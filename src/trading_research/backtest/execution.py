"""Turning a stream of signals into a stream of trades.

Everything before this module treats each observation independently: a model
emits a probability at every row, and the evaluation so far has charged a full
round trip for each one it acted on. That is a deliberate simplification, and
it is wrong in a specific way — consecutive observations produce almost the
same signal, so acting on all of them means paying for one opinion many times.

Two rules fix that, and both come from how a real system behaves:

**One position at a time.** A signal arriving while a position is open is
ignored. Without this, a two-minute horizon on a 100 ms grid opens twelve
hundred overlapping positions for a single sustained view.

**A cooldown after closing.** Having just traded on a view, wait before acting
on it again. Otherwise the same slow-moving signal reopens immediately and the
position is effectively never closed, only re-charged.

Both raise profit per trade by trading less. Neither creates an edge that was
not there: the total is what it is, and thinning redistributes it across fewer
trades. Whether that turns a losing strategy into a winning one depends on
whether the edge per *opinion* clears the cost, which is the honest question —
and is what these rules make measurable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading_research.labels.directional import HOLD


@dataclass(frozen=True)
class Trade:
    """One completed round trip."""

    entry_index: int
    exit_index: int
    direction: int
    entry_spread_bp: float
    move_bp: float
    #: What ended the trade: "clock", "take_profit", "stop_loss", "reversal" or
    #: "end_of_data". Recorded because a rule that never fires and a rule that
    #: fires on every trade produce the same summary line and mean opposite
    #: things.
    exit_reason: str = "clock"


@dataclass(frozen=True)
class ThinningRules:
    """How signals are converted into trades.

    ``hold_periods``
        How long a position stays open, in observations. Normally the label
        horizon: the model predicted a move over that window, so the position
        is held for it and no longer.
    ``cooldown_periods``
        Observations to wait after closing before trading again.
    ``allow_reversal``
        Whether an opposite signal may close a position early. Off by default:
        it doubles the cost of a reversal and, on a noisy signal, mostly
        converts churn into more churn.
    ``take_profit_bp`` / ``stop_loss_bp``
        Exit as soon as the position is this far in front of, or behind, its
        entry. Both need the price path, so ``thin`` must be given ``mid``.

        Neither is free. A take-profit caps the winners and leaves the losers
        to run their full course, which is why a tight one *lowers* the average
        outcome while raising the hit rate — measured here at 5 bp: 69% of
        trades won and the mean fell by more than half. A stop-loss does the
        mirror image, and a tight one turns ordinary noise into a realised
        loss. They are options, not improvements.
    """

    hold_periods: int
    cooldown_periods: int = 0
    allow_reversal: bool = False
    take_profit_bp: float | None = None
    stop_loss_bp: float | None = None

    def __post_init__(self) -> None:
        if self.hold_periods < 1:
            raise ValueError(f"hold_periods must be at least 1, got {self.hold_periods}")
        if self.cooldown_periods < 0:
            raise ValueError(f"cooldown_periods must be non-negative, got {self.cooldown_periods}")
        for name in ("take_profit_bp", "stop_loss_bp"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when set, got {value}")

    @property
    def needs_price_path(self) -> bool:
        """True when an exit can happen before the clock runs out."""
        return self.take_profit_bp is not None or self.stop_loss_bp is not None


def thin(
    decision: np.ndarray,
    forward_bp: np.ndarray,
    spread_bp: np.ndarray,
    rules: ThinningRules,
    mid: np.ndarray | None = None,
) -> list[Trade]:
    """Walk the signal stream in time order, opening trades where allowed.

    A single forward pass, because that is the only way the state — in a
    position, in cooldown, free — can be correct. A vectorised version would
    have to know when positions close, which depends on when they opened.

    Without a take-profit or stop-loss, a trade's outcome is ``forward_bp`` at
    entry: the realised move over exactly the horizon the model was trained to
    predict, so the trade is scored against what it bet on.

    With either set, the outcome depends on the path, and ``mid`` becomes
    required. The exit is the first of take-profit, stop-loss and the clock —
    checked in that order at each step. Both are evaluated on the *same*
    observation, so a bar that would have triggered both is resolved
    optimistically. That is a real limitation of bar data rather than a choice,
    and it flatters the result: with 100 ms observations the window in which it
    matters is small, but it is not zero, and a wider grid would make it
    material.
    """
    n = len(decision)
    if rules.needs_price_path and mid is None:
        raise ValueError("take_profit_bp or stop_loss_bp needs the price path; pass mid=")

    trades: list[Trade] = []
    position_until = -1  # index at which the current position closes on the clock
    free_from = 0  # index from which trading is allowed again
    open_at = -1
    open_direction = HOLD

    def close(at: int, reason: str) -> None:
        nonlocal open_direction, free_from
        if mid is not None and reason != "clock":
            move = open_direction * (mid[at] / mid[open_at] - 1.0) * 1e4
        else:
            move = float(forward_bp[open_at])
        trades.append(
            Trade(
                entry_index=open_at,
                exit_index=at,
                direction=open_direction,
                entry_spread_bp=float(spread_bp[open_at]),
                move_bp=float(move),
                exit_reason=reason,
            )
        )
        open_direction = HOLD
        free_from = at + rules.cooldown_periods

    for i in range(n):
        # Path exits are checked before the clock, since either can only end
        # the trade earlier than it would have ended anyway.
        if open_direction != HOLD and rules.needs_price_path and mid is not None:
            move = open_direction * (mid[i] / mid[open_at] - 1.0) * 1e4
            if rules.take_profit_bp is not None and move >= rules.take_profit_bp:
                close(i, "take_profit")
            elif rules.stop_loss_bp is not None and move <= -rules.stop_loss_bp:
                close(i, "stop_loss")

        if open_direction != HOLD and i >= position_until:
            close(i, "clock")

        signal = int(decision[i])
        if signal == HOLD:
            continue

        if open_direction != HOLD:
            if rules.allow_reversal and signal != open_direction:
                close(i, "reversal")
            else:
                continue

        if i < free_from:
            continue
        if np.isnan(forward_bp[i]) or np.isnan(spread_bp[i]):
            continue

        open_direction = signal
        open_at = i
        position_until = i + rules.hold_periods

    # A position still open when the data runs out is closed rather than
    # dropped. Its outcome is already known — `forward_bp` at entry covers the
    # whole holding period, and rows whose outcome is unknown were never
    # entered — so discarding it would quietly lose real trades, and the ones
    # nearest the end of every block at that.
    if open_direction != HOLD:
        open_direction_kept = open_direction
        at = min(position_until, n - 1)
        move = float(forward_bp[open_at])
        trades.append(
            Trade(
                entry_index=open_at,
                exit_index=at,
                direction=open_direction_kept,
                entry_spread_bp=float(spread_bp[open_at]),
                move_bp=move,
                exit_reason="end_of_data",
            )
        )

    return trades


def trades_to_frame(trades: list[Trade]) -> pd.DataFrame:
    """Tabulate trades, with an empty frame of the right shape when there are none."""
    if not trades:
        return pd.DataFrame(
            columns=[
                "entry_index",
                "exit_index",
                "direction",
                "entry_spread_bp",
                "move_bp",
                "exit_reason",
            ]
        )
    return pd.DataFrame([t.__dict__ for t in trades])


def score(trades: list[Trade], costs: object) -> dict[str, float]:
    """Profit and loss for a set of thinned trades.

    Reported per trade as well as in total, because thinning changes the trade
    count by design: a total that improves purely because fewer trades were
    taken says nothing, while an improvement per trade says the surviving
    signals were the better ones.
    """
    if not trades:
        return {
            "trades": 0.0,
            "gross_bp": 0.0,
            "cost_bp": 0.0,
            "net_bp": 0.0,
            "gross_per_trade_bp": float("nan"),
            "net_per_trade_bp": float("nan"),
            "hit_rate": float("nan"),
        }

    frame = trades_to_frame(trades)
    gross = (frame["direction"] * frame["move_bp"]).to_numpy()
    cost = np.asarray(costs.round_trip_bp(frame["entry_spread_bp"]))  # type: ignore[attr-defined]
    net = gross - cost

    return {
        "trades": float(len(trades)),
        "gross_bp": float(gross.sum()),
        "cost_bp": float(cost.sum()),
        "net_bp": float(net.sum()),
        "gross_per_trade_bp": float(gross.mean()),
        "net_per_trade_bp": float(net.mean()),
        "hit_rate": float((gross > 0).mean()),
    }
