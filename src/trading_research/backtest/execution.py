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

from trading_research.labels.directional import BUY, HOLD


@dataclass(frozen=True)
class Trade:
    """One completed round trip."""

    entry_index: int
    exit_index: int
    direction: int
    entry_spread_bp: float
    move_bp: float
    #: What ended the trade: "clock", "take_profit", "stop_loss",
    #: "trailing_stop", "signal_decay", "flip", "reversal" or "end_of_data".
    #: Recorded because a rule that never fires and a rule that fires on every
    #: trade produce the same summary line and mean opposite things.
    exit_reason: str = "clock"


@dataclass(frozen=True)
class ThinningRules:
    """How signals are converted into trades.

    Entry is one decision — act or do not — and it is governed by the
    confidence threshold upstream. Exit is where the choices are, and there are
    more of them than there look to be.

    Position management
    -------------------
    ``hold_periods``
        How long a position stays open, in observations, when nothing else
        closes it. The natural value is the label horizon: the model predicted
        a move over that window, so the position is held for it and no longer.
    ``cooldown_periods``
        Observations to wait after closing before trading again.
    ``allow_reversal``
        Whether an opposite signal may close a position **and open the
        opposite one**. Off by default: it doubles the cost of a reversal and,
        on a noisy signal, mostly converts churn into more churn.
    ``hold_scale_by_confidence``
        Makes the clock depend on how sure the model was at entry rather than
        being one number for every trade. At *k*, a position opened with
        confidence *c* is held for ``hold_periods * (1 + k * (c - 0.5) * 2)``
        observations, so a barely-cleared signal is held for less than the
        default and a strong one for more. Zero keeps the fixed clock.

        The case for it is that confidence and horizon are related: a model
        nearly certain of a direction is usually seeing a larger or more
        persistent move than one that just cleared the threshold, and giving
        both the same holding period wastes the first and overstays the second.

    Price-path exits
    ----------------
    ``take_profit_bp``
        Close once the position is this far in front. Caps the winners while
        leaving the losers to run their full course, which is why a tight one
        *lowers* the average outcome while raising the hit rate — measured here
        at 4 bp on BTCUSDT: 75% of trades won, against 54% for the untouched
        clock, and gross edge per trade fell from 2.97 bp to 1.27.
    ``stop_loss_bp``
        Close once the position is this far behind. The mirror image: a tight
        one converts ordinary noise into a realised loss.
    ``trailing_stop_bp``
        Close once the position has given back this much from the best level it
        reached. Unlike a fixed take-profit it does not cap the winners, which
        is the usual argument for it; what it does instead is convert a winner
        into a smaller winner every time the path is noisy, and at these
        horizons the path is nothing but noise.

    Signal exits
    ------------
    These read the model's ongoing opinion rather than the price, so ``thin``
    must be given ``p_buy`` and ``p_sell``. They are the only rules here that
    use the fact that a model keeps predicting after the trade is opened —
    everything else treats the entry decision as final.

    ``exit_below_confidence``
        Close when the probability of the direction being held falls below
        this. The position was opened because the model was confident; this
        closes it when the model stops being confident, rather than waiting for
        a clock that was set by the label horizon.
    ``exit_on_flip``
        Close when the model's probability for the opposite direction exceeds
        the one being held. Distinct from ``allow_reversal``: this goes flat
        and respects the cooldown, rather than immediately paying to enter the
        other side.

    Marker exits
    ------------
    ``thin`` also accepts an ``exit_when`` boolean array. Anything computable
    per observation can drive it — volatility collapsing, the spread widening,
    a feature reverting — and the position closes when it is true. Kept as an
    array rather than a rule so that this module stays ignorant of features:
    what counts as a marker is the caller's question, not execution's.

    Every one of these is an option, not an improvement. Which combination is
    best is a question for validation — see
    :mod:`trading_research.validation.exits`.
    """

    hold_periods: int
    cooldown_periods: int = 0
    allow_reversal: bool = False
    hold_scale_by_confidence: float = 0.0
    take_profit_bp: float | None = None
    stop_loss_bp: float | None = None
    trailing_stop_bp: float | None = None
    exit_below_confidence: float | None = None
    exit_on_flip: bool = False

    def __post_init__(self) -> None:
        if self.hold_periods < 1:
            raise ValueError(f"hold_periods must be at least 1, got {self.hold_periods}")
        if self.cooldown_periods < 0:
            raise ValueError(f"cooldown_periods must be non-negative, got {self.cooldown_periods}")
        for name in ("take_profit_bp", "stop_loss_bp", "trailing_stop_bp"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when set, got {value}")
        if self.hold_scale_by_confidence < 0:
            raise ValueError(
                f"hold_scale_by_confidence must be non-negative, got {self.hold_scale_by_confidence}"
            )
        if self.exit_below_confidence is not None and not 0.0 < self.exit_below_confidence < 1.0:
            raise ValueError(
                f"exit_below_confidence must be in (0, 1), got {self.exit_below_confidence}"
            )

    @property
    def needs_price_path(self) -> bool:
        """True when an exit can happen before the clock runs out.

        Includes the signal exits: they decide *when* to leave from the model's
        opinion, but the outcome still has to be priced from the path, and
        using the label's forward return for a trade that ended early would
        credit it with a move it was not there for.
        """
        return (
            self.take_profit_bp is not None
            or self.stop_loss_bp is not None
            or self.trailing_stop_bp is not None
            or self.needs_probabilities
        )

    @property
    def needs_probabilities(self) -> bool:
        """True when an exit reads the model's ongoing opinion."""
        return (
            self.exit_below_confidence is not None
            or self.exit_on_flip
            or self.hold_scale_by_confidence > 0
        )


def thin(
    decision: np.ndarray,
    forward_bp: np.ndarray,
    spread_bp: np.ndarray,
    rules: ThinningRules,
    mid: np.ndarray | None = None,
    p_buy: np.ndarray | None = None,
    p_sell: np.ndarray | None = None,
    exit_when: np.ndarray | None = None,
) -> list[Trade]:
    """Walk the signal stream in time order, opening trades where allowed.

    A single forward pass, because that is the only way the state — in a
    position, in cooldown, free — can be correct. A vectorised version would
    have to know when positions close, which depends on when they opened.

    With no early-exit rule set, a trade's outcome is ``forward_bp`` at entry:
    the realised move over exactly the horizon the model was trained to
    predict, so the trade is scored against what it bet on.

    With any early exit the outcome depends on the path, ``mid`` becomes
    required, and the signal exits additionally need ``p_buy`` and ``p_sell``.

    ``exit_when`` is an optional boolean array: where it is true, an open
    position closes with reason ``"marker"``. It is how a feature-driven exit
    is expressed without this module knowing what a feature is.

    Order of checks
    ---------------
    At each observation, in this order: take-profit, stop-loss, trailing stop,
    signal exits, marker, then the clock. The order is a choice and it matters, because
    two rules can be satisfied on the same observation and bar data cannot say
    which came first. It resolves ties optimistically — the profitable exit
    wins — which flatters the result. At 100 ms the window in which it matters
    is small, but it is not zero, and a wider grid would make it material.
    """
    n = len(decision)
    if rules.needs_price_path and mid is None:
        raise ValueError("this exit rule needs the price path; pass mid=")
    if rules.needs_probabilities and (p_buy is None or p_sell is None):
        raise ValueError("this exit rule needs the model's opinion; pass p_buy= and p_sell=")

    trades: list[Trade] = []
    position_until = -1  # index at which the current position closes on the clock
    free_from = 0  # index from which trading is allowed again
    open_at = -1
    open_direction = HOLD
    best_move_bp = 0.0  # high-water mark since entry, for the trailing stop

    def raw_move(index: int) -> float:
        """Price change since entry, in basis points, unsigned by direction.

        Stored on the trade, because ``score`` applies the direction itself and
        a value that already had it applied would be signed twice. That is not
        hypothetical: it was the shape of a real bug here, and it inverted the
        profit of every short that exited early while leaving every long
        correct — so the totals stayed plausible and only shorts were wrong.
        """
        assert mid is not None
        return (mid[index] / mid[open_at] - 1.0) * 1e4

    def position_move(index: int) -> float:
        """The same change from the position's point of view: positive is ahead.

        This is what the exit *conditions* need — a stop-loss asks whether the
        position is behind, not whether the price fell.
        """
        return open_direction * raw_move(index)

    def close(at: int, reason: str) -> None:
        nonlocal open_direction, free_from, best_move_bp
        # A clock exit after exactly ``hold_periods`` rows is scored on
        # ``forward_bp``, which is by construction the move over those rows.
        # Any other duration — a hold scaled by confidence, or the data ending
        # first — is scored on the price path, because ``forward_bp`` measures
        # the fixed horizon and not the rows actually held: the earlier version
        # credited a position held twice as long with the move of the unscaled
        # one. Without a path the fixed-horizon move is the only number there
        # is, and it is used with that caveat.
        exact_clock = reason == "clock" and at - open_at == rules.hold_periods
        move = float(forward_bp[open_at]) if exact_clock or mid is None else raw_move(at)
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
        best_move_bp = 0.0
        free_from = at + rules.cooldown_periods

    for i in range(n):
        # Early exits are checked before the clock, since each of them can only
        # end the trade sooner than it would have ended anyway.
        if open_direction != HOLD and mid is not None:
            move = position_move(i)
            best_move_bp = max(best_move_bp, move)

            if rules.take_profit_bp is not None and move >= rules.take_profit_bp:
                close(i, "take_profit")
            elif rules.stop_loss_bp is not None and move <= -rules.stop_loss_bp:
                close(i, "stop_loss")
            elif (
                rules.trailing_stop_bp is not None
                # Only after the position has been in front: a trailing stop
                # armed from entry is a stop-loss wearing a different name.
                and best_move_bp > 0
                and move <= best_move_bp - rules.trailing_stop_bp
            ):
                close(i, "trailing_stop")

        if open_direction != HOLD and rules.needs_probabilities:
            assert p_buy is not None and p_sell is not None
            held = p_buy[i] if open_direction == BUY else p_sell[i]
            against = p_sell[i] if open_direction == BUY else p_buy[i]
            if rules.exit_below_confidence is not None and held < rules.exit_below_confidence:
                close(i, "signal_decay")
            elif rules.exit_on_flip and against > held:
                close(i, "flip")

        if open_direction != HOLD and exit_when is not None and bool(exit_when[i]):
            close(i, "marker")

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
        best_move_bp = 0.0

        hold = rules.hold_periods
        if rules.hold_scale_by_confidence > 0 and p_buy is not None and p_sell is not None:
            # Confidence in the direction taken, mapped from [0.5, 1] onto
            # [0, 1] so that a signal which barely cleared the threshold scales
            # the clock by about one and a near-certain one by 1 + k.
            entry_confidence = p_buy[i] if signal == BUY else p_sell[i]
            stretch = 1.0 + rules.hold_scale_by_confidence * max(
                0.0, (float(entry_confidence) - 0.5) * 2.0
            )
            hold = max(1, round(rules.hold_periods * stretch))
        position_until = i + hold

    # A position still open when the data runs out is closed rather than
    # dropped. Its outcome is already known — `forward_bp` at entry covers the
    # whole holding period, and rows whose outcome is unknown were never
    # entered — so discarding it would quietly lose real trades, and the ones
    # nearest the end of every block at that.
    if open_direction != HOLD:
        trades.append(
            Trade(
                entry_index=open_at,
                exit_index=min(position_until, n - 1),
                direction=open_direction,
                entry_spread_bp=float(spread_bp[open_at]),
                move_bp=float(forward_bp[open_at]),
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


def score(trades: list[Trade], costs: object, *, days: float | None = None) -> dict[str, float]:
    """Profit and loss for a set of thinned trades.

    Reported per trade as well as in total, because thinning changes the trade
    count by design: a total that improves purely because fewer trades were
    taken says nothing, while an improvement per trade says the surviving
    signals were the better ones.

    Pass ``days`` to get the figures a desk asks for before any of these —
    trades per day, profit per day, and the worst drawdown along the way. See
    :mod:`trading_research.backtest.metrics`.
    """
    if not trades:
        empty = {
            "trades": 0.0,
            "gross_bp": 0.0,
            "cost_bp": 0.0,
            "net_bp": 0.0,
            "gross_per_trade_bp": float("nan"),
            "net_per_trade_bp": float("nan"),
            "hit_rate": float("nan"),
        }
        if days is not None:
            empty.update({"trades_per_day": 0.0, "net_bp_per_day": 0.0, "max_drawdown_bp": 0.0})
        return empty

    frame = trades_to_frame(trades)
    gross = (frame["direction"] * frame["move_bp"]).to_numpy()
    cost = np.asarray(costs.round_trip_bp(frame["entry_spread_bp"]))  # type: ignore[attr-defined]
    net = gross - cost

    result = {
        "trades": float(len(trades)),
        "gross_bp": float(gross.sum()),
        "cost_bp": float(cost.sum()),
        "net_bp": float(net.sum()),
        "gross_per_trade_bp": float(gross.mean()),
        "net_per_trade_bp": float(net.mean()),
        "hit_rate": float((gross > 0).mean()),
    }
    if days is not None:
        from trading_research.backtest.metrics import summarise

        # The hit rate here is on net rather than gross: a trade that moved the
        # right way but not far enough to pay for itself is not a win.
        extended = summarise(net, days=days)
        result.update({k: v for k, v in extended.items() if k not in result})
        result["net_hit_rate"] = extended["hit_rate"]
    return result
