"""Execution as a mode: crossing, posting and quoting behind one interface.

Everything upstream of execution produces an opinion: a direction, how strong
it is, and when it became known. How that opinion meets the market is a
separate choice, and this project has built three answers to it:

``taker``
    Cross the spread on the signal and cross back when the hold ends:
    :func:`~trading_research.backtest.execution.thin` and
    :class:`~trading_research.backtest.costs.TakerCosts` on the five-second
    grid. Every result before §30 was measured this way, and it stays the
    default everywhere, so every earlier number reproduces unchanged.
``passive_entry``
    Post at the touch on the signal and fill only once the queue ahead has
    traded through: :func:`~trading_research.backtest.maker.simulate` on the
    same grid, with the venue's prints aggregated onto it.
``market_maker``
    Quote both sides continuously in event time through
    :mod:`trading_research.market_making.simulator`: a queue per order,
    latency, fills only from prints, inventory limits, fees and funding. The
    quoter is one of the pre-registered S0 to S3 (:mod:`.market_making`), or,
    when a signal is given, S1 leaning on the signal's strength.

Two event-time modes serve the 2025 short-horizon study, on days its own
loader read (they read nothing themselves):

``event_taker``
    The event-time taker executor, :func:`trading_research.backtest.taker.execute_day`:
    the clip walked across the visible book at decision + latency and walked
    back after the hold, funding settled on the position held. Paired with a
    regime policy as ``taker``.
``signal_entry``
    The signal-entry quoter, :class:`trading_research.market_making.signal_entry.SignalEntryQuoter`,
    in the event-time simulator: one clip at the touch per decision, a taker
    exit a hold after the fill. Paired as ``passive_entry``; under
    ``guard_pull`` the simulator also cancels a resting entry at the first
    event after each flag.

Every mode returns the same table, ``attempts`` (symbol, day, ts, filled,
gross_bp, cost_bp, net_bp, path), which is the shape
:func:`~trading_research.evaluation.significance.assess` reads. A mode cannot
win by being measured differently: misses count as attempts that earned
nothing, and the market maker contributes one row per simulated day, its net
over the notional it traded.

Regime policy
-------------
A run also chooses what to do after a regime-break flag: ``none``,
``guard_pull`` (no new entries, or no quotes, inside the window after a flag)
or ``guard_widen`` (the market maker's half-spread doubled inside it). A taker
or a single passive entry has no spread of its own to widen, so that pairing is
refused rather than given a meaning after the fact; :data:`VALID_PAIRS` lists
the seven combinations that exist.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date
from enum import StrEnum
from typing import cast

import numpy as np
import pandas as pd

from trading_research.backtest.costs import TakerCosts
from trading_research.backtest.execution import ThinningRules, thin
from trading_research.backtest.maker import MakerCosts, PostingRules, simulate
from trading_research.backtest.taker import TakerBook, TakerDay, TakerSettings, execute_day
from trading_research.market_making.events import DayEvents
from trading_research.market_making.prereg import Access
from trading_research.market_making.quoters import FLAG_SIGNAL
from trading_research.market_making.signal_entry import SignalEntryQuoter, entry_tapes
from trading_research.market_making.signals import SignalTape
from trading_research.market_making.simulator import (
    EXTRA_COUNTERS,
    SimConfig,
    _summary,
    simulate_day,
)
from trading_research.pipeline import market_making as mm

NS_PER_S = 1_000_000_000
NS_PER_DAY = 86_400 * NS_PER_S


class ExecutionMode(StrEnum):
    TAKER = "taker"
    PASSIVE_ENTRY = "passive_entry"
    MARKET_MAKER = "market_maker"
    EVENT_TAKER = "event_taker"
    SIGNAL_ENTRY = "signal_entry"

    @property
    def family(self) -> str:
        """The execution a regime policy pairs with (:data:`VALID_PAIRS`): the
        event-time taker is a taker, the signal-entry quoter a passive entry."""
        return _FAMILY.get(self.value, self.value)


_FAMILY: dict[str, str] = {"event_taker": "taker", "signal_entry": "passive_entry"}


#: What a run does after a regime-break flag.
REGIME_POLICIES: tuple[str, ...] = ("none", "guard_pull", "guard_widen")

#: The (execution, regime policy) pairs that mean something.
VALID_PAIRS: tuple[tuple[str, str], ...] = (
    ("taker", "none"),
    ("taker", "guard_pull"),
    ("passive_entry", "none"),
    ("passive_entry", "guard_pull"),
    ("market_maker", "none"),
    ("market_maker", "guard_pull"),
    ("market_maker", "guard_widen"),
)

#: The columns every mode's attempts carry, in order.
ATTEMPT_COLUMNS: tuple[str, ...] = (
    "symbol",
    "day",
    "ts",
    "filled",
    "gross_bp",
    "cost_bp",
    "net_bp",
    "path",
)


def check_pair(mode: ExecutionMode | str, regime_policy: str) -> ExecutionMode:
    """The mode, after refusing a regime policy it cannot carry out."""
    mode = ExecutionMode(mode)
    if regime_policy not in REGIME_POLICIES:
        raise ValueError(f"regime policy must be one of {REGIME_POLICIES}, got {regime_policy!r}")
    if (mode.family, regime_policy) not in VALID_PAIRS:
        raise ValueError(
            f"{mode.value} has no spread of its own to widen; "
            f"use guard_pull or none (valid pairs: {VALID_PAIRS})"
        )
    return mode


@dataclass(frozen=True)
class SignalStream:
    """Decisions from any rule or model, in event time.

    ``ts`` is when each decision became known (int64 ns since the epoch);
    ``direction`` is -1, 0 or +1; ``strength`` is whatever the producer
    measures its conviction in (bp or a probability margin), read by the
    market maker's lean and by nothing else; ``hold_ns`` is how long a position
    opened on it is meant to last.
    """

    symbol: str
    ts: np.ndarray
    direction: np.ndarray
    strength: np.ndarray
    hold_ns: int

    def __post_init__(self) -> None:
        if not (len(self.ts) == len(self.direction) == len(self.strength)):
            raise ValueError("a signal stream needs one direction and strength per timestamp")
        if len(self.ts) > 1 and np.any(np.diff(np.asarray(self.ts, dtype=np.int64)) < 0):
            raise ValueError("signal timestamps must be in time order")
        if not np.isin(np.asarray(self.direction), (-1, 0, 1)).all():
            raise ValueError("direction must be -1, 0 or +1")
        if self.hold_ns <= 0:
            raise ValueError("the hold must be positive")

    def __len__(self) -> int:
        return len(self.ts)

    @classmethod
    def from_grid(
        cls,
        symbol: str,
        labels_ns: np.ndarray,
        decision: np.ndarray,
        *,
        hold_rows: int,
        strength: np.ndarray | None = None,
        grid_seconds: int = 5,
    ) -> SignalStream:
        """The rows of a gridded decision array that carry a decision."""
        decision = np.asarray(decision)
        if len(decision) != len(labels_ns):
            raise ValueError("one decision per grid label")
        keep = decision != 0
        values = np.ones(len(decision)) if strength is None else np.asarray(strength, dtype=float)
        return cls(
            symbol,
            np.asarray(labels_ns, dtype=np.int64)[keep],
            decision[keep].astype(np.int8),
            values[keep],
            int(hold_rows) * grid_seconds * NS_PER_S,
        )

    def on_grid(self, labels_ns: np.ndarray) -> np.ndarray:
        """A decision per grid row: each signal acts at the first label at or after it.

        A signal is never moved earlier than it was known. Two signals reaching
        the same row leave the later one.
        """
        labels = np.asarray(labels_ns, dtype=np.int64)
        out = np.zeros(len(labels), dtype=int)
        rows = np.searchsorted(labels, np.asarray(self.ts, dtype=np.int64), side="left")
        inside = rows < len(labels)
        out[rows[inside]] = np.asarray(self.direction)[inside]
        return out


@dataclass(frozen=True)
class GridMarket:
    """One instrument on the five-second grid: what taker and passive entry read.

    ``labels_ns`` are the row labels; ``sell_at_bid`` and ``buy_at_ask`` are
    the aggressive volume that would consume a resting bid and ask on each
    row (:func:`aggressive_flow`), needed by passive entry only.
    """

    symbol: str
    labels_ns: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    bid_size: np.ndarray
    ask_size: np.ndarray
    sell_at_bid: np.ndarray | None = None
    buy_at_ask: np.ndarray | None = None
    tick: float = 0.0

    def __post_init__(self) -> None:
        n = len(self.labels_ns)
        for name in ("bid", "ask", "bid_size", "ask_size", "sell_at_bid", "buy_at_ask"):
            array = getattr(self, name)
            if array is not None and len(array) != n:
                raise ValueError(f"{name} has {len(array)} rows, the grid has {n}")

    def __len__(self) -> int:
        return len(self.labels_ns)

    def window(self, start: int, stop: int) -> GridMarket:
        """Rows ``[start, stop)``, every array cut the same way."""
        cut = slice(start, stop)

        def part(array: np.ndarray | None) -> np.ndarray | None:
            return None if array is None else np.asarray(array)[cut]

        return GridMarket(
            self.symbol,
            np.asarray(self.labels_ns)[cut],
            np.asarray(self.bid)[cut],
            np.asarray(self.ask)[cut],
            np.asarray(self.bid_size)[cut],
            np.asarray(self.ask_size)[cut],
            sell_at_bid=part(self.sell_at_bid),
            buy_at_ask=part(self.buy_at_ask),
            tick=self.tick,
        )

    @property
    def mid(self) -> np.ndarray:
        return (np.asarray(self.bid) + np.asarray(self.ask)) / 2.0

    @property
    def spread_bp(self) -> np.ndarray:
        return (np.asarray(self.ask) - np.asarray(self.bid)) / self.mid * 1e4

    def forward_bp(self, rows: int) -> np.ndarray:
        """The mid's move over the next ``rows`` rows, bp; NaN where it runs out."""
        mid = self.mid
        out = np.full(len(mid), np.nan)
        out[:-rows] = (mid[rows:] / mid[:-rows] - 1.0) * 1e4
        return out


def aggressive_flow(
    trades: pd.DataFrame, timestamps: pd.Series, bid: np.ndarray, ask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per row, the volume that would consume a resting bid and a resting ask.

    A print is assigned to the row whose interval contains it, and counted only
    if it happened at a price a resting order would have been standing at. A
    sell that traded *below* the best bid swept through it, so it counts too;
    one that traded above it never touched the queue.
    """
    edges = timestamps.to_numpy()
    slot = np.searchsorted(edges, trades["timestamp"].to_numpy(), side="right") - 1
    inside = (slot >= 0) & (slot < len(edges))
    slot, aggressor = slot[inside], trades["aggressor"].to_numpy()[inside]
    price, size = trades["price"].to_numpy()[inside], trades["size"].to_numpy()[inside]

    sell_at_bid = np.zeros(len(edges))
    buy_at_ask = np.zeros(len(edges))
    selling = (aggressor == -1) & (price <= bid[slot])
    buying = (aggressor == 1) & (price >= ask[slot])
    np.add.at(sell_at_bid, slot[selling], size[selling])
    np.add.at(buy_at_ask, slot[buying], size[buying])
    return sell_at_bid, buy_at_ask


def guarded_rows(labels_ns: np.ndarray, flags_ns: np.ndarray, window_ns: int) -> np.ndarray:
    """True on rows inside ``(flag, flag + window]`` of some flag.

    The window opens strictly after the flag is effective, as the market
    maker's :class:`~trading_research.market_making.quoters.RegimeGuard` reads
    it, so a row at the flag's own instant is not yet guarded.
    """
    labels = np.asarray(labels_ns, dtype=np.int64)
    flags = np.unique(np.asarray(flags_ns, dtype=np.int64))
    if len(flags) == 0 or len(labels) == 0:
        return np.zeros(len(labels), dtype=bool)
    latest = np.searchsorted(flags, labels, side="left") - 1
    known = latest >= 0
    out = np.zeros(len(labels), dtype=bool)
    out[known] = labels[known] - flags[latest[known]] <= window_ns
    return out


#: Columns of the grid modes' per-attempt frames, before symbol, day and ts.
GRID_COLUMNS = (
    "entry_index",
    "exit_index",
    "direction",
    "filled",
    "gross_bp",
    "cost_bp",
    "net_bp",
    "path",
)


def run_taker(
    decision: np.ndarray,
    forward_bp: np.ndarray,
    spread_bp: np.ndarray,
    rules: ThinningRules,
    costs: TakerCosts,
    *,
    mid: np.ndarray | None = None,
    p_buy: np.ndarray | None = None,
    p_sell: np.ndarray | None = None,
) -> pd.DataFrame:
    """Taker execution on a grid: :func:`thin`, then one round trip charged per trade.

    Exactly what the searches did before execution became an axis — the same
    call, the same cost per trade — so a caller routed through here reproduces
    its earlier numbers to the last bit.
    """
    trades = thin(decision, forward_bp, spread_bp, rules, mid=mid, p_buy=p_buy, p_sell=p_sell)
    rows = []
    for trade in trades:
        # A scalar spread in, a scalar cost out; the cast is for the type checker.
        cost = float(cast(float, costs.round_trip_bp(trade.entry_spread_bp)))
        gross = trade.direction * trade.move_bp
        rows.append(
            (
                trade.entry_index,
                trade.exit_index,
                trade.direction,
                True,
                gross,
                cost,
                gross - cost,
                trade.exit_reason,
            )
        )
    return pd.DataFrame(rows, columns=list(GRID_COLUMNS))


def run_passive(
    decision: np.ndarray,
    market: GridMarket,
    rules: PostingRules,
    costs: MakerCosts,
) -> pd.DataFrame:
    """Passive entry on a grid: :func:`simulate`, every attempt a row.

    A miss is a row with ``filled`` False and nothing earned, so the per-row
    mean is :func:`~trading_research.backtest.maker.score`'s net per attempt.
    """
    if market.sell_at_bid is None or market.buy_at_ask is None:
        raise ValueError("passive entry needs the prints on the grid (sell_at_bid, buy_at_ask)")
    attempts = simulate(
        decision,
        np.asarray(market.bid),
        np.asarray(market.ask),
        np.asarray(market.bid_size),
        np.asarray(market.ask_size),
        np.asarray(market.sell_at_bid),
        np.asarray(market.buy_at_ask),
        rules,
        tick=market.tick,
    )
    spread_bp = market.spread_bp
    rows = []
    for attempt in attempts:
        if not attempt.filled:
            rows.append(
                (attempt.signal_index, -1, attempt.direction, False, 0.0, 0.0, 0.0, "unfilled")
            )
            continue
        gross = (attempt.exit_price / attempt.fill_price - 1.0) * 1e4 * attempt.direction
        cost = costs.round_trip_bp(
            passive_exit=not attempt.exit_crossed,
            spread_bp=float(spread_bp[attempt.exit_index]),
        )
        rows.append(
            (
                attempt.signal_index,
                attempt.exit_index,
                attempt.direction,
                True,
                gross,
                cost,
                gross - cost,
                attempt.exit_reason,
            )
        )
    return pd.DataFrame(rows, columns=list(GRID_COLUMNS))


@dataclass(frozen=True)
class ExecutionResult:
    """What a mode did with a signal, in the one shape every mode shares.

    ``days`` is the market maker's per-day table (its decomposition, fills,
    flags), and ``None`` for the grid modes.
    """

    attempts: pd.DataFrame
    summary: dict[str, float]
    mode: ExecutionMode
    days: pd.DataFrame | None = None


def summarise(attempts: pd.DataFrame) -> dict[str, float]:
    """Per-attempt figures, misses included, so a low fill rate is in the number."""
    if attempts.empty:
        return {"attempts": 0.0, "fills": 0.0, "fill_rate": math.nan, "net_bp": 0.0}
    filled = attempts["filled"].astype(bool)
    return {
        "attempts": float(len(attempts)),
        "fills": float(filled.sum()),
        "fill_rate": float(filled.mean()),
        "gross_per_attempt_bp": float(attempts["gross_bp"].mean()),
        "cost_per_attempt_bp": float(attempts["cost_bp"].mean()),
        "net_per_attempt_bp": float(attempts["net_bp"].mean()),
        "net_bp": float(attempts["net_bp"].sum()),
        "days": float(attempts["day"].nunique()),
    }


def _stamped(symbol: str, labels_ns: np.ndarray, rows: pd.DataFrame) -> pd.DataFrame:
    """A grid mode's rows with the symbol, the day and the time of each entry."""
    ts = np.asarray(labels_ns, dtype=np.int64)[rows["entry_index"].to_numpy(dtype=np.int64)]
    out = rows.assign(symbol=symbol, ts=ts, day=pd.to_datetime(ts).strftime("%Y-%m-%d"))
    return out[[*ATTEMPT_COLUMNS, "entry_index", "exit_index", "direction"]]


def execute(
    signals: SignalStream | None,
    mode: ExecutionMode | str,
    *,
    market: GridMarket | None = None,
    symbol: str | None = None,
    days: Sequence[date] = (),
    regime_policy: str = "none",
    flags_ns: np.ndarray | None = None,
    guard_window_min: float = 15.0,
    grid_seconds: int = 5,
    taker: TakerCosts | None = None,
    taker_rules: ThinningRules | None = None,
    maker: MakerCosts | None = None,
    posting: PostingRules | None = None,
    plan: mm.MakerPlan | None = None,
    sources: mm.MakerSources | None = None,
    access: Access | None = None,
    workers: int = 1,
    books: Sequence[TakerBook] = (),
    taker_settings: TakerSettings | None = None,
    events: Sequence[DayEvents] = (),
    sim_config: SimConfig | None = None,
    passive_exit: bool = False,
) -> ExecutionResult:
    """Execute ``signals`` in ``mode`` and return the attempts and their summary.

    The event-time modes read days already loaded by the study's loader:
    ``event_taker`` one :class:`~trading_research.backtest.taker.TakerBook` per
    day in ``books`` (``taker_settings`` defaults to the registered ones);
    ``signal_entry`` one :class:`~trading_research.market_making.events.DayEvents`
    per day in ``events`` under ``sim_config``. A decision on a day without
    one is not executed. Their ``days`` table holds one row per day, funding,
    counters and flags included.

    The grid modes read ``market`` and map each signal to the first grid row
    at or after it. The market maker reads ``days`` of event-time data from
    ``sources`` under ``plan``; ``signals`` there are optional and, given with
    ``plan.lean``, make S1 lean on them. ``flags_ns`` are regime-flag times
    (int64 ns); a guard policy without them builds the flags from the book for
    the market maker and is refused for the grid modes.
    """
    mode = check_pair(mode, regime_policy)
    if mode in (ExecutionMode.EVENT_TAKER, ExecutionMode.SIGNAL_ENTRY):
        if signals is None:
            raise ValueError(f"{mode.value} executes a signal: give signals")
        ts, side = _decisions(signals, regime_policy, flags_ns, guard_window_min)
        if mode is ExecutionMode.EVENT_TAKER:
            return _event_taker(signals, ts, side, books, taker_settings)
        guarded = regime_policy == "guard_pull"
        return _signal_entry(signals, ts, side, events, sim_config, guarded, flags_ns, passive_exit)
    if mode is ExecutionMode.MARKET_MAKER:
        return _market_maker(
            signals, symbol, days, regime_policy, flags_ns, guard_window_min,
            plan, sources, access, workers,
        )  # fmt: skip
    if signals is None or market is None:
        raise ValueError(f"{mode.value} executes a signal on a grid: give signals and market")
    decision = signals.on_grid(market.labels_ns)
    if regime_policy == "guard_pull":
        if flags_ns is None:
            raise ValueError("a guard policy on the grid needs the flag times")
        window = int(guard_window_min * 60 * NS_PER_S)
        decision[guarded_rows(market.labels_ns, flags_ns, window)] = 0
    hold_rows = max(1, round(signals.hold_ns / (grid_seconds * NS_PER_S)))
    if mode is ExecutionMode.TAKER:
        rules = taker_rules or ThinningRules(hold_periods=hold_rows)
        forward = market.forward_bp(rules.hold_periods)
        decision[~np.isfinite(forward)] = 0
        rows = run_taker(
            decision,
            np.nan_to_num(forward),
            market.spread_bp,
            rules,
            taker or TakerCosts(),
            mid=market.mid if rules.needs_price_path else None,
        )
    else:
        rows = run_passive(
            decision, market, posting or PostingRules(hold_rows=hold_rows), maker or MakerCosts()
        )
    attempts = _stamped(signals.symbol, market.labels_ns, rows)
    return ExecutionResult(attempts, summarise(attempts), mode)


def _market_maker(
    signals: SignalStream | None,
    symbol: str | None,
    days: Sequence[date],
    regime_policy: str,
    flags_ns: np.ndarray | None,
    guard_window_min: float,
    plan: mm.MakerPlan | None,
    sources: mm.MakerSources | None,
    access: Access | None,
    workers: int,
) -> ExecutionResult:
    if plan is None or not days:
        raise ValueError("the market maker needs a plan and the days to quote")
    name = symbol or (signals.symbol if signals is not None else None)
    if name is None:
        raise ValueError("the market maker needs a symbol")
    sources = sources or mm.MakerSources()
    access = access or mm.access_for(days)
    if regime_policy != "none":
        plan = replace(plan, regime_policy=regime_policy, guard_window_min=guard_window_min)
    tapes: dict[str, SignalTape] = {}
    if plan.regime_policy != "none" or plan.strategy == "S3":
        if flags_ns is None:
            tapes[FLAG_SIGNAL] = mm.flag_tape(name, days, sources, access)
        else:
            stamps = np.unique(np.asarray(flags_ns, dtype=np.int64))
            tapes[FLAG_SIGNAL] = SignalTape(FLAG_SIGNAL, stamps, stamps / 1e9)
    if signals is not None and plan.lean:
        signed = np.asarray(signals.direction, dtype=np.float64) * signals.strength
        tapes.update(mm.signal_tapes(signals.ts, signed))
        plan = replace(plan, lean_window_s=signals.hold_ns / NS_PER_S)
    frame = mm.run(name, days, plan, sources, access=access, workers=workers, tapes=tapes or None)
    attempts = mm.attempts(frame)
    summary = {**summarise(attempts), **mm.decomposition(frame)}
    return ExecutionResult(attempts, summary, ExecutionMode.MARKET_MAKER, frame)


# ---------------------------------------------------------------------------
# The event-time modes: the taker executor and the signal-entry quoter
# ---------------------------------------------------------------------------


def taker_attempts(symbol: str, day: TakerDay) -> pd.DataFrame:
    """One row per round trip of the event-time executor, in the attempts' shape.

    ``ts`` is the decision's stamp; ``cost_bp`` is the fees less the funding
    received, so ``net_bp`` = ``gross_bp`` - ``cost_bp``; ``path`` is the
    exit's reason (``hold`` or ``day_end``). Every entry the executor made
    filled; a skipped entry is a counter, not an attempt.
    """
    trades = day.trades
    notional = trades["notional"].to_numpy(dtype=np.float64)
    fees_bp = trades["fees"].to_numpy(dtype=np.float64) / notional * 1e4
    funding_bp = trades["funding"].to_numpy(dtype=np.float64) / notional * 1e4
    frame = pd.DataFrame(
        {
            "symbol": symbol,
            "day": day.day.isoformat(),
            "ts": trades["decision_ns"].to_numpy(dtype=np.int64),
            "filled": np.ones(len(trades), dtype=bool),
            "gross_bp": trades["gross_bp"].to_numpy(dtype=np.float64),
            "cost_bp": fees_bp - funding_bp,
            "net_bp": trades["net_bp"].to_numpy(dtype=np.float64),
            "path": trades["exit_reason"].astype(str).to_numpy(),
            "direction": trades["side"].to_numpy(dtype=np.int64),
            "notional": notional,
            "fees_bp": fees_bp,
            "funding_bp": funding_bp,
        }
    )
    return frame[[*ATTEMPT_COLUMNS, *EVENT_EXTRA_COLUMNS]]


def signal_entry_attempts(
    symbol: str, day: date, table: pd.DataFrame, decision_ns: np.ndarray
) -> pd.DataFrame:
    """One row per posted entry of the signal-entry quoter (its :meth:`attempts`).

    An entry without a fill is an attempt that earned nothing (``unfilled``);
    a filled one is ``round_trip``. Funding is not in the attempts: the quoter
    leaves it in the day's totals (the result's ``days`` table). ``ts`` is the
    decision's stamp, recovered exactly from the tape's seconds.
    """
    stamps = np.asarray(decision_ns, dtype=np.int64)
    exact = dict(zip((stamps / 1e9).tolist(), stamps.tolist(), strict=True))
    notional = table["notional"].to_numpy(dtype=np.float64)
    fees_bp = table["fees"].to_numpy(dtype=np.float64) / notional * 1e4
    filled = table["filled"].to_numpy(dtype=bool)
    frame = pd.DataFrame(
        {
            "symbol": symbol,
            "day": day.isoformat(),
            "ts": np.array([exact[s] for s in table["decision_s"]], dtype=np.int64),
            "filled": filled,
            "gross_bp": table["gross"].to_numpy(dtype=np.float64) / notional * 1e4,
            "cost_bp": fees_bp,
            "net_bp": table["net"].to_numpy(dtype=np.float64) / notional * 1e4,
            "path": np.where(filled, "round_trip", "unfilled"),
            "direction": table["side"].to_numpy(dtype=np.int64),
            "notional": notional,
            "fees_bp": fees_bp,
            "funding_bp": np.zeros(len(table)),
        }
    )
    return frame[[*ATTEMPT_COLUMNS, *EVENT_EXTRA_COLUMNS]]


#: Columns the event-time modes' attempts carry after :data:`ATTEMPT_COLUMNS`.
EVENT_EXTRA_COLUMNS: tuple[str, ...] = ("direction", "notional", "fees_bp", "funding_bp")


def _decisions(
    signals: SignalStream, regime_policy: str, flags_ns: np.ndarray | None, guard_window_min: float
) -> tuple[np.ndarray, np.ndarray]:
    """The stream's entry decisions (stamps, sides), the guarded ones removed."""
    ts = np.asarray(signals.ts, dtype=np.int64)
    side = np.asarray(signals.direction, dtype=np.int64)
    keep = side != 0
    if regime_policy == "guard_pull":
        if flags_ns is None:
            raise ValueError("a guard policy needs the flag times")
        keep &= ~guarded_rows(ts, flags_ns, int(guard_window_min * 60 * NS_PER_S))
    return ts[keep], side[keep]


def _on_day(ts: np.ndarray, day: date) -> np.ndarray:
    start = int(pd.Timestamp(day).value)
    return (ts >= start) & (ts < start + NS_PER_DAY)


def _event_taker(
    signals: SignalStream,
    ts: np.ndarray,
    side: np.ndarray,
    books: Sequence[TakerBook],
    settings: TakerSettings | None,
) -> ExecutionResult:
    attempts, days = [], []
    for book in sorted(books, key=lambda b: b.day):
        mine = _on_day(ts, book.day)
        day = execute_day(book, ts[mine], side[mine], signals.hold_ns, settings)
        attempts.append(taker_attempts(signals.symbol, day))
        days.append(
            {
                "symbol": signals.symbol,
                "day": book.day.isoformat(),
                "trades": float(len(day.trades)),
                "net_usdt": day.net,
                "turnover": day.turnover,
                "net_bp_day": day.day_bp(),
                "flags": ",".join(day.flags),
                **day.counters,
            }
        )
    return _event_result(attempts, days, ExecutionMode.EVENT_TAKER)


def _signal_entry(
    signals: SignalStream,
    ts: np.ndarray,
    side: np.ndarray,
    events: Sequence[DayEvents],
    config: SimConfig | None,
    guarded: bool,
    flags_ns: np.ndarray | None,
    passive_exit: bool,
) -> ExecutionResult:
    if config is None:
        raise ValueError("the signal-entry quoter needs the simulator's settings")
    if guarded:
        config = config.with_(cancel_entries_on_flag=True)
    flags = np.unique(np.asarray(flags_ns if flags_ns is not None else [], dtype=np.int64))
    attempts, days = [], []
    for day_events in sorted(events, key=lambda e: e.day):
        mine = _on_day(ts, day_events.day)
        tapes = entry_tapes(ts[mine], side[mine]) if mine.any() else {}
        today = flags[_on_day(flags, day_events.day)]
        if guarded and len(today):
            tapes[FLAG_SIGNAL] = SignalTape(FLAG_SIGNAL, today, today / 1e9)
        quoter = SignalEntryQuoter(signals.hold_ns / NS_PER_S, passive_exit=passive_exit)
        result = simulate_day(day_events, quoter, config, tapes or None)
        table = quoter.attempts(result.fills)
        attempts.append(signal_entry_attempts(signals.symbol, day_events.day, table, ts[mine]))
        row = {"symbol": signals.symbol, **_summary(result, quoter.name)}
        row.update({f"quoter_{k}": float(v) for k, v in sorted(quoter.counts.items())})
        row.update({k: float(result.counters.get(k, 0.0)) for k in EXTRA_COUNTERS})
        days.append(row)
    return _event_result(attempts, days, ExecutionMode.SIGNAL_ENTRY)


def _event_result(
    attempts: list[pd.DataFrame], days: list[dict[str, object]], mode: ExecutionMode
) -> ExecutionResult:
    columns = [*ATTEMPT_COLUMNS, *EVENT_EXTRA_COLUMNS]
    frames = [frame for frame in attempts if len(frame)]
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    return ExecutionResult(table, summarise(table), mode, pd.DataFrame(days))
