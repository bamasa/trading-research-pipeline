"""The event-time taker executor: a clip crossed on a decision and crossed back after the hold.

Every taker fill of the short-horizon study comes from here; the older
``pipeline/execution.run_taker`` prices mid to mid and is not used for it. The
rules, in the order they apply to one decision at a row stamp ``t``:

* **Entry hours.** No entry before 00:01 or after 23:58 UTC
  (``entries_outside_hours``).
* **One position at a time.** A decision is taken only once the previous
  position is closed, at the stamp of the snapshot its exit walked or later
  (``entries_busy`` otherwise).
* **Entry.** The order arrives at ``t`` + latency and walks the first snapshot
  stamped at or after its arrival, across the visible levels, at their
  volume-weighted price for the clip (:func:`.walk.walk_book`), with nothing
  added on top. A clip larger than that snapshot's visible depth on the side
  taken is skipped (``entries_skipped``); no entry is skipped for its
  snapshot's age. An entry no snapshot follows is not made
  (``entries_no_snapshot``).
* **Exit.** Decided at the first row stamped at or after ``t + hold`` (at
  ``t + hold`` itself if the book has no such row), or at the day's flatten
  time if that comes first; it arrives after the latency and walks the first
  snapshot at or after its arrival whatever its age. Size beyond the visible
  levels is charged at the deepest level's price
  (``flatten_beyond_visible_book``). An exit no snapshot follows walks the
  last one; if that snapshot is more than ``stale_after_ns`` older than the
  order, the day is flagged ``flatten_stale``.
* **Late snapshots.** An order whose walked snapshot is stamped more than
  ``late_snapshot_ns`` after its arrival (a feed gap) is counted as
  ``entry_late_snapshot`` or ``exit_late_snapshot``; it is a measurement and
  excludes nothing.
* **Money.** Each fill pays the taker fee on its notional. Funding is
  settled at each settlement on the position held then (an entry filled at a
  settlement's exact stamp is not yet held; an exit filled there still is),
  at the mid of the last snapshot strictly before it. The account's
  identity is checked after every change and at the day's end, with turnover
  equal to the sum of the fill notionals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, time

import numpy as np
import pandas as pd

from trading_research.backtest.costs import BYBIT_BASE, FeeTier
from trading_research.backtest.walk import book_arrays, walk_book
from trading_research.market_making.accounting import Account, IdentityError
from trading_research.market_making.events import NS_PER_S, DayEvents, day_start_ns, to_ns

#: A walked snapshot stamped more than this after the order's arrival is late.
LATE_SNAPSHOT_NS = NS_PER_S
#: Counter names, shared with the simulator's taker orders.
ENTRY_LATE, EXIT_LATE = "entry_late_snapshot", "exit_late_snapshot"
BEYOND_BOOK = "flatten_beyond_visible_book"


def _time_ns(moment: time) -> int:
    return (moment.hour * 3600 + moment.minute * 60 + moment.second) * NS_PER_S + (
        moment.microsecond * 1000
    )


@dataclass(frozen=True)
class TakerBook:
    """One day's snapshots, prices in quote units, and its funding settlements."""

    day: date
    ts: np.ndarray
    bid_px: np.ndarray
    bid_sz: np.ndarray
    ask_px: np.ndarray
    ask_sz: np.ndarray
    funding_ts: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    funding_rate: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def __post_init__(self) -> None:
        rows = len(self.ts)
        for plane in (self.bid_px, self.bid_sz, self.ask_px, self.ask_sz):
            if plane.ndim != 2 or plane.shape[0] != rows:
                raise ValueError("every plane needs one (depth,) row per snapshot")
        if rows == 0:
            raise ValueError("a day with no snapshot cannot be traded")
        if rows > 1 and np.any(np.diff(self.ts) <= 0):
            raise ValueError("snapshot stamps must be strictly increasing")
        if len(self.funding_ts) != len(self.funding_rate):
            raise ValueError("one funding rate per settlement")

    @property
    def mid(self) -> np.ndarray:
        return 0.5 * (self.bid_px[:, 0] + self.ask_px[:, 0])

    @classmethod
    def from_events(cls, events: DayEvents) -> TakerBook:
        """The simulator's day, its integer ticks turned back into prices."""
        tick = events.spec.tick
        return cls(
            events.day,
            np.asarray(events.book_ts, dtype=np.int64),
            events.bid_px * tick,
            np.asarray(events.bid_sz, dtype=np.float64),
            events.ask_px * tick,
            np.asarray(events.ask_sz, dtype=np.float64),
            np.asarray(events.funding_ts, dtype=np.int64),
            np.asarray(events.funding_rate, dtype=np.float64),
        )

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        day: date,
        *,
        depth: int = 10,
        funding: pd.DataFrame | None = None,
    ) -> TakerBook:
        """From a book frame (``timestamp`` and the schema's level columns) and,
        optionally, a funding frame (``timestamp``, ``rate``)."""
        stamps = to_ns(frame["timestamp"])
        bid_px, bid_sz, ask_px, ask_sz = book_arrays(frame, depth)
        if funding is None:
            return cls(day, stamps, bid_px, bid_sz, ask_px, ask_sz)
        return cls(
            day,
            stamps,
            bid_px,
            bid_sz,
            ask_px,
            ask_sz,
            to_ns(funding["timestamp"]),
            funding["rate"].to_numpy(dtype=np.float64),
        )


@dataclass(frozen=True, kw_only=True)
class TakerSettings:
    """What the executor depends on besides the book. Defaults are the registered ones."""

    clip: float = 0.010
    fees: FeeTier = BYBIT_BASE
    latency_ns: int = 10_000_000
    late_snapshot_ns: int = LATE_SNAPSHOT_NS
    stale_after_ns: int = 5 * NS_PER_S
    no_entry_before: time = time(0, 1)
    no_entry_after: time = time(23, 58)
    flatten_at: time = time(23, 59)

    def __post_init__(self) -> None:
        if not self.clip > 0:
            raise ValueError(f"the clip must be positive, got {self.clip}")
        if self.latency_ns < 0:
            raise ValueError("latency cannot be negative")
        if not self.no_entry_after <= self.flatten_at:
            raise ValueError("the flatten cannot come before the last entry")


#: The per-trade table's columns, in order.
TRADE_COLUMNS: tuple[str, ...] = (
    "decision_ns",
    "side",
    "size",
    "entry_arrival_ns",
    "entry_snapshot_ns",
    "entry_price",
    "entry_mid",
    "exit_decision_ns",
    "exit_arrival_ns",
    "exit_snapshot_ns",
    "exit_price",
    "exit_mid",
    "exit_reason",
    "exit_beyond",
    "entry_late",
    "exit_late",
    "notional",
    "gross",
    "fees",
    "funding",
    "net",
    "gross_bp",
    "net_bp",
)

#: Counters every day reports, zero when nothing happened.
COUNTERS: tuple[str, ...] = (
    "decisions",
    "entries",
    "entries_outside_hours",
    "entries_busy",
    "entries_skipped",
    "entries_no_snapshot",
    ENTRY_LATE,
    EXIT_LATE,
    BEYOND_BOOK,
    "funding_settlements_with_position",
)


@dataclass(frozen=True)
class TakerDay:
    """One day of the executor: its round trips, counters and flags."""

    day: date
    trades: pd.DataFrame
    counters: dict[str, float]
    #: ``flatten_stale`` when an exit had no book within ``stale_after_ns``.
    flags: tuple[str, ...]
    net: float
    turnover: float

    @property
    def excluded(self) -> bool:
        return bool(self.flags)

    def day_bp(self, column: str = "net") -> float:
        """The day's value: the sum of ``column`` (USDT) over the entry notionals, x 1e4.

        NaN for a day without a trade, which has no value in a per-trade series.
        """
        notional = float(self.trades["notional"].sum())
        if notional <= 0:
            return float("nan")
        return float(self.trades[column].sum()) / notional * 1e4


def _side(book: TakerBook, side: int, row: int) -> tuple[np.ndarray, np.ndarray]:
    """The levels a taker on ``side`` (+1 buys, -1 sells) walks at ``row``."""
    if side > 0:
        return book.ask_px[row], book.ask_sz[row]
    return book.bid_px[row], book.bid_sz[row]


def execute_day(
    book: TakerBook,
    decision_ns: np.ndarray,
    side: np.ndarray,
    hold_ns: int,
    settings: TakerSettings | None = None,
) -> TakerDay:
    """Execute one day's decisions (row stamps, time order) with sides +1 / -1."""
    settings = settings or TakerSettings()
    decisions = np.asarray(decision_ns, dtype=np.int64)
    sides = np.asarray(side, dtype=np.int64)
    if len(decisions) != len(sides):
        raise ValueError("one side per decision")
    if len(decisions) > 1 and np.any(np.diff(decisions) < 0):
        raise ValueError("decisions must be in time order")
    if not np.isin(sides, (-1, 1)).all():
        raise ValueError("a side is +1 (buy) or -1 (sell)")
    if hold_ns <= 0:
        raise ValueError("the hold must be positive")
    ts, mid, n = book.ts, book.mid, len(book.ts)
    start = day_start_ns(book.day)
    first_ok = start + _time_ns(settings.no_entry_before)
    last_ok = start + _time_ns(settings.no_entry_after)
    flatten_ns = start + _time_ns(settings.flatten_at)
    latency, clip, fee_bp = settings.latency_ns, settings.clip, settings.fees.taker_bp
    entry_rows = np.searchsorted(ts, decisions + latency, side="left")
    hold_rows = np.searchsorted(ts, decisions + hold_ns, side="left")
    counters = dict.fromkeys(COUNTERS, 0.0)
    account = Account()
    free_at = np.iinfo(np.int64).min
    stale = False
    rows: list[tuple[object, ...]] = []
    for k in range(len(decisions)):
        t, s = int(decisions[k]), int(sides[k])
        counters["decisions"] += 1
        if t < first_ok or t > last_ok:
            counters["entries_outside_hours"] += 1
            continue
        if t < free_at:
            counters["entries_busy"] += 1
            continue
        e = int(entry_rows[k])
        if e >= n:
            counters["entries_no_snapshot"] += 1
            continue
        entry = walk_book(*_side(book, s, e), clip)
        if entry.beyond[0] > 1e-9 * clip:
            counters["entries_skipped"] += 1
            continue
        h = int(hold_rows[k])
        exit_decision, reason = (int(ts[h]) if h < n else t + hold_ns), "hold"
        if exit_decision > flatten_ns:
            exit_decision, reason = flatten_ns, "day_end"
        exit_arrival = exit_decision + latency
        x = max(int(np.searchsorted(ts, exit_arrival, side="left")), e)
        if x >= n:
            x = n - 1
            stale = stale or exit_arrival - int(ts[x]) > settings.stale_after_ns
        exit_walk = walk_book(*_side(book, -s, x), clip)
        entry_ts, exit_ts = int(ts[e]), max(int(ts[x]), exit_arrival)
        entry_price, exit_price = float(entry.vwap[0]), float(exit_walk.vwap[0])
        fees = account.fill(s, entry_price, clip, fee_bp, maker=False)
        account.check_identity(float(mid[e]))
        funding = _settle(book, account, entry_ts, exit_ts, counters)
        fees += account.fill(-s, exit_price, clip, fee_bp, maker=False)
        account.check_identity(float(mid[x]))
        beyond = float(exit_walk.beyond[0])
        counters[BEYOND_BOOK] += float(beyond > 0.0)
        entry_late = entry_ts - (t + latency) > settings.late_snapshot_ns
        exit_late = int(ts[x]) - exit_arrival > settings.late_snapshot_ns
        counters[ENTRY_LATE] += float(entry_late)
        counters[EXIT_LATE] += float(exit_late)
        counters["entries"] += 1
        free_at = exit_ts
        notional = entry_price * clip
        gross = s * (exit_price - entry_price) * clip
        net = gross - fees + funding
        rows.append(
            (
                t,
                s,
                clip,
                t + latency,
                entry_ts,
                entry_price,
                float(mid[e]),
                exit_decision,
                exit_arrival,
                int(ts[x]),
                exit_price,
                float(mid[x]),
                reason,
                beyond,
                entry_late,
                exit_late,
                notional,
                gross,
                fees,
                funding,
                net,
                gross / notional * 1e4,
                net / notional * 1e4,
            )
        )
    return _finish(book, account, rows, counters, stale)


def _settle(
    book: TakerBook, account: Account, entry_ts: int, exit_ts: int, counters: dict[str, float]
) -> float:
    """Funding on the position held from ``entry_ts`` (exclusive) to ``exit_ts``
    (inclusive), each settlement marked at the mid strictly before it."""
    stamps = book.funding_ts
    first = int(np.searchsorted(stamps, entry_ts, side="right"))
    last = int(np.searchsorted(stamps, exit_ts, side="right"))
    paid = 0.0
    for j in range(first, last):
        row = int(np.searchsorted(book.ts, stamps[j], side="left")) - 1
        mark = float(book.mid[row])
        paid += account.settle_funding(float(book.funding_rate[j]), mark)
        account.check_identity(mark)
        counters["funding_settlements_with_position"] += 1
    return paid


def _finish(
    book: TakerBook,
    account: Account,
    rows: list[tuple[object, ...]],
    counters: dict[str, float],
    stale: bool,
) -> TakerDay:
    trades = pd.DataFrame(rows, columns=list(TRADE_COLUMNS))
    if account.position != 0.0:
        raise IdentityError(f"the day ended holding {account.position!r}")
    account.check_identity(float(book.mid[-1]))
    turnover = float((trades["entry_price"] + trades["exit_price"]).mul(trades["size"]).sum())
    scale = max(1.0, account.taker_turnover)
    if abs(turnover - account.taker_turnover) > 1e-9 * scale:
        raise IdentityError(
            f"turnover {account.taker_turnover!r} differs from the fills' notionals {turnover!r}"
        )
    if abs(float(trades["net"].sum()) - account.cash) > 1e-9 * scale:
        raise IdentityError(f"the trades' net differs from cash {account.cash!r}")
    return TakerDay(
        book.day,
        trades,
        counters,
        ("flatten_stale",) if stale else (),
        account.cash,
        account.taker_turnover,
    )
