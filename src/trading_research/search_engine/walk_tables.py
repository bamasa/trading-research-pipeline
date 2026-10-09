"""Walk tables: every 100 ms row's taker prices for the clip, and each hold's walked P&L.

Stage B scores 2,400 cells on 150 days by looking up, never by re-executing.
For one day, :class:`WalkTable` walks both sides of every row's book for the
clip with :func:`trading_research.backtest.walk.walk_book`, the function the
event-time taker executor (:mod:`trading_research.backtest.taker`) prices
every fill with: the buy VWAP (asks), the sell VWAP (bids) and what each walk
left beyond the visible book. For one day and hold, :class:`HoldTable` applies
the executor's rules to a decision at every row and stores the long and the
short round trip's net in USDT and in bp of the entry notional, fees and
funding included, with the arithmetic in the executor's own order, so a
looked-up trade equals the executor's to the last bit:

* the entry walks the first row stamped at or after ``t`` + latency; it is
  skipped when the clip goes beyond the visible book (more than ``1e-9`` of
  the clip) and not made when no row follows;
* the exit is decided at the first row stamped at or after ``t`` + hold (at
  ``t`` + hold itself when none follows), or at the flatten time if that comes
  first, and walks the first row at or after the decision plus latency, never
  one before the entry's, the last row when the book ends;
* the next decision is accepted from ``max(exit row's stamp, exit arrival)``;
  ``free_row`` is the first row stamped at or after it;
* no entry is decided before 00:01 or after 23:58 UTC;
* funding is settled on the position held in ``(entry stamp, exit stamp]`` at
  the mid of the last row strictly before each settlement.

Both tables live in memory one day at a time and are never stored.
:func:`execute_rows` runs the executor's decision loop over a table, for the
parity test and for Null B's route; :func:`parity` is the stage-0 test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, time

import numpy as np

from trading_research.backtest.taker import TakerBook, TakerSettings, execute_day
from trading_research.backtest.walk import walk_book
from trading_research.market_making.events import day_start_ns

#: An entry whose walk leaves more than this share of the clip beyond the
#: visible book is skipped, as the executor skips it.
SKIP_SHARE = 1e-9
LONG, SHORT = 0, 1


def clock_ns(moment: time) -> int:
    """A time of day as nanoseconds after midnight."""
    seconds = moment.hour * 3600 + moment.minute * 60 + moment.second
    return seconds * 1_000_000_000 + moment.microsecond * 1000


def side_index(side: np.ndarray) -> np.ndarray:
    """0 for a long (+1), 1 for a short (-1): the first axis of a hold table."""
    return np.where(np.asarray(side) > 0, LONG, SHORT)


@dataclass(frozen=True)
class WalkTable:
    """One day: each row's buy and sell VWAP for the clip and what lay beyond."""

    day: date
    ts: np.ndarray
    mid: np.ndarray
    buy_vwap: np.ndarray
    sell_vwap: np.ndarray
    buy_beyond: np.ndarray
    sell_beyond: np.ndarray
    clip: float
    funding_ts: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    funding_rate: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @classmethod
    def from_book(cls, book: TakerBook, clip: float) -> WalkTable:
        """Walk both sides of every row of the executor's book for ``clip``."""
        buy = walk_book(book.ask_px, book.ask_sz, clip)
        sell = walk_book(book.bid_px, book.bid_sz, clip)
        return cls(
            book.day,
            np.asarray(book.ts, dtype=np.int64),
            book.mid,
            buy.vwap,
            sell.vwap,
            buy.beyond,
            sell.beyond,
            float(clip),
            np.asarray(book.funding_ts, dtype=np.int64),
            np.asarray(book.funding_rate, dtype=np.float64),
        )

    @property
    def rows(self) -> int:
        return len(self.ts)

    @property
    def buy_exceeds(self) -> np.ndarray:
        """Buying the clip at this row goes beyond the visible asks."""
        return self.buy_beyond > SKIP_SHARE * self.clip

    @property
    def sell_exceeds(self) -> np.ndarray:
        """Selling the clip at this row goes beyond the visible bids."""
        return self.sell_beyond > SKIP_SHARE * self.clip

    @property
    def start_ns(self) -> int:
        return day_start_ns(self.day)


@dataclass(frozen=True)
class HoldTable:
    """One day and hold: the executor's round trip for a decision at every row.

    Arrays have one value per row (the decision row); ``net``, ``notional`` and
    ``pnl_bp`` are ``(2, rows)``, long first. Where no entry is possible
    (``has_entry`` false) the money is NaN.
    """

    walk: WalkTable
    hold_ns: int
    settings: TakerSettings
    entry_row: np.ndarray
    exit_row: np.ndarray
    free_ns: np.ndarray
    free_row: np.ndarray
    allowed: np.ndarray
    stale: np.ndarray
    skip: np.ndarray
    net: np.ndarray
    notional: np.ndarray
    pnl_bp: np.ndarray

    @property
    def rows(self) -> int:
        return self.walk.rows

    @property
    def has_entry(self) -> np.ndarray:
        return self.entry_row < self.rows

    @property
    def busy_until(self) -> np.ndarray:
        """The last row a position decided here keeps busy: ``thin_entries``'
        ``busy_until``, so the next decision may be taken from ``free_row``."""
        return self.free_row - 1

    def takeable(self, side: np.ndarray) -> np.ndarray:
        """Rows where a decision on ``side`` would be filled if the book were free:
        inside entry hours, a snapshot to enter on, the clip within the book."""
        side = np.asarray(side)
        rows = np.arange(self.rows)
        return self.allowed & self.has_entry & ~self.skip[side_index(side), rows]


def hold_table(walk: WalkTable, hold_ns: int, settings: TakerSettings | None = None) -> HoldTable:
    """Apply the executor's rules to a decision at every row of ``walk``."""
    settings = settings or TakerSettings()
    if hold_ns <= 0:
        raise ValueError("the hold must be positive")
    if settings.clip != walk.clip:
        raise ValueError(f"the table walked {walk.clip}, the settings trade {settings.clip}")
    ts, n, latency, clip = walk.ts, walk.rows, settings.latency_ns, settings.clip
    start = walk.start_ns
    allowed = (ts >= start + clock_ns(settings.no_entry_before)) & (
        ts <= start + clock_ns(settings.no_entry_after)
    )
    flatten_ns = start + clock_ns(settings.flatten_at)
    entry = np.searchsorted(ts, ts + latency, side="left").astype(np.int64)
    known = entry < n
    e = np.minimum(entry, n - 1)
    h = np.searchsorted(ts, ts + hold_ns, side="left")
    decided = np.where(h < n, ts[np.minimum(h, n - 1)], ts + hold_ns)
    decided = np.minimum(decided, flatten_ns)
    arrival = decided + latency
    x = np.maximum(np.searchsorted(ts, arrival, side="left"), e)
    ended = x >= n
    x = np.minimum(x, n - 1)
    stale = known & ended & (arrival - ts[n - 1] > settings.stale_after_ns)
    entry_ts, exit_ts = ts[e], np.maximum(ts[x], arrival)
    free_row = np.searchsorted(ts, exit_ts, side="left").astype(np.int64)
    fee_bp = settings.fees.taker_bp
    marks = walk.mid[np.searchsorted(ts, walk.funding_ts, side="left") - 1]
    net = np.full((2, n), np.nan)
    notional = np.full((2, n), np.nan)
    skip = np.zeros((2, n), dtype=bool)
    sides = ((LONG, 1, walk.buy_vwap, walk.sell_vwap, walk.buy_exceeds),
             (SHORT, -1, walk.sell_vwap, walk.buy_vwap, walk.sell_exceeds))  # fmt: skip
    for index, s, enter_at, exit_at, exceeds in sides:
        entry_price, exit_price = enter_at[e], exit_at[x]
        fees = entry_price * clip * fee_bp * 1e-4
        fees = fees + exit_price * clip * fee_bp * 1e-4
        position = 0.0 + s * clip
        funding = np.zeros(n)
        for stamp, rate, mark in zip(walk.funding_ts, walk.funding_rate, marks, strict=True):
            held = (entry_ts < stamp) & (stamp <= exit_ts)
            funding = np.where(held, funding + -position * mark * rate, funding)
        gross = s * (exit_price - entry_price) * clip
        value = gross - fees + funding
        notional[index] = np.where(known, entry_price * clip, np.nan)
        net[index] = np.where(known, value, np.nan)
        skip[index] = known & exceeds[e]
    pnl_bp = net / notional * 1e4
    return HoldTable(walk, int(hold_ns), settings, np.where(known, entry, n), x, exit_ts,
                     free_row, allowed, stale, skip, net, notional, pnl_bp)  # fmt: skip


#: The executor's decision counters a table run reports.
TABLE_COUNTERS: tuple[str, ...] = (
    "decisions",
    "entries",
    "entries_outside_hours",
    "entries_busy",
    "entries_skipped",
    "entries_no_snapshot",
)


@dataclass(frozen=True)
class TableTrades:
    """The round trips taken from a hold table, in time order; money on the
    trade's own side except the two walked P&Ls, which are both sides'."""

    rows: np.ndarray
    side: np.ndarray
    net: np.ndarray
    notional: np.ndarray
    pnl_long_bp: np.ndarray
    pnl_short_bp: np.ndarray
    stale: bool
    counters: dict[str, int]

    @property
    def trades(self) -> int:
        return len(self.rows)

    @property
    def net_bp(self) -> np.ndarray:
        return np.where(self.side > 0, self.pnl_long_bp, self.pnl_short_bp)

    @property
    def g_bp(self) -> np.ndarray:
        """G per trade: ``side x (pnl_long_bp - pnl_short_bp) / 2``, the costs out."""
        return self.side * (self.pnl_long_bp - self.pnl_short_bp) / 2.0

    def day_values(self) -> dict[str, float | int]:
        """The cell-day's ledger values: ``net_bp`` is the day's net over its entry
        notional (``TakerDay.day_bp``), ``G_bp`` and the two walked P&Ls the means
        over the day's trades; all 0 on a day without a trade."""
        skips = int(self.counters["entries_skipped"])
        if self.trades == 0:
            zero = {"net_bp": 0.0, "G_bp": 0.0, "pnl_long_bp": 0.0, "pnl_short_bp": 0.0}
            return {**zero, "trades": 0, "skips": skips}
        return {
            "net_bp": float(self.net.sum() / self.notional.sum() * 1e4),
            "G_bp": float(self.g_bp.mean()),
            "pnl_long_bp": float(self.pnl_long_bp.mean()),
            "pnl_short_bp": float(self.pnl_short_bp.mean()),
            "trades": self.trades,
            "skips": skips,
        }


def take(
    table: HoldTable, rows: np.ndarray, side: np.ndarray, counters: dict[str, int]
) -> TableTrades:
    """The trades at decision ``rows`` already known to be free and takeable."""
    rows = np.asarray(rows, dtype=np.int64)
    side = np.asarray(side, dtype=np.int64)
    own = side_index(side)
    return TableTrades(
        rows,
        side,
        table.net[own, rows],
        table.notional[own, rows],
        table.pnl_bp[LONG, rows],
        table.pnl_bp[SHORT, rows],
        bool(table.stale[rows].any()),
        counters,
    )


def execute_rows(table: HoldTable, rows: np.ndarray, side: np.ndarray) -> TableTrades:
    """The executor's decision loop over a table: decisions at row positions in
    time order, sides +1 / -1; outside hours, busy, unfillable and beyond-book
    decisions are counted as the executor counts them and take nothing."""
    rows = np.asarray(rows, dtype=np.int64)
    side = np.asarray(side, dtype=np.int64)
    if rows.shape != side.shape:
        raise ValueError("one side per decision")
    if len(rows) and (np.any(np.diff(rows) < 0) or rows[0] < 0 or rows[-1] >= table.rows):
        raise ValueError("decision rows must be row positions in time order")
    if not np.isin(side, (-1, 1)).all():
        raise ValueError("a side is +1 (buy) or -1 (sell)")
    ts, allowed, entry, free = table.walk.ts, table.allowed, table.entry_row, table.free_ns
    skip, n = table.skip, table.rows
    counters = dict.fromkeys(TABLE_COUNTERS, 0)
    free_at = np.iinfo(np.int64).min
    taken: list[int] = []
    for k, (row, s) in enumerate(zip(rows.tolist(), side.tolist(), strict=True)):
        counters["decisions"] += 1
        if not allowed[row]:
            counters["entries_outside_hours"] += 1
        elif ts[row] < free_at:
            counters["entries_busy"] += 1
        elif entry[row] >= n:
            counters["entries_no_snapshot"] += 1
        elif skip[LONG if s > 0 else SHORT, row]:
            counters["entries_skipped"] += 1
        else:
            taken.append(k)
            counters["entries"] += 1
            free_at = int(free[row])
    return take(table, rows[taken], side[taken], counters)


@dataclass(frozen=True)
class Parity:
    """Executor against table, decision by decision; ``ok`` when nothing differs."""

    trades: int
    mismatches: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.mismatches


def entry_parity(
    book: TakerBook,
    rows: np.ndarray,
    side: np.ndarray,
    table: HoldTable,
) -> Parity:
    """Execute the same decisions with ``backtest.taker.execute_day`` and with
    the table, and list every difference: which decisions became trades, their
    entry and exit snapshots, net, notional and net bp (to the last bit), the
    counters and the stale-day flag."""
    rows = np.asarray(rows, dtype=np.int64)
    ours = execute_rows(table, rows, side)
    day = execute_day(book, book.ts[rows], side, table.hold_ns, table.settings)
    theirs, out = day.trades, []
    if len(theirs) != ours.trades:
        out.append(f"trades: executor {len(theirs)}, table {ours.trades}")
    else:
        ts = table.walk.ts
        checks = {
            "decision_ns": ts[ours.rows],
            "side": ours.side,
            "entry_snapshot_ns": ts[table.entry_row[ours.rows]],
            "exit_snapshot_ns": ts[table.exit_row[ours.rows]],
            "net": ours.net,
            "notional": ours.notional,
            "net_bp": ours.net_bp,
        }
        for column, values in checks.items():
            if not np.array_equal(theirs[column].to_numpy(), values):
                out.append(f"{column} differs")
    for name in TABLE_COUNTERS:
        if int(day.counters[name]) != ours.counters[name]:
            out.append(f"{name}: executor {day.counters[name]:g}, table {ours.counters[name]}")
    if day.excluded != ours.stale:
        out.append(f"stale day: executor {day.excluded}, table {ours.stale}")
    return Parity(ours.trades, tuple(out))
