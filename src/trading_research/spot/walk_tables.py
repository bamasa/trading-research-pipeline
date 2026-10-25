"""Print walk tables: both sides' round trip for a decision at every row of the print grid.

The print slots' placebos, K-drift's all-long and all-short runs and Null A's
hour flips are scored by looking up, on the pattern of
:mod:`trading_research.search_engine.walk_tables`. For one day, hold and
settings (tier, latency, bracket), :func:`print_hold_table` applies the print
executor's rules (:mod:`.print_taker`) to a decision at every row and stores,
for each side, the entry fill's print, the exit's decision, print and fill
time, the time the position keeps the account busy until, the prices, fees and
the net in quote and in bp of the entry notional, with the arithmetic in the
executor's own order, so a looked-up trade equals the executor's to the last
bit. Fills come from the executor's rule (``print_taker``'s one import of the
fill function) and nothing else.

On a book a taker's exit time does not depend on its side; on prints it does,
since a buy and a sell fill at different prints. So the table keeps the busy
time per side (``free_ns``, ``(2, rows)``), and :func:`execute_rows` re-applies
the one-position rule to the run's own fills: a decision stamped before the
run's own previous exit fill is skipped and counted (``entries_busy``).
:func:`print_entry_parity` is the stage-0 test against the executor.

G and the two walked P&Ls are the 2025 ledger's: per trade,
``G = side x (pnl_long_bp - pnl_short_bp) / 2`` at the trade's decision row. On
prints one side can have no fill at a row where the other has one; such a
trade has no G, and the day's means are over the trades that have one.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from trading_research.market_making.events import day_start_ns
from trading_research.search_engine.walk_tables import Parity
from trading_research.spot.costs import fill_fee
from trading_research.spot.print_taker import (
    BEYOND_SHARE,
    NS_PER_DAY,
    PRINT_COUNTERS,
    DayPrints,
    PrintSettings,
    clock_ns,
    day_sweeps,
    execute_print_day,
    fills,
    no_print_exits,
)

LONG, SHORT = 0, 1
SIDES: tuple[tuple[int, int], ...] = ((LONG, 1), (SHORT, -1))


def side_index(side: np.ndarray) -> np.ndarray:
    """0 for a long (+1), 1 for a short (-1): the first axis of a table."""
    return np.where(np.asarray(side) > 0, LONG, SHORT)


@dataclass(frozen=True)
class PrintHoldTable:
    """One day and hold: the print executor's round trip for a decision at every row.

    Per-row arrays have one value per row of the print grid; per-side arrays are
    ``(2, rows)``, long first. Where a side has no entry (``has_entry`` false)
    its stamps are -1 and its money NaN.
    """

    day: date
    row_ns: np.ndarray
    hold_ns: int
    settings: PrintSettings
    allowed: np.ndarray
    has_entry: np.ndarray
    entry_print_ns: np.ndarray
    entry_price: np.ndarray
    entry_beyond: np.ndarray
    entry_late: np.ndarray
    exit_decision_ns: np.ndarray
    exit_print_ns: np.ndarray
    exit_price: np.ndarray
    exit_beyond: np.ndarray
    exit_late: np.ndarray
    exit_no_print: np.ndarray
    exit_either_side: np.ndarray
    #: The exit's fill time: the next decision on that side's run may come at it.
    free_ns: np.ndarray
    notional: np.ndarray
    gross: np.ndarray
    fees: np.ndarray
    net: np.ndarray
    pnl_bp: np.ndarray

    @property
    def rows(self) -> int:
        return len(self.row_ns)

    def takeable(self, side: np.ndarray) -> np.ndarray:
        """Rows where a decision on ``side`` would be filled if the account were free."""
        own = side_index(side)
        return self.allowed & self.has_entry[own, np.arange(self.rows)]


def print_hold_table(
    prints: DayPrints, row_ns: np.ndarray, hold_ns: int, settings: PrintSettings
) -> PrintHoldTable:
    """Apply the print executor's rules to a decision at every row of ``row_ns``."""
    if hold_ns <= 0:
        raise ValueError("the hold must be positive")
    ts = np.asarray(row_ns, dtype=np.int64)
    if len(ts) > 1 and np.any(np.diff(ts) <= 0):
        raise ValueError("row stamps must be strictly increasing")
    n, latency, clip, rule = len(ts), settings.latency_ns, settings.clip, settings.rule
    start = day_start_ns(prints.day)
    allowed = (ts >= start + clock_ns(settings.no_entry_before)) & (
        ts <= start + clock_ns(settings.no_entry_after)
    )
    flatten, day_end = start + clock_ns(settings.flatten_at), start + NS_PER_DAY
    sw = day_sweeps(prints, settings)
    out = {name: np.full((2, n), -1, dtype=np.int64) for name in
           ("entry_print_ns", "exit_decision_ns", "exit_print_ns", "free_ns")}  # fmt: skip
    for name in ("entry_price", "entry_beyond", "exit_price", "exit_beyond", "notional",
                 "gross", "fees", "net"):  # fmt: skip
        out[name] = np.full((2, n), np.nan)
    flags = {name: np.zeros((2, n), dtype=bool) for name in
             ("has_entry", "entry_late", "exit_late", "exit_no_print", "exit_either_side")}  # fmt: skip
    for index, s in SIDES:
        arrival = ts + latency
        entry = fills.fill(sw[s], arrival, until_ns=flatten, decision_ns=ts, rule=rule)
        k = np.flatnonzero(entry.found)
        e_ns, e_price = entry.print_ns[k], entry.price[k]
        due = e_ns + hold_ns
        h = np.searchsorted(ts, due, side="left")
        decided = np.where(h < n, ts[np.minimum(h, n - 1)], due)
        decided = np.minimum(decided, flatten)
        x_arrival = decided + latency
        exit_ = fills.fill(sw[-s], x_arrival, until_ns=day_end, decision_ns=decided, rule=rule)
        found = exit_.found
        x_ns, x_price = exit_.print_ns.copy(), exit_.price.copy()
        x_beyond = np.where(found, exit_.beyond, 0.0)
        either = np.zeros(len(k), dtype=bool)
        if (~found).any():
            _, stamp, price, side_none = no_print_exits(sw[-s], sw[s], x_arrival[~found])
            x_ns[~found], x_price[~found], either[~found] = stamp, price, side_none
        x_fill = np.where(found, x_ns, x_arrival)
        fee_e = settings.fee_bp(e_ns)
        fee_x = settings.fee_bp(x_fill)
        fees = fill_fee(e_price * clip, fee_e) + fill_fee(x_price * clip, fee_x)
        gross = s * (x_price - e_price) * clip
        values = {"entry_print_ns": e_ns, "exit_decision_ns": decided, "exit_print_ns": x_ns,
                  "free_ns": x_fill, "entry_price": e_price, "entry_beyond": entry.beyond[k],
                  "exit_price": x_price, "exit_beyond": x_beyond, "notional": e_price * clip,
                  "gross": gross, "fees": fees, "net": gross - fees}  # fmt: skip
        for name, value in values.items():
            out[name][index, k] = value
        flags["has_entry"][index, k] = True
        flags["entry_late"][index, k] = e_ns - arrival[k] > settings.late_print_ns
        flags["exit_late"][index, k] = x_ns - x_arrival > settings.late_print_ns
        flags["exit_no_print"][index, k] = ~found
        flags["exit_either_side"][index, k] = either
    pnl_bp = out["net"] / out["notional"] * 1e4
    return PrintHoldTable(prints.day, ts, int(hold_ns), settings, allowed, flags["has_entry"],
                          out["entry_print_ns"], out["entry_price"], out["entry_beyond"],
                          flags["entry_late"], out["exit_decision_ns"], out["exit_print_ns"],
                          out["exit_price"], out["exit_beyond"], flags["exit_late"],
                          flags["exit_no_print"], flags["exit_either_side"], out["free_ns"],
                          out["notional"], out["gross"], out["fees"], out["net"], pnl_bp)  # fmt: skip


@dataclass(frozen=True)
class PrintTableTrades:
    """The round trips taken from a print table, in time order: money on the trade's
    own side, and both sides' walked P&L at its decision row."""

    rows: np.ndarray
    side: np.ndarray
    net: np.ndarray
    gross: np.ndarray
    notional: np.ndarray
    turnover: np.ndarray
    pnl_long_bp: np.ndarray
    pnl_short_bp: np.ndarray
    counters: dict[str, int]

    @property
    def trades(self) -> int:
        return len(self.rows)

    @property
    def net_bp(self) -> np.ndarray:
        return np.where(self.side > 0, self.pnl_long_bp, self.pnl_short_bp)

    @property
    def g_bp(self) -> np.ndarray:
        """G per trade, ``side x (pnl_long_bp - pnl_short_bp) / 2``; NaN where the other
        side has no fill at the trade's row."""
        return self.side * (self.pnl_long_bp - self.pnl_short_bp) / 2.0

    def day_values(self) -> dict[str, float | int]:
        """The day's ledger values as in 2025: ``net_bp`` the day's net over its entry
        notional, ``gross_bp`` the same before fees, ``gross_bp_turnover`` the gross
        over both sides' notional; ``G_bp`` and the two walked P&Ls the means over the
        trades that have them (``g_trades`` of them); all 0 on a day without a trade."""
        skips = int(self.counters["entries_busy"])
        if self.trades == 0:
            zero = dict.fromkeys(("net_bp", "gross_bp", "gross_bp_turnover", "G_bp",
                                  "pnl_long_bp", "pnl_short_bp"), 0.0)  # fmt: skip
            return {**zero, "trades": 0, "g_trades": 0, "skips": skips}
        g = self.g_bp
        has = np.isfinite(g)
        both = has.any()
        return {
            "net_bp": float(self.net.sum() / self.notional.sum() * 1e4),
            "gross_bp": float(self.gross.sum() / self.notional.sum() * 1e4),
            "gross_bp_turnover": float(self.gross.sum() / self.turnover.sum() * 1e4),
            "G_bp": float(g[has].mean()) if both else 0.0,
            "pnl_long_bp": float(self.pnl_long_bp[has].mean()) if both else 0.0,
            "pnl_short_bp": float(self.pnl_short_bp[has].mean()) if both else 0.0,
            "trades": self.trades,
            "g_trades": int(has.sum()),
            "skips": skips,
        }


def execute_rows(table: PrintHoldTable, rows: np.ndarray, side: np.ndarray) -> PrintTableTrades:
    """The print executor's decision loop over a table: decisions at row positions in
    time order, sides +1 / -1, each run keeping its own busy periods."""
    rows = np.asarray(rows, dtype=np.int64)
    side = np.asarray(side, dtype=np.int64)
    if rows.shape != side.shape:
        raise ValueError("one side per decision")
    if len(rows) and (np.any(np.diff(rows) < 0) or rows[0] < 0 or rows[-1] >= table.rows):
        raise ValueError("decision rows must be row positions in time order")
    if not np.isin(side, (-1, 1)).all():
        raise ValueError("a side is +1 (buy) or -1 (sell)")
    ts, counters = table.row_ns, dict.fromkeys(PRINT_COUNTERS, 0)
    free_at = np.iinfo(np.int64).min
    taken: list[int] = []
    for k, (row, s) in enumerate(zip(rows.tolist(), side.tolist(), strict=True)):
        own = LONG if s > 0 else SHORT
        counters["decisions"] += 1
        if not table.allowed[row]:
            counters["entries_outside_hours"] += 1
        elif ts[row] < free_at:
            counters["entries_busy"] += 1
        elif not table.has_entry[own, row]:
            counters["entries_no_print"] += 1
        else:
            taken.append(k)
            free_at = int(table.free_ns[own, row])
    r, s_ = rows[taken], side[taken]
    own_ = side_index(s_)
    counters["entries"] = len(taken)
    limit = BEYOND_SHARE * table.settings.clip
    for name, values in (("exits_no_print", table.exit_no_print),
                         ("exits_no_print_either_side", table.exit_either_side),
                         ("entry_late_print", table.entry_late), ("exit_late_print", table.exit_late),
                         ("fill_beyond_sweep", table.entry_beyond > limit),
                         ("fill_beyond_sweep", table.exit_beyond > limit)):  # fmt: skip
        counters[name] += int(values[own_, r].sum())
    notional = table.notional[own_, r]
    exit_notional = table.exit_price[own_, r] * table.settings.clip
    return PrintTableTrades(r, s_, table.net[own_, r], table.gross[own_, r], notional,
                            notional + exit_notional, table.pnl_bp[LONG, r],
                            table.pnl_bp[SHORT, r], counters)  # fmt: skip


def print_entry_parity(
    prints: DayPrints, rows: np.ndarray, side: np.ndarray, table: PrintHoldTable
) -> Parity:
    """Execute the same decisions with :func:`.print_taker.execute_print_day` and with
    the table, and list every difference: which decisions became trades, their entry
    and exit prints, exit decisions, busy-until times, prices, net, notional and net bp
    (to the last bit), and every counter."""
    rows = np.asarray(rows, dtype=np.int64)
    side = np.asarray(side, dtype=np.int64)
    ours = execute_rows(table, rows, side)
    day = execute_print_day(prints, table.row_ns, table.row_ns[rows], side, table.hold_ns,
                            table.settings)  # fmt: skip
    theirs, out = day.trades, []
    if len(theirs) != ours.trades:
        out.append(f"trades: executor {len(theirs)}, table {ours.trades}")
    else:
        own, r = side_index(ours.side), ours.rows
        checks = {
            "decision_ns": table.row_ns[r],
            "side": ours.side,
            "entry_print_ns": table.entry_print_ns[own, r],
            "exit_decision_ns": table.exit_decision_ns[own, r],
            "exit_print_ns": table.exit_print_ns[own, r],
            "exit_fill_ns": table.free_ns[own, r],
            "entry_price": table.entry_price[own, r],
            "exit_price": table.exit_price[own, r],
            "net": ours.net,
            "notional": ours.notional,
            "net_bp": ours.net_bp,
        }
        for column, values in checks.items():
            if not np.array_equal(theirs[column].to_numpy(), values):
                out.append(f"{column} differs")
    for name in PRINT_COUNTERS:
        if int(day.counters[name]) != ours.counters[name]:
            out.append(f"{name}: executor {day.counters[name]:g}, table {ours.counters[name]}")
    return Parity(ours.trades, tuple(out))
