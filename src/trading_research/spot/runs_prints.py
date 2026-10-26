"""One day of a print row (S25-PF, S25-PB, S25-PFT): every run the statuses and the page read.

The true run is the print executor (:func:`trading_research.spot.print_taker.execute_print_day`)
on the own cell's decisions at the verdict tier. Its placebos, K-drift's two
runs and Null A's flips take the true run's entry decision times and re-apply
the one-position rule to their own fills through the print walk table of the
hold (:func:`trading_research.spot.walk_tables.execute_rows`); the neighbourhood
cells (S25-PF only), the 110 ms re-run, the four brackets and the long-only
variant are full runs of the executor. Every run's void conditions, its own
(K-units on every fill, K-acct) and its fills' on the spot account (K-acct,
K-cap, K-inv), join the day's record (:mod:`.run_checks`); G and the two walked
P&Ls at the true trades are recorded as in 2025. See :mod:`.runs` for the record.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from datetime import date
from functools import partial
from typing import Any

import numpy as np
import pandas as pd

from trading_research.spot import account as acc
from trading_research.spot.calibration import capacity_ratio, trade_fills
from trading_research.spot.costs import FeeTable
from trading_research.spot.fills import PrintTape
from trading_research.spot.plan import Row
from trading_research.spot.print_taker import PrintDay, execute_print_day, print_settings
from trading_research.spot.run_checks import Voids, print_table_fills
from trading_research.spot.runs import (
    RowModels,
    decisions,
    hour_flips,
    quantiles,
    random_sides,
    rank_ic,
    shift_rows,
    signal,
    table_bp,
    tier_values,
)
from trading_research.spot.walk_tables import (
    PrintHoldTable,
    PrintTableTrades,
    execute_rows,
    print_hold_table,
)

BRACKET_RUNS: tuple[str, ...] = ("optimistic", "pessimistic", "stale", "worse_of_two")
LATENCY_RERUN_NS = 110_000_000
RANDOM_DRAWS = 200


def side_notional(trades: pd.DataFrame) -> tuple[float, float]:
    """The notional of the day's buys and of its sells, entries and exits alike."""
    if not len(trades):
        return 0.0, 0.0
    side = trades["side"].to_numpy(dtype=np.int64)
    size = trades["size"].to_numpy(dtype=np.float64)
    entry = trades["entry_price"].to_numpy(dtype=np.float64) * size
    exit_ = trades["exit_price"].to_numpy(dtype=np.float64) * size
    buys = float(entry[side > 0].sum() + exit_[side < 0].sum())
    sells = float(entry[side < 0].sum() + exit_[side > 0].sum())
    return buys, sells


def _run(tape: PrintTape, grid: np.ndarray, rows: np.ndarray, side: np.ndarray, hold_ns: int,
         settings: Any) -> PrintDay:  # fmt: skip
    return execute_print_day(tape, grid, grid[rows], side, hold_ns, settings)


def _full(voids: Voids, call: Callable[[], PrintDay]) -> PrintDay | None:
    """A full executor run with its void conditions: the run's own (K-units, K-acct)
    and its fills' on the spot account (K-acct, K-cap, K-inv)."""
    day = voids.run(call)
    if day is not None:
        voids.account(acc.print_fills(day), day.net)
    return day


def _bp(day: PrintDay | None) -> float:
    return math.nan if day is None else day.day_bp()


def _table_run(
    voids: Voids, table: PrintHoldTable, rows: np.ndarray, side: np.ndarray, fees: FeeTable
) -> PrintTableTrades:
    """A walk-table run with its void conditions: each fill's fee and the account."""
    found = execute_rows(table, rows, side)
    replayed = voids.run(lambda: print_table_fills(table, found, fees))
    if replayed is not None:
        voids.account(*replayed)
    return found


def shifted(
    table: PrintHoldTable, start_ns: int, tod_ns: np.ndarray, side: np.ndarray,
    voids: Voids | None = None, fees: FeeTable | None = None,
) -> float:  # fmt: skip
    """Another day's signal tape on this day's table, with its own one-position state."""
    rows = shift_rows(table.row_ns, start_ns, tod_ns)
    inside = rows < table.rows
    rows, side = rows[inside], np.asarray(side)[inside]
    if voids is None or fees is None:
        return table_bp(execute_rows(table, rows, side))
    return table_bp(_table_run(voids, table, rows, side, fees))


def print_day(
    row: Row,
    fits: RowModels,
    tape: PrintTape,
    grid: np.ndarray,
    probas: Mapping[tuple[float, float], np.ndarray],
    forward: Mapping[float, np.ndarray],
    prints: pd.DataFrame,
    table: FeeTable,
    *,
    day: date,
    null_a: int = 0,
) -> tuple[dict[str, Any], PrintHoldTable | None]:
    """The day's record, and the own cell's walk table (for the one-day shift). The
    record's ``void`` holds every void condition any run of the day fired."""
    own = row.own
    rows, side = decisions(probas[own.fit], fits.rules[own])
    settings = print_settings(table, row.instrument)
    record: dict[str, Any] = {"day": day.isoformat(), "row": row.name, "void": [],
                              "decisions": len(rows), "grid_rows": len(grid)}  # fmt: skip
    voids = Voids(tape)
    true = _full(voids, lambda: _run(tape, grid, rows, side, own.hold_ns, settings))
    if true is None:
        return record | {"void": voids.sorted()}, None
    trades = true.trades
    record.update(value=true.day_bp(), trades=len(trades), counters=dict(true.counters),
                  net_quote=float(true.net), turnover=float(true.turnover))  # fmt: skip
    record.update(_money(trades))
    table50 = print_hold_table(tape, grid, own.hold_ns, settings)
    t_rows = np.searchsorted(grid, trades["decision_ns"].to_numpy(dtype=np.int64))
    t_side = trades["side"].to_numpy(dtype=np.int64)

    def placebo(sides: np.ndarray) -> tuple[float, int]:
        found = _table_run(voids, table50, t_rows, sides, table)
        return table_bp(found), int(found.counters["entries_busy"])

    record["flipped"], skipped_flip = placebo(-t_side)
    record["all_long"], skipped_long = placebo(np.ones_like(t_side))
    record["all_short"], skipped_short = placebo(-np.ones_like(t_side))
    draws = [placebo(random_sides(len(t_rows), k, day)) for k in range(RANDOM_DRAWS)]
    record["random"] = [v for v, _ in draws]
    record["skips"] = {"flipped": skipped_flip, "all_long": skipped_long,
                       "all_short": skipped_short, "random": sum(s for _, s in draws)}  # fmt: skip
    start = tape.start_ns
    record["null_a"] = [
        placebo(t_side * hour_flips(grid[t_rows], start, k, day))[0] for k in range(null_a)
    ]
    late = print_settings(table, row.instrument, latency_ns=LATENCY_RERUN_NS)
    record["latency"] = _bp(_full(voids, lambda: _run(tape, grid, rows, side, own.hold_ns, late)))
    record.update(_variants(row, fits, tape, grid, probas, rows, side, table, voids))
    record.update(g_values(table50, t_rows, t_side))
    scored = np.isfinite(probas[own.fit]).all(axis=1)
    record["ic"] = rank_ic(signal(probas[own.fit])[scored], forward[own.hold_s][scored])
    record["capacity"] = capacity_ratio(*trade_fills(trades), prints) if len(trades) else 0.0
    record["fills"] = 2 * len(trades)
    record["beyond_sweep"] = int(true.counters["fill_beyond_sweep"])
    record["tiers"] = tier_values(trades, table, row.instrument, lambda tier: (
        table.taker_bp(row.instrument, tier, trades["entry_print_ns"].to_numpy(np.int64)),
        table.taker_bp(row.instrument, tier, trades["exit_fill_ns"].to_numpy(np.int64))))  # fmt: skip
    record["signal"] = {"tod_ns": (grid[rows] - start).tolist(), "side": side.tolist()}
    record["void"] = voids.sorted()
    return record, table50


def g_values(table: Any, rows: np.ndarray, side: np.ndarray) -> dict[str, Any]:
    """G and the two walked P&Ls at the true trades' decision rows, from the own
    cell's table, as the 2025 ledger records them (0 on a day without a trade)."""
    from trading_research.search_engine.walk_tables import TABLE_COUNTERS, take

    if isinstance(table, PrintHoldTable):
        values = execute_rows(table, rows, side).day_values()
    else:
        values = take(table, rows, side, dict.fromkeys(TABLE_COUNTERS, 0)).day_values()
    return {
        k: values[k] for k in ("G_bp", "pnl_long_bp", "pnl_short_bp", "g_trades") if k in values
    }


def _money(trades: pd.DataFrame) -> dict[str, Any]:
    """The true run's money beside the verdict, and its waits and realised holds."""
    s = acc.day_summary(trades)
    out: dict[str, Any] = {"gross_bp": s.gross_bp, "gross_bp_turnover": s.gross_bp_turnover,
                           "net_bp_long": s.net_bp_long, "net_bp_short": s.net_bp_short,
                           "trades_long": s.trades_long, "trades_short": s.trades_short,
                           "gross_quote": s.gross, "fees_quote": s.fees,
                           "entry_notional": s.entry_notional,
                           "net_per_100_clip": s.net_per_100_clip}  # fmt: skip
    for name, column in (("entry_wait_s", "entry_wait_ns"), ("exit_wait_s", "exit_wait_ns"),
                         ("hold_realised_s", "hold_realised_ns")):  # fmt: skip
        values = trades[column].to_numpy(dtype=np.float64) / 1e9 if len(trades) else np.zeros(0)
        if name == "exit_wait_s" and "exit_no_print" in trades:  # a fallback exit waits not
            values = values[~trades["exit_no_print"].to_numpy(dtype=bool)]
        out[name] = quantiles(values)
    if "exit_fallback_age_ns" in trades:  # the fallback prints' ages, reported apart
        ages = trades["exit_fallback_age_ns"].to_numpy(dtype=np.float64)
        out["exit_fallback_age_s"] = quantiles(ages[ages >= 0] / 1e9)
    return out


def inventory(tape: PrintTape) -> dict[str, float]:
    """The held clip marked from the day's first print to its last."""
    first, last = acc.first_last_print(tape)
    return {"inventory_pnl": 0.010 * (last - first), "inventory_pnl_bp": (last / first - 1) * 1e4}


def _variants(
    row: Row,
    fits: RowModels,
    tape: PrintTape,
    grid: np.ndarray,
    probas: Mapping[tuple[float, float], np.ndarray],
    rows: np.ndarray,
    side: np.ndarray,
    table: FeeTable,
    voids: Voids,
) -> dict[str, Any]:
    """The brackets, K-fill's worse-of-two sums, the long-only variant, the cells,
    each a full run whose void conditions join the day's."""
    own = row.own
    out: dict[str, Any] = {"brackets": {}, **inventory(tape)}
    for bracket in BRACKET_RUNS:
        settings = print_settings(table, row.instrument, bracket=bracket)
        day = _full(voids, partial(_run, tape, grid, rows, side, own.hold_ns, settings))
        out["brackets"][bracket] = _bp(day)
        if bracket == "worse_of_two" and day is not None:
            buys, sells = side_notional(day.trades)
            out["worse"] = {"net_quote": float(day.trades["net"].sum()),
                            "entry_notional": float(day.trades["notional"].sum()),
                            "buy_notional": buys, "sell_notional": sells}  # fmt: skip
    verdict = print_settings(table, row.instrument)
    longs, sides = acc.long_only(grid[rows], side)
    out["long_only"] = _bp(_full(voids, lambda: execute_print_day(
        tape, grid, longs, sides, own.hold_ns, verdict)))  # fmt: skip
    if len(row.cells) > 1:
        cells = []
        for cell in row.cells:
            if cell not in fits.rules or cell.fit not in probas:
                cells.append(math.nan)
                continue
            r, s = decisions(probas[cell.fit], fits.rules[cell])
            cells.append(_bp(_full(voids, partial(_run, tape, grid, r, s, cell.hold_ns, verdict))))
        out["neighbourhood"] = cells
    return out
