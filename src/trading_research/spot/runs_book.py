"""One day of a book row (S25-BB, S25-T) on Bybit's spot book, and S25-PB's book walk.

The true run is the 2025 event-time taker executor through this study's fee
table (:func:`trading_research.spot.costs.execute_book_day`). On a book a
taker's exit time does not depend on its side, so the placebos, K-drift's runs
and Null A's flips are looked up in the 2025 walk tables at the true run's entry
decision times, as in 2025 (an entry beyond the visible book on the placebo's
own side skipped and counted); the 110 ms re-run and the neighbourhood cells
run the executor's decision loop over walk tables (``execute_rows``). A day any
of these runs flags ``flatten_stale`` is stale: it is excluded on Bybit. Every
run's void conditions (K-units on every fill from a fresh look-up of the fee
table, K-acct, K-cap and K-inv on its fills) join the day's record.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from trading_research.backtest.taker import TakerBook
from trading_research.search_engine.walk_tables import (
    HoldTable,
    TableTrades,
    WalkTable,
    execute_rows,
    take,
)
from trading_research.search_engine.walk_tables import side_index as book_side_index
from trading_research.spot import access as sa
from trading_research.spot import account as acc
from trading_research.spot.calibration import capacity_ratio, trade_fills
from trading_research.spot.costs import (
    FeeTable,
    book_hold_table,
    book_settings,
    execute_book_day,
)
from trading_research.spot.fills import PrintTape
from trading_research.spot.plan import Row
from trading_research.spot.run_checks import Voids, book_table_fills
from trading_research.spot.runs import (
    RowModels,
    decisions,
    hour_flips,
    random_sides,
    rank_ic,
    shift_rows,
    signal,
    table_bp,
    tier_values,
)
from trading_research.spot.runs_prints import (
    LATENCY_RERUN_NS,
    RANDOM_DRAWS,
    _money,
    g_values,
    inventory,
)


def lookup(
    table: HoldTable, rows: np.ndarray, side: np.ndarray, voids: Voids | None = None,
    fees: FeeTable | None = None,
) -> tuple[float, int, bool]:  # fmt: skip
    """A placebo's day at the true run's entry rows on ``side``: its value, the entries
    it skips beyond the visible book on its own side, and whether a kept one is stale.
    With ``voids``, the kept trades' fills are checked (K-units) and replayed on the
    spot account."""
    own = book_side_index(side)
    skipped = table.skip[own, rows] if len(rows) else np.zeros(0, dtype=bool)
    kept, own, kept_side = rows[~skipped], own[~skipped], np.asarray(side)[~skipped]
    if not len(kept):
        return math.nan, int(skipped.sum()), False
    if voids is not None and fees is not None:
        _account_table(voids, table, take(table, kept, kept_side, {}), fees)
    net, notional = table.net[own, kept], table.notional[own, kept]
    return (
        float(net.sum() / notional.sum() * 1e4),
        int(skipped.sum()),
        bool(table.stale[kept].any()),
    )


def _account_table(voids: Voids, table: HoldTable, found: TableTrades, fees: FeeTable) -> None:
    settings = table.settings
    instrument, tier = getattr(settings, "instrument", sa.BYBIT), getattr(settings, "tier", None)
    tier = tier or fees.verdict_tier(instrument)
    replayed = voids.run(lambda: book_table_fills(table, found, fees, instrument, tier))
    if replayed is not None:
        voids.account(*replayed)


def _table_run(
    voids: Voids, table: HoldTable, rows: np.ndarray, side: np.ndarray, fees: FeeTable
) -> TableTrades:
    found = execute_rows(table, rows, side)
    _account_table(voids, table, found, fees)
    return found


def shifted(
    table: HoldTable, start_ns: int, tod_ns: np.ndarray, side: np.ndarray,
    voids: Voids | None = None, fees: FeeTable | None = None,
) -> tuple[float, bool]:  # fmt: skip
    """Another day's signal tape on this day's table, with its own one-position state."""
    rows = shift_rows(table.walk.ts, start_ns, tod_ns)
    inside = rows < table.rows
    rows, side = rows[inside], np.asarray(side)[inside]
    found = execute_rows(table, rows, side)
    if voids is not None and fees is not None:
        _account_table(voids, table, found, fees)
    return table_bp(found), bool(found.stale)


def book_day(
    row: Row,
    fits: RowModels,
    book: TakerBook,
    probas: Mapping[tuple[float, float], np.ndarray],
    forward: Mapping[float, np.ndarray],
    prints: pd.DataFrame,
    marks: PrintTape,
    table: FeeTable,
    *,
    day: date,
    null_a: int = 0,
) -> tuple[dict[str, Any], HoldTable | None, pd.DataFrame]:
    """The day's record, the own cell's hold table (for the one-day shift) and the
    true run's trades (for FILL-S)."""
    own = row.own
    rows, side = decisions(probas[own.fit], fits.rules[own])
    settings = book_settings(table, sa.BYBIT, day)
    record: dict[str, Any] = {"day": day.isoformat(), "row": row.name, "void": [],
                              "decisions": len(rows), "grid_rows": len(book.ts)}  # fmt: skip
    voids = Voids(marks)
    true = voids.run(lambda: execute_book_day(book, book.ts[rows], side, own.hold_ns, settings,
                                              table))  # fmt: skip
    if true is None:
        return record | {"void": voids.sorted()}, None, pd.DataFrame()
    trades = true.trades
    stale = [bool(true.excluded)]
    record.update(value=true.day_bp(), trades=len(trades), counters=dict(true.counters),
                  net_quote=float(true.net), turnover=float(true.turnover), **inventory(marks))  # fmt: skip
    record.update(_money_book(trades))
    voids.account(acc.book_fills(true, settings), true.net)
    walk = WalkTable.from_book(book, settings.clip)
    tables: dict[int, HoldTable] = {}

    def at(hold_ns: int) -> HoldTable:
        if hold_ns not in tables:
            tables[hold_ns] = book_hold_table(walk, hold_ns, settings, table)
        return tables[hold_ns]

    t_rows = np.searchsorted(book.ts, trades["decision_ns"].to_numpy(dtype=np.int64))
    t_side = trades["side"].to_numpy(dtype=np.int64)

    def placebo(sides: np.ndarray) -> tuple[float, int]:
        value, skipped, flagged = lookup(at(own.hold_ns), t_rows, sides, voids, table)
        stale.append(flagged)
        return value, skipped

    record["flipped"], s1 = placebo(-t_side)
    record["all_long"], s2 = placebo(np.ones_like(t_side))
    record["all_short"], s3 = placebo(-np.ones_like(t_side))
    draws = [placebo(random_sides(len(t_rows), k, day)) for k in range(RANDOM_DRAWS)]
    record["random"] = [v for v, _ in draws]
    record["skips"] = {"flipped": s1, "all_long": s2, "all_short": s3,
                       "random": sum(s for _, s in draws)}  # fmt: skip
    start = int(walk.start_ns)
    record["null_a"] = [
        placebo(t_side * hour_flips(book.ts[t_rows], start, k, day))[0] for k in range(null_a)
    ]
    late = book_settings(table, sa.BYBIT, day, latency_ns=LATENCY_RERUN_NS)
    rerun = _table_run(voids, book_hold_table(walk, own.hold_ns, late, table), rows, side, table)
    record["latency"] = table_bp(rerun)
    stale.append(bool(rerun.stale))
    cells = []
    for cell in row.cells:
        if cell not in fits.rules or cell.fit not in probas:
            cells.append(math.nan)
            continue
        r, s = decisions(probas[cell.fit], fits.rules[cell])
        ran = _table_run(voids, at(cell.hold_ns), r, s, table)
        stale.append(bool(ran.stale))
        cells.append(table_bp(ran))
    record["neighbourhood"] = cells
    longs, sides = acc.long_only(book.ts[rows], side)
    long_run = voids.run(lambda: execute_book_day(book, longs, sides, own.hold_ns, settings, table))
    record["long_only"] = math.nan if long_run is None else long_run.day_bp()
    if long_run is not None:
        voids.account(acc.book_fills(long_run, settings), long_run.net)
        stale.append(bool(long_run.excluded))
    record.update(g_values(at(own.hold_ns), t_rows, t_side))
    scored = np.isfinite(probas[own.fit]).all(axis=1)
    record["ic"] = rank_ic(signal(probas[own.fit])[scored], forward[own.hold_s][scored])
    record["capacity"] = capacity_ratio(*trade_fills(trades), prints) if len(trades) else 0.0
    record["tiers"] = tier_values(trades, table, sa.BYBIT, lambda tier: (
        table.day_fees(sa.BYBIT, day, tier).taker_bp,) * 2)  # fmt: skip
    record["signal"] = {"tod_ns": (book.ts[rows] - start).tolist(), "side": side.tolist()}
    record["stale"] = any(stale)
    record["void"] = voids.sorted()
    return record, at(own.hold_ns), trades


def _money_book(trades: pd.DataFrame) -> dict[str, Any]:
    out = _money(trades.assign(entry_wait_ns=0, exit_wait_ns=0, hold_realised_ns=0))
    return {k: v for k, v in out.items() if k not in ("entry_wait_s", "exit_wait_s",
                                                       "hold_realised_s")}  # fmt: skip
