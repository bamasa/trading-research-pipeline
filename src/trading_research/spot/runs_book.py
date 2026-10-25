"""One day of a book row (S25-BB, S25-T) on Bybit's spot book, and S25-PB's book walk.

The true run is the 2025 event-time taker executor through this study's fee
table (:func:`trading_research.spot.costs.execute_book_day`). On a book a
taker's exit time does not depend on its side, so the placebos, K-drift's runs
and Null A's flips are looked up in the 2025 walk tables at the true run's entry
decision times, as in 2025 (an entry beyond the visible book on the placebo's
own side skipped and counted); the 110 ms re-run and the neighbourhood cells
run the executor's decision loop over walk tables (``execute_rows``). A day any
of these runs flags ``flatten_stale`` is stale: it is excluded on Bybit.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from trading_research.backtest.taker import TakerBook
from trading_research.market_making.accounting import IdentityError
from trading_research.search_engine.walk_tables import HoldTable, WalkTable, execute_rows
from trading_research.search_engine.walk_tables import side_index as book_side_index
from trading_research.spot import access as sa
from trading_research.spot import account as acc
from trading_research.spot.calibration import capacity_ratio, trade_fills
from trading_research.spot.costs import (
    FeeTable,
    FeeUnitsError,
    book_hold_table,
    book_settings,
    execute_book_day,
)
from trading_research.spot.fills import PrintTape
from trading_research.spot.plan import Row
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
from trading_research.spot.runs_prints import LATENCY_RERUN_NS, RANDOM_DRAWS, _money, inventory


def lookup(table: HoldTable, rows: np.ndarray, side: np.ndarray) -> tuple[float, int, bool]:
    """A placebo's day at the true run's entry rows on ``side``: its value, the entries
    it skips beyond the visible book on its own side, and whether a kept one is stale."""
    own = book_side_index(side)
    skipped = table.skip[own, rows] if len(rows) else np.zeros(0, dtype=bool)
    kept, own = rows[~skipped], own[~skipped]
    if not len(kept):
        return math.nan, int(skipped.sum()), False
    net, notional = table.net[own, kept], table.notional[own, kept]
    return (
        float(net.sum() / notional.sum() * 1e4),
        int(skipped.sum()),
        bool(table.stale[kept].any()),
    )


def shifted(
    table: HoldTable, start_ns: int, tod_ns: np.ndarray, side: np.ndarray
) -> tuple[float, bool]:
    """Another day's signal tape on this day's table, with its own one-position state."""
    rows = shift_rows(table.walk.ts, start_ns, tod_ns)
    inside = rows < table.rows
    found = execute_rows(table, rows[inside], np.asarray(side)[inside])
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
    try:
        true = execute_book_day(book, book.ts[rows], side, own.hold_ns, settings, table)
    except FeeUnitsError:
        return record | {"void": ["K-units"]}, None, pd.DataFrame()
    except IdentityError:
        return record | {"void": ["K-acct"]}, None, pd.DataFrame()
    trades = true.trades
    stale = [bool(true.excluded)]
    record.update(value=true.day_bp(), trades=len(trades), counters=dict(true.counters),
                  net_quote=float(true.net), turnover=float(true.turnover), **inventory(marks))  # fmt: skip
    record.update(_money_book(trades))
    try:
        found = acc.account_day(
            acc.book_fills(true, settings), true.net, *acc.first_last_print(marks)
        )
        record["void"] = list(found.void)
    except ValueError:
        record["void"] = ["K-cap"]
    walk = WalkTable.from_book(book, settings.clip)
    tables: dict[int, HoldTable] = {}

    def at(hold_ns: int) -> HoldTable:
        if hold_ns not in tables:
            tables[hold_ns] = book_hold_table(walk, hold_ns, settings, table)
        return tables[hold_ns]

    t_rows = np.searchsorted(book.ts, trades["decision_ns"].to_numpy(dtype=np.int64))
    t_side = trades["side"].to_numpy(dtype=np.int64)

    def placebo(sides: np.ndarray) -> tuple[float, int]:
        value, skipped, flagged = lookup(at(own.hold_ns), t_rows, sides)
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
    rerun = execute_rows(book_hold_table(walk, own.hold_ns, late, table), rows, side)
    record["latency"] = table_bp(rerun)
    stale.append(bool(rerun.stale))
    cells = []
    for cell in row.cells:
        if cell not in fits.rules or cell.fit not in probas:
            cells.append(math.nan)
            continue
        r, s = decisions(probas[cell.fit], fits.rules[cell])
        ran = execute_rows(at(cell.hold_ns), r, s)
        stale.append(bool(ran.stale))
        cells.append(table_bp(ran))
    record["neighbourhood"] = cells
    longs, sides = acc.long_only(book.ts[rows], side)
    long_run = execute_book_day(book, longs, sides, own.hold_ns, settings, table)
    record["long_only"] = long_run.day_bp()
    stale.append(bool(long_run.excluded))
    scored = np.isfinite(probas[own.fit]).all(axis=1)
    record["ic"] = rank_ic(signal(probas[own.fit])[scored], forward[own.hold_s][scored])
    record["capacity"] = capacity_ratio(*trade_fills(trades), prints) if len(trades) else 0.0
    record["tiers"] = tier_values(trades, table, sa.BYBIT, lambda tier: (
        table.day_fees(sa.BYBIT, day, tier).taker_bp,) * 2)  # fmt: skip
    record["signal"] = {"tod_ns": (book.ts[rows] - start).tolist(), "side": side.tolist()}
    record["stale"] = any(stale)
    return record, at(own.hold_ns), trades


def _money_book(trades: pd.DataFrame) -> dict[str, Any]:
    out = _money(trades.assign(entry_wait_ns=0, exit_wait_ns=0, hold_realised_ns=0))
    return {k: v for k, v in out.items() if k not in ("entry_wait_s", "exit_wait_s",
                                                       "hold_realised_s")}  # fmt: skip
