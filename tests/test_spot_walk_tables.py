"""Print walk tables equal the print executor, entry by entry, side-dependent busy periods included."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from test_spot_print_taker import hand_prints
from test_spot_prints_fixture import grid_of, make_prints
from trading_research.market_making.events import NS_PER_S
from trading_research.spot.costs import load_fee_table
from trading_research.spot.fills import BRACKETS, PrintTape
from trading_research.spot.print_taker import execute_print_day, print_settings
from trading_research.spot.walk_tables import (
    LONG,
    SHORT,
    PrintHoldTable,
    execute_rows,
    print_entry_parity,
    print_hold_table,
)

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "spot_2026.yaml"
TABLE = load_fee_table(CONFIG)
BINANCE = "binance_spot_BTCFDUSD"


def table_for(seed: int, hold_s: int, **kw: Any) -> tuple[PrintTape, PrintHoldTable]:
    prints = make_prints(seed)
    settings = print_settings(TABLE, BINANCE, **kw)
    return prints, print_hold_table(prints, grid_of(prints), hold_s * NS_PER_S, settings)


def decisions(table: PrintHoldTable, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(table.rows, size=min(count, table.rows), replace=False))
    return rows.astype(np.int64), rng.choice(np.array([-1, 1]), size=len(rows))


@pytest.mark.parametrize("hold_s", [20, 50, 120])
@pytest.mark.parametrize("seed", [0, 1])
def test_parity_with_the_print_executor(hold_s: int, seed: int) -> None:
    prints, table = table_for(seed, hold_s)
    rows, side = decisions(table, table.rows // 2, seed)
    parity = print_entry_parity(prints, rows, side, table)
    assert parity.ok, parity.mismatches
    assert parity.trades >= 10


@pytest.mark.parametrize("bracket", BRACKETS)
def test_parity_under_every_bracket_and_at_110_ms(bracket: str) -> None:
    for latency in (10_000_000, 110_000_000):
        prints, table = table_for(2, 50, bracket=bracket, latency_ns=latency)
        rows, side = decisions(table, table.rows // 2, 2)
        parity = print_entry_parity(prints, rows, side, table)
        assert parity.ok, (bracket, latency, parity.mismatches)


def test_placebos_keep_their_own_busy_periods() -> None:
    prints, table = table_for(3, 50)
    both = table.has_entry.all(axis=0)
    assert (table.free_ns[LONG, both] != table.free_ns[SHORT, both]).mean() > 0.9
    rows, side = decisions(table, table.rows // 2, 3)
    runs = {"true": side, "flipped": -side, "all_long": np.ones_like(side),
            "all_short": -np.ones_like(side)}  # fmt: skip
    busy = {}
    for name, sides in runs.items():
        parity = print_entry_parity(prints, rows, sides, table)
        assert parity.ok, (name, parity.mismatches)
        busy[name] = execute_rows(table, rows, sides).counters["entries_busy"]
    assert len(set(busy.values())) > 1  # the skips differ with the side


def test_a_handmade_side_dependent_run() -> None:
    prints = hand_prints([(43200.0 + k, -1, 60_000.0 - k, 0.02) for k in range(240)]
                         + [(43440.5, 1, 59_800.0, 0.02), (43441.0, -1, 59_799.0, 0.02),
                            (43500.0, 1, 59_801.0, 0.02), (43501.0, -1, 59_800.0, 0.02),
                            (43600.0, 1, 59_802.0, 0.02), (43700.0, -1, 59_803.0, 0.02)])  # fmt: skip
    row_ns = np.unique(prints.ts)
    table = print_hold_table(prints, row_ns, 50 * NS_PER_S, print_settings(TABLE, BINANCE))
    rows = np.searchsorted(row_ns, prints.ts[[1, 60, 241]])
    for side, trades, busy in (([-1, -1, 1], 2, 1), ([1, 1, -1], 1, 2)):
        parity = print_entry_parity(prints, rows, np.array(side), table)
        assert parity.ok and parity.trades == trades, parity.mismatches
        assert execute_rows(table, rows, np.array(side)).counters["entries_busy"] == busy


def test_every_row_is_the_executors_single_decision() -> None:
    prints, table = table_for(4, 20)
    takeable = np.flatnonzero(table.allowed & table.has_entry.all(axis=0))
    for row in np.random.default_rng(4).choice(takeable, 40, replace=False):
        for side, index in ((1, LONG), (-1, SHORT)):
            day = execute_print_day(prints, table.row_ns, table.row_ns[[row]], np.array([side]),
                                    table.hold_ns, table.settings)  # fmt: skip
            assert day.trades["net_bp"].iloc[0] == table.pnl_bp[index, row]
            assert day.trades["exit_fill_ns"].iloc[0] == table.free_ns[index, row]


def test_day_values_are_the_2025_ledgers() -> None:
    prints, table = table_for(5, 50)
    rows, side = decisions(table, table.rows // 2, 5)
    taken = execute_rows(table, rows, side)
    values = taken.day_values()
    day = execute_print_day(prints, table.row_ns, table.row_ns[rows], side, table.hold_ns,
                            table.settings)  # fmt: skip
    assert values["net_bp"] == pytest.approx(day.day_bp("net"), rel=1e-12)
    assert values["gross_bp"] == pytest.approx(day.day_bp("gross"), rel=1e-12)
    assert values["gross_bp_turnover"] == pytest.approx(day.trades["gross"].sum() / day.turnover
                                                        * 1e4, rel=1e-9)  # fmt: skip
    g = taken.side * (taken.pnl_long_bp - taken.pnl_short_bp) / 2
    has = np.isfinite(g)
    assert values["G_bp"] == pytest.approx(g[has].mean()) and values["g_trades"] == has.sum()
    net_bp = np.where(taken.side > 0, taken.pnl_long_bp, taken.pnl_short_bp)
    np.testing.assert_allclose((taken.net_bp - (taken.pnl_long_bp + taken.pnl_short_bp) / 2)[has],
                               g[has], rtol=1e-9, atol=1e-9)  # fmt: skip
    np.testing.assert_array_equal(taken.net_bp, net_bp)
    empty = execute_rows(table, rows[:0], side[:0]).day_values()
    assert empty["trades"] == 0 and empty["net_bp"] == 0.0
