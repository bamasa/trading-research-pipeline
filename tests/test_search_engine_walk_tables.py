"""Walk tables equal the event-time taker executor, decision by decision."""

from __future__ import annotations

import numpy as np
import pytest

from _engine_books import CLIP, decisions, make_book
from trading_research.backtest.taker import TakerSettings, execute_day
from trading_research.backtest.walk import walk_book
from trading_research.market_making.events import NS_PER_S
from trading_research.search_engine.walk_tables import (
    LONG,
    SHORT,
    WalkTable,
    entry_parity,
    execute_rows,
    hold_table,
)
from trading_research.strategies.ev_gate import thin_entries

HOLDS_S = (5, 20, 50, 120)


def test_the_table_walks_every_row_as_the_executor_walks_one() -> None:
    book = make_book()
    walk = WalkTable.from_book(book, CLIP)
    for row in np.random.default_rng(0).choice(len(book.ts), 50, replace=False):
        buy = walk_book(book.ask_px[row], book.ask_sz[row], CLIP)
        sell = walk_book(book.bid_px[row], book.bid_sz[row], CLIP)
        assert (walk.buy_vwap[row], walk.buy_beyond[row]) == (buy.vwap[0], buy.beyond[0])
        assert (walk.sell_vwap[row], walk.sell_beyond[row]) == (sell.vwap[0], sell.beyond[0])
    assert walk.buy_exceeds.any() and walk.sell_exceeds.any()


@pytest.mark.parametrize("hold_s", HOLDS_S)
@pytest.mark.parametrize("late_stretch", [True, False])
def test_parity_with_the_executor(hold_s: int, late_stretch: bool) -> None:
    book = make_book(seed=hold_s, late_stretch=late_stretch)
    table = hold_table(WalkTable.from_book(book, CLIP), hold_s * NS_PER_S)
    rows, side = decisions(book, len(book.ts) // 3, seed=hold_s)
    parity = entry_parity(book, rows, side, table)
    assert parity.ok, parity.mismatches
    assert parity.trades >= 3


def test_the_parity_covers_every_counter() -> None:
    book = make_book(seed=8, thin_share=0.2)
    table = hold_table(WalkTable.from_book(book, CLIP), 5 * NS_PER_S)
    rows, side = decisions(book, len(book.ts) // 2, seed=8)
    assert entry_parity(book, rows, side, table).ok
    counters = execute_rows(table, rows, side).counters
    assert counters["entries_busy"] > 0 and counters["entries_outside_hours"] > 0
    assert counters["entries_skipped"] > 0 and counters["entries"] > 50


def test_an_exit_after_the_book_ends_flags_the_day_in_both() -> None:
    book = make_book(seed=7, late_stretch=False)
    table = hold_table(WalkTable.from_book(book, CLIP), 120 * NS_PER_S)
    last = np.flatnonzero(table.takeable(np.ones(len(book.ts))))[-1]
    parity = entry_parity(book, np.array([last]), np.array([1]), table)
    assert parity.ok and parity.trades == 1
    assert table.stale[last] and table.free_row[last] == len(book.ts)


def test_every_row_is_the_executors_single_decision() -> None:
    book = make_book(seed=3)
    table = hold_table(WalkTable.from_book(book, CLIP), 50 * NS_PER_S)
    takeable = np.flatnonzero(table.allowed & table.has_entry & ~table.skip.any(axis=0))
    for row in np.random.default_rng(1).choice(takeable, 40, replace=False):
        for side, index in ((1, LONG), (-1, SHORT)):
            day = execute_day(book, book.ts[[row]], np.array([side]), 50 * NS_PER_S)
            assert day.trades["net_bp"].iloc[0] == table.pnl_bp[index, row]
            assert day.trades["net"].iloc[0] == table.net[index, row]


def test_funding_reaches_a_position_held_over_a_settlement() -> None:
    book = make_book(seed=4, funding_rate=5e-3)
    table = hold_table(WalkTable.from_book(book, CLIP), 120 * NS_PER_S)
    eight = book.funding_ts[1]
    rows = np.flatnonzero((book.ts > eight - 60 * NS_PER_S) & (book.ts < eight - 5 * NS_PER_S))
    rows = rows[table.has_entry[rows]]
    quiet = hold_table(
        WalkTable.from_book(make_book(seed=4, funding_rate=0.0), CLIP), 120 * NS_PER_S
    )
    paid = table.net[:, rows] - quiet.net[:, rows]
    assert (paid[LONG] < 0).all() and (paid[SHORT] > 0).all()  # longs pay a positive rate


def test_thin_entries_over_busy_until_takes_what_the_executor_takes() -> None:
    book = make_book(seed=5)
    table = hold_table(WalkTable.from_book(book, CLIP), 20 * NS_PER_S)
    rows, side = decisions(book, len(book.ts) // 2, seed=5)
    sides = np.zeros(len(book.ts), dtype=np.int64)
    sides[rows] = side
    sides[~table.takeable(sides)] = 0
    accepted = thin_entries(sides, table.busy_until)
    ours = execute_rows(table, rows, side)
    np.testing.assert_array_equal(accepted, ours.rows)


def test_day_values_follow_the_executor_day() -> None:
    book = make_book(seed=6)
    table = hold_table(WalkTable.from_book(book, CLIP), 5 * NS_PER_S)
    rows, side = decisions(book, 600, seed=6)
    ours = execute_rows(table, rows, side)
    day = execute_day(book, book.ts[rows], side, 5 * NS_PER_S)
    values = ours.day_values()
    assert values["net_bp"] == pytest.approx(day.day_bp("net"), rel=1e-12)
    assert values["trades"] == len(day.trades)
    assert values["skips"] == day.counters["entries_skipped"]
    g = side_g = ours.side * (ours.pnl_long_bp - ours.pnl_short_bp) / 2
    assert values["G_bp"] == pytest.approx(g.mean()) and side_g.shape == (ours.trades,)
    nothing = execute_rows(table, rows[:0], side[:0]).day_values()
    assert nothing == {"net_bp": 0.0, "G_bp": 0.0, "pnl_long_bp": 0.0, "pnl_short_bp": 0.0,
                       "trades": 0, "skips": 0}  # fmt: skip


def test_bad_input_is_refused() -> None:
    book = make_book()
    walk = WalkTable.from_book(book, CLIP)
    with pytest.raises(ValueError, match="hold"):
        hold_table(walk, 0)
    with pytest.raises(ValueError, match="walked"):
        hold_table(walk, NS_PER_S, TakerSettings(clip=0.02))
    table = hold_table(walk, NS_PER_S)
    with pytest.raises(ValueError, match="time order"):
        execute_rows(table, np.array([5, 3]), np.array([1, 1]))
    with pytest.raises(ValueError, match="side"):
        execute_rows(table, np.array([3, 5]), np.array([1, 0]))
