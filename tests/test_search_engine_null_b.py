"""Null B's walk-table route, and its parity with the executor at s = 0."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from _engine_books import CLIP, decisions, make_book
from trading_research.market_making.events import NS_PER_S
from trading_research.search_engine.null_b import (
    Entries,
    NullBRoute,
    entries_of,
    replay,
    stage0_parity,
)
from trading_research.search_engine.walk_tables import (
    HoldTable,
    WalkTable,
    execute_rows,
    hold_table,
)
from trading_research.validation.choice import ChoiceError, neighbourhood_index
from trading_research.validation.placebo_search import null_b, null_b_pairs, null_b_shifts

DAYS = [date(2025, 2, 3) + timedelta(days=i) for i in range(10)]
HOLDS = (5 * NS_PER_S, 20 * NS_PER_S)
CELLS = ("a", "b", "c")
CELL_HOLDS = (HOLDS[0], HOLDS[1], HOLDS[0])
SHIFTS = null_b_shifts(days=10, shift_range=(2, 7))


@pytest.mark.parametrize("hold_s", [5, 20, 50, 120])
def test_stage0_parity_holds_on_a_day(hold_s: int) -> None:
    book = make_book(seed=10 + hold_s, thin_share=0.05)
    rows, side = decisions(book, len(book.ts) // 3, seed=hold_s)
    parity = stage0_parity(book, rows, side, hold_s * NS_PER_S)
    assert parity.ok, parity.mismatches
    assert parity.trades > 0


def test_an_entry_is_decided_at_the_first_row_at_or_after_its_time_of_day() -> None:
    source, target = make_book(seed=1, day=DAYS[0]), make_book(seed=2, day=DAYS[1])
    s_table = hold_table(WalkTable.from_book(source, CLIP), HOLDS[1])
    t_table = hold_table(WalkTable.from_book(target, CLIP), HOLDS[1])
    rows, side = decisions(source, 400, seed=3)
    entries = entries_of(s_table, execute_rows(s_table, rows, side))
    routed = replay(t_table, entries)
    mapped = t_table.walk.start_ns + entries.tod_ns
    at = np.searchsorted(target.ts, mapped)
    assert routed.trades > 5
    for row in routed.rows:
        wanted = mapped[at == row]
        assert (target.ts[row] >= wanted).all() and (
            row == 0 or (target.ts[row - 1] < wanted).all()
        )


def test_an_entry_inside_the_mapped_previous_position_is_skipped_and_counted() -> None:
    table = hold_table(WalkTable.from_book(make_book(seed=4, thin_share=0.0), CLIP), HOLDS[1])
    first = int(np.flatnonzero(table.takeable(np.ones(table.rows)))[0])
    tod = table.walk.ts[first] - table.walk.start_ns
    entries = Entries(np.array([tod, tod + NS_PER_S, tod + 30 * NS_PER_S]), np.array([1, -1, 1]))
    routed = replay(table, entries)
    assert routed.trades == 2 and routed.counters["entries_busy"] == 1
    late = Entries(np.array([25 * 3600 * NS_PER_S]), np.array([1]))
    assert replay(table, late).counters["entries_no_snapshot"] == 1


def _world() -> tuple[np.ndarray, dict[int, dict[int, HoldTable]], dict[int, dict[int, Entries]]]:
    usable = np.ones(len(DAYS), dtype=bool)
    usable[6] = False
    tables, entries = {}, {}
    for i, day in enumerate(DAYS):
        walk = WalkTable.from_book(make_book(seed=20 + i, day=day, thin_share=0.05), CLIP)
        tables[i] = {hold: hold_table(walk, hold) for hold in HOLDS}
        entries[i] = {}
        for c, hold in enumerate(CELL_HOLDS):
            if (i + c) % 4 == 0:
                continue  # a cell without a trade that day
            rows, side = decisions(make_book(seed=20 + i, day=day), 300, seed=100 * i + c)
            entries[i][c] = entries_of(tables[i][hold], execute_rows(tables[i][hold], rows, side))
    return usable, tables, entries


def _route(
    usable: np.ndarray, entries: dict[int, dict[int, Entries]], directory: Path | None = None
) -> NullBRoute:
    route = NullBRoute(CELLS, CELL_HOLDS, usable, shifts=SHIFTS, directory=directory)
    for day, by_cell in entries.items():
        route.add_source_day(day, by_cell)
    return route


def test_shifted_outcomes_are_the_pairs_replayed() -> None:
    usable, tables, entries = _world()
    route = _route(usable, entries)
    with pytest.raises(ChoiceError, match="not filled"):
        route.shifted(SHIFTS[0])
    for target in range(len(DAYS)):
        route.add_target_day(target, tables[target])
    for shift in SHIFTS:
        cells = route.shifted(shift)
        pairs = null_b_pairs(usable, shift)
        assert cells.days == len(pairs.target) and route.excluded(shift) == pairs.excluded == 2
        for col, (source, target) in enumerate(zip(pairs.source, pairs.target, strict=True)):
            for c, hold in enumerate(CELL_HOLDS):
                if c not in entries[source]:
                    assert cells.trades[c, col] == 0 and np.isnan(cells.net_bp[c, col])
                    continue
                values = replay(tables[target][hold], entries[source][c]).day_values()
                assert cells.trades[c, col] == values["trades"]
                if values["trades"]:
                    assert cells.net_bp[c, col] == values["net_bp"]
                    assert cells.g_bp[c, col] == values["G_bp"]
    assert route.busy.sum() > 0
    table = pd.DataFrame({"x": [0.0, 1.0, 2.0], "group": [0, 0, 0]})
    draws = null_b(
        route.shifted,
        {"S": np.arange(3)},
        neighbourhood_index(table, ["x"], by=["group"]),
        shifts=SHIFTS,
        min_trades_per_day=0,
    )
    assert draws.replications == len(SHIFTS)


def test_a_route_on_disk_resumes_and_refilling_is_idempotent(tmp_path: Path) -> None:
    usable, tables, entries = _world()
    memory = _route(usable, entries)
    first = _route(usable, entries, tmp_path)
    for target in range(len(DAYS)):
        memory.add_target_day(target, tables[target])
        if target < 5:
            first.add_target_day(target, tables[target])
    second = _route(usable, entries, tmp_path)
    assert second.done[:5].all() and not second.done[5:].any()
    for target in range(4, len(DAYS)):
        second.add_target_day(target, tables[target])
    for shift in SHIFTS:
        ours, theirs = second.shifted(shift), memory.shifted(shift)
        np.testing.assert_array_equal(ours.trades, theirs.trades)
        np.testing.assert_array_equal(ours.net_bp, theirs.net_bp)
    np.testing.assert_array_equal(second.busy, memory.busy)


def test_missing_source_entries_are_refused() -> None:
    usable, tables, _ = _world()
    route = NullBRoute(CELLS, CELL_HOLDS, usable, shifts=SHIFTS)
    with pytest.raises(ValueError, match="entries were not added"):
        route.add_target_day(5, tables[5])
    with pytest.raises(ValueError, match="one hold per cell"):
        NullBRoute(CELLS, CELL_HOLDS[:2], usable)
