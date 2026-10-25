"""One day of a print row and of a book row on a synthetic reader-test day."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from test_spot_day_fixture import DAYS, TABLE, fetch_days, load, probas
from trading_research.features.timed import stamps_ns
from trading_research.spot import access as sa
from trading_research.spot import days as dy
from trading_research.spot import synthetic as syn
from trading_research.spot.costs import book_settings, execute_book_day, taker_book
from trading_research.spot.plan import OWN_CELL, ROWS
from trading_research.spot.print_taker import execute_print_day, print_settings
from trading_research.spot.runs import RowModels, decisions
from trading_research.spot.runs_book import book_day
from trading_research.spot.runs_prints import print_day, shifted
from trading_research.strategies.hold_first import HoldFirstRule

RULE = HoldFirstRule(0.5, 0.5)


@pytest.fixture(autouse=True)
def _offline():  # type: ignore[no-untyped-def]
    with syn.no_network():
        yield


@pytest.fixture(scope="module")
def day(tmp_path_factory: pytest.TempPathFactory) -> dy.DayInputs:
    base = tmp_path_factory.mktemp("spot_day")
    return load(base, fetch_days(base), DAYS[0])


def models(row: str) -> RowModels:
    cells = ROWS[row].cells
    return RowModels(row, {}, dict.fromkeys(cells, RULE), None)


def test_a_print_row_runs_the_executor_and_its_placebos(day: dy.DayInputs) -> None:
    tape = dy.tape(day, sa.BINANCE)
    grid = dy.grid_of(day.binance.frame)
    proba = probas(len(grid), seed=1)
    fits = {c.fit: proba for c in ROWS["S25-PF"].cells}
    forward = {OWN_CELL.hold_s: np.linspace(-1, 1, len(grid))}
    record, table = print_day(ROWS["S25-PF"], models("S25-PF"), tape, grid, fits, forward,
                              day.binance.frame, TABLE, day=day.day, null_a=3)  # fmt: skip
    rows, side = decisions(proba, RULE)
    direct = execute_print_day(tape, grid, grid[rows], side, OWN_CELL.hold_ns,
                               print_settings(TABLE, sa.BINANCE))  # fmt: skip
    assert record["trades"] == len(direct.trades) > 0
    assert record["value"] == pytest.approx(direct.day_bp())
    assert len(record["random"]) == 200 and len(record["null_a"]) == 3
    assert len(record["neighbourhood"]) == 12 and record["void"] == []
    assert record["fills"] == 2 * record["trades"] and record["capacity"] > 0
    assert set(record["brackets"]) == {"optimistic", "pessimistic", "stale", "worse_of_two"}
    assert record["worse"]["entry_notional"] > 0 and "regular" in record["tiers"]
    start = dy.midnight_ns(day.day)
    same = shifted(
        table, start, np.asarray(record["signal"]["tod_ns"]), np.asarray(record["signal"]["side"])
    )
    assert math.isfinite(same)


def test_a_day_without_a_decision_has_no_value(day: dy.DayInputs) -> None:
    tape = dy.tape(day, sa.BYBIT)
    grid = dy.grid_of(day.bybit.frame)
    quiet = {OWN_CELL.fit: np.full((len(grid), 3), np.nan)}
    record, _ = print_day(ROWS["S25-PB"], models("S25-PB"), tape, grid, quiet,
                          {OWN_CELL.hold_s: np.zeros(len(grid))}, day.bybit.frame, TABLE, day=day.day)  # fmt: skip
    assert record["trades"] == 0 and math.isnan(record["value"]) and math.isnan(record["flipped"])
    assert record["capacity"] == 0.0 and all(math.isnan(v) for v in record["random"])


def test_a_book_row_runs_the_book_executor_and_the_walk_tables(day: dy.DayInputs) -> None:
    book = taker_book(day.book, depth=10)
    proba = probas(len(book.ts), seed=2, share=0.02)
    fits = {c.fit: proba for c in ROWS["S25-BB"].cells}
    marks = dy.tape(day, sa.BYBIT)
    record, table, trades = book_day(ROWS["S25-BB"], models("S25-BB"), book, fits,
                                     {OWN_CELL.hold_s: np.zeros(len(book.ts))}, day.bybit.frame,
                                     marks, TABLE, day=day.day, null_a=2)  # fmt: skip
    rows, side = decisions(proba, RULE)
    direct = execute_book_day(book, book.ts[rows], side, OWN_CELL.hold_ns,
                              book_settings(TABLE, sa.BYBIT, day.day), TABLE)  # fmt: skip
    assert record["trades"] == len(trades) == len(direct.trades) > 0
    assert record["value"] == pytest.approx(direct.day_bp())
    assert len(record["neighbourhood"]) == 12 and len(record["null_a"]) == 2
    assert record["stale"] is False and table is not None
    assert record["inventory_pnl_bp"] == pytest.approx(
        (marks.price[-1] / marks.price[0] - 1) * 1e4)  # fmt: skip


def test_the_cold_day_is_not_scored_for_61_minutes(tmp_path: Path, day: dy.DayInputs) -> None:
    assert not day.warm.get(sa.BINANCE)
    assert day.scored_from(sa.BINANCE) == dy.midnight_ns(day.day) + 61 * 60 * 1_000_000_000
    frame = dy.store_frame("binance", day)
    assert np.array_equal(stamps_ns(frame), dy.grid_of(day.binance.frame))
