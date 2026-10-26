"""Void conditions on every run of a row's day, and K-units on every fill from a fresh
look-up of the fee table."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from test_spot_day_fixture import DAYS, TABLE, fetch_days, load, probas
from trading_research.market_making.accounting import IdentityError
from trading_research.spot import access as sa
from trading_research.spot import costs, runs_book, runs_prints
from trading_research.spot import days as dy
from trading_research.spot import synthetic as syn
from trading_research.spot.costs import FeeUnitsError, book_settings, execute_book_day, taker_book
from trading_research.spot.plan import OWN_CELL, ROWS
from trading_research.spot.print_taker import PrintSettings, execute_print_day, print_settings
from trading_research.spot.runs import RowModels, decisions
from trading_research.strategies.hold_first import HoldFirstRule

RULE = HoldFirstRule(0.5, 0.5)


@pytest.fixture(autouse=True)
def _offline():  # type: ignore[no-untyped-def]
    with syn.no_network():
        yield


@pytest.fixture(scope="module")
def day(tmp_path_factory: pytest.TempPathFactory) -> dy.DayInputs:
    base = tmp_path_factory.mktemp("spot_voids")
    return load(base, fetch_days(base), DAYS[0])


def models(row: str) -> RowModels:
    return RowModels(row, {}, dict.fromkeys(ROWS[row].cells, RULE), None)


def _print_row(day: dy.DayInputs) -> dict:
    tape, grid = dy.tape(day, sa.BINANCE), dy.grid_of(day.binance.frame)
    proba = probas(len(grid), seed=1)
    fits = {c.fit: proba for c in ROWS["S25-PF"].cells}
    record, _ = runs_prints.print_day(
        ROWS["S25-PF"], models("S25-PF"), tape, grid, fits,
        {OWN_CELL.hold_s: np.linspace(-1, 1, len(grid))}, day.binance.frame, TABLE, day=day.day,
    )  # fmt: skip
    return record


def test_an_identity_error_in_the_latency_rerun_voids_the_row(
    day: dy.DayInputs, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = runs_prints.execute_print_day

    def failing(tape, rows, decisions_, side, hold, settings):  # type: ignore[no-untyped-def]
        if settings.latency_ns == runs_prints.LATENCY_RERUN_NS:
            raise IdentityError("a fault in the 110 ms re-run only")
        return real(tape, rows, decisions_, side, hold, settings)

    monkeypatch.setattr(runs_prints, "execute_print_day", failing)
    record = _print_row(day)
    assert record["void"] == ["K-acct"] and np.isnan(record["latency"])
    assert record["trades"] > 0  # the true run is kept; the void ranks first


def test_an_account_fault_in_a_bracket_run_voids_the_row(
    day: dy.DayInputs, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trading_research.spot import run_checks

    real = run_checks.acc.account_day
    calls = {"n": 0}

    def account_day(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        found = real(*args, **kwargs)
        if calls["n"] == 3:  # a run after the true one
            return type(found)(found.clip, found.first_price, found.last_price, found.checks,
                               2 * found.clip, found.net)  # fmt: skip
        return found

    monkeypatch.setattr(run_checks.acc, "account_day", account_day)
    assert _print_row(day)["void"] == ["K-cap"]


def test_a_print_fill_whose_fee_differs_from_a_fresh_look_up_fires_k_units(
    day: dy.DayInputs,
) -> None:
    class Cheap(PrintSettings):
        def fee_bp(self, ns: np.ndarray) -> np.ndarray:  # the executor's own look-up, wrong
            return np.full(len(ns), 9.0)

    tape, grid = dy.tape(day, sa.BINANCE), dy.grid_of(day.binance.frame)
    rows, side = decisions(probas(len(grid), seed=1), RULE)
    good = print_settings(TABLE, sa.BINANCE)
    execute_print_day(tape, grid, grid[rows], side, OWN_CELL.hold_ns, good)
    cheap = Cheap(table=TABLE, instrument=sa.BINANCE, tier="regular")
    with pytest.raises(FeeUnitsError):
        execute_print_day(tape, grid, grid[rows], side, OWN_CELL.hold_ns, cheap)


def test_a_book_fill_whose_fee_differs_fires_k_units_per_fill(
    day: dy.DayInputs, monkeypatch: pytest.MonkeyPatch
) -> None:
    book = taker_book(day.book, depth=10)
    rows, side = decisions(probas(len(book.ts), seed=2, share=0.02), RULE)
    settings = book_settings(TABLE, sa.BYBIT, day.day)
    execute_book_day(book, book.ts[rows], side, OWN_CELL.hold_ns, settings, TABLE)
    real = costs.execute_day

    def tampered(*args, **kwargs):  # type: ignore[no-untyped-def]
        found = real(*args, **kwargs)
        found.trades.loc[found.trades.index[0], "fees"] *= 1.0 + 1e-9
        return found

    monkeypatch.setattr(costs, "execute_day", tampered)
    with pytest.raises(FeeUnitsError):
        execute_book_day(book, book.ts[rows], side, OWN_CELL.hold_ns, settings, TABLE)


def test_a_book_rows_walk_table_runs_join_its_voids(
    day: dy.DayInputs, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise FeeUnitsError("a table fill charged another fee")

    monkeypatch.setattr(runs_book, "book_table_fills", broken)
    book = taker_book(day.book, depth=10)
    proba = probas(len(book.ts), seed=2, share=0.02)
    fits = {c.fit: proba for c in ROWS["S25-BB"].cells}
    record, _, trades = runs_book.book_day(
        ROWS["S25-BB"], models("S25-BB"), book, fits, {OWN_CELL.hold_s: np.zeros(len(book.ts))},
        day.bybit.frame, dy.tape(day, sa.BYBIT), TABLE, day=day.day,
    )  # fmt: skip
    assert len(trades) > 0 and record["void"] == ["K-units"]


def test_the_day_fee_of_a_book_fill_is_the_days(day: dy.DayInputs) -> None:
    start = dy.midnight_ns(date(2026, 3, 3))
    found = costs.book_fill_fee_bp(TABLE, sa.BYBIT, "non_vip", np.array([start, start + 1]))
    assert found.tolist() == [10.0, 10.0]
