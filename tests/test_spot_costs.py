"""The spot fee table: as registered, by instant and by day, and the book executor's settings."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from _engine_books import decisions, make_book
from trading_research.backtest.costs import BYBIT_BASE
from trading_research.backtest.taker import TakerBook, TakerSettings
from trading_research.market_making.events import NS_PER_S
from trading_research.search_engine.walk_tables import WalkTable, hold_table
from trading_research.spot.costs import (
    FEE_NOT_SOURCED,
    BookSettings,
    FeeNotSourced,
    FeeTable,
    FeeUnitsError,
    SettingsRefused,
    book_hold_table,
    book_settings,
    check_fill_fees,
    day_bounds_ns,
    execute_book_day,
    fill_fee,
    instant_ns,
    instrument_key,
    load_fee_table,
)

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "spot_2026.yaml"
BINANCE, BYBIT = "binance_spot_BTCFDUSD", "bybit_spot_BTCUSDT"
#: The registration's table, transcribed by hand from its section "Fees".
REGISTERED = {
    BINANCE: [
        ("2024-04-25T06:00:00Z", "2026-01-29T00:00:00Z", ("B3", "B4", "B6", "B8"),
         {"regular": (0.0, 10.0), "vip1": (0.0, 10.0), "vip2_to_vip9": (0.0, 0.0),
          "spot_liquidity_provider": (0.0, 0.0)}),
        ("2026-01-29T00:00:00Z", "2026-10-22T00:00:00Z", ("B4", "B5", "B6", "B7", "B8"),
         {"regular": (0.0, 10.0), "vip3": (0.0, 6.0), "vip9": (0.0, 2.3)}),
    ],
    BYBIT: [
        ("2025-01-31T00:00:00Z", "2026-10-22T00:00:00Z", ("Y1", "Y2", "Y6"),
         {"non_vip": (10.0, 10.0), "vip1": (6.75, 8.0), "vip2": (6.5, 7.75),
          "vip3": (6.25, 7.5), "vip4": (5.0, 6.0), "vip5": (4.0, 5.0)}),
    ],
}  # fmt: skip
#: The last instant the registered intervals cover; later intervals are the fee amendment's.
REGISTERED_END = "2026-10-22T00:00:00Z"


def config() -> dict[str, Any]:
    with CONFIG.open(encoding="utf-8") as handle:
        loaded: dict[str, Any] = yaml.safe_load(handle)
    return loaded


def registered_only() -> FeeTable:
    """The table from the registered intervals alone, whatever the fee amendment adds."""
    fees = config()["fees"]
    end = instant_ns(REGISTERED_END)
    intervals = {k: [r for r in rows if instant_ns(r["end"]) <= end]
                 for k, rows in fees["intervals"].items()}  # fmt: skip
    return FeeTable.from_config({**fees, "intervals": intervals})


def days(start: date, end: date) -> list[date]:
    return [start + timedelta(days=k) for k in range((end - start).days + 1)]


def test_the_table_is_the_registered_one() -> None:
    table = load_fee_table(CONFIG)
    assert table.verdict == {BINANCE: "regular", BYBIT: "non_vip"}
    for instrument, rows in REGISTERED.items():
        got = [iv for iv in table.intervals if iv.instrument == instrument]
        assert len(got) >= len(rows)
        for iv, (start, end, sources, tiers) in zip(got, rows, strict=False):
            assert (iv.start_ns, iv.end_ns) == (instant_ns(start), instant_ns(end))
            assert iv.sources == sources
            assert {t: (m, k) for t, m, k in iv.tiers} == tiers
        for later in got[len(rows) :]:  # only the fee amendment's, for P's days
            assert later.start_ns >= instant_ns(REGISTERED_END)
    assert table.undated == {BYBIT: {"supreme_vip": (3.0, 4.5), "pro6_institutional": (0.5, 1.5)}}


def test_every_registered_day_is_covered_at_the_verdict_tier() -> None:
    blocks = config()["blocks"]
    wanted = list(blocks["reader_test"]["days"])
    for name in ("W", "D", "H"):
        wanted += days(blocks[name]["start"], blocks[name]["end"])
    assert len(wanted) == 3 + 30 + 92 + 75
    table = load_fee_table(CONFIG)
    for instrument in (BINANCE, BYBIT):
        for day in wanted:
            assert table.covered(instrument, day), (instrument, day)
            assert table.exclusion(instrument, day) is None
            assert table.day_fees(instrument, day).taker_bp == 10.0
    assert table.day_fees(BINANCE, date(2026, 8, 8)).maker_bp == 0.0
    assert table.day_fees(BYBIT, date(2026, 8, 8)).maker_bp == 10.0


def test_a_day_after_the_registered_intervals_is_excluded() -> None:
    table = registered_only()
    for instrument in (BINANCE, BYBIT):
        assert table.covered(instrument, date(2026, 10, 21))
        assert table.exclusion(instrument, date(2026, 10, 22)) == FEE_NOT_SOURCED
        assert table.exclusion(instrument, date(2026, 11, 1)) == FEE_NOT_SOURCED
        with pytest.raises(FeeNotSourced):
            table.day_fees(instrument, date(2026, 11, 1))
        with pytest.raises(FeeNotSourced):
            table.taker_bp(instrument, table.verdict_tier(instrument), [instant_ns(REGISTERED_END)])
    assert not table.covered(BYBIT, date(2025, 1, 30))  # before the first capture's next day


def test_fees_by_instant_and_the_boundary_of_2026_01_29() -> None:
    table = load_fee_table(CONFIG)
    boundary = instant_ns("2026-01-29T00:00:00Z")
    before, at = np.array([boundary - 1]), np.array([boundary])
    assert table.fees_bp(BINANCE, "vip2_to_vip9", before) == (0.0, 0.0)
    with pytest.raises(FeeNotSourced):
        table.fees_bp(BINANCE, "vip2_to_vip9", at)
    assert table.taker_bp(BINANCE, "vip9", at)[0] == 2.3
    assert table.taker_bp(BINANCE, "regular", np.r_[before, at]).tolist() == [10.0, 10.0]
    assert not table.covered(BINANCE, date(2024, 4, 25))  # the terms start at 06:00
    assert table.covered(BINANCE, date(2024, 4, 26))
    assert not table.covered(BINANCE, date(2026, 1, 28), "vip3")
    assert table.covered(BINANCE, date(2026, 1, 29), "vip3")


def synthetic(rows: list[tuple[str, str, dict[str, list[float]]]]) -> FeeTable:
    """A one-venue table on Bybit's key, from ``(start, end, tiers)`` rows."""
    intervals = [{"start": a, "end": b, "sources": ["Y1"], "tiers": t} for a, b, t in rows]
    return FeeTable.from_config(
        {"verdict_tier": {BYBIT: "non_vip"}, "intervals": {BYBIT: intervals}}
    )


def test_a_split_day_charges_by_instant_and_the_book_the_higher_fee() -> None:
    table = synthetic([
        ("2026-11-01T00:00:00Z", "2026-11-05T12:00:00Z", {"non_vip": [8.0, 10.0]}),
        ("2026-11-05T12:00:00Z", "2026-11-10T00:00:00Z", {"non_vip": [6.0, 7.5]}),
    ])  # fmt: skip
    day = date(2026, 11, 5)
    start, _ = day_bounds_ns(day)
    noon = start + 12 * 3600 * NS_PER_S
    assert table.taker_bp(BYBIT, "non_vip", np.array([noon - 1, noon])).tolist() == [10.0, 7.5]
    fees = table.day_fees(BYBIT, day)
    assert (fees.maker_bp, fees.taker_bp) == (8.0, 10.0)
    assert table.day_fees(BYBIT, date(2026, 11, 6)).taker_bp == 7.5
    assert table.exclusion(BYBIT, date(2026, 11, 10)) == FEE_NOT_SOURCED


def test_a_gap_inside_a_day_excludes_it_and_an_overlap_takes_the_higher_fee() -> None:
    table = synthetic([
        ("2026-11-01T00:00:00Z", "2026-11-02T06:00:00Z", {"non_vip": [10.0, 10.0]}),
        ("2026-11-02T06:00:01Z", "2026-11-04T00:00:00Z", {"non_vip": [10.0, 10.0]}),
        ("2026-11-03T00:00:00Z", "2026-11-04T00:00:00Z", {"non_vip": [6.0, 12.0]}),
    ])  # fmt: skip
    assert table.exclusion(BYBIT, date(2026, 11, 2)) == FEE_NOT_SOURCED
    assert table.covered(BYBIT, date(2026, 11, 3))
    maker, taker = table.fees_bp(BYBIT, "non_vip", np.array([day_bounds_ns(date(2026, 11, 3))[0]]))
    assert (maker[0], taker[0]) == (10.0, 12.0)


def test_bounds_and_keys_are_refused_unless_unambiguous() -> None:
    assert instrument_key("bybit-spot:BTCUSDT") == BYBIT == instrument_key(BYBIT)
    with pytest.raises(KeyError):
        instrument_key("BTCUSDT")
    with pytest.raises(ValueError, match="UTC"):
        instant_ns("2026-11-01T00:00:00")
    with pytest.raises(ValueError, match="UTC"):
        instant_ns("2026-11-01T00:00:00+03:00")
    assert instant_ns(datetime(1970, 1, 1, 0, 0, 1, tzinfo=UTC)) == NS_PER_S
    with pytest.raises(ValueError, match="end after"):
        synthetic([("2026-11-02T00:00:00Z", "2026-11-01T00:00:00Z", {"non_vip": [1.0, 1.0]})])
    with pytest.raises(ValueError, match=r"fees\.sources"):
        FeeTable.from_config({"verdict_tier": {}, "sources": {"Y1": {}}, "intervals": {BYBIT: [
            {"start": "2026-11-01T00:00:00Z", "end": "2026-11-02T00:00:00Z",
             "sources": ["Z9"], "tiers": {"non_vip": [1.0, 1.0]}}]}})  # fmt: skip


SPOT_DAY = date(2026, 5, 4)


def spot_book(seed: int = 0) -> TakerBook:
    """A synthetic book on a covered day, without funding (spot has none)."""
    b = make_book(seed, day=SPOT_DAY)
    return TakerBook(b.day, b.ts, b.bid_px, b.bid_sz, b.ask_px, b.ask_sz)


def test_book_settings_come_from_the_table_and_a_default_is_refused() -> None:
    table = load_fee_table(CONFIG)
    settings = book_settings(table, "bybit-spot:BTCUSDT", SPOT_DAY)
    assert isinstance(settings, BookSettings)
    assert (settings.fees.maker_bp, settings.fees.taker_bp, settings.tier) == (
        10.0,
        10.0,
        "non_vip",
    )
    assert (settings.clip, settings.latency_ns) == (0.010, 10_000_000)
    assert book_settings(table, BYBIT, SPOT_DAY, latency_ns=110_000_000).latency_ns == 110_000_000
    assert book_settings(table, BYBIT, SPOT_DAY, "vip5").fees.taker_bp == 5.0
    book = spot_book()
    rows, side = decisions(book, 400)
    hold = 20 * NS_PER_S
    refused: list[object] = [
        None,
        TakerSettings(),
        BookSettings(instrument=BYBIT, tier="non_vip", day=SPOT_DAY),  # the default 5.5 bp
        book_settings(table, BYBIT, date(2026, 5, 5)),
    ]
    for wrong in refused:
        with pytest.raises(SettingsRefused):
            execute_book_day(book, book.ts[rows], side, hold, wrong, table)
        with pytest.raises(SettingsRefused):
            book_hold_table(WalkTable.from_book(book, 0.010), hold, wrong, table)
    assert TakerSettings().fees == BYBIT_BASE


def test_the_book_executor_charges_the_tables_fee_on_every_fill() -> None:
    table = load_fee_table(CONFIG)
    book = spot_book(3)
    rows, side = decisions(book, 600, seed=3)
    settings = book_settings(table, BYBIT, SPOT_DAY)
    day = execute_book_day(book, book.ts[rows], side, 20 * NS_PER_S, settings, table)
    trades = day.trades
    assert len(trades) > 20 and day.counters["funding_settlements_with_position"] == 0
    entry = trades["entry_price"] * trades["size"]
    exit_ = trades["exit_price"] * trades["size"]
    np.testing.assert_allclose(trades["fees"], (entry + exit_) * 10.0 * 1e-4, rtol=1e-12)
    walked = book_hold_table(WalkTable.from_book(book, 0.010), 20 * NS_PER_S, settings, table)
    plain = hold_table(WalkTable.from_book(book, 0.010), 20 * NS_PER_S, settings)
    np.testing.assert_array_equal(walked.net, plain.net)
    gross = np.asarray(walked.gross)
    fees = gross - walked.net
    known = np.isfinite(fees[0])
    long_fees = (walked.walk.buy_vwap[walked.entry_row[known]] + walked.walk.sell_vwap[
        walked.exit_row[known]]) * 0.010 * 10.0 * 1e-4  # fmt: skip
    np.testing.assert_allclose(fees[0][known], long_fees, rtol=1e-9)


def test_a_fee_off_the_table_is_a_units_error() -> None:
    notional = np.array([650.0, 651.0])
    check_fill_fees(fill_fee(notional, 10.0), notional * 10.0 * 1e-4)
    with pytest.raises(FeeUnitsError, match="fill 1"):
        check_fill_fees(fill_fee(notional, 10.0), fill_fee(notional, [10.0, 5.5]))
    with pytest.raises(FeeUnitsError):
        check_fill_fees(fill_fee(notional, 10.0), fill_fee(notional[:1], 10.0))
