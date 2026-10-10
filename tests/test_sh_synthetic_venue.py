"""The dry run's venue: the study's own fetch stage reads it, and nothing reaches the network."""

from __future__ import annotations

import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from trading_research.data import fetch_log as fl
from trading_research.data import fetch_stage as fs
from trading_research.data import study_access as sa
from trading_research.data.disk import StageBudget
from trading_research.pipeline.short_horizon import synthetic as syn

DAYS = (date(2024, 12, 30), date(2024, 12, 31))
GAP = date(2024, 12, 29)
SMALL = StageBudget(archives_in_flight_gb=0.001, books_gb=0.001, prints_gb=0.001)


def test_a_stage_fetches_the_synthetic_days_and_misses_the_rest(tmp_path: Path) -> None:
    market = syn.SyntheticMarket(frozenset(DAYS), step_ms=60_000)
    venue = syn.Venue(market)
    roots = fs.StudyRoots.under(tmp_path / "data")
    log = fl.FetchLog(tmp_path / "log.json")
    plan = fs.StagePlan(book=(GAP, *DAYS), prints=(GAP, *DAYS), bars=DAYS, funding=DAYS)
    with syn.serving(venue):
        fs.run_stage(
            "BTCUSDT", plan, roots, log, budget=SMALL, book_workers=1, print_workers=1,
            expected_present=market.present(),
        )  # fmt: skip
    status = {(r.plane, r.day): r.status for r in log.of()}
    assert status["book", GAP.isoformat()] == fl.MISSING_PLANE
    assert {status["book", d.isoformat()] for d in DAYS} == {fl.ON_DISK}
    assert {status["funding", d.isoformat()] for d in DAYS} == {fl.ON_DISK}
    assert not venue.unknown and log.complete()
    book = pd.read_parquet(roots.book / "BTCUSDT" / f"{DAYS[0]}.parquet")
    assert len(book) == 1440 and (book["ask_price_0"] > book["bid_price_0"]).all()
    stamps = pd.DatetimeIndex(book["timestamp"])
    assert stamps[0].hour == 0 and stamps[-1].hour == 23


def test_the_venue_answers_the_same_bytes_every_time() -> None:
    market = syn.SyntheticMarket(frozenset(DAYS), step_ms=60_000)
    assert market.book_archive(DAYS[0]) == market.book_archive(DAYS[0])
    assert market.prints_archive(DAYS[1]) == market.prints_archive(DAYS[1])


def test_an_unknown_request_is_a_404_and_is_kept() -> None:
    venue = syn.Venue(syn.SyntheticMarket(frozenset(DAYS)))
    with syn.serving(venue), pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen("https://example.org/anything")
    assert venue.unknown == ["https://example.org/anything"]


def test_reader_test_days_are_short_ob200_archives() -> None:
    day = sa.READER_TEST_DAYS[0]
    venue = syn.Venue(syn.SyntheticMarket(frozenset(), step_ms=60_000), frozenset({day}))
    from trading_research.data import bybit

    assert venue.body(bybit.archive_url("BTCUSDT", day)) is not None
    assert venue.body(bybit.archive_url("BTCUSDT", day, "ob500")) is None


def test_no_socket_connects_while_the_venue_serves() -> None:
    import socket

    venue = syn.Venue(syn.SyntheticMarket(frozenset(DAYS)))
    with syn.serving(venue), pytest.raises(OSError, match="no network"):
        socket.create_connection(("example.org", 443), timeout=1)


def _lines(market: syn.SyntheticMarket, day: date) -> list[dict[str, Any]]:
    import json

    return [json.loads(line) for line in market.book_lines(day)]


def test_trend_episodes_show_at_the_touch_and_move_the_price() -> None:
    day = DAYS[0]
    plain = syn.SyntheticMarket(frozenset(DAYS), step_ms=60_000)
    trending = syn.SyntheticMarket(frozenset(DAYS), step_ms=60_000, episodes_per_day=6)
    assert plain.path(day).signal is not None and not plain.path(day).signal.any()
    p = trending.path(day)
    assert p.signal is not None and set(np.unique(p.signal)) <= {-1, 0, 1} and p.signal.any()
    moves = np.diff(p.bid0)
    assert moves[p.signal[:-1] == 1].mean() > 1.0 > moves[p.signal[:-1] == 0].mean() > -1.0
    for i, message in enumerate(_lines(trending, day)[1:], start=1):
        shown = int(p.signal[i])
        if shown == 0:
            continue
        levels = message["data"]["b" if shown > 0 else "a"]
        touch = syn._price(int(p.bid0[i]) + (0 if shown > 0 else 1))
        assert any(price == touch and float(size) >= trending.signal_size for price, size in levels)


def test_a_burst_day_has_louder_hourly_returns_and_a_wider_spread(tmp_path: Path) -> None:
    from trading_research.data import bybit

    day = DAYS[1]
    plain = syn.SyntheticMarket(frozenset(DAYS), step_ms=60_000)
    bursting = syn.SyntheticMarket(
        frozenset(DAYS), step_ms=60_000, burst_days=frozenset({day}), burst_hours=(2, 8)
    )
    _, quiet = plain.hourly_closes()
    _, loud = bursting.hourly_closes()
    start = (day - syn.STREAM_START).days * 24
    ratio = np.diff(np.log(loud)) / np.diff(np.log(quiet))  # diff j is the return of hour j + 1
    assert np.allclose(ratio[start + 1 : start + 7], 10.0)
    assert np.allclose(ratio[start - 24 : start + 1], 1.0)
    spread = bursting.path(day).spread
    assert spread is not None and spread[0] == 1 and spread[-1] == bursting.burst_spread_ticks
    assert plain.path(day).spread is not None and (plain.path(day).spread == 1).all()
    archive = tmp_path / "book.zip"
    archive.write_bytes(bursting.book_archive(day))
    book = bybit.reconstruct(archive, symbol="BTCUSDT")
    gap = (book["ask_price_0"] - book["bid_price_0"]).round(1)
    hour = pd.DatetimeIndex(book["timestamp"]).hour
    assert (gap[hour < 2] == 0.1).all() and (gap[hour >= 2] == 0.8).all()
    inverted = syn.SyntheticMarket(frozenset(DAYS), step_ms=60_000, episodes_per_day=40,
                                   burst_days=frozenset({day}), burst_hours=(12, 18))  # fmt: skip
    p = inverted.path(day)
    assert p.signal is not None
    moves, shown = np.diff(p.bid0), p.signal[:-1]
    after = p.stamps_ms[:-1] >= syn._ms(day, 12)
    assert moves[~after & (shown == 1)].mean() > 1.0 > -1.0 > moves[after & (shown == 1)].mean()
    assert moves[~after & (shown == -1)].mean() < -1.0 < 1.0 < moves[after & (shown == -1)].mean()


def test_a_gap_day_skips_three_sequence_numbers(tmp_path: Path) -> None:
    from trading_research.data import bybit

    market = syn.SyntheticMarket(frozenset(DAYS), step_ms=60_000, gap_days=frozenset({DAYS[0]}))
    counted = {}
    for day in DAYS:
        archive = tmp_path / f"{day}.zip"
        archive.write_bytes(market.book_archive(day))
        counted[day] = bybit.reconstruct(archive, symbol="BTCUSDT").attrs["sequence_gaps"]
    assert counted == {DAYS[0]: 3, DAYS[1]: 0}
    sequence = [m["data"]["u"] for m in _lines(market, DAYS[0])]
    assert sequence[0] == 1 and sequence[-1] == len(sequence) + 3
