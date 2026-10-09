"""The dry run's venue: the study's own fetch stage reads it, and nothing reaches the network."""

from __future__ import annotations

import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

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
