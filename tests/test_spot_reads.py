"""The reads through the ledger: refused out of order, the fetch records, P's publication check."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from trading_research.data import fetch_log as fl
from trading_research.market_making.prereg import HeldOutLocked
from trading_research.spot import access as sa
from trading_research.spot import reads
from trading_research.spot import synthetic as syn
from trading_research.spot.context import SpotStudy
from trading_research.spot.state import StageOrderError


def test_a_block_is_refused_before_the_amendment(tmp_path: Path) -> None:
    study = SpotStudy(tmp_path)
    with pytest.raises(StageOrderError, match="needs stage 0"):
        reads.open_block(study, "H")
    with pytest.raises(StageOrderError, match="needs stage 0"):
        reads.read_block(study, "H")
    with pytest.raises(HeldOutLocked, match="not written"):
        reads.h_results(study)


def test_the_fetch_records_hold_each_files_sha256_and_time(tmp_path: Path) -> None:
    log = fl.FetchLog(tmp_path / "log.json")
    day = date(2026, 8, 8)
    record = log.get("bybit_book", day)
    record.add(fl.OK, now=datetime(2026, 10, 26, tzinfo=UTC), sha256="ab" * 32, size=3)
    log.get("binance_prints", day).add(fl.MISSING, now=record.attempts[0].when)
    found = reads.fetch_records(log, [day])
    assert list(found) == ["bybit_book/2026-08-08"] and found["bybit_book/2026-08-08"]["sha256"]
    assert reads.fetch_records(log, [date(2026, 8, 9)]) == {}


def test_p_waits_for_its_last_day_on_every_plane(tmp_path: Path) -> None:
    published = date(2026, 12, 31)
    venue = syn.SpotVenue(syn.SpotMarket(frozenset({published})))
    study = SpotStudy(tmp_path, venue=venue)
    check = reads.published(study)
    with syn.serving(venue):
        assert check(sa.BYBIT, "book", published) and check(sa.BINANCE, "prints", published)
        assert not check(sa.BYBIT, "prints", date(2026, 12, 30))
