"""The spot archives: registered URLs, daily files only, the data root, HEAD and the head check."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from pathlib import Path

import pytest
import yaml

from trading_research.spot import archives as ar
from trading_research.spot import synthetic as syn

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "configs" / "spot_2026.yaml").read_text(encoding="utf-8"))
DAY = date(2026, 3, 3)


@pytest.fixture(autouse=True)
def _offline() -> Iterator[None]:
    with syn.no_network():
        yield


@pytest.mark.parametrize("name", sorted(ar.PLANES))
def test_urls_and_first_days_are_the_registered_ones(name: str) -> None:
    plane = ar.PLANES[name]
    registered = CONFIG["data"][name]  # the plane names are the configuration's keys
    assert plane.url_template == registered["url"]
    assert plane.first_day == registered["present"][0]
    instrument = CONFIG["instruments"][registered["instrument"]]["id"]
    assert plane.instrument == instrument
    assert plane.url(DAY) == registered["url"].format(symbol=plane.symbol, day=DAY.isoformat())


def test_binance_is_asked_for_daily_files_only() -> None:
    url = ar.BINANCE_PRINTS.url(DAY)
    assert "/daily/aggTrades/BTCFDUSD/" in url and "monthly" not in url
    assert url.endswith("BTCFDUSD-aggTrades-2026-03-03.zip")
    assert ar.checksum_url(ar.BINANCE_PRINTS, DAY) == f"{url}.CHECKSUM"
    with pytest.raises(ValueError, match="no published checksum"):
        ar.checksum_url(ar.BYBIT_BOOK, DAY)


def test_files_land_under_the_spot_root_keyed_by_venue_plane_and_symbol(tmp_path: Path) -> None:
    assert ar.day_path(ar.BYBIT_BOOK, DAY) == Path(
        "data/spot/bybit/book/BTCUSDT/2026-03-03.parquet"
    )
    assert ar.day_path(ar.BYBIT_PRINTS, DAY, tmp_path) == (
        tmp_path / "bybit" / "prints" / "BTCUSDT" / "2026-03-03.parquet"
    )
    assert ar.day_path(ar.BINANCE_PRINTS, DAY).parts[:4] == ("data", "spot", "binance", "prints")
    forbidden = {Path(f"data/{p}/BTCUSDT") for p in ("book", "trades", "bars", "funding")}
    for plane in ar.PLANES.values():
        assert not any(ar.day_path(plane, DAY).is_relative_to(f) for f in forbidden)
    assert {p.instrument for p in ar.PLANES.values()} == set(ar.INSTRUMENTS)
    assert {ar.BYBIT_BOOK.kind, ar.BYBIT_PRINTS.kind} == {ar.BOOK, ar.PRINTS}


def test_head_reads_headers_only_and_a_missing_day_is_a_status() -> None:
    market = syn.SpotMarket(frozenset({DAY}), step_ms=600_000)
    venue = syn.SpotVenue(market)
    found = ar.coverage(ar.BYBIT_BOOK, [DAY, date(2026, 3, 4)], opener=venue)
    assert found[DAY].status == 200
    assert found[DAY].size == len(market.book_archive(DAY))
    assert found[DAY].last_modified is not None
    assert found[date(2026, 3, 4)].status == 404
    assert {method for method, _ in venue.asked} == {"HEAD"}


def test_the_head_check_lists_the_registered_coverage() -> None:
    present = ar.listed_present(ROOT / ar.HEAD_CHECK)
    reader_test = set(CONFIG["data"]["reader_test_days"])
    for name in ar.PLANES:
        assert reader_test <= present[name]
        assert date(2026, 10, 21) in present[name] and date(2026, 10, 22) not in present[name]
    assert date(2025, 4, 28) not in present["bybit_book"]
    assert date(2025, 4, 29) in present["bybit_book"]
    assert date(2025, 1, 1) in present["binance_prints"]
    block_d = CONFIG["blocks"]["D"]
    days = (block_d["end"] - block_d["start"]).days + 1
    for name in ar.PLANES:
        assert sum(block_d["start"] <= d <= block_d["end"] for d in present[name]) == days
