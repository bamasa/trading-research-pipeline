"""The per-day spot fetchers: only a 404 is missing, bad archives are classified, archives deleted."""

from __future__ import annotations

import gzip
import io
import urllib.error
import zipfile
from collections.abc import Iterable, Iterator
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from trading_research.data import fetch_log as fl
from trading_research.market_making.prereg import HeldOutLocked
from trading_research.spot import archives as ar
from trading_research.spot import binance, bybit_book, bybit_prints
from trading_research.spot import synthetic as syn

DAY = date(2026, 3, 3)
MISSING = date(2026, 3, 4)
MARKET = syn.SpotMarket(frozenset({DAY}), step_ms=600_000, rpi_every=5)
FETCHERS = {"bybit_book": bybit_book.fetch_day, "bybit_prints": bybit_prints.fetch_day,
            "binance_prints": binance.fetch_day}  # fmt: skip


@pytest.fixture(autouse=True)
def _offline() -> Iterator[None]:
    with syn.no_network():
        yield


class Guard:
    """Permits the days it is given, in every plane."""

    def __init__(self, days: Iterable[date]) -> None:
        self.days, self.asked = frozenset(days), []

    def require(
        self, instrument: str, days: Iterable[date], plane: str, *, what: str = "read"
    ) -> None:
        for day in days:
            self.asked.append((instrument, day, plane, what))
            if day not in self.days:
                raise HeldOutLocked(f"cannot {what} {instrument} {plane} on {day}")


class _Body(io.BytesIO):
    def __init__(self, body: bytes) -> None:
        super().__init__(body)
        self.headers = {"Content-Length": str(len(body))}


class Faulty:
    """The venue, with some URLs answered by other bytes or by an error."""

    def __init__(self, replace: dict[str, bytes | Exception]) -> None:
        self.venue, self.replace = syn.SpotVenue(MARKET), replace

    def __call__(self, url: str, timeout: float | None = None) -> Any:
        if url in self.replace:
            value = self.replace[url]
            if isinstance(value, Exception):
                raise value
            return _Body(value)
        return self.venue(url, timeout)


def _run(name: str, tmp_path: Path, opener: Any, day: date = DAY) -> Any:
    archives = tmp_path / "in_flight"
    out = FETCHERS[name](
        day, tmp_path / "spot", archives, access=Guard({DAY, MISSING}), opener=opener
    )
    assert not archives.exists() or not any(archives.iterdir())  # deleted whatever happened
    return out


@pytest.mark.parametrize("name", sorted(FETCHERS))
def test_a_day_is_fetched_converted_and_its_archive_deleted(name: str, tmp_path: Path) -> None:
    outcome = _run(name, tmp_path, syn.SpotVenue(MARKET))
    assert (outcome.plane, outcome.outcome) == (name, fl.OK)
    assert outcome.sha256 and outcome.size
    frame = pd.read_parquet(ar.day_path(ar.PLANES[name], DAY, tmp_path / "spot"))
    assert len(frame) == outcome.stats["rows"] > 0


def test_the_stats_carry_what_stage_0_records(tmp_path: Path) -> None:
    book = _run("bybit_book", tmp_path, syn.SpotVenue(MARKET)).stats
    assert {"messages", "median_interval_ms", "sequence_gaps", "clip_beyond_levels_share",
            "peak_rss_bytes"} <= set(book)  # fmt: skip
    prints = _run("bybit_prints", tmp_path, syn.SpotVenue(MARKET)).stats
    assert prints["rpi"] == prints["rows"] // 5 and prints["rpi_share"] > 0
    assert _run("binance_prints", tmp_path, syn.SpotVenue(MARKET)).stats["unit"] == "us"
    written = pd.read_parquet(ar.day_path(ar.BYBIT_BOOK, DAY, tmp_path / "spot"))
    assert written.attrs["sequence_gaps"] == 0


@pytest.mark.parametrize("name", sorted(FETCHERS))
def test_only_a_404_is_a_missing_plane(name: str, tmp_path: Path) -> None:
    assert _run(name, tmp_path, syn.SpotVenue(MARKET), MISSING).outcome == fl.MISSING
    url = ar.PLANES[name].url(DAY)
    for error in (
        urllib.error.URLError("reset"),
        urllib.error.HTTPError(url, 503, "busy", None, None),
    ):
        assert _run(name, tmp_path, Faulty({url: error})).outcome == fl.TRANSIENT


@pytest.mark.parametrize("name", sorted(FETCHERS))
def test_bytes_that_do_not_decompress_are_corrupt(name: str, tmp_path: Path) -> None:
    url = ar.PLANES[name].url(DAY)
    replace: dict[str, bytes | Exception] = {url: b"not an archive"}
    if name == "binance_prints":
        replace[ar.checksum_url(ar.BINANCE_PRINTS, DAY)] = MARKET.binance_checksum(DAY)
    outcome = _run(name, tmp_path, Faulty(replace))
    assert outcome.outcome == fl.CORRUPT and outcome.sha256


def _zip(text: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as packed:
        packed.writestr("x.csv", text)
    return buffer.getvalue()


def test_an_archive_that_decompresses_and_does_not_parse_failed_its_replay(tmp_path: Path) -> None:
    book = Faulty({ar.BYBIT_BOOK.url(DAY): _zip("not json\n")})
    assert _run("bybit_book", tmp_path, book).outcome == fl.REPLAY_FAILED
    layout = gzip.compress(b"timestamp,symbol,side,size,price\n1.5,BTCUSDT,Buy,1,10\n")
    assert (
        _run("bybit_prints", tmp_path, Faulty({ar.BYBIT_PRINTS.url(DAY): layout})).outcome
        == fl.REPLAY_FAILED
    )


def test_binance_is_read_only_against_its_published_checksum(tmp_path: Path) -> None:
    checksum = ar.checksum_url(ar.BINANCE_PRINTS, DAY)
    wrong = Faulty({checksum: b"0" * 64 + b"  x.zip\n"})
    assert _run("binance_prints", tmp_path, wrong).outcome == fl.CORRUPT
    gone = Faulty({checksum: urllib.error.HTTPError(checksum, 404, "Not Found", None, None)})
    outcome = _run("binance_prints", tmp_path, gone)
    assert outcome.outcome == fl.TRANSIENT and "checksum not fetched" in outcome.detail
    assert not ar.day_path(ar.BINANCE_PRINTS, DAY, tmp_path / "spot").exists()


@pytest.mark.parametrize("name", sorted(FETCHERS))
def test_a_refused_day_raises_before_any_request(name: str, tmp_path: Path) -> None:
    venue = syn.SpotVenue(MARKET)
    with pytest.raises(HeldOutLocked):
        FETCHERS[name](DAY, tmp_path, tmp_path / "a", access=Guard(()), opener=venue)
    assert venue.asked == []
