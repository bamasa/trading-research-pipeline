"""The local spot venue: registered URLs only, the probed layouts, the same bytes, no socket."""

from __future__ import annotations

import gzip
import io
import itertools
import json
import socket
import urllib.error
import urllib.request
import zipfile
from datetime import date

import pytest

from trading_research.spot import archives as ar
from trading_research.spot import synthetic as syn

DAY = date(2026, 3, 3)
MARKET = syn.SpotMarket(frozenset({DAY}), step_ms=600_000)


def test_serving_routes_urlopen_to_the_venue_and_refuses_every_socket() -> None:
    venue = syn.SpotVenue(MARKET)
    with syn.serving(venue):
        with urllib.request.urlopen(ar.BYBIT_PRINTS.url(DAY)) as response:
            assert gzip.decompress(response.read()) == MARKET.bybit_prints_csv(DAY)
        with pytest.raises(OSError, match="no network"):
            socket.create_connection(("example.org", 443), timeout=1)
        with socket.socket() as raw, pytest.raises(OSError, match="no network"):
            raw.connect(("example.org", 443))
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen("https://example.org/")
    assert venue.unknown == ["https://example.org/"]


def test_only_the_registered_urls_and_days_are_answered() -> None:
    venue = syn.SpotVenue(MARKET)
    assert venue.body(ar.BYBIT_BOOK.url(DAY)) == MARKET.book_archive(DAY)
    assert venue.body(ar.BYBIT_BOOK.url(date(2026, 3, 4))) is None  # a missing day
    assert venue.unknown == []
    linear = ar.BYBIT_BOOK.url(DAY).replace("/spot/", "/linear/")
    monthly = ar.BINANCE_PRINTS.url(DAY).replace("/daily/", "/monthly/")
    assert venue.body(linear) is None and venue.body(monthly) is None
    assert venue.unknown == [linear, monthly]


def test_the_layouts_are_the_probed_ones() -> None:
    first = json.loads(MARKET.book_lines(DAY)[0])
    assert set(first) == {"topic", "ts", "type", "data", "cts"} and first["type"] == "snapshot"
    assert set(first["data"]) == {"s", "b", "a", "u", "seq"}
    header = MARKET.bybit_prints_csv(DAY).split(b"\n", 1)[0]
    assert header == b"id,timestamp,price,volume,side,rpi"
    with zipfile.ZipFile(io.BytesIO(MARKET.binance_archive(DAY))) as packed:
        row = packed.read(packed.namelist()[0]).split(b"\n", 1)[0].split(b",")
    assert len(row) == 8 and len(row[5]) == 16  # no header, microseconds in 2026
    old = syn.SpotMarket(frozenset({date(2024, 6, 1)}), step_ms=600_000)
    assert len(old.binance_rows(date(2024, 6, 1))[0].split(",")[5]) == 13  # milliseconds


def test_the_same_bytes_every_time_and_a_head_without_a_body() -> None:
    again = syn.SpotMarket(frozenset({DAY}), step_ms=600_000)
    assert again.book_archive(DAY) == MARKET.book_archive(DAY)
    assert again.binance_checksum(DAY) == MARKET.binance_checksum(DAY)
    venue = syn.SpotVenue(MARKET)
    request = urllib.request.Request(ar.BINANCE_PRINTS.url(DAY), method="HEAD")
    with venue(request) as response:
        assert response.read() == b""
        assert int(response.headers["Content-Length"]) == len(MARKET.binance_archive(DAY))


def test_faults_on_request() -> None:
    faulty = syn.SpotMarket(
        frozenset({DAY}), step_ms=600_000, gap_days=frozenset({DAY}), empty_days=frozenset({DAY})
    )
    updates = [json.loads(line)["data"]["u"] for line in faulty.book_lines(DAY)]
    assert sum(b - a != 1 for a, b in itertools.pairwise(updates)) == 3
    assert faulty.bybit_prints_csv(DAY) == b"id,timestamp,price,volume,side,rpi\n"
    assert faulty.binance_rows(DAY) == []
    assert set(faulty.present()) == set(ar.PLANES)
