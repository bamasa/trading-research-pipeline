"""Bybit's funding history: parsing, paging backwards, and the settlement grid."""

from __future__ import annotations

import io
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from trading_research.data import bybit_funding
from trading_research.data.bybit_funding import (
    PAGE_LIMIT,
    BybitFundingError,
    download_range,
    fetch,
    load_day,
    parse,
    settlement_interval_h,
)

HOUR_MS = 3_600_000


def _ms(day: date, hour: int = 0) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp() * 1000) + (
        hour * HOUR_MS
    )


def _payload(stamps: list[int], rate: float = 1e-4) -> dict[str, Any]:
    """Newest first, as the venue returns it."""
    rows = [
        {"symbol": "TESTUSDT", "fundingRate": str(rate), "fundingRateTimestamp": str(s)}
        for s in sorted(stamps, reverse=True)
    ]
    return {"retCode": 0, "retMsg": "OK", "result": {"category": "linear", "list": rows}}


class _Venue:
    """Answers each request with the newest ``PAGE_LIMIT`` settlements in range."""

    def __init__(self, stamps: list[int]) -> None:
        self.stamps = stamps
        self.urls: list[str] = []

    def __call__(self, url: str, timeout: float) -> io.BytesIO:
        self.urls.append(url)
        query = dict(part.split("=") for part in url.split("?")[1].split("&"))
        start, end = int(query["startTime"]), int(query["endTime"])
        inside = sorted((s for s in self.stamps if start <= s <= end), reverse=True)
        return io.BytesIO(json.dumps(_payload(inside[:PAGE_LIMIT])).encode())


def test_a_response_is_parsed_oldest_first() -> None:
    day = date(2024, 2, 15)
    frame = parse(_payload([_ms(day, 16), _ms(day, 0), _ms(day, 8)], rate=-2.5e-4))
    assert list(frame["timestamp"].dt.hour) == [0, 8, 16]
    assert frame["rate"].tolist() == [-2.5e-4] * 3


def test_a_refusal_from_the_venue_raises() -> None:
    with pytest.raises(BybitFundingError, match="retCode"):
        parse({"retCode": 10001, "retMsg": "params error"})


def test_a_long_span_is_fetched_backwards_a_page_at_a_time() -> None:
    start, end = date(2024, 1, 1), date(2024, 3, 31)
    stamps = [_ms(start) + 8 * HOUR_MS * i for i in range(91 * 3)]
    venue = _Venue(stamps)
    frame = fetch("TESTUSDT", start, end, opener=venue)
    assert len(frame) == len(stamps)
    assert frame["timestamp"].is_monotonic_increasing
    assert len(venue.urls) == 2


def test_a_span_is_written_a_file_per_day(tmp_path: Path) -> None:
    start, end = date(2024, 2, 14), date(2024, 2, 16)
    stamps = [_ms(start) + 8 * HOUR_MS * i for i in range(9)]
    written = download_range("TESTUSDT", start, end, tmp_path, opener=_Venue(stamps))
    assert [p.stem for p in written] == ["2024-02-14", "2024-02-15", "2024-02-16"]
    day = load_day("TESTUSDT", date(2024, 2, 15), tmp_path)
    assert list(day["timestamp"].dt.hour) == [0, 8, 16]
    with pytest.raises(BybitFundingError, match="no funding"):
        load_day("TESTUSDT", date(2024, 2, 17), tmp_path)


def test_a_day_before_listing_gets_no_file(tmp_path: Path) -> None:
    start, end = date(2024, 2, 14), date(2024, 2, 15)
    written = download_range(
        "TESTUSDT", start, end, tmp_path, opener=_Venue([_ms(end, h) for h in (0, 8, 16)])
    )
    assert [p.stem for p in written] == ["2024-02-15"]


def test_the_settlement_interval_is_read_and_checked() -> None:
    day = date(2024, 2, 15)
    eight = pd.to_datetime([_ms(day, h) for h in (0, 8, 16)], unit="ms", utc=True)
    assert settlement_interval_h(eight) == 8
    four = pd.to_datetime([_ms(day, h) for h in (0, 4, 8, 12)], unit="ms", utc=True)
    assert settlement_interval_h(four) == 4
    assert settlement_interval_h(eight[:1]) is None
    with pytest.raises(BybitFundingError, match="not uniform"):
        settlement_interval_h(pd.to_datetime([_ms(day, h) for h in (0, 8, 12)], unit="ms"))
    with pytest.raises(BybitFundingError, match="boundaries"):
        settlement_interval_h(pd.to_datetime([_ms(day, h) for h in (1, 9, 17)], unit="ms"))


def test_the_request_names_the_symbol_and_the_span() -> None:
    url = bybit_funding.request_url("BICOUSDT", 1, 2)
    assert "category=linear" in url and "symbol=BICOUSDT" in url
    assert "startTime=1" in url and "endTime=2" in url
