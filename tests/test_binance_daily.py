"""Hourly bars from daily archives only.

Binance publishes klines both as daily and as monthly archives, and the
downloader prefers the month and caches it. For a study that may read a day only
once its block is opened, a cached month is a later block's bars on disk early.
The daily-only path must never ask for, read or write a monthly file.
"""

from __future__ import annotations

import io
import urllib.error
import zipfile
from datetime import UTC, date, datetime, timedelta
from email.message import Message
from pathlib import Path

import pandas as pd
import pytest

from trading_research.data import binance
from trading_research.data.binance import (
    ArchiveSpec,
    BinanceArchiveError,
    download_range,
    monthly_archives_beyond,
)

SPEC = ArchiveSpec(market="futures-um", kind="klines", symbol="TESTUSDT", interval="1h")
DAY = date(2024, 12, 2)


def _klines(day: date, hours: int = 24) -> bytes:
    start = datetime(day.year, day.month, day.day, tzinfo=UTC)
    lines = [
        "open_time,open,high,low,close,volume,close_time,quote_volume,count,"
        "taker_buy_volume,taker_buy_quote_volume,ignore"
    ]
    for h in range(hours):
        open_ms = int((start + timedelta(hours=h)).timestamp() * 1000)
        close_ms = open_ms + 3_600_000 - 1
        lines.append(f"{open_ms},100,101,99,100.5,10,{close_ms},1000,5,4,400,0")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("bars.csv", "\n".join(lines) + "\n")
    return buffer.getvalue()


def _http_error(url: str, code: int) -> BinanceArchiveError:
    try:
        raise urllib.error.HTTPError(url, code, "status", Message(), io.BytesIO(b""))
    except urllib.error.HTTPError as exc:
        try:
            raise BinanceArchiveError(f"{url}: HTTP {code}") from exc
        except BinanceArchiveError as wrapped:
            return wrapped


@pytest.fixture
def requested(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Serve every daily and monthly kline archive, recording what was asked."""
    urls: list[str] = []

    def fetch(url: str, *, verify: bool = True) -> bytes:
        urls.append(url)
        return _klines(DAY)

    monkeypatch.setattr(binance, "fetch_archive", fetch)
    return urls


def test_the_default_prefers_and_caches_the_month(tmp_path: Path, requested: list[str]) -> None:
    """The hazard the option exists for: one day fetched, the whole month kept."""
    download_range(SPEC, DAY, DAY, tmp_path, keep_raw=False)
    assert "/monthly/" in requested[0]
    assert monthly_archives_beyond(SPEC, tmp_path, DAY)


def test_daily_only_asks_for_the_day_alone(tmp_path: Path, requested: list[str]) -> None:
    results = download_range(SPEC, DAY, DAY, tmp_path, keep_raw=False, daily_only=True)
    assert requested == [SPEC.url(DAY, "daily")]
    assert results[0].output is not None
    bars = pd.read_parquet(results[0].output)
    assert len(bars) == 24
    assert (bars["open_time"].dt.date == DAY).all()
    assert not list((tmp_path / "_archives").rglob("*.zip"))


def test_daily_only_never_reads_a_cached_month(tmp_path: Path, requested: list[str]) -> None:
    cache = tmp_path / "_archives" / SPEC.symbol / SPEC.directory
    cache.mkdir(parents=True)
    (cache / SPEC.filename(DAY.replace(day=1), "monthly")).write_bytes(b"not read")
    download_range(SPEC, DAY, DAY, tmp_path, keep_raw=True, daily_only=True)
    assert requested == [SPEC.url(DAY, "daily")]
    assert (cache / SPEC.filename(DAY, "daily")).exists()  # the day's own archive


@pytest.mark.parametrize("code", [404, 503])
def test_a_failed_day_carries_its_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    def fetch(url: str, *, verify: bool = True) -> bytes:
        raise _http_error(url, code)

    monkeypatch.setattr(binance, "fetch_archive", fetch)
    results = download_range(SPEC, DAY, DAY, tmp_path, keep_raw=False, daily_only=True)
    assert results[0].output is None
    assert results[0].http_status == code


def test_a_monthly_only_kind_has_no_daily_path(tmp_path: Path, requested: list[str]) -> None:
    funding = ArchiveSpec(market="futures-um", kind="fundingRate", symbol="TESTUSDT")
    results = download_range(funding, DAY, DAY, tmp_path, daily_only=True)
    assert results[0].output is None
    assert "monthly only" in (results[0].skipped or "")
    assert requested == []


def test_months_past_the_last_permitted_day_are_found(tmp_path: Path) -> None:
    cache = tmp_path / "_archives" / SPEC.symbol / SPEC.directory
    cache.mkdir(parents=True)
    for name in ("2025-05", "2025-06", "2025-07", "2025-07-08"):
        (cache / f"TESTUSDT-1h-{name}.zip").write_bytes(b"")
    found = monthly_archives_beyond(SPEC, tmp_path, date(2025, 6, 30))
    assert [p.name for p in found] == ["TESTUSDT-1h-2025-07.zip"]
    assert [p.name for p in monthly_archives_beyond(SPEC, tmp_path, date(2025, 6, 29))] == [
        "TESTUSDT-1h-2025-06.zip",
        "TESTUSDT-1h-2025-07.zip",
    ]


def test_ensure_bars_passes_daily_only_through(tmp_path: Path, requested: list[str]) -> None:
    from trading_research.data.ensure import ensure_bars

    plan = ensure_bars("TESTUSDT", "1h", DAY, DAY, root=tmp_path, daily_only=True, on_progress=None)
    assert plan.complete
    assert requested == [SPEC.url(DAY, "daily")]
    assert monthly_archives_beyond(SPEC, tmp_path, DAY) == []
