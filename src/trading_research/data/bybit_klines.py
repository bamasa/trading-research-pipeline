"""Bybit's hourly closes of a USDT perpetual, from the public kline endpoint.

``/v5/market/kline`` serves at most a thousand bars a request, newest first,
without credentials. A span is fetched forwards a page at a time and written
one file per day (``open_time``, ``close``), as the other loaders here are, so
a day on disk is never fetched again. A bar is labelled by its open, like
Binance's klines, so both venues' hours line up.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

ENDPOINT = "https://api.bybit.com/v5/market/kline"
PAGE_LIMIT = 1000
_HOUR_MS = 3_600_000

Opener = Callable[..., Any]


class BybitKlineError(RuntimeError):
    """The bars could not be fetched or made sense of."""


def request_url(symbol: str, start_ms: int, end_ms: int) -> str:
    query = urllib.parse.urlencode(
        {"category": "linear", "symbol": symbol, "interval": "60", "start": start_ms,
         "end": end_ms, "limit": PAGE_LIMIT}
    )  # fmt: skip
    return f"{ENDPOINT}?{query}"


def parse(payload: dict[str, Any]) -> pd.DataFrame:
    """One response into ``open_time`` (UTC) and ``close``, oldest first."""
    if payload.get("retCode") != 0:
        raise BybitKlineError(f"retCode={payload.get('retCode')!r}: {payload.get('retMsg')!r}")
    rows = payload.get("result", {}).get("list", [])
    frame = pd.DataFrame(
        {
            "open_time": pd.to_datetime([int(r[0]) for r in rows], unit="ms", utc=True),
            "close": pd.Series([float(r[4]) for r in rows], dtype="float64"),
        }
    )
    return frame.sort_values("open_time", kind="stable").reset_index(drop=True)


def _ms(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp() * 1000)


def fetch(
    symbol: str, start: date, end: date, *, opener: Opener = urllib.request.urlopen
) -> pd.DataFrame:
    """Every hourly bar of ``symbol`` opening from ``start`` to the end of ``end``."""
    first, last = _ms(start), _ms(end + timedelta(days=1)) - _HOUR_MS
    pages, cursor = [], first
    while cursor <= last:
        top = min(cursor + (PAGE_LIMIT - 1) * _HOUR_MS, last)
        with opener(request_url(symbol, cursor, top), timeout=30.0) as response:
            page = parse(json.loads(response.read()))
        if not page.empty:
            pages.append(page)
        cursor = top + _HOUR_MS
    if not pages:
        return parse({"retCode": 0, "result": {"list": []}})
    frame = pd.concat(pages, ignore_index=True).drop_duplicates("open_time")
    return frame.sort_values("open_time", kind="stable").reset_index(drop=True)


def day_path(symbol: str, day: date, root: Path) -> Path:
    return root / symbol / "klines-1h" / f"{day.isoformat()}.parquet"


def download_range(
    symbol: str, start: date, end: date, root: Path, *, opener: Opener = urllib.request.urlopen
) -> None:
    """Fetch the days of the span not on disk and write one file per day with bars."""
    days = [start + timedelta(days=k) for k in range((end - start).days + 1)]
    missing = [d for d in days if not day_path(symbol, d, root).exists()]
    if not missing:
        return
    frame = fetch(symbol, missing[0], missing[-1], opener=opener)
    for day, rows in frame.groupby(frame["open_time"].dt.date, sort=True):
        target = day_path(symbol, day, root)
        if day in missing:
            target.parent.mkdir(parents=True, exist_ok=True)
            rows.reset_index(drop=True).to_parquet(target, index=False)


def load(symbol: str, start: date, end: date, root: Path) -> pd.Series:
    """Hourly closes from disk, indexed by the bar's open."""
    days = [start + timedelta(days=k) for k in range((end - start).days + 1)]
    files = [day_path(symbol, d, root) for d in days]
    frames = [pd.read_parquet(f) for f in files if f.exists()]
    if not frames:
        return pd.Series(dtype="float64")
    f = pd.concat(frames, ignore_index=True)
    return pd.Series(f["close"].to_numpy(), index=pd.DatetimeIndex(f["open_time"])).sort_index()
