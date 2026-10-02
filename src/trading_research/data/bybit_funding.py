"""Bybit's funding history, which a position held through a settlement pays.

A perpetual has no expiry, so the venue ties its price to the index by charging
one side and paying the other at fixed settlement times — every eight hours, at
00:00, 08:00 and 16:00 UTC, for the instruments studied here. A positive rate
means longs pay shorts. For a market maker that holds inventory across a
settlement this is a cash flow like any other, and leaving it out would credit
or charge the strategy with money it never paid or received.

Unlike the book and the prints, Bybit does not publish funding as daily
archives. It is served by the public REST endpoint, without credentials, newest
first and at most two hundred rows a request — about sixty-six days at three
settlements a day — so a span is fetched backwards a page at a time and cut into
the same one-file-per-day layout as everything else, so the plan, resume and
skip logic in :mod:`trading_research.data.ensure` is shared.

The settlement interval is not assumed. It is read from the history and checked
(:func:`settlement_interval_h`), because a venue can change it per instrument,
and a simulator settling every eight hours on an instrument that settles every
four would get a third of the cash flow wrong without anything failing.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

#: The public endpoint. ``category=linear`` is the USDT perpetuals.
ENDPOINT = "https://api.bybit.com/v5/market/funding/history"

#: The most rows the endpoint returns in one response.
PAGE_LIMIT = 200

_MS_PER_DAY = 86_400_000


class BybitFundingError(RuntimeError):
    """The history could not be fetched or made sense of."""


Opener = Callable[..., Any]


def request_url(symbol: str, start_ms: int, end_ms: int, *, limit: int = PAGE_LIMIT) -> str:
    """The URL for one page: settlements in ``[start_ms, end_ms]``, newest first."""
    query = urllib.parse.urlencode(
        {
            "category": "linear",
            "symbol": symbol,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": limit,
        }
    )
    return f"{ENDPOINT}?{query}"


def parse(payload: dict[str, Any]) -> pd.DataFrame:
    """One response into ``timestamp`` (UTC) and ``rate``, oldest first.

    ``rate`` is a fraction per settlement, as the venue publishes it: 0.0001 is
    one basis point of notional, paid by longs to shorts when positive.
    """
    if payload.get("retCode") != 0:
        raise BybitFundingError(
            f"venue returned retCode={payload.get('retCode')!r}: {payload.get('retMsg')!r}"
        )
    rows = payload.get("result", {}).get("list", [])
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [int(row["fundingRateTimestamp"]) for row in rows], unit="ms", utc=True
            ),
            "rate": pd.Series([float(row["fundingRate"]) for row in rows], dtype="float64"),
        }
    )
    return frame.sort_values("timestamp", kind="stable").reset_index(drop=True)


def _ms(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp() * 1000)


def fetch(
    symbol: str,
    start: date,
    end: date,
    *,
    timeout: float = 30.0,
    opener: Opener = urllib.request.urlopen,
) -> pd.DataFrame:
    """Every settlement of ``symbol`` from ``start`` to ``end`` inclusive.

    Pages backwards from the end of the span: each response holds the newest
    rows up to its ``endTime``, so the next request ends one millisecond before
    the oldest row received. ``opener`` exists so the paging can be tested
    without the network.
    """
    if end < start:
        raise ValueError(f"end {end} is before start {start}")
    first = _ms(start)
    last = _ms(end) + _MS_PER_DAY - 1
    pages: list[pd.DataFrame] = []
    cursor = last
    while cursor >= first:
        url = request_url(symbol, first, cursor)
        try:
            with opener(url, timeout=timeout) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            raise BybitFundingError(f"{symbol}: {exc.code} {exc.reason}") from exc
        except OSError as exc:
            raise BybitFundingError(f"{symbol}: {exc}") from exc
        page = parse(payload)
        if page.empty:
            break
        pages.append(page)
        oldest = int(page["timestamp"].iloc[0].value // 1_000_000)
        if len(page) < PAGE_LIMIT:
            break
        cursor = oldest - 1
    if not pages:
        return parse({"retCode": 0, "result": {"list": []}})
    frame = pd.concat(pages, ignore_index=True)
    frame = frame.drop_duplicates(subset="timestamp").sort_values("timestamp", kind="stable")
    return frame.reset_index(drop=True)


def day_path(symbol: str, day: date, root: Path | str) -> Path:
    return Path(root) / symbol / f"{day.isoformat()}.parquet"


def download_range(
    symbol: str,
    start: date,
    end: date,
    root: Path | str,
    *,
    timeout: float = 30.0,
    opener: Opener = urllib.request.urlopen,
) -> list[Path]:
    """Fetch a span and write one file per day that has settlements.

    A day with no settlement — before the instrument was listed — gets no file,
    so :func:`trading_research.data.ensure.plan` keeps reporting it as missing
    rather than mistaking it for a day that was fetched and found empty.
    """
    frame = fetch(symbol, start, end, timeout=timeout, opener=opener)
    written: list[Path] = []
    if frame.empty:
        return written
    days = frame["timestamp"].dt.date
    for day, rows in frame.groupby(days, sort=True):
        if not isinstance(day, date) or day < start or day > end:
            continue
        target = day_path(symbol, day, root)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(f".{os.getpid()}.part")
        rows.reset_index(drop=True).to_parquet(partial, index=False)
        partial.replace(target)
        written.append(target)
    return written


def load_day(symbol: str, day: date, root: Path | str) -> pd.DataFrame:
    """One instrument-day of settlements, as written by :func:`download_range`."""
    path = day_path(symbol, day, root)
    if not path.exists() or path.stat().st_size == 0:
        raise BybitFundingError(f"no funding for {symbol} on {day} under {root}")
    return pd.read_parquet(path)


def settlement_interval_h(timestamps: pd.Series | pd.DatetimeIndex) -> int | None:
    """The settlement interval the history shows, in hours, or None if it cannot say.

    Every gap between consecutive settlements must be the same whole number of
    hours, and every settlement must fall on a multiple of it from midnight UTC.
    Anything else raises: an instrument whose interval changed inside the span
    needs a decision, not a guess.
    """
    index = pd.DatetimeIndex(timestamps)
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    stamps = np.sort(index.as_unit("s").to_numpy().astype(np.int64))
    if len(stamps) < 2:
        return None
    gaps = set(np.diff(stamps).tolist())
    if len(gaps) != 1:
        raise BybitFundingError(f"settlement gaps are not uniform: {sorted(gaps)} seconds")
    gap = gaps.pop()
    if gap <= 0 or gap % 3600:
        raise BybitFundingError(f"a settlement gap of {gap} seconds is not a whole number of hours")
    if np.any(stamps % gap):
        raise BybitFundingError(f"settlements do not fall on {gap // 3600}-hour boundaries")
    return int(gap // 3600)
