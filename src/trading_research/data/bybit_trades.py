"""Bybit's public trade prints, which are what fills a resting order.

The order-book archives this project already uses say what was *offered*. They
cannot say what was *taken*, and a maker model needs exactly that: an order
resting in a queue moves up it only when someone crosses the spread and consumes
the size ahead of it.

Bybit publishes the prints separately, and cheaply — a day of a mid-cap
instrument is a few hundred kilobytes against two hundred megabytes for its
book. The column that matters is ``side``, which names the **aggressor**: a
``Buy`` row is a buyer lifting an offer, so it consumes the ask queue; a
``Sell`` row hits a bid and consumes the bid queue. Getting that backwards would
fill every resting order at exactly the wrong moment, so it is asserted in the
tests rather than trusted.
"""

from __future__ import annotations

import gzip
import io
import os
import urllib.error
import urllib.request
from collections.abc import Iterator
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ARCHIVE = "https://public.bybit.com/trading/{symbol}/{symbol}{day}.csv.gz"


class BybitTradesError(RuntimeError):
    """The archive could not be fetched or made sense of."""


def archive_url(symbol: str, day: date) -> str:
    return ARCHIVE.format(symbol=symbol, day=day.isoformat())


def _days(start: date, end: date) -> Iterator[date]:
    day = start
    while day <= end:
        yield day
        day += timedelta(days=1)


def parse(raw: bytes) -> pd.DataFrame:
    """Turn one day's CSV into the project's trade contract.

    ``aggressor`` is +1 when a buyer crossed and -1 when a seller did, which is
    the sign convention the rest of the project uses for direction. Bybit's own
    ``side`` column is the aggressor's side, not the maker's — the opposite of
    Binance's ``is_buyer_maker``, and the kind of detail that inverts a result
    silently.
    """
    frame = pd.read_csv(io.BytesIO(raw), usecols=["timestamp", "side", "size", "price"])
    if frame.empty:
        raise BybitTradesError("archive contained no rows")
    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(frame["timestamp"], unit="s", utc=True),
            "price": frame["price"].astype("float64"),
            "size": frame["size"].astype("float64"),
            "aggressor": (frame["side"].astype("string") == "Buy").map({True: 1, False: -1}),
        }
    )
    return out.sort_values("timestamp").reset_index(drop=True)


def download_day(symbol: str, day: date, out_dir: Path | str, *, timeout: float = 120.0) -> Path:
    """Fetch and convert one day, reusing what is already on disk."""
    out = Path(out_dir) / symbol
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"{day.isoformat()}.parquet"
    if target.exists() and target.stat().st_size > 0:
        return target

    try:
        with urllib.request.urlopen(archive_url(symbol, day), timeout=timeout) as response:
            raw = gzip.decompress(response.read())
    except urllib.error.HTTPError as exc:
        raise BybitTradesError(f"{symbol} {day}: {exc.code} {exc.reason}") from exc
    except OSError as exc:
        raise BybitTradesError(f"{symbol} {day}: {exc}") from exc

    # The temporary name carries the process id: two workers asked for the
    # same day would otherwise write the same ".part" file and one would
    # rename it out from under the other.
    partial = target.with_suffix(f".{os.getpid()}.part")
    parse(raw).to_parquet(partial, index=False)
    partial.replace(target)
    return target


def download_range(
    symbol: str, start: date, end: date, out_dir: Path | str, *, on_day: object = None
) -> list[Path]:
    """Fetch a span, skipping days the archive does not have.

    A missing day is reported and passed over rather than ending the run: an
    instrument listed mid-span has no archive before it existed, which is not an
    error.
    """
    written = []
    for day in _days(start, end):
        try:
            written.append(download_day(symbol, day, out_dir))
        except BybitTradesError as exc:
            if callable(on_day):
                on_day(day, str(exc))
            continue
        if callable(on_day):
            on_day(day, "ok")
    return written


def load(symbol: str, root: Path | str) -> pd.DataFrame:
    files = sorted((Path(root) / symbol).glob("*.parquet"))
    if not files:
        raise BybitTradesError(f"no trades for {symbol} under {root}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
