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

    The sort is **stable**: prints sharing a timestamp keep the order the
    archive wrote them in. Distinct timestamps are only a third to two thirds of
    a day's prints, so ties are the norm, and pandas' default sort is free to
    shuffle them — a resting order's fills would then depend on the sorting
    algorithm. Bybit's ``trdMatchID`` is kept as ``match_id`` when the archive
    carries it, so a print can be traced back to the venue's own record. Files
    written before this change have neither property; the market-making loader
    imposes its own deterministic order within a timestamp, so it does not
    depend on either.
    """
    wanted = {"timestamp", "side", "size", "price", "trdMatchID"}
    frame = pd.read_csv(io.BytesIO(raw), usecols=lambda column: column in wanted)
    missing = sorted({"timestamp", "side", "size", "price"} - set(frame.columns))
    if missing:
        raise BybitTradesError(f"archive lacks columns {missing}")
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
    if "trdMatchID" in frame.columns:
        out["match_id"] = frame["trdMatchID"].astype("string")
    return out.sort_values("timestamp", kind="stable").reset_index(drop=True)


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


def day_path(symbol: str, day: date, root: Path | str) -> Path:
    """Where :func:`download_day` writes one instrument-day."""
    return Path(root) / symbol / f"{day.isoformat()}.parquet"


def load_day(symbol: str, day: date, root: Path | str) -> pd.DataFrame:
    """One instrument-day of prints, without touching any other day.

    The unit the market-making simulator works in: it reads one day, simulates
    it and lets it go, so its memory is one day's prints however long the span.
    A missing or empty file raises rather than returning an empty frame, since
    a day without prints is a day that cannot be simulated, not a quiet one.
    """
    path = day_path(symbol, day, root)
    if not path.exists() or path.stat().st_size == 0:
        raise BybitTradesError(f"no trades for {symbol} on {day} under {root}")
    return pd.read_parquet(path)


def load(symbol: str, root: Path | str) -> pd.DataFrame:
    """Every day of one instrument's prints, concatenated.

    Memory grows with the span: BTCUSDT runs to about 110 million rows over the
    days on disk here, several gigabytes once read. Anything that can work a
    day at a time should use :func:`load_day` instead.
    """
    files = sorted((Path(root) / symbol).glob("*.parquet"))
    if not files:
        raise BybitTradesError(f"no trades for {symbol} under {root}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
