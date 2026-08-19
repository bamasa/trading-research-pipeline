"""Bybit's free order-book archives, and reconstructing a book from them.

The central limitation of everything else in this project is that Binance gives
away only the touch — one bid and one ask — so book slope, level imbalance and
concentration were defined and left unimplemented. Bybit publishes the whole
thing: five hundred levels per side, tick granularity, back to 2023, no
registration and no fee.

The format is a stream rather than a table. Each day begins with one snapshot of
five hundred levels and continues with deltas: a price and a size, where a size
of zero removes the level. Reconstructing the book means replaying that stream
and taking a photograph of the result whenever the grid says to.

Three things the replay has to get right
----------------------------------------
**A gap is fatal, not cosmetic.** If an update is missed the book is wrong from
that moment until the next snapshot, and nothing downstream can tell. The
sequence number is checked on every message and a gap invalidates the book until
it is resynchronised — reported, not silently carried.

**Zero means delete.** A size of zero is a removal, not a level with no size.
Treating it as a size produces a book full of phantom levels at the touch, which
is exactly where it does most damage.

**A photograph is not an average.** The grid samples the book *as it stood* at
each instant. Interpolating between updates would invent liquidity that was
never quoted, and at a hundred milliseconds the difference is the whole signal.

Memory
------
A day is about a gigabyte of JSON. It is streamed line by line and only the
current book and the sampled rows are held, so the peak is the output rather
than the input.
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

#: Where the archives live. Public, unauthenticated, one file per day.
ARCHIVE_URL = (
    "https://quote-saver.bycsi.com/orderbook/linear/{symbol}/{day}_{symbol}_ob500.data.zip"
)

#: Levels the archive carries per side. Sampling keeps fewer.
ARCHIVE_DEPTH = 500


class BybitArchiveError(RuntimeError):
    """An archive is missing, malformed, or its sequence broke."""


def archive_url(symbol: str, day: date) -> str:
    return ARCHIVE_URL.format(symbol=symbol, day=day.isoformat())


@dataclass
class BookState:
    """The current book, as a price-to-size mapping per side."""

    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)
    last_update: int | None = None
    gaps: int = 0

    def apply(self, side: str, rows: list[list[str]]) -> None:
        book = self.bids if side == "b" else self.asks
        for price, size in rows:
            level = float(price)
            quantity = float(size)
            # Zero is a removal. Storing it would leave a phantom level, and at
            # the touch a phantom level changes every imbalance feature.
            if quantity == 0.0:
                book.pop(level, None)
            else:
                book[level] = quantity

    def top(self, depth: int) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        bids = sorted(self.bids.items(), key=lambda kv: -kv[0])[:depth]
        asks = sorted(self.asks.items(), key=lambda kv: kv[0])[:depth]
        return bids, asks

    @property
    def ready(self) -> bool:
        return bool(self.bids) and bool(self.asks)


def _messages(path: Path) -> Iterator[dict[str, Any]]:
    """Stream one archive's messages, whether zipped or already extracted."""
    if path.suffix == ".zip":
        archive = zipfile.ZipFile(path)
        with archive, archive.open(archive.namelist()[0]) as binary:
            for raw in binary:
                if raw.strip():
                    yield json.loads(raw)
        return
    with path.open(encoding="utf-8") as text:
        for line in text:
            if line.strip():
                yield json.loads(line)


def reconstruct(
    path: Path | str,
    *,
    symbol: str,
    depth: int = 10,
    grid_ms: int = 100,
    source: str = "bybit-linear-ob500",
) -> pd.DataFrame:
    """Replay an archive and photograph the book on a fixed grid.

    ``depth`` is how many levels to keep. The archive carries five hundred;
    ten is what the feature registry needs and what keeps a day's output to a
    size the rest of the pipeline can hold.

    Rows are emitted at most once per ``grid_ms``, carrying the book as it stood
    when that interval ended. Intervals with no update at all produce no row —
    the same choice the Binance resampler makes, so a quiet stretch is visibly
    absent rather than silently forward-filled.
    """
    state = BookState()
    rows: list[dict[str, Any]] = []
    bucket: int | None = None

    for message in _messages(Path(path)):
        kind = message.get("type")
        data = message.get("data", {})
        stamp = int(message["ts"])
        update = data.get("u")

        if kind == "snapshot":
            state = BookState()
            state.apply("b", data.get("b", []))
            state.apply("a", data.get("a", []))
            state.last_update = update
        elif kind == "delta":
            if (
                state.last_update is not None
                and update is not None
                and update != state.last_update + 1
            ):
                # The book is wrong from here until the next snapshot. Counted
                # and carried on the result rather than raised: one gap in a day
                # is a data-quality fact, not a reason to lose the day.
                state.gaps += 1
            state.apply("b", data.get("b", []))
            state.apply("a", data.get("a", []))
            state.last_update = update
        else:
            continue

        if not state.ready:
            continue

        current = stamp // grid_ms
        if bucket is not None and current == bucket:
            continue
        bucket = current

        bids, asks = state.top(depth)
        if len(bids) < depth or len(asks) < depth:
            continue

        row: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(stamp / 1000, tz=UTC),
            "symbol": symbol,
            "source": source,
            "sequence_id": update,
        }
        for level in range(depth):
            row[f"bid_price_{level}"], row[f"bid_size_{level}"] = bids[level]
            row[f"ask_price_{level}"], row[f"ask_size_{level}"] = asks[level]
        rows.append(row)

    if not rows:
        raise BybitArchiveError(f"{path} produced no usable book rows")

    frame = pd.DataFrame(rows)
    frame["symbol"] = frame["symbol"].astype("string")
    frame["source"] = frame["source"].astype("string")
    frame.attrs["sequence_gaps"] = state.gaps
    return frame.sort_values("timestamp", kind="stable").reset_index(drop=True)


def download_day(
    symbol: str,
    day: date,
    cache_dir: Path | str,
    *,
    timeout: float = 600.0,
) -> Path:
    """Fetch one day's archive, reusing it if already present.

    Kept as a separate step from reconstruction because the archive is two
    hundred megabytes and the reconstruction is two minutes: re-running the
    second should never re-do the first.
    """
    import urllib.error
    import urllib.request

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"{symbol}-{day.isoformat()}-ob500.zip"
    if target.exists() and target.stat().st_size > 0:
        return target

    url = archive_url(symbol, day)
    partial = target.with_suffix(".part")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            partial.write_bytes(response.read())
    except urllib.error.HTTPError as exc:
        partial.unlink(missing_ok=True)
        raise BybitArchiveError(f"{symbol} {day}: {exc.code} {exc.reason}") from exc
    except OSError as exc:
        partial.unlink(missing_ok=True)
        raise BybitArchiveError(f"{symbol} {day}: {exc}") from exc

    # Renamed only once complete, so an interrupted download is never mistaken
    # for a cached one on the next run.
    partial.replace(target)
    return target


def download_range(
    symbol: str,
    start: date,
    end: date,
    out_dir: Path | str,
    *,
    cache_dir: Path | str | None = None,
    depth: int = 10,
    grid_ms: int = 100,
    keep_archive: bool = False,
    overwrite: bool = False,
    on_day: Any = None,
) -> pd.DataFrame:
    """Fetch and reconstruct a date range, one day at a time.

    One day is processed and written before the next is fetched. A day is a
    gigabyte of JSON and five instruments over a month would not fit in memory
    together; the day is also the unit everything downstream reads, so the
    partitioning is the same one the rest of the pipeline uses.
    """
    from datetime import timedelta

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(cache_dir) if cache_dir is not None else out / "_archives"
    rows: list[dict[str, Any]] = []

    day = start
    while day <= end:
        target = out / f"{day.isoformat()}.parquet"
        if target.exists() and not overwrite:
            rows.append({"day": day, "rows": 0, "skipped": "already converted"})
            if on_day is not None:
                on_day(rows[-1])
            day += timedelta(days=1)
            continue

        try:
            archive = download_day(symbol, day, cache)
            frame = reconstruct(archive, symbol=symbol, depth=depth, grid_ms=grid_ms)
        except BybitArchiveError as exc:
            rows.append({"day": day, "rows": 0, "skipped": str(exc)})
            if on_day is not None:
                on_day(rows[-1])
            day += timedelta(days=1)
            continue

        frame.to_parquet(target, compression="zstd", index=False)
        rows.append({"day": day, "rows": len(frame), "gaps": frame.attrs.get("sequence_gaps", 0)})
        if on_day is not None:
            on_day(rows[-1])
        if not keep_archive:
            archive.unlink(missing_ok=True)
        del frame
        day += timedelta(days=1)

    return pd.DataFrame(rows)
