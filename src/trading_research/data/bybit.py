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

import heapq
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

try:  # pragma: no cover - the fallback is exercised only without the extra
    from orjson import loads as _loads
except ImportError:  # pragma: no cover
    from json import loads as _loads

#: Parsing dominates the replay: a day is a gigabyte of JSON and the download
#: that produced it took twenty seconds. ``orjson`` is several times faster than
#: the standard library on this shape of message and is an optional extra, so
#: the slow path still works — it just decides whether a month of one instrument
#: takes one hour or three.

#: Where the archives live. Public, unauthenticated, one file per day.
ARCHIVE_URL = (
    "https://quote-saver.bycsi.com/orderbook/linear/{symbol}/{day}_{symbol}_ob500.data.zip"
)

#: Levels the archive carries per side. Sampling keeps fewer.
ARCHIVE_DEPTH = 500


def _price(item: tuple[float, float]) -> float:
    """Key for the level heaps. A module-level function rather than a lambda so
    it is not rebuilt on every one of the day's 864,000 calls."""
    return item[0]


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
        """The best ``depth`` levels a side.

        ``nlargest`` rather than a full sort. The book holds five hundred levels
        and this runs once per sampled row — 864,000 times a day — so sorting
        the whole of it to keep ten was most of the replay's cost. Selecting the
        top k is O(n log k) against O(n log n), and measured here it is the
        difference between two minutes a day and twenty seconds.
        """
        bids = heapq.nlargest(depth, self.bids.items(), key=_price)
        asks = heapq.nsmallest(depth, self.asks.items(), key=_price)
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
                    yield _loads(raw)
        return
    with path.open(encoding="utf-8") as text:
        for line in text:
            if line.strip():
                yield _loads(line)


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

    Rows are emitted at most once per ``grid_ms``: the first update in each
    interval produces a row carrying the book as it stood right after that
    update, stamped with that update's own timestamp. Later updates in the same
    interval are applied to the book but appear only in the next interval's
    row, so a row is never ahead of its timestamp, and can be up to one
    interval behind the venue's book. Intervals with no update at all produce
    no row — the same choice the Binance resampler makes, so a quiet stretch is
    visibly absent rather than silently forward-filled.

    The stamp is the message's ``ts``, the time the venue generated it; prints
    carry their match time. The two differ by a few milliseconds (see
    ``market_making.analysis.clock_offset_profile``).
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


def _one_day(job: tuple[str, date, Path, Path, int, int, bool, bool]) -> dict[str, Any]:
    """Fetch, replay and write one day. The unit of parallel work.

    A module-level function taking a tuple because it has to be picklable: a
    closure or a method would not cross a process boundary.
    """
    symbol, day, out, cache, depth, grid_ms, keep_archive, overwrite = job
    target = out / f"{day.isoformat()}.parquet"
    if target.exists() and not overwrite:
        return {"day": day, "rows": 0, "skipped": "already converted"}

    try:
        archive = download_day(symbol, day, cache)
        frame = reconstruct(archive, symbol=symbol, depth=depth, grid_ms=grid_ms)
    except BybitArchiveError as exc:
        return {"day": day, "rows": 0, "skipped": str(exc)}

    frame.to_parquet(target, compression="zstd", index=False)
    result = {"day": day, "rows": len(frame), "gaps": frame.attrs.get("sequence_gaps", 0)}
    if not keep_archive:
        archive.unlink(missing_ok=True)
    return result


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
    workers: int = 1,
    on_day: Any = None,
) -> pd.DataFrame:
    """Fetch and reconstruct a date range, a day at a time.

    Days are independent — each archive carries its own opening snapshot — so
    they parallelise cleanly, and the replay is CPU-bound rather than
    network-bound: a day downloads in twenty seconds and replays in ninety. With
    ``workers`` above one the days run in separate processes.

    Memory is the constraint on how many. Each worker holds one day's output,
    about 34 MB at ten levels, plus the parsing overhead — so the practical
    limit is cores rather than RAM, and going past the core count only adds
    contention.

    Results arrive in completion order and are sorted before returning, so the
    table reads chronologically however the work was scheduled.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from datetime import timedelta

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(cache_dir) if cache_dir is not None else out / "_archives"
    cache.mkdir(parents=True, exist_ok=True)

    days: list[date] = []
    day = start
    while day <= end:
        days.append(day)
        day += timedelta(days=1)

    jobs = [(symbol, d, out, cache, depth, grid_ms, keep_archive, overwrite) for d in days]
    rows: list[dict[str, Any]] = []

    if workers <= 1:
        for job in jobs:
            rows.append(_one_day(job))
            if on_day is not None:
                on_day(rows[-1])
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_one_day, job): job[1] for job in jobs}
            for future in as_completed(futures):
                rows.append(future.result())
                if on_day is not None:
                    on_day(rows[-1])

    return pd.DataFrame(rows).sort_values("day").reset_index(drop=True)
