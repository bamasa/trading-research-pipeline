"""Bybit's spot ``ob200`` book: downloaded, replayed through ``data/bybit.reconstruct``, counted.

The format probe of 2026-10-25 found the linear archive's shape (a snapshot,
then deltas of ``[price, size]`` strings with an update id ``u``), so the
replay is :func:`trading_research.data.bybit.reconstruct`, unchanged, on the
registered grid: ten levels, at most one row per 100 ms interval at its first
update, stamped with that update's venue time, no forward fill. The rows carry
the source tag ``bybit-spot-ob200`` and the venue-qualified instrument
``bybit-spot:BTCUSDT`` in their ``symbol`` column.

**Message statistics**, on the pattern of
:func:`trading_research.data.bybit.archive_message_stats` (:func:`message_stats`):
the counts by type, the median and largest interval between messages, the
first and last stamp, and the sequence gaps counted as ``reconstruct`` counts
them. Stage 0 records them on the reader-test days.

**Ten levels.** :func:`beyond_levels_share` is the share of book rows in
which the clip exceeds the visible levels on either side; stage 0 records it,
and above :data:`WALK_THRESHOLD` the walk replays all 200 levels
(``replay(depth=200)``, :func:`replay_walk`: the ten-level grid's rows with the
deeper levels the archive holds) while the features keep ten. The share is over
the grid's rows: an interval whose first update leaves a side with fewer than
ten levels has no row, as in 2025, so it is never traded nor walked.
"""

from __future__ import annotations

import re
import zipfile
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_research.data import bybit
from trading_research.data import fetch_log as fl
from trading_research.data.failures import Opener
from trading_research.data.study_fetch import DayOutcome, peak_rss_bytes
from trading_research.spot import fetch
from trading_research.spot.archives import BOOK, BYBIT_BOOK, BYBIT_SPOT, day_path
from trading_research.spot.instrument import CLIP_BTC

SOURCE = "bybit-spot-ob200"
LEVELS = 10
ARCHIVE_LEVELS = 200
GRID_MS = 100
#: Above this share of rows beyond ten levels, the walk uses all 200.
WALK_THRESHOLD = 0.001


def replay(archive: Path | str, *, depth: int = LEVELS, grid_ms: int = GRID_MS) -> pd.DataFrame:
    """One day's archive as book rows on the registered grid: ``data/bybit.reconstruct``
    at ten levels; at a deeper walk, :func:`replay_walk`, the same rows with the
    deeper levels the archive holds."""
    if depth <= LEVELS:
        return bybit.reconstruct(
            archive, symbol=BYBIT_SPOT, depth=depth, grid_ms=grid_ms, source=SOURCE
        )
    return replay_walk(archive, depth=depth, grid_ms=grid_ms)


def replay_walk(archive: Path | str, *, depth: int, grid_ms: int = GRID_MS) -> pd.DataFrame:
    """The ten-level grid's rows carrying up to ``depth`` levels for the walk.

    ``data/bybit.reconstruct`` at ``depth`` skips an interval whose first update
    leaves a side with fewer than ``depth`` levels, so a 200-level replay would
    lose rows of the registered ten-level grid. Here the book is replayed as
    ``reconstruct`` replays it (its :class:`~trading_research.data.bybit.BookState`
    and message reader, the same snapshot, delta, gap and interval rules), a row
    is emitted where the ten-level grid has one, and a level the archive does not
    hold at that moment is left empty (NaN, which the walk reads as no depth).
    The first ten levels of every row equal ``reconstruct(depth=10)``'s; the rows
    with an empty deep level are counted (``attrs["deep_levels_missing_rows"]``).
    """
    state = bybit.BookState()
    rows: list[list[Any]] = []
    bucket: int | None = None
    short = 0
    gaps = 0  # outside the book's state, as reconstruct keeps it: a snapshot resets the book, not the count
    for message in bybit._messages(Path(archive)):
        kind, data, stamp = message.get("type"), message.get("data", {}), int(message["ts"])
        update = data.get("u")
        if kind == "snapshot":
            state = bybit.BookState()
            state.apply("b", data.get("b", []))
            state.apply("a", data.get("a", []))
            state.last_update = update
        elif kind == "delta":
            if (
                state.last_update is not None
                and update is not None
                and update != state.last_update + 1
            ):
                gaps += 1
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
        if len(bids) < LEVELS or len(asks) < LEVELS:  # the ten-level grid's rule
            continue
        short += len(bids) < depth or len(asks) < depth
        empty = [(np.nan, np.nan)] * depth
        row: list[Any] = [stamp, update]
        for bid, ask in zip([*bids, *empty][:depth], [*asks, *empty][:depth], strict=True):
            row += [*bid, *ask]
        rows.append(row)
    if not rows:
        raise bybit.BybitArchiveError(f"{archive} produced no usable book rows")
    columns = ["ms", "sequence_id"]
    for level in range(depth):
        columns += [f"bid_price_{level}", f"bid_size_{level}", f"ask_price_{level}",
                    f"ask_size_{level}"]  # fmt: skip
    frame = pd.DataFrame(rows, columns=columns)
    stamps = pd.to_datetime(frame.pop("ms"), unit="ms", utc=True)
    frame.insert(0, "timestamp", stamps)
    frame.insert(1, "symbol", pd.Series(BYBIT_SPOT, index=frame.index, dtype="string"))
    frame.insert(2, "source", pd.Series(SOURCE, index=frame.index, dtype="string"))
    frame.attrs["sequence_gaps"] = gaps
    frame.attrs["deep_levels_missing_rows"] = short
    return frame.sort_values("timestamp", kind="stable").reset_index(drop=True)


@dataclass(frozen=True)
class SpotMessageStats:
    """What a day's message stream looked like, beside the book it builds."""

    messages: int
    snapshots: int
    deltas: int
    #: Lines that are neither a snapshot nor a delta (skipped by the replay).
    other: int
    median_interval_ms: float | None
    max_interval_ms: int | None
    first_ts: int | None
    last_ts: int | None
    sequence_gaps: int


_TYPE = re.compile(rb'"type"\s*:\s*"([a-z]+)"')
_TS = re.compile(rb'"ts"\s*:\s*(\d+)')
_UPDATE = re.compile(rb'"u"\s*:\s*(\d+)')


def _lines(path: Path) -> Iterator[bytes]:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as packed, packed.open(packed.namelist()[0]) as binary:
            yield from (raw for raw in binary if raw.strip())
        return
    with path.open("rb") as binary:
        yield from (raw for raw in binary if raw.strip())


def message_stats(path: Path | str) -> SpotMessageStats:
    """Count an archive's messages, their intervals and its sequence gaps."""
    stamps: list[int] = []
    snapshots = deltas = other = gaps = 0
    last: int | None = None
    for raw in _lines(Path(path)):
        kind, stamp, update = _TYPE.search(raw), _TS.search(raw), _UPDATE.search(raw)
        name = kind.group(1).decode() if kind else None
        found = int(update.group(1)) if update else None
        if name == "snapshot":
            snapshots += 1
            last = found
        elif name == "delta":
            deltas += 1
            if last is not None and found is not None and found != last + 1:
                gaps += 1
            last = found
        else:
            other += 1
            continue
        if stamp is not None:
            stamps.append(int(stamp.group(1)))
    intervals = np.diff(np.asarray(stamps, dtype=np.int64))
    return SpotMessageStats(
        messages=snapshots + deltas,
        snapshots=snapshots,
        deltas=deltas,
        other=other,
        median_interval_ms=float(np.median(intervals)) if len(intervals) else None,
        max_interval_ms=int(intervals.max()) if len(intervals) else None,
        first_ts=stamps[0] if stamps else None,
        last_ts=stamps[-1] if stamps else None,
        sequence_gaps=gaps,
    )


def beyond_levels_share(
    frame: pd.DataFrame, *, clip: float = CLIP_BTC, depth: int = LEVELS
) -> float:
    """Share of rows in which ``clip`` exceeds the ``depth`` visible levels on either side."""
    if frame.empty:
        raise ValueError("no book rows to measure")
    bids = np.nansum(frame[[f"bid_size_{k}" for k in range(depth)]].to_numpy(np.float64), axis=1)
    asks = np.nansum(frame[[f"ask_size_{k}" for k in range(depth)]].to_numpy(np.float64), axis=1)
    return float(np.mean((bids < clip) | (asks < clip)))


def fetch_day(
    day: date,
    root: Path | str,
    archive_dir: Path | str,
    *,
    access: fetch.AccessGuard,
    opener: Opener | None = None,
    timeout: float = 120.0,
    depth: int = LEVELS,
    grid_ms: int = GRID_MS,
) -> DayOutcome:
    """Fetch, check, count, replay and write one book day; delete its archive whatever happens.

    The stats carry the archive's message statistics, the replayed rows, the
    sequence gaps, the share of rows beyond ten levels and the worker's peak
    memory after the day.
    """
    access.require(BYBIT_SPOT, [day], BOOK, what="fetch")
    name = BYBIT_BOOK.name
    archive = Path(archive_dir) / f"{name}-{day.isoformat()}.zip"
    got, missed = fetch.download(
        name, day, BYBIT_BOOK.url(day), archive, opener=opener, timeout=timeout
    )
    if got is None:
        assert missed is not None
        return missed
    try:
        try:
            fetch.check_zip(archive)
        except fetch.UNREADABLE_ARCHIVE as exc:
            return fetch.failed(name, day, fl.CORRUPT, exc, got)
        try:
            stats: dict[str, object] = asdict(message_stats(archive))
            frame = replay(archive, depth=depth, grid_ms=grid_ms)
        except (bybit.BybitArchiveError, ValueError, KeyError, TypeError) as exc:
            return fetch.failed(name, day, fl.REPLAY_FAILED, exc, got)
        fetch.write_parquet(frame, day_path(BYBIT_BOOK, day, root))
        stats.update(
            rows=len(frame),
            sequence_gaps=int(frame.attrs.get("sequence_gaps", 0)),
            deep_levels_missing_rows=int(frame.attrs.get("deep_levels_missing_rows", 0)),
            clip_beyond_levels_share=beyond_levels_share(frame, depth=depth),
            peak_rss_bytes=peak_rss_bytes(),
        )
        return fetch.done(name, day, got, stats)
    finally:
        archive.unlink(missing_ok=True)
