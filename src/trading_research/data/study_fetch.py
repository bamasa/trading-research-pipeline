"""The 2025 study's fetch: every day the venue has, on disk, or a reason it is not.

On the pattern of :mod:`trading_research.data.ensure` (only what is missing is
fetched, and no archive is kept after its replay), with two differences that
decide which days a study keeps:

- **a failed download is not an exclusion.** Only an HTTP 404 makes a missing
  plane (:mod:`trading_research.data.failures`); every other failure deletes the
  partial file and is retried with backoff until the day is on disk or one of
  the two terminal rules of :mod:`trading_research.data.fetch_log` ends it;
- **every attempt is recorded**, with the sha256 and UTC download time of the
  file used and what the day produced (rows, sequence gaps, the archive's
  message counts and median interval, the worker's peak memory).

The functions here fetch one plane of one day and return what happened; they
never raise for a failed transfer, a missing day or a bad archive. Anything
else (a full disk, a bug) propagates and stops the stage, whose log keeps every
attempt so far.
"""

from __future__ import annotations

import gzip
import hashlib
import resource
import sys
import traceback
import zipfile
import zlib
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from trading_research.data import bybit, bybit_trades
from trading_research.data import fetch_log as fl
from trading_research.data.failures import Download, Opener, download_to, is_missing_plane

BOOK = "book"
PRINTS = "prints"
BARS = "bars"
FUNDING = "funding"

#: Exceptions that say an archive which arrived does not decompress.
_UNREADABLE_ARCHIVE = (zipfile.BadZipFile, zlib.error, EOFError, OSError)


@dataclass(frozen=True)
class DayOutcome:
    """One attempt at one plane of one day, as a worker returns it."""

    plane: str
    day: date
    outcome: str
    finished_at: datetime
    detail: str = ""
    sha256: str | None = None
    size: int | None = None
    stats: dict[str, Any] = field(default_factory=dict)


def _now() -> datetime:
    return datetime.now(UTC)


def _describe(exc: BaseException) -> str:
    return "".join(traceback.format_exception_only(type(exc), exc)).strip()[:300]


def peak_rss_bytes() -> int:
    """This process's peak resident memory so far, in bytes.

    ``ru_maxrss`` is in bytes on macOS and in kilobytes on Linux.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _write_parquet(frame: pd.DataFrame, target: Path) -> None:
    """Written under a temporary name and renamed, so a file on disk is whole."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"{target.name}.part")
    frame.to_parquet(partial, compression="zstd", index=False)
    partial.replace(target)


@dataclass(frozen=True)
class Seen:
    """The file an attempt judged, for the record."""

    sha256: str
    size: int

    def fields(self) -> dict[str, Any]:
        return {"sha256": self.sha256, "size": self.size}


def _fetched(
    plane: str, day: date, url: str, archive: Path, opener: Opener, timeout: float
) -> tuple[Download | None, DayOutcome | None]:
    """Download, or the outcome that replaces it: a 404 or a failed transfer."""
    try:
        return download_to(url, archive, timeout=timeout, opener=opener), None
    except Exception as exc:  # classified, never swallowed: only a 404 is missing
        outcome = fl.MISSING if is_missing_plane(exc) else fl.TRANSIENT
        return None, DayOutcome(plane, day, outcome, _now(), detail=_describe(exc))


def book_path(symbol: str, day: date, root: Path | str) -> Path:
    """Where a replayed book day lives: the layout :mod:`ensure` plans over."""
    return Path(root) / symbol / f"{day.isoformat()}.parquet"


def fetch_book_day(
    symbol: str,
    day: date,
    root: Path | str,
    archive_dir: Path | str,
    *,
    depth: int = 10,
    grid_ms: int = 100,
    timeout: float = 120.0,
    opener: Opener | None = None,
) -> DayOutcome:
    """Fetch, check, replay and write one book day; delete its archive whatever happens.

    The archive name follows the day (:func:`bybit.archive_kind`). An archive
    that arrives and does not decompress is ``corrupt`` (retried, unreadable
    when the same bytes fail twice); one that decompresses and whose replay
    raises is ``replay_failed`` (unreadable at once). The replay is
    :func:`bybit.reconstruct`, unchanged. The stats carry the replayed rows and
    sequence gaps, the archive's message counts and median interval, and the
    worker's peak memory after the day.
    """
    import urllib.request

    kind = bybit.archive_kind(day)
    archive = Path(archive_dir) / f"{symbol}-{day.isoformat()}-{kind}.zip"
    url = bybit.archive_url(symbol, day, kind)
    download, failed = _fetched(BOOK, day, url, archive, opener or urllib.request.urlopen, timeout)
    if download is None:
        assert failed is not None
        return failed
    seen = Seen(download.sha256, download.size)
    try:
        try:
            with zipfile.ZipFile(archive) as packed:
                bad = packed.testzip()
            if bad is not None:
                raise zipfile.BadZipFile(f"member {bad} fails its checksum")
        except _UNREADABLE_ARCHIVE as exc:
            return DayOutcome(BOOK, day, fl.CORRUPT, _now(), _describe(exc), **seen.fields())
        try:
            stats: dict[str, Any] = asdict(bybit.archive_message_stats(archive))
            frame = bybit.reconstruct(
                archive, symbol=symbol, depth=depth, grid_ms=grid_ms, source=bybit.source_tag(kind)
            )
        except (bybit.BybitArchiveError, ValueError, KeyError, TypeError) as exc:
            return DayOutcome(BOOK, day, fl.REPLAY_FAILED, _now(), _describe(exc), **seen.fields())
        _write_parquet(frame, book_path(symbol, day, root))
        stats.update(
            archive=kind,
            rows=len(frame),
            sequence_gaps=int(frame.attrs.get("sequence_gaps", 0)),
            peak_rss_bytes=peak_rss_bytes(),
        )
        return DayOutcome(BOOK, day, fl.OK, _now(), stats=stats, **seen.fields())
    finally:
        archive.unlink(missing_ok=True)


def fetch_prints_day(
    symbol: str,
    day: date,
    root: Path | str,
    archive_dir: Path | str,
    *,
    timeout: float = 120.0,
    opener: Opener | None = None,
) -> DayOutcome:
    """Fetch, check, parse and write one day of prints; delete the archive after."""
    import urllib.request

    archive = Path(archive_dir) / f"{symbol}{day.isoformat()}.csv.gz"
    url = bybit_trades.archive_url(symbol, day)
    download, failed = _fetched(
        PRINTS, day, url, archive, opener or urllib.request.urlopen, timeout
    )
    if download is None:
        assert failed is not None
        return failed
    seen = Seen(download.sha256, download.size)
    try:
        try:
            raw = gzip.decompress(archive.read_bytes())
        except _UNREADABLE_ARCHIVE as exc:
            return DayOutcome(PRINTS, day, fl.CORRUPT, _now(), _describe(exc), **seen.fields())
        try:
            frame = bybit_trades.parse(raw)
        except (bybit_trades.BybitTradesError, ValueError, KeyError, TypeError) as exc:
            return DayOutcome(
                PRINTS, day, fl.REPLAY_FAILED, _now(), _describe(exc), **seen.fields()
            )
        del raw
        _write_parquet(frame, bybit_trades.day_path(symbol, day, root))
        return DayOutcome(PRINTS, day, fl.OK, _now(), stats={"rows": len(frame)}, **seen.fields())
    finally:
        archive.unlink(missing_ok=True)


def fetch_bars_day(
    symbol: str,
    day: date,
    root: Path | str,
    *,
    interval: str = "1h",
) -> DayOutcome:
    """One day of Binance USDT-M futures bars, from that day's own archive only.

    Through :func:`binance.download_range` with ``daily_only``, so no monthly
    archive is asked for, read or cached. Bars follow the books' rule: only a
    404 is a missing plane, and any other failure is retried.
    """
    from trading_research.data.binance import ArchiveSpec, download_range

    spec = ArchiveSpec(market="futures-um", kind="klines", symbol=symbol, interval=interval)
    try:
        result = download_range(spec, day, day, root, keep_raw=False, daily_only=True)[0]
    except Exception as exc:  # a parse of a verified archive that still fails
        return DayOutcome(BARS, day, fl.TRANSIENT, _now(), _describe(exc))
    if result.output is None:
        outcome = fl.MISSING if result.http_status == 404 else fl.TRANSIENT
        return DayOutcome(BARS, day, outcome, _now(), result.skipped or "")
    stats = {"rows": result.rows}
    if result.sha256 is None:  # converted on an earlier run: the file is the record
        return DayOutcome(BARS, day, fl.OK, _now(), "already converted", file_sha256(result.output))
    return DayOutcome(BARS, day, fl.OK, _now(), "", result.sha256, result.bytes_downloaded, stats)


def fetch_funding_span(
    symbol: str,
    days: list[date],
    root: Path | str,
    *,
    opener: Opener | None = None,
) -> list[DayOutcome]:
    """Funding for ``days``, fetched as one span from the first to the last.

    The venue serves funding from a REST endpoint, not as daily archives, so one
    request covers the span and each day gets the same outcome when it fails.
    A day the answered history has no settlement for is a missing plane, as is
    every day when the venue answers 404; any other failure is retried.
    """
    import urllib.request

    from trading_research.data import bybit_funding

    if not days:
        return []
    first, last = min(days), max(days)
    try:
        written = bybit_funding.download_range(
            symbol, first, last, root, opener=opener or urllib.request.urlopen
        )
    except Exception as exc:  # classified, never swallowed: only a 404 is missing
        outcome = fl.MISSING if is_missing_plane(exc) else fl.TRANSIENT
        return [DayOutcome(FUNDING, d, outcome, _now(), _describe(exc)) for d in days]
    paths = {date.fromisoformat(p.stem): p for p in written}
    out = []
    for day in days:
        path = paths.get(day)
        if path is None:
            out.append(DayOutcome(FUNDING, day, fl.MISSING, _now(), "no settlement in the history"))
            continue
        rows = len(pd.read_parquet(path))
        sha = file_sha256(path)
        out.append(
            DayOutcome(FUNDING, day, fl.OK, _now(), "", sha, path.stat().st_size, {"rows": rows})
        )
    return out
