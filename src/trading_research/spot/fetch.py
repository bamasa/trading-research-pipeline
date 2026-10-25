"""What every spot fetcher shares: the access it asks, the download it classifies, the file it writes.

On the 2025 study's pattern (:mod:`trading_research.data.study_fetch`), whose
failure rules the spot registration adopts:

- **only an HTTP 404 is a missing plane** (:func:`trading_research.data.failures.is_missing_plane`);
  any other failure of the transfer is ``transient`` and is retried with
  backoff by :func:`trading_research.data.fetch_stage.run_plane`, until the day
  is on disk or a terminal rule of :mod:`trading_research.data.fetch_log` ends it;
- **an archive that arrives and does not decompress** is ``corrupt`` (unreadable
  once the same bytes fail twice); one that decompresses and whose parser or
  replay raises is ``replay_failed`` (unreadable at once);
- **every archive is deleted after its replay or conversion**, whatever
  happened, so only converted days stay on disk.

A per-day fetcher returns a :class:`~trading_research.data.study_fetch.DayOutcome`
and never raises for a failed transfer, a missing day or a bad archive;
anything else (a full disk, a bug, a refused day) propagates and stops the
stage. Before any request it asks its access (:class:`AccessGuard`), so a day
the access refuses costs nothing and reveals nothing.
"""

from __future__ import annotations

import traceback
import urllib.request
import zipfile
import zlib
from collections.abc import Iterable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Protocol

import pandas as pd

from trading_research.data import fetch_log as fl
from trading_research.data.failures import Download, Opener, download_to, is_missing_plane
from trading_research.data.study_fetch import DayOutcome

#: Exceptions that say an archive which arrived does not decompress.
UNREADABLE_ARCHIVE: tuple[type[BaseException], ...] = (
    zipfile.BadZipFile,
    zlib.error,
    EOFError,
    OSError,
)


class AccessGuard(Protocol):
    """What the fetchers and loaders ask before touching a day.

    ``instrument`` is venue-qualified (``bybit-spot:BTCUSDT``), ``plane`` is
    ``book`` or ``prints``. It raises (the study's access raises
    :class:`~trading_research.market_making.prereg.HeldOutLocked`) unless every
    day is permitted.
    """

    def require(
        self, instrument: str, days: Iterable[date], plane: str, *, what: str = "read"
    ) -> None: ...


def now() -> datetime:
    return datetime.now(UTC)


def describe(exc: BaseException) -> str:
    return "".join(traceback.format_exception_only(type(exc), exc)).strip()[:300]


def download(
    plane: str,
    day: date,
    url: str,
    target: Path,
    *,
    opener: Opener | None,
    timeout: float,
) -> tuple[Download | None, DayOutcome | None]:
    """Download ``url``, or the outcome that replaces it: a 404 or a failed transfer."""
    try:
        got = download_to(url, target, timeout=timeout, opener=opener or urllib.request.urlopen)
    except Exception as exc:  # classified, never swallowed: only a 404 is missing
        outcome = fl.MISSING if is_missing_plane(exc) else fl.TRANSIENT
        return None, DayOutcome(plane, day, outcome, now(), detail=describe(exc))
    if not got.complete:  # pragma: no cover - download_to raises first
        return None, DayOutcome(plane, day, fl.TRANSIENT, now(), detail="short body")
    return got, None


def failed(plane: str, day: date, outcome: str, exc: BaseException, got: Download) -> DayOutcome:
    """An archive that arrived and then failed, with the bytes it was."""
    return DayOutcome(plane, day, outcome, now(), describe(exc), got.sha256, got.size)


def done(
    plane: str, day: date, got: Download, stats: dict[str, object], detail: str = ""
) -> DayOutcome:
    return DayOutcome(plane, day, fl.OK, now(), detail, got.sha256, got.size, dict(stats))


def write_parquet(frame: pd.DataFrame, target: Path) -> None:
    """Written under a temporary name and renamed, so a file on disk is whole."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"{target.name}.part")
    frame.to_parquet(partial, compression="zstd", index=False)
    partial.replace(target)


def check_zip(path: Path) -> None:
    """Raise unless the zip opens and every member passes its CRC."""
    with zipfile.ZipFile(path) as packed:
        bad = packed.testzip()
    if bad is not None:
        raise zipfile.BadZipFile(f"member {bad} fails its checksum")
