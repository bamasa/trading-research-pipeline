"""Disk: the check before each stage, and the deletions that keep the peak bounded.

A stage that runs out of disk half way leaves partial files, archives that were
never replayed and a log that says the day is pending; worse, a feature store
that fills the disk can fail on its last day after hours of work. So each stage
first checks that free space exceeds its budget (archives in flight, replayed
books, print files, features) by a fixed margin, and refuses to start otherwise
(:func:`check_disk`).

The default per-item sizes are the registration's: a replayed ten-level day at
most 60 MB, print files at most a quarter of their archives, the largest book
archive of the development period 0.92 GB, one archive in flight per worker.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

GB = 10**9

#: Free space that must remain above a stage's budget.
MARGIN_GB = 10.0

#: Registered sizes, used as defaults.
LARGEST_BOOK_ARCHIVE_GB = 0.92
BOOK_DAY_GB_MAX = 0.060
PRINT_PARQUET_SHARE_MAX = 0.25


class InsufficientDisk(RuntimeError):
    """Free disk does not exceed the stage's budget by the margin."""


@dataclass(frozen=True)
class StageBudget:
    """What a stage will put on disk at its peak, in GB."""

    archives_in_flight_gb: float
    books_gb: float
    prints_gb: float
    features_gb: float = 0.0

    @property
    def total_gb(self) -> float:
        return self.archives_in_flight_gb + self.books_gb + self.prints_gb + self.features_gb


def stage_budget(
    *,
    book_days: int,
    print_archives_gb: float,
    workers: int,
    features_gb: float = 0.0,
    largest_archive_gb: float = LARGEST_BOOK_ARCHIVE_GB,
    book_day_gb: float = BOOK_DAY_GB_MAX,
    print_share: float = PRINT_PARQUET_SHARE_MAX,
) -> StageBudget:
    """A stage's budget from its day count, its print archives and its workers."""
    if workers < 1 or book_days < 0 or print_archives_gb < 0 or features_gb < 0:
        raise ValueError("a budget needs at least one worker and no negative size")
    return StageBudget(
        archives_in_flight_gb=workers * largest_archive_gb,
        books_gb=book_days * book_day_gb,
        prints_gb=print_archives_gb * print_share,
        features_gb=features_gb,
    )


def free_gb(path: Path | str) -> float:
    """Free space on the volume holding ``path`` (or its nearest existing parent)."""
    probe = Path(path).resolve()
    while not probe.exists():
        probe = probe.parent
    return shutil.disk_usage(probe).free / GB


def check_disk(
    path: Path | str,
    budget: StageBudget,
    *,
    margin_gb: float = MARGIN_GB,
    free: float | None = None,
) -> float:
    """Refuse the stage unless free disk exceeds its budget by ``margin_gb``.

    Returns the free space found, for the record. ``free`` overrides the
    measurement (for tests and for a dry run's report).
    """
    available = free_gb(path) if free is None else free
    needed = budget.total_gb + margin_gb
    if available <= needed:
        raise InsufficientDisk(
            f"{available:.1f} GB free under {path}; the stage needs more than {needed:.1f} GB "
            f"({budget.total_gb:.1f} GB budget + {margin_gb:.0f} GB margin)"
        )
    return available


def _remove(paths: Iterable[Path]) -> int:
    freed = 0
    for path in paths:
        if path.is_file():
            freed += path.stat().st_size
            path.unlink()
    return freed


def delete_days(directory: Path | str, days: Iterable[date]) -> int:
    """Delete ``<directory>/<day>.parquet`` for each day; returns the bytes freed."""
    root = Path(directory)
    return _remove(root / f"{day.isoformat()}.parquet" for day in days)


def delete_partials(root: Path | str) -> int:
    """Delete files an interrupted run left half written (``*.part``) under ``root``."""
    base = Path(root)
    return _remove(list(base.rglob("*.part"))) if base.exists() else 0


def delete_archives(archive_dir: Path | str) -> int:
    """Delete archives left in the pass-through directory by an interrupted run.

    The fetch deletes each archive after its replay, so anything left here is
    from a run that stopped mid-day; the day is still pending in its log and is
    fetched again.
    """
    base = Path(archive_dir)
    if not base.exists():
        return 0
    kept = [p for p in base.iterdir() if p.is_file() and p.name.endswith((".zip", ".gz"))]
    return _remove(kept) + delete_partials(base)
