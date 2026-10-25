"""The spot study's fetch of a stage: the disk check, then every plane until each day is resolved.

On the pattern of :mod:`trading_research.data.ensure` (only what the log has
not resolved is fetched, and nothing is fetched silently: :func:`pending` says
what a stage would fetch before it does), under the 2025 study's failure rules,
which the spot registration adopts (§Data budget):

- **every day of the stage is asked for**, every day the head check listed as
  present among them; a 404 on a listed day is a missing plane, noted as such;
- **only an HTTP 404 is a missing plane**; any other failure is retried with
  backoff (:func:`trading_research.data.fetch_stage.run_plane`) until the day is
  on disk or a terminal rule of :mod:`trading_research.data.fetch_log` ends it,
  and the log, rewritten after every attempt, carries a restart on;
- **archives are deleted** after their replay or conversion (each fetcher does
  it), and what an interrupted run left in flight is deleted before a stage;
- **the disk check**: before each stage, stage 0 included, free disk must
  exceed the stage's budget by the registered 10 GB, or the stage waits,
  measuring again every ``poll_s``; every measurement is appended to a record
  (:func:`wait_for_disk`).

**The budget.** The registration gives the study's disk peak as 12 to 19 GB and
"about 29 GB" of free disk for stage 1. Every stage's budget here is that
peak's upper end, :data:`STAGE_BUDGET_GB`, so each stage starts only above
29 GB free: files of earlier stages stay on disk, so the peak is cumulative.

Book days replay on at most :data:`BOOK_WORKERS_MAX` processes.
"""

from __future__ import annotations

import functools
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from trading_research.data import disk
from trading_research.data import fetch_log as fl
from trading_research.data.failures import Opener
from trading_research.data.fetch_stage import run_plane
from trading_research.data.study_fetch import DayOutcome
from trading_research.spot import binance, bybit_book, bybit_prints
from trading_research.spot.archives import PLANES, day_path
from trading_research.spot.fetch import AccessGuard

#: Free disk that must remain above a stage's budget (``fetch.disk_margin_gb``).
DISK_MARGIN_GB = 10.0
#: Every stage's budget: the registered peak's upper end (``fetch.disk_peak_gb``).
STAGE_BUDGET_GB = 19.0
#: ``fetch.replay_workers_max``.
BOOK_WORKERS_MAX = 4
#: The planes in the order a stage fetches them.
ORDER: tuple[str, ...] = ("bybit_book", "bybit_prints", "binance_prints")
FETCHERS: dict[str, Callable[..., DayOutcome]] = {
    "bybit_book": bybit_book.fetch_day,
    "bybit_prints": bybit_prints.fetch_day,
    "binance_prints": binance.fetch_day,
}


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class DiskCheck:
    """One measurement of free disk at a stage's start."""

    stage: int
    at: str  # UTC, ISO 8601
    free_gb: float
    budget_gb: float
    margin_gb: float

    @property
    def passed(self) -> bool:
        return self.free_gb > self.budget_gb + self.margin_gb


def record_disk(check: DiskCheck, path: Path | str) -> None:
    """Append ``check`` to the JSON list at ``path``, atomically."""
    target = Path(path)
    kept = json.loads(target.read_text(encoding="utf-8")) if target.exists() else []
    kept.append({**asdict(check), "passed": check.passed})
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"{target.name}.part")
    partial.write_text(json.dumps(kept, indent=1) + "\n", encoding="utf-8")
    partial.replace(target)


def wait_for_disk(
    stage: int,
    data_root: Path | str,
    record: Path | str,
    *,
    budget_gb: float = STAGE_BUDGET_GB,
    margin_gb: float = DISK_MARGIN_GB,
    measure: Callable[[Path | str], float] = disk.free_gb,
    clock: Callable[[], datetime] = _now,
    sleep: Callable[[float], None] = time.sleep,
    poll_s: float = 600.0,
    max_checks: int | None = None,
) -> DiskCheck:
    """Measure free disk until it exceeds the budget by the margin; record every measurement.

    With ``max_checks``, raise :class:`trading_research.data.disk.InsufficientDisk`
    after that many failed measurements instead of waiting on.
    """
    checks = 0
    while True:
        check = DiskCheck(stage, clock().isoformat(), measure(data_root), budget_gb, margin_gb)
        record_disk(check, record)
        checks += 1
        if check.passed:
            return check
        if max_checks is not None and checks >= max_checks:
            raise disk.InsufficientDisk(
                f"{check.free_gb:.1f} GB free; stage {stage} needs more than "
                f"{budget_gb + margin_gb:.1f} GB ({budget_gb:.1f} GB budget + {margin_gb:.0f} GB margin)"
            )
        sleep(poll_s)


@dataclass(frozen=True)
class SpotRoots:
    """Where converted days land, and the pass-through directory of archives in flight."""

    data: Path
    archives: Path

    @classmethod
    def under(cls, data_root: Path | str) -> SpotRoots:
        base = Path(data_root)
        return cls(data=base, archives=base / "_archives_in_flight")


def _check_planes(days: Mapping[str, Sequence[date]]) -> None:
    unknown = sorted(set(days) - set(PLANES))
    if unknown:
        raise ValueError(f"unknown planes {unknown}; the planes are {sorted(PLANES)}")


def pending(days: Mapping[str, Sequence[date]], log: fl.FetchLog) -> dict[str, list[date]]:
    """The days of each plane a stage would still ask for: those its log has not resolved."""
    _check_planes(days)

    def resolved(name: str, day: date) -> bool:
        record = log.records.get(log.key(name, day))
        return record is not None and record.resolved

    return {name: [d for d in wanted if not resolved(name, d)] for name, wanted in days.items()}


def files_on_disk(days: Mapping[str, Sequence[date]], data_root: Path | str) -> list[Path]:
    """Every converted file of ``days`` already on disk: a block opened for fetching has none."""
    _check_planes(days)
    paths = [day_path(PLANES[name], d, data_root) for name, wanted in days.items() for d in wanted]
    return [p for p in paths if p.exists()]


def run_stage(
    stage: int,
    days: Mapping[str, Sequence[date]],
    roots: SpotRoots,
    log: fl.FetchLog,
    *,
    access: AccessGuard,
    disk_record: Path | str,
    expected_present: Mapping[str, frozenset[date]] | None = None,
    opener: Opener | None = None,
    book_workers: int = 1,
    print_workers: int = 1,
    timeout: float = 120.0,
    clock: Callable[[], datetime] = _now,
    sleep: Callable[[float], None] = time.sleep,
    measure: Callable[[Path | str], float] = disk.free_gb,
    poll_s: float = 600.0,
    max_disk_checks: int | None = None,
) -> fl.FetchLog:
    """Fetch ``days`` (plane name to days) until every day is on disk or terminally excluded.

    The whole plan is checked against ``access`` before the disk is measured;
    each fetcher asks again before its request. ``expected_present`` is the
    head check's answer (:func:`trading_research.spot.archives.listed_present`).
    ``opener`` serves every archive (the synthetic venue in a dry run, which
    fetches with one worker, in-process).
    """
    _check_planes(days)
    if not 1 <= book_workers <= BOOK_WORKERS_MAX or print_workers < 1:
        raise ValueError(f"book workers run 1 to {BOOK_WORKERS_MAX}, print workers at least 1")
    for name, wanted in days.items():
        plane = PLANES[name]
        access.require(plane.instrument, wanted, plane.kind, what="fetch")
    wait_for_disk(stage, roots.data, disk_record, measure=measure, clock=clock, sleep=sleep,
                  poll_s=poll_s, max_checks=max_disk_checks)  # fmt: skip
    disk.delete_archives(roots.archives)
    present = expected_present or {}
    for name in ORDER:
        if name not in days:
            continue
        fetch_one = functools.partial(
            FETCHERS[name],
            root=roots.data,
            archive_dir=roots.archives,
            access=access,
            opener=opener,
            timeout=timeout,
        )
        listed = present.get(name, frozenset())
        run_plane(
            name,
            days[name],
            fetch_one,
            log,
            workers=book_workers if name == "bybit_book" else print_workers,
            expected_present=[d for d in days[name] if d in listed],
            clock=clock,
            sleep=sleep,
        )
    return log
