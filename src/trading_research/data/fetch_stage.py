"""Running one plane of a stage's fetch to completion, and the memory that caps its workers.

A stage is complete only when every day is on disk or ended by a terminal rule
(:mod:`trading_research.data.fetch_log`). :func:`run_plane` therefore loops:
it runs every day that is due, records each outcome in the log as it arrives,
and sleeps until the next retry is due, for as long as any day is pending. A
restarted run reads the same log and carries on.

**Replay workers.** :func:`trading_research.data.bybit.reconstruct` holds a
day's rows in memory until the day ends, so a replay worker's peak grows with
the day's row count, not its archive's size. The book plane runs on at most
:data:`REPLAY_WORKERS_MAX` processes, fewer when that many times the peak
resident memory of one replayed day would exceed
:data:`REPLAY_MEMORY_BUDGET_BYTES` (:func:`replay_workers`). The peak is
measured in a fresh process (:func:`replay_peak_rss`), so the figure is one
replay's and not whatever the measuring process held before.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from trading_research.data import fetch_log as fl
from trading_research.data.disk import StageBudget
from trading_research.data.failures import Opener
from trading_research.data.study_fetch import BOOK, DayOutcome, peak_rss_bytes

REPLAY_WORKERS_MAX = 4
REPLAY_MEMORY_BUDGET_BYTES = 24 * 10**9

#: A 404 on a day the existence check found present is reported as such.
FOUND_PRESENT_NOTE = "404 on a day the HEAD check found present"


def replay_workers(
    peak_rss: int,
    *,
    max_workers: int = REPLAY_WORKERS_MAX,
    budget_bytes: int = REPLAY_MEMORY_BUDGET_BYTES,
) -> int:
    """Book-replay workers: at most ``max_workers``, and fewer while that many
    peaks would exceed ``budget_bytes``; never fewer than one."""
    if peak_rss <= 0:
        raise ValueError(f"a peak of {peak_rss} bytes is not a measurement")
    return max(1, min(max_workers, budget_bytes // peak_rss))


def _peak_after(fn: Callable[..., Any], args: tuple[Any, ...]) -> int:
    fn(*args)
    return peak_rss_bytes()


def measure_peak_rss(fn: Callable[..., Any], *args: Any) -> int:
    """Run ``fn(*args)`` in a fresh spawned process and return its peak RSS in bytes.

    ``fn`` must be importable by name (a module-level function), since it
    crosses a process boundary.
    """
    import multiprocessing

    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=context) as pool:
        return int(pool.submit(_peak_after, fn, args).result())


def _replay(archive: Path, symbol: str, depth: int, grid_ms: int) -> None:
    from trading_research.data.bybit import reconstruct

    reconstruct(archive, symbol=symbol, depth=depth, grid_ms=grid_ms)


def replay_peak_rss(
    archive: Path | str, *, symbol: str, depth: int = 10, grid_ms: int = 100
) -> int:
    """Peak resident memory of replaying one archive, alone in a fresh process."""
    return measure_peak_rss(_replay, Path(archive), symbol, depth, grid_ms)


def _now() -> datetime:
    return datetime.now(UTC)


def _outcomes(
    days: Sequence[date], fetch_one: Callable[[date], DayOutcome], workers: int
) -> Iterator[DayOutcome]:
    if workers <= 1:
        for day in days:
            yield fetch_one(day)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch_one, day) for day in days]
        for future in as_completed(futures):
            yield future.result()


def _record(
    log: fl.FetchLog, outcome: DayOutcome, expected_present: frozenset[date]
) -> fl.DayRecord:
    record = log.get(outcome.plane, outcome.day)
    record.add(
        outcome.outcome,
        now=outcome.finished_at,
        detail=outcome.detail,
        sha256=outcome.sha256,
        size=outcome.size,
    )
    if outcome.outcome == fl.OK:
        record.stats = dict(outcome.stats)
    elif record.status == fl.MISSING_PLANE and outcome.day in expected_present:
        record.stats["note"] = FOUND_PRESENT_NOTE
    log.save()
    return record


def run_plane(
    plane: str,
    days: Sequence[date],
    fetch_one: Callable[[date], DayOutcome],
    log: fl.FetchLog,
    *,
    workers: int = 1,
    expected_present: Sequence[date] = (),
    clock: Callable[[], datetime] = _now,
    sleep: Callable[[float], None] = time.sleep,
    on_record: Callable[[fl.DayRecord], None] | None = None,
) -> fl.FetchLog:
    """Fetch ``days`` of one plane until each is on disk or terminally excluded.

    ``fetch_one`` must be picklable when ``workers`` exceeds one (a
    :func:`functools.partial` of a module-level fetcher is). Days already
    resolved in ``log`` are not fetched again; a pending day waits for its
    backoff. Returns only when :meth:`FetchLog.complete` holds for the days.
    """
    if plane == BOOK and workers > REPLAY_WORKERS_MAX:
        raise ValueError(f"the book replay runs on at most {REPLAY_WORKERS_MAX} workers")
    present = frozenset(expected_present)
    records = [log.get(plane, day) for day in days]
    log.save()
    while pending := [r for r in records if not r.resolved]:
        now = clock()
        due = [date.fromisoformat(r.day) for r in pending if r.due(now) <= now]
        if not due:
            wake = min(r.due(now) for r in pending)
            sleep(max((wake - now).total_seconds(), 0.0))
            continue
        for outcome in _outcomes(due, fetch_one, workers):
            record = _record(log, outcome, present)
            if on_record is not None:
                on_record(record)
    return log


def run_funding(
    days: Sequence[date],
    fetch_span: Callable[[list[date]], list[DayOutcome]],
    log: fl.FetchLog,
    *,
    clock: Callable[[], datetime] = _now,
    sleep: Callable[[float], None] = time.sleep,
) -> fl.FetchLog:
    """Funding for ``days``: the pending days fetched as one span until resolved.

    ``fetch_span`` is :func:`trading_research.data.study_fetch.fetch_funding_span`
    with its symbol and root bound. The days share each attempt, so they share
    the backoff and the terminal rules.
    """
    from trading_research.data.study_fetch import FUNDING

    records = [log.get(FUNDING, day) for day in days]
    log.save()
    while pending := [r for r in records if not r.resolved]:
        now = clock()
        wake = max(r.due(now) for r in pending)
        if wake > now:
            sleep((wake - now).total_seconds())
            continue
        for outcome in fetch_span([date.fromisoformat(r.day) for r in pending]):
            _record(log, outcome, frozenset())
    return log


@dataclass(frozen=True)
class StudyRoots:
    """Where each plane lands; the repository's usual layout by default."""

    book: Path
    trades: Path
    bars: Path
    funding: Path
    #: Pass-through directory: an archive lives here only until its replay.
    archives: Path

    @classmethod
    def under(cls, data_root: Path | str) -> StudyRoots:
        base = Path(data_root)
        return cls(
            book=base / "book",
            trades=base / "trades",
            bars=base / "bars",
            funding=base / "funding",
            archives=base / "_archives_in_flight",
        )


@dataclass(frozen=True)
class StagePlan:
    """The days a stage fetches, per plane (E's bars without its book, say)."""

    book: tuple[date, ...] = ()
    prints: tuple[date, ...] = ()
    bars: tuple[date, ...] = ()
    funding: tuple[date, ...] = ()


def head_check_present(path: Path | str) -> dict[str, frozenset[date]]:
    """Days the existence check found present, per archive (``ob500``, ``ob200``, ``prints``)."""
    import csv

    present: dict[str, set[date]] = {"ob500": set(), "ob200": set(), "prints": set()}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for archive, days in present.items():
                if row[f"{archive}_status"].strip() == "200":
                    days.add(date.fromisoformat(row["day"]))
    return {archive: frozenset(days) for archive, days in present.items()}


def files_on_disk(
    symbol: str, plan: StagePlan, roots: StudyRoots, *, interval: str = "1h"
) -> list[Path]:
    """Every file of ``plan`` already on disk: a block opened for fetching has none."""
    from trading_research.data import bybit_funding, bybit_trades
    from trading_research.data.study_fetch import book_path

    candidates = [book_path(symbol, d, roots.book) for d in plan.book]
    candidates += [bybit_trades.day_path(symbol, d, roots.trades) for d in plan.prints]
    candidates += [roots.bars / symbol / f"klines-{interval}" / f"{d}.parquet" for d in plan.bars]
    candidates += [bybit_funding.day_path(symbol, d, roots.funding) for d in plan.funding]
    return [p for p in candidates if p.exists()]


def run_stage(
    symbol: str,
    plan: StagePlan,
    roots: StudyRoots,
    log: fl.FetchLog,
    *,
    budget: StageBudget,
    book_workers: int,
    print_workers: int = 8,
    expected_present: dict[str, frozenset[date]] | None = None,
    opener: Opener | None = None,
    clock: Callable[[], datetime] = _now,
    sleep: Callable[[float], None] = time.sleep,
) -> fl.FetchLog:
    """Check the disk, clear what an interrupted run left, then fetch every plane.

    ``budget`` is the stage's :class:`trading_research.data.disk.StageBudget`;
    the stage refuses to start unless free disk exceeds it by the margin.
    Books run on ``book_workers`` (at most four; :func:`replay_workers` says how
    many), prints on ``print_workers``, bars one day at a time and funding as
    one span. ``opener`` serves the book and print archives (a
    :class:`trading_research.data.failures.DirectoryOpener` replays archives
    already on disk). Returns when every planned day is on disk or terminally
    excluded.
    """
    import functools

    from trading_research.data import disk
    from trading_research.data import study_fetch as sf

    disk.check_disk(roots.book, budget)
    disk.delete_archives(roots.archives)
    present = expected_present or {}
    book_present = [d for d in plan.book if d in present.get(_archive_of(d), frozenset())]
    loop: dict[str, Any] = {"clock": clock, "sleep": sleep}
    archives: dict[str, Any] = {"archive_dir": roots.archives, "opener": opener}
    book = functools.partial(sf.fetch_book_day, symbol, root=roots.book, **archives)
    run_plane(
        sf.BOOK, plan.book, book, log, workers=book_workers, expected_present=book_present, **loop
    )
    prints = functools.partial(sf.fetch_prints_day, symbol, root=roots.trades, **archives)
    run_plane(
        sf.PRINTS,
        plan.prints,
        prints,
        log,
        workers=print_workers,
        expected_present=[d for d in plan.prints if d in present.get("prints", frozenset())],
        **loop,
    )
    bars = functools.partial(sf.fetch_bars_day, symbol, root=roots.bars)
    run_plane(sf.BARS, plan.bars, bars, log, workers=1, **loop)
    funding = functools.partial(sf.fetch_funding_span, symbol, root=roots.funding)
    run_funding(plan.funding, funding, log, **loop)
    return log


def _archive_of(day: date) -> str:
    from trading_research.data.bybit import archive_kind

    return archive_kind(day)
