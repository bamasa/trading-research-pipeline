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
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from trading_research.data import fetch_log as fl
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
