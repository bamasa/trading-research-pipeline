"""Data that fetches itself when it is missing.

Nothing under ``data/`` is in version control — order books run to hundreds of
megabytes a day and belong nowhere near a repository. That leaves a gap that
research code usually fills badly: a script assumes the files are there, fails
with a path error when they are not, and the reader is left to reconstruct which
of a dozen download commands they were supposed to have run, in what order, for
which dates.

This module closes it. Every entry point states what it needs, this checks what
is already on disk, and only the missing days are fetched. Running an experiment
on a clean checkout therefore works — slowly the first time, instantly
afterwards.

Three properties worth stating, because each is a decision rather than an
accident.

**Only the gaps are fetched.** A day already present is never re-downloaded, so
an interrupted run resumes rather than restarts, and re-running an experiment
costs nothing.

**A missing day is not an error.** An instrument listed halfway through the span
has no archive before it existed, and a venue that has not published yesterday
yet has no archive either. Those are reported and skipped; only a total absence
of data raises.

**Nothing is fetched silently.** A function that quietly downloads twenty
gigabytes because a default said so is worse than one that fails. Every call
reports what it will fetch before it starts, and ``dry_run`` answers the
question without acting on it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

#: Where each kind of data lives, relative to the repository root.
BOOK_ROOT = Path("data/book")
UNIVERSE_ROOT = Path("data/universe")
TRADES_ROOT = Path("data/trades")


class DataUnavailable(RuntimeError):
    """Nothing could be fetched, so the caller cannot proceed."""


@dataclass
class FetchPlan:
    """What a request needs, and how much of it is already here."""

    symbol: str
    present: list[date] = field(default_factory=list)
    missing: list[date] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.missing

    def describe(self) -> str:
        return f"{self.symbol}: {len(self.present)} day(s) present, {len(self.missing)} to fetch"


def _days(start: date, end: date) -> list[date]:
    if end < start:
        raise ValueError(f"end {end} is before start {start}")
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def plan(symbol: str, start: date, end: date, root: Path) -> FetchPlan:
    """Which days of ``symbol`` are on disk and which are not."""
    directory = root / symbol
    out = FetchPlan(symbol=symbol)
    for day in _days(start, end):
        path = directory / f"{day.isoformat()}.parquet"
        if path.exists() and path.stat().st_size > 0:
            out.present.append(day)
        else:
            out.missing.append(day)
    return out


def _fetch_days(
    symbol: str,
    days: Sequence[date],
    fetch: Callable[[str, date], None],
    *,
    on_progress: Callable[[str], None] | None = None,
) -> tuple[int, list[date]]:
    """Fetch each day, collecting the ones the venue does not have."""
    fetched, unavailable = 0, []
    for day in days:
        try:
            fetch(symbol, day)
        except Exception as exc:  # a missing day must not end the run
            unavailable.append(day)
            if on_progress:
                on_progress(f"  {symbol} {day}: unavailable ({type(exc).__name__})")
            continue
        fetched += 1
        if on_progress and fetched % 10 == 0:
            on_progress(f"  {symbol}: {fetched}/{len(days)} fetched")
    return fetched, unavailable


def ensure_book(
    symbol: str,
    start: date,
    end: date,
    *,
    root: Path = BOOK_ROOT,
    depth: int = 10,
    grid_ms: int = 100,
    dry_run: bool = False,
    on_progress: Callable[[str], None] | None = print,
) -> FetchPlan:
    """Multi-level order books for one instrument, fetching what is missing.

    ``depth`` is levels a side. Ten is what the depth features need and costs
    about two minutes of replay a day; one is enough for a screen and costs
    eleven seconds, so a caller wanting breadth should ask for one.
    """
    from trading_research.data.bybit import download_range

    wanted = plan(symbol, start, end, root)
    if wanted.complete:
        return wanted
    if on_progress:
        on_progress(wanted.describe())
    if dry_run:
        return wanted

    def fetch(name: str, day: date) -> None:
        download_range(name, day, day, root / name, depth=depth, grid_ms=grid_ms, workers=1)

    fetched, unavailable = _fetch_days(symbol, wanted.missing, fetch, on_progress=on_progress)
    if fetched == 0 and not wanted.present:
        raise DataUnavailable(
            f"no order-book data for {symbol} between {start} and {end}; "
            f"{len(unavailable)} day(s) unavailable from the archive"
        )
    return plan(symbol, start, end, root)


def ensure_trades(
    symbol: str,
    start: date,
    end: date,
    *,
    root: Path = TRADES_ROOT,
    dry_run: bool = False,
    on_progress: Callable[[str], None] | None = print,
) -> FetchPlan:
    """Public trade prints, which are what a maker model needs and a book cannot say."""
    from trading_research.data.bybit_trades import download_day

    wanted = plan(symbol, start, end, root)
    if wanted.complete:
        return wanted
    if on_progress:
        on_progress(wanted.describe())
    if dry_run:
        return wanted

    def fetch(name: str, day: date) -> None:
        download_day(name, day, root)

    fetched, unavailable = _fetch_days(symbol, wanted.missing, fetch, on_progress=on_progress)
    if fetched == 0 and not wanted.present:
        raise DataUnavailable(
            f"no trade prints for {symbol} between {start} and {end}; "
            f"{len(unavailable)} day(s) unavailable"
        )
    return plan(symbol, start, end, root)


def ensure_universe(
    symbols: Sequence[str],
    start: date,
    end: date,
    *,
    root: Path = UNIVERSE_ROOT,
    workers: int = 6,
    dry_run: bool = False,
    on_progress: Callable[[str], None] | None = print,
) -> dict[str, FetchPlan]:
    """Top of book for many instruments over one span.

    Parallel across instrument-days, since each archive is independent and the
    work is network-bound. Instruments whose archives are entirely absent are
    reported and dropped rather than raising: a cross-sectional study wants the
    instruments that exist, not a failure because one did not.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from trading_research.data.bybit import download_range

    plans = {s: plan(s, start, end, root) for s in symbols}
    jobs = [(s, day) for s, p in plans.items() for day in p.missing]
    if on_progress:
        have = sum(len(p.present) for p in plans.values())
        on_progress(
            f"universe: {len(symbols)} instruments, {have} instrument-day(s) present, "
            f"{len(jobs)} to fetch"
        )
    if dry_run or not jobs:
        return plans

    def one(job: tuple[str, date]) -> tuple[str, date, bool]:
        name, day = job
        try:
            download_range(name, day, day, root / name, depth=1, grid_ms=1000, workers=1)
        except Exception:
            return name, day, False
        return name, day, True

    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(one, job) for job in jobs]):
            future.result()
            done += 1
            if on_progress and done % 25 == 0:
                on_progress(f"  {done}/{len(jobs)} instrument-day(s)")

    refreshed = {s: plan(s, start, end, root) for s in symbols}
    usable = {s: p for s, p in refreshed.items() if p.present}
    if not usable:
        raise DataUnavailable(
            f"none of the {len(symbols)} instruments had data between {start} and {end}"
        )
    if on_progress and len(usable) < len(symbols):
        dropped = sorted(set(symbols) - set(usable))
        on_progress(f"  no data at all for: {', '.join(dropped)}")
    return usable
