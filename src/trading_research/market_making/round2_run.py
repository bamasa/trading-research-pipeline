"""The second round's readers, each behind a :class:`.round2.Access`.

Everything here that opens a file of an instrument-day first asks the access
given whether that instrument may be read on that day, before anything is
opened:

* :func:`screen` computes round one's admission statistics
  (:mod:`.screen`, unchanged) on D, and H's coverage from the file system
  alone, without opening a file of H;
* :func:`markouts` scores every print of each day from its resting side at
  the gate's horizon (:func:`.gate.print_markouts`) and keeps the result, so
  the gate's state tapes, its hour masks and its persistence check need no
  second read of the book;
* :func:`run_plans` simulates :class:`~.heldout.Spec` lists on instrument-days
  through round one's job runner (:func:`.heldout.run_jobs`), one loaded day
  per job, each spec with its own tapes; every job writes its rows as it
  finishes under a directory keyed by everything the rows depend on, so a run
  that stops resumes without simulating a finished day again.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from trading_research.market_making import gate, heldout
from trading_research.market_making import screen as screen_module
from trading_research.market_making.events import DayUnavailable, day_start_ns, load_day
from trading_research.market_making.round2 import Access, block_days
from trading_research.market_making.simulator import source_fingerprint


def _require_access(access: object) -> Access:
    if not isinstance(access, Access):
        raise TypeError("the second round reads only through a round2.Access")
    return access


def screen(
    symbols: Sequence[str],
    *,
    access: object,
    book_roots: Sequence[Path],
    trades_root: Path,
    workers: int = 1,
    on_progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """The admission statistics of every instrument on D, one row each, with
    the coverage of D and of H (H from the file system only)."""
    permit = _require_access(access)
    d_days, h_days = block_days("D"), block_days("H")
    for symbol in symbols:
        permit.require(symbol, d_days, what="screen")
    roots = tuple(Path(r) for r in book_roots)
    jobs = [(s, day, roots, Path(trades_root)) for s in symbols for day in d_days]
    if workers <= 1:
        results = [screen_module._one_day(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(screen_module._one_day, jobs))
    rows = []
    for symbol in symbols:
        mine = [r for r, job in zip(results, jobs, strict=True) if job[0] == symbol]
        loaded = [r for r in mine if isinstance(r, screen_module.DayStatistics)]
        row: dict[str, Any] = {"symbol": symbol}
        row.update(screen_module.combine(loaded) if loaded else {})
        row["days_unavailable"] = sum(isinstance(r, str) for r in mine)
        row["coverage_D"] = screen_module.coverage(
            symbol, d_days, book_roots=roots, trades_root=trades_root
        )
        row["coverage_H"] = screen_module.coverage(
            symbol, h_days, book_roots=roots, trades_root=trades_root
        )
        rows.append(row)
        if on_progress:
            on_progress(f"  screened {symbol}: {len(loaded)} days")
    return pd.DataFrame(rows)


def _markout_day(job: tuple[str, date, tuple[Path, ...], Path, Path]) -> str | None:
    """Score one instrument-day's prints and write them; the reason if the day
    cannot be loaded. Module level so a worker process can receive it."""
    symbol, day, book_roots, trades_root, target = job
    if target.exists():
        return None
    try:
        events = load_day(
            symbol, day, book_roots=book_roots, trades_root=trades_root, funding_root=None
        )
    except DayUnavailable as exc:
        return str(exc)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".tmp")
    gate.print_markouts(events).frame().to_parquet(partial, index=False)
    partial.replace(target)
    return None


def markouts(
    symbol: str,
    days: Sequence[date],
    *,
    access: object,
    book_roots: Sequence[Path],
    trades_root: Path,
    out: Path,
    workers: int = 1,
) -> tuple[dict[date, gate.PrintMarkouts], dict[date, str]]:
    """Every print of each day, scored from its resting side, by day; and the
    days that could not be loaded, with the reason. Kept under ``out`` in a
    folder named for the source that computed them."""
    permit = _require_access(access)
    days = sorted(set(days))
    permit.require(symbol, days, what="score the prints of")
    folder = Path(out) / source_fingerprint()[:12] / symbol
    roots = tuple(Path(r) for r in book_roots)
    jobs = [
        (symbol, d, roots, Path(trades_root), folder / f"{d.isoformat()}.parquet") for d in days
    ]
    if workers <= 1:
        reasons = [_markout_day(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            reasons = list(pool.map(_markout_day, jobs))
    loaded, missing = {}, {}
    for (_, day, _, _, target), reason in zip(jobs, reasons, strict=True):
        if reason is not None:
            missing[day] = reason
            continue
        frame = pd.read_parquet(target)
        loaded[day] = gate.PrintMarkouts.from_frame(frame, day_start_ns(day))
    return loaded, missing


def spec_digest(symbol: str, days: Sequence[date], specs: Sequence[heldout.Spec]) -> str:
    """A hash of everything a plan's rows depend on: the instrument, the days,
    each spec (label, quoter, settings, tapes, flags) and the source."""
    described = []
    for spec in specs:
        tapes = {name: tape.fingerprint() for name, tape in sorted((spec.tapes or {}).items())}
        described.append(
            {
                "label": spec.label,
                "quoter": repr(spec.make_quoter()),
                "config": asdict(spec.config),
                "group": spec.group,
                "tapes": tapes,
                "built": [list(b) for b in spec.built],
                "oracle": spec.oracle,
                "keep": spec.keep,
            }
        )
    payload = {
        "symbol": symbol,
        "days": [d.isoformat() for d in sorted(set(days))],
        "specs": described,
        "source": source_fingerprint(),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()[:16]


def run_plans(
    plans: Sequence[tuple[str, Sequence[date], Sequence[heldout.Spec], int]],
    *,
    access: object,
    book_roots: Sequence[Path],
    trades_root: Path,
    funding_root: Path | None,
    out_root: Path,
    workers: int = 1,
    market_horizons_s: tuple[float, ...] = (),
    on_progress: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, dict[str, Path]]:
    """Simulate each plan, ``(symbol, days, specs, specs per job)``, in one pool.

    Every instrument's days are checked against ``access`` before any job
    starts. Jobs run in the order of the plans, so a caller puts the busiest
    instrument first. Returns every row, and each instrument's output folder
    (where kept fills, orders and equity are).
    """
    permit = _require_access(access)
    for symbol, days, _, _ in plans:
        permit.require(symbol, days, what="simulate")
    jobs: list[heldout.Job] = []
    folders: dict[str, Path] = {}
    for symbol, days, specs, per_job in plans:
        if not specs:
            continue
        folder = Path(out_root) / f"{symbol}_{spec_digest(symbol, days, specs)}"
        folders[symbol] = folder
        jobs += heldout.build_jobs(
            symbol,
            days,
            specs,
            book_roots=book_roots,
            trades_root=trades_root,
            funding_root=funding_root,
            out=folder,
            specs_per_job=per_job,
            market_horizons_s=market_horizons_s,
        )
    if on_progress:
        on_progress(f"  {len(jobs)} jobs on {workers} workers")
    rows = heldout.run_jobs(jobs, workers=workers, on_progress=on_progress, every=50)
    return rows, folders
