"""The held-out runner: tapes built from the loaded day, fill placement, jobs.

The development runs went through :func:`.simulator.run_cells`, which keeps a
day's summary and nothing a verdict on the held-out block also needs: which
fills came from an order placed inside the spread (H1's K3), the orders
themselves (fill ratios), and the oracle rungs of the advantage ladder, which
:func:`~.simulator.run_cells` refuses by design. This module runs the same
simulator, one loaded instrument-day per job, for a list of :class:`Spec`:

* a spec can ask for a tape that only the loaded day can give —
  :func:`stale_spread_tape` (H1's stale-trigger placebo) or
  :func:`forecast_tape` (the ladder's oracle fair value);
* a spec marked ``oracle`` runs a quoter that reads the future, through
  :func:`~.simulator.simulate_day` with ``allow_oracle``, and its row is
  stamped as an upper bound;
* a spec marked ``keep`` writes its fills (each with its placement), orders
  and minute equity next to the summary, for the markouts, the inventory and
  the decomposition;
* every job writes its rows to disk as it finishes, so a run that stops can be
  resumed without simulating a finished day again.

Nothing here decides which days may be read: the caller passes days its
:class:`~.prereg.Access` permits, and :func:`run_jobs` checks them again.
"""

from __future__ import annotations

import gc
import json
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_research.market_making import analysis
from trading_research.market_making.events import NS_PER_S, DayEvents, DayUnavailable, load_day
from trading_research.market_making.quoters import FORECAST_SIGNAL, STALE_SPREAD_SIGNAL, Quoter
from trading_research.market_making.signals import SignalTape
from trading_research.market_making.simulator import (
    EXTRA_COUNTERS,
    DayResult,
    SimConfig,
    _peak_rss_mb,
    _summary,
    _unavailable_row,
    day_tapes,
    simulate_day,
)


def last_rows(events: DayEvents) -> tuple[np.ndarray, np.ndarray]:
    """Distinct snapshot times, and the row of the last snapshot at each."""
    ts = events.book_ts
    last = np.r_[ts[1:] != ts[:-1], True] if len(ts) else np.zeros(0, dtype=bool)
    rows = np.flatnonzero(last)
    return ts[rows], rows


def stale_spread_tape(events: DayEvents, lag_s: float = 60.0) -> SignalTape:
    """The touch spread in ticks, known ``lag_s`` after its snapshot.

    A decision at ``t`` reads it strictly before ``t``: the spread of the last
    snapshot strictly before ``t - lag_s``.
    """
    ts, rows = last_rows(events)
    spread = (events.ask_px[rows, 0] - events.bid_px[rows, 0]).astype(np.float64)
    return SignalTape(STALE_SPREAD_SIGNAL, ts + round(lag_s * NS_PER_S), spread)


def noise_seed(base: int, symbol: str, day: date) -> list[int]:
    """The ladder's noise seed for one instrument-day: the registered seed,
    the day and the instrument, so every rung sees the same draws."""
    return [int(base), day.toordinal(), zlib.crc32(symbol.encode())]


def forecast_tape(
    events: DayEvents,
    r2: float,
    *,
    seed: list[int] | int,
    horizon_s: float = 1.0,
) -> SignalTape:
    """An oracle forecast of the mid's change over the next ``horizon_s``.

    With ``d`` the realised change from each snapshot to the last snapshot at
    or before ``horizon_s`` later, and ``s2`` its mean square over the day, the
    forecast is ``r2 * (d + e)`` with ``e`` Gaussian noise of variance
    ``s2 * (1 - r2) / r2``: its R-squared against ``d`` is ``r2`` in
    expectation, and it is the best linear forecast given ``d + e``. At
    ``r2 = 1`` it is ``d`` itself. Stamped one nanosecond before its snapshot,
    so the decision at that snapshot reads it. Uses the future: only for the
    ladder.
    """
    if not 0.0 < r2 <= 1.0:
        raise ValueError(f"R-squared must be in (0, 1], got {r2}")
    ts, rows = last_rows(events)
    mid = analysis.book_mid(events)
    later = analysis.mid_at(events.book_ts, mid, ts + round(horizon_s * NS_PER_S))
    change = later - mid[rows]
    if r2 < 1.0:
        finite = np.isfinite(change)
        scale = float(np.sqrt(np.mean(change[finite] ** 2))) if finite.any() else 0.0
        noise = np.random.default_rng(seed).standard_normal(len(ts))
        change = r2 * (change + scale * np.sqrt((1.0 - r2) / r2) * noise)
    return SignalTape(FORECAST_SIGNAL, ts - 1, change)


#: Where a passive fill's order stood when it went live, against its side's touch.
INSIDE, TOUCH, BEHIND, TAKER = "inside", "touch", "behind", "taker"


def placement(fills: pd.DataFrame, orders: pd.DataFrame, events: DayEvents) -> np.ndarray:
    """Per fill: ``inside`` the spread, at the ``touch`` or ``behind`` it, for
    the order's price against its side's touch in the snapshot it arrived on
    (the last at or before it went live); ``taker`` for flattens and exits."""
    out = np.full(len(fills), TAKER, dtype=object)
    if fills.empty or orders.empty:
        return out
    by_oid = orders.set_index("oid")
    maker = fills["oid"].to_numpy(dtype=np.int64) > 0
    oids = fills["oid"].to_numpy(dtype=np.int64)[maker]
    live = by_oid.loc[oids, "live_ns"].to_numpy(dtype=np.int64)
    price = by_oid.loc[oids, "price"].to_numpy(dtype=np.int64)
    side = by_oid.loc[oids, "side"].to_numpy(dtype=np.int64)
    row = np.maximum(np.searchsorted(events.book_ts, live, side="right") - 1, 0)
    touch = np.where(side > 0, events.bid_px[row, 0], events.ask_px[row, 0]).astype(np.int64)
    better = side * (price - touch)
    out[maker] = np.where(better > 0, INSIDE, np.where(better == 0, TOUCH, BEHIND))
    return out


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Spec:
    """One configuration to simulate on a day.

    ``tapes`` are the external signals for the whole span; the job builder
    hands each day only its slice. ``built`` names tapes made from the loaded
    day: ``("stale_spread", lag_s)`` or ``("forecast", r2)``. ``oracle`` admits
    a quoter that reads the future (the ladder's rungs) and stamps its rows;
    ``keep`` writes the day's fills, orders and equity beside the summary.
    ``group`` says what the row is for (``main``, ``placebo``, ``fee``...).
    """

    label: str
    make_quoter: Callable[[], Quoter]
    config: SimConfig
    group: str = "main"
    tapes: Mapping[str, SignalTape] | None = None
    built: tuple[tuple[str, float], ...] = ()
    oracle: bool = False
    keep: bool = False


@dataclass(frozen=True)
class Job:
    """One instrument-day and the specs to run on it, loaded once."""

    key: str
    symbol: str
    day: date
    specs: tuple[Spec, ...]
    book_roots: tuple[Path, ...]
    trades_root: Path
    funding_root: Path | None
    out: Path
    noise_base: int = 0
    #: Horizons of the market-wide passive benchmark to compute on this day;
    #: empty for none.
    market_horizons_s: tuple[float, ...] = ()


def build_jobs(
    symbol: str,
    days: Sequence[date],
    specs: Sequence[Spec],
    *,
    book_roots: Sequence[Path],
    trades_root: Path,
    funding_root: Path | None,
    out: Path,
    specs_per_job: int = 8,
    market_horizons_s: tuple[float, ...] = (),
    noise_base: int = 0,
) -> list[Job]:
    """Cut the specs on each day into jobs; the first job of a day also
    computes the market-wide benchmark when ``market_horizons_s`` is given."""
    labels = [s.label for s in specs]
    if len(set(labels)) != len(labels):
        raise ValueError("spec labels must be unique per instrument")
    chunk = max(1, int(specs_per_job))
    jobs = []
    for day in sorted(set(days)):
        sliced = [replace(s, tapes=day_tapes(s.tapes, day)) for s in specs]
        for k, first in enumerate(range(0, max(len(sliced), 1), chunk)):
            jobs.append(
                Job(
                    key=f"{symbol}_{day.isoformat()}_{k:03d}",
                    symbol=symbol,
                    day=day,
                    specs=tuple(sliced[first : first + chunk]),
                    book_roots=tuple(Path(r) for r in book_roots),
                    trades_root=Path(trades_root),
                    funding_root=None if funding_root is None else Path(funding_root),
                    out=Path(out),
                    noise_base=noise_base,
                    market_horizons_s=market_horizons_s if k == 0 else (),
                )
            )
    return jobs


def _market_row(job: Job, events: DayEvents) -> dict[str, Any]:
    """The day's market statistics and the market-wide passive benchmark."""
    from trading_research.market_making.screen import day_statistics

    stats = day_statistics(events)
    row: dict[str, Any] = {
        "symbol": job.symbol,
        "day": job.day.isoformat(),
        "group": "market",
        "label": "market",
        "status": "ok",
        "seconds": stats.seconds,
        "spread_bp_seconds": stats.spread_bp_seconds,
        "tick_bp_seconds": stats.tick_bp_seconds,
        "two_tick_seconds": stats.two_tick_seconds,
        "prints": float(stats.prints),
        "sequence_gaps": float(stats.sequence_gaps),
        "median_touch": float(np.nanmedian(stats.touch)),
    }
    benchmark = analysis.market_wide_markouts(events, job.market_horizons_s)
    for _, line in benchmark.iterrows():
        name = f"{float(line['horizon_s']):g}s"
        row[f"benchmark_bp_{name}"] = float(line["markout_bp"])
        row[f"benchmark_volume_{name}"] = float(line["volume"])
    return row


def _keep(spec: Spec, job: Job, fills: pd.DataFrame, result: DayResult) -> None:
    folder = job.out / "keep" / spec.label / job.symbol
    folder.mkdir(parents=True, exist_ok=True)
    stem = job.day.isoformat()
    fills.to_parquet(folder / f"{stem}.fills.parquet", index=False)
    result.orders.to_parquet(folder / f"{stem}.orders.parquet", index=False)
    result.equity.to_parquet(folder / f"{stem}.equity.parquet", index=False)


def _simulate(spec: Spec, job: Job, events: DayEvents) -> dict[str, Any]:
    tapes = dict(spec.tapes or {})
    for kind, value in spec.built:
        if kind == "stale_spread":
            tapes[STALE_SPREAD_SIGNAL] = stale_spread_tape(events, value)
        elif kind == "forecast":
            seed = noise_seed(job.noise_base, job.symbol, job.day)
            tapes[FORECAST_SIGNAL] = forecast_tape(events, value, seed=seed)
        else:
            raise ValueError(f"unknown day tape {kind!r}")
    quoter = spec.make_quoter()
    result = simulate_day(events, quoter, spec.config, tapes or None, allow_oracle=spec.oracle)
    row = _summary(result, quoter.name)
    for name in EXTRA_COUNTERS:
        row[name] = float(result.counters.get(name, 0.0))
    for name, count in sorted(getattr(quoter, "counts", {}).items()):
        row[f"quoter_{name}"] = float(count)
    summarise = getattr(quoter, "summarise", None)
    if callable(summarise):
        for name, value in sorted(summarise(result.fills).items()):
            row[f"quoter_{name}"] = float(value)
    fills = result.fills.copy()
    fills["placement"] = placement(fills, result.orders, events)
    for where in (INSIDE, TOUCH, BEHIND, TAKER):
        mine = fills["placement"] == where
        row[f"fills_{where}"] = float(mine.sum())
        row[f"volume_{where}"] = float(fills.loc[mine, "size"].sum())
    live = result.orders[result.orders["outcome"] != "rejected"]
    row["orders_live"] = float(len(live))
    row["orders_with_fill"] = float((live["filled"] > 0).sum())
    row.update({"label": spec.label, "group": spec.group, "oracle": spec.oracle})
    if spec.keep:
        _keep(spec, job, fills, result)
    return row


def run_job(job: Job) -> list[dict[str, Any]]:
    """Every spec of a job on its instrument-day; the rows are also written to
    ``out/rows/<key>.json``, and a job whose file exists is read back."""
    target = job.out / "rows" / f"{job.key}.json"
    if target.exists():
        loaded: list[dict[str, Any]] = json.loads(target.read_text(encoding="utf-8"))
        return loaded
    rows: list[dict[str, Any]] = []
    try:
        events = load_day(
            job.symbol,
            job.day,
            book_roots=job.book_roots,
            trades_root=job.trades_root,
            funding_root=job.funding_root,
        )
    except DayUnavailable as exc:
        for spec in job.specs:
            row = _unavailable_row(job.symbol, job.day, spec.label, str(exc))
            row.update({"label": spec.label, "group": spec.group, "oracle": spec.oracle})
            rows.append(row)
        events = None
    if events is not None:
        if job.market_horizons_s:
            rows.append(_market_row(job, events))
        for spec in job.specs:
            rows.append(_simulate(spec, job, events))
            gc.collect()
    for row in rows:
        row["worker_peak_rss_mb"] = _peak_rss_mb()
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".tmp")
    partial.write_text(json.dumps(rows, sort_keys=True, default=float), encoding="utf-8")
    partial.replace(target)
    return rows


def run_jobs(
    jobs: Sequence[Job],
    *,
    workers: int = 1,
    day_guard: Callable[[Sequence[date]], None] | None = None,
    on_progress: Callable[[str], None] | None = None,
    every: int = 25,
) -> pd.DataFrame:
    """Run every job, each in a fresh process when ``workers`` is above one.

    ``day_guard`` is called with every day before anything is read. Jobs start
    in the order given, so a caller puts the busiest instrument-days first to
    avoid a long tail; the rows do not depend on the order.
    """
    if day_guard is not None:
        day_guard(sorted({job.day for job in jobs}))
    import time

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    done = 0

    def note() -> None:
        if on_progress and (done % every == 0 or done == len(jobs)):
            elapsed = time.perf_counter() - started
            on_progress(f"  {done}/{len(jobs)} jobs, {elapsed:,.0f} s")

    if workers <= 1:
        for job in jobs:
            rows.extend(run_job(job))
            done += 1
            note()
    else:
        import multiprocessing

        context = multiprocessing.get_context("spawn")
        with context.Pool(processes=workers, maxtasksperchild=1) as pool:
            for batch in pool.imap_unordered(run_job, jobs, chunksize=1):
                rows.extend(batch)
                done += 1
                note()
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["group", "label", "symbol", "day"], kind="stable").reset_index(
        drop=True
    )


def kept(out: Path, label: str, symbol: str, what: str) -> pd.DataFrame:
    """Every kept day of one spec on one instrument, concatenated with its day:
    ``what`` is ``fills``, ``orders`` or ``equity``."""
    folder = Path(out) / "keep" / label / symbol
    frames = []
    for path in sorted(folder.glob(f"*.{what}.parquet")):
        frame = pd.read_parquet(path)
        frame.insert(0, "day", path.name.split(".")[0])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
