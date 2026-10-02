"""Instrument admission for the market-making study, computed on block D only.

The pre-registration admits an instrument to the market-making verdicts (S0-S3,
H1, H3) only where quoting can pay at all: a time-weighted spread of at least
twice the base maker fee, a spread of two ticks or more at least a quarter of
the time, enough prints to be filled, and data on at least 90% of the days of D
and of H. H2's instruments need only to be in §27's universe, outside its
exclusions, with the same coverage. This module computes the statistics and
applies the rule.

Statistics, per instrument over the days of D
---------------------------------------------
* ``tick_bp`` — the tick over the mid, time-weighted;
* ``spread_bp_time_weighted`` — the touch spread over the mid, each snapshot
  weighted by the time until the next one (the last until midnight);
* ``share_at_two_ticks_or_more`` — the time-weighted share of a spread of two
  ticks or more;
* ``median_touch_over_median_print`` — the median touch size (the mean of the
  best bid and ask sizes, sampled at every whole second, as the simulator's
  trailing touch is) over the median print size;
* ``prints_per_day`` — the mean;
* ``market_wide_passive_markout_bp_{1,5,30}s`` — every print scored from its
  resting side (:func:`.analysis.market_wide_markouts`), volume-weighted over
  the block;
* ``coverage_D``, ``coverage_H`` — the share of the block's days with both a
  book file and a print file on disk.

Two values the amendment freezes come from the same pass: the clip's notional
cap, a tenth of the median touch notional over D, and ``sigma_ref``, the median
over D's quoting hours of the one-minute volatility the simulator computes.

What is read
------------
The statistics read D's files, one instrument-day at a time, after the
:class:`~trading_research.market_making.prereg.Access` given permits each day.
Coverage of H is the one statistic about H that the rule needs, and
:func:`coverage` computes it from the file system alone — whether a non-empty
file exists for the day — without opening any file.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_research.market_making.events import NS_PER_DAY, NS_PER_S, load_day

#: The registered thresholds of the market-making admission rule.
SPREAD_BP_MIN = 4.0
SHARE_TWO_TICKS_MIN = 0.25
PRINTS_PER_DAY_MIN = 5_000.0
COVERAGE_MIN = 0.90

#: Markout horizons of the market-wide benchmark, seconds.
HORIZONS_S = (1.0, 5.0, 30.0)

#: The clip's notional cap is this share of D's median touch notional.
CLIP_SHARE_OF_MEDIAN_TOUCH = 0.10


def coverage(
    symbol: str,
    days: Sequence[date],
    *,
    book_roots: Sequence[Path],
    trades_root: Path,
) -> float:
    """Share of ``days`` with a non-empty book file and print file on disk.

    Reads the file system's directory entries only: no file is opened, so it
    says nothing about a day beyond whether its data was fetched.
    """
    if not days:
        raise ValueError("no days to cover")
    present = 0
    for day in days:
        name = f"{day.isoformat()}.parquet"
        book = any(_non_empty(Path(root) / symbol / name) for root in book_roots)
        trades = _non_empty(Path(trades_root) / symbol / name)
        present += int(book and trades)
    return present / len(days)


def _non_empty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


@dataclass(frozen=True)
class DayStatistics:
    """What one instrument-day contributes to the block's statistics."""

    symbol: str
    day: date
    seconds: float
    spread_bp_seconds: float
    tick_bp_seconds: float
    two_tick_seconds: float
    prints: int
    sequence_gaps: int
    #: The touch, its notional and the one-minute volatility at every whole
    #: second of the day (volatility over the quoting hours only).
    touch: np.ndarray
    touch_notional: np.ndarray
    vol_bp_1m: np.ndarray
    #: Distinct print sizes and their counts, for an exact median.
    size_values: np.ndarray
    size_counts: np.ndarray
    #: Market-wide passive markout per horizon: volume-weighted mean and volume.
    markout_bp: tuple[float, ...]
    markout_volume: tuple[float, ...]


def day_statistics(
    events: Any, *, warmup_s: float = 600.0, stop_s: float = 86_280.0
) -> DayStatistics:
    """The statistics of one loaded day (a :class:`.events.DayEvents`)."""
    from trading_research.market_making import analysis
    from trading_research.market_making.simulator import SimConfig, _per_second

    tick = events.spec.tick
    book_ts = events.book_ts
    bid = events.bid_px[:, 0].astype(np.float64)
    ask = events.ask_px[:, 0].astype(np.float64)
    mid = (bid + ask) * 0.5 * tick
    end = events.day_start_ns + NS_PER_DAY
    duration = np.diff(np.append(book_ts, max(end, int(book_ts[-1])))).astype(np.float64) / NS_PER_S
    spread_ticks = ask - bid
    spread_bp = spread_ticks * tick / mid * 1e4
    tick_bp = tick / mid * 1e4

    config = SimConfig(clip_notional=1.0)
    vol, _ = _per_second(events, config)
    vol_array = np.asarray(vol, dtype=np.float64)
    seconds = np.arange(len(vol_array))
    quoting = (seconds >= warmup_s) & (seconds < stop_s)
    boundaries = events.day_start_ns + np.arange(86_400, dtype=np.int64) * NS_PER_S
    row = np.searchsorted(book_ts, boundaries, side="right") - 1
    known = row >= 0
    touch = np.full(len(boundaries), np.nan)
    touch[known] = 0.5 * (events.bid_sz[row[known], 0] + events.ask_sz[row[known], 0])
    touch_mid = np.full(len(boundaries), np.nan)
    touch_mid[known] = mid[row[known]]

    values, counts = np.unique(np.asarray(events.trade_sz, dtype=np.float64), return_counts=True)
    benchmark = analysis.market_wide_markouts(events, HORIZONS_S)
    return DayStatistics(
        symbol=events.spec.symbol,
        day=events.day,
        seconds=float(duration.sum()),
        spread_bp_seconds=float(np.dot(spread_bp, duration)),
        tick_bp_seconds=float(np.dot(tick_bp, duration)),
        two_tick_seconds=float(duration[spread_ticks >= 2].sum()),
        prints=len(events.trade_ts),
        sequence_gaps=int(events.sequence_gaps),
        touch=touch,
        touch_notional=touch * touch_mid,
        vol_bp_1m=vol_array[quoting],
        size_values=values,
        size_counts=counts,
        markout_bp=tuple(float(v) for v in benchmark["markout_bp"]),
        markout_volume=tuple(float(v) for v in benchmark["volume"]),
    )


def _one_day(job: tuple[str, date, tuple[Path, ...], Path]) -> DayStatistics | str:
    """Load and summarise one instrument-day; a reason string if it cannot be."""
    from trading_research.market_making.events import DayUnavailable

    symbol, day, book_roots, trades_root = job
    try:
        events = load_day(
            symbol, day, book_roots=book_roots, trades_root=trades_root, funding_root=None
        )
    except DayUnavailable as exc:
        return str(exc)
    return day_statistics(events)


def _weighted_median(values: np.ndarray, counts: np.ndarray) -> float:
    order = np.argsort(values)
    values, counts = values[order], counts[order]
    cumulative = np.cumsum(counts)
    half = cumulative[-1] / 2.0
    return float(values[int(np.searchsorted(cumulative, half, side="left"))])


def combine(days: Sequence[DayStatistics]) -> dict[str, float]:
    """The block's statistics from its days."""
    if not days:
        raise ValueError("no days to combine")
    seconds = sum(d.seconds for d in days)
    sizes = (
        pd.concat([pd.Series(d.size_counts, index=d.size_values) for d in days])
        .groupby(level=0)
        .sum()
    )
    median_print = _weighted_median(sizes.index.to_numpy(dtype=np.float64), sizes.to_numpy())
    touch = np.concatenate([d.touch for d in days])
    notional = np.concatenate([d.touch_notional for d in days])
    vol = np.concatenate([d.vol_bp_1m for d in days])
    out: dict[str, float] = {
        "days": float(len(days)),
        "tick_bp": sum(d.tick_bp_seconds for d in days) / seconds,
        "spread_bp_time_weighted": sum(d.spread_bp_seconds for d in days) / seconds,
        "share_at_two_ticks_or_more": sum(d.two_tick_seconds for d in days) / seconds,
        "median_touch": float(np.nanmedian(touch)),
        "median_print": median_print,
        "median_touch_over_median_print": float(np.nanmedian(touch)) / median_print,
        "prints_per_day": float(np.mean([d.prints for d in days])),
        "median_touch_notional_usdt": float(np.nanmedian(notional)),
        "sigma_ref_bp_1m": float(np.nanmedian(vol)),
        "sequence_gap_days": float(sum(d.sequence_gaps > 0 for d in days)),
    }
    out["clip_notional_usdt"] = CLIP_SHARE_OF_MEDIAN_TOUCH * out["median_touch_notional_usdt"]
    for k, horizon in enumerate(HORIZONS_S):
        weights = np.array([d.markout_volume[k] for d in days])
        values = np.array([d.markout_bp[k] for d in days])
        ok = np.isfinite(values) & (weights > 0)
        out[f"market_wide_passive_markout_bp_{horizon:g}s"] = (
            float(np.average(values[ok], weights=weights[ok])) if ok.any() else float("nan")
        )
    return out


def screen(
    symbols: Sequence[str],
    *,
    d_days: Sequence[date],
    h_days: Sequence[date],
    book_roots: Sequence[Path],
    trades_root: Path,
    access: object,
    workers: int = 1,
    on_progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """The admission statistics of every instrument, one row each.

    ``d_days`` are read (after ``access`` permits them); ``h_days`` are only
    checked for presence on disk.
    """
    from trading_research.market_making.prereg import Access

    if not isinstance(access, Access):
        raise TypeError("the screen needs a prereg.Access")
    access.require(d_days, what="screen")
    jobs = [
        (symbol, day, tuple(Path(r) for r in book_roots), Path(trades_root))
        for symbol in symbols
        for day in sorted(set(d_days))
    ]
    if workers <= 1:
        results = [_one_day(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_one_day, jobs))
    rows = []
    for symbol in symbols:
        mine = [r for r, job in zip(results, jobs, strict=True) if job[0] == symbol]
        loaded = [r for r in mine if isinstance(r, DayStatistics)]
        row: dict[str, Any] = {"symbol": symbol}
        row.update(combine(loaded) if loaded else {})
        row["days_unavailable"] = sum(isinstance(r, str) for r in mine)
        row["coverage_D"] = coverage(symbol, d_days, book_roots=book_roots, trades_root=trades_root)
        row["coverage_H"] = coverage(symbol, h_days, book_roots=book_roots, trades_root=trades_root)
        rows.append(row)
        if on_progress:
            on_progress(f"  screened {symbol}: {len(loaded)} days")
    return pd.DataFrame(rows)


def admit(table: pd.DataFrame, *, universe: Sequence[str], exclude: Sequence[str]) -> pd.DataFrame:
    """Apply the registered admission rule; adds ``mm_admitted`` and ``h2_admitted``."""
    out = table.copy()
    covered = (out["coverage_D"] >= COVERAGE_MIN) & (out["coverage_H"] >= COVERAGE_MIN)
    out["mm_admitted"] = (
        (out["spread_bp_time_weighted"] >= SPREAD_BP_MIN)
        & (out["share_at_two_ticks_or_more"] >= SHARE_TWO_TICKS_MIN)
        & (out["prints_per_day"] >= PRINTS_PER_DAY_MIN)
        & covered
    )
    out["h2_admitted"] = (
        out["symbol"].isin(list(universe)) & ~out["symbol"].isin(list(exclude)) & covered
    )
    return out
