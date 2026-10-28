"""Reported beside R, never defining ``b``: the second detector and the mean channel.

* **The second detector**: ``validation/changepoint.detect`` at its defaults
  (window 2000, threshold 20, reference 40, minimum gap 20,000, its four
  statistics) on the 1 s grid of BTCUSDT's mid and spread from the Bybit book
  (``data.grid.to_grid``, ``label="right"``). It runs continuously over D,
  **restarts on H's first day** (E's book is never fetched), runs continuously
  over H, **restarts on F's first day**, 2025-08-21 (no ``ob500`` state carried
  into ``ob200`` data), and runs continuously from F through P. An alarm within
  a day of 2025-08-21 is labelled "at the archive change". Each alarm is stamped
  at the label of the last grid row of the block that fired it.
* **The mean channel**: the frozen detector's own run, its ``odds_mean`` against
  8.93 (``regime_flags.mean_channel``): each window's first step at or above it.

A day enters the 1 s stream when its book is on disk and excluded for nothing
but sequence gaps (the grid carries the last book forward across a gap); each
day is cut to its own UTC hours and gridded on its own, and kept under the
run's work folder so a later block's run reads it again without its book.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from trading_research.data import study_access as sa

#: The second detector's continuous spans: the blocks each run reads, in order.
SPANS: dict[str, tuple[str, ...]] = {"D": ("D",), "H": ("H",), "F": ("F",), "F-P": ("F", "P")}
#: The span a block's read runs (D's is run in development, by stage C).
SPAN_OF_BLOCK: dict[str, str] = {"H": "H", "F": "F", "P": "F-P"}
ARCHIVE_CHANGE_LABEL = "at the archive change"
NS_PER_DAY = 86_400 * 10**9


def grid_day(book: pd.DataFrame, day: date) -> pd.DataFrame:
    """One day's book on the 1 s grid, labelled by each bin's end: ``timestamp``
    (ns), ``mid`` and ``spread_bp``; rows outside the day's UTC hours are cut first."""
    from trading_research.data.grid import to_grid
    from trading_research.market_making.events import to_ns

    start = pd.Timestamp(day.isoformat(), tz="UTC").value
    stamps = pd.DatetimeIndex(pd.to_datetime(book["timestamp"], utc=True))
    ns = to_ns(stamps)
    inside = (ns >= start) & (ns < start + NS_PER_DAY)
    frame = pd.DataFrame({"timestamp": stamps[inside],
                          "bid_price_0": book.loc[inside, "bid_price_0"].to_numpy(dtype=np.float64),
                          "ask_price_0": book.loc[inside, "ask_price_0"].to_numpy(dtype=np.float64)})  # fmt: skip
    if frame.empty:
        return pd.DataFrame({"timestamp": np.zeros(0, np.int64), "mid": [], "spread_bp": []})
    grid = to_grid(frame, seconds=1, label="right")
    bid, ask = grid["bid_price_0"].to_numpy(), grid["ask_price_0"].to_numpy()
    mid = (bid + ask) / 2.0
    labels = to_ns(pd.DatetimeIndex(pd.to_datetime(grid["timestamp"], utc=True)))
    return pd.DataFrame({"timestamp": labels, "mid": mid, "spread_bp": (ask - bid) / mid * 1e4})


def cached_grid(study: Any, day: date, access: sa.StudyAccess) -> pd.DataFrame | None:
    """A day's 1 s grid, kept under the run's work folder; None when the day's book
    is missing or excluded for anything but sequence gaps."""
    from trading_research.data import study_loader as sl

    access.require(study.symbol, [day], sa.BOOK, what="return the cached grid of")
    key = sa.cache_key(access)  # a read session's grids apart from development's
    folder = study.dir("detector2", "grid", key) if key else study.dir("detector2", "grid")
    path = folder / f"{day.isoformat()}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    found = sl.load_book_day(
        study.symbol, day, study.roots.book, max_sequence_gaps=None, access=access
    )
    if found.frame is None or set(found.exclusions) - {sl.SEQUENCE_GAPS}:
        return None
    grid = grid_day(found.frame, day)
    partial = path.with_name(path.name + ".part")
    grid.to_parquet(partial, index=False)
    partial.replace(path)
    return grid


def at_archive_change(stamp: pd.Timestamp) -> bool:
    change = pd.Timestamp(sa.ARCHIVE_CHANGE.isoformat(), tz="UTC").value
    return bool(abs(stamp.value - change) <= NS_PER_DAY)


def run_stream(grids: Sequence[pd.DataFrame], span: str) -> dict[str, Any]:
    """``changepoint.detect`` at its defaults over the grids as one stream."""
    from trading_research.validation.changepoint import detect

    kept = [g for g in grids if len(g)]
    frame = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame(columns=["timestamp"])
    rows = len(frame)
    out: dict[str, Any] = {"span": span, "rows": rows, "alarms": []}
    needed = 2000 * (40 + 2)
    if rows < needed:
        return out | {
            "note": f"{rows} grid rows, fewer than the {needed} the detector needs: not monitored"
        }
    labels = frame["timestamp"].to_numpy(dtype=np.int64)
    for found in detect(frame["mid"].to_numpy(), frame["spread_bp"].to_numpy()):
        stamp = pd.Timestamp(int(labels[found.index - 1]), tz="UTC")
        out["alarms"].append({"timestamp": stamp.isoformat(), "statistic": found.statistic,
                              "direction": int(found.direction), "excursion": float(found.excursion),
                              "label": ARCHIVE_CHANGE_LABEL if at_archive_change(stamp) else ""})  # fmt: skip
    return out


def span_days(study: Any, span: str) -> list[date]:
    return [d for block in SPANS[span] for d in study.scale.days(block)]


def run_span(study: Any, span: str, access: sa.StudyAccess) -> dict[str, Any]:
    """The second detector over one continuous span, with the days it read."""
    return run_days(study, span_days(study, span), access, span)


def run_days(study: Any, days: Sequence[date], access: sa.StudyAccess, span: str) -> dict[str, Any]:
    """The second detector over ``days`` as one continuous stream."""
    grids = [(d, cached_grid(study, d, access)) for d in days]
    found = run_stream([g for _, g in grids if g is not None], span)
    return found | {"days": len(days), "days_read": sum(g is not None for _, g in grids)}


def mean_channel(close: pd.Series, end: date, profile: Sequence[float]) -> list[dict[str, Any]]:
    """The frozen detector's mean channel over the stream to ``end``: each window's
    first step at or above 8.93, as ``regime_flags.mean_channel`` reads it."""
    from trading_research.validation import regime_flags as rf

    stamp = pd.Timestamp((end + timedelta(days=1)).isoformat(), tz="UTC")
    table = rf.mean_channel(
        rf.run_detector(close, np.asarray(profile, dtype=np.float64), end=stamp)
    )
    return [{"timestamp": pd.Timestamp(t).isoformat(), "window": int(w), "odds_mean": float(o)}
            for t, w, o in zip(table["timestamp"], table["window"], table["odds_mean"], strict=True)]  # fmt: skip


def beside_r(block: Any) -> dict[str, Any]:
    """What a block's read reports beside R: its span of the second detector over the
    block's planned days (P's after F's, its span running on from F; a stand-in for
    P on days outside P, the rehearsal's, reads its own days only), and the frozen
    detector's mean channel over the stream to the block's end."""
    from trading_research.pipeline.short_horizon import flags

    end = block.end or sa.BLOCKS[block.name][1]
    close = flags.hourly_close(block.study.symbol, block.study.roots.bars, end, block.access)
    profile = list(block.value(("detector", "hour_profile")))
    span = SPAN_OF_BLOCK.get(block.name)
    days = list(block.days_planned)
    if block.name == "P" and days and sa.block_of(days[0]) == "P":
        days = [*block.study.scale.days("F"), *days]
    found = None if span is None else run_days(block.study, days, block.access, span)
    return {"second_detector": found, "mean_channel": mean_channel(close, end, profile)}
