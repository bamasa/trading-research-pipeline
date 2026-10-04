"""The second round's registered statistics: its metric, its pooling, its new kills.

Pure functions of numbers the simulator already produced; nothing reads market
data. What ``docs/preregistration/market_making_round2.md`` fixes:

* **the primary metric**, net per 100 USDT of clip a day: for instrument ``i``
  on day ``t``, ``y_it = net_it * 100 / clip_i`` with ``clip_i`` the frozen clip
  notional cap (:func:`per_clip`); pooled, the mean of ``y_it`` over the
  admitted instruments with a usable day ``t`` (:func:`pooled_daily`); a paired
  difference is taken per instrument-day where both days are usable, then
  pooled the same way (:func:`paired_daily`);
* **the minimum detectable effect**, ``2 * sigma_D / sqrt(n)`` (:func:`mde`);
* **K-queue**, a positive verdict that is not positive under pessimistic
  cancellation attribution and ``all_ahead`` growth together, applied to every
  sign claim as K-pess is (:func:`k_queue`);
* **K-nbhd**, the held-out metric recomputed with each cell of the chosen
  cell's D neighbourhood has a median at or below zero (:func:`k_nbhd`, and
  :func:`k_nbhd_per_instrument` for S1, whose neighbourhood is per instrument);
* **the gate's choice on D**: the highest neighbourhood median among the cells
  the admitted instruments fill at least five times a day on average, ties
  broken by the cell's own value (:func:`choose_gate_cell`).

The rest of round one's statistics (:mod:`.verdicts`) apply unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from trading_research.market_making.gate import GateCell, cell_neighbourhood


def usable(rows: pd.DataFrame) -> pd.DataFrame:
    """The rows that may enter a verdict: simulated, and not excluded."""
    excluded = rows["excluded"].astype(bool)
    return rows[(rows["status"] == "ok") & ~excluded]


def per_clip(rows: pd.DataFrame, clips: Mapping[str, float], column: str = "net") -> pd.DataFrame:
    """``symbol``, ``day`` and ``y``: ``column`` per 100 USDT of the frozen clip,
    for every usable instrument-day."""
    ok = usable(rows)
    missing = sorted(set(ok["symbol"]) - set(clips))
    if missing:
        raise KeyError(f"no frozen clip for {missing}")
    clip = ok["symbol"].map(dict(clips)).astype(np.float64)
    if (clip <= 0).any():
        raise ValueError("a clip must be positive")
    y = ok[column].astype(np.float64) * 100.0 / clip
    return pd.DataFrame({"symbol": ok["symbol"], "day": ok["day"], "y": y}).reset_index(drop=True)


def pooled_daily(rows: pd.DataFrame, clips: Mapping[str, float], column: str = "net") -> pd.Series:
    """The pooled daily metric: each day's mean of ``y_it`` over the instruments
    with a usable day."""
    frame = per_clip(rows, clips, column)
    return frame.groupby("day")["y"].mean().rename(f"{column}_per_100_clip")


def paired_daily(
    rows: pd.DataFrame,
    base: pd.DataFrame,
    clips: Mapping[str, float],
    column: str = "net",
) -> pd.Series:
    """``rows`` minus ``base`` per instrument-day where both are usable, per 100
    USDT of clip, pooled over instruments each day as the metric is."""
    a = per_clip(rows, clips, column)
    b = per_clip(base, clips, column)
    both = a.merge(b, on=["symbol", "day"], suffixes=("", "_base"))
    both["d"] = both["y"] - both["y_base"]
    return both.groupby("day")["d"].mean().rename(f"{column}_paired_per_100_clip")


def usdt_daily(rows: pd.DataFrame, column: str = "net") -> pd.Series:
    """The pooled USDT sum, reported beside the primary: each day's total."""
    return usable(rows).groupby("day")[column].sum()


def mde(daily: pd.Series | Sequence[float], n: int) -> dict[str, float]:
    """``sigma_D``, the standard deviation of a daily series over D, and the
    minimum detectable effect ``2 * sigma_D / sqrt(n)`` for ``n`` held-out days."""
    values = np.asarray(daily, dtype=np.float64)
    values = values[np.isfinite(values)]
    sigma = float(values.std(ddof=1)) if len(values) > 1 else math.nan
    return {"days": float(len(values)), "sigma_d": sigma, "mde": 2.0 * sigma / math.sqrt(n)}


def k_queue(default: Sequence[float], joint: Sequence[float]) -> bool:
    """K-queue: every sign claim positive under the default rules, and some
    claim not positive under the joint pessimistic queue bracket."""
    return bool(all(v > 0 for v in default) and not all(v > 0 for v in joint))


def k_nbhd(values: Sequence[float]) -> tuple[float, bool]:
    """K-nbhd for one claim: the median of the held-out metric over the chosen
    cell's neighbourhood, and whether it fires (median at or below zero). A
    neighbourhood with no finite value fires: nothing shows it positive."""
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return math.nan, True
    median = float(np.median(array))
    return median, median <= 0


def k_nbhd_per_instrument(values: Mapping[str, Sequence[float]]) -> tuple[float, bool]:
    """K-nbhd where each instrument has its own neighbourhood (S1): each
    instrument's median over its cells, pooled as the metric pools, by the
    mean over instruments; it fires when that is at or below zero."""
    medians = []
    for cells in values.values():
        array = np.asarray(cells, dtype=np.float64)
        array = array[np.isfinite(array)]
        if len(array):
            medians.append(float(np.median(array)))
    if not medians:
        return math.nan, True
    pooled = float(np.mean(medians))
    return pooled, pooled <= 0


def neighbourhood_medians(
    scores: Mapping[GateCell, float], cells: Sequence[GateCell] | None = None
) -> dict[GateCell, float]:
    """Each cell's registered neighbourhood median over the cells in ``scores``
    with a finite value (``cells`` restricts which cells get a median)."""
    known = [c for c, v in scores.items() if math.isfinite(v)]
    out = {}
    for cell in cells if cells is not None else known:
        near = [scores[c] for c in cell_neighbourhood(cell, known)]
        out[cell] = float(np.median(near)) if near else math.nan
    return out


def choose_gate_cell(
    values: Mapping[GateCell, float],
    fills: Mapping[GateCell, float],
    *,
    min_fills: float = 5.0,
) -> tuple[GateCell, dict[str, object]]:
    """The registered choice of one gate cell on D.

    ``values`` is each cell's objective (the mean over D of the pooled daily
    metric) and ``fills`` the mean over the admitted instruments of its passive
    fills a day. Cells under ``min_fills`` take no part; among the rest, the
    highest neighbourhood median wins, ties broken by the cell's own value and
    then by the registered order of the cells. Returns the cell and the record
    the amendment freezes: the neighbourhood's cells with their values, its
    median, the cell's own value and the outright peak.
    """
    taking_part = {
        c: float(v)
        for c, v in values.items()
        if math.isfinite(v) and fills.get(c, 0.0) >= min_fills
    }
    if not taking_part:
        raise ValueError(f"no gate cell reaches {min_fills} passive fills a day")
    medians = neighbourhood_medians(taking_part)
    order = list(values)
    chosen = max(taking_part, key=lambda c: (medians[c], taking_part[c], -order.index(c)))
    peak = max(taking_part, key=lambda c: (taking_part[c], -order.index(c)))

    def row(cell: GateCell) -> list[object]:
        record = cell.record()
        return [*record.values(), float(f"{taking_part[cell]:.6g}")]

    near = cell_neighbourhood(chosen, list(taking_part))
    return chosen, {
        "axes": [*chosen.record(), "value"],
        "chosen": list(chosen.record().values()),
        "median": float(f"{medians[chosen]:.6g}"),
        "own": float(f"{taking_part[chosen]:.6g}"),
        "cells": [row(c) for c in near],
        "peak": row(peak),
        "cells_taking_part": len(taking_part),
    }
