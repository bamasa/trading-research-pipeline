"""The ten-level book tensor: forty columns, scaled by values fitted on W.

Each row of the replayed book becomes forty numbers, ten levels a side, a price
and a size for each:

- **price**: the level's distance from the touch on its own side
  (``ask_i - ask_0``, ``bid_0 - bid_i``), divided by the larger of ``sigma * mid``
  and a spread floor. ``sigma`` is the standard deviation of the mid's
  row-to-row log changes over the trailing 60 s on row stamps; the floor is the
  median quoted spread on the warm-up block W. Level 0's distance is zero by
  this definition; the columns are kept so that every level has the same
  layout.
- **size**: ``log(1 + size)``, z-scored against that level's mean and standard
  deviation on W.

Why fixed calibration values rather than rolling ones
------------------------------------------------------
A whole-day statistic leaks the rest of the day into its morning, and a rolling
one makes the scale of a level depend on the hour. The values here are fitted
once, on W, which no scored day follows from, and are written to the
fitted-values file; on every later block they are read, never refitted. The
spread floor stops a quiet minute, whose ``sigma`` is near zero, from blowing a
one-tick distance up into a huge number.

The 60 s ``sigma`` is the only rolling state. Its window is in time, so the
tensor is registered with ``lookback_s=60`` and one row more (the first change
in the window reads the row before it). Prices are differenced in float64 and
the variance is pandas' online estimator over log changes near zero, not the
``E[x^2] - E[x]^2`` of prices near 1e5, which loses most of its digits there.

Columns are ordered ask distances 0-9, bid distances 0-9, ask sizes 0-9, bid
sizes 0-9.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from trading_research.features.timed import (
    TimedFeature,
    TimedRegistry,
    register_builder,
    rolling_time,
    stamps_ns,
)

LEVELS: int = 10
SIGMA_WINDOW_S: float = 60.0


def tensor_columns(levels: int = LEVELS) -> list[str]:
    """The tensor's column names, in the network's channel order."""
    names = [f"lob_ask_dist_{i}" for i in range(levels)]
    names += [f"lob_bid_dist_{i}" for i in range(levels)]
    names += [f"lob_ask_size_{i}" for i in range(levels)]
    names += [f"lob_bid_size_{i}" for i in range(levels)]
    return names


def _log_sizes(rows: pd.DataFrame, levels: int) -> np.ndarray:
    """``log(1 + size)`` for asks 0..L-1 then bids 0..L-1, one row per snapshot."""
    columns = [f"ask_size_{i}" for i in range(levels)] + [f"bid_size_{i}" for i in range(levels)]
    return np.log1p(rows[columns].to_numpy(dtype="float64"))


@dataclass(frozen=True)
class TensorCalibration:
    """The values fitted on W: the spread floor and each size column's moments."""

    spread_floor: float
    size_mean: tuple[float, ...]
    size_std: tuple[float, ...]
    levels: int = LEVELS
    sigma_window_s: float = SIGMA_WINDOW_S

    def __post_init__(self) -> None:
        if not self.spread_floor > 0:
            raise ValueError(f"spread_floor must be positive, got {self.spread_floor}")
        if len(self.size_mean) != 2 * self.levels or len(self.size_std) != 2 * self.levels:
            raise ValueError(f"need {2 * self.levels} size moments, one per level and side")
        if any(not s > 0 for s in self.size_std):
            raise ValueError("every size standard deviation must be positive")

    def to_dict(self) -> dict[str, object]:
        """For the fitted-values file."""
        return {
            "spread_floor": self.spread_floor,
            "size_mean": list(self.size_mean),
            "size_std": list(self.size_std),
            "levels": self.levels,
            "sigma_window_s": self.sigma_window_s,
        }

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> TensorCalibration:
        mean = values["size_mean"]
        std = values["size_std"]
        if not isinstance(mean, list | tuple) or not isinstance(std, list | tuple):
            raise TypeError("size_mean and size_std must be lists")
        return cls(
            spread_floor=float(str(values["spread_floor"])),
            size_mean=tuple(float(v) for v in mean),
            size_std=tuple(float(v) for v in std),
            levels=int(str(values.get("levels", LEVELS))),
            sigma_window_s=float(str(values.get("sigma_window_s", SIGMA_WINDOW_S))),
        )


@dataclass
class TensorCalibrator:
    """Accumulates W's days one at a time and returns the calibration.

    W is a month of 100 ms rows, too large to hold as one frame. The quoted
    spread takes few distinct values (whole ticks), so its median is exact
    from counts; the size moments are merged day by day with Chan's update,
    which keeps the variance exact without holding the rows.
    """

    levels: int = LEVELS
    decimals: int = 8
    spread_counts: dict[float, int] = field(default_factory=dict)
    n: int = 0
    mean: np.ndarray | None = None
    m2: np.ndarray | None = None

    def add(self, rows: pd.DataFrame) -> None:
        spread = rows["ask_price_0"].to_numpy("float64") - rows["bid_price_0"].to_numpy("float64")
        values, counts = np.unique(np.round(spread, self.decimals), return_counts=True)
        for value, count in zip(values.tolist(), counts.tolist(), strict=True):
            self.spread_counts[value] = self.spread_counts.get(value, 0) + int(count)

        x = _log_sizes(rows, self.levels)
        n_b = len(x)
        if n_b == 0:
            return
        mean_b = x.mean(axis=0)
        m2_b = ((x - mean_b) ** 2).sum(axis=0)
        if self.mean is None or self.m2 is None:
            self.n, self.mean, self.m2 = n_b, mean_b, m2_b
            return
        total = self.n + n_b
        delta = mean_b - self.mean
        self.mean = self.mean + delta * n_b / total
        self.m2 = self.m2 + m2_b + delta**2 * self.n * n_b / total
        self.n = total

    def spread_median(self) -> float:
        values = np.array(sorted(self.spread_counts))
        cumulative = np.cumsum([self.spread_counts[v] for v in values])
        total = int(cumulative[-1])
        lower = values[np.searchsorted(cumulative, (total - 1) // 2 + 1)]
        upper = values[np.searchsorted(cumulative, total // 2 + 1)]
        return float((lower + upper) / 2)

    def result(self) -> TensorCalibration:
        if self.mean is None or self.m2 is None or self.n == 0:
            raise ValueError("no rows were added")
        return TensorCalibration(
            spread_floor=self.spread_median(),
            size_mean=tuple(float(v) for v in self.mean),
            size_std=tuple(float(v) for v in np.sqrt(self.m2 / self.n)),
            levels=self.levels,
        )


def fit_tensor_calibration(days: Iterable[pd.DataFrame], levels: int = LEVELS) -> TensorCalibration:
    """Fit the calibration on W, given as an iterable of day frames."""
    calibrator = TensorCalibrator(levels=levels)
    for rows in days:
        calibrator.add(rows)
    return calibrator.result()


def mid_log_change_std(rows: pd.DataFrame, window_s: float = SIGMA_WINDOW_S) -> np.ndarray:
    """Std of the mid's row-to-row log changes over the trailing window on stamps."""
    stamps = stamps_ns(rows)
    mid = (rows["bid_price_0"].to_numpy("float64") + rows["ask_price_0"].to_numpy("float64")) / 2
    change = np.full(len(mid), np.nan)
    change[1:] = np.log1p(np.diff(mid) / mid[:-1])
    return rolling_time(change, stamps, window_s).std().to_numpy()


def lob_tensor(rows: pd.DataFrame, calibration: TensorCalibration) -> pd.DataFrame:
    """The forty tensor columns for every row, NaN until ``sigma`` is defined."""
    levels = calibration.levels
    ask = rows[[f"ask_price_{i}" for i in range(levels)]].to_numpy("float64")
    bid = rows[[f"bid_price_{i}" for i in range(levels)]].to_numpy("float64")
    mid = (ask[:, 0] + bid[:, 0]) / 2
    sigma = mid_log_change_std(rows, calibration.sigma_window_s)
    scale = np.maximum(sigma * mid, calibration.spread_floor)
    ask_dist = (ask - ask[:, :1]) / scale[:, None]
    bid_dist = (bid[:, :1] - bid) / scale[:, None]
    sizes = (_log_sizes(rows, levels) - np.asarray(calibration.size_mean)) / np.asarray(
        calibration.size_std
    )
    values = np.hstack([ask_dist, bid_dist, sizes])
    return pd.DataFrame(values, columns=tensor_columns(levels), index=rows.index)


def register_tensor(
    calibration: TensorCalibration, registry: TimedRegistry | None = None
) -> list[TimedFeature]:
    """Register the forty columns, with the 60 s window and one row declared."""

    def builder(rows: pd.DataFrame, _: pd.DataFrame | None) -> pd.DataFrame:
        return lob_tensor(rows, calibration)

    return register_builder(
        builder,
        tensor_columns(calibration.levels),
        plane="book",
        lookback_s=calibration.sigma_window_s,
        lookback_rows=1,
        description="ten-level book tensor",
        registry=registry,
    )
