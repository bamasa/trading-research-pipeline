"""The registration's inputs, target and cost, as arrays of bar by instrument.

Every input at bar *t* is a function of bars up to *t* alone. The target runs
from the mid one bar after the decision to the mid eleven bars after it, so the
last bar's bid-ask bounce is not in it, and the cost is the taker fee twice
plus half the spread at entry and at exit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from trading_research.xs_deep.panel import Panel

#: Bars of history the networks read.
WINDOW = 60
#: Bars from the decision to the entry, and from the entry to the exit.
ENTRY_LAG, HOLD = 1, 10
#: Decision times are every tenth bar; training rows every fifth.
DECIDE_EVERY, TRAIN_EVERY = 10, 5
#: Bybit's base taker fee, bp a side: the benchmark's.
TAKER_BP = 5.5
#: Inputs are clipped here after standardising, the target's z-score at 3.
CLIP_INPUT, CLIP_TARGET = 5.0, 3.0

TABULAR = (
    "rel_1", "rel_5", "rel_10", "rel_30", "rel_60", "idx_10", "idx_60",
    "vol_60", "idx_vol_60", "spread", "imbalance", "imbalance_10",
)  # fmt: skip
SERIES = ("rel_1m", "idx_1m", "spread", "imbalance")
REL_10 = TABULAR.index("rel_10")


@dataclass
class Frame:
    """Inputs and outcomes on the panel's bars; NaN where undefined."""

    tab: np.ndarray  # (T, N, 12)
    series: np.ndarray  # (T, N, 4)
    seq_ok: np.ndarray  # (T, N): the last WINDOW bars of every series are finite
    target: np.ndarray  # (T, N): relative return, entry to exit, bp
    fwd: np.ndarray  # (T, N): the instrument's own return, entry to exit, bp
    cost: np.ndarray  # (T, N): a leg's round trip, bp

    def usable(self) -> np.ndarray:
        """(T, N): every input and outcome of a row is defined."""
        inputs = np.isfinite(self.tab).all(axis=-1) & self.seq_ok
        return inputs & np.isfinite(self.target) & np.isfinite(self.cost)

    def windows(self, rows: np.ndarray) -> np.ndarray:
        """(len(rows), N, 4, WINDOW): the series' last WINDOW bars at each row."""
        view = sliding_window_view(self.series, WINDOW, axis=0)  # (T-W+1, N, 4, W)
        return np.ascontiguousarray(view[rows - (WINDOW - 1)])


def _rolling(x: np.ndarray, k: int, how: str) -> np.ndarray:
    frame = pd.DataFrame(x).rolling(k, min_periods=k)
    return np.asarray(getattr(frame, how)().to_numpy(), dtype="float64")


def _row_mean(x: np.ndarray) -> np.ndarray:
    count = np.isfinite(x).sum(axis=1)
    total = np.nansum(x, axis=1)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def build(panel: Panel) -> Frame:
    logm = np.log(panel.mid)
    r1 = np.vstack([np.full((1, logm.shape[1]), np.nan), np.diff(logm, axis=0) * 1e4])
    idx1 = _row_mean(r1)
    rel1 = r1 - idx1[:, None]
    n = logm.shape[1]
    idx_col = np.repeat(idx1[:, None], n, axis=1)
    tab = np.stack(
        [
            rel1,
            _rolling(rel1, 5, "sum"),
            _rolling(rel1, 10, "sum"),
            _rolling(rel1, 30, "sum"),
            _rolling(rel1, 60, "sum"),
            _rolling(idx_col, 10, "sum"),
            _rolling(idx_col, 60, "sum"),
            _rolling(rel1, 60, "std"),
            _rolling(idx_col, 60, "std"),
            panel.spread_bp,
            panel.imbalance,
            _rolling(panel.imbalance, 10, "mean"),
        ],
        axis=-1,
    )
    series = np.stack([rel1, idx_col, panel.spread_bp, panel.imbalance], axis=-1)
    bad = (~np.isfinite(series).all(axis=-1)).astype("float64")
    seq_ok = _rolling(bad, WINDOW, "sum") == 0

    def ahead(x: np.ndarray, k: int) -> np.ndarray:
        out = np.full_like(x, np.nan)
        out[: len(x) - k] = x[k:]
        return out

    entry, exit_ = ENTRY_LAG, ENTRY_LAG + HOLD
    fwd = (ahead(logm, exit_) - ahead(logm, entry)) * 1e4
    target = fwd - _row_mean(fwd)[:, None]
    cost = 2 * TAKER_BP + ahead(panel.spread_bp, entry) / 2 + ahead(panel.spread_bp, exit_) / 2
    return Frame(tab=tab, series=series, seq_ok=seq_ok, target=target, fwd=fwd, cost=cost)


def zscore_rows(y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Each row's target as a z-score across its usable instruments, clipped."""
    y = np.where(mask, y, np.nan)
    mu = np.nanmean(y, axis=1, keepdims=True)
    sd = np.nanstd(y, axis=1, keepdims=True)
    z = (y - mu) / np.where(sd > 0, sd, 1.0)
    return np.where(mask, np.clip(z, -CLIP_TARGET, CLIP_TARGET), 0.0)


@dataclass
class Scaler:
    """Means and deviations of the training window, per input."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, mask: np.ndarray) -> Scaler:
        rows = x[mask]
        sd = rows.std(axis=0)
        return cls(rows.mean(axis=0), np.where(sd > 0, sd, 1.0))

    def apply(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Standardised and clipped, inputs last; zero where ``mask`` is false."""
        z = np.clip((x - self.mean) / self.std, -CLIP_INPUT, CLIP_INPUT)
        keep = mask.reshape(mask.shape + (1,) * (x.ndim - mask.ndim))
        return np.where(keep, np.nan_to_num(z), 0.0).astype("float32")
