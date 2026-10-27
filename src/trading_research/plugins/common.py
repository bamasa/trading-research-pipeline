"""Small pieces the built-in adapters and the conformance suite share."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

SECONDS_PER_DAY = 86_400


def grid_seconds(frame: pd.DataFrame) -> float:
    """The frame's row spacing in seconds: the median gap between labels."""
    stamps = pd.DatetimeIndex(frame["timestamp"])
    if len(stamps) < 2:
        return 1.0
    gaps = np.diff(stamps.as_unit("ns").to_numpy(dtype="datetime64[ns]").view(np.int64)) / 1e9
    return float(np.median(gaps))


def rows_per_day(frame: pd.DataFrame) -> float:
    return SECONDS_PER_DAY / grid_seconds(frame)


def days_spanned(frame: pd.DataFrame) -> float:
    """How many days the frame's rows cover, as a fraction."""
    return max(len(frame) / rows_per_day(frame), 1e-9)


def book_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Every registered book-plane feature on the frame, and their longest look-back."""
    from trading_research.features import REGISTRY, build

    names = REGISTRY.names("book")
    out = build(frame, names, plane="book")
    return out.drop(columns=["timestamp"], errors="ignore"), REGISTRY.max_lookback(names)


def forward_bp(frame: pd.DataFrame, horizon: int) -> np.ndarray:
    """The mid's log move over the next ``horizon`` rows, bp; NaN where it runs out."""
    mid = frame["mid"].to_numpy(dtype="float64")
    out = np.full(len(mid), np.nan)
    if horizon < len(mid):
        out[:-horizon] = np.log(mid[horizon:] / mid[:-horizon]) * 1e4
    return out


def finite_rows(x: pd.DataFrame) -> np.ndarray:
    return np.asarray(np.isfinite(x.to_numpy(dtype="float64")).all(axis=1))


def predict_rows(model: Any, x: pd.DataFrame) -> np.ndarray:
    """``predict_proba`` on the rows with every input present; NaN rows elsewhere."""
    ok = finite_rows(x)
    out = np.full((len(x), 3), np.nan)
    if ok.any():
        out[ok] = np.asarray(model.predict_proba(x.loc[ok]), dtype="float64")
    return out
