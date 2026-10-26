"""One day's model inputs, filter, strategy inputs and labels, as arrays on disk.

A day keeps the rows the original kept, 200 to N-200 of its grid: the
auxiliary inputs drop 200 rows at each end and the book inputs 100. Its arrays:

* ``ts``: the row's boundary, nanoseconds since the epoch;
* ``x``: ``(rows, 56)`` float32, the sixteen auxiliary inputs then the forty
  book inputs, in the original's order;
* ``volatility``: the filter's input (a row below 2e-5 has no label);
* ``rel50`` and ``rel100``: the labels' relative change at 50 and 100 rows
  (5 s and 10 s), NaN where a side lacks the rows;
* ``rv_50``, ``mid_mean_50``, ``spread_rolling_10``: the gate's inputs, as the
  original stored them (z-scores);
* ``mid``: the touch mid, for reporting only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_research.replication_eth.features import AUX_FEATURES, aux_frame
from trading_research.replication_eth.lob import LEVELS, START, lob_features, relative_change

#: Rows dropped at each end of a day, as in the original.
TRIM = 200

#: The volatility filter: a row below it carries no label.
MIN_VOLATILITY = 2e-5

#: Arrays a stored day carries.
KEYS = (
    "ts",
    "x",
    "volatility",
    "rel50",
    "rel100",
    "rv_50",
    "mid_mean_50",
    "spread_rolling_10",
    "mid",
)


def _book(frame: pd.DataFrame, side: str, kind: str) -> np.ndarray:
    return frame[[f"{side}_{kind}_{i}" for i in range(LEVELS)]].to_numpy(dtype=np.float64)


def build_day(frame: pd.DataFrame, *, depth: int) -> dict[str, np.ndarray]:
    """The stored arrays of one gridded day (:func:`.grid.grid_day`'s output)."""
    n = len(frame)
    if n <= 2 * TRIM + 1:
        raise ValueError(f"a day of {n} rows is too short")
    aux = aux_frame(frame, depth)
    book = lob_features(
        _book(frame, "ask", "price"),
        _book(frame, "ask", "size"),
        _book(frame, "bid", "price"),
        _book(frame, "bid", "size"),
    )
    keep = np.arange(TRIM, n - TRIM)
    mid = ((frame["ask_price_0"] + frame["bid_price_0"]) / 2.0).to_numpy()
    x = np.hstack([aux[list(AUX_FEATURES)].to_numpy()[keep], book[keep - START]])
    return {
        "ts": frame["ts"].to_numpy(dtype=np.int64)[keep],
        "x": x.astype(np.float32),
        "volatility": aux["volatility"].to_numpy()[keep],
        "rel50": relative_change(mid, 50)[keep],
        "rel100": relative_change(mid, 100)[keep],
        "rv_50": aux["rv_50"].to_numpy()[keep],
        "mid_mean_50": aux["mid_mean_50"].to_numpy()[keep],
        "spread_rolling_10": aux["spread_rolling_10"].to_numpy()[keep],
        "mid": mid[keep],
    }


def save_day(arrays: dict[str, np.ndarray], path: Path | str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.stem + ".part.npz")
    stored: dict[str, Any] = {key: arrays[key] for key in KEYS}
    np.savez(partial, **stored)
    partial.replace(target)
    return target


def load_day(path: Path | str) -> dict[str, np.ndarray]:
    with np.load(Path(path)) as stored:
        return {key: stored[key] for key in KEYS}
