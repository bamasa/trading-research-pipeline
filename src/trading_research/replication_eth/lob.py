"""The original's forty normalised book inputs and its three-class labels.

Re-typed from the author's 2025 book preprocessing (ten levels a side):

* Prices: each level's distance from the best price on its side, divided by the
  standard deviation of the best price over the previous 50 rows (the row
  itself excluded), floored at half the mean spread over those rows; level 0
  is instead the best price's z-score on those 50 rows.
* Sizes: each level's z-score on the previous 100 rows, its standard deviation
  floored per column at the larger of the day's 10th percentile and 5 % of the
  day's median of that standard deviation (a floor read from the whole day, as
  in the original).
* Labels: the mean mid over the next ``k`` rows against the mean over the last
  ``k`` (the row included), relative, against ``+-alpha``: 1 down, 2 flat, 3 up.
  The change is stored and the threshold applied when a day is read.

Rows 0 to 99 have no full history and are not produced; the caller keeps rows
200 to N-200, where the auxiliary inputs and these overlap in the original.
"""

from __future__ import annotations

import numpy as np

PRICE_WINDOW, VOLUME_WINDOW, LEVELS = 50, 100, 10
START = max(PRICE_WINDOW, VOLUME_WINDOW)
EPS = 1e-12


def _trailing(x: np.ndarray, window: int, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation of ``x`` over the ``window`` rows before each row."""
    shape = (len(x) + 1, *x.shape[1:])
    s, s2 = np.zeros(shape), np.zeros(shape)
    s[1:] = np.cumsum(x, axis=0, dtype=np.float64)
    s2[1:] = np.cumsum(x * x, axis=0, dtype=np.float64)
    mean = (s[rows] - s[rows - window]) / window
    var = np.maximum((s2[rows] - s2[rows - window]) / window - mean**2, EPS)
    return mean, np.sqrt(var)


def lob_features(
    ask_px: np.ndarray, ask_sz: np.ndarray, bid_px: np.ndarray, bid_sz: np.ndarray
) -> np.ndarray:
    """``(N - 100, 40)`` float32 for rows 100 onwards: ask prices, ask sizes,
    bid prices, bid sizes, ten levels each."""
    n = len(ask_px)
    rows = np.arange(START, n)
    best_ask, best_bid = ask_px[:, 0].astype(np.float64), bid_px[:, 0].astype(np.float64)
    ask_mean, ask_std = _trailing(best_ask, PRICE_WINDOW, rows)
    bid_mean, bid_std = _trailing(best_bid, PRICE_WINDOW, rows)
    spread = best_ask - best_bid
    spread_mean, _ = _trailing(spread, PRICE_WINDOW, rows)
    positive = spread[spread > 0]
    whole_day = np.median(positive) if positive.size else 1.0
    floor = 0.5 * np.where(spread_mean > 0, spread_mean, whole_day)
    ask_std, bid_std = np.maximum(ask_std, floor), np.maximum(bid_std, floor)
    asks = (ask_px[rows] - best_ask[rows, None]) / ask_std[:, None]
    bids = (best_bid[rows, None] - bid_px[rows]) / bid_std[:, None]
    asks[:, 0] = (best_ask[rows] - ask_mean) / ask_std
    bids[:, 0] = (best_bid[rows] - bid_mean) / bid_std
    sizes = np.hstack([ask_sz, bid_sz]).astype(np.float64)
    size_mean, size_std = _trailing(sizes, VOLUME_WINDOW, rows)
    size_floor = np.maximum(np.percentile(size_std, 10, axis=0), 0.05 * np.median(size_std, axis=0))
    size_std = np.maximum(size_std, np.maximum(size_floor, 1e-6)[None, :])
    z = (sizes[rows] - size_mean) / size_std
    out = np.hstack([asks, z[:, :LEVELS], bids, z[:, LEVELS:]])
    return out.astype(np.float32)


def relative_change(mid: np.ndarray, k: int) -> np.ndarray:
    """Mean mid over the next ``k`` rows against the last ``k``, relative; NaN
    on rows without ``k`` rows on either side."""
    m = np.asarray(mid, dtype=np.float64)
    n = len(m)
    out = np.full(n, np.nan)
    t = np.arange(k - 1, n - k)
    if t.size == 0:
        return out
    c = np.zeros(n + 1)
    c[1:] = np.cumsum(m)
    past = (c[t + 1] - c[t - (k - 1)]) / k
    future = (c[t + k + 1] - c[t + 1]) / k
    out[t] = (future - past) / np.where(past != 0.0, past, 1.0)
    return out


def labels(rel: np.ndarray, alpha: float) -> np.ndarray:
    """1 down / 2 flat / 3 up against ``+-alpha``; a NaN change is flat."""
    out = np.full(len(rel), 2, dtype=np.int64)
    with np.errstate(invalid="ignore"):
        out[rel > alpha] = 3
        out[rel < -alpha] = 1
    return out
