"""The print grid and the touch proxies made of prints.

**The print grid** (the registration, "Common settings"; S25-PF, S25-PB,
S25-PFT): one row per 100 ms interval that holds at least one print, stamped
at its first print and carrying the state right after every print with that
stamp; later prints of the interval belong to the next row; an interval
without a print has no row. Intervals are ``[k * 100 ms, (k + 1) * 100 ms)``
from the epoch, so midnight starts one, as ``data/bybit.reconstruct`` cuts
the book grid (``stamp // grid_ms``). Prints are aligned backward, as in
2025: row ``t`` owns the prints in ``(s_{t-1}, s_t]``, which is
:func:`trading_research.features.prints.bucket_prints`, reused unchanged.

**The touch proxies**: the ask proxy is the price of the last
buyer-initiated print stamped at or before a moment, the bid proxy the last
seller-initiated one's, the mid proxy their mean. Every print counts, those
flagged ``rpi`` included (they stay in the features; only the fill rule skips
them). A proxy is NaN until its side has printed in the prints given, so a
day's grid built with the previous day's tail carries its proxies across
midnight and a cold day's does not.

A row's proxies are state, as a book row's levels are: they may come from a
print older than any window, and a feature reads them from the row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_research.features.prints import print_stamps_ns

GRID_MS: int = 100
PROXY_COLUMNS: tuple[str, ...] = ("ask_proxy", "bid_proxy", "mid_proxy")


def grid_stamps_ns(prints: pd.DataFrame, grid_ms: int = GRID_MS) -> np.ndarray:
    """The print grid's row stamps: the first print stamp of each occupied interval."""
    if grid_ms <= 0:
        raise ValueError(f"grid_ms must be positive, got {grid_ms}")
    ts = print_stamps_ns(prints)
    if not len(ts):
        return np.zeros(0, dtype=np.int64)
    bins = ts // (grid_ms * 1_000_000)
    first = np.ones(len(ts), dtype=bool)
    first[1:] = bins[1:] != bins[:-1]
    return np.asarray(ts[first], dtype=np.int64)


def proxies_at(
    ts: np.ndarray, price: np.ndarray, side: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(ask, bid, mid)`` proxies right after every print stamped at or before each target."""
    out = []
    for aggressor in (1, -1):
        mine = np.flatnonzero(np.asarray(side) == aggressor)
        j = np.searchsorted(np.asarray(ts)[mine], targets, side="right") - 1
        value = np.full(len(targets), np.nan)
        known = j >= 0
        value[known] = np.asarray(price, dtype=np.float64)[mine[j[known]]]
        out.append(value)
    ask, bid = out
    return ask, bid, (ask + bid) / 2.0


def touch_proxies(prints: pd.DataFrame, stamps_ns: np.ndarray) -> pd.DataFrame:
    """The three proxies at each stamp, from every print given."""
    ask, bid, mid = proxies_at(
        print_stamps_ns(prints),
        prints["price"].to_numpy(dtype=np.float64),
        prints["aggressor"].to_numpy(dtype=np.int64),
        np.asarray(stamps_ns, dtype=np.int64),
    )
    return pd.DataFrame({"ask_proxy": ask, "bid_proxy": bid, "mid_proxy": mid})


def print_grid(prints: pd.DataFrame, grid_ms: int = GRID_MS) -> pd.DataFrame:
    """The grid's rows: ``timestamp`` and the touch proxies, one row per occupied interval.

    ``prints`` follow the print contract, in archive order; they may span
    midnight (a day with the previous day's tail), and the caller keeps the
    day's rows.
    """
    stamps = grid_stamps_ns(prints, grid_ms)
    frame = touch_proxies(prints, stamps)
    frame.insert(0, "timestamp", pd.to_datetime(stamps, unit="ns", utc=True))
    return frame
