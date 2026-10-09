"""Walking the visible book for one clip.

A taker order for a clip larger than the touch fills level by level: the touch
first, then the next price, until the clip is done. Its price is the
volume-weighted average of what it took. The walked-book label
(:mod:`trading_research.labels.taker_vwap`) and the event-time taker executor
must price an order identically, so both call :func:`walk`; a label priced one
way and fills priced another would make the label's edge a property of the
mismatch.

What the visible book cannot fill
---------------------------------
A clip larger than the ten visible levels is reported in ``exceeds``, and its
remainder is charged at the last visible level's price. The executor skips such
an entry and counts it; an exit cannot be skipped, so it pays that price and is
counted too (the simulator's ``flatten_beyond_visible_book``). The function
reports and charges; the policy belongs to the caller.

Units
-----
Inputs are the book's own float64 prices and sizes. A non-positive or
non-finite price is refused rather than walked: it is either a feed fault, which
the day's validation excludes, or a normalised column that reached a cost, which
the K-units rule voids.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

BP: float = 1e4

Side = Literal["ask", "bid"]


@dataclass(frozen=True)
class Walk:
    """The result of walking one side of each snapshot for a clip."""

    #: Volume-weighted price paid (asks) or received (bids) per unit of the clip.
    vwap: np.ndarray
    #: The clip was larger than the visible size; the rest was charged at the
    #: last visible level's price.
    exceeds: np.ndarray
    #: Levels the clip reached, 1 for a fill at the touch.
    levels: np.ndarray


def walk(prices: np.ndarray, sizes: np.ndarray, clip: float) -> Walk:
    """Walk each row of ``prices``/``sizes`` (best level first) for ``clip``."""
    if not clip > 0:
        raise ValueError(f"clip must be positive, got {clip}")
    prices = np.atleast_2d(np.asarray(prices, dtype="float64"))
    sizes = np.atleast_2d(np.asarray(sizes, dtype="float64"))
    if prices.shape != sizes.shape:
        raise ValueError(f"prices {prices.shape} and sizes {sizes.shape} differ in shape")
    if not np.all(np.isfinite(prices)) or np.any(prices <= 0):
        raise ValueError("a walked price is non-positive or not finite; raw book prices only")
    if not np.all(np.isfinite(sizes)) or np.any(sizes < 0):
        raise ValueError("a walked size is negative or not finite")

    cumulative = np.cumsum(sizes, axis=1)
    before = cumulative - sizes
    take = np.clip(clip - before, 0.0, sizes)
    visible = cumulative[:, -1]
    remainder = np.maximum(clip - visible, 0.0)
    notional = (take * prices).sum(axis=1) + remainder * prices[:, -1]
    reached = (before < clip).sum(axis=1)
    return Walk(vwap=notional / clip, exceeds=remainder > 0, levels=reached.astype(np.int64))


def book_side(rows: pd.DataFrame, side: Side, depth: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """The ``depth`` visible levels of one side as (prices, sizes) arrays."""
    prices = np.column_stack([rows[f"{side}_price_{i}"].to_numpy("float64") for i in range(depth)])
    sizes = np.column_stack([rows[f"{side}_size_{i}"].to_numpy("float64") for i in range(depth)])
    return prices, sizes


def walk_rows(rows: pd.DataFrame, side: Side, clip: float, depth: int = 10) -> Walk:
    """Walk one side of every row: ``ask`` to buy the clip, ``bid`` to sell it."""
    prices, sizes = book_side(rows, side, depth)
    return walk(prices, sizes, clip)


def walk_cost_bp(rows: pd.DataFrame, clip: float, depth: int = 10) -> np.ndarray:
    """``w_t``: buying the clip and selling it back on the same snapshot, in bp of mid.

    The spread plus what the clip pays beyond the touch on both sides, before
    fees. On a crossed snapshot the round trip would earn money, which no
    order could; the cost there is NaN, and a gate reading it stays closed.
    """
    buy = walk_rows(rows, "ask", clip, depth).vwap
    sell = walk_rows(rows, "bid", clip, depth).vwap
    mid = (rows["bid_price_0"].to_numpy("float64") + rows["ask_price_0"].to_numpy("float64")) / 2
    cost = (buy - sell) / mid * BP
    return np.where(cost < 0, np.nan, cost)
