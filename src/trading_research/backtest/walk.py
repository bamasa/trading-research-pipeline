"""Walking a visible book for a size: the one function every taker price comes from.

A taker order of size ``q`` takes the best level first, then the next, until
``q`` is filled; its price is the volume-weighted average of what it took.
:func:`walk_book` does that for many snapshots at once, one row each, and is
the only implementation in the repository: the event-time taker executor
(:mod:`.taker`), the market-making simulator's taker orders and the walked-book
label all call it, so an entry, an exit and a label computed on the same
snapshot cannot disagree about the price.

What lies beyond the visible levels is unknown. The walk charges any size left
after the last visible level at that level's price and reports how much was
left (``beyond``); whether that is acceptable is the caller's decision. The
taker executor skips an entry that would need it and charges an exit for it, as
the simulator's flatten does.

Prices may be in any unit (integer ticks in the simulator, quote currency in a
book frame); the walk never rounds. A missing level (a non-finite price or
size, or a size of zero) is empty; the price charged beyond the book is the
deepest level's that has a price. The order of the arithmetic is fixed: take
the smaller of what is left and the level's size, add take times price to the
cost, subtract take from what is left, level by level, so the result is the
same to the last bit as a loop over one snapshot.

The label's entry points
------------------------
:func:`walk`, :func:`walk_rows` and :func:`walk_cost_bp` are the forms the
walked-book label (:mod:`trading_research.labels.taker_vwap`) and the EV gate
read. They walk with :func:`walk_book` and differ only in what they accept: a
non-positive or non-finite price is refused rather than treated as an empty
level, because on a label's input it is either a feed fault, which the day's
validation excludes, or a normalised column that reached a cost, which the
K-units rule voids. They also report how many levels the size reached.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd

from trading_research.data.schema import (
    Side,
    ask_price_col,
    ask_size_col,
    bid_price_col,
    bid_size_col,
)

BP: float = 1e4


class BookWalk(NamedTuple):
    """One walk per snapshot. Arrays of one value per row."""

    #: Volume-weighted price of the size, in the units of the prices given.
    vwap: np.ndarray
    #: Size left after the last visible level, charged at that level's price.
    beyond: np.ndarray
    #: Total visible size on the side walked.
    visible: np.ndarray


def walk_book(prices: np.ndarray, sizes: np.ndarray, size: float | np.ndarray) -> BookWalk:
    """Walk each row's levels (best first) for ``size``; one :class:`BookWalk` row per row.

    ``prices`` and ``sizes`` are ``(rows, depth)`` or one snapshot's ``(depth,)``;
    ``size`` is one positive size for every row or one per row.
    """
    px = np.asarray(prices, dtype=np.float64)
    sz = np.asarray(sizes, dtype=np.float64)
    if px.ndim == 1:
        px, sz = px[None, :], sz[None, :]
    if px.shape != sz.shape or px.ndim != 2 or px.shape[1] == 0:
        raise ValueError(f"prices and sizes must share a (rows, depth) shape, got {px.shape}")
    wanted = np.broadcast_to(np.asarray(size, dtype=np.float64), (px.shape[0],))
    if not (wanted > 0).all():
        raise ValueError("the size walked must be positive")
    priced = np.isfinite(px)
    empty = ~(priced & np.isfinite(sz) & (sz > 0))
    last_price = _deepest_price(px, priced)
    if empty.any():
        sz = np.where(empty, 0.0, sz)
        px = np.where(empty, 0.0, px)
    left = wanted.copy()
    cost = np.zeros(px.shape[0])
    for level in range(px.shape[1]):
        take = np.minimum(left, sz[:, level])
        cost += take * px[:, level]
        left -= take
    beyond = np.where(left > 0.0, left, 0.0)
    cost += beyond * last_price
    return BookWalk(cost / wanted, beyond, sz.sum(axis=1))


def _deepest_price(px: np.ndarray, priced: np.ndarray) -> np.ndarray:
    """The price of each row's deepest priced level (NaN for a side with none)."""
    depth = px.shape[1]
    if priced.all():
        return px[:, depth - 1]
    index = depth - 1 - np.argmax(priced[:, ::-1], axis=1)
    out = px[np.arange(px.shape[0]), index]
    return np.where(priced.any(axis=1), out, np.nan)


def book_arrays(
    frame: pd.DataFrame, depth: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(bid_px, bid_sz, ask_px, ask_sz)``, each ``(rows, depth)``, from a book frame."""

    def block(name: Callable[[int], str]) -> np.ndarray:
        columns = [name(level) for level in range(depth)]
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise KeyError(f"the book frame lacks {missing}")
        return frame[columns].to_numpy(dtype=np.float64)

    return block(bid_price_col), block(bid_size_col), block(ask_price_col), block(ask_size_col)


def round_trip_cost_bp(
    bid_px: np.ndarray,
    bid_sz: np.ndarray,
    ask_px: np.ndarray,
    ask_sz: np.ndarray,
    size: float | np.ndarray,
) -> np.ndarray:
    """The cost of buying ``size`` and selling it back on the same snapshot, bp of mid.

    Both legs walk the book with :func:`walk_book`; computed from prices, so it
    is in basis points whatever the book's units.
    """
    buy = walk_book(ask_px, ask_sz, size).vwap
    sell = walk_book(bid_px, bid_sz, size).vwap
    best_bid = np.atleast_2d(np.asarray(bid_px, dtype=np.float64))[:, 0]
    best_ask = np.atleast_2d(np.asarray(ask_px, dtype=np.float64))[:, 0]
    mid = 0.5 * (best_bid + best_ask)
    return (buy - sell) / mid * BP


@dataclass(frozen=True)
class Walk:
    """The label's view of walking one side of each snapshot for a clip."""

    #: Volume-weighted price paid (asks) or received (bids) per unit of the clip.
    vwap: np.ndarray
    #: The clip was larger than the visible size; the rest was charged at the
    #: last visible level's price.
    exceeds: np.ndarray
    #: Levels the clip reached, 1 for a fill at the touch.
    levels: np.ndarray


def _raw_book(prices: np.ndarray, sizes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(prices, sizes)`` as 2-d float64, refused unless they are a raw book's."""
    px = np.atleast_2d(np.asarray(prices, dtype=np.float64))
    sz = np.atleast_2d(np.asarray(sizes, dtype=np.float64))
    if px.shape != sz.shape:
        raise ValueError(f"prices {px.shape} and sizes {sz.shape} differ in shape")
    if not np.all(np.isfinite(px)) or np.any(px <= 0):
        raise ValueError("a walked price is non-positive or not finite; raw book prices only")
    if not np.all(np.isfinite(sz)) or np.any(sz < 0):
        raise ValueError("a walked size is negative or not finite")
    return px, sz


def walk(prices: np.ndarray, sizes: np.ndarray, clip: float) -> Walk:
    """Walk each row of ``prices``/``sizes`` (best level first) for ``clip``."""
    if not clip > 0:
        raise ValueError(f"clip must be positive, got {clip}")
    px, sz = _raw_book(prices, sizes)
    result = walk_book(px, sz, clip)
    before = np.cumsum(sz, axis=1) - sz
    reached = (before < clip).sum(axis=1)
    return Walk(vwap=result.vwap, exceeds=result.beyond > 0, levels=reached.astype(np.int64))


def book_side(rows: pd.DataFrame, side: Side, depth: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """The ``depth`` visible levels of one side as (prices, sizes) arrays."""
    price, size = (ask_price_col, ask_size_col) if side == "ask" else (bid_price_col, bid_size_col)
    prices = rows[[price(level) for level in range(depth)]].to_numpy(dtype=np.float64)
    sizes = rows[[size(level) for level in range(depth)]].to_numpy(dtype=np.float64)
    return prices, sizes


def walk_rows(rows: pd.DataFrame, side: Side, clip: float, depth: int = 10) -> Walk:
    """Walk one side of every row: ``ask`` to buy the clip, ``bid`` to sell it."""
    prices, sizes = book_side(rows, side, depth)
    return walk(prices, sizes, clip)


def walk_cost_bp(rows: pd.DataFrame, clip: float, depth: int = 10) -> np.ndarray:
    """``w_t``: buying the clip and selling it back on the same snapshot, in bp of mid.

    :func:`round_trip_cost_bp` on a raw book: the spread plus what the clip pays
    beyond the touch on both sides, before fees. On a crossed snapshot the round
    trip would earn money, which no order could; the cost there is NaN, and a
    gate reading it stays closed.
    """
    if not clip > 0:
        raise ValueError(f"clip must be positive, got {clip}")
    bid_px, bid_sz = _raw_book(*book_side(rows, "bid", depth))
    ask_px, ask_sz = _raw_book(*book_side(rows, "ask", depth))
    cost = round_trip_cost_bp(bid_px, bid_sz, ask_px, ask_sz, clip)
    return np.where(cost < 0, np.nan, cost)
