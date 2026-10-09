"""Synthetic one-day books for the search engine's tests.

Rows come about every 100 ms with a few milliseconds of jitter, in stretches
that cover the day's edges and a funding settlement: the first minutes after
midnight (entries open at 00:01), the minutes around 08:00, a stretch with a
feed gap, and the last minutes before the 23:59 flatten. Sizes are drawn so
that the clip sometimes goes beyond the visible book.
"""

from __future__ import annotations

from datetime import date

import numpy as np

from trading_research.backtest.taker import TakerBook
from trading_research.market_making.events import NS_PER_S, day_start_ns

DAY = date(2024, 2, 11)
CLIP = 0.010
TICK = 0.1


def _stretch(rng: np.random.Generator, start_s: float, seconds: float) -> np.ndarray:
    rows = int(seconds * 10)
    base = np.round(start_s * NS_PER_S) + np.arange(rows) * 100_000_000
    return (base + rng.integers(0, 5, rows) * 1_000_000).astype(np.int64)


def make_book(
    seed: int = 0,
    *,
    depth: int = 5,
    thin_share: float = 0.03,
    funding_rate: float = 1e-4,
    day: date = DAY,
    late_stretch: bool = True,
) -> TakerBook:
    """One day's synthetic book with stretches at 00:00, 08:00, 12:00 and, unless
    ``late_stretch`` is false (the book then ends at 12:02), 23:57."""
    rng = np.random.default_rng(seed)
    stretches = [(20.0, 160.0), (8 * 3600 - 90.0, 200.0), (12 * 3600, 60.0),
                 (12 * 3600 + 75.0, 60.0), (23 * 3600 + 57 * 60 + 30, 125.0)]  # fmt: skip
    if not late_stretch:
        stretches = stretches[:-1]
    offsets = np.concatenate([_stretch(rng, a, b) for a, b in stretches])
    ts = day_start_ns(day) + offsets
    n = len(ts)
    mid_ticks = 500_000 + np.cumsum(rng.integers(-1, 2, n))
    half = rng.integers(1, 3, n)
    steps = np.arange(depth)
    bid_px = (mid_ticks - half)[:, None] - steps[None, :]
    ask_px = (mid_ticks + half)[:, None] + steps[None, :]
    sizes = np.round(rng.exponential(0.02, (2, n, depth)) + 0.001, 3)
    thin = rng.random((2, n)) < thin_share
    sizes[thin] = 0.001
    funding_ts = np.array([day_start_ns(day) + h * 3600 * NS_PER_S for h in (0, 8, 16)])
    return TakerBook(
        day,
        ts,
        bid_px * TICK,
        sizes[0],
        ask_px * TICK,
        sizes[1],
        funding_ts.astype(np.int64),
        np.full(3, funding_rate),
    )


def decisions(book: TakerBook, count: int, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """``count`` decision rows in time order and a random side for each."""
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(len(book.ts), size=count, replace=False))
    return rows.astype(np.int64), rng.choice(np.array([-1, 1]), size=count)
