"""Synthetic spot prints for the spot executor's tests, and the checks of the fixture itself.

One day of prints in stretches that cover the day's edges: the minutes after
midnight (entries open at 00:01), a busy stretch, a thin stretch whose prints
come seconds apart (late prints), a stretch in which only sellers trade (so a
short's exit waits and busy periods depend on the side), and the last minutes
before and after the 23:59 flatten. Some prints share a stamp (a sweep), some
are flagged unreachable, and sizes are drawn so the clip sometimes exceeds a
sweep. :func:`grid_of` builds the print grid's row stamps: one row per 100 ms
interval holding a print, stamped at its first print, by ``spot/grid.py``.
The prints pass through the print contract (``spot/prints.py``), as the loader
gives them, and become the rule's tape by ``PrintTape.from_frame``.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from trading_research.market_making.events import NS_PER_S, day_start_ns
from trading_research.spot import prints as contract
from trading_research.spot.fills import PrintTape
from trading_research.spot.grid import grid_stamps_ns

DAY = date(2026, 6, 3)
CLIP = 0.010
#: (start s after midnight, seconds, mean gap s, share of buys)
STRETCHES = (
    (5.0, 400.0, 0.4, 0.5),
    (6 * 3600.0, 900.0, 0.15, 0.5),
    (9 * 3600.0, 600.0, 3.0, 0.5),
    (12 * 3600.0, 240.0, 0.3, 0.0),
    (12 * 3600.0 + 240.0, 300.0, 0.3, 0.5),
    (23 * 3600 + 56 * 60.0, 239.0, 0.5, 0.5),
)


def make_frame(seed: int = 0, *, day: date = DAY, unreachable_share: float = 0.02) -> pd.DataFrame:
    """One synthetic day of prints in archive order, in the print contract."""
    rng = np.random.default_rng(seed)
    stamps, sides = [], []
    for start, seconds, gap, buys in STRETCHES:
        count = int(seconds / gap)
        t = start + np.cumsum(rng.exponential(gap, count))
        t = t[t < start + seconds]
        stamps.append(np.round(t * 1e3).astype(np.int64) * 1_000_000)
        sides.append(np.where(rng.random(len(t)) < buys, 1, -1))
    ts, side = np.concatenate(stamps), np.concatenate(sides)
    swept = rng.random(len(ts)) < 0.15  # a print repeated at its stamp: a sweep
    ts, side = np.repeat(ts, 1 + swept), np.repeat(side, 1 + swept)
    order = np.argsort(ts, kind="stable")
    ts, side = ts[order] + day_start_ns(day), side[order]
    mid = 60_000.0 + np.cumsum(rng.normal(0.0, 1.5, len(ts)))
    price = np.round(mid + side * rng.integers(1, 4, len(ts)) * 0.01, 2)
    size = np.round(rng.exponential(0.006, len(ts)) + 0.0005, 4)
    rpi = rng.random(len(ts)) < unreachable_share
    return contract.assemble(pd.Series(pd.to_datetime(ts, utc=True)), pd.Series(price),
                             pd.Series(size), side.astype(np.int8), pd.Series(np.arange(len(ts))),
                             rpi, contract.BINANCE_SOURCE)  # fmt: skip


def make_prints(seed: int = 0, *, day: date = DAY, unreachable_share: float = 0.02) -> PrintTape:
    """The same day as the print rule's tape."""
    return PrintTape.from_frame(make_frame(seed, day=day, unreachable_share=unreachable_share), day)


def grid_of(prints: PrintTape) -> np.ndarray:
    """The print grid's row stamps (``spot/grid.py``): each 100 ms interval's first print."""
    return grid_stamps_ns(pd.DataFrame({"timestamp": pd.to_datetime(prints.ts, utc=True)}))


def test_the_fixture_covers_what_the_executor_must_meet() -> None:
    prints = make_prints()
    rows = grid_of(prints)
    start = day_start_ns(DAY)
    assert rows[0] < start + 60 * NS_PER_S and rows[-1] > start + (23 * 3600 + 59 * 60) * NS_PER_S
    assert np.all(np.diff(rows) > 0) and np.isin(rows, prints.ts).all()
    assert (np.diff(prints.ts) == 0).sum() > 100  # sweeps
    assert prints.rpi.sum() > 10
    interval = (prints.ts - start) // 100_000_000  # the grid counted from midnight
    assert np.array_equal(rows, prints.ts[np.r_[True, interval[1:] != interval[:-1]]])
    seller_only = (prints.ts >= start + 12 * 3600 * NS_PER_S) & (
        prints.ts < start + (12 * 3600 + 240) * NS_PER_S
    )
    assert seller_only.sum() > 100 and (prints.side[seller_only] == -1).all()
    assert prints.size.min() > 0 and (prints.size < CLIP).mean() > 0.5
