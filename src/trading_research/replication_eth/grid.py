"""A day of the venue's book and prints on the original's 100 ms grid.

The original's rows are snapshots stamped on whole 100 ms boundaries, each one
carrying the trades printed since the previous snapshot. The pipeline's replay
(:func:`trading_research.data.bybit.reconstruct`) emits at most one row per
100 ms, stamped by the update that produced it. Here every boundary takes the
latest replayed row stamped at or before it, so a row never shows a book the
venue had not yet published, and the prints stamped in (previous boundary,
boundary].

The original's trade side is +1 for a seller crossing and -1 for a buyer (its
first printed trade sits at the bid with side 1.0), the opposite of the
pipeline's ``aggressor``; the signed volume below keeps the original's sign.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from trading_research.market_making.events import day_start_ns, to_ns

#: The original's snapshot interval.
GRID_NS = 100_000_000

#: Boundaries in a day, 00:00:00.1 to 23:59:59.9.
BOUNDARIES = 863_999

#: Levels a side kept for the book totals. The original's rows are not printed
#: with their length; twenty is the depth of the venue's 100 ms partial-book
#: stream, which matches rows stamped on every 100 ms boundary.
DEPTH = 20

#: Per-boundary trade aggregates the features read.
TRADE_COLUMNS = ("trade_count", "total_trade_volume", "sell_count", "signed_volume")


def level_columns(depth: int = DEPTH) -> list[str]:
    """The replay's level columns, ask and bid prices and sizes, best first."""
    names = []
    for side in ("ask", "bid"):
        for kind in ("price", "size"):
            names += [f"{side}_{kind}_{level}" for level in range(depth)]
    return names


def boundaries(day: date) -> np.ndarray:
    """The day's 100 ms boundaries, in nanoseconds since the epoch."""
    return day_start_ns(day) + GRID_NS * np.arange(1, BOUNDARIES + 1, dtype=np.int64)


def trade_aggregates(prints: pd.DataFrame, day: date) -> pd.DataFrame:
    """Prints summed per boundary: count, volume, sells and signed volume.

    A print stamped exactly on a boundary belongs to that boundary's row.
    """
    start = day_start_ns(day)
    stamps = to_ns(prints["timestamp"])
    slot = -((start - stamps) // GRID_NS)  # ceil((stamp - start) / GRID_NS)
    keep = (slot >= 1) & (slot <= BOUNDARIES)
    slot = slot[keep] - 1
    size = prints["size"].to_numpy(dtype=np.float64)[keep]
    aggressor = prints["aggressor"].to_numpy(dtype=np.int64)[keep]
    n = BOUNDARIES
    return pd.DataFrame(
        {
            "trade_count": np.bincount(slot, minlength=n).astype(np.float64),
            "total_trade_volume": np.bincount(slot, weights=size, minlength=n),
            "sell_count": np.bincount(slot, weights=(aggressor < 0) * 1.0, minlength=n),
            "signed_volume": np.bincount(slot, weights=-aggressor * size, minlength=n),
        }
    )


def grid_day(
    book: pd.DataFrame, prints: pd.DataFrame, day: date, depth: int = DEPTH
) -> pd.DataFrame:
    """One row per boundary that has a book: ``ts`` (ns), levels and trade sums.

    Boundaries before the day's first replayed row are dropped.
    """
    stamps = to_ns(book["timestamp"])
    if len(stamps) > 1 and np.any(np.diff(stamps) < 0):
        raise ValueError("the replayed book must be in time order")
    grid = boundaries(day)
    source = np.searchsorted(stamps, grid, side="right") - 1
    have = source >= 0
    columns = level_columns(depth)
    levels = book[columns].to_numpy(dtype=np.float64)[source[have]]
    out = pd.DataFrame(levels, columns=columns)
    out.insert(0, "ts", grid[have])
    trades = trade_aggregates(prints, day).loc[have].reset_index(drop=True)
    return pd.concat([out, trades], axis=1)
