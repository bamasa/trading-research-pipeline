"""Prints aligned backward onto the replayed book's rows.

The replayed book (:func:`trading_research.data.bybit.reconstruct`) has event-time
rows: a row is stamped with the update that opened its 100 ms interval, and
carries the book as it stood right after that update. A print belongs to the
**first row stamped at or after it**, so row *t* holds the prints in
``(s_{t-1}, s_t]``. Everything in that interval had happened when the row was
stamped, and nothing after it had.

:func:`trading_research.features.flow.align_to_grid` uses the other rule: a
print goes to the last row stamped at or before it. On a regular grid whose
rows close intervals that is harmless; on these rows it gives row *t* the
prints after its own stamp, up to the next row's, which a decision at *t*
could not have seen. It is kept for the older grids it was written for, and the
study's flow features are built from :func:`bucket_prints` below instead. The
print-cut leakage test (:mod:`trading_research.validation.leakage_timed`)
tells the two rules apart.

Prints carry the Bybit contract of :mod:`trading_research.data.bybit_trades`:
``timestamp``, ``price``, ``size`` and ``aggressor`` (+1 a buyer crossed, -1 a
seller did).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_research.features.timed import stamps_ns

BP: float = 1e4


def print_stamps_ns(prints: pd.DataFrame) -> np.ndarray:
    """Print stamps as int64 nanoseconds, checked non-decreasing."""
    return stamps_ns(prints)


def print_rows(prints: pd.DataFrame, stamps: np.ndarray) -> np.ndarray:
    """The row each print belongs to: the first stamped at or after it.

    A print after the last row belongs to no row yet, and gets ``len(stamps)``.
    """
    return np.searchsorted(stamps, print_stamps_ns(prints), side="left").astype(np.int64)


def prints_through(stamps: np.ndarray, print_stamps: np.ndarray) -> np.ndarray:
    """For each row, how many prints are stamped at or before it."""
    return np.searchsorted(print_stamps, stamps, side="right").astype(np.int64)


def bucket_prints(prints: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Per-row sums of the prints each row owns, under the backward rule.

    The columns are those of :func:`trading_research.features.flow.align_to_grid`,
    so :func:`trading_research.features.flow.build_flow_features` runs on the
    result unchanged.
    """
    stamps = stamps_ns(rows)
    slot = print_rows(prints, stamps)
    inside = slot < len(stamps)
    slot = slot[inside]
    aggressor = prints["aggressor"].to_numpy(dtype="float64")[inside]
    size = prints["size"].to_numpy(dtype="float64")[inside]
    price = prints["price"].to_numpy(dtype="float64")[inside]
    notional = size * price

    n = len(stamps)
    buy = aggressor > 0
    columns: dict[str, np.ndarray] = {}
    columns["buy_volume"] = np.bincount(slot[buy], weights=size[buy], minlength=n)
    columns["sell_volume"] = np.bincount(slot[~buy], weights=size[~buy], minlength=n)
    columns["trade_count"] = np.bincount(slot, minlength=n).astype("float64")
    columns["signed_notional"] = np.bincount(slot, weights=aggressor * notional, minlength=n)
    columns["traded_notional"] = np.bincount(slot, weights=notional, minlength=n)
    largest = np.zeros(n)
    np.maximum.at(largest, slot, notional)
    columns["largest_trade"] = largest

    volume = columns["buy_volume"] + columns["sell_volume"]
    mid = (rows["bid_price_0"].to_numpy(dtype="float64") + rows["ask_price_0"].to_numpy()) / 2.0
    active = volume > 0
    deviation = np.zeros(n)
    vwap = columns["traded_notional"][active] / volume[active]
    deviation[active] = (vwap / mid[active] - 1.0) * BP
    columns["vwap_deviation_bp"] = deviation
    return pd.DataFrame(columns, index=rows.index)
