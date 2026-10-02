"""Markouts and the profit decomposition, computed after the loop.

Nothing here feeds back into a decision: the simulator finishes the day first,
and these read its fills against the book that followed them. That separation
is what lets a markout look forward without the simulator ever doing so.

Markouts
--------
Per fill, ``side * (mid(t + h) - p) / p * 1e4`` in basis points, with ``mid``
the last snapshot at or before ``t + h``. Positive means the fill was worth
having ``h`` seconds later. The market-wide benchmark scores every print in the
tape the same way from its resting side, independently of the simulator, so a
strategy's markouts can be compared with what resting at all would have earned.

The decomposition
-----------------
Per fill, ``spread = side * (mid_ref - p) * size`` with ``mid_ref`` the last
snapshot mid at or before the fill, and ``adverse = side * (mid(t + H) -
mid_ref) * size`` at the decomposition horizon ``H``. ``inventory`` is what is
left of the gross trading profit — the mark-to-market of inventory held beyond
``H`` — so ``spread + adverse + inventory - fees + funding`` is the net by
construction, and the simulator checks that net against the cash. Only
``spread + adverse`` together (the ``H``-second markout) is free of the
book's hundred-millisecond staleness; the split between them is indicative.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from trading_research.market_making.events import NS_PER_S, DayEvents


def book_mid(events: DayEvents) -> np.ndarray:
    """Mid of every snapshot, in price units."""
    return (events.bid_px[:, 0] + events.ask_px[:, 0]) * (0.5 * events.spec.tick)


def mid_at(book_ts: np.ndarray, mid: np.ndarray, at_ns: np.ndarray) -> np.ndarray:
    """Mid of the last snapshot at or before each time; NaN before the first."""
    at = np.asarray(at_ns, dtype=np.int64)
    index = np.searchsorted(book_ts, at, side="right") - 1
    out = np.full(len(at), np.nan)
    known = index >= 0
    out[known] = mid[index[known]]
    return out


def horizon_column(horizon_s: float) -> str:
    return f"markout_{horizon_s:g}s"


def markouts(
    fills: pd.DataFrame, book_ts: np.ndarray, mid: np.ndarray, horizons_s: Sequence[float]
) -> pd.DataFrame:
    """``fills`` with one markout column per horizon, in basis points."""
    out = fills.copy()
    ts = fills["ts"].to_numpy(dtype=np.int64)
    side = fills["side"].to_numpy(dtype=np.float64)
    price = fills["price"].to_numpy(dtype=np.float64)
    for horizon in horizons_s:
        later = mid_at(book_ts, mid, ts + round(horizon * NS_PER_S))
        out[horizon_column(horizon)] = side * (later - price) / price * 1e4
    return out


def decompose(
    fills: pd.DataFrame,
    *,
    book_ts: np.ndarray,
    mid: np.ndarray,
    horizon_s: float,
    final_position: float,
    final_mark: float,
    fees: float,
    funding: float,
) -> dict[str, float]:
    """Split a day's net into spread, adverse, inventory, fees and funding."""
    ts = fills["ts"].to_numpy(dtype=np.int64)
    side = fills["side"].to_numpy(dtype=np.float64)
    price = fills["price"].to_numpy(dtype=np.float64)
    size = fills["size"].to_numpy(dtype=np.float64)
    reference = fills["mid_ref"].to_numpy(dtype=np.float64)
    later = mid_at(book_ts, mid, ts + round(horizon_s * NS_PER_S))

    spread = float(np.sum(side * (reference - price) * size))
    adverse = float(np.sum(side * (later - reference) * size))
    held = final_position * final_mark if final_position != 0.0 else 0.0
    gross = float(np.sum(-side * price * size)) + held
    inventory = gross - spread - adverse
    return {
        "spread": spread,
        "adverse": adverse,
        "inventory": inventory,
        "fees": fees,
        "funding": funding,
        "gross": gross,
        "making": spread + adverse - fees,
        "net": gross - fees + funding,
    }


def summarise_markouts(fills: pd.DataFrame, horizons_s: Sequence[float]) -> pd.DataFrame:
    """Volume-weighted markouts by fill path, with fill counts and volume."""
    rows = []
    for path, group in fills.groupby("path", sort=True):
        weights = group["size"].to_numpy(dtype=np.float64)
        row: dict[str, object] = {
            "path": path,
            "fills": len(group),
            "volume": float(weights.sum()),
        }
        for horizon in horizons_s:
            values = group[horizon_column(horizon)].to_numpy(dtype=np.float64)
            ok = np.isfinite(values)
            row[horizon_column(horizon)] = (
                float(np.average(values[ok], weights=weights[ok])) if ok.any() else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def market_wide_markouts(events: DayEvents, horizons_s: Sequence[float]) -> pd.DataFrame:
    """Every print scored from its resting side, volume-weighted, per horizon.

    The benchmark a strategy's own markouts are read against: what being the
    passive side of the tape's prints was worth, with no simulator involved.
    """
    mid = book_mid(events)
    ts = events.trade_ts
    price = events.trade_px * events.spec.tick
    resting = -events.trade_aggressor.astype(np.float64)
    size = events.trade_sz
    rows = []
    for horizon in horizons_s:
        later = mid_at(events.book_ts, mid, ts + round(horizon * NS_PER_S))
        value = resting * (later - price) / price * 1e4
        ok = np.isfinite(value)
        rows.append(
            {
                "horizon_s": float(horizon),
                "markout_bp": float(np.average(value[ok], weights=size[ok]))
                if ok.any()
                else np.nan,
                "prints": int(ok.sum()),
                "volume": float(size[ok].sum()),
            }
        )
    return pd.DataFrame(rows)
