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


def clock_offset_profile(
    events: DayEvents, deltas_ms: Sequence[int] = tuple(range(-50, 51, 5))
) -> pd.DataFrame:
    """How well the prints explain the touch's size changes, per clock shift.

    The simulator assumes a snapshot reflects every print stamped at or before
    it. If book rows are stamped later than the matches they reflect, a print
    lands before a snapshot that does not show it yet (the queue model carries
    such prints forward); if earlier, a snapshot shows prints stamped after it,
    which would be look-ahead.

    For each shift ``delta`` every print is moved by ``delta`` milliseconds and,
    over snapshot intervals in which the touch price did not move, the change
    of the touch size is compared with the prints at the touch inside the
    interval. The shift with the smallest mean squared residual is the
    effective offset: positive means the book lags the prints, negative that it
    leads them, zero what the simulator assumes. Read it on a development day
    only.
    """
    book_ts = events.book_ts
    rows = []
    planes = (
        (1, events.bid_px[:, 0], events.bid_sz[:, 0]),
        (-1, events.ask_px[:, 0], events.ask_sz[:, 0]),
    )
    for side, price, size in planes:
        still = price[1:] == price[:-1]
        hitting = events.trade_aggressor == -side
        trade_ts = events.trade_ts[hitting]
        trade_px = events.trade_px[hitting]
        trade_sz = events.trade_sz[hitting]
        for delta in deltas_ms:
            shifted = trade_ts + int(delta) * 1_000_000
            # Interval k covers (book_ts[k - 1], book_ts[k]].
            interval = np.searchsorted(book_ts, shifted, side="left")
            inside = (interval >= 1) & (interval < len(book_ts))
            k = interval[inside]
            at_touch = trade_px[inside] == price[k]
            traded = np.bincount(
                k[at_touch], weights=trade_sz[inside][at_touch], minlength=len(book_ts)
            )[1:]
            residual = (size[:-1] - traded - size[1:])[still]
            rows.append(
                {
                    "side": side,
                    "delta_ms": int(delta),
                    "mean_sq_residual": float(np.mean(residual**2)) if len(residual) else np.nan,
                    "share_negative": float(np.mean(residual < -1e-12))
                    if len(residual)
                    else np.nan,
                }
            )
    frame = pd.DataFrame(rows)
    return frame.groupby("delta_ms", as_index=False)[["mean_sq_residual", "share_negative"]].mean()


def best_clock_offset_ms(profile: pd.DataFrame) -> int:
    """The shift with the smallest residual in a :func:`clock_offset_profile`."""
    best = int(np.argmin(profile["mean_sq_residual"].to_numpy(dtype=np.float64)))
    return int(profile["delta_ms"].to_numpy(dtype=np.int64)[best])
