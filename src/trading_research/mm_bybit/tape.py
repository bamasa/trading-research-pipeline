"""One day as the quoters and the engine see it: prints, proxies, rows, volatility.

Built once per day from the day's prints and settlements, then shared by every
cell. Prices are kept twice: in USDT and as integer ticks of the day's own tick
(``events.derive_tick`` over the day's distinct prices), so the fill rule's
"strictly below the order's price" is an integer comparison.

* The **touch proxies** are the spot study's: after each print, the price of the
  last buyer-initiated print (the ask proxy) and of the last seller-initiated
  one (the bid proxy), NaN until that side has printed.
* The **rows** are the spot study's print grid (``spot/grid.grid_stamps_ns``,
  100 ms): a row's state is after every print stamped at or before it.
* The **volatility** is the simulator's per-second form on the mid proxy: the
  mid proxy at each whole second, one-second log returns, an EWMA of their
  squares (60 s half-life), scaled to a minute, in bp.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from trading_research.market_making.events import (
    NS_PER_DAY,
    NS_PER_S,
    day_start_ns,
    derive_lot,
    derive_tick,
    to_ns,
)
from trading_research.spot.fills import PrintTape
from trading_research.spot.grid import GRID_MS, grid_stamps_ns

SECONDS = 86_400


@dataclass(frozen=True)
class DayTape:
    day: date
    start_ns: int
    tick: float
    lot: float
    ts: np.ndarray  # int64 ns, non-decreasing
    price: np.ndarray  # float64 USDT
    px: np.ndarray  # int64 ticks
    size: np.ndarray  # float64 BTC
    agg: np.ndarray  # int64, +1 buyer-initiated, -1 seller-initiated
    last_buy: np.ndarray  # float64 ticks after each print (the ask proxy), NaN before
    last_sell: np.ndarray  # float64 ticks after each print (the bid proxy), NaN before
    row_ts: np.ndarray  # int64 decision stamps
    row_end: np.ndarray  # int64 index of the last print stamped at or before each row
    sec_mid: np.ndarray  # mid proxy (USDT) at each whole second 0..86400
    sec_vol: np.ndarray  # one-minute volatility (bp) at each whole second
    sec_spread: np.ndarray  # proxy spread (ticks) at each whole second
    funding_ts: np.ndarray
    funding_rate: np.ndarray
    prints: PrintTape  # the day's prints for the flatten's side-correct rule

    def index_at_or_before(self, t: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.ts, np.asarray(t, dtype=np.int64), side="right") - 1

    def mid_at(self, t: np.ndarray, *, strictly_before: bool = False) -> np.ndarray:
        """The mid proxy (USDT) from prints stamped at or before (or before) each time."""
        stamps = np.asarray(t, dtype=np.int64)
        if strictly_before:
            idx = np.searchsorted(self.ts, stamps, side="left") - 1
        else:
            idx = np.searchsorted(self.ts, stamps, side="right") - 1
        out = np.full(len(idx), np.nan)
        ok = idx >= 0
        out[ok] = (self.last_buy[idx[ok]] + self.last_sell[idx[ok]]) * 0.5 * self.tick
        return out

    def second_of(self, t: int) -> int:
        return min(max((t - self.start_ns) // NS_PER_S, 0), SECONDS)

    def max_gap_s(self, first_s: float = 600.0, last_s: float = 86_340.0) -> float:
        """The longest gap between consecutive prints stamped in the window."""
        lo, hi = self.start_ns + int(first_s * NS_PER_S), self.start_ns + int(last_s * NS_PER_S)
        inside = self.ts[(self.ts >= lo) & (self.ts < hi)]
        if len(inside) < 2:
            return float("inf")
        return float(np.diff(inside).max()) / NS_PER_S


def _forward(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The last value where ``mask`` holds, at or before each position (NaN before)."""
    idx = np.where(mask, np.arange(len(values)), -1)
    idx = np.maximum.accumulate(idx) if len(idx) else idx
    out = np.full(len(values), np.nan)
    ok = idx >= 0
    out[ok] = values[idx[ok]]
    return out


def per_second(
    ts: np.ndarray, mid: np.ndarray, start_ns: int, half_life_s: float = 60.0
) -> tuple[np.ndarray, np.ndarray]:
    """The mid proxy at each whole second and the volatility read there."""
    boundaries = start_ns + np.arange(SECONDS + 1, dtype=np.int64) * NS_PER_S
    idx = np.searchsorted(ts, boundaries, side="right") - 1
    sec_mid = np.full(len(boundaries), np.nan)
    ok = idx >= 0
    sec_mid[ok] = mid[idx[ok]]
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.diff(np.log(sec_mid), prepend=np.nan)
    variance = pd.Series(returns**2).ewm(halflife=half_life_s, adjust=False, ignore_na=True).mean()
    return sec_mid, np.sqrt(variance.to_numpy() * 60.0) * 1e4


def build(
    day: date,
    prints: pd.DataFrame,
    funding_ts: np.ndarray,
    funding_rate: np.ndarray,
    *,
    grid_ms: int = GRID_MS,
    half_life_s: float = 60.0,
) -> DayTape:
    """The day's tape from its prints (the contract, in time order) and settlements."""
    ts = to_ns(prints["timestamp"])
    start = day_start_ns(day)
    if len(ts) == 0 or ts[0] < start or ts[-1] >= start + NS_PER_DAY:
        raise ValueError(f"{day}: prints must be non-empty and stamped on the day")
    price = prints["price"].to_numpy(dtype=np.float64)
    size = prints["size"].to_numpy(dtype=np.float64)
    agg = prints["aggressor"].to_numpy(dtype=np.int64)
    tick = derive_tick(price)
    px = np.round(price / tick).astype(np.int64)
    last_buy = _forward(px.astype(np.float64), agg > 0)
    last_sell = _forward(px.astype(np.float64), agg < 0)
    mid = (last_buy + last_sell) * 0.5 * tick
    sec_mid, sec_vol = per_second(ts, mid, start, half_life_s)
    boundaries = start + np.arange(SECONDS + 1, dtype=np.int64) * NS_PER_S
    idx = np.searchsorted(ts, boundaries, side="right") - 1
    spread = np.full(len(boundaries), np.nan)
    ok = idx >= 0
    spread[ok] = last_buy[idx[ok]] - last_sell[idx[ok]]
    row_ts = grid_stamps_ns(prints, grid_ms)
    return DayTape(
        day, start, float(tick), float(derive_lot(size)), ts, price, px, size, agg,
        last_buy, last_sell, row_ts, np.searchsorted(ts, row_ts, side="right") - 1,
        sec_mid, sec_vol, spread,
        np.asarray(funding_ts, dtype=np.int64), np.asarray(funding_rate, dtype=np.float64),
        PrintTape.from_frame(prints, day),
    )  # fmt: skip
