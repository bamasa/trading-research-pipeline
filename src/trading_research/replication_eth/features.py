"""The original's sixteen auxiliary inputs, its volatility filter and strategy inputs.

Re-typed from the author's 2025 preprocessing, one day at a time, with the same
pandas operations in the same order, so the edge cases (a first row with no
history, a second with no trades) come out as they did there:

* ``*_lag_1_log`` is ``log((x + 1e-6) / (x.shift(1) + 1e-6))``, infinities and
  gaps set to zero and rounded to nine decimals, for the best ask and bid, the
  depth summed over the row's levels, the volume printed and the number of
  sells printed in the 100 ms.
* ``mid_lag_1_log`` is the *level* ``log(mid)`` one row back: the original
  computes a log change under that name and then overwrites it with this.
* Every input except ``book_imbalance`` and ``volatility`` is then a z-score on
  the trailing 100 rows (the row itself included), its standard deviation
  floored at 1e-9 and then increased by 1e-9; this includes the three strategy
  inputs ``rv_50``, ``mid_mean_50`` and ``spread_rolling_10``, which the
  original's gate then reads as if they were raw.
* ``volatility`` is the 50-row standard deviation of the change in the mid of
  the volume-weighted prices of one ETH on the first ten levels, each rounded
  to the 0.01 tick; it is not normalised.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: The model's sixteen auxiliary inputs, in the original's order.
AUX_FEATURES: tuple[str, ...] = (
    "ask_rate_lvl0_lag_1_log",
    "bid_rate_lvl0_lag_1_log",
    "spread",
    "mid_lag_1_log",
    "ask_depth_total_lag_1_log",
    "bid_depth_total_lag_1_log",
    "book_imbalance",
    "queue_imbalance",
    "ofi",
    "microprice_dev",
    "impact_coeff_1s",
    "return_1s",
    "trade_flow_imbalance_1s",
    "total_trade_volume_lag_1_log",
    "sell_count_lag_1_log",
    "volatility_1s",
)

#: Inputs the original's gate reads, as z-scores (see the module docstring).
STRATEGY_FEATURES: tuple[str, ...] = ("rv_50", "mid_mean_50", "spread_rolling_10")

#: Left as they are by the normalisation.
NOT_NORMALISED = ("book_imbalance", "volatility")

W_NORM, EPS_NORM, EPS_LAG = 100, 1e-9, 1e-6
PRICE_TICK, LOT, VWAP_LEVELS = 0.01, 1.0, 10


def lag_log(x: pd.Series) -> pd.Series:
    ratio = ((x + EPS_LAG) / (x.shift(1) + EPS_LAG)).to_numpy(dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = pd.Series(np.log(ratio), index=x.index)
    return out.replace([np.inf, -np.inf], 0).fillna(0).round(9)


def exec_price(prices: np.ndarray, sizes: np.ndarray, lot: float = LOT) -> np.ndarray:
    """Each row's volume-weighted price of ``lot`` on its levels, on the tick.

    A row whose levels hold less than ``lot`` gets the weighted price of all of
    them, unrounded, as in the original.
    """
    cum = np.cumsum(sizes, axis=1)
    enough = cum[:, -1] >= lot
    j = np.argmax(cum >= lot, axis=1)
    rows = np.arange(len(prices))
    before = np.where(j > 0, cum[rows, np.maximum(j - 1, 0)], 0.0)
    spent = np.cumsum(prices * sizes, axis=1)
    spent_before = np.where(j > 0, spent[rows, np.maximum(j - 1, 0)], 0.0)
    price = (spent_before + prices[rows, j] * (lot - before)) / lot
    price = np.round(price / PRICE_TICK) * PRICE_TICK
    whole = spent[:, -1] / cum[:, -1]
    return np.where(enough, price, whole)


def zscore(x: pd.Series) -> pd.Series:
    mean = x.rolling(W_NORM, min_periods=1).mean()
    std = x.rolling(W_NORM, min_periods=1).std().replace(0, EPS_NORM)
    return (x - mean) / (std + EPS_NORM)


def aux_frame(day: pd.DataFrame, depth: int) -> pd.DataFrame:
    """The day's auxiliary, volatility and strategy columns, every row, as stored."""
    a0, b0 = day["ask_price_0"], day["bid_price_0"]
    qa, qb = day["ask_size_0"], day["bid_size_0"]
    ask_total = day[[f"ask_size_{i}" for i in range(depth)]].sum(axis=1)
    bid_total = day[[f"bid_size_{i}" for i in range(depth)]].sum(axis=1)
    mid = (a0 + b0) / 2.0
    f = pd.DataFrame(index=day.index)
    f["ask_rate_lvl0_lag_1_log"] = lag_log(a0)
    f["bid_rate_lvl0_lag_1_log"] = lag_log(b0)
    f["spread"] = a0 - b0
    f["mid_lag_1_log"] = np.log(mid + 1e-12).shift(1).fillna(0)
    f["ask_depth_total_lag_1_log"] = lag_log(ask_total)
    f["bid_depth_total_lag_1_log"] = lag_log(bid_total)
    imbalance = (bid_total - ask_total) / (bid_total + ask_total)
    f["book_imbalance"] = imbalance.replace([np.inf, -np.inf], 0).fillna(0).round(9)
    f["queue_imbalance"] = (qb - qa) / (qb + qa + 1e-12)
    f["ofi"] = qb.diff().fillna(0) - qa.diff().fillna(0)
    micro = (a0 * qb + b0 * qa) / (qb + qa + 1e-12)
    f["microprice_dev"] = micro - mid
    flow = day["signed_volume"].rolling(10, min_periods=1).sum().fillna(0)
    ret_1s = mid.pct_change(periods=10).fillna(0)
    f["impact_coeff_1s"] = (ret_1s / (flow / mid + 1e-12)).clip(-10, 10)
    f["return_1s"] = ret_1s
    f["trade_flow_imbalance_1s"] = flow
    f["total_trade_volume_lag_1_log"] = lag_log(day["total_trade_volume"])
    f["sell_count_lag_1_log"] = lag_log(day["sell_count"])
    f["volatility_1s"] = mid.pct_change().fillna(0).rolling(10).std().fillna(0)
    f["rv_50"] = mid.pct_change().fillna(0).rolling(50, min_periods=1).std().fillna(0)
    f["mid_mean_50"] = mid.rolling(50, min_periods=1).mean().fillna(0)
    f["spread_rolling_10"] = f["spread"].rolling(10, min_periods=1).mean().fillna(0)
    levels = range(VWAP_LEVELS)
    ea = exec_price(
        day[[f"ask_price_{i}" for i in levels]].to_numpy(),
        day[[f"ask_size_{i}" for i in levels]].to_numpy(),
    )
    eb = exec_price(
        day[[f"bid_price_{i}" for i in levels]].to_numpy(),
        day[[f"bid_size_{i}" for i in levels]].to_numpy(),
    )
    vol = pd.Series((ea + eb) / 2.0, index=day.index).pct_change().rolling(50).std()
    for name in f.columns:
        if name not in NOT_NORMALISED:
            f[name] = zscore(f[name])
    f["volatility"] = vol.fillna(0)
    return f.fillna(0).round(9)
