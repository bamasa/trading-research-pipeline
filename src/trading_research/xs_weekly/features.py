"""The universe each day, the twelve features, the four series and the target, from a :class:`Panel`.

Day *t*'s decision is at its 00:00 UTC open; every input reads bars up to day
*t - 1* alone. The target runs from day *t*'s open to day *t + 7*'s, or to the
coin's last close before it if it stops trading, minus the universe's mean.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading_research.xs_weekly.panel import Panel

UNIVERSE, MIN_HISTORY, VOLUME_DAYS = 100, 120, 30
HOLD, WINDOW = 7, 60
TABULAR = (
    "rel_1", "rel_7", "rel_28", "rel_60", "vol_7", "vol_28", "vol_60",
    "mkt_7", "mkt_28", "volume_7_60", "funding_7", "funding_28",
)  # fmt: skip
SERIES = ("rel_1d", "mkt_1d", "volume", "funding")
REL_7, REL_28, VOL_28 = TABULAR.index("rel_7"), TABULAR.index("rel_28"), TABULAR.index("vol_28")


def _past(x: np.ndarray, k: int, how: str) -> np.ndarray:
    """A rolling statistic over the ``k`` days ending the day before."""
    r = getattr(pd.DataFrame(x).rolling(k, min_periods=k), how)()
    return np.asarray(r.shift(1).to_numpy(), dtype="float64")


def _row_mean(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    v = np.where(mask & np.isfinite(x), x, 0.0)
    n = (mask & np.isfinite(x)).sum(axis=1)
    return np.where(n > 0, v.sum(axis=1) / np.maximum(n, 1), np.nan)


@dataclass
class Frame:
    universe: np.ndarray  # (D, C) bool
    tab: np.ndarray  # (D, C, 12)
    series: np.ndarray  # (D, C, 4): day d's values, read at d + 1 onward
    target: np.ndarray  # (D, C) relative log return, open d to open d + 7
    gross: np.ndarray  # (D, C) the coin's own log return over the same week
    funding_week: np.ndarray  # (D, C) funding bp over days d .. d + 6


def universe(p: Panel) -> np.ndarray:
    history = pd.DataFrame(np.isfinite(p.close)).cumsum().shift(1).fillna(0).to_numpy()
    volume = _past(np.nan_to_num(p.quote_volume), VOLUME_DAYS, "mean")
    ok = (history >= MIN_HISTORY) & np.isfinite(p.open) & np.isfinite(volume)
    ranked = pd.DataFrame(np.where(ok, volume, np.nan)).rank(axis=1, ascending=False).to_numpy()
    return ok & (ranked <= UNIVERSE)


def exit_price(p: Panel) -> np.ndarray:
    """Day d + 7's open, or the last close before it if the coin stopped trading."""
    d = len(p.days)
    last_close = pd.DataFrame(p.close).ffill().to_numpy()
    out = np.full_like(p.open, np.nan)
    out[: d - HOLD] = np.where(
        np.isfinite(p.open[HOLD:]), p.open[HOLD:], last_close[HOLD - 1 : d - 1]
    )
    return out


def build(p: Panel) -> Frame:
    uni = universe(p)
    logc = np.log(p.close)
    r = np.vstack([np.full((1, logc.shape[1]), np.nan), np.diff(logc, axis=0)])
    mkt = _row_mean(r, uni)
    rel = r - mkt[:, None]
    mkt_col = np.repeat(mkt[:, None], r.shape[1], axis=1)
    lv = np.log(np.where(p.quote_volume > 0, p.quote_volume, np.nan))
    tab = np.stack(
        [_past(rel, 1, "sum"), _past(rel, 7, "sum"), _past(rel, 28, "sum"), _past(rel, 60, "sum"),
         _past(rel, 7, "std"), _past(rel, 28, "std"), _past(rel, 60, "std"),
         _past(mkt_col, 7, "sum"), _past(mkt_col, 28, "sum"),
         _past(lv, 7, "mean") - _past(lv, 60, "mean"),
         _past(p.funding_bp, 7, "mean"), _past(p.funding_bp, 28, "mean")],
        axis=-1,
    )  # fmt: skip
    lv60 = pd.DataFrame(lv).rolling(WINDOW, min_periods=20).mean().to_numpy()
    series = np.stack([rel, mkt_col, lv - lv60, p.funding_bp], axis=-1)
    gross = np.log(exit_price(p) / p.open)
    target = gross - _row_mean(gross, uni)[:, None]
    fw = pd.DataFrame(np.nan_to_num(p.funding_bp)).rolling(HOLD).sum().shift(-(HOLD - 1)).to_numpy()
    return Frame(uni, tab, series, target, gross, fw)


def windows(frame: Frame, day: int, coins: np.ndarray) -> np.ndarray:
    """(len(coins), WINDOW, 4): each coin's series over the WINDOW days before ``day``."""
    w = frame.series[day - WINDOW : day][:, coins]  # (W, n, 4)
    return np.nan_to_num(np.transpose(w, (1, 0, 2)))
