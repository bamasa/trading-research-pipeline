"""One row per (month, candidate coin): eleven features from before the month, and its funding.

Every feature of month *m* reads data strictly before *m*'s first hour; the
target is the funding a day over *m* itself, in bp, the quantity the carry's
rules rank on (``strategies/carry.scores``).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from trading_research.strategies.carry import Market

FEATURES = (
    "f1", "f3", "f_trend", "f_sd", "f_pos", "basis_1", "basis_end",
    "ret_1", "ret_3", "vol_1", "rank_f3",
)  # fmt: skip
#: Months of history a row needs.
HISTORY = 3

Readable = Callable[[str, pd.Period], bool]


def _bounds(first: pd.Period, last: pd.Period) -> tuple[pd.Timestamp, pd.Timestamp]:
    a = first.start_time.tz_localize("UTC")
    b = (last + 1).start_time.tz_localize("UTC")
    return a, b


def _within(s: pd.Series, a: pd.Timestamp, b: pd.Timestamp) -> pd.Series:
    return s[(s.index >= a) & (s.index < b)]


def funding_a_day(mk: Market, first: pd.Period, last: pd.Period) -> float:
    """Funding a day, bp, from the start of ``first`` to the end of ``last``."""
    a, b = _bounds(first, last)
    days = (b - a).days
    return float(_within(mk.funding, a, b).sum()) * 1e4 / days


def row(mk: Market, month: pd.Period) -> dict[str, float]:
    """The features of ``month`` (all but ``rank_f3``) and its target."""
    m1, m3 = month - 1, month - HISTORY
    a1, b = _bounds(m1, m1)
    a3, _ = _bounds(m3, m1)
    rates3 = _within(mk.funding, a3, b)
    daily = rates3.groupby(pd.DatetimeIndex(rates3.index).floor("1D")).sum() * 1e4
    spot, perp = _within(mk.spot, a1, b), _within(mk.perp, a1, b)
    basis = (perp / spot.reindex(perp.index) - 1).dropna() * 1e4
    spot3 = _within(mk.spot, a3, b)
    logret = np.log(spot).diff().dropna()
    f1, f3 = funding_a_day(mk, m1, m1), funding_a_day(mk, m3, m1)
    return {
        "f1": f1,
        "f3": f3,
        "f_trend": f1 - f3,
        "f_sd": float(daily.std(ddof=1)) if len(daily) > 1 else np.nan,
        "f_pos": float((rates3 > 0).mean()) if len(rates3) else np.nan,
        "basis_1": float(basis.mean()) if len(basis) else np.nan,
        "basis_end": float(basis[basis.index >= b - np.timedelta64(24, "h")].mean())
        if len(basis)
        else np.nan,
        "ret_1": float(np.log(spot.iloc[-1] / spot.iloc[0])) if len(spot) > 1 else np.nan,
        "ret_3": float(np.log(spot3.iloc[-1] / spot3.iloc[0])) if len(spot3) > 1 else np.nan,
        "vol_1": float(logret.std(ddof=1) * 1e4) if len(logret) > 1 else np.nan,
        "target": funding_a_day(mk, month, month),
    }


def table(markets: dict[str, Market], months: pd.PeriodIndex, readable: Readable) -> pd.DataFrame:
    """Every candidate's row for every month: readable in the month and its history."""
    rows = []
    for month in months:
        for coin, mk in markets.items():
            history = [month - k for k in range(1, HISTORY + 1)]
            if readable(coin, month) and all(readable(coin, m) for m in history):
                rows.append({"month": month, "coin": coin, **row(mk, month)})
    out = pd.DataFrame(rows)
    out["rank_f3"] = out.groupby("month")["f3"].rank(pct=True)
    return out.dropna(subset=list(FEATURES)).reset_index(drop=True)
