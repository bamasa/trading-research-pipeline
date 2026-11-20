"""Walk-forward folds, the legs a ranking trades, and the daily numbers read from them.

A fold fits on 14 days and scores the next 7. Its training rows are every
fifth bar whose exit falls inside the training days, so no target reaches into
the scored days; its scored rows are every tenth bar of the scored days. The
inputs are standardised with the training rows' means and deviations.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from trading_research.market_making.verdicts import t_sf
from trading_research.pairs.stats import holm_adjusted
from trading_research.xs_deep.features import (
    DECIDE_EVERY,
    ENTRY_LAG,
    HOLD,
    TRAIN_EVERY,
    WINDOW,
    Frame,
    Scaler,
    zscore_rows,
)
from trading_research.xs_deep.rankers import Batch, make

TRAIN_DAYS, TEST_DAYS = 14, 7
#: Instruments long and short at each decision time.
LEGS = 3
#: A decision time is used when at least this many instruments are usable.
MIN_INSTRUMENTS = 10


def folds(days: list[date]) -> list[tuple[list[date], list[date]]]:
    """(training days, scored days) of each fold within one contiguous span."""
    out, i = [], 0
    while i + TRAIN_DAYS < len(days):
        out.append((days[i : i + TRAIN_DAYS], days[i + TRAIN_DAYS : i + TRAIN_DAYS + TEST_DAYS]))
        i += TEST_DAYS
    return out


def rows(
    times: pd.DatetimeIndex, usable: np.ndarray, days: list[date], every: int, *, train: bool
) -> np.ndarray:
    """Bars of ``days`` on the ``every``-minute clock with enough usable instruments."""
    day = np.asarray((times - np.timedelta64(1, "m")).date)
    minute = np.asarray(times.hour * 60 + times.minute)
    t = np.arange(len(times))
    wanted = np.array(days, dtype=object)
    keep = np.isin(day, wanted) & (minute % every == 0) & (t >= WINDOW - 1)
    keep &= usable.sum(axis=1) >= MIN_INSTRUMENTS
    if train:
        exit_bar = np.minimum(t + ENTRY_LAG + HOLD, len(times) - 1)
        keep &= np.isin(day[exit_bar], wanted) & (t + ENTRY_LAG + HOLD < len(times))
    return t[keep]


def batch(frame: Frame, at: np.ndarray, usable: np.ndarray, tab: Scaler, seq: Scaler) -> Batch:
    mask = usable[at]
    return Batch(
        tab=tab.apply(frame.tab[at], mask),
        seq=seq.apply(np.moveaxis(frame.windows(at), 2, 3), mask),
        mask=mask,
        y=zscore_rows(frame.target[at], mask),
    )


def fold(
    name: str, frame: Frame, times: pd.DatetimeIndex, fit_days: list[date], score_days: list[date]
) -> tuple[np.ndarray, np.ndarray]:
    """One fold: the scored bars and the ranker's scores there, NaN where unusable."""
    usable = frame.usable()
    fit_at = rows(times, usable, fit_days, TRAIN_EVERY, train=True)
    score_at = rows(times, usable, score_days, DECIDE_EVERY, train=False)
    fit_mask = usable[fit_at]
    tab = Scaler.fit(frame.tab[fit_at], fit_mask)
    # The series' statistics at the training bars stand for their windows'.
    seq = Scaler.fit(frame.series[fit_at], fit_mask)
    ranker = make(name)
    ranker.fit(batch(frame, fit_at, usable, tab, seq))
    scored = batch(frame, score_at, usable, tab, seq)
    return score_at, np.where(scored.mask, ranker.score(scored), np.nan)


def decisions(frame: Frame, at: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    """Per decision time: the coefficient, and the mean gross, cost and net of its legs."""
    out = []
    for t, s in zip(at, scores, strict=True):
        ok = np.isfinite(s)
        idx = np.flatnonzero(ok)
        order = idx[np.argsort(s[ok], kind="stable")]
        short, long_ = order[:LEGS], order[-LEGS:]
        side = np.r_[-np.ones(LEGS), np.ones(LEGS)]
        legs = np.r_[short, long_]
        gross = side * frame.fwd[t, legs]
        cost = frame.cost[t, legs]
        ic = spearman(s[ok], frame.target[t, ok])
        out.append((t, float(ic), gross.mean(), cost.mean(), (gross - cost).mean()))
    return pd.DataFrame(out, columns=["bar", "ic", "gross", "cost", "net"])


def one_sided(daily: pd.Series) -> dict[str, float]:
    """Mean of a daily series, its t over days, and the one-sided p of mean > 0."""
    x = daily.dropna().to_numpy(dtype="float64")
    n = len(x)
    sd = x.std(ddof=1) if n > 1 else float("nan")
    t = x.mean() / (sd / np.sqrt(n)) if n > 1 and sd > 0 else float("nan")
    p = t_sf(float(t), n - 1) if np.isfinite(t) else float("nan")
    return {"mean": float(x.mean()), "t": float(t), "p": p, "days": n}


def holm(p: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values by name."""
    names = list(p)
    return dict(zip(names, map(float, holm_adjusted([p[k] for k in names])), strict=True))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation; ties are broken by position, which continuous scores do not have."""
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a, kind="stable"), kind="stable").astype("float64")
    rb = np.argsort(np.argsort(b, kind="stable"), kind="stable").astype("float64")
    return float(np.corrcoef(ra, rb)[0, 1])
