"""The choosers' replays through the carry search's engine, and the numbers C1 and C2 read."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from trading_research.carry_ml.features import Readable
from trading_research.market_making.verdicts import t_sf
from trading_research.pairs.stats import holm_adjusted
from trading_research.strategies.carry import Market, Rule, replay

#: One coin, the whole account, no floor: the journal's rule's shape.
ONE_COIN = Rule(slots=1, lookback=3, threshold_bp_a_day=0.0)
PLACEBOS = 19


def picks(scored: pd.DataFrame) -> dict[pd.Period, str]:
    """Each month's highest-scored coin; ties go to the coin listed first."""
    best = scored.loc[scored.groupby("month")["score"].idxmax()]
    return dict(zip(best["month"], best["coin"], strict=True))


def run(
    markets: dict[str, Market],
    months: pd.PeriodIndex,
    chosen: dict[pd.Period, str],
    readable: Readable,
) -> pd.DataFrame:
    """The carry holding ``chosen[month]`` each month: each month's return on the account."""
    return replay(markets, months, ONE_COIN, readable, chooser=lambda m: [chosen[m]])


def placebo(
    markets: dict[str, Market],
    months: pd.PeriodIndex,
    rows: pd.DataFrame,
    readable: Readable,
    seed: int,
) -> pd.DataFrame:
    """The same carry on a coin drawn at random each month from the month's candidates."""
    rng = np.random.default_rng(seed)
    chosen = {m: str(rng.choice(sorted(rows.loc[rows["month"] == m, "coin"]))) for m in months}
    return run(markets, months, chosen, readable)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return math.nan
    ra = np.argsort(np.argsort(a, kind="stable"), kind="stable")
    rb = np.argsort(np.argsort(b, kind="stable"), kind="stable")
    return float(np.corrcoef(ra, rb)[0, 1])


def forecast_quality(scored: pd.DataFrame, rows: pd.DataFrame) -> pd.Series:
    """Each month's rank correlation across candidates of the score and the realised funding."""
    joined = scored.merge(rows[["month", "coin", "target"]], on=["month", "coin"])
    out = {
        month: spearman(g["score"].to_numpy(), g["target"].to_numpy())
        for month, g in joined.groupby("month")
    }
    return pd.Series(out, dtype="float64")


def one_sided(x: pd.Series | np.ndarray) -> dict[str, float]:
    """Mean, *t* against zero and the one-sided *p* of mean > 0, months as observations."""
    v = np.asarray(x, dtype="float64")
    v = v[np.isfinite(v)]
    n = len(v)
    sd = float(v.std(ddof=1)) if n > 1 else math.nan
    t = float(v.mean() / (sd / math.sqrt(n))) if n > 1 and sd > 0 else math.nan
    return {"mean": float(v.mean()), "t": t, "p": t_sf(t, n - 1) if n > 1 else math.nan,
            "months": n}  # fmt: skip


def holm(p: dict[str, float]) -> dict[str, float]:
    names = list(p)
    return dict(zip(names, map(float, holm_adjusted([p[k] for k in names])), strict=True))
