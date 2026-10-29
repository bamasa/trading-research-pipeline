"""Every pair among the symbols, judged on the formation window only.

For each ordered pair ``(a, b)`` (``a`` before ``b`` in the order the symbols
were given; ``a`` is regressed on ``b``): the correlation of bar log returns,
the hedge ratio, the Engle-Granger t and p-value, the half-life of the
residual, and the residual's standard deviation set against the round trip of
both legs (four taker fills). A residual whose typical deviation is smaller
than what a round trip costs cannot pay for itself however reliably it reverts.

Many pairs are tested at once, so the p-values are corrected across all of
them by Holm's step-down procedure (family-wise level ``alpha``); a pair
*passes* only on its Holm-adjusted p-value. Pairs are ranked: passing first,
then by adjusted p-value, then by the Engle-Granger t (more negative first).

The screen reads the window it is handed and nothing else. A pair that
cointegrates on its formation window is a candidate, not a result: the
backtest (:mod:`.backtest`) is what trades it on the bars after.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pandas as pd

from trading_research.pairs.costs import LegCosts
from trading_research.pairs.stats import (
    DEFAULT_LAGS,
    DEFAULT_SIMS,
    PairsError,
    engle_granger,
    half_life,
    holm_adjusted,
    return_correlation,
)

#: Family-wise level of the screen.
ALPHA = 0.05

COLUMNS = (
    "rank",
    "a",
    "b",
    "nobs",
    "correlation",
    "beta",
    "eg_t",
    "critical_5",
    "p_value",
    "p_holm",
    "passes",
    "half_life_bars",
    "spread_sd_bp",
    "round_trip_cost_bp",
    "sd_over_cost",
)


def screen(
    log_closes: pd.DataFrame,
    costs: LegCosts,
    *,
    alpha: float = ALPHA,
    lags: int = DEFAULT_LAGS,
    sims: int = DEFAULT_SIMS,
) -> pd.DataFrame:
    """One row per pair of columns of ``log_closes`` (the formation window), ranked."""
    names = list(log_closes.columns)
    if len(names) < 2:
        raise PairsError(f"a screen needs two symbols or more, got {names}")
    rows = []
    for a, b in combinations(names, 2):
        la = log_closes[a].to_numpy(dtype=np.float64)
        lb = log_closes[b].to_numpy(dtype=np.float64)
        try:
            test = engle_granger(la, lb, lags=lags, sims=sims)
        except PairsError as exc:
            raise PairsError(f"{a} on {b}: {exc}") from exc
        beta = test.hedge.beta
        sd_bp = test.hedge.residual_sd * 1e4
        cost = costs.round_trip_bp(beta)
        rows.append(
            {
                "a": a,
                "b": b,
                "nobs": test.nobs,
                "correlation": return_correlation(la, lb),
                "beta": beta,
                "eg_t": test.t,
                "critical_5": test.critical_5,
                "p_value": test.p_value,
                "half_life_bars": half_life(test.hedge.residual(la, lb)),
                "spread_sd_bp": sd_bp,
                "round_trip_cost_bp": cost,
                "sd_over_cost": sd_bp / cost if cost > 0 else math.inf,
            }
        )
    table = pd.DataFrame(rows)
    table["p_holm"] = holm_adjusted(table["p_value"].to_numpy())
    table["passes"] = table["p_holm"] <= alpha
    table = table.sort_values(
        ["passes", "p_holm", "eg_t"], ascending=[False, True, True], kind="stable"
    ).reset_index(drop=True)
    table["rank"] = np.arange(1, len(table) + 1)
    return table.loc[:, list(COLUMNS)]
