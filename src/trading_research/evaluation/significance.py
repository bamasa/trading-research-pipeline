"""How many observations there actually were.

This module exists because of a specific mistake, made in this project on 21
August 2026 and caught only by data that had not been used to make it.

A cross-sectional strategy was measured on 3,383 trades across 26 instruments
over 14 days. Treating each trade as an observation gives a mean of +6.89 basis
points with a *t* of 4.12, which reads as settled. It was not settled. The
instruments carried the same market-wide bet -- their per-day results correlate
at +0.47 -- so 26 instruments on one day are closer to one observation than to
26, and 14 days of them are 14 observations rather than 3,383. Aggregated that
way the mean is +6.18 with a *t* of **1.18**, which reads as eight heads in
fourteen coin flips. The strategy was then run on the following six weeks and
lost money on every instrument.

The arithmetic was never in doubt; the unit of observation was. Both figures
below are computed for anything this project reports from now on, and the
smaller one is the one to believe.

What clustering does to a standard error
----------------------------------------
For *n* observations with average pairwise correlation *r*, the effective count
is approximately

    n_eff = n / (1 + (n - 1) * r)

At r = 0, all n count. At r = 1, one does. The 26 instruments here sit at
r = 0.47, giving about 2 independent bets per day rather than 26 -- and since
the days themselves are what vary, the honest denominator is the number of days.

Why not just use the day-level figure and stop
-----------------------------------------------
Because the two disagreeing is itself the diagnostic. A strategy whose trade-level
and cluster-level statistics agree is taking genuinely independent bets. One
where they diverge by a factor of three is taking the same bet repeatedly, and
that is worth knowing before the position is sized, not after.

Accepting and rejecting are not symmetric
------------------------------------------
A cluster-level statistic this conservative will rarely establish that something
works, and that is the intended asymmetry. Accepting a strategy requires
significance; rejecting one requires only that it fail a condition stated in
advance. When the reversion candidate was run on fresh data it lost on 26
instruments of 26 with a cluster-level *t* of just -1.73 — not a significant
loss — and it was refuted anyway, because a median at or below zero was written
down as its kill condition before the test. Demanding significance to reject as
well as to accept is how a strategy survives on the strength of nobody being
able to prove it does not work.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class SignificanceError(ValueError):
    """The sample cannot support the statistic asked for."""


@dataclass(frozen=True)
class Significance:
    """A result, measured both ways."""

    #: Observations if every row counts.
    trades: int
    trade_mean_bp: float
    trade_t: float
    #: Observations after grouping rows that share a bet.
    clusters: int
    cluster_mean_bp: float
    cluster_t: float
    #: Mean pairwise correlation between series within a cluster.
    correlation: float
    #: Independent bets per cluster, from that correlation.
    effective_per_cluster: float
    positive_clusters: int

    @property
    def inflation(self) -> float:
        """How much treating trades as independent overstates the evidence."""
        if self.cluster_t == 0 or not np.isfinite(self.cluster_t):
            return float("nan")
        return float(self.trade_t / self.cluster_t)

    @property
    def verdict(self) -> str:
        """What the cluster-level figure supports, in one line.

        The threshold is 3 rather than the customary 2. With a couple of dozen
        clusters and returns as fat-tailed as these, a *t* of 2 is reached often
        enough by chance that it settles nothing -- as this project established
        by reaching one and then losing money for six weeks.
        """
        if not np.isfinite(self.cluster_t):
            return "not enough clusters to say anything"
        if abs(self.cluster_t) < 2.0:
            return (
                f"indistinguishable from chance ({self.positive_clusters}/{self.clusters} positive)"
            )
        if abs(self.cluster_t) < 3.0:
            return "suggestive, and this project has been wrong at exactly this level"
        return "significant at the cluster level"

    def to_dict(self) -> dict[str, float]:
        return {
            "trades": float(self.trades),
            "trade_mean_bp": self.trade_mean_bp,
            "trade_t": self.trade_t,
            "clusters": float(self.clusters),
            "cluster_mean_bp": self.cluster_mean_bp,
            "cluster_t": self.cluster_t,
            "correlation_within_cluster": self.correlation,
            "effective_bets_per_cluster": self.effective_per_cluster,
            "positive_clusters": float(self.positive_clusters),
            "inflation": self.inflation,
        }


def _t_statistic(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if len(finite) < 2:
        return float("nan"), float("nan")
    error = float(np.std(finite, ddof=1) / np.sqrt(len(finite)))
    mean = float(np.mean(finite))
    return mean, (mean / error if error > 0 else float("nan"))


def mean_pairwise_correlation(panel: pd.DataFrame) -> float:
    """Average correlation between the columns of a cluster-by-series table.

    NaN when fewer than two series have any variation — an honest answer, since
    a single series has no correlation with anything and a constant one has none
    with anything either.
    """
    usable = panel.loc[:, panel.std() > 0]
    if usable.shape[1] < 2:
        return float("nan")
    matrix = usable.corr().to_numpy()
    upper = matrix[np.triu_indices_from(matrix, k=1)]
    finite = upper[np.isfinite(upper)]
    return float(np.mean(finite)) if len(finite) else float("nan")


def effective_count(n: int, correlation: float) -> float:
    """Independent observations among ``n`` correlated ones."""
    if n < 1:
        return 0.0
    if not np.isfinite(correlation):
        return float(n)
    # Correlation below zero would inflate the count, which is not a claim this
    # is willing to make from a noisy estimate.
    r = max(0.0, min(float(correlation), 0.999))
    return float(n / (1.0 + (n - 1) * r))


def assess(
    trades: pd.DataFrame,
    *,
    value: str = "net_bp",
    cluster: str = "day",
    series: str | None = "symbol",
) -> Significance:
    """Measure a result by trade and by cluster, and report both.

    ``cluster`` is what makes results dependent — a day, for a strategy whose
    bet is market-wide; an episode or a week for something slower. Choosing it
    is a judgement about the strategy, and getting it wrong in the permissive
    direction is what this module exists to prevent.

    ``series`` names the column whose members share a bet within a cluster,
    used only to report how correlated they are. Pass ``None`` when there is
    one series.
    """
    for column in (value, cluster):
        if column not in trades.columns:
            raise SignificanceError(f"{column!r} is not a column: {list(trades.columns)[:8]}")
    if trades.empty:
        raise SignificanceError("no trades to assess")

    trade_mean, trade_t = _t_statistic(trades[value].to_numpy(dtype="float64"))

    by_cluster = trades.groupby(cluster)[value].mean()
    if len(by_cluster) < 2:
        raise SignificanceError(
            f"{len(by_cluster)} cluster(s) of {cluster!r}: nothing can be said about a single one"
        )
    cluster_mean, cluster_t = _t_statistic(by_cluster.to_numpy(dtype="float64"))

    correlation = float("nan")
    if series is not None and series in trades.columns:
        panel = trades.pivot_table(index=cluster, columns=series, values=value, aggfunc="mean")
        correlation = mean_pairwise_correlation(panel)
        members = panel.shape[1]
    else:
        members = 1

    return Significance(
        trades=len(trades),
        trade_mean_bp=trade_mean,
        trade_t=trade_t,
        clusters=len(by_cluster),
        cluster_mean_bp=cluster_mean,
        cluster_t=cluster_t,
        correlation=correlation,
        effective_per_cluster=effective_count(members, correlation),
        positive_clusters=int((by_cluster > 0).sum()),
    )
