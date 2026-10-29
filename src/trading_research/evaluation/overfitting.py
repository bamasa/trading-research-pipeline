"""What a search's best result is worth, given how many results it looked at.

Reported beside every number the development period produces, gating nothing:

* **N_eff**, the number of independent trials among the N scored, from the
  mean pairwise correlation of the trials' daily series
  (:func:`effective_trials`, on ``significance.effective_count``);
* the **deflated Sharpe ratio** (Bailey and Lopez de Prado, 2014): the
  probability that a daily series' Sharpe ratio is above the maximum expected
  from N_eff trials of no skill, corrected for the series' skewness and
  kurtosis (:func:`deflated_sharpe_ratio`);
* the **probability of backtest overfitting** by combinatorially symmetric
  cross-validation (Bailey, Borwein, Lopez de Prado and Zhu, 2017): split the
  days into blocks, choose on every half of the blocks, and count how often
  the choice ranks below the median on the other half (:func:`cscv_pbo`).

Sharpe ratios here are per day, not annualised; the trial count and the
variance of Sharpe ratios across trials are in the same unit.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd

from trading_research.evaluation.significance import effective_count, mean_pairwise_correlation

EULER_GAMMA = 0.5772156649015329
_NORMAL = NormalDist()


def effective_trials(n_trials: int, panel: pd.DataFrame) -> float:
    """N_eff = ``effective_count(n_trials, mean_pairwise_correlation(panel))``,
    ``panel`` being days x trials (the daily G of every trial)."""
    return effective_count(n_trials, mean_pairwise_correlation(panel))


def sharpe_ratios(panel: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Each column's per-day Sharpe ratio (mean over sample sd), NaN where it
    has fewer than two finite days or no dispersion."""
    values = np.asarray(panel, dtype=np.float64)
    finite = np.isfinite(values)
    n = finite.sum(axis=0)
    mean = np.where(finite, values, 0.0).sum(axis=0) / np.maximum(n, 1)
    centred = np.where(finite, values - mean, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        sd = np.sqrt((centred**2).sum(axis=0) / (n - 1))
        out = mean / sd
    return np.where((n >= 2) & np.isfinite(out), out, np.nan)


def sharpe_variance(panel: pd.DataFrame | np.ndarray) -> float:
    """The variance across trials of their per-day Sharpe ratios (ddof 1)."""
    ratios = sharpe_ratios(panel)
    ratios = ratios[np.isfinite(ratios)]
    return float(ratios.var(ddof=1)) if len(ratios) > 1 else math.nan


def expected_maximum_sharpe(n_trials: float, variance: float) -> float:
    """The expected maximum Sharpe ratio of ``n_trials`` independent trials of
    no skill whose Sharpe ratios have this variance:

        sqrt(V) * ((1 - g) * z(1 - 1/N) + g * z(1 - 1/(N e)))

    with ``z`` the standard normal quantile and ``g`` Euler's constant. One
    trial or fewer selects nothing, and the expected maximum is 0.
    """
    if not math.isfinite(variance) or variance < 0:
        return math.nan
    if n_trials <= 1:
        return 0.0
    first = _NORMAL.inv_cdf(1.0 - 1.0 / n_trials)
    second = _NORMAL.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(variance) * ((1.0 - EULER_GAMMA) * first + EULER_GAMMA * second)


@dataclass(frozen=True)
class DeflatedSharpe:
    """A daily series' Sharpe ratio against the best of N_eff null trials."""

    days: int
    sharpe: float
    skewness: float
    #: Kurtosis, not excess: 3 for a normal series.
    kurtosis: float
    n_trials: float
    #: The expected maximum Sharpe ratio under no skill.
    benchmark: float
    #: The probability that the true Sharpe ratio exceeds the benchmark.
    dsr: float


def deflated_sharpe_ratio(
    values: np.ndarray | pd.Series | list[float], n_trials: float, variance: float
) -> DeflatedSharpe:
    """The deflated Sharpe ratio of a daily series.

    ``Phi((SR - SR0) * sqrt(T - 1) / sqrt(1 - skew * SR + (kurt - 1) / 4 * SR^2))``
    with ``SR0 = expected_maximum_sharpe(n_trials, variance)``. Moments are the
    series' own (population skewness and kurtosis). NaN with fewer than three
    days, no dispersion, or a non-positive denominator.
    """
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    n = len(array)
    benchmark = expected_maximum_sharpe(n_trials, variance)
    nan = DeflatedSharpe(n, math.nan, math.nan, math.nan, n_trials, benchmark, math.nan)
    if n < 3:
        return nan
    sd = float(array.std(ddof=1))
    population = float(array.std(ddof=0))
    if sd <= 0:
        return nan
    centred = array - array.mean()
    skew = float(np.mean(centred**3) / population**3)
    kurt = float(np.mean(centred**4) / population**4)
    sharpe = float(array.mean() / sd)
    denominator = 1.0 - skew * sharpe + (kurt - 1.0) / 4.0 * sharpe**2
    if denominator <= 0 or not math.isfinite(benchmark):
        return DeflatedSharpe(n, sharpe, skew, kurt, n_trials, benchmark, math.nan)
    z = (sharpe - benchmark) * math.sqrt(n - 1) / math.sqrt(denominator)
    return DeflatedSharpe(n, sharpe, skew, kurt, n_trials, benchmark, _NORMAL.cdf(z))


#: The registered CSCV partition: 10 blocks of the 150 test days (252 splits).
CSCV_BLOCKS = 10


def cscv_splits(days: int, blocks: int = CSCV_BLOCKS) -> list[tuple[np.ndarray, np.ndarray]]:
    """Every (in-sample, out-of-sample) pair of day positions: the days cut
    into ``blocks`` contiguous blocks in calendar order, each half of the
    blocks in turn in-sample and the rest out-of-sample."""
    if blocks < 2 or blocks % 2 or blocks > days:
        raise ValueError(f"{blocks} blocks cannot split {days} days into two halves")
    parts = np.array_split(np.arange(days), blocks)
    out = []
    for chosen in itertools.combinations(range(blocks), blocks // 2):
        inside = np.concatenate([parts[i] for i in chosen])
        outside = np.concatenate([parts[i] for i in range(blocks) if i not in chosen])
        out.append((inside, outside))
    return out


def relative_rank_logit(chosen: int, scores: np.ndarray) -> float:
    """The logit of the chosen candidate's relative rank among ``scores``.

    The rank is 1 for the worst and N for the best, ties averaged; the
    relative rank is rank / (N + 1) and the logit log(w / (1 - w)), below 0
    when the choice sits below the median.
    """
    values = np.asarray(scores, dtype=np.float64)
    target = values[chosen]
    rank = float((values < target).sum()) + (float((values == target).sum()) + 1.0) / 2.0
    w = rank / (len(values) + 1.0)
    return math.log(w / (1.0 - w))


@dataclass(frozen=True)
class Pbo:
    """The probability of backtest overfitting and what it was made of."""

    pbo: float
    #: Splits with a choice; a split whose in-sample half admits none is skipped.
    splits: int
    skipped: int
    logits: np.ndarray


def cscv_pbo(
    days: int,
    choose: Callable[[np.ndarray], tuple[int | None, np.ndarray]],
    score: Callable[[np.ndarray], np.ndarray],
    blocks: int = CSCV_BLOCKS,
) -> Pbo:
    """PBO by combinatorially symmetric cross-validation.

    ``choose(in_sample_days)`` applies the selection rule and returns the
    chosen candidate and the candidates it chose among; ``score(days)`` gives
    every candidate's performance on those days. In each split the chosen
    candidate is ranked by its out-of-sample score among the same candidates.
    PBO is the share of splits whose logit is at or below 0.
    """
    logits = []
    skipped = 0
    for inside, outside in cscv_splits(days, blocks):
        chosen, candidates = choose(inside)
        if chosen is None or len(candidates) < 2:
            skipped += 1
            continue
        candidates = np.asarray(candidates, dtype=np.int64)
        position = int(np.flatnonzero(candidates == chosen)[0])
        logits.append(relative_rank_logit(position, score(outside)[candidates]))
    array = np.asarray(logits, dtype=np.float64)
    pbo = float((array <= 0).mean()) if len(array) else math.nan
    return Pbo(pbo, len(array), skipped, array)


def cscv_pbo_of_means(returns: np.ndarray, blocks: int = CSCV_BLOCKS) -> Pbo:
    """The textbook case: days x trials, the trial with the highest mean is
    chosen in-sample and ranked by its mean out-of-sample."""
    values = np.asarray(returns, dtype=np.float64)
    every = np.arange(values.shape[1])

    def choose(days: np.ndarray) -> tuple[int | None, np.ndarray]:
        return int(np.argmax(values[days].mean(axis=0))), every

    return cscv_pbo(values.shape[0], choose, lambda days: values[days].mean(axis=0), blocks)
