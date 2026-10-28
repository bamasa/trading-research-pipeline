"""The statistics a pair is judged by: hedge, cointegration, half-life, correlation.

The hedge ratio and the half-life are the ones the ``spread`` strategy plugin
already fits (:func:`trading_research.strategies.spread.fit_hedge`, least
squares of one log price on the other with an intercept, and
:func:`~trading_research.strategies.spread.fit_ou`, the AR(1) of the residual);
this module calls them rather than fitting its own, so a pair screened here and
a pair traded by the plugin carry the same beta.

Engle-Granger
-------------
The residual of the hedge regression is tested for a unit root with an
augmented Dickey-Fuller regression without a constant (the intercept is
already in the hedge),

    d e(t) = rho * e(t-1) + sum_j gamma_j * d e(t-j) + u(t),

and the statistic is the t of ``rho``. Its distribution under no cointegration
is not the Dickey-Fuller one, because the residual was fitted to look
stationary. The p-value here is read from that null distribution simulated by
this same code: two independent Gaussian random walks of the same length, the
same hedge regression, the same lags (:func:`eg_null`). It is checked against
MacKinnon's (2010) response-surface critical values in the tests
(:data:`MACKINNON_2010_C`). The simulation is seeded and cached by length, so
the same input always gives the same p-value; its resolution is ``1 / sims``.

No package beyond numpy is needed, and none is added.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from trading_research.strategies.spread import Hedge, SpreadError, fit_hedge, fit_ou

#: Augmentation lags of the Dickey-Fuller regression, by default. Hourly
#: residuals carry little serial correlation in their changes; one lag absorbs it.
DEFAULT_LAGS = 1

#: Simulated null draws per series length.
DEFAULT_SIMS = 4000

#: Seed of the null simulation, fixed so a p-value is a function of the input.
NULL_SEED = 20200106

#: MacKinnon (2010), Table 1, constant case: response-surface coefficients
#: ``(b_inf, b_1, b_2, b_3)`` of the critical value ``b_inf + b_1/T + b_2/T^2 + b_3/T^3``,
#: by number of series N (1 is the plain Dickey-Fuller test, 2 the residual of a
#: two-series cointegrating regression) and level.
MACKINNON_2010_C: dict[int, dict[float, tuple[float, float, float, float]]] = {
    1: {
        0.01: (-3.43035, -6.5393, -16.786, -79.433),
        0.05: (-2.86154, -2.8903, -4.234, -40.040),
        0.10: (-2.56677, -1.5384, -2.809, 0.0),
    },
    2: {
        0.01: (-3.89644, -10.9519, -22.527, 0.0),
        0.05: (-3.33613, -6.1101, -6.823, 0.0),
        0.10: (-3.04445, -4.2412, -2.720, 0.0),
    },
}


class PairsError(ValueError):
    """The input cannot support the statistic or the backtest asked for."""


def mackinnon_critical(n_series: int, level: float, nobs: int) -> float:
    """MacKinnon's (2010) critical value for ``n_series`` at ``level`` and ``nobs``."""
    b = MACKINNON_2010_C[n_series][level]
    return b[0] + b[1] / nobs + b[2] / nobs**2 + b[3] / nobs**3


def _adf_design(x: np.ndarray, lags: int, constant: bool) -> tuple[np.ndarray, np.ndarray]:
    """Rows of the Dickey-Fuller regression for a batch ``x`` of shape (m, n)."""
    dx = np.diff(x, axis=1)
    n = x.shape[1]
    if n - 1 - lags < 10 + lags:
        raise PairsError(f"{n} observations cannot support {lags} lag(s)")
    y = dx[:, lags:]
    columns = [x[:, lags : n - 1]]
    for j in range(1, lags + 1):
        columns.append(dx[:, lags - j : n - 1 - j])
    if constant:
        columns.append(np.ones_like(y))
    return np.stack(columns, axis=2), y


def adf_t(x: np.ndarray, *, lags: int = DEFAULT_LAGS, constant: bool = True) -> np.ndarray:
    """The Dickey-Fuller t of the lagged level, for one series or a batch (rows).

    ``constant`` adds an intercept (the plain test of a series); the
    Engle-Granger residual test uses none.
    """
    batch = np.atleast_2d(np.asarray(x, dtype=np.float64))
    design, y = _adf_design(batch, lags, constant)
    xtx = np.einsum("mtk,mtl->mkl", design, design)
    xty = np.einsum("mtk,mt->mk", design, y)
    inverse = np.linalg.inv(xtx)
    beta = np.einsum("mkl,ml->mk", inverse, xty)
    residual = y - np.einsum("mtk,mk->mt", design, beta)
    dof = y.shape[1] - design.shape[2]
    s2 = np.sum(residual**2, axis=1) / dof
    t = beta[:, 0] / np.sqrt(s2 * inverse[:, 0, 0])
    return t if np.ndim(x) > 1 else t[:1]


def _hedge_residuals(log_a: np.ndarray, log_b: np.ndarray) -> np.ndarray:
    """OLS residual of each row of ``log_a`` on the same row of ``log_b``, with an intercept."""
    xb = log_b - log_b.mean(axis=1, keepdims=True)
    ya = log_a - log_a.mean(axis=1, keepdims=True)
    beta = np.sum(xb * ya, axis=1) / np.sum(xb * xb, axis=1)
    return ya - beta[:, None] * xb


@lru_cache(maxsize=32)
def eg_null(nobs: int, lags: int = DEFAULT_LAGS, sims: int = DEFAULT_SIMS) -> np.ndarray:
    """The Engle-Granger t under no cointegration, sorted: ``sims`` simulated draws.

    Two independent Gaussian random walks of ``nobs`` steps, the hedge
    regression and the residual test exactly as :func:`engle_granger` runs them.
    """
    rng = np.random.default_rng(np.random.SeedSequence([NULL_SEED, nobs, lags]))
    out = []
    chunk = max(1, min(sims, 2_000_000 // max(nobs, 1)))
    done = 0
    while done < sims:
        m = min(chunk, sims - done)
        a = np.cumsum(rng.standard_normal((m, nobs)), axis=1)
        b = np.cumsum(rng.standard_normal((m, nobs)), axis=1)
        out.append(adf_t(_hedge_residuals(a, b), lags=lags, constant=False))
        done += m
    draws = np.sort(np.concatenate(out))
    draws.setflags(write=False)
    return draws


def null_p_value(t: float, null: np.ndarray) -> float:
    """Share of the null at or below ``t``, counting the observation itself."""
    if not math.isfinite(t):
        return math.nan
    below = int(np.searchsorted(null, t, side="right"))
    return (below + 1) / (len(null) + 1)


@dataclass(frozen=True)
class Cointegration:
    """An Engle-Granger test of one ordered pair on one window."""

    hedge: Hedge
    t: float
    p_value: float
    lags: int
    nobs: int
    #: MacKinnon's 5% critical value for two series at this length, for reference.
    critical_5: float


def engle_granger(
    log_a: np.ndarray,
    log_b: np.ndarray,
    *,
    lags: int = DEFAULT_LAGS,
    sims: int = DEFAULT_SIMS,
) -> Cointegration:
    """Hedge ``log_a`` on ``log_b`` and test the residual for a unit root."""
    a = np.asarray(log_a, dtype=np.float64)
    b = np.asarray(log_b, dtype=np.float64)
    if len(a) != len(b):
        raise PairsError(f"legs of different length: {len(a)} and {len(b)}")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise PairsError("a leg has a missing or non-positive price on this window")
    try:
        hedge = fit_hedge(a, b)
    except SpreadError as exc:
        raise PairsError(str(exc)) from exc
    residual = hedge.residual(a, b)
    if not hedge.residual_sd > 1e-12:
        raise PairsError("the legs move as one on this window: the residual is constant")
    t = float(adf_t(residual, lags=lags, constant=False)[0])
    return Cointegration(
        hedge=hedge,
        t=t,
        p_value=null_p_value(t, eg_null(len(a), lags, sims)),
        lags=lags,
        nobs=len(a),
        critical_5=mackinnon_critical(2, 0.05, len(a)),
    )


def half_life(residual: np.ndarray) -> float:
    """Bars for half a deviation of the residual to decay (AR(1)); inf without a pull."""
    try:
        return float(fit_ou(np.asarray(residual, dtype=np.float64), step=1).half_life)
    except SpreadError:
        return math.inf


def return_correlation(log_a: np.ndarray, log_b: np.ndarray) -> float:
    """Correlation of the two legs' bar log returns."""
    ra, rb = np.diff(np.asarray(log_a, float)), np.diff(np.asarray(log_b, float))
    if len(ra) < 3 or np.std(ra) == 0 or np.std(rb) == 0:
        return math.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def holm_adjusted(p_values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Holm's step-down adjusted p-values, in the input order; NaN stays NaN and counts in m.

    A hypothesis passes at family level ``alpha`` exactly when its adjusted
    p-value is at most ``alpha``, as
    :func:`trading_research.market_making.verdicts.holm` decides it.
    """
    p = np.asarray(p_values, dtype=np.float64)
    m = len(p)
    order = np.argsort(np.where(np.isnan(p), np.inf, p), kind="stable")
    adjusted = np.full(m, np.nan)
    running = 0.0
    for rank, index in enumerate(order):
        if np.isnan(p[index]):
            break
        running = max(running, min(1.0, (m - rank) * p[index]))
        adjusted[index] = running
    return adjusted
