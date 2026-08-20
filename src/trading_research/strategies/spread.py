"""Two instruments, one residual, and the speed at which it comes back.

Directional prediction on a single perpetual is what the rest of this project
spent itself on, and §26 is the result: a gross edge of a few basis points
against a cost several times larger. Pairs trading attacks a different quantity.
The spread between two instruments that move together is not a random walk — it
is pulled back towards a level — and mean reversion is a far stronger statistical
effect than direction at these horizons.

It is not free. Two legs means two round trips, so the cost to beat doubles.
The bet is that the residual is predictable by more than a factor of two better
than the direction of either leg.

The residual
------------
For instruments *a* and *b*, the hedge ratio β is fitted by least squares on log
prices over a trailing window, and the residual is

    z(t) = log P_a(t) - alpha - beta * log P_b(t)

Beta is refitted on the training window only and carried forward, because a hedge
ratio refitted on the period being traded is fitted to the answer.

Ornstein-Uhlenbeck
------------------
The residual is modelled as a mean-reverting process

    dz = theta * (mu - z) dt + sigma * dW

fitted by regressing the next residual on the current one -- the discrete
form z(t+1) = a + b*z(t) + noise, giving theta = -log(b) per step and a
half-life of log(2)/theta. The half-life is the number
this contributes that a plain z-score does not: it says how long a position must
be held for the reversion to happen, and therefore whether the trade can pay for
two round trips before the edge decays.

A pair whose half-life is far longer than the holding period is not tradeable
here whatever its z-score, and a pair whose half-life is one row is noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class SpreadError(ValueError):
    """The pair cannot be fitted, or the fit is degenerate."""


@dataclass(frozen=True)
class Hedge:
    """A fitted linear relationship between two log price series."""

    alpha: float
    beta: float
    #: Standard deviation of the residual on the fitting window, in log units.
    residual_sd: float
    n: int

    def residual(self, log_a: np.ndarray, log_b: np.ndarray) -> np.ndarray:
        return log_a - self.alpha - self.beta * log_b


@dataclass(frozen=True)
class OrnsteinUhlenbeck:
    """Mean reversion speed of a residual, and what it implies."""

    mu: float
    theta: float
    sigma: float
    #: Rows for half the deviation to decay. Infinite when there is no pull.
    half_life: float
    #: Share of variance the autoregression explains. Near zero means the
    #: "mean reversion" is the regression fitting noise.
    r_squared: float

    @property
    def reverts(self) -> bool:
        return np.isfinite(self.half_life) and self.theta > 0.0


def fit_hedge(log_a: np.ndarray, log_b: np.ndarray) -> Hedge:
    """Least-squares hedge ratio on the window given, and nothing else."""
    ok = np.isfinite(log_a) & np.isfinite(log_b)
    if ok.sum() < 100:
        raise SpreadError(f"only {int(ok.sum())} usable rows to fit a hedge on")
    x, y = log_b[ok], log_a[ok]
    if np.std(x) == 0:
        raise SpreadError("the hedge instrument did not move on this window")
    beta, alpha = np.polyfit(x, y, 1)
    residual = y - alpha - beta * x
    return Hedge(
        alpha=float(alpha), beta=float(beta), residual_sd=float(np.std(residual)), n=int(ok.sum())
    )


def fit_ou(residual: np.ndarray, *, step: int = 1) -> OrnsteinUhlenbeck:
    """Fit the discrete mean-reverting form to a residual series.

    ``step`` samples the series before fitting. Consecutive five-second
    residuals are nearly identical, and fitting on all of them makes θ tiny and
    the half-life enormous for arithmetic reasons rather than economic ones.
    """
    values = residual[np.isfinite(residual)][::step]
    if len(values) < 100:
        raise SpreadError(f"only {len(values)} usable residuals")
    x, y = values[:-1], values[1:]
    if np.std(x) == 0:
        raise SpreadError("residual is constant")

    b, a = np.polyfit(x, y, 1)
    predicted = a + b * x
    residual_var = float(np.var(y - predicted))
    r_squared = 1.0 - residual_var / float(np.var(y)) if np.var(y) > 0 else 0.0

    # b >= 1 means the deviation grows rather than decays: not a reverting pair.
    if b <= 0 or b >= 1:
        return OrnsteinUhlenbeck(
            mu=float(np.mean(values)),
            theta=0.0,
            sigma=float(np.std(values)),
            half_life=float("inf"),
            r_squared=float(r_squared),
        )
    theta = -np.log(b) / step
    return OrnsteinUhlenbeck(
        mu=float(a / (1.0 - b)),
        theta=float(theta),
        sigma=float(np.sqrt(residual_var)),
        half_life=float(np.log(2.0) / theta),
        r_squared=float(r_squared),
    )


def z_score(residual: np.ndarray, ou: OrnsteinUhlenbeck) -> np.ndarray:
    """Deviation from the fitted mean, in fitted standard deviations.

    Uses the *fitted* mean and dispersion rather than a rolling window of the
    series being traded — a rolling z-score computed on the traded period would
    centre the signal on the answer.
    """
    scale = ou.sigma if ou.sigma > 0 else float("nan")
    return (residual - ou.mu) / scale


def signal(z: np.ndarray, *, entry: float = 2.0, exit_at: float = 0.5) -> np.ndarray:
    """Enter against the deviation, leave when it has mostly closed.

    +1 means long the first leg and short the hedge, which is what a residual
    *below* its mean calls for. The position is held through the band between
    ``exit_at`` and ``entry`` rather than flipping every row, since crossing the
    entry threshold repeatedly is how a mean-reversion backtest manufactures
    trades it would have paid for many times over.
    """
    if entry <= exit_at:
        raise SpreadError(f"entry {entry} must exceed exit {exit_at}")
    out = np.zeros(len(z), dtype=int)
    position = 0
    for i, value in enumerate(z):
        if not np.isfinite(value):
            out[i] = position
            continue
        if position == 0:
            if value <= -entry:
                position = 1
            elif value >= entry:
                position = -1
        elif (position == 1 and value >= -exit_at) or (position == -1 and value <= exit_at):
            position = 0
        out[i] = position
    return out


def rank_pairs(
    log_prices: pd.DataFrame, *, step: int = 12, min_r_squared: float = 0.01
) -> pd.DataFrame:
    """Every pair in the frame, ranked by how usable its reversion looks.

    Fitted on whatever window is handed in — which must be a training window.
    The returned table is a *candidate list*, not a result: a pair that reverts
    on the fitting window is the least surprising thing in statistics.
    """
    names = list(log_prices.columns)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            try:
                hedge = fit_hedge(log_prices[a].to_numpy(), log_prices[b].to_numpy())
                residual = hedge.residual(log_prices[a].to_numpy(), log_prices[b].to_numpy())
                ou = fit_ou(residual, step=step)
            except SpreadError:
                continue
            if not ou.reverts or ou.r_squared < min_r_squared:
                continue
            rows.append(
                {
                    "a": a,
                    "b": b,
                    "beta": hedge.beta,
                    "residual_sd_bp": hedge.residual_sd * 1e4,
                    "half_life_rows": ou.half_life,
                    "r_squared": ou.r_squared,
                    # What one standard deviation of the residual is worth, in
                    # basis points. Anything smaller than a round trip is not a
                    # trade however reliably it reverts.
                    "amplitude_bp": hedge.residual_sd * 1e4,
                }
            )
    table = pd.DataFrame(rows)
    return table.sort_values("half_life_rows").reset_index(drop=True) if len(table) else table
