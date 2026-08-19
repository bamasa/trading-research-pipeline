"""Predicting how sure the model is, and trading only when it is sure enough.

Every model here has produced one number per observation: a direction, or an
expected edge. None of them said how much to believe it, and the pipeline has
been selecting trades by *how large* the prediction is. Those are different
things. A large prediction the model has no business making is exactly the one
worth skipping, and a modest prediction it is confident about may be the better
trade.

The rule this enables is the one a desk actually writes:

    go long when   mu - k * sigma  >  cost
    go short when  mu + k * sigma  < -cost

with ``mu`` the expected edge, ``sigma`` the model's own uncertainty about it,
and ``k`` how many standard deviations of margin is required. Nothing is traded
on a prediction whose lower bound does not clear the round trip. The threshold
that everywhere else in this project is swept on validation disappears — it is
replaced by an arithmetic comparison in basis points, with ``k`` as the one
stated assumption.

Fitting the variance without lying to it
----------------------------------------
A variance model trained on the mean model's *in-sample* residuals learns that
the mean model is far better than it is, because those residuals are the ones it
has already fitted. The variance then comes out too small everywhere and the
gate opens on everything, which is worse than having no gate at all — it is a
gate that reports confidence it has not earned.

So the training block is split chronologically: the mean is fitted on the front,
its residuals are measured on the tail it has never seen, and the variance model
learns from those. The same arrangement as the probability calibrator, for the
same reason.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trading_research.models.base import CLASSES, Model

BUY_COLUMN = CLASSES.index(1)
SELL_COLUMN = CLASSES.index(-1)
HOLD_COLUMN = CLASSES.index(0)


class HeteroscedasticRegressor(Model):
    """Predicts an expected edge and how uncertain that estimate is.

    Two regressors. The first predicts the target; the second predicts the
    squared error the first will make, fitted on residuals from data the first
    never saw. The square root of the second is the standard deviation the
    decision rule needs.

    ``holdout_fraction`` is how much of the training block is reserved for
    measuring residuals. Too little and the variance model has nothing to learn
    from; too much and the mean model is weakened to serve it.
    """

    def __init__(
        self,
        mean_model: Model,
        variance_model: Model,
        *,
        holdout_fraction: float = 0.3,
        floor_bp: float = 0.1,
    ) -> None:
        super().__init__(name=f"hetero({mean_model.name}, {variance_model.name})")
        if not 0.1 <= holdout_fraction <= 0.6:
            raise ValueError(f"holdout_fraction must be in [0.1, 0.6], got {holdout_fraction}")
        if floor_bp <= 0:
            raise ValueError(f"floor_bp must be positive, got {floor_bp}")
        self.mean_model = mean_model
        self.variance_model = variance_model
        self.holdout_fraction = holdout_fraction
        # A predicted variance of zero would let any prediction through the
        # gate, however small. The floor is what "no uncertainty at all" is
        # allowed to mean, and it is stated rather than discovered.
        self.floor_bp = floor_bp

    def fit(self, x: pd.DataFrame, y: pd.Series) -> HeteroscedasticRegressor:
        usable = np.isfinite(y.to_numpy(dtype="float64"))
        x, y = x.loc[usable], y.loc[usable]
        split = int(len(x) * (1 - self.holdout_fraction))
        if split < 200 or len(x) - split < 200:
            raise ValueError(
                f"need at least 200 rows each side of the residual split, "
                f"got {split} and {len(x) - split}"
            )

        # Chronological, not random: neighbouring rows are near-duplicates, so
        # a random split would leave the variance model measuring residuals on
        # moments the mean model has effectively already seen.
        front_x, front_y = x.iloc[:split], y.iloc[:split]
        tail_x, tail_y = x.iloc[split:], y.iloc[split:]

        self.mean_model.fit(front_x, front_y)
        residual = tail_y.to_numpy(dtype="float64") - self._mean_of(tail_x)
        self.variance_model.fit(tail_x, pd.Series(residual**2, index=tail_x.index))

        # The mean model is then refitted on everything: the split existed to
        # produce honest residuals, not to hold data back from the final fit.
        self.mean_model.fit(x, y)
        self.fitted_ = True
        return self

    def _mean_of(self, x: pd.DataFrame) -> np.ndarray:
        if hasattr(self.mean_model, "predict_edge_bp"):
            return np.asarray(self.mean_model.predict_edge_bp(x), dtype="float64")
        raise TypeError(f"{self.mean_model.name} does not predict basis points")

    def predict_mean_bp(self, x: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self._mean_of(x)

    def predict_sigma_bp(self, x: pd.DataFrame) -> np.ndarray:
        """Standard deviation of the mean model's error, in basis points."""
        self._check_fitted()
        if hasattr(self.variance_model, "predict_edge_bp"):
            variance = np.asarray(self.variance_model.predict_edge_bp(x), dtype="float64")
        else:
            raise TypeError(f"{self.variance_model.name} does not predict basis points")
        # A regressor fitted on squared residuals can predict a negative number;
        # clipping at the floor is the only sensible reading of that.
        return np.sqrt(np.clip(variance, self.floor_bp**2, None))

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """A confidence that already carries the uncertainty.

        The ratio of the mean to its own standard deviation, squashed. This
        keeps the model usable by the rest of the pipeline, but the decision
        rule that motivates the class is :class:`ConfidenceBound`, which
        compares against a cost rather than a swept threshold.
        """
        mean = self.predict_mean_bp(x)
        sigma = self.predict_sigma_bp(x)
        ratio = np.tanh(np.abs(mean) / np.maximum(sigma, self.floor_bp))
        confidence = 1.0 / 3.0 + (0.9 - 1.0 / 3.0) * ratio

        proba = np.zeros((len(x), len(CLASSES)))
        up = mean >= 0
        proba[up, BUY_COLUMN] = confidence[up]
        proba[~up, SELL_COLUMN] = confidence[~up]
        proba[:, HOLD_COLUMN] = 1.0 - proba.sum(axis=1)
        return proba

    def describe(self) -> dict[str, Any]:
        out = super().describe()
        out.update(
            {
                "kind": "heteroscedastic",
                "holdout_fraction": self.holdout_fraction,
                "floor_bp": self.floor_bp,
                "mean_model": self.mean_model.describe(),
                "variance_model": self.variance_model.describe(),
            }
        )
        return out


class ConfidenceBound(Model):
    """Trade only where the prediction's lower bound clears the cost.

        long   when  mu - k*sigma >  cost
        short  when  mu + k*sigma < -cost

    What this rejects is different from what a size threshold rejects. A
    threshold on ``|mu|`` discards *small* predictions; this discards *unreliable*
    ones, and a large prediction the model is unsure about is exactly the trade
    worth skipping. The two filters disagree on most observations, which is the
    whole reason for the class.

    ``k`` is the only free parameter and it is an assumption rather than a fitted
    value: at zero the rule reduces to expected value against cost, and each unit
    asks for one more standard deviation of margin.
    """

    def __init__(self, model: HeteroscedasticRegressor, *, cost_bp: float, k: float = 1.0) -> None:
        super().__init__(name=f"bound({model.name}, k={k:g})")
        if cost_bp <= 0:
            raise ValueError(f"cost_bp must be positive, got {cost_bp}")
        if k < 0:
            raise ValueError(f"k must be non-negative, got {k}")
        self.model = model
        self.cost_bp = cost_bp
        self.k = k

    def fit(self, x: pd.DataFrame, y: pd.Series) -> ConfidenceBound:
        self.model.fit(x, y)
        self.fitted_ = True
        return self

    def bounds(self, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper bound of the predicted edge, in basis points."""
        mean = self.model.predict_mean_bp(x)
        margin = self.k * self.model.predict_sigma_bp(x)
        return mean - margin, mean + margin

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        lower, upper = self.bounds(x)
        long = lower > self.cost_bp
        short = upper < -self.cost_bp

        # Confidence is how far past the cost the bound reaches, so a trade
        # barely clearing it is distinguishable from one clearing it twice over
        # and the rest of the pipeline can still be selective if it wants.
        surplus = np.where(long, lower - self.cost_bp, np.where(short, -upper - self.cost_bp, 0.0))
        confidence = np.where(
            long | short,
            1.0 / 3.0 + (0.9 - 1.0 / 3.0) * np.tanh(surplus / self.cost_bp),
            1.0 / 3.0,
        )

        proba = np.zeros((len(x), len(CLASSES)))
        proba[long, BUY_COLUMN] = confidence[long]
        proba[short, SELL_COLUMN] = confidence[short]
        proba[:, HOLD_COLUMN] = 1.0 - proba.sum(axis=1)
        return proba

    def describe(self) -> dict[str, Any]:
        out = super().describe()
        out.update({"cost_bp": self.cost_bp, "k": self.k, "model": self.model.describe()})
        return out
