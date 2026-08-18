"""Models that predict a number, wrapped to look like the rest.

The regression targets — ``magnitude`` and ``net_pnl`` — ask for basis points,
not a class. Everything downstream of the model works in three-class
probabilities: the confidence threshold, the thinning, the exit rules that read
the model's ongoing opinion. Rewriting that path for regression would give two
code paths and two sets of bugs, so the prediction is converted instead.

The conversion is the interesting part
--------------------------------------
A predicted basis-point figure carries information a probability does not: its
scale is the same as the cost. That makes the trade decision writable directly —
act when the expected value exceeds the round trip — with no threshold to sweep.

But the pipeline wants a confidence, so the prediction is mapped onto one by
comparing it with the cost: a prediction of exactly the round trip maps to the
middle of the range, larger predictions towards the top, smaller towards the
class prior. Squashed with ``tanh``, so nothing saturates and the ordering
survives — a hard clip destroyed the ranking of the rule strategies and would
do the same here.

The mapping is monotonic, so sweeping a confidence threshold on the converted
scores is exactly equivalent to sweeping an expected-value threshold on the raw
predictions. Nothing is lost by going through it, and the rest of the pipeline
stays one path.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trading_research.models.base import CLASSES, Model

#: Confidence at zero expected edge — no opinion, all three classes equal.
NEUTRAL = 1.0 / 3.0

#: Ceiling, below one so that a threshold can always exclude the top of the
#: range. Matches the rule strategies for the same reason.
MAX_CONFIDENCE = 0.9


def edge_to_proba(edge_bp: np.ndarray, scale_bp: float) -> np.ndarray:
    """Map an expected edge in basis points onto three class probabilities.

    ``scale_bp`` is what counts as a large edge — normally the round-trip cost,
    so a prediction of exactly the cost lands halfway up the range.
    """
    edge_bp = np.nan_to_num(edge_bp, nan=0.0)
    magnitude = np.tanh(np.abs(edge_bp) / max(scale_bp, 1e-9))
    directional = NEUTRAL + (MAX_CONFIDENCE - NEUTRAL) * magnitude
    remainder = 1.0 - directional

    proba = np.zeros((len(edge_bp), len(CLASSES)))
    up = edge_bp >= 0
    proba[up, CLASSES.index(1)] = directional[up]
    proba[up, CLASSES.index(-1)] = remainder[up] / 2
    proba[~up, CLASSES.index(-1)] = directional[~up]
    proba[~up, CLASSES.index(1)] = remainder[~up] / 2
    proba[:, CLASSES.index(0)] = 1.0 - proba.sum(axis=1)
    return proba


class RegressionModel(Model):
    """Base for models fitted on a basis-point target.

    ``scale_bp`` is set at construction rather than learned: it is the cost of
    trading, which is known, and fitting it would let the model choose what
    counts as a large move.
    """

    def __init__(self, name: str, *, scale_bp: float = 11.0) -> None:
        super().__init__(name=name)
        self.scale_bp = scale_bp
        self.estimator_: Any = None

    def _build(self) -> Any:
        raise NotImplementedError

    def fit(self, x: pd.DataFrame, y: pd.Series) -> RegressionModel:
        usable = np.isfinite(y.to_numpy(dtype="float64"))
        if usable.sum() < 100:
            raise ValueError(f"only {int(usable.sum())} usable rows to fit on")
        self.estimator_ = self._build()
        self.estimator_.fit(x.loc[usable], y.loc[usable])
        self.fitted_ = True
        return self

    def predict_edge_bp(self, x: pd.DataFrame) -> np.ndarray:
        """The raw prediction, in basis points. What the target was in."""
        self._check_fitted()
        return np.asarray(self.estimator_.predict(x), dtype="float64")

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return edge_to_proba(self.predict_edge_bp(x), self.scale_bp)

    def describe(self) -> dict[str, Any]:
        out = super().describe()
        out.update({"kind": "regression", "scale_bp": self.scale_bp})
        return out


class RidgeBaseline(RegressionModel):
    """Linear, regularised. The regression counterpart of the logistic baseline.

    Ridge rather than plain least squares because the generated feature set is
    heavily collinear — lags and rolling statistics of the same primitive — and
    an unregularised fit on it produces large offsetting coefficients that move
    wildly between folds.
    """

    def __init__(self, alpha: float = 1.0, scale_bp: float = 11.0) -> None:
        super().__init__("ridge", scale_bp=scale_bp)
        self.alpha = alpha

    def _build(self) -> Any:
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))


class GradientBoostedRegressor(RegressionModel):
    """Gradient boosting on the basis-point target.

    Hyperparameters match the classifier's so the comparison between targets is
    a comparison between targets rather than between tunings.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        scale_bp: float = 11.0,
    ) -> None:
        super().__init__("xgboost_regressor", scale_bp=scale_bp)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate

    def _build(self) -> Any:
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=1,
            verbosity=0,
        )
