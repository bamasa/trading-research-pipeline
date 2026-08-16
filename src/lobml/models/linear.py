"""Logistic regression baseline.

The first model worth beating. It is linear, it is fast, and its coefficients
can be read, which makes it a useful check on whether a feature carries the
sign it is supposed to: a queue-imbalance coefficient with the wrong sign means
something is wrong upstream, and that is far easier to notice here than inside
a boosted ensemble.

The scaler lives inside ``fit``. Standardising before splitting is one of the
quietest leaks there is — the mean and standard deviation carry information
from the test period into training, nothing is out of order, and every metric
improves slightly. Keeping the scaler inside the model makes that mistake
awkward to write by accident.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from lobml.models.base import CLASSES, Model


class LogisticBaseline(Model):
    """Multinomial logistic regression on standardised features.

    ``class_weight="balanced"`` by default. Under taker costs most moments are
    HOLD, often overwhelmingly so, and an unweighted fit responds by predicting
    HOLD everywhere — technically accurate and completely uninformative.
    Weighting does not create signal, but it keeps the fit from collapsing
    before the decision threshold has a chance to do its job.
    """

    def __init__(
        self,
        *,
        C: float = 1.0,
        max_iter: int = 200,
        class_weight: str | None = "balanced",
        seed: int = 42,
    ) -> None:
        super().__init__(name="logistic")
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.seed = seed
        self.pipeline_: Any = None
        self.columns_: list[str] | None = None

    def fit(self, x: pd.DataFrame, y: pd.Series) -> LogisticBaseline:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        self.columns_ = list(x.columns)
        self.pipeline_ = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=self.C,
                        max_iter=self.max_iter,
                        class_weight=self.class_weight,
                        random_state=self.seed,
                    ),
                ),
            ]
        )
        self.pipeline_.fit(x.to_numpy(), y.to_numpy())
        self.fitted_ = True
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        assert self.pipeline_ is not None
        if self.columns_ is not None and list(x.columns) != self.columns_:
            raise ValueError(
                f"{self.name}: feature columns differ from those seen in fit; "
                f"a reordered matrix silently changes what every coefficient means"
            )
        return _align_columns(self.pipeline_.predict_proba(x.to_numpy()), self.pipeline_.classes_)

    def coefficients(self) -> pd.DataFrame:
        """Coefficients per class, for inspection.

        Worth reading rather than skipping. A feature whose sign contradicts its
        definition points at a bug upstream, and it is visible here in a way it
        never is in an ensemble.
        """
        self._check_fitted()
        assert self.pipeline_ is not None and self.columns_ is not None
        clf = self.pipeline_.named_steps["clf"]
        return pd.DataFrame(
            clf.coef_.T, index=self.columns_, columns=[int(c) for c in clf.classes_]
        )

    def describe(self) -> dict[str, Any]:
        out = super().describe()
        out.update({"C": self.C, "max_iter": self.max_iter, "class_weight": self.class_weight})
        return out


def _align_columns(proba: np.ndarray, model_classes: np.ndarray) -> np.ndarray:
    """Reorder a probability matrix into :data:`CLASSES` order.

    scikit-learn orders its output by sorted class label, which happens to
    match here, but relying on coincidence is how buy and sell get swapped. The
    mapping is made explicit, and a class missing from a training block is
    filled with zeros rather than shifting every other column left.
    """
    out = np.zeros((proba.shape[0], len(CLASSES)))
    lookup = {int(c): i for i, c in enumerate(model_classes)}
    for target, cls in enumerate(CLASSES):
        source = lookup.get(int(cls))
        if source is not None:
            out[:, target] = proba[:, source]
    return out
