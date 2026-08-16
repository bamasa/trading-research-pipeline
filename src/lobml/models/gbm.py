"""Gradient-boosted trees baseline.

The workhorse for tabular features of this kind: it finds interactions the
linear model cannot, needs no scaling, and tolerates features on wildly
different scales.

Hyperparameters are deliberately modest and fixed. A large search would make
the comparison against the linear baseline a comparison of search budgets, and
on a sample this size a deep ensemble mostly learns the particular fortnight it
was fitted on. Depth and tree count are kept small for the same reason.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from lobml.models.base import CLASSES, Model
from lobml.models.linear import _align_columns


class GradientBoostedBaseline(Model):
    """XGBoost multi-class classifier over the feature frame."""

    def __init__(
        self,
        *,
        n_estimators: int = 300,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: float = 20.0,
        reg_lambda: float = 2.0,
        balance_classes: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__(name="xgboost")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.balance_classes = balance_classes
        self.seed = seed
        self.model_: Any = None
        self.columns_: list[str] | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, x: pd.DataFrame, y: pd.Series) -> GradientBoostedBaseline:
        from xgboost import XGBClassifier

        self.columns_ = list(x.columns)
        # XGBoost needs contiguous labels from zero; the -1/0/+1 encoding is
        # mapped here and mapped straight back on the way out, so the rest of
        # the pipeline never sees the internal encoding.
        present = np.array(sorted(set(y.unique())), dtype=int)
        self.classes_ = present
        encoded = pd.Series(np.searchsorted(present, y.to_numpy().astype(int)), index=y.index)

        weights = None
        if self.balance_classes:
            counts = encoded.value_counts()
            inverse = {cls: len(encoded) / (len(counts) * n) for cls, n in counts.items()}
            weights = encoded.map(inverse).to_numpy()

        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            reg_lambda=self.reg_lambda,
            objective="multi:softprob",
            num_class=len(present),
            tree_method="hist",
            random_state=self.seed,
            n_jobs=-1,
            verbosity=0,
        )
        self.model_.fit(x.to_numpy(), encoded.to_numpy(), sample_weight=weights)
        self.fitted_ = True
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        assert self.model_ is not None and self.classes_ is not None
        if self.columns_ is not None and list(x.columns) != self.columns_:
            raise ValueError(f"{self.name}: feature columns differ from those seen in fit")
        return _align_columns(self.model_.predict_proba(x.to_numpy()), self.classes_)

    def importances(self) -> pd.Series:
        """Gain-based feature importance, for the report."""
        self._check_fitted()
        assert self.model_ is not None and self.columns_ is not None
        return pd.Series(self.model_.feature_importances_, index=self.columns_).sort_values(
            ascending=False
        )

    def describe(self) -> dict[str, Any]:
        out = super().describe()
        out.update(
            {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "subsample": self.subsample,
                "colsample_bytree": self.colsample_bytree,
                "min_child_weight": self.min_child_weight,
                "reg_lambda": self.reg_lambda,
                "balance_classes": self.balance_classes,
            }
        )
        return out


__all__ = ["CLASSES", "GradientBoostedBaseline"]
