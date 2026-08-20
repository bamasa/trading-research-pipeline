"""Randomised tree ensembles, and a second boosting implementation.

Two families that the project has not had, both chosen for what they contribute
to an ensemble rather than for their solo performance.

**Randomised forests.** A random forest decorrelates its members by giving each
tree a bootstrap sample and each split a random subset of features. Extremely
randomised trees go further and pick the split *threshold* at random too. The
second sounds worse and often is not: on features that are mostly noise, choosing
the best threshold is largely fitting that noise, and giving it up costs a little
bias and buys a lot of variance. On order-book features, where the honest
information coefficient is a few hundredths, that is usually the right trade.

**LightGBM.** A second gradient-boosting implementation beside XGBoost, growing
trees leaf-wise rather than level-wise and binning features histogram-style. It
is here for stacking rather than for competition: two boosters that split
differently make different mistakes, and different mistakes is the only thing
that makes combining models worth anything.

All three expose the project's ``Model`` interface, so they drop into the same
searches, the same walk-forward and the same bias-variance decomposition as
everything else.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trading_research.models.base import CLASSES, Model, clean


class _SklearnEnsemble(Model):
    """Shared plumbing: fit, and map class probabilities onto the project's order."""

    def __init__(self, name: str, **params: Any) -> None:
        super().__init__(name=name)
        self.params = params
        self.estimator_: Any = None

    def _build(self) -> Any:
        raise NotImplementedError

    def fit(self, x: pd.DataFrame, y: pd.Series) -> _SklearnEnsemble:
        x, y = clean(x, y)
        if y.nunique() < 2:
            raise ValueError(f"{self.name}: training block has one class only")
        self.estimator_ = self._build()
        self.estimator_.fit(x, y)
        self.fitted_ = True
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        raw = self.estimator_.predict_proba(x)
        # The estimator's class order is whatever it saw; the project's is
        # fixed. Mapping by label rather than by position keeps a training block
        # that happened to contain no sells from silently shifting every column.
        out = np.zeros((len(x), len(CLASSES)))
        for position, label in enumerate(self.estimator_.classes_):
            out[:, CLASSES.index(int(label))] = raw[:, position]
        return out

    def describe(self) -> dict[str, Any]:
        return {"model": self.name, **self.params}


class RandomForest(_SklearnEnsemble):
    """Bagged trees with feature subsampling at each split."""

    def __init__(
        self,
        *,
        n_estimators: int = 300,
        max_depth: int | None = 6,
        min_samples_leaf: int = 200,
        max_features: str | float = "sqrt",
        class_weight: str | None = "balanced_subsample",
        seed: int = 42,
    ) -> None:
        super().__init__(
            "random_forest",
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            seed=seed,
        )

    def _build(self) -> Any:
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=self.params["n_estimators"],
            max_depth=self.params["max_depth"],
            # Large leaves on purpose. A leaf of twenty rows on five-second book
            # data is a leaf of about one minute, which memorises a moment
            # rather than learning a pattern.
            min_samples_leaf=self.params["min_samples_leaf"],
            max_features=self.params["max_features"],
            class_weight=self.params["class_weight"],
            random_state=self.params["seed"],
            n_jobs=-1,
        )


class ExtraTrees(_SklearnEnsemble):
    """Random forest with random split thresholds as well as random features."""

    def __init__(
        self,
        *,
        n_estimators: int = 400,
        max_depth: int | None = 8,
        min_samples_leaf: int = 200,
        max_features: str | float = "sqrt",
        class_weight: str | None = "balanced_subsample",
        seed: int = 42,
    ) -> None:
        super().__init__(
            "extra_trees",
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            seed=seed,
        )

    def _build(self) -> Any:
        from sklearn.ensemble import ExtraTreesClassifier

        return ExtraTreesClassifier(
            n_estimators=self.params["n_estimators"],
            max_depth=self.params["max_depth"],
            min_samples_leaf=self.params["min_samples_leaf"],
            max_features=self.params["max_features"],
            class_weight=self.params["class_weight"],
            random_state=self.params["seed"],
            n_jobs=-1,
        )


class LightGBMBaseline(Model):
    """Leaf-wise histogram boosting — a second booster that errs differently."""

    def __init__(
        self,
        *,
        n_estimators: int = 300,
        num_leaves: int = 15,
        learning_rate: float = 0.05,
        min_child_samples: int = 200,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 5.0,
        seed: int = 42,
    ) -> None:
        super().__init__(name="lightgbm")
        # Typed explicitly rather than as dict[str, float]: LightGBM's counts
        # are integers and a float silently widens them.
        self.params: dict[str, Any] = {
            "n_estimators": int(n_estimators),
            "num_leaves": int(num_leaves),
            "learning_rate": float(learning_rate),
            "min_child_samples": int(min_child_samples),
            "subsample": float(subsample),
            "colsample_bytree": float(colsample_bytree),
            "reg_lambda": float(reg_lambda),
            "seed": int(seed),
        }
        self.estimator_: Any = None
        self.classes_: np.ndarray | None = None

    def fit(self, x: pd.DataFrame, y: pd.Series) -> LightGBMBaseline:
        from lightgbm import LGBMClassifier

        x, y = clean(x, y)
        if y.nunique() < 2:
            raise ValueError("lightgbm: training block has one class only")
        self.estimator_ = LGBMClassifier(
            n_estimators=self.params["n_estimators"],
            num_leaves=self.params["num_leaves"],
            learning_rate=self.params["learning_rate"],
            min_child_samples=self.params["min_child_samples"],
            subsample=self.params["subsample"],
            subsample_freq=1,
            colsample_bytree=self.params["colsample_bytree"],
            reg_lambda=self.params["reg_lambda"],
            random_state=self.params["seed"],
            class_weight="balanced",
            # Single-threaded on purpose. LightGBM ships its own OpenMP
            # runtime, and on macOS loading it into a process that already has
            # XGBoost's or PyTorch's segfaults as soon as it opens a thread
            # pool. The suite hits that whenever a torch test runs first. One
            # thread costs a little speed and makes the model usable next to
            # the rest of the project.
            n_jobs=1,
            verbose=-1,
        )
        self.estimator_.fit(x, y)
        self.classes_ = self.estimator_.classes_
        self.fitted_ = True
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        raw = self.estimator_.predict_proba(x)
        out = np.zeros((len(x), len(CLASSES)))
        assert self.classes_ is not None
        for position, label in enumerate(self.classes_):
            out[:, CLASSES.index(int(label))] = raw[:, position]
        return out

    def describe(self) -> dict[str, Any]:
        return {"model": self.name, **self.params}
