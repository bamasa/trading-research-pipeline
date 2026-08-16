"""One interface for every model, so comparisons are like for like.

The purpose is discipline rather than abstraction. A model here fits on a
training block, produces class probabilities, and nothing else. In particular
it does not choose its own decision threshold: that happens on the validation
block, through the same code for every model, so a comparison cannot turn into
a comparison of how carefully each one was tuned.

Anything fitted lives inside ``fit``. Scalers especially: fitting a scaler on
the whole dataset before splitting is a leak that no amount of careful
splitting afterwards can undo, and putting the scaler inside the model makes
that mistake awkward to write.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from trading_research.labels.directional import BUY, HOLD, SELL

#: Column order for probability output. Fixed once, here, because a silently
#: reordered probability matrix turns buy signals into sell signals and nothing
#: about the resulting numbers looks wrong.
CLASSES: tuple[int, int, int] = (SELL, HOLD, BUY)


@dataclass
class Model(ABC):
    """A three-class classifier over features, predicting sell / hold / buy."""

    name: str = "model"
    fitted_: bool = field(default=False, init=False, repr=False)

    @abstractmethod
    def fit(self, x: pd.DataFrame, y: pd.Series) -> Model:
        """Fit on one training block. Must not see validation or test data."""

    @abstractmethod
    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Return an ``(n, 3)`` array of probabilities in :data:`CLASSES` order."""

    def _check_fitted(self) -> None:
        if not self.fitted_:
            raise RuntimeError(f"{self.name} has not been fitted")

    def describe(self) -> dict[str, Any]:
        return {"name": self.name}


def clean(x: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Drop rows where any feature or the label is missing.

    Warm-up rows and the unlabelled tail are dropped rather than imputed. An
    imputed feature at the start of a block is a guess presented to the model as
    an observation, and the model has no way to tell the difference.
    """
    mask = x.notna().all(axis=1) & y.notna()
    return x.loc[mask], y.loc[mask]


class AlwaysHold(Model):
    """Never trades.

    The baseline that matters most under taker costs. Trading is expensive, so
    "do nothing" is a genuinely strong strategy, and any model that fails to
    beat it net of costs has not earned its complexity. Reporting it alongside
    the others keeps that comparison unavoidable.
    """

    def __init__(self) -> None:
        super().__init__(name="always_hold")

    def fit(self, x: pd.DataFrame, y: pd.Series) -> AlwaysHold:  # noqa: ARG002
        self.fitted_ = True
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        out = np.zeros((len(x), 3))
        out[:, CLASSES.index(HOLD)] = 1.0
        return out


class ClassPrior(Model):
    """Predicts the training class frequencies, ignoring the features.

    Separates two things that are easy to confuse: how much of a model's
    apparent skill comes from the features, and how much from simply knowing
    that most moments are HOLD. A model that barely beats this one is mostly
    reproducing the class balance.
    """

    def __init__(self) -> None:
        super().__init__(name="class_prior")
        self.prior_: np.ndarray | None = None

    def fit(self, x: pd.DataFrame, y: pd.Series) -> ClassPrior:  # noqa: ARG002
        counts = np.array([float((y == c).sum()) for c in CLASSES])
        total = counts.sum()
        if total == 0:
            raise ValueError("cannot fit on an empty training block")
        self.prior_ = counts / total
        self.fitted_ = True
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        assert self.prior_ is not None
        return np.tile(self.prior_, (len(x), 1))

    def describe(self) -> dict[str, Any]:
        out = super().describe()
        if self.prior_ is not None:
            out["prior"] = {str(c): float(p) for c, p in zip(CLASSES, self.prior_, strict=True)}
        return out
