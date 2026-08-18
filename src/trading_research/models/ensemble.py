"""Averaging several models into one.

The model comparison in the results says two things that point here. The linear
model has the best gross edge and the fewest ways to go wrong; the network has
the highest hit rate and swings by more between runs than the models differ by.
An average of several fits is the standard answer to the second problem, and it
costs the first one very little.

What averaging does and does not do
-----------------------------------
It reduces variance, not bias. If three models make independent errors around
the same signal, the mean is closer to that signal than any of them; if they all
miss the same thing — which, fitted on the same features and the same rows, they
largely do — the mean misses it too. So the case for it here is stability rather
than accuracy: a result that does not depend on which fit happened to run.

Probabilities are averaged rather than votes counted. A vote throws away how
sure each model was, which is the only thing the downstream confidence threshold
has to work with.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from trading_research.models.base import Model


class EnsembleModel(Model):
    """Averages the probabilities of several fitted models.

    ``weights`` default to equal. Fitting weights on validation is possible and
    deliberately not done: with three members and a validation block this size,
    the weights would be fitting the block rather than the models.
    """

    def __init__(self, members: Sequence[Model], *, weights: Sequence[float] | None = None) -> None:
        if len(members) < 2:
            raise ValueError(f"an ensemble needs at least two members, got {len(members)}")
        if weights is not None and len(weights) != len(members):
            raise ValueError(f"{len(weights)} weights for {len(members)} members")

        super().__init__(name="+".join(m.name for m in members))
        self.members = list(members)
        self.weights = (
            np.full(len(members), 1.0 / len(members))
            if weights is None
            else np.asarray(weights, dtype="float64") / float(np.sum(weights))
        )

    def fit(self, x: pd.DataFrame, y: pd.Series) -> EnsembleModel:
        for member in self.members:
            member.fit(x, y)
        self.fitted_ = True
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        stacked = np.stack([m.predict_proba(x) for m in self.members])
        averaged = np.tensordot(self.weights, stacked, axes=(0, 0))
        # Members can disagree enough that rounding leaves the row slightly off
        # one; renormalising keeps the contract the rest of the pipeline relies
        # on rather than letting a 0.9999 row quietly fail a threshold.
        total = averaged.sum(axis=1, keepdims=True)
        return np.divide(averaged, total, out=np.zeros_like(averaged), where=total > 0)

    def describe(self) -> dict[str, Any]:
        out = super().describe()
        out.update(
            {
                "members": [m.describe() for m in self.members],
                "weights": self.weights.tolist(),
            }
        )
        return out


def build_ensemble(names: Sequence[str], **params: Any) -> EnsembleModel:
    """Build an ensemble from model names, as a config would name them."""
    from trading_research.pipeline.stages import build_model

    return EnsembleModel([build_model(name, **params.get(name, {})) for name in names])
