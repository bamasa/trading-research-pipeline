"""Rules and models combined, rather than compared.

§14 measured the two families against each other and the rules won. That framed
it as a choice, which is the wrong frame: a rule and a model see different
things about the same moment, and the useful question is what happens when both
are consulted.

Three ways to combine them, and they encode different beliefs about which one
knows what.

``Agreement``
    Trade only when every member points the same way. The rule supplies a
    direction it has held for decades; the model supplies a second opinion from
    the rest of the book. Disagreement means standing aside, which is the
    cheapest thing a strategy can do and the one this cost floor rewards.

``Gated``
    One member decides the direction, another decides whether to act. Built for
    the case where a rule is a better direction-finder than the model — which
    §14 says it is here — while the model is better at telling a moment worth
    trading from one that is not.

``Blend``
    A weighted average of probabilities, which is the ensemble in
    :mod:`trading_research.models.ensemble` with members of different kinds. It
    is the softest of the three: it dilutes a confident member with an unsure
    one rather than letting either veto.

What none of these can do is create signal. If both members are wrong at the
same moments — and fitted on the same book they largely are — combining them
produces a strategy that is wrong at those moments with more ceremony.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from trading_research.models.base import CLASSES, Model

BUY_COLUMN = CLASSES.index(1)
SELL_COLUMN = CLASSES.index(-1)
HOLD_COLUMN = CLASSES.index(0)


def _direction_and_confidence(proba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Which way a member points, and how strongly."""
    buy, sell = proba[:, BUY_COLUMN], proba[:, SELL_COLUMN]
    direction = np.where(buy >= sell, 1, -1)
    return direction, np.maximum(buy, sell)


def _from_direction(direction: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    """Rebuild a probability matrix from a direction and a strength."""
    confidence = np.clip(confidence, 0.0, 1.0)
    proba = np.zeros((len(direction), len(CLASSES)))
    up = direction > 0
    proba[up, BUY_COLUMN] = confidence[up]
    proba[~up, SELL_COLUMN] = confidence[~up]
    proba[:, HOLD_COLUMN] = 1.0 - confidence
    return proba


class CompositeModel(Model):
    """Shared plumbing for strategies built from other strategies."""

    def __init__(self, name: str, members: Sequence[Model]) -> None:
        if len(members) < 2:
            raise ValueError(f"a composite needs at least two members, got {len(members)}")
        super().__init__(name=name)
        self.members = list(members)

    def fit(self, x: pd.DataFrame, y: pd.Series) -> CompositeModel:
        for member in self.members:
            member.fit(x, y)
        self.fitted_ = True
        return self

    def describe(self) -> dict[str, Any]:
        out = super().describe()
        out["members"] = [m.describe() for m in self.members]
        return out


class Agreement(CompositeModel):
    """Act only where every member agrees on the direction.

    Confidence is the *weakest* member's, not the average. A chain is only as
    strong as its least convinced link, and averaging would let one very
    confident member drag a doubtful one over the threshold — which is the
    behaviour this class exists to prevent.
    """

    def __init__(self, members: Sequence[Model]) -> None:
        super().__init__("&".join(m.name for m in members), members)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        votes = [_direction_and_confidence(m.predict_proba(x)) for m in self.members]
        directions = np.stack([d for d, _ in votes])
        confidences = np.stack([c for _, c in votes])

        agreed = (directions == directions[0]).all(axis=0)
        # Where members disagree the composite says nothing: HOLD at the class
        # prior, which no threshold will act on.
        confidence = np.where(agreed, confidences.min(axis=0), 0.0)
        return _from_direction(directions[0], confidence)


class Gated(CompositeModel):
    """One member points, another decides whether the moment is worth it.

    ``direction_from`` supplies the side; ``gate_by`` supplies the confidence.
    The gate can still veto by being unsure, and it cannot flip the trade — it
    has no vote on direction, only on whether there is one.
    """

    def __init__(self, direction_from: Model, gate_by: Model) -> None:
        super().__init__(f"{direction_from.name}|{gate_by.name}", [direction_from, gate_by])

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        direction, _ = _direction_and_confidence(self.members[0].predict_proba(x))
        gate_direction, gate_confidence = _direction_and_confidence(
            self.members[1].predict_proba(x)
        )
        # The gate is only allowed to endorse. Where it points the other way its
        # confidence is not evidence for this trade, so it counts as zero.
        endorsed = gate_direction == direction
        return _from_direction(direction, np.where(endorsed, gate_confidence, 0.0))
