"""Strategies that use what a target actually gives them.

A three-class direction target and a regression on basis points carry different
information, and the pipeline until now threw most of it away: every prediction
was reduced to "which way, and how confident", one unit was traded, and the exit
was a clock. That wastes the regression targets entirely — a number in basis
points is an expected value, and an expected value can be compared with a cost
and used to size a position — and it wastes the barrier target, which knows the
trading rule and was then traded with a different one.

Three constructions, each matched to what its target knows.

Meta-labelling
--------------
The standard two-stage arrangement. A **primary** decides the direction; a
**secondary**, trained on whether the primary's trades actually won, decides
whether to take this one. The division of labour is the point: finding
direction and judging a specific trade are different problems, and §14 found
the best direction-finder here is a rule with no parameters while the models
were better at neither.

The secondary's target is binary — did the primary's trade at this moment make
money — which is what the triple-barrier label computes when its barriers are
the trading rule's. It is fitted only on the moments the primary wanted to
trade, so it never sees the 99% of rows where nothing was going to happen, and
its positive class is balanced in a way the raw target never is.

Expected value
--------------
For a regression target the decision needs no threshold at all:

    trade when  |predicted edge|  >  cost * margin

The threshold everywhere else in this project is swept on validation because a
probability has no natural scale. A prediction in basis points has one — the
cost is in the same units — so the rule is written down rather than fitted, and
one degree of freedom disappears from the search.

Size then follows from the same number: a trade expected to make twice the cost
carries twice the position of one expected to make exactly the cost, capped so
that a single confident prediction cannot become the whole book.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trading_research.models.base import CLASSES, Model

BUY_COLUMN = CLASSES.index(1)
SELL_COLUMN = CLASSES.index(-1)
HOLD_COLUMN = CLASSES.index(0)


def direction_of(proba: np.ndarray) -> np.ndarray:
    """Which way a set of class probabilities points."""
    return np.where(proba[:, BUY_COLUMN] >= proba[:, SELL_COLUMN], 1, -1)


class MetaLabelled(Model):
    """A primary picks the side; a secondary decides whether to act.

    ``primary`` supplies direction and nothing else — its confidence is
    discarded, which is deliberate. If the primary's confidence were also used
    the two stages would be answering the same question twice, and the
    secondary's job is to answer a different one: *given* that we would trade
    here, does that trade win.

    The secondary is fitted only on the moments the primary wanted, using a
    binary label built from the realised outcome of those trades. That is what
    makes the arrangement worth the extra model — it turns a 99%-HOLD problem
    into a roughly balanced one, and it lets a weak direction signal be
    combined with a separate judgement about when to trust it.
    """

    def __init__(
        self,
        primary: Model,
        secondary: Model,
        *,
        min_primary_confidence: float = 0.0,
    ) -> None:
        super().__init__(name=f"meta({primary.name}->{secondary.name})")
        self.primary = primary
        self.secondary = secondary
        self.min_primary_confidence = min_primary_confidence
        self.acted_share_: float = float("nan")

    def fit(self, x: pd.DataFrame, y: pd.Series) -> MetaLabelled:
        """Fit the primary on the direction target, the secondary on outcomes.

        ``y`` is the ordinary three-class target. The secondary's label is
        derived from it: on the rows the primary wanted to trade, did the
        target agree with the primary's direction.
        """
        self.primary.fit(x, y)
        side = direction_of(self.primary.predict_proba(x))

        target = y.to_numpy(dtype="float64")
        wanted = np.isfinite(target)
        if self.min_primary_confidence > 0:
            strength = np.max(self.primary.predict_proba(x)[:, [SELL_COLUMN, BUY_COLUMN]], axis=1)
            wanted &= strength >= self.min_primary_confidence

        # Won means the target moved the way the primary pointed. A target of
        # zero — the move did not clear the cost — is a loss, not a neutral:
        # the trade was taken and paid its costs for nothing.
        won = (np.sign(target) == side) & (target != 0)
        self.acted_share_ = float(wanted.mean())

        if wanted.sum() < 200 or len(np.unique(won[wanted])) < 2:
            raise ValueError(
                f"the primary produced {int(wanted.sum())} usable moments with "
                f"{len(np.unique(won[wanted]))} outcome classes; the secondary needs both"
            )

        # Trained on the primary's moments only. Fitting it everywhere would
        # teach it about rows the strategy will never be at.
        self.secondary.fit(
            x.loc[wanted], pd.Series(won[wanted].astype(float), index=x.index[wanted])
        )
        self.fitted_ = True
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        side = direction_of(self.primary.predict_proba(x))
        secondary = self.secondary.predict_proba(x)
        # The secondary was fitted on a {0, 1} target, so its "buy" column is
        # the probability that the primary's trade wins, whichever way it faces.
        confidence = secondary[:, BUY_COLUMN]

        proba = np.zeros((len(x), len(CLASSES)))
        up = side > 0
        proba[up, BUY_COLUMN] = confidence[up]
        proba[~up, SELL_COLUMN] = confidence[~up]
        proba[:, HOLD_COLUMN] = 1.0 - confidence
        return proba

    def describe(self) -> dict[str, Any]:
        out = super().describe()
        out.update(
            {
                "primary": self.primary.describe(),
                "secondary": self.secondary.describe(),
                "acted_share": self.acted_share_,
            }
        )
        return out


class ExpectedValue(Model):
    """Trade a regression prediction directly, with no fitted threshold.

    The rule is arithmetic rather than a hyperparameter:

        act when |predicted edge| > cost * margin

    A prediction in basis points is on the same scale as the cost, so nothing
    needs sweeping. ``margin`` above one asks for a buffer — trading only where
    the prediction beats the cost by a stated factor — which is a stated
    assumption rather than a number found by searching the validation block.

    ``predict_proba`` maps the surplus over cost onto a confidence so the rest
    of the pipeline is unchanged, but the decision has already been made by then
    and no threshold downstream can improve it.
    """

    def __init__(self, regressor: Model, *, cost_bp: float, margin: float = 1.0) -> None:
        super().__init__(name=f"ev({regressor.name}, x{margin:g})")
        if cost_bp <= 0:
            raise ValueError(f"cost_bp must be positive, got {cost_bp}")
        if margin <= 0:
            raise ValueError(f"margin must be positive, got {margin}")
        self.regressor = regressor
        self.cost_bp = cost_bp
        self.margin = margin

    @property
    def hurdle_bp(self) -> float:
        return self.cost_bp * self.margin

    def fit(self, x: pd.DataFrame, y: pd.Series) -> ExpectedValue:
        self.regressor.fit(x, y)
        self.fitted_ = True
        return self

    def predict_edge_bp(self, x: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        if hasattr(self.regressor, "predict_edge_bp"):
            return np.asarray(self.regressor.predict_edge_bp(x), dtype="float64")
        raise TypeError(f"{self.regressor.name} does not predict basis points")

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        edge = self.predict_edge_bp(x)
        acts = np.abs(edge) > self.hurdle_bp
        # Confidence is how far past the hurdle the prediction is, squashed so
        # it stays inside the range a threshold could exclude.
        surplus = np.tanh(np.clip(np.abs(edge) - self.hurdle_bp, 0, None) / max(self.cost_bp, 1e-9))
        confidence = np.where(acts, 1.0 / 3.0 + (0.9 - 1.0 / 3.0) * surplus, 1.0 / 3.0)

        proba = np.zeros((len(edge), len(CLASSES)))
        up = edge >= 0
        proba[up, BUY_COLUMN] = confidence[up]
        proba[~up, SELL_COLUMN] = confidence[~up]
        remainder = 1.0 - proba.sum(axis=1)
        proba[:, HOLD_COLUMN] = remainder
        return proba

    def position_size(self, x: pd.DataFrame, *, cap: float = 3.0) -> np.ndarray:
        """How large a position the prediction justifies, in units.

        Proportional to the surplus over the hurdle, so a trade expected to make
        twice the cost carries twice the position of one expected to make
        exactly the cost. Capped, because the largest predictions are the least
        reliable — they are where the model is extrapolating — and an uncapped
        rule puts the most weight exactly there.
        """
        edge = np.abs(self.predict_edge_bp(x))
        return np.clip((edge - self.hurdle_bp) / max(self.cost_bp, 1e-9), 0.0, cap)

    def describe(self) -> dict[str, Any]:
        out = super().describe()
        out.update(
            {
                "cost_bp": self.cost_bp,
                "margin": self.margin,
                "hurdle_bp": self.hurdle_bp,
                "regressor": self.regressor.describe(),
            }
        )
        return out
