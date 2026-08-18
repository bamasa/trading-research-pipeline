"""Rule-based strategies, with no fitting at all.

A repository that only compares machine-learning models answers a narrower
question than it appears to. The ones below are what a desk would actually run
first: they have no parameters to fit beyond a window length, they have been
public for decades, and if a learned model cannot beat them then the learning is
not what is producing whatever result there is.

They are wrapped as models rather than kept as a separate code path, so they go
through the identical confidence threshold, thinning, cost model and
walk-forward as everything else. A comparison where the baseline is scored by
different machinery is not a comparison.

What "probability" means here
-----------------------------
Each rule produces a signal strength in [-1, 1] which is mapped onto the three
class probabilities. That is a convenience, not a claim: the number is a
normalised indicator, not a calibrated belief, and it should not be read as one.
What it buys is that the same threshold sweep that selects the strongest
learned signals also selects the strongest rule signals, so "trade only when the
indicator is extreme" needs no special handling.

``fit`` records what the rule needs from the training window — a scale for
normalisation — and nothing else. None of these look at the label.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trading_research.models.base import CLASSES, Model

#: Probability assigned to each class when a rule has no opinion.
NEUTRAL = 1.0 / 3.0

#: Probability at full signal strength. Chosen to sit inside the threshold
#: sweep's range so that a rule can be made as selective as a learned model.
MAX_CONFIDENCE = 0.9


def _to_proba(strength: np.ndarray) -> np.ndarray:
    """Map a signal in [-1, 1] onto three class probabilities.

    Interpolates from one third at zero signal — no opinion, all three classes
    equal — up to ``MAX_CONFIDENCE`` at full strength. The ceiling is below one
    on purpose: the threshold sweep runs to 0.95, and a rule whose strongest
    signal sat at 1.0 could never be excluded by any threshold, which would make
    it incomparable with the learned models rather than merely better.
    """
    strength = np.clip(np.nan_to_num(strength, nan=0.0), -1.0, 1.0)
    magnitude = np.abs(strength)
    directional = NEUTRAL + (MAX_CONFIDENCE - NEUTRAL) * magnitude
    remainder = 1.0 - directional

    proba = np.zeros((len(strength), len(CLASSES)))
    up = strength >= 0
    proba[up, CLASSES.index(1)] = directional[up]
    proba[up, CLASSES.index(-1)] = remainder[up] / 2
    proba[~up, CLASSES.index(-1)] = directional[~up]
    proba[~up, CLASSES.index(1)] = remainder[~up] / 2
    proba[:, CLASSES.index(0)] = 1.0 - proba.sum(axis=1)
    return proba


class RuleModel(Model):
    """Shared plumbing: normalise a raw signal by its training-window scale."""

    #: Feature the rule reads. Set by each subclass.
    column: str = ""

    #: Normalised magnitude below which the rule says nothing. Zero means the
    #: rule always has an opinion, however faint. A deadzone is what makes a
    #: rule selective, and it belongs after normalisation rather than inside
    #: the raw signal — a threshold on a raw feature means something different
    #: on every instrument.
    deadzone: float = 0.0

    def __init__(self, name: str, *, column: str | None = None) -> None:
        super().__init__(name=name)
        if column is not None:
            self.column = column
        self.scale_: float = 1.0

    def signal(self, x: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def fit(self, x: pd.DataFrame, y: pd.Series) -> RuleModel:  # noqa: ARG002
        # The label is deliberately ignored. All the training window provides is
        # a scale, so that "large" means the same thing across instruments.
        raw = self.signal(x)
        finite = raw[np.isfinite(raw)]
        # A robust scale: two standard deviations of the training window puts
        # the bulk of the signal inside [-1, 1] without one outlier setting it.
        self.scale_ = float(np.std(finite)) * 2.0 if finite.size else 1.0
        if self.scale_ <= 0:
            self.scale_ = 1.0
        self.fitted_ = True
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        normalised = self.signal(x) / self.scale_
        if self.deadzone > 0:
            normalised = np.where(np.abs(normalised) >= self.deadzone, normalised, 0.0)
        return _to_proba(normalised)

    def describe(self) -> dict[str, Any]:
        out = super().describe()
        out.update(
            {
                "kind": "rule",
                "column": self.column,
                "scale": self.scale_,
                "deadzone": self.deadzone,
            }
        )
        return out

    def _column(self, x: pd.DataFrame) -> np.ndarray:
        if self.column not in x.columns:
            raise KeyError(f"{self.name} needs the column {self.column!r}")
        return x[self.column].to_numpy(dtype="float64")


class Momentum(RuleModel):
    """Trade in the direction the price has just moved.

    The oldest idea in the book, and the one most likely to be already priced.
    Reads a recent return: if the last stretch went up, go long.
    """

    column = "log_mid_ret20"

    def __init__(self, column: str | None = None) -> None:
        super().__init__("momentum", column=column)

    def signal(self, x: pd.DataFrame) -> np.ndarray:
        return self._column(x)


class MeanReversion(RuleModel):
    """Trade against a move away from the recent average.

    The mirror of momentum, and at short horizons in a liquid book usually the
    better-supported story: an isolated print that pushes the mid away from
    where trading has been happening tends to come back.
    """

    column = "log_mid_ret20"

    def __init__(self, column: str | None = None) -> None:
        super().__init__("mean_reversion", column=column)

    def signal(self, x: pd.DataFrame) -> np.ndarray:
        return -self._column(x)


class OrderFlowRule(RuleModel):
    """Trade with the imbalance at the touch.

    The one microstructure rule that needs no model, and the strongest single
    feature in this study: queue imbalance correlates with the next second's
    return at about 0.27. This is that correlation traded directly, and it is
    the honest baseline for any learned model built on the same book.
    """

    column = "queue_imbalance"

    def __init__(self, column: str | None = None) -> None:
        super().__init__("order_flow", column=column)

    def signal(self, x: pd.DataFrame) -> np.ndarray:
        return self._column(x)


class Breakout(RuleModel):
    """Trade a move that clears the recent range.

    Distinct from momentum in what it ignores: small moves inside the range
    produce nothing, so it trades rarely and only on the larger moves — which
    is the shape a cost floor rewards, whatever the direction turns out to be.
    """

    column = "log_mid_ret50"

    #: Divided by this to make the move comparable across regimes: a 5 bp move
    #: is a breakout in a calm hour and nothing in a volatile one.
    scale_column = "log_mid_vol50"

    #: Only the tail counts. Inside the deadzone there is no range to have
    #: broken out of, and the rule stands aside rather than trading noise.
    deadzone = 1.0

    def __init__(self, column: str | None = None) -> None:
        super().__init__("breakout", column=column)

    def signal(self, x: pd.DataFrame) -> np.ndarray:
        move = self._column(x)
        if self.scale_column in x.columns:
            volatility = x[self.scale_column].to_numpy(dtype="float64")
            with np.errstate(divide="ignore", invalid="ignore"):
                move = np.where(volatility > 0, move / volatility, 0.0)
        return move


class SpreadCapture(RuleModel):
    """Trade only when the spread is unusually tight, in the imbalance direction.

    Not a maker strategy — everything here still crosses — but it encodes the
    same intuition: the cost of a round trip is the binding constraint, so act
    when it is temporarily smallest. Included because it is the cheapest test of
    whether cost timing alone can matter.
    """

    column = "queue_imbalance"

    def __init__(self, column: str | None = None) -> None:
        super().__init__("spread_capture", column=column)

    def signal(self, x: pd.DataFrame) -> np.ndarray:
        imbalance = self._column(x)
        if "spread_bp" not in x.columns:
            return imbalance
        spread = x["spread_bp"].to_numpy(dtype="float64")
        tight = spread <= np.nanquantile(spread, 0.3)
        return np.where(tight, imbalance, 0.0)
