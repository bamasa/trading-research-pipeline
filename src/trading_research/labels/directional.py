"""Directional labels.

A label answers "what should have been done here?", which means it necessarily
depends on the future. That is the one place in the pipeline where reading
forward is correct, and it is why labels live apart from features: a feature
that looked forward would be a bug, and keeping the two in separate modules
means the leakage checker can assert opposite properties of each without
special cases.

The label declares its ``horizon``, and that number does real work. The split
uses it to purge each block's tail, because the last *H* rows of a training
block have labels determined by prices in the block after it. Deriving the
purge from the label rather than from a config field is what keeps the two from
drifting apart when the horizon changes.

Threshold and costs
-------------------
A three-class target with a threshold — sell, hold, buy — says nothing useful
unless the threshold is at least the cost of a round trip. Below that, "the
price rose by more than the threshold" and "acting on it would have made money"
are different statements, and a model that predicts the first perfectly still
loses money.

The threshold is therefore a parameter here, and
:func:`round_trip_cost_bp` computes the floor it has to clear. Results are
reported across a range of thresholds rather than at one tuned value, since the
shape of that curve says more than any single point on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from trading_research.data.schema import mid_price

BP: Final = 1e4

#: Class encoding. Kept as -1/0/+1 rather than 0/1/2 so that the sign of the
#: label is the direction of the trade, which makes position arithmetic in the
#: backtest read as arithmetic rather than as a lookup.
SELL: Final = -1
HOLD: Final = 0
BUY: Final = 1

CLASS_NAMES: Final = {SELL: "sell", HOLD: "hold", BUY: "buy"}


def round_trip_cost_bp(
    fee_bp_per_side: float, slippage_bp: float, spread_bp: float | pd.Series
) -> pd.Series | float:
    """Cost of entering and leaving a position, in basis points.

    Two sides of fees, the spread paid once on the way in for a taker, and
    slippage on each side. This is the floor a labelling threshold has to clear
    before "the price moved" means "money could have been made".
    """
    return 2.0 * fee_bp_per_side + spread_bp + 2.0 * slippage_bp


@dataclass(frozen=True)
class DirectionalLabel:
    """Forward mid return over ``horizon`` observations, cut into three classes.

    ``horizon`` is in observations, not wall-clock time. On a 100 ms grid a
    horizon of 20 is two seconds; on raw updates it is however long twenty
    updates take. The grid is recorded in the dataset manifest so the two are
    never confused.
    """

    horizon: int
    threshold_bp: float

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError(f"horizon must be at least 1, got {self.horizon}")
        if self.threshold_bp < 0:
            raise ValueError(f"threshold_bp must be non-negative, got {self.threshold_bp}")

    @property
    def purge(self) -> int:
        """Rows a split must drop from each block tail.

        Exactly the horizon: a label at row *t* is determined by the price at
        *t + horizon*, so the last ``horizon`` rows of a block were decided by
        the block that follows.
        """
        return self.horizon

    def forward_return_bp(self, book: pd.DataFrame) -> pd.Series:
        """Log return of mid from now to ``horizon`` observations ahead, in bp.

        The final ``horizon`` rows are ``NaN``: their outcome is not in the
        data. Left as ``NaN`` rather than filled, so they are dropped knowingly
        instead of being trained on as though the price had not moved.
        """
        mid = mid_price(book)
        future = mid.shift(-self.horizon)
        return pd.Series(BP * np.log(future.to_numpy() / mid.to_numpy()), index=book.index)

    def __call__(self, book: pd.DataFrame) -> pd.Series:
        """Return the three-class label as -1 / 0 / +1, with NaN where unknown."""
        move = self.forward_return_bp(book)
        label = pd.Series(np.nan, index=book.index, dtype="float64")
        label = label.mask(move.notna() & (move.abs() <= self.threshold_bp), float(HOLD))
        label = label.mask(move > self.threshold_bp, float(BUY))
        label = label.mask(move < -self.threshold_bp, float(SELL))
        return label.rename("label")

    def describe(self) -> dict[str, object]:
        return {
            "kind": "directional",
            "horizon": self.horizon,
            "threshold_bp": self.threshold_bp,
            "purge": self.purge,
            "classes": {str(k): v for k, v in CLASS_NAMES.items()},
        }


def class_balance(label: pd.Series) -> pd.Series:
    """Share of each class among labelled rows.

    Worth looking at before anything else. A threshold that leaves 99% HOLD
    makes accuracy meaningless and gives a classifier almost nothing to learn
    from, and that is a property of the threshold rather than of the market.
    """
    counts = label.dropna().value_counts(normalize=True)
    return counts.rename(index=lambda k: CLASS_NAMES[int(k)]).sort_index()


def threshold_for_share(book: pd.DataFrame, horizon: int, target_share: float) -> float:
    """Threshold that leaves roughly ``target_share`` of rows non-HOLD.

    Separates *how often to trade* from *when to trade*. Choosing a threshold
    directly conflates the two: a change that trades twice as often may look
    better simply because it traded more, and comparing two configurations at
    different trade frequencies compares mostly the frequencies.

    Fixing the frequency first and letting the threshold follow makes the
    comparison meaningful. Must be computed on training data only — it reads
    the distribution of outcomes, which the test period is not allowed to
    inform.
    """
    if not 0.0 < target_share < 1.0:
        raise ValueError(f"target_share must be in (0, 1), got {target_share}")
    move = DirectionalLabel(horizon=horizon, threshold_bp=0.0).forward_return_bp(book)
    return float(np.nanquantile(move.abs().to_numpy(), 1.0 - target_share))
