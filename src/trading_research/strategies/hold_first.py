"""The hold-first decision rule of stratum C, with one threshold per side.

A three-class model gives ``p_long``, ``p_hold`` and ``p_short``. The rule goes
long when ``p_long >= theta_long`` and ``p_long > p_short``, short when
``p_short >= theta_short`` and ``p_short > p_long``, and holds otherwise. Hold
is the default: a side trades only when its own probability clears its own
threshold, whatever ``p_hold`` is.

The thresholds are set from a rate, not tuned on outcomes. On the inner
validation, before the market gate, the regime guard and the one-position rule,
each side's threshold is the value at which that side would enter on half of
the target number of trades a day: the ``k``-th largest ``p_long`` among the
rows where ``p_long > p_short``, ``k`` being half the target rate times the
validation's days. The rule then reads only probabilities, so the thresholds
see no label of the validation rows, and the gate and the guard can only remove
decision rows: a gated or guarded cell trades less than its target.

A side whose class is missing from a fit has probability 0 everywhere, has no
candidate row, and gets an infinite threshold: it never trades with that model.

Thresholds come from inner validation rows inside the development block only
(the pattern of :func:`trading_research.labels.directional.threshold_for_share`).
Choosing the rate itself is the search's job, by neighbourhood, not here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading_research.labels.directional import BUY, HOLD, SELL
from trading_research.models.base import CLASSES


def count_days(timestamps: pd.Series | Sequence[pd.Timestamp]) -> int:
    """Distinct UTC calendar days among the validation rows."""
    stamps = pd.DatetimeIndex(pd.to_datetime(pd.Series(timestamps), utc=True))
    return int(stamps.normalize().nunique())


def side_threshold(p_side: np.ndarray, p_other: np.ndarray, entries: int) -> float:
    """The threshold at which ``entries`` rows would enter on this side."""
    p_side = np.asarray(p_side, dtype="float64")
    p_other = np.asarray(p_other, dtype="float64")
    candidates = p_side[np.isfinite(p_side) & (p_side > p_other)]
    if entries <= 0 or len(candidates) == 0:
        return float("inf")
    ordered = np.sort(candidates)[::-1]
    return float(ordered[min(entries, len(ordered)) - 1])


@dataclass(frozen=True)
class HoldFirstRule:
    """Per-side thresholds on the class probabilities."""

    theta_long: float
    theta_short: float

    def decide(self, p_long: np.ndarray, p_short: np.ndarray) -> np.ndarray:
        """+1 long, -1 short, 0 hold, for each row."""
        p_long = np.asarray(p_long, dtype="float64")
        p_short = np.asarray(p_short, dtype="float64")
        out = np.full(len(p_long), HOLD, dtype=np.int8)
        out[(p_long >= self.theta_long) & (p_long > p_short)] = BUY
        out[(p_short >= self.theta_short) & (p_short > p_long)] = SELL
        return out

    def decide_proba(self, proba: np.ndarray) -> np.ndarray:
        """The same, from an ``(n, 3)`` array in the models' class order."""
        proba = np.asarray(proba, dtype="float64")
        return self.decide(proba[:, CLASSES.index(BUY)], proba[:, CLASSES.index(SELL)])

    def entries_per_day(self, proba: np.ndarray, days: float) -> tuple[float, float]:
        """Long and short decision rows a day, before any gate or position rule."""
        sides = self.decide_proba(proba)
        return float((sides == BUY).sum() / days), float((sides == SELL).sum() / days)

    def to_dict(self) -> dict[str, float]:
        return {"theta_long": self.theta_long, "theta_short": self.theta_short}


def fit_hold_first(proba: np.ndarray, *, days: float, trades_per_day: float) -> HoldFirstRule:
    """Thresholds giving each side half of ``trades_per_day`` on the inner validation."""
    if days <= 0:
        raise ValueError(f"days must be positive, got {days}")
    if trades_per_day <= 0:
        raise ValueError(f"trades_per_day must be positive, got {trades_per_day}")
    proba = np.asarray(proba, dtype="float64")
    entries = max(round(trades_per_day / 2.0 * days), 1)
    p_long = proba[:, CLASSES.index(BUY)]
    p_short = proba[:, CLASSES.index(SELL)]
    return HoldFirstRule(
        theta_long=side_threshold(p_long, p_short, entries),
        theta_short=side_threshold(p_short, p_long, entries),
    )
