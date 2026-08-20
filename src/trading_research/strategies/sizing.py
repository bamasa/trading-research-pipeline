"""What to do with an edge estimate, once you have one.

A direction and a threshold is the crudest possible use of a forecast: act, or
do not. The systematic literature spends most of its attention on the next
question — given an estimate of the edge and an estimate of how uncertain it is,
how large should the position be — and that question has answers that do not
require the forecast to be any better than it already is.

Three of them are here.

**Volatility scaling.** Position size inversely proportional to predicted
volatility, so each trade carries the same risk rather than the same notional.
This does not improve the average trade; it improves the *ratio* by stopping
the noisiest moments from dominating the variance. It is the single most widely
used position rule in systematic trading and it costs nothing to apply.

**Confidence scaling.** Size proportional to the model's own estimate of its
edge, so a marginal signal takes a small position and a strong one takes a
large one. Where the model's confidence is genuinely informative this converts
ranking skill into money without needing better classification.

**Fractional Kelly.** Size proportional to edge over variance, capped. The
uncapped version is famously fragile to estimation error in exactly the way a
noisy edge estimate is worst; the fraction is the standard defence and is
mandatory here rather than optional.

What none of them fix
---------------------
Every rule below is multiplicative in the edge. If the expected edge per trade
is negative after costs, scaling it changes how fast money is lost and not
whether. Sizing is a way to earn more from an edge that exists — it is never a
way to manufacture one, and this module is documented that way because the
temptation to read it otherwise is exactly how leverage ruins people.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class SizingError(ValueError):
    """The sizing rule was given parameters it cannot honour."""


@dataclass(frozen=True)
class VolatilityTarget:
    """Scale positions so each carries the same predicted risk.

    ``target_bp`` is the risk per trade the book is willing to run, measured as
    the predicted move over the holding period. ``max_leverage`` caps the result
    so a quiet forecast cannot ask for an unbounded position — the failure mode
    of every volatility-scaled strategy is a period of unusually low predicted
    volatility immediately before a jump.
    """

    target_bp: float = 10.0
    max_leverage: float = 3.0
    floor_bp: float = 1.0

    def __post_init__(self) -> None:
        if self.target_bp <= 0:
            raise SizingError(f"target_bp must be positive, got {self.target_bp}")
        if self.max_leverage <= 0:
            raise SizingError(f"max_leverage must be positive, got {self.max_leverage}")

    def scale(self, predicted_volatility_bp: np.ndarray) -> np.ndarray:
        """Multiplier per row, in [0, max_leverage]."""
        volatility = np.maximum(np.asarray(predicted_volatility_bp, dtype="float64"), self.floor_bp)
        return np.clip(self.target_bp / volatility, 0.0, self.max_leverage)


@dataclass(frozen=True)
class ConfidenceScaled:
    """Size on the model's own estimate of how strong the signal is.

    ``floor`` is the confidence at which the position is zero and ``ceiling``
    the one at which it is full, so the rule is a ramp rather than a step. Below
    the floor nothing is taken, which makes this a threshold rule as well as a
    sizing rule.
    """

    floor: float = 0.34
    ceiling: float = 0.60
    max_leverage: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.floor < self.ceiling <= 1.0:
            raise SizingError(f"need 0 <= floor < ceiling <= 1, got {self.floor}, {self.ceiling}")

    def scale(self, confidence: np.ndarray) -> np.ndarray:
        ramp = (np.asarray(confidence, dtype="float64") - self.floor) / (self.ceiling - self.floor)
        return np.clip(ramp, 0.0, 1.0) * self.max_leverage


@dataclass(frozen=True)
class FractionalKelly:
    """Edge over variance, scaled down and capped.

    The full Kelly fraction maximises long-run growth *given the true edge*, and
    an edge estimated from a few thousand noisy trades is not the true edge.
    Overestimating it by a factor of two under full Kelly is enough to turn
    growth into ruin, so ``fraction`` is applied and ``max_leverage`` bounds the
    result regardless.
    """

    fraction: float = 0.25
    max_leverage: float = 2.0

    def __post_init__(self) -> None:
        if not 0.0 < self.fraction <= 1.0:
            raise SizingError(f"fraction must be in (0, 1], got {self.fraction}")

    def scale(self, edge_bp: np.ndarray, volatility_bp: np.ndarray) -> np.ndarray:
        edge = np.asarray(edge_bp, dtype="float64")
        variance = np.maximum(np.asarray(volatility_bp, dtype="float64"), 1e-9) ** 2
        raw = self.fraction * edge / variance
        # Negative edge means no position, not a reversed one: the sign of the
        # trade is the strategy's business, and this rule only sets size.
        return np.clip(raw, 0.0, self.max_leverage)


def apply_scale(net_bp: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Scale realised per-trade results, with the identity that matters stated.

    A multiplier changes the magnitude of each outcome and never its sign, so
    the mean of the scaled series is positive only when the underlying edge is.
    Any apparent improvement from sizing is a change in the *distribution* —
    better ratio, smaller drawdown — not a change in whether the strategy makes
    money.
    """
    net = np.asarray(net_bp, dtype="float64")
    factor = np.asarray(scale, dtype="float64")
    if len(net) != len(factor):
        raise SizingError(f"net has {len(net)} rows, scale has {len(factor)}")
    return net * factor
