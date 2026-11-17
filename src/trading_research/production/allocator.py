"""The balancer: how much of the capital each running strategy may put at risk.

Weights are inverse to each strategy's volatility per round trip, so every
strategy contributes about the same risk; a strategy whose state is
``reduced`` counts at half its weight and a ``halted`` one at none. No weight
exceeds ``max_weight``, and the weights sum to ``max_gross`` while any
strategy may trade. The cap is applied by water-filling: a capped strategy is
fixed at the cap and the remainder is spread over the others in the same
proportions, until no weight is over it; when every strategy is at the cap
the remainder stays undeployed rather than breaking it.

The volatility is the backtest's ``sd_bp`` until a strategy has closed
``min_trades`` round trips live, and the live figure after; a strategy is
never sized up because it has been lucky, only because it has been calm.

The result is a scale on each strategy's position limit: the clip and the
limit the backtest validated are the scale's 1.0 at an equal split, and the
balancer moves the limit, never the clip.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

STATE_FACTOR = {"running": 1.0, "reduced": 0.5, "halted": 0.0}


@dataclass(frozen=True)
class Allocation:
    name: str
    weight: float
    capital: float
    #: The strategy's position limit relative to an equal split of ``max_gross``.
    scale: float
    sd_bp: float
    state: str


def allocate(
    sd_bp: Mapping[str, float],
    states: Mapping[str, str],
    *,
    capital: float,
    max_gross: float,
    max_weight: float,
) -> dict[str, Allocation]:
    """Weights, capital and limit scales for every strategy in ``sd_bp``."""
    if not sd_bp:
        return {}
    for name, sd in sd_bp.items():
        if not sd > 0:
            raise ValueError(f"{name}: volatility must be positive, got {sd}")
        if states.get(name, "running") not in STATE_FACTOR:
            raise ValueError(f"{name}: unknown state {states.get(name)!r}")
    raw = {name: STATE_FACTOR[states.get(name, "running")] / sd for name, sd in sd_bp.items()}
    weights = dict.fromkeys(raw, 0.0)
    free = {name for name, value in raw.items() if value > 0}
    budget = max_gross
    while free and budget > 1e-12:
        total = sum(raw[name] for name in free)
        share = {name: budget * raw[name] / total for name in free}
        over = {name for name, value in share.items() if value > max_weight}
        if not over:
            weights.update(share)
            break
        for name in over:
            weights[name] = max_weight
            budget -= max_weight
        free -= over
    equal = max_gross / len(sd_bp)
    return {
        name: Allocation(
            name=name,
            weight=weights[name],
            capital=weights[name] * capital,
            scale=weights[name] / equal,
            sd_bp=sd_bp[name],
            state=states.get(name, "running"),
        )
        for name in sd_bp
    }
