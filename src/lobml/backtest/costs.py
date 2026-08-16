"""Transaction costs, in basis points.

One cost model, used in three places: when labelling, when deciding whether a
prediction is worth acting on, and when computing profit and loss. Backtests
routinely overstate results by labelling against a mid-price move that the
spread would have eaten, so the three must agree by construction rather than by
someone remembering to keep them in step.

Taker only
----------
This project models taking liquidity. Every entry and every exit crosses the
spread and pays the taker fee.

That is the harder assumption and the honest one for a directional model. A
maker strategy is not simply cheaper: a resting order fills only when someone
trades against it, which happens preferentially when the market is about to
move through it. Modelling that needs queue position and adverse-selection
assumptions this project does not have data for, and quoting maker economics
without them would flatter every result.

Published Binance USD-M futures rates are 0.02% maker and 0.05% taker — 2 and 5
basis points respectively. Taking on both sides therefore costs 10 bp before
the spread is paid at all, which is the single most important number in this
repository: it is what any predicted move has to clear before the prediction is
worth anything.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

#: Binance USD-M futures taker fee at the standard tier, in basis points.
BINANCE_UM_TAKER_BP: float = 5.0


@dataclass(frozen=True)
class TakerCosts:
    """Round-trip cost of entering and leaving a position by taking liquidity.

    All figures are basis points of notional.

    ``fee_bp_per_side``
        Exchange fee. Charged on entry and on exit.
    ``slippage_bp``
        Everything the quoted book does not capture: the price moving between
        the decision and the order arriving, and the order walking past the top
        level when it is larger than what rests there. Charged per side.
    ``half_spread_multiplier``
        How much of the quoted spread a crossing order pays. One means the
        order crosses fully — buy at the ask, sell at the bid — which is the
        default because that is what taking liquidity means.
    """

    fee_bp_per_side: float = BINANCE_UM_TAKER_BP
    slippage_bp: float = 0.5
    half_spread_multiplier: float = 1.0

    def __post_init__(self) -> None:
        for name in ("fee_bp_per_side", "slippage_bp", "half_spread_multiplier"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative, got {getattr(self, name)}")

    def round_trip_bp(self, spread_bp: float | pd.Series) -> float | pd.Series:
        """Total cost of a round trip, given the spread prevailing at the time.

        The spread is charged once, not twice. Entering pays half of it against
        mid, and exiting on the other side pays the other half, so a full round
        trip costs one spread — the common mistake of charging it twice
        overstates costs about as badly as ignoring it understates them.
        """
        return (
            2.0 * self.fee_bp_per_side
            + self.half_spread_multiplier * spread_bp
            + 2.0 * self.slippage_bp
        )

    def entry_bp(self, spread_bp: float | pd.Series) -> float | pd.Series:
        """Cost of the entry leg alone, against mid."""
        return (
            self.fee_bp_per_side + 0.5 * self.half_spread_multiplier * spread_bp + self.slippage_bp
        )

    def describe(self) -> dict[str, object]:
        return {
            "execution": "taker",
            "fee_bp_per_side": self.fee_bp_per_side,
            "slippage_bp": self.slippage_bp,
            "half_spread_multiplier": self.half_spread_multiplier,
            "note": (
                "Taker on both legs. Maker economics are not modelled: a resting "
                "order fills preferentially when the market is about to move "
                "through it, and that adverse selection cannot be estimated from "
                "this data."
            ),
        }


def breakeven_share(forward_move_bp: pd.Series, cost_bp: float | pd.Series) -> float:
    """Share of moments whose future move is larger than the cost of trading.

    An upper bound on how often a strategy could possibly be right to trade,
    and it assumes the direction is predicted perfectly. Useful as a reality
    check before any model is fitted: if only a small fraction of moments could
    pay for themselves under perfect foresight, no classifier is going to
    rescue the horizon.
    """
    move = forward_move_bp.abs()
    return float((move > cost_bp).mean())
