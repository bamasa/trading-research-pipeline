"""What one fill of one leg costs, read from the repository's fee tables.

A pair trade is four taker fills: each leg entered and each leg left. Every
fill pays the taker fee and a slippage, both in bp of that leg's notional, as
:class:`trading_research.backtest.costs.TakerCosts` charges one side
(:meth:`~trading_research.backtest.costs.TakerCosts.entry_bp`). Bars carry no
quoted spread, so the spread is not charged separately: ``slippage_bp`` is
everything between the bar's open and the price actually paid, the half-spread
included, and should be set with that in mind.

The fee comes from the tables the repository already holds, never from a
number of this module's own: Binance USD-M's taker fee
(:data:`~trading_research.backtest.costs.BINANCE_UM_TAKER_BP`, the default,
since the public bars are Binance's), or a named row of Bybit's linear
schedule through :func:`trading_research.fees.tier`, or a fee stated on the
command line.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from trading_research.backtest.costs import BINANCE_UM_TAKER_BP, TakerCosts


@dataclass(frozen=True)
class LegCosts:
    """Taker fee and slippage per fill, in bp of the leg's notional, and where the fee came from."""

    taker: TakerCosts
    source: str

    @property
    def per_fill_bp(self) -> float:
        """Fee plus slippage of one fill (no quoted spread on bars)."""
        return cast(float, self.taker.entry_bp(0.0))

    def round_trip_bp(self, beta: float) -> float:
        """Four fills, in bp of the first leg's notional: the second leg is ``|beta|`` of it."""
        return 2.0 * self.per_fill_bp * (1.0 + abs(beta))

    def describe(self) -> dict[str, object]:
        return {
            "fee_bp_per_fill": self.taker.fee_bp_per_side,
            "slippage_bp_per_fill": self.taker.slippage_bp,
            "cost_bp_per_fill": self.per_fill_bp,
            "fee_source": self.source,
            "fills_per_trade": 4,
        }


def leg_costs(
    *,
    fee_bp: float | None = None,
    tier: str | None = None,
    slippage_bp: float = 0.5,
) -> LegCosts:
    """The cost of a fill: a stated fee, a Bybit linear tier by name, or Binance USD-M's taker fee."""
    if fee_bp is not None and tier is not None:
        raise ValueError("state a fee or name a tier, not both")
    if tier is not None:
        from trading_research import fees

        row = fees.tier(tier)
        taker = TakerCosts(fee_bp_per_side=row.taker_bp, slippage_bp=slippage_bp)
        return LegCosts(taker, f"{row.venue} tier {row.tier} ({row.source.split(';')[0]})")
    if fee_bp is not None:
        return LegCosts(TakerCosts(fee_bp_per_side=fee_bp, slippage_bp=slippage_bp), "stated")
    return LegCosts(
        TakerCosts(fee_bp_per_side=BINANCE_UM_TAKER_BP, slippage_bp=slippage_bp),
        "binance_usdm_futures standard taker (backtest/costs.py BINANCE_UM_TAKER_BP)",
    )
