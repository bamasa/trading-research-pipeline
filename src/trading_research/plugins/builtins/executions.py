"""The execution modes of :mod:`trading_research.pipeline.execution`, registered as plugins.

Each adapter builds the mode's cost model from one row of the fee table
(:data:`trading_research.backtest.costs.BYBIT_LINEAR_TIERS`, the base tier by
default) and calls :func:`trading_research.pipeline.execution.execute`
unchanged. Decisions arrive per row of the market's frame, with the hold in
rows; the event-time modes receive them stamped at the rows' labels.

``taker``
    Cross on the decision row and cross back ``hold`` rows later, one position
    at a time (:func:`~trading_research.backtest.execution.thin`); a round trip
    pays two taker fees, the spread and slippage on both legs.
``passive``
    Post at the touch and fill once the visible queue ahead has traded through,
    on the grid (:func:`trading_research.backtest.maker.simulate`); maker fee in,
    a taker exit after the hold.
``maker``
    The same intent in event time, in the market-making simulator: one clip at
    the touch per decision, a queue per order, latency, fills only from prints
    under the simulator's fill rule (queue by default), the taker exit a hold
    after the fill (:class:`~trading_research.market_making.signal_entry.SignalEntryQuoter`).
``event_taker``
    The event-time taker: the clip walks the visible book at the decision plus
    latency and walks back after the hold (:func:`trading_research.backtest.taker.execute_day`).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from trading_research.backtest.costs import FeeTier, fee_tier
from trading_research.plugins.market import MarketData
from trading_research.plugins.registry import register_execution


def _tier(fees: FeeTier | str) -> FeeTier:
    return fees if isinstance(fees, FeeTier) else fee_tier(fees)


class _Execution:
    fill_rule = ""

    def __init__(self, *, fees: FeeTier | str = "base", slippage_bp: float = 0.5) -> None:
        self.fees = _tier(fees)
        self.slippage_bp = slippage_bp

    def describe(self) -> dict[str, Any]:
        return {
            "fill_rule": self.fill_rule,
            "fee_tier": self.fees.name,
            "maker_bp": self.fees.maker_bp,
            "taker_bp": self.fees.taker_bp,
            "slippage_bp": self.slippage_bp,
        }


@register_execution("taker")
class Taker(_Execution):
    """Cross on the decision and cross back after the hold, on the grid."""

    fill_rule = (
        "crosses at the touch on the decision row, back after the hold; one position at a time"
    )

    def __init__(self, *, cooldown: int = 0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.cooldown = cooldown

    def execute(self, decisions: np.ndarray, market: MarketData, *, hold: int) -> Any:
        from trading_research.backtest.costs import TakerCosts
        from trading_research.backtest.execution import ThinningRules
        from trading_research.pipeline.execution import execute

        return execute(
            market.signals(decisions, hold),
            "taker",
            market=market.grid_market(),
            grid_seconds=market.grid_seconds,
            taker=TakerCosts(fee_bp_per_side=self.fees.taker_bp, slippage_bp=self.slippage_bp),
            taker_rules=ThinningRules(hold_periods=hold, cooldown_periods=self.cooldown),
        )


@register_execution("passive")
class Passive(_Execution):
    """Post at the touch and fill when the queue ahead has traded, on the grid."""

    fill_rule = (
        "posts at the touch; fills once prints at the price consume the visible queue ahead; "
        "cancelled after the timeout; a taker exit after the hold"
    )

    def __init__(self, *, timeout: int = 60, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.timeout = timeout

    def execute(self, decisions: np.ndarray, market: MarketData, *, hold: int) -> Any:
        from trading_research.backtest.maker import MakerCosts, PostingRules
        from trading_research.pipeline.execution import execute

        costs = MakerCosts(
            fee_bp_per_side=self.fees.maker_bp,
            taker_fee_bp_per_side=self.fees.taker_bp,
            slippage_bp=self.slippage_bp,
        )
        return execute(
            market.signals(decisions, hold),
            "passive_entry",
            market=market.grid_market(),
            grid_seconds=market.grid_seconds,
            maker=costs,
            posting=PostingRules(timeout_rows=self.timeout, hold_rows=hold),
        )


@register_execution("maker")
class Maker(_Execution):
    """One clip at the touch per decision in the event-time simulator; a taker exit."""

    def __init__(
        self, *, clip: float = 0.01, rule: str = "queue", latency_ms: float = 10.0, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        from trading_research.market_making.queue import FillRule

        self.clip = clip
        self.rule = FillRule(rule)
        self.latency_ms = latency_ms
        self.fill_rule = (
            f"event time: one clip posted at the touch after the decision and latency; "
            f"fills from prints under the simulator's '{self.rule.value}' rule; "
            f"a taker exit the hold after the fill"
        )

    def execute(self, decisions: np.ndarray, market: MarketData, *, hold: int) -> Any:
        from trading_research.market_making.simulator import SimConfig
        from trading_research.pipeline.execution import execute

        latency = round(self.latency_ms * 1_000_000)
        config = SimConfig(
            clip_btc=self.clip,
            fees=self.fees,
            fill_rule=self.rule,
            order_latency_ns=latency,
            cancel_latency_ns=latency,
            flatten_slippage_bp=self.slippage_bp,
        )
        return execute(
            market.signals(decisions, hold),
            "signal_entry",
            events=market.day_events(),
            sim_config=config,
        )


@register_execution("event_taker")
class EventTaker(_Execution):
    """The clip walked across the visible book at the decision and back after the hold."""

    fill_rule = (
        "event time: the clip walks the first snapshot after the decision and latency, "
        "and walks back the first snapshot after the hold"
    )

    def __init__(self, *, clip: float = 0.01, latency_ms: float = 10.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.clip = clip
        self.latency_ms = latency_ms

    def execute(self, decisions: np.ndarray, market: MarketData, *, hold: int) -> Any:
        from trading_research.backtest.taker import TakerSettings
        from trading_research.pipeline.execution import execute

        settings = TakerSettings(
            clip=self.clip, fees=self.fees, latency_ns=round(self.latency_ms * 1_000_000)
        )
        return execute(
            market.signals(decisions, hold),
            "event_taker",
            books=market.taker_books(),
            taker_settings=settings,
        )
