"""Choosing which strategy to run tomorrow, and whether to run one at all.

Everything else in this project picks one configuration and keeps it for the
whole test span. That is not what a desk does. A desk has several strategies,
watches how each is doing, and moves between them — and, on a bad enough
stretch, stands down.

This is that decision, made once per day and made only from days already
finished. On each day it looks back over a trailing window of realised results,
picks the strategy that did best, and runs it tomorrow. If nothing in the window
cleared ``minimum_edge_bp``, it trades nothing.

Standing aside is the important part
------------------------------------
Under a cost floor, not trading has a real expected value — zero, which beats
most of what is on offer here. A selector that must always pick something is
forced to run its least bad option through periods when none of them works, and
that is where a strategy with a small edge does most of its losing.

The lookahead this could have had, and does not
-----------------------------------------------
The rule is only worth anything if the choice for day *d* uses days strictly
before *d*. Picking the best strategy over a window that includes the day being
traded is a way of reading the answer, and it produces a beautiful equity curve
on any data at all. The window here ends the day before, and the tests check it
directly rather than trusting the loop.

What it cannot do
-----------------
Switching between strategies cannot manufacture edge that none of them has.
If every strategy is losing, the selector picks whichever lost least recently,
which is a different thing from picking one that will win. It helps when
strategies take turns working, and turns into an expensive way to chase noise
when they do not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd


class AllocationError(ValueError):
    """The daily results cannot support a selection."""


#: Name used when the selector declines to trade.
STAND_ASIDE = "stand_aside"


@dataclass
class AllocatorConfig:
    """How the daily choice is made.

    ``lookback_days``
        How much recent history the choice reads. Short reacts quickly and
        chases noise; long is stable and slow to notice a strategy has stopped
        working.
    ``minimum_edge_bp``
        Trailing net per trade a strategy must show before it is run at all.
        Zero means "do not run a strategy that has been losing" — the natural
        floor, and the one that makes standing aside possible. ``None`` removes
        the floor entirely and forces a choice every day, which is the control
        this is compared against rather than a setting anyone would want.
    ``minimum_trades``
        Trades a strategy must have taken in the window for its figure to
        count. Without it the selector chases whoever took two lucky trades.
    ``objective``
        Column ranked. Per-trade by default, since strategies here trade at
        very different rates.
    """

    lookback_days: int = 5
    minimum_edge_bp: float | None = 0.0
    minimum_trades: int = 20
    objective: str = "net_per_trade_bp"

    def __post_init__(self) -> None:
        if self.lookback_days < 1:
            raise ValueError(f"lookback_days must be at least 1, got {self.lookback_days}")
        if self.minimum_trades < 1:
            raise ValueError(f"minimum_trades must be at least 1, got {self.minimum_trades}")

    @property
    def label(self) -> str:
        floor = "none" if self.minimum_edge_bp is None else f"{self.minimum_edge_bp:g}bp"
        return f"look{self.lookback_days}d/min{floor}"


@dataclass
class Allocation:
    """What was chosen for one day, and on what evidence."""

    day: date
    chosen: str
    trailing_edge_bp: float
    trailing_trades: float
    considered: int
    reason: str = ""


@dataclass
class AllocationResult:
    """The whole sequence of daily choices, and what they produced."""

    allocations: list[Allocation]
    daily: pd.DataFrame
    config: AllocatorConfig
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def switches(self) -> int:
        """How often the selector changed its mind."""
        names = [a.chosen for a in self.allocations]
        return sum(1 for a, b in pairwise(names) if a != b)

    @property
    def days_aside(self) -> int:
        return sum(1 for a in self.allocations if a.chosen == STAND_ASIDE)

    def summary(self) -> str:
        return (
            f"{len(self.allocations)} days, {self.switches} switches, "
            f"{self.days_aside} stood aside, "
            f"net {self.metrics.get('net_bp', float('nan')):.0f} bp "
            f"over {self.metrics.get('trades', 0):.0f} trades"
        )


def allocate(
    daily: pd.DataFrame,
    config: AllocatorConfig | None = None,
) -> AllocationResult:
    """Walk the days, choosing a strategy for each from the ones before it.

    ``daily`` carries one row per strategy per day, with at least ``day``,
    ``strategy``, ``trades``, ``net_bp`` and the objective column. It is the
    realised result of running every strategy every day — which a backtest has
    and a live system would not, but the *choice* only ever reads the past, so
    the sequence of choices is one a live system could have made.
    """
    config = config or AllocatorConfig()
    required = {"day", "strategy", "trades", "net_bp", config.objective}
    missing = required - set(daily.columns)
    if missing:
        raise AllocationError(f"daily results are missing {sorted(missing)}")

    frame = daily.copy()
    frame["day"] = pd.to_datetime(frame["day"]).dt.date
    days = sorted(frame["day"].unique())
    if len(days) <= config.lookback_days:
        raise AllocationError(
            f"{len(days)} days cannot support a {config.lookback_days}-day lookback"
        )

    allocations: list[Allocation] = []
    realised: list[dict[str, Any]] = []

    for day in days[config.lookback_days :]:
        window_days = [d for d in days if d < day][-config.lookback_days :]
        window = frame[frame["day"].isin(window_days)]

        # Pooled over the window, not averaged per day: a strategy that traded
        # once on four days and forty times on the fifth is one strategy, and
        # averaging its daily figures would weight the quiet days equally.
        pooled = window.groupby("strategy").agg(trades=("trades", "sum"), net_bp=("net_bp", "sum"))
        pooled = pooled[pooled["trades"] >= config.minimum_trades]
        pooled["edge"] = pooled["net_bp"] / pooled["trades"]

        if pooled.empty:
            allocations.append(
                Allocation(day, STAND_ASIDE, float("nan"), 0.0, 0, "nothing traded enough")
            )
            continue

        best = str(pooled["edge"].idxmax())
        edge = float(pooled["edge"].loc[best])
        traded = float(pooled["trades"].loc[best])
        if config.minimum_edge_bp is not None and edge < config.minimum_edge_bp:
            allocations.append(
                Allocation(
                    day,
                    STAND_ASIDE,
                    edge,
                    traded,
                    len(pooled),
                    f"best trailing edge {edge:.2f} below {config.minimum_edge_bp:g}",
                )
            )
            continue

        allocations.append(Allocation(day, best, edge, traded, len(pooled)))
        today = frame[(frame["day"] == day) & (frame["strategy"] == best)]
        if not today.empty:
            realised.append(
                {
                    "day": day,
                    "strategy": str(best),
                    "trades": float(today["trades"].iloc[0]),
                    "net_bp": float(today["net_bp"].iloc[0]),
                }
            )

    result = pd.DataFrame(realised)
    metrics: dict[str, float] = {"days": float(len(allocations))}
    if not result.empty:
        from trading_research.backtest.metrics import summarise

        # Drawdown on the daily sequence rather than trade by trade: the
        # selector acts daily, so the day is the unit its risk is felt in.
        daily_net = result["net_bp"].to_numpy()
        metrics.update(summarise(daily_net, days=len(allocations)))
        metrics["trades"] = float(result["trades"].sum())
        metrics["trades_per_day"] = float(result["trades"].sum() / len(allocations))
        metrics["net_per_trade_bp"] = (
            float(result["net_bp"].sum() / result["trades"].sum())
            if result["trades"].sum()
            else float("nan")
        )
        metrics["days_traded"] = float(len(result))

    return AllocationResult(allocations=allocations, daily=result, config=config, metrics=metrics)


def default_configs() -> tuple[AllocatorConfig, ...]:
    """Settings worth comparing: how much history, and how picky about it.

    ``None`` is included as the control — a selector forced to trade every day
    — because the value of standing aside is only visible against something
    that cannot.
    """
    return tuple(
        AllocatorConfig(lookback_days=look, minimum_edge_bp=floor)
        for look in (3, 5, 10)
        for floor in (None, 0.0, 1.0)
    )


def per_day_results(
    trades: pd.DataFrame,
    strategy: str,
    *,
    day_column: str = "day",
) -> pd.DataFrame:
    """Collapse a strategy's trades into one row per day, for the allocator."""
    if trades.empty:
        return pd.DataFrame(
            columns=[day_column, "strategy", "trades", "net_bp", "net_per_trade_bp"]
        )
    grouped = trades.groupby(day_column).agg(trades=("net_bp", "size"), net_bp=("net_bp", "sum"))
    grouped["strategy"] = strategy
    grouped["net_per_trade_bp"] = grouped["net_bp"] / grouped["trades"].replace(0, np.nan)
    return grouped.reset_index()
