"""One signal executed three ways, each through the repository's own executor.

``taker`` and ``passive`` run on the five-second grid through
:func:`trading_research.pipeline.execution.execute`, the market built from the
best bid and ask and the prints as ``reproduce`` builds it
(:func:`~trading_research.data.grid.to_grid`,
:func:`~trading_research.pipeline.execution.aggressive_flow`); ``maker`` quotes
both sides in event time through the same function's ``market_maker`` mode,
which is ``mm-backtest``'s path: a
:class:`~trading_research.pipeline.market_making.MakerPlan` under a
:class:`~trading_research.market_making.simulator.SimConfig` at the stated fee
tier, the days opened through
:func:`~trading_research.pipeline.market_making.access_for` (so a held-out day
of the market-making study is refused here as there), and, given a signal, S1
leaning on it as ``experiments/grand_search.py --execution-axis`` leans it.

Every mode returns :class:`~trading_research.pipeline.execution.ExecutionResult`:
one row per attempt (a miss counts, earning nothing), and for the market maker
one row per day in basis points of its turnover.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from trading_research.backtest.costs import FeeTier, TakerCosts, fee_tier
from trading_research.backtest.execution import ThinningRules
from trading_research.backtest.maker import MakerCosts, PostingRules
from trading_research.pipeline.execution import ExecutionResult, GridMarket, SignalStream, execute
from trading_research.walkthrough.layout import Dataset

#: The three ways a signal meets the market, by the names the walkthrough uses,
#: and the :class:`~trading_research.pipeline.execution.ExecutionMode` of each.
MODES: dict[str, str] = {"taker": "taker", "passive": "passive_entry", "maker": "market_maker"}
GRID_SECONDS = 5
NS = 1_000_000_000


@dataclass(frozen=True)
class Costs:
    """The fee tier and the slippage every mode is charged, stated once."""

    tier: FeeTier
    slippage_bp: float = 0.5

    @classmethod
    def named(cls, tier: str = "base", slippage_bp: float = 0.5) -> Costs:
        """A row of Bybit's linear schedule (``backtest/costs.py``), by name."""
        return cls(fee_tier(tier), slippage_bp)

    @property
    def taker(self) -> TakerCosts:
        return TakerCosts(fee_bp_per_side=self.tier.taker_bp, slippage_bp=self.slippage_bp)

    @property
    def maker(self) -> MakerCosts:
        # MakerCosts refuses a negative fee, so a rebate tier is refused here, not clamped.
        return MakerCosts(
            fee_bp_per_side=self.tier.maker_bp,
            taker_fee_bp_per_side=self.tier.taker_bp,
            slippage_bp=self.slippage_bp,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "tier": self.tier.name,
            "maker_bp": self.tier.maker_bp,
            "taker_bp": self.tier.taker_bp,
            "slippage_bp": self.slippage_bp,
            "source": "backtest/costs.py BYBIT_LINEAR_TIERS (trp fees)",
        }


def _prints(dataset: Dataset, symbol: str, day: date) -> pd.DataFrame | None:
    """One day's prints with their aggressor: the ``trades`` plane, else the raw trades."""
    if dataset.trades is not None and (dataset.trades / symbol / f"{day}.parquet").exists():
        return pd.read_parquet(dataset.trades / symbol / f"{day}.parquet")
    path = dataset.raw / symbol / "aggTrades" / f"{day}.parquet"
    if not path.exists():
        return None
    raw = pd.read_parquet(path)
    return pd.DataFrame(
        {
            "timestamp": raw["timestamp"],
            "price": raw["price"],
            "size": raw["quantity"],
            "aggressor": np.where(raw["is_buyer_maker"].to_numpy(dtype=bool), -1, 1),
        }
    )


def grid_market(dataset: Dataset, symbol: str, days: Sequence[date]) -> GridMarket:
    """``symbol``'s best bid and ask on the five-second grid over ``days``, with the
    volume that would consume a resting bid or ask on each row."""
    from trading_research.data.grid import to_grid
    from trading_research.market_making.events import to_ns
    from trading_research.pipeline.execution import aggressive_flow

    columns = ["timestamp", "bid_price_0", "ask_price_0", "bid_size_0", "ask_size_0"]
    parts = [
        pd.read_parquet(dataset.raw / symbol / "bookTicker" / f"{d}.parquet", columns=columns)
        for d in sorted(days)
    ]
    book = to_grid(pd.concat(parts, ignore_index=True), GRID_SECONDS).reset_index(drop=True)
    stamps = pd.Series(to_ns(pd.DatetimeIndex(book["timestamp"])))
    bid, ask = book["bid_price_0"].to_numpy(), book["ask_price_0"].to_numpy()
    sell, buy = np.zeros(len(book)), np.zeros(len(book))
    for day in sorted(days):
        prints = _prints(dataset, symbol, day)
        if prints is None or prints.empty:
            continue
        prints = prints.assign(timestamp=to_ns(prints["timestamp"]))
        day_sell, day_buy = aggressive_flow(prints, stamps, bid, ask)
        sell += day_sell
        buy += day_buy
    steps = np.diff(np.unique(bid[np.isfinite(bid)]))
    return GridMarket(
        symbol,
        stamps.to_numpy(),
        bid,
        ask,
        book["bid_size_0"].to_numpy(),
        book["ask_size_0"].to_numpy(),
        sell_at_bid=sell,
        buy_at_ask=buy,
        tick=float(steps[steps > 0].min()) if (steps > 0).any() else 0.0,
    )


def stream(
    symbol: str, frame: pd.DataFrame, decision: np.ndarray, hold_rows: int, strength_bp: float
) -> SignalStream:
    """Decisions on the prepared rows as a :class:`SignalStream` (each at its row's stamp)."""
    from trading_research.market_making.events import to_ns

    ts = to_ns(pd.DatetimeIndex(frame["timestamp"]))
    keep = np.asarray(decision) != 0
    return SignalStream(
        symbol,
        ts[keep],
        np.asarray(decision)[keep].astype(np.int8),
        np.full(int(keep.sum()), float(strength_bp)),
        int(hold_rows) * GRID_SECONDS * NS,
    )


def maker_plan(
    dataset: Dataset, symbol: str, days: Sequence[date], costs: Costs, *, lean: bool
) -> Any:
    """S1 as ``mm-backtest`` runs it by hand, its clip and volatility reference set by
    the market-making registration's rules on ``days`` (which must be search days):
    the clip a tenth of the median touch notional, sigma the median one-minute
    volatility of the mid in bp."""
    from trading_research.market_making.simulator import SimConfig
    from trading_research.pipeline import market_making as mm

    touch, vol = [], []
    for day in sorted(days):
        book = pd.read_parquet(
            dataset.raw / symbol / "bookTicker" / f"{day}.parquet",
            columns=["timestamp", "bid_price_0", "ask_price_0", "bid_size_0", "ask_size_0"],
        )
        mid = (book["bid_price_0"] + book["ask_price_0"]) / 2
        touch.append(float(np.median(np.minimum(book["bid_size_0"], book["ask_size_0"]) * mid)))
        minute = mid.groupby(book["timestamp"].dt.floor("1min")).last()
        vol.append(float(np.nanmedian(np.abs(np.diff(np.log(minute.to_numpy()))) * 1e4)))
    config = SimConfig(clip_notional=0.10 * float(np.median(touch)), soft_limit_clips=6.0)
    params = {"skew_bp": 2.0, "k": 0.5, "min_edge_bp": 4.0}  # mm-backtest's defaults for S1
    return mm.MakerPlan(
        "S1",
        config.with_(fees=costs.tier),
        params,
        float(np.median(vol)),
        lean=1.0 if lean else 0.0,
    )


def plan_values(plan: Any) -> dict[str, Any]:
    """What a frozen choice records of a market-maker plan, enough to build it again."""
    return {
        "strategy": plan.strategy,
        "clip_notional": plan.config.clip_notional,
        "soft_limit_clips": plan.config.soft_limit_clips,
        "params": dict(plan.params),
        "sigma_ref": plan.sigma_ref,
        "lean": plan.lean,
    }


def plan_from(values: dict[str, Any], costs: Costs) -> Any:
    """The plan :func:`plan_values` recorded, at ``costs``' tier."""
    from trading_research.market_making.simulator import SimConfig
    from trading_research.pipeline import market_making as mm

    config = SimConfig(
        clip_notional=float(values["clip_notional"]),
        soft_limit_clips=float(values["soft_limit_clips"]),
    )
    return mm.MakerPlan(
        values["strategy"], config.with_(fees=costs.tier), dict(values["params"]),
        float(values["sigma_ref"]), lean=float(values["lean"]),
    )  # fmt: skip


def run_mode(
    mode: str,
    dataset: Dataset,
    symbol: str,
    days: Sequence[date],
    costs: Costs,
    *,
    signal: SignalStream | None = None,
    hold_rows: int = 12,
    market: GridMarket | None = None,
    plan: Any = None,
) -> ExecutionResult:
    """``signal`` executed in ``mode`` (taker, passive or maker) over ``days``."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {tuple(MODES)}, got {mode!r}")
    if mode == "maker":
        from trading_research.pipeline import market_making as mm

        if not dataset.has_event_time(symbol, days):
            raise ValueError(
                f"the market maker needs book, prints and funding for {symbol} on every day"
            )
        plan = plan or maker_plan(dataset, symbol, days, costs, lean=signal is not None)
        sources = mm.MakerSources((dataset.book,), dataset.trades, dataset.funding)  # type: ignore[arg-type]
        return execute(
            signal if (signal is not None and plan.lean) else None,
            "market_maker",
            symbol=symbol,
            days=list(days),
            plan=plan,
            sources=sources,
            access=mm.access_for(list(days)),
        )
    if signal is None:
        raise ValueError(f"{mode} executes a signal: give one")
    market = market or grid_market(dataset, symbol, days)
    if mode == "taker":
        return execute(
            signal, "taker", market=market, taker=costs.taker,
            taker_rules=ThinningRules(hold_periods=hold_rows),
        )  # fmt: skip
    return execute(
        signal, "passive_entry", market=market, maker=costs.maker,
        posting=PostingRules(timeout_rows=hold_rows, hold_rows=hold_rows),
    )  # fmt: skip


def at_fee(result: ExecutionResult, mode: str, base: Costs, other: FeeTier) -> float:
    """The net per attempt of the same attempts at another tier, decisions unchanged.

    A taker's round trip moves by twice the difference of the taker fees; a
    passive fill pays a maker fee in and, when it crossed out, a taker fee out
    (``backtest/maker.MakerCosts``); the market maker's day moves by the maker
    difference on its maker turnover and the taker difference on its flattens.
    Exact for decisions that do not read the fee, which the quoter does: for
    the market maker it is the same quotes re-costed.
    """
    attempts = result.attempts
    if attempts.empty:
        return float("nan")
    d_taker = other.taker_bp - base.tier.taker_bp
    d_maker = other.maker_bp - base.tier.maker_bp
    if mode == "taker":
        return float((attempts["net_bp"] - 2 * d_taker).mean())
    if mode == "passive":
        filled = attempts["filled"].astype(bool).to_numpy()
        crossed = filled & (attempts["path"] != "posted_out").to_numpy()
        exit_shift = np.where(crossed, d_taker, np.where(filled, d_maker, 0.0))
        shift = np.where(filled, d_maker, 0.0) + exit_shift
        return float((attempts["net_bp"] - shift).mean())
    days = result.days
    if days is None or days.empty:
        return float("nan")
    from trading_research.pipeline import market_making as mm

    used = mm.usable(days)
    turnover = (used["maker_turnover"] + used["taker_turnover"]).to_numpy(dtype=float)
    extra = d_maker * used["maker_turnover"].to_numpy(dtype=float) + d_taker * used[
        "taker_turnover"
    ].to_numpy(dtype=float)
    net = used["net"].to_numpy(dtype=float) - extra * 1e-4
    with np.errstate(divide="ignore", invalid="ignore"):
        per = np.where(turnover > 0, net / turnover * 1e4, 0.0)
    return float(np.mean(per)) if len(per) else float("nan")


def with_tier(costs: Costs, tier: FeeTier) -> Costs:
    return replace(costs, tier=tier)
