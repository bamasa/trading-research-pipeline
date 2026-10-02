"""Own size and inventory: the clip, the soft and hard limits, and flat days."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time

import numpy as np

from trading_research.market_making.quoters import MarketView, Quote, Quotes, TouchQuoter
from trading_research.market_making.simulator import SimConfig, simulate_day
from trading_research.market_making.synthetic import hand_built_market, random_market

#: A clip of exactly two on the random markets: the notional cap binds at a mid
#: near 100 and the touch cap never does.
FUZZ = SimConfig(clip_notional=250.0, clip_touch_share=1e9, warmup_s=0.0)


@dataclass(frozen=True)
class Fixed:
    bid: int | None
    ask: int | None = None
    name: str = "fixed"
    uses_future: bool = False

    def quotes(self, view: MarketView, position: float) -> Quotes:
        return Quotes(Quote(self.bid, view.clip), Quote(self.ask, view.clip))


def _book(best_bid: int, best_ask: int, size: float = 10.0):
    return (
        [(best_bid - i, size) for i in range(3)],
        [(best_ask + i, size) for i in range(3)],
    )


def test_inventory_never_exceeds_the_hard_limit() -> None:
    """Random streams, short and long latencies, tight and loose limits."""
    flattened = 0
    for seed in range(40):
        market = random_market(seed, n_snapshots=300, prints_per_snapshot=1.5)
        for soft in (1.0, 2.0, 4.0):
            for latency in (10_000_000, 250_000_000):
                config = FUZZ.with_(
                    soft_limit_clips=soft, order_latency_ns=latency, cancel_latency_ns=latency
                )
                result = simulate_day(market, TouchQuoter(), config)
                fills = result.fills
                assert (fills["hard_limit"] == 2.0 * (soft + 1)).all()
                passive = fills[fills["maker"]]
                assert (passive["position_after"].abs() <= passive["hard_limit"] + 1e-9).all()
                limit = fills[fills["path"] == "flatten"]
                assert (limit["position_after"].abs() <= limit["soft_limit"] + 1e-9).all()
                assert result.counters["max_abs_position"] <= 2.0 * (soft + 1) + 1e-9
                flattened += int(result.counters.get("flattens_soft_limit", 0))
    assert flattened > 0  # the flatten back to the soft limit was exercised


def test_the_growing_side_is_not_quoted_beyond_the_soft_limit() -> None:
    for seed in range(40):
        market = random_market(seed, n_snapshots=300, prints_per_snapshot=1.5)
        result = simulate_day(market, TouchQuoter(), FUZZ.with_(soft_limit_clips=2.0))
        orders = result.orders
        grown = orders["side"] * orders["position_at_decision"] + orders["size"]
        assert (grown <= 4.0 + 1e-9).all()
        assert (orders["size"] == 2.0).all()


def test_a_sweep_past_the_soft_limit_is_flattened_back_to_it() -> None:
    """A replace leaves the old bid in flight while the new one arrives; one
    sweep fills both, the position is two clips against a soft limit of one,
    and a taker order takes it back to one."""
    market = hand_built_market(
        [(0.0, *_book(100, 102)), (100.0, *_book(100, 102))],
        [(120.0, 98, 5.0, -1)],
    )

    @dataclass(frozen=True)
    class MovesDown:
        name: str = "moves-down"
        uses_future: bool = False

        def quotes(self, view: MarketView, position: float) -> Quotes:
            bid = 100 if view.ts == market.book_ts[0] else 99
            return Quotes(Quote(bid, view.clip), Quote(None, 0.0))

    config = SimConfig(
        clip_notional=1e12, warmup_s=0.0, soft_limit_clips=1.0, cancel_latency_ns=50_000_000
    )
    result = simulate_day(market, MovesDown(), config)
    fills = result.fills
    assert fills["path"].tolist()[:3] == ["through", "through", "flatten"]
    assert fills["position_after"].tolist()[:3] == [1.0, 2.0, 1.0]
    assert result.counters["flattens_soft_limit"] == 1


def test_days_start_and_end_flat() -> None:
    """Quoting stops and the position is flattened at the stated times, and
    nothing is left open or held at the end."""
    for seed in range(10):
        market = random_market(seed, n_snapshots=600, prints_per_snapshot=1.5)
        config = FUZZ.with_(
            soft_limit_clips=3.0, stop_quoting_at=time(0, 0, 40), flatten_at=time(0, 0, 50)
        )
        result = simulate_day(market, TouchQuoter(), config)
        start = market.day_start_ns
        fills = result.fills
        assert (fills.loc[fills["ts"] >= start + 41 * 10**9, "maker"] == False).all()  # noqa: E712
        assert result.orders["decided_ns"].max() < start + 40 * 10**9
        assert (result.orders["outcome"] != "open").all()
        equity = result.equity
        assert equity["position"].iloc[-1] == 0.0
        assert (equity.loc[equity["ts"] >= start + 50 * 10**9, "position"] == 0.0).all()
        assert result.counters.get("flattens_day_end", 0) <= 1


def test_quotes_beyond_the_visible_book_are_not_placed() -> None:
    market = hand_built_market([(0.0, *_book(100, 102))], [(50.0, 102, 1.0, 1)])
    result = simulate_day(market, Fixed(bid=90), SimConfig(clip_notional=1e12, warmup_s=0.0))
    assert result.orders.empty
    assert result.counters["not_placed_beyond_book"] >= 1


def test_a_clip_below_one_lot_is_not_quoted() -> None:
    market = hand_built_market([(0.0, *_book(100, 102))], [(50.0, 102, 1.0, 1)])
    result = simulate_day(market, Fixed(bid=100), SimConfig(clip_notional=0.5, warmup_s=0.0))
    assert result.orders.empty
    assert result.counters["not_quoted_below_lot"] >= 1


def test_the_clip_adapts_to_the_trailing_touch() -> None:
    """A tenth of the median touch, rounded down to the lot."""
    small = hand_built_market([(0.0, *_book(100, 102, size=37.0))], [(50.0, 102, 1.0, 1)])
    result = simulate_day(small, Fixed(bid=100), SimConfig(clip_notional=1e12, warmup_s=0.0))
    assert result.orders["size"].tolist() == [3.0]
    capped = simulate_day(small, Fixed(bid=100), SimConfig(clip_notional=2.0, warmup_s=0.0))
    assert capped.orders["size"].tolist() == [1.0]  # 2 USDT at a price of 1.0
    assert np.isclose(result.counters["clip_over_touch"], 3.0 / 37.0)
