"""Own size and inventory: the clip, the soft and hard limits, and flat days."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time

import numpy as np
import pytest

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
                # A flatten arrives after the order latency and always moves the
                # position toward zero.
                before = fills["position_after"].shift(1, fill_value=0.0)
                limit = fills["path"] == "flatten"
                moved = fills.loc[limit, "position_after"].abs() < before[limit].abs()
                assert moved.all()
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
    sweep at 120 ms fills both, the position is two clips against a soft limit
    of one, and a taker order takes it back to one. The taker pays the order
    latency and walks the first snapshot at or after its arrival (200 ms, best
    bid 0.97), not the book the sweep had just consumed."""
    before, after = _book(100, 102), _book(97, 102)
    market = hand_built_market(
        [(0.0, *before), (100.0, *before), (200.0, *after), (300.0, *after)],
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
    flatten = fills.iloc[2]
    assert flatten["ts"] - market.day_start_ns == 200_000_000
    assert flatten["queue_wait_ns"] == 80_000_000  # sent at 120 ms
    assert flatten["price"] == pytest.approx(0.97 * (1 - 0.5e-4))
    assert flatten["mid_ref"] == pytest.approx((0.97 + 1.02) / 2)


def test_a_flatten_never_reverses_the_position() -> None:
    """A flatten is reduce-only: if passive fills shrank the position while it
    travelled, it sells only what is left."""
    for seed in range(40):
        market = random_market(seed, n_snapshots=300, prints_per_snapshot=1.5)
        config = FUZZ.with_(
            soft_limit_clips=1.0, order_latency_ns=250_000_000, cancel_latency_ns=250_000_000
        )
        fills = simulate_day(market, TouchQuoter(), config).fills
        before = fills["position_after"].shift(1, fill_value=0.0)
        flatten = fills["path"] == "flatten"
        assert (fills.loc[flatten, "position_after"] * before[flatten] >= 0).all()


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


def test_the_hard_limit_holds_when_the_clip_shrinks() -> None:
    """The touch is 100 for the first second (clip 10, hard limit 20) and 20
    after it (clip 2, hard limit 4). The bid placed at a clip of 10 keeps its
    price, but not its size: it is cancelled and re-placed at 2, so the sale at
    3.5 s cannot take the position past the hard limit."""
    big, small = _book(100, 102, size=100.0), _book(100, 102, size=20.0)
    snapshots = [(0.0, *big)] + [(1000.0 + 100.0 * i, *small) for i in range(30)]
    market = hand_built_market(snapshots, [(3500.0, 99, 50.0, -1)])
    config = SimConfig(
        clip_notional=1e12,
        clip_touch_share=0.1,
        warmup_s=0.0,
        soft_limit_clips=1.0,
        touch_window_s=1,
    )
    result = simulate_day(market, Fixed(bid=100), config)
    passive = result.fills[result.fills["maker"]]
    assert (passive["position_after"].abs() <= passive["hard_limit"] + 1e-9).all()
    assert passive["size"].tolist() == [2.0]
    assert result.counters["cancelled_for_limits"] >= 1


def test_own_quotes_never_lock_or_cross() -> None:
    """A quoter asking for a bid above its ask is pulled apart around the
    middle, one tick each side; both quotes then rest and fill without ever
    meeting."""
    book = _book(100, 103)
    market = hand_built_market(
        [(0.0, *book), (100.0, *book), (200.0, *book)],
        [(150.0, 100, 1.0, -1), (160.0, 103, 1.0, 1)],
    )
    config = SimConfig(clip_notional=1e12, clip_touch_share=1e9, warmup_s=0.0)
    result = simulate_day(market, Fixed(bid=102, ask=101), config)
    orders = result.orders
    assert set(orders.loc[orders["side"] == 1, "price"]) == {101}
    assert set(orders.loc[orders["side"] == -1, "price"]) == {102}
    assert result.counters["self_cross_clamped"] >= 1
    maker = result.fills[result.fills["maker"]]
    assert sorted(maker["price"].round(2).tolist()) == [1.01, 1.02]


def test_an_order_arriving_at_our_own_opposite_order_is_rejected() -> None:
    """The ask at 101 is withdrawn at 100 ms, but its cancel takes 50 ms; the
    bid at 101 sent at the same moment arrives after 10 ms to find our own ask
    still there, and is rejected rather than trading with it."""
    book = _book(100, 103)
    market = hand_built_market(
        [(0.0, *book), (100.0, *book), (200.0, *book)],
        [(250.0, 103, 1.0, 1)],
    )

    @dataclass(frozen=True)
    class Flip:
        name: str = "flip"
        uses_future: bool = False

        def quotes(self, view: MarketView, position: float) -> Quotes:
            if view.ts == market.book_ts[0]:
                return Quotes(Quote(None, 0.0), Quote(101, view.clip))
            return Quotes(Quote(101, view.clip), Quote(None, 0.0))

    config = SimConfig(
        clip_notional=1e12, clip_touch_share=1e9, warmup_s=0.0, cancel_latency_ns=50_000_000
    )
    result = simulate_day(market, Flip(), config)
    assert result.counters["rejected_self_cross"] == 1
    assert result.fills[result.fills["maker"]].empty


def test_the_day_end_flatten_crosses_the_spread_and_walks_the_depth() -> None:
    book = _book(100, 102, size=1.0)
    market = hand_built_market(
        [(0.0, *book), (100.0, *book)] + [(1_000.0 * i, *book) for i in range(2, 70)],
        [(150.0, 99, 50.0, -1)],
    )
    config = SimConfig(
        clip_notional=1e12,
        clip_touch_share=1e9,
        warmup_s=0.0,
        soft_limit_clips=1e6,
        stop_quoting_at=time(0, 0, 30),
        flatten_at=time(0, 0, 40),
        suspend_after_pause_s=1e6,
    )

    @dataclass(frozen=True)
    class BidOnce:
        name: str = "bid-once"
        uses_future: bool = False

        def quotes(self, view: MarketView, position: float) -> Quotes:
            return Quotes(Quote(100 if position == 0 else None, 3.0), Quote(None, 0.0))

    result = simulate_day(market, BidOnce(), config)
    flatten = result.fills[result.fills["path"] == "flatten"]
    assert len(flatten) == 1
    row = flatten.iloc[0]
    assert row["ts"] - market.day_start_ns == 41 * 10**9  # first book after 40 s + 10 ms
    walked = (1.00 + 0.99 + 0.98) / 3 * (1 - 0.5e-4)  # 3 units over 3 levels of 1
    assert row["price"] == pytest.approx(walked)
    assert row["fee"] == pytest.approx(row["price"] * 3 * 5.5e-4)
    assert not result.excluded or result.flags == ("no_funding",)


def test_a_day_end_flatten_with_no_fresh_book_is_flagged() -> None:
    """The book ends at 100 ms; the flatten at two minutes has nothing newer to
    walk than a two-minute-old snapshot, so the day is flagged and excluded."""
    book = _book(100, 102)
    late = [(30_000.0 + 1000.0 * i, 80, 1.0, 1) for i in range(5)]
    market = hand_built_market(
        [(0.0, *book), (100.0, *book)],
        [(150.0, 99, 1.0, -1), *late],
    )
    config = SimConfig(
        clip_notional=1e12,
        clip_touch_share=1e9,
        warmup_s=0.0,
        stop_quoting_at=time(0, 1),
        flatten_at=time(0, 2),
    )
    result = simulate_day(market, Fixed(bid=100), config)
    assert "flatten_stale" in result.flags
    assert result.excluded
    assert result.equity["position"].iloc[-1] == 0.0


def test_a_day_end_flatten_after_a_pause_walks_the_first_book_after_it() -> None:
    """No snapshot from 100 ms to 150 s: the flatten due at 120 s waits for the
    150 s snapshot and walks it, and the day is not flagged."""
    book, later = _book(100, 102), _book(95, 97)
    market = hand_built_market(
        [(0.0, *book), (100.0, *book), (150_000.0, *later), (151_000.0, *later)],
        [(50.0, 99, 1.0, -1)],
    )
    config = SimConfig(
        clip_notional=1e12,
        clip_touch_share=1e9,
        warmup_s=0.0,
        stop_quoting_at=time(0, 1),
        flatten_at=time(0, 2),
    )
    result = simulate_day(market, Fixed(bid=100), config)
    flatten = result.fills[result.fills["path"] == "flatten"].iloc[0]
    assert flatten["ts"] - market.day_start_ns == 150 * 10**9
    assert flatten["price"] == pytest.approx(0.95 * (1 - 0.5e-4))
    assert "flatten_stale" not in result.flags
