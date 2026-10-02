"""The queue: an own order fills only from prints, and only once the size ahead is gone.

Unit tests drive :mod:`trading_research.market_making.queue` directly with
hand-chosen numbers; the fuzz tests drive the whole simulator over random
streams and check what must hold whatever the stream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from trading_research.market_making.orders import Order
from trading_research.market_making.queue import (
    QUEUE,
    THROUGH,
    ArrivalGrowth,
    CancelAttribution,
    CrossedPolicy,
    QueuePriority,
    on_arrival,
    on_print,
    on_snapshot,
)
from trading_research.market_making.quoters import MarketView, Quote, Quotes, TouchQuoter
from trading_research.market_making.simulator import SimConfig, simulate_day
from trading_research.market_making.synthetic import hand_built_market, random_market

CONFIG = SimConfig(clip_notional=1e12, warmup_s=0.0)
#: A clip of two on the random markets (mid near 100), whatever their touch.
FUZZ = SimConfig(clip_notional=250.0, clip_touch_share=1e9, warmup_s=0.0)


def _order(side: int = 1, price: int = 100, size: float = 2.0, live_ns: int = 25) -> Order:
    return Order(
        oid=1,
        side=side,
        price=price,
        size=size,
        remaining=size,
        decided_ns=0,
        live_ns=live_ns,
    )


def _arrived(level: float, *, side: int = 1, size: float = 2.0) -> Order:
    order = _order(side=side, size=size)
    on_arrival(order, level, inside_spread=False, priority=QueuePriority.TAIL)
    order.arrival_snapshot_ns = 0
    return order


@dataclass(frozen=True)
class Fixed:
    """Quotes fixed prices with the simulator's clip."""

    bid: int | None
    ask: int | None = None
    name: str = "fixed"
    uses_future: bool = False

    def quotes(self, view: MarketView, position: float) -> Quotes:
        return Quotes(Quote(self.bid, view.clip), Quote(self.ask, view.clip))


def _book(best_bid: int, best_ask: int, size: float = 10.0, depth: int = 3):
    return (
        [(best_bid - i, size) for i in range(depth)],
        [(best_ask + i, size) for i in range(depth)],
    )


# ---------------------------------------------------------------------------
# Arrival
# ---------------------------------------------------------------------------


def test_joining_a_visible_level_starts_at_its_tail() -> None:
    assert _arrived(7.0).queue_ahead == 7.0


def test_inside_the_spread_starts_at_the_front() -> None:
    order = _order()
    on_arrival(order, 0.0, inside_spread=True, priority=QueuePriority.TAIL)
    assert order.queue_ahead == 0.0
    filled, path = on_print(order, 100, 1.0, -1)
    assert (filled, path) == (1.0, QUEUE)


def test_the_front_priority_rung_ignores_the_visible_queue() -> None:
    order = _order()
    on_arrival(order, 9.0, inside_spread=False, priority=QueuePriority.FRONT)
    assert order.queue_ahead == 0.0
    on_snapshot(
        order,
        20.0,
        rule=CancelAttribution.PROPORTIONAL,
        growth=ArrivalGrowth.ALL_AHEAD,
        snapshot_ns=100,
    )
    assert order.queue_ahead == 0.0


def test_an_order_beyond_the_visible_book_waits_for_the_level_to_appear() -> None:
    order = _order()
    on_arrival(order, None, inside_spread=False, priority=QueuePriority.TAIL)
    assert on_print(order, 100, 50.0, -1) == (0.0, "")
    on_snapshot(
        order,
        6.0,
        rule=CancelAttribution.OPTIMISTIC,
        growth=ArrivalGrowth.NONE,
        snapshot_ns=100,
    )
    assert order.queue_ahead == 6.0  # the tail of what is visible now


# ---------------------------------------------------------------------------
# Prints
# ---------------------------------------------------------------------------


def test_prints_on_the_wrong_side_do_not_advance_the_queue() -> None:
    """A buyer lifting the offer at our bid's price does not touch the bid queue."""
    order = _arrived(5.0)
    assert on_print(order, 100, 3.0, +1) == (0.0, "")
    assert order.queue_ahead == 5.0


def test_prints_at_other_prices_do_not_advance_the_queue() -> None:
    """A sale above our bid does not reach it."""
    order = _arrived(5.0)
    assert on_print(order, 101, 30.0, -1) == (0.0, "")
    assert order.queue_ahead == 5.0
    ask = _arrived(5.0, side=-1)
    assert on_print(ask, 99, 30.0, +1) == (0.0, "")
    assert ask.queue_ahead == 5.0


def test_partial_fills_sum_to_the_print_beyond_the_queue() -> None:
    order = _arrived(3.0, size=5.0)
    fills = [on_print(order, 100, size, -1)[0] for size in (2.0, 2.0, 1.5, 10.0)]
    assert fills == [0.0, 1.0, 1.5, 2.5]
    assert sum(fills) == 5.0
    assert order.remaining == 0.0


def test_a_print_through_the_price_fills_at_the_limit_and_clears_the_queue() -> None:
    order = _arrived(40.0, size=3.0)
    filled, path = on_print(order, 98, 2.0, -1)
    assert (filled, path) == (2.0, THROUGH)
    assert order.queue_ahead == 0.0
    ask = _arrived(40.0, side=-1, size=3.0)
    assert on_print(ask, 103, 7.0, +1) == (3.0, THROUGH)


def test_a_print_through_fills_at_the_order_price_in_the_simulator() -> None:
    bids, asks = _book(100, 102)
    market = hand_built_market(
        [(0.0, bids, asks), (100.0, bids, asks)],
        [(150.0, 99, 0.4, -1)],
    )
    result = simulate_day(market, Fixed(bid=100), CONFIG)
    fill = result.fills.iloc[0]
    assert fill["path"] == THROUGH
    assert fill["price"] == pytest.approx(100 * 0.01)
    assert fill["size"] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------


def _snapshot(order: Order, level: float, rule: CancelAttribution, growth: ArrivalGrowth) -> None:
    on_snapshot(order, level, rule=rule, growth=growth, snapshot_ns=100)


def test_orders_joining_after_us_do_not_move_us_back() -> None:
    order = _arrived(5.0)
    _snapshot(order, 5.0, CancelAttribution.PROPORTIONAL, ArrivalGrowth.PRO_RATA_TIME)
    for level in (12.0, 40.0, 90.0):
        on_snapshot(
            order,
            level,
            rule=CancelAttribution.PESSIMISTIC,
            growth=ArrivalGrowth.ALL_AHEAD,
            snapshot_ns=200,
        )
        assert order.queue_ahead == 5.0


def test_growth_in_the_arrival_interval_is_split_by_time() -> None:
    """Live 25 ms into a 100 ms interval: a quarter of the growth joined first."""
    results = {}
    for growth in ArrivalGrowth:
        order = _arrived(10.0)  # snapshot at 0, live at 25 ns of a 100 ns interval
        _snapshot(order, 18.0, CancelAttribution.PROPORTIONAL, growth)
        results[growth] = order.queue_ahead
    assert results[ArrivalGrowth.NONE] == 10.0
    assert results[ArrivalGrowth.PRO_RATA_TIME] == pytest.approx(12.0)
    assert results[ArrivalGrowth.ALL_AHEAD] == 18.0


def test_cancellation_attribution_brackets_on_one_snapshot() -> None:
    """Level 10 -> 4 with 5 ahead: 6 cancelled."""
    results = {}
    for rule in CancelAttribution:
        order = _arrived(10.0)
        order.queue_ahead = 5.0
        order.in_arrival_interval = False
        _snapshot(order, 4.0, rule, ArrivalGrowth.NONE)
        results[rule] = order.queue_ahead
    assert results[CancelAttribution.PESSIMISTIC] == 4.0  # clamped to the level
    assert results[CancelAttribution.PROPORTIONAL] == pytest.approx(2.0)
    assert results[CancelAttribution.OPTIMISTIC] == 0.0


def test_prints_at_the_price_are_not_counted_again_as_cancellations() -> None:
    order = _arrived(10.0)
    order.in_arrival_interval = False
    on_print(order, 100, 4.0, -1)
    assert order.queue_ahead == 6.0
    _snapshot(order, 6.0, CancelAttribution.OPTIMISTIC, ArrivalGrowth.NONE)
    assert order.queue_ahead == 6.0


def test_an_empty_level_leaves_nothing_ahead_under_every_rule() -> None:
    for rule in CancelAttribution:
        order = _arrived(10.0)
        order.in_arrival_interval = False
        _snapshot(order, 0.0, rule, ArrivalGrowth.NONE)
        assert order.queue_ahead == 0.0


def _random_walk(seed: int, rule: CancelAttribution, growth: ArrivalGrowth) -> tuple[float, bool]:
    """Drive one order through a random sequence; return its fills and whether
    every invariant held."""
    rng = np.random.default_rng(seed)
    level = float(rng.integers(0, 20))
    order = _order(size=float(rng.integers(1, 6)))
    on_arrival(order, level, inside_spread=False, priority=QueuePriority.TAIL)
    order.arrival_snapshot_ns = 0
    filled = 0.0
    ok = True
    for step in range(60):
        before = order.queue_ahead
        if rng.random() < 0.6:
            price = 100 + int(rng.integers(-1, 2))
            got, _ = on_print(order, price, float(rng.integers(1, 8)), int(rng.choice([-1, 1])))
            filled += got
            ok &= order.queue_ahead >= 0.0 and order.queue_ahead <= before
        else:
            level = float(max(0, level + rng.integers(-8, 9)))
            first = order.in_arrival_interval
            on_snapshot(order, level, rule=rule, growth=growth, snapshot_ns=100 * (step + 1))
            ok &= 0.0 <= order.queue_ahead <= level
            ok &= first or order.queue_ahead <= before
    return filled, ok


@pytest.mark.parametrize("growth", list(ArrivalGrowth))
@pytest.mark.parametrize("rule", list(CancelAttribution))
def test_queue_ahead_is_never_negative_and_never_exceeds_the_level(
    rule: CancelAttribution, growth: ArrivalGrowth
) -> None:
    """Never negative; at most the visible level after every snapshot; and it
    only ever falls, except at the first snapshot after arrival."""
    for seed in range(200):
        assert _random_walk(seed, rule, growth)[1], seed


def test_cancellation_attribution_orders_the_fills_of_one_order() -> None:
    """For one order on one stream, pessimistic <= proportional <= optimistic."""
    for seed in range(200):
        fills = {
            rule: _random_walk(seed, rule, ArrivalGrowth.PRO_RATA_TIME)[0]
            for rule in CancelAttribution
        }
        assert fills[CancelAttribution.PESSIMISTIC] <= fills[CancelAttribution.PROPORTIONAL]
        assert fills[CancelAttribution.PROPORTIONAL] <= fills[CancelAttribution.OPTIMISTIC]


def _maker_volume(market, quoter, config: SimConfig) -> dict[CancelAttribution, float]:
    return {
        rule: float(
            simulate_day(market, quoter, config.with_(cancel_attribution=rule))
            .fills.query("maker")["size"]
            .sum()
        )
        for rule in CancelAttribution
    }


def test_cancellation_attribution_brackets() -> None:
    """Through the whole simulator: one order a side, resting all day at fixed
    prices and too large to complete, fills no more under the pessimistic rule
    than under the proportional one, and no more under that than under the
    optimistic one, on every stream.

    For a quoter that re-quotes after each fill the ordering is not a theorem
    per stream — a fill changes what is quoted next, and the paths diverge — so
    for the touch quoter it is checked in total over the streams.
    """
    held = FUZZ.with_(clip_notional=1e12, soft_limit_clips=1e6)
    strict = 0
    for seed in range(40):
        # A touch that stays put, with prints only at it or short of it, so the
        # queue rather than trade-throughs decides the fills.
        market = random_market(seed, n_snapshots=400, max_step=0, through_share=0.0)
        quoter = Fixed(bid=int(market.bid_px[0, 0]), ask=int(market.ask_px[0, 0]))
        volume = _maker_volume(market, quoter, held)
        assert volume[CancelAttribution.PESSIMISTIC] <= volume[CancelAttribution.PROPORTIONAL]
        assert volume[CancelAttribution.PROPORTIONAL] <= volume[CancelAttribution.OPTIMISTIC]
        strict += volume[CancelAttribution.PESSIMISTIC] < volume[CancelAttribution.OPTIMISTIC]
    assert strict > 30

    total = dict.fromkeys(CancelAttribution, 0.0)
    for seed in range(40):
        for rule, value in _maker_volume(
            random_market(seed, n_snapshots=400), TouchQuoter(), FUZZ
        ).items():
            total[rule] += value
    assert total[CancelAttribution.PESSIMISTIC] < total[CancelAttribution.PROPORTIONAL]
    assert total[CancelAttribution.PROPORTIONAL] < total[CancelAttribution.OPTIMISTIC]


# ---------------------------------------------------------------------------
# The whole simulator, fuzzed
# ---------------------------------------------------------------------------


def test_no_fill_without_a_print_at_or_through_the_price() -> None:
    """200 random streams: every passive fill has a print at its timestamp, on
    the side that hits it, at its price (a queue fill) or through it."""
    checked = 0
    for seed in range(200):
        market = random_market(seed, n_snapshots=150, prints_per_snapshot=2.0)
        result = simulate_day(market, TouchQuoter(), FUZZ)
        orders = result.orders.set_index("oid")
        for fill in result.fills.query("maker").itertuples():
            price = int(orders.loc[fill.oid, "price"])
            here = market.trade_ts == fill.ts
            hitting = market.trade_aggressor[here] == -fill.side
            beyond = (price - market.trade_px[here][hitting]) * fill.side
            if fill.path == QUEUE:
                assert np.any(beyond == 0), (seed, fill)
            else:
                assert fill.path == THROUGH
                assert np.any(beyond > 0), (seed, fill)
            assert fill.price == pytest.approx(price * market.spec.tick)
            checked += 1
    assert checked > 5000


def test_a_crossed_snapshot_alone_never_fills() -> None:
    """The historical book cannot contain our order, so a snapshot whose ask is
    at our bid says nothing about whether we traded."""
    bids, asks = _book(100, 102)
    crossed_bids, crossed_asks = _book(99, 100)
    market = hand_built_market(
        [(0.0, bids, asks), (100.0, bids, asks), (200.0, crossed_bids, crossed_asks)],
        [(150.0, 102, 1.0, 1), (250.0, 100, 1.0, 1)],
    )
    result = simulate_day(market, Fixed(bid=100), CONFIG)
    assert result.fills.query("maker").empty
    assert result.counters["crossed_episodes"] == 1
    assert len(result.crossed) == 1


def test_assume_filled_reports_more_fills_than_trade_tape() -> None:
    bids, asks = _book(100, 102)
    crossed_bids, crossed_asks = _book(99, 100)
    market = hand_built_market(
        [(0.0, bids, asks), (100.0, bids, asks), (200.0, crossed_bids, crossed_asks)],
        [(150.0, 102, 1.0, 1)],
    )
    tape = simulate_day(market, Fixed(bid=100), CONFIG)
    assumed = simulate_day(
        market, Fixed(bid=100), CONFIG.with_(crossed_policy=CrossedPolicy.ASSUME_FILLED)
    )
    assert len(assumed.fills.query("maker")) > len(tape.fills.query("maker"))
    assert assumed.fills.query("maker")["path"].tolist() == ["crossed"]
