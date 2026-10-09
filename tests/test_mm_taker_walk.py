"""The simulator's taker orders: the shared walk, late snapshots, a fixed clip, the order hook."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from trading_research.backtest import walk
from trading_research.backtest.taker import TakerBook
from trading_research.market_making import simulator
from trading_research.market_making.orders import OwnOrder
from trading_research.market_making.quoters import NO_QUOTE, MarketView, Quote, Quotes, TouchQuoter
from trading_research.market_making.simulator import SimConfig, simulate_day
from trading_research.market_making.synthetic import hand_built_market, random_market

MS = 1_000_000


def _old_walk(prices: np.ndarray, sizes: np.ndarray, size: float) -> tuple[float, bool]:
    """The simulator's walk before it called the shared function, as it was written."""
    depth = len(prices)
    left = size
    cost = 0.0
    for level in range(depth):
        take = min(left, float(sizes[level]))
        cost += take * float(prices[level])
        left -= take
        if left <= 0.0:
            break
    beyond = left > 0.0
    if beyond:
        cost += left * float(prices[depth - 1])
    return cost / size, beyond


def test_the_shared_walk_equals_the_simulators_former_walk_to_the_last_bit() -> None:
    rng = np.random.default_rng(7)
    for seed in range(5):
        events = random_market(seed, n_snapshots=400)
        for row in rng.integers(0, len(events.book_ts), 50):
            for prices, sizes in (
                (events.ask_px[row], events.ask_sz[row]),
                (events.bid_px[row], events.bid_sz[row]),
            ):
                size = float(rng.integers(1, 200))
                vwap, beyond = _old_walk(prices, sizes, size)
                walked = walk.walk_book(prices, sizes, size)
                assert walked.vwap[0] == vwap
                assert bool(walked.beyond[0] > 0.0) == beyond


def test_flatten_fills_are_priced_by_the_executors_walk() -> None:
    """With no slippage on top, every taker fill of the simulator is the walk of
    the snapshot it executed on, and the taker executor's walk of the same
    snapshot in prices gives the same price."""
    assert simulator.walk_book is walk.walk_book
    config = SimConfig(
        clip_notional=250.0, clip_touch_share=1e9, warmup_s=0.0, soft_limit_clips=1.0
    ).with_(flatten_slippage_bp=0.0)
    checked = 0
    for seed in range(10):
        events = random_market(seed, n_snapshots=300, prints_per_snapshot=1.5)
        book = TakerBook.from_events(events)
        fills = simulate_day(events, TouchQuoter(), config).fills
        for fill in fills[fills["path"] == "flatten"].itertuples():
            row = int(np.searchsorted(events.book_ts, fill.ts, side="right")) - 1
            ticks = events.ask_px[row] if fill.side == 1 else events.bid_px[row]
            sizes = events.ask_sz[row] if fill.side == 1 else events.bid_sz[row]
            expected = walk.walk_book(ticks, sizes, fill.size).vwap[0] * events.spec.tick
            assert fill.price == expected
            prices = book.ask_px[row] if fill.side == 1 else book.bid_px[row]
            in_prices = walk.walk_book(prices, sizes, fill.size).vwap[0]
            assert fill.price == pytest.approx(in_prices, rel=1e-12)
            checked += 1
    assert checked > 20


@dataclass
class BuyThenCross:
    """Bids at the touch while flat; crosses out as soon as it holds anything."""

    name: str = "buy-then-cross"
    uses_future: bool = False
    seen: list[tuple[int, int, int]] = field(default_factory=list)

    def observe(self, own: tuple[OwnOrder, ...], opened_ns: int) -> None:
        self.seen.append((len(own), opened_ns, 0))

    def quotes(self, view: MarketView, position: float) -> Quotes:
        orders_out, opened, _ = self.seen[-1]
        self.seen[-1] = (orders_out, opened, view.ts)
        if position > 0:
            return Quotes(NO_QUOTE, NO_QUOTE, -position)
        return Quotes(Quote(view.best_bid, view.clip), NO_QUOTE)


def _book(best_bid: int, best_ask: int, size: float = 10.0):
    return (
        [(best_bid - i, size) for i in range(3)],
        [(best_ask + i, size) for i in range(3)],
    )


def _late_market(gap_ms: float):
    book = _book(100, 102)
    return hand_built_market(
        [(0.0, *book), (100.0, *book), (200.0, *book), (200.0 + gap_ms, *book)],
        [(150.0, 99, 5.0, -1)],
    )


def test_a_taker_order_walking_a_snapshot_a_second_late_is_counted() -> None:
    config = SimConfig(clip_btc=1.0, warmup_s=0.0, soft_limit_clips=1.0)
    late = simulate_day(_late_market(1_600.0), BuyThenCross(), config)
    assert late.counters["exit_late_snapshot"] == 1
    flatten = late.fills[late.fills["path"] == "flatten"].iloc[0]
    assert flatten["ts"] - _late_market(1_600.0).day_start_ns == 1_800 * MS
    on_time = simulate_day(_late_market(100.0), BuyThenCross(), config)
    assert on_time.counters.get("exit_late_snapshot", 0.0) == 0.0


def test_the_quoter_is_told_its_orders_out_and_when_the_position_opened() -> None:
    market = _late_market(100.0)
    quoter = BuyThenCross()
    simulate_day(market, quoter, SimConfig(clip_btc=1.0, warmup_s=0.0, soft_limit_clips=1.0))
    start = market.day_start_ns
    assert quoter.seen[0] == (0, -1, start)
    assert quoter.seen[1] == (1, -1, start + 100 * MS)
    # The bid filled at 150 ms: nothing out, and the position opened then.
    assert quoter.seen[2] == (0, start + 150 * MS, start + 200 * MS)


def test_a_fixed_clip_replaces_the_notional_and_touch_caps() -> None:
    events = random_market(3, n_snapshots=300, prints_per_snapshot=1.5)
    config = SimConfig(clip_btc=2.0, clip_touch_share=1e-9, warmup_s=0.0, soft_limit_clips=1.0)
    result = simulate_day(events, TouchQuoter(), config)
    assert (result.orders["size"] == 2.0).all()
    assert result.counters["max_abs_position"] <= 4.0 + 1e-9
    with pytest.raises(ValueError, match="exactly one"):
        SimConfig()
    with pytest.raises(ValueError, match="exactly one"):
        SimConfig(clip_notional=1.0, clip_btc=1.0)
    with pytest.raises(ValueError, match="positive"):
        SimConfig(clip_btc=0.0)
