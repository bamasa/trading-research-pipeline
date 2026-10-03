"""The strategies' quoting rules, on hand-built views and synthetic markets.

Each rule the pre-registration states is checked where it can be seen
directly: S1's skew moves both quotes against the inventory and never improves
the touch; S2 improves one tick only when the spread and the gate allow it;
S3 pulls its quotes for exactly its window after a flag; S4 leans against the
index's move; X1 posts the fading side at the touch, exits at the opposite
touch and crosses for whatever is left when the exit times out.
"""

from __future__ import annotations

import math
from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

from trading_research.market_making.events import NS_PER_S, day_start_ns
from trading_research.market_making.quoters import (
    DIRECTION_SIGNAL,
    FLAG_SIGNAL,
    INDEX_SIGNAL,
    LEAN_TRIGGER_SIGNAL,
    NO_QUOTE,
    TRIGGER_SIGNAL,
    InsideQuoter,
    MarketView,
    Quote,
    Quotes,
    RegimeGuard,
    ReversionLean,
    SignalExecutor,
    SkewQuoter,
)
from trading_research.market_making.signals import SignalTape
from trading_research.market_making.simulator import SimConfig, simulate_day
from trading_research.market_making.synthetic import (
    SYNTHETIC_DAY,
    hand_built_market,
    random_market,
)

#: One tick is one basis point at a price of 1.0.
TICK = 0.0001
T0 = day_start_ns(SYNTHETIC_DAY) + 3600 * NS_PER_S


def view(
    *,
    bid: int = 10_000,
    ask: int = 10_004,
    vol: float = 5.0,
    clip: float = 1.0,
    soft: float = 6.0,
    ts: int = T0,
    tick: float = TICK,
    signals: dict[str, float] | None = None,
) -> MarketView:
    bid_px = np.array([bid - i for i in range(10)], dtype=np.int64)
    ask_px = np.array([ask + i for i in range(10)], dtype=np.int64)
    sizes = np.full(10, 50.0)
    return MarketView(
        ts,
        tick,
        bid_px,
        sizes,
        ask_px,
        sizes,
        bid,
        ask,
        (bid + ask) * 0.5 * tick,
        vol,
        50.0,
        clip,
        soft,
        MappingProxyType(signals or {}),
    )


def s1(**changes: float) -> SkewQuoter:
    base = SkewQuoter(skew_bp=5.0, k=0.0, min_edge_bp=3.0, sigma_ref=5.0, maker_bp=2.0)
    return replace(base, **changes)


# -- S1 ----------------------------------------------------------------------


def test_skew_lowers_both_quotes_when_long_and_raises_them_when_short() -> None:
    narrow = {"bid": 10_000, "ask": 10_002}
    quoter = s1()
    flat = quoter.quotes(view(**narrow), 0.0)
    long = quoter.quotes(view(**narrow), 3.0)
    short = quoter.quotes(view(**narrow), -3.0)
    assert long.bid.price < flat.bid.price and long.ask.price < flat.ask.price
    assert short.bid.price > flat.bid.price and short.ask.price > flat.ask.price
    # The reservation price moves by skew_bp * q / Q_soft of the mid, in bp.
    mid = view(**narrow).mid
    shift = (quoter.reservation(view(**narrow), 3.0) - mid) / mid * 1e4
    assert shift == pytest.approx(-5.0 * 3.0 / 6.0)


def test_skew_scales_with_volatility_inside_its_clamp() -> None:
    quoter = s1()
    base = view(vol=5.0)
    moves = {}
    for vol in (1.0, 5.0, 10.0, 40.0):
        here = view(vol=vol)
        moves[vol] = (quoter.reservation(here, 3.0) - here.mid) / here.mid * 1e4
    assert moves[10.0] == pytest.approx(4.0 * moves[5.0])  # (10 / 5) ** 2
    assert moves[40.0] == pytest.approx(4.0 * moves[5.0])  # clamped at 4
    assert moves[1.0] == pytest.approx(0.25 * moves[5.0])  # clamped at 0.25
    assert quoter.reservation(base, 0.0) == pytest.approx(base.mid)


def test_s1_never_improves_the_touch() -> None:
    rng = np.random.default_rng(3)
    for _ in range(500):
        bid = 10_000 + int(rng.integers(-50, 50))
        here = view(bid=bid, ask=bid + int(rng.integers(1, 12)), vol=float(rng.uniform(0.5, 30)))
        quoter = s1(
            skew_bp=float(rng.choice([2, 5, 10])),
            k=float(rng.choice([0, 0.5, 1, 2])),
            min_edge_bp=float(rng.choice([0, 1, 2, 4])),
        )
        out = quoter.quotes(here, float(rng.uniform(-6, 6)))
        assert out.bid.price <= here.best_bid
        assert out.ask.price >= here.best_ask
        assert out.bid.size == here.clip and out.ask.size == here.clip


def test_the_gate_keeps_s1_off_a_one_tick_touch_it_cannot_pay_for() -> None:
    """A one-tick spread of 1.8 bp cannot clear a 2 bp maker fee: both quotes
    stand behind the touch."""
    tick = 0.00018
    here = view(bid=5555, ask=5556, tick=tick)
    out = s1(min_edge_bp=0.0).quotes(here, 0.0)
    assert out.bid.price < here.best_bid
    assert out.ask.price > here.best_ask


def test_s1_does_not_quote_without_a_volatility_or_a_limit() -> None:
    assert s1().quotes(view(vol=math.nan), 0.0) == Quotes(NO_QUOTE, NO_QUOTE)
    assert s1().quotes(view(soft=0.0), 0.0) == Quotes(NO_QUOTE, NO_QUOTE)


# -- S2 ----------------------------------------------------------------------


def s2(**changes: object) -> InsideQuoter:
    base = InsideQuoter(skew_bp=0.0, k=0.0, min_edge_bp=0.0, m_ticks=3, sigma_ref=5.0, maker_bp=1.5)
    return replace(base, **changes)


def test_inside_improves_only_when_the_spread_and_the_gate_allow() -> None:
    wide = view(bid=10_000, ask=10_006)  # six ticks of 1 bp; best + 1 is 2 bp from mid
    out = s2().quotes(wide, 0.0)  # a gate of 1.5 bp
    assert (out.bid.price, out.ask.price) == (10_001, 10_005)
    # The same spread with a gate of 2.5 bp: one tick inside no longer clears it.
    gated = s2(min_edge_bp=1.0).quotes(wide, 0.0)
    assert gated.bid.price <= 10_000 and gated.ask.price >= 10_006
    # A spread narrower than m: no improvement, whatever the gate.
    narrow = view(bid=10_000, ask=10_002)
    assert s2(m_ticks=3, maker_bp=0.0).quotes(narrow, 0.0).bid.price <= 10_000
    # The twin is S1 at the same parameters: the inside rule alone differs.
    twin = s2(inside=False)
    assert twin.quotes(wide, 0.0) == twin.base.quotes(wide, 0.0)
    assert twin.quotes(wide, 0.0).bid.price == 10_000


def test_at_two_ticks_only_the_side_that_reduces_inventory_improves() -> None:
    two = view(bid=10_000, ask=10_002)
    quoter = s2(m_ticks=2, maker_bp=0.0)
    flat = quoter.quotes(two, 0.0)
    assert (flat.bid.price, flat.ask.price) == (10_001, 10_002)  # the bid when flat
    long = quoter.quotes(two, 2.0)
    assert (long.bid.price, long.ask.price) == (10_000, 10_001)  # selling reduces |q|
    short = quoter.quotes(two, -2.0)
    assert (short.bid.price, short.ask.price) == (10_001, 10_002)


def test_an_improved_quote_is_always_post_only_safe() -> None:
    rng = np.random.default_rng(5)
    for _ in range(500):
        bid = 10_000 + int(rng.integers(-30, 30))
        here = view(bid=bid, ask=bid + int(rng.integers(1, 9)), vol=float(rng.uniform(0.5, 10)))
        quoter = s2(
            m_ticks=int(rng.choice([2, 3, 4])),
            maker_bp=float(rng.choice([0.0, 2.0])),
            skew_bp=float(rng.choice([0, 5])),
        )
        out = quoter.quotes(here, float(rng.uniform(-6, 6)))
        assert out.bid.price < here.best_ask and out.ask.price > here.best_bid
        assert out.bid.price < out.ask.price
        assert out.bid.price <= here.best_bid + 1 and out.ask.price >= here.best_ask - 1


# -- S3 ----------------------------------------------------------------------


def test_the_guard_pulls_for_exactly_its_window() -> None:
    flag = T0 / NS_PER_S
    guard = RegimeGuard(s1(), "pull", window_min=15.0)
    for seconds, pulled in ((-1, False), (1, True), (899, True), (900, True), (901, False)):
        here = view(ts=T0 + seconds * NS_PER_S, signals={FLAG_SIGNAL: flag})
        if seconds < 0:
            here = view(ts=T0 + seconds * NS_PER_S)  # the flag is not known yet
        out = guard.quotes(here, 0.0)
        assert (out == Quotes(NO_QUOTE, NO_QUOTE)) is pulled, seconds


def test_widening_doubles_the_half_spread() -> None:
    narrow = view(bid=10_000, ask=10_002, signals={FLAG_SIGNAL: T0 / NS_PER_S - 1})
    base = s1()
    widened = RegimeGuard(base, "widen", window_min=60.0).quotes(narrow, 0.0)
    expected = base.quote_prices(narrow, 0.0, delta_scale=2.0)
    assert (widened.bid.price, widened.ask.price) == expected
    assert widened.bid.price < base.quotes(narrow, 0.0).bid.price


def test_the_guard_places_no_order_inside_a_window_in_the_simulator() -> None:
    market = random_market(9, n_snapshots=3000, prints_per_snapshot=1.0)
    start = int(market.book_ts[0])
    effective = start + 100 * NS_PER_S
    window_s = 60.0
    tape = SignalTape(FLAG_SIGNAL, np.array([effective]), np.array([effective / NS_PER_S]))
    config = SimConfig(clip_notional=1e6, clip_touch_share=1e9, warmup_s=0.0, vol_half_life_s=5.0)
    quoter = RegimeGuard(InsideQuoter(2.0, 0.0, 0.0, 2, 1.0, maker_bp=0.0), "pull", window_min=1.0)
    result = simulate_day(market, quoter, config, {FLAG_SIGNAL: tape})
    decided = result.orders["decided_ns"].to_numpy()
    inside = (decided > effective) & (decided <= effective + window_s * NS_PER_S)
    assert not inside.any()
    assert (decided <= effective).any() and (decided > effective + window_s * NS_PER_S).any()
    # Every order resting when the flag became known was cancelled at once.
    cancels = result.orders["cancel_ns"].to_numpy()
    resting = (result.orders["live_ns"] <= effective) & (
        (result.orders["done_ns"] > effective) | (result.orders["done_ns"] < 0)
    )
    assert resting.any()
    assert (cancels[resting.to_numpy()] <= effective + 200_000_000).all()


# -- S4 ----------------------------------------------------------------------


def lean(**changes: object) -> ReversionLean:
    base = ReversionLean(s1(), beta=-0.5, lam=1.0, one_sided=False)
    return replace(base, **changes)


def lean_view(s: float, *, age_s: float = 30.0, **kwargs: object) -> MarketView:
    signals = {INDEX_SIGNAL: s, LEAN_TRIGGER_SIGNAL: T0 / NS_PER_S - age_s}
    return view(bid=10_000, ask=10_002, signals=signals, **kwargs)  # type: ignore[arg-type]


def test_the_lean_quotes_against_the_index_move() -> None:
    plain = s1().quotes(lean_view(0.0), 0.0)
    up = lean().quotes(lean_view(40.0), 0.0)  # the index rose: lean to sell
    down = lean().quotes(lean_view(-40.0), 0.0)
    assert up.bid.price < plain.bid.price and up.ask.price < plain.ask.price
    assert down.bid.price > plain.bid.price and down.ask.price > plain.ask.price
    # The shift is lam * beta * s in bp of the reservation price: -20 bp here,
    # seen on the bid (the ask is held at the touch, which S1 never improves).
    assert up.bid.price - plain.bid.price == pytest.approx(-20, abs=1)
    assert up.ask.price == 10_002
    # The flipped placebo leans the other way.
    flipped = lean(lam=-1.0).quotes(lean_view(40.0), 0.0)
    assert flipped.bid.price > plain.bid.price


def test_the_lean_ends_with_its_window() -> None:
    plain = s1().quotes(lean_view(40.0), 0.0)
    stale = lean().quotes(lean_view(40.0, age_s=601.0), 0.0)
    assert stale == plain
    fresh = lean().quotes(lean_view(40.0, age_s=599.0), 0.0)
    assert fresh != plain


def test_one_sided_lean_does_not_quote_the_side_that_follows_the_move() -> None:
    quoter = lean(one_sided=True)
    after_rise = quoter.quotes(lean_view(40.0), 0.0)
    assert after_rise.bid == NO_QUOTE and after_rise.ask.price is not None
    after_fall = quoter.quotes(lean_view(-40.0), 0.0)
    assert after_fall.ask == NO_QUOTE and after_fall.bid.price is not None
    outside = quoter.quotes(lean_view(40.0, age_s=700.0), 0.0)
    assert outside.bid.price is not None and outside.ask.price is not None


# -- X1 ----------------------------------------------------------------------


def x1_view(seconds: float, *, trigger: float = 0.0, direction: float = -1.0) -> MarketView:
    signals = {TRIGGER_SIGNAL: T0 / NS_PER_S + trigger, DIRECTION_SIGNAL: direction}
    return view(ts=T0 + round(seconds * NS_PER_S), signals=signals)


def test_x1_posts_the_fading_side_at_the_touch_and_waits_ten_minutes() -> None:
    executor = SignalExecutor()
    first = executor.quotes(x1_view(0.1, direction=-1.0), 0.0)  # the index rose: sell
    assert first.bid == NO_QUOTE
    assert first.ask == Quote(10_004, 1.0)
    assert executor.quotes(x1_view(599.0), 0.0).ask.price == 10_004
    assert executor.quotes(x1_view(601.0), 0.0) == Quotes(NO_QUOTE, NO_QUOTE)
    assert executor.counts == {"attempts": 1, "misses": 1}
    buyer = SignalExecutor()
    assert buyer.quotes(x1_view(0.1, direction=1.0), 0.0) == Quotes(Quote(10_000, 1.0), NO_QUOTE)


def test_x1_exits_at_the_opposite_touch_and_crosses_after_the_timeout() -> None:
    executor = SignalExecutor()
    executor.quotes(x1_view(0.1, direction=1.0), 0.0)
    exit_quote = executor.quotes(x1_view(30.0, direction=1.0), 0.8)  # partly filled
    assert exit_quote == Quotes(NO_QUOTE, Quote(10_004, 0.8))
    assert executor.quotes(x1_view(629.0), 0.8).ask.price == 10_004
    crossed = executor.quotes(x1_view(630.0), 0.8)
    assert crossed.cross == pytest.approx(-0.8)
    assert crossed.bid == NO_QUOTE and crossed.ask == NO_QUOTE
    assert executor.quotes(x1_view(630.1), 0.8).cross == 0.0  # sent once
    assert executor.quotes(x1_view(631.0), 0.0) == Quotes(NO_QUOTE, NO_QUOTE)
    assert executor.counts["exits_crossed"] == 1
    # The next trigger opens a new attempt.
    again = executor.quotes(x1_view(1300.0, trigger=1290.0, direction=-1.0), 0.0)
    assert again.ask.price == 10_004 and executor.counts["attempts"] == 2


def test_x1_skips_a_trigger_while_an_attempt_is_open() -> None:
    executor = SignalExecutor()
    executor.quotes(x1_view(0.1), 0.0)
    executor.quotes(x1_view(100.0, trigger=90.0), 0.0)
    assert executor.counts == {"attempts": 1, "skipped_busy": 1}


def _x1_market(exit_print: bool) -> object:
    """A quiet four-tick market; a trigger to buy at 01:00; a sale ten seconds
    later fills the bid, and, if asked, a purchase later fills the exit at the
    ask."""
    bid_levels = [(10_000 - i, 10.0) for i in range(10)]
    ask_levels = [(10_004 + i, 10.0) for i in range(10)]
    snapshots = [(float(ms), bid_levels, ask_levels) for ms in range(0, 7_200_000, 500)]
    prints = [(3_610_000.0, 10_000, 15.0, -1)]  # beyond the ten ahead of us: fills us
    if exit_print:
        prints.append((3_700_000.0, 10_004, 15.0, 1))
    prints.append((7_000_000.0, 10_004, 1.0, 1))  # a print far later: the stream goes on
    return hand_built_market(snapshots, prints, tick=TICK)


@pytest.mark.parametrize("exit_print", [True, False])
def test_x1_in_the_simulator_buys_the_fade_and_gets_out(exit_print: bool) -> None:
    market = _x1_market(exit_print)
    trigger_ns = day_start_ns(SYNTHETIC_DAY) + 3600 * NS_PER_S
    tapes = {
        TRIGGER_SIGNAL: SignalTape(
            TRIGGER_SIGNAL, np.array([trigger_ns]), np.array([trigger_ns / NS_PER_S])
        ),
        DIRECTION_SIGNAL: SignalTape(DIRECTION_SIGNAL, np.array([trigger_ns]), np.array([1.0])),
    }
    config = SimConfig(clip_notional=1e9, clip_touch_share=0.2, warmup_s=0.0)
    executor = SignalExecutor()
    result = simulate_day(market, executor, config, tapes)  # type: ignore[arg-type]
    fills = result.fills
    assert fills["side"].iloc[0] == 1 and bool(fills["maker"].iloc[0])  # the bid, passively
    assert fills["price"].iloc[0] == pytest.approx(10_000 * TICK)
    second = fills.iloc[1]
    if exit_print:
        assert second["side"] == -1 and bool(second["maker"])
        assert second["price"] == pytest.approx(10_004 * TICK)
        assert executor.counts.get("exits_crossed", 0) == 0
    else:
        # Ten minutes after the fill was seen, a reduce-only taker sale at the bid.
        assert second["side"] == -1 and not bool(second["maker"])
        assert second["ts"] - fills["ts"].iloc[0] == pytest.approx(600 * NS_PER_S, abs=1e9)
        assert result.counters["flattens_strategy"] == 1
        assert executor.counts["exits_crossed"] == 1
    assert fills["position_after"].iloc[-1] == 0.0
    table = executor.attempts(fills)
    assert len(table) == 1
    assert table["net"].sum() == pytest.approx(result.decomposition["net"])


def test_a_cross_never_grows_the_position() -> None:
    """The simulator honours a quoter's cross only as far as it reduces the
    position: from flat, a request to buy is refused."""

    class AlwaysCross:
        name = "cross"
        uses_future = False

        def quotes(self, view: MarketView, position: float) -> Quotes:
            return Quotes(NO_QUOTE, NO_QUOTE, 5.0)

    market = random_market(2, n_snapshots=200)
    config = SimConfig(clip_notional=1e6, clip_touch_share=1e9, warmup_s=0.0)
    result = simulate_day(market, AlwaysCross(), config)
    assert result.fills.empty
    assert result.counters["crosses_refused"] > 0


def test_attempt_table_assigns_fills_to_the_attempt_that_owns_them() -> None:
    executor = SignalExecutor()
    executor.log.extend([(1.0, 1, 100, 1.0, 2.0), (2.0, -1, 200, 1.0, 2.0)])
    fills = pd.DataFrame(
        {
            "ts": [110, 150, 250],
            "side": [1, -1, -1],
            "price": [1.0, 1.01, 1.0],
            "size": [2.0, 2.0, 2.0],
            "fee": [0.0, 0.0, 0.0],
        }
    )
    table = executor.attempts(fills)
    assert table["net"].tolist() == pytest.approx([0.02, 2.0])
    assert table["net_bp"].tolist() == pytest.approx([100.0, 1e4])
    assert table["filled"].tolist() == [True, True]
