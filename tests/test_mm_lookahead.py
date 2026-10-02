"""No look-ahead: nothing after a decision can change it.

The simulator is run twice on streams that agree up to a cutoff and differ
completely after it. Every view a quoter was given, every order decided and
every fill booked before the cutoff must be identical, bit for bit. A rolling
statistic computed over the whole day, a centred window or a signal joined at
the start of its bin would each fail this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from trading_research.data.grid import to_grid
from trading_research.market_making.events import NS_PER_S, DayEvents, assemble
from trading_research.market_making.quoters import MarketView, Quote, Quotes, TouchQuoter
from trading_research.market_making.signals import SignalTape
from trading_research.market_making.simulator import (
    OracleRefused,
    SimConfig,
    run_days,
    simulate_day,
)
from trading_research.market_making.synthetic import SYNTHETIC_DAY, random_market

#: A clip set by the trailing touch (a short window, so it moves inside the
#: stream) and capped by notional; a short volatility half-life for the same reason.
CONFIG = SimConfig(
    clip_notional=600.0,
    clip_touch_share=0.5,
    warmup_s=0.0,
    touch_window_s=20,
    vol_half_life_s=5.0,
)


@dataclass
class Recording:
    """The touch quoter, writing down everything it was shown."""

    name: str = "recording"
    uses_future: bool = False
    seen: list[tuple[Any, ...]] = field(default_factory=list)

    def quotes(self, view: MarketView, position: float) -> Quotes:
        self.seen.append(
            (
                view.ts,
                view.best_bid,
                view.best_ask,
                tuple(view.bid_px),
                tuple(view.ask_sz),
                view.mid,
                view.vol_bp_1m,
                view.touch_median_1h,
                view.clip,
                view.soft_limit,
                tuple(sorted(view.signals.items())),
                position,
            )
        )
        return TouchQuoter().quotes(view, position)


def _splice(early: DayEvents, late: DayEvents, cutoff: int) -> DayEvents:
    """``early`` before ``cutoff`` and ``late`` from it on."""
    book_early = early.book_ts < cutoff
    book_late = late.book_ts >= cutoff
    trade_early = early.trade_ts < cutoff
    trade_late = late.trade_ts >= cutoff

    def join(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.concatenate([a, b])

    return assemble(
        early.spec,
        early.day,
        book_ts=join(early.book_ts[book_early], late.book_ts[book_late]),
        bid_px=join(early.bid_px[book_early], late.bid_px[book_late]),
        bid_sz=join(early.bid_sz[book_early], late.bid_sz[book_late]),
        ask_px=join(early.ask_px[book_early], late.ask_px[book_late]),
        ask_sz=join(early.ask_sz[book_early], late.ask_sz[book_late]),
        trade_ts=join(early.trade_ts[trade_early], late.trade_ts[trade_late]),
        trade_px=join(early.trade_px[trade_early], late.trade_px[trade_late]),
        trade_sz=join(early.trade_sz[trade_early], late.trade_sz[trade_late]),
        trade_aggressor=join(early.trade_aggressor[trade_early], late.trade_aggressor[trade_late]),
    )


def _tape(market: DayEvents, seed: int) -> SignalTape:
    rng = np.random.default_rng(seed)
    stamps = market.day_start_ns + np.arange(1, 200, dtype=np.int64) * NS_PER_S
    return SignalTape("index", stamps, rng.normal(size=len(stamps)))


@pytest.mark.parametrize("seed", range(10))
def test_mutating_the_future_does_not_change_past_decisions(seed: int) -> None:
    original = random_market(seed, n_snapshots=900, prints_per_snapshot=1.5)
    other = random_market(seed + 1_000, n_snapshots=900, prints_per_snapshot=1.5)
    cutoff = int(original.book_ts[450]) + 37_000_000
    mutated = _splice(original, other, cutoff)
    tape, other_tape = _tape(original, seed), _tape(original, seed + 1_000)
    later = other_tape.ts >= cutoff
    mutated_tape = SignalTape("index", tape.ts, np.where(later, other_tape.values, tape.values))

    first, second = Recording(), Recording()
    a = simulate_day(original, first, CONFIG, {"index": tape})
    b = simulate_day(mutated, second, CONFIG, {"index": mutated_tape})

    # Compared as text, so that NaN (no value yet) equals NaN.
    before = [repr(view) for view in first.seen if view[0] < cutoff]
    assert len(before) > 100
    assert before == [repr(view) for view in second.seen if view[0] < cutoff]
    assert any(view[0] >= cutoff for view in first.seen)

    decided = ["oid", "side", "price", "size", "decided_ns", "live_ns", "position_at_decision"]
    pd.testing.assert_frame_equal(
        a.orders.loc[a.orders["decided_ns"] < cutoff, decided].reset_index(drop=True),
        b.orders.loc[b.orders["decided_ns"] < cutoff, decided].reset_index(drop=True),
    )
    booked = [
        "ts", "oid", "side", "price", "size", "maker", "fee", "path",
        "queue_wait_ns", "position_after", "mid_ref",
    ]  # fmt: skip
    fills_a = a.fills.loc[a.fills["ts"] < cutoff, booked].reset_index(drop=True)
    assert len(fills_a) > 0
    pd.testing.assert_frame_equal(
        fills_a, b.fills.loc[b.fills["ts"] < cutoff, booked].reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        a.equity[a.equity["ts"] < cutoff].reset_index(drop=True),
        b.equity[b.equity["ts"] < cutoff].reset_index(drop=True),
    )


def _clock_book(seconds: float, step_ms: int = 700) -> pd.DataFrame:
    """A book whose bid is the second it was observed at."""
    start = pd.Timestamp(SYNTHETIC_DAY.isoformat(), tz="UTC")
    offsets = np.arange(0, int(seconds * 1000), step_ms)
    stamps = start + pd.to_timedelta(offsets, unit="ms")
    return pd.DataFrame(
        {"timestamp": stamps, "bid_price_0": offsets / 1000.0, "ask_price_0": offsets / 1000.0}
    )


def test_grid_signals_are_joined_at_the_end_of_their_bin() -> None:
    """Each value a decision reads was observed strictly before the decision.
    Joined at the bin's start instead, a value from up to five seconds later
    would be read."""
    book = _clock_book(60.0)
    start = int(pd.Timestamp(SYNTHETIC_DAY.isoformat(), tz="UTC").value)
    decisions = start + np.arange(1, 55_000, 37, dtype=np.int64) * 1_000_000
    elapsed = (decisions - start) / 1e9

    right = SignalTape.from_grid(
        "clock", to_grid(book, 5, label="right"), "bid_price_0", seconds=5, label="right"
    )
    seen = right.strictly_before(decisions)
    known = ~np.isnan(seen)
    assert known.mean() > 0.9
    assert np.all(seen[known] < elapsed[known])

    left_grid = to_grid(book, 5, label="left")
    relabelled = SignalTape.from_grid("clock", left_grid, "bid_price_0", seconds=5, label="left")
    np.testing.assert_array_equal(relabelled.ts, right.ts)
    np.testing.assert_array_equal(relabelled.values, right.values)

    naive = SignalTape(
        "clock",
        relabelled.ts - 5 * NS_PER_S,  # the left labels, taken at face value
        relabelled.values,
    )
    peeked = naive.strictly_before(decisions)
    ahead = peeked[~np.isnan(peeked)] - elapsed[~np.isnan(peeked)]
    assert ahead.max() > 4.0  # the look-ahead the right label removes


def test_the_simulator_hands_quoters_only_past_signals() -> None:
    market = random_market(2, n_snapshots=300)
    stamps = market.day_start_ns + np.arange(0, 40_000, 250, dtype=np.int64) * 1_000_000
    tape = SignalTape("clock", stamps, stamps.astype(np.float64))
    recording = Recording()
    simulate_day(market, recording, CONFIG, {"clock": tape})
    for view in recording.seen:
        ((name, value),) = view[10]
        assert name == "clock"
        assert value < view[0]


@dataclass(frozen=True)
class Oracle:
    name: str = "oracle"
    uses_future: bool = True

    def quotes(self, view: MarketView, position: float) -> Quotes:
        return TouchQuoter().quotes(view, position)


def test_oracle_forecasts_are_refused_outside_the_ladder(tmp_path: Path) -> None:
    market = random_market(0)
    with pytest.raises(OracleRefused):
        simulate_day(market, Oracle(), CONFIG)
    assert simulate_day(market, Oracle(), CONFIG, allow_oracle=True).oracle
    assert not simulate_day(market, TouchQuoter(), CONFIG).oracle
    with pytest.raises(OracleRefused):
        run_days(
            "X",
            [SYNTHETIC_DAY],
            Oracle,
            CONFIG,
            book_roots=[tmp_path],
            trades_root=tmp_path,
            funding_root=None,
        )


def test_the_reference_mid_of_a_fill_is_the_book_before_it() -> None:
    """A sale through our bid at 100 ms, and a snapshot at 100 ms that already
    shows the move: the fill's reference mid is the book before the print."""
    from trading_research.market_making.synthetic import hand_built_market

    def book(best_bid: int, best_ask: int):
        return (
            [(best_bid - i, 10.0) for i in range(3)],
            [(best_ask + i, 10.0) for i in range(3)],
        )

    market = hand_built_market(
        [(0.0, *book(100, 102)), (100.0, *book(98, 100)), (200.0, *book(98, 100))],
        [(100.0, 99, 1.0, -1)],
    )

    @dataclass(frozen=True)
    class Bid100:
        name: str = "bid-100"
        uses_future: bool = False

        def quotes(self, view: MarketView, position: float) -> Quotes:
            return Quotes(Quote(100, view.clip), Quote(None, 0.0))

    fills = simulate_day(market, Bid100(), SimConfig(clip_notional=1e12, warmup_s=0.0)).fills
    fill = fills[fills["maker"]].iloc[0]
    assert fill["mid_ref"] == pytest.approx(1.01)
    assert fill["markout_1s"] == pytest.approx((0.99 - 1.00) / 1.00 * 1e4)
