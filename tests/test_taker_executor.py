"""The event-time taker executor: timing, depth, one position, late books, funding, money."""

from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd
import pytest

from trading_research.backtest import taker, walk
from trading_research.backtest.costs import FeeTier
from trading_research.backtest.taker import TakerBook, TakerSettings, execute_day
from trading_research.market_making.events import NS_PER_S, day_start_ns
from trading_research.market_making.synthetic import SYNTHETIC_DAY, random_market

START = day_start_ns(SYNTHETIC_DAY)
MINUTE = 60 * NS_PER_S
MS = 1_000_000


def _book(stamps_ms: list[float], bid: float = 100.0, ask: float = 100.5, size: float = 1.0):
    """Two levels a side, 0.5 apart, the touch moving by ``bid`` per row if given as a list."""
    rows = len(stamps_ms)
    bids = np.full(rows, bid) if np.isscalar(bid) else np.asarray(bid, dtype=float)
    asks = np.full(rows, ask) if np.isscalar(ask) else np.asarray(ask, dtype=float)
    return TakerBook(
        SYNTHETIC_DAY,
        np.array([START + round(t * MS) for t in stamps_ms], dtype=np.int64),
        np.stack([bids, bids - 0.5], axis=1),
        np.full((rows, 2), size),
        np.stack([asks, asks + 0.5], axis=1),
        np.full((rows, 2), size),
    )


def _ms(t_ms: float) -> int:
    return START + round(t_ms * MS)


def test_a_long_round_trip_known_answer() -> None:
    """Decision at 00:01:00.000; the entry walks the 100 ms row (ask 100.5 for
    the first unit, 101.0 for the second); the exit is decided at the first row
    at or after t + 1 s and walks the row after its arrival."""
    rows = [60_000.0 + 100 * i for i in range(30)]
    bids = [100.0] * 11 + [102.0] * 19
    book = _book(rows, bid=bids, ask=[b + 0.5 for b in bids])
    settings = TakerSettings(clip=2.0, fees=FeeTier("t", 0.0, 5.0))
    day = execute_day(book, np.array([_ms(60_000)]), np.array([1]), NS_PER_S, settings)
    (trade,) = day.trades.itertuples()
    assert trade.entry_snapshot_ns == _ms(60_100)
    assert trade.entry_price == pytest.approx(100.75)
    assert trade.exit_decision_ns == _ms(61_000)
    assert trade.exit_snapshot_ns == _ms(61_100)
    assert trade.exit_price == pytest.approx(101.75)  # bids 102 and 101.5
    assert trade.gross == pytest.approx(2.0)
    assert trade.fees == pytest.approx((100.75 + 101.75) * 2 * 5e-4)
    assert trade.net == pytest.approx(day.net)
    assert trade.net_bp == pytest.approx(trade.net / (100.75 * 2) * 1e4)
    assert day.turnover == pytest.approx((100.75 + 101.75) * 2)
    assert day.day_bp("gross") == pytest.approx(2.0 / 201.5 * 1e4)


def test_every_fill_walks_a_snapshot_at_or_after_its_decision_plus_the_latency() -> None:
    """The K-latency unit test, at 10 ms and at 110 ms, on random books."""
    for seed in range(5):
        events = random_market(seed, n_snapshots=900, lot=0.001)
        book = TakerBook.from_events(events)
        rng = np.random.default_rng(seed)
        stamps = np.sort(rng.choice(book.ts[:-50], 60, replace=False))
        sides = rng.choice([-1, 1], len(stamps))
        for latency in (10 * MS, 110 * MS):
            settings = TakerSettings(clip=0.002, latency_ns=latency, no_entry_before=time(0, 0))
            day = execute_day(book, stamps, sides, 2 * NS_PER_S, settings)
            trades = day.trades
            assert len(trades) > 0
            assert (trades["entry_snapshot_ns"] >= trades["decision_ns"] + latency).all()
            assert (trades["exit_snapshot_ns"] >= trades["exit_decision_ns"] + latency).all()
            assert (trades["exit_decision_ns"] >= trades["decision_ns"] + 2 * NS_PER_S).all()


def test_one_position_at_a_time() -> None:
    """A decision while the position is open is busy; one at the stamp of the
    snapshot the exit walked is free."""
    rows = [60_000.0 + 100 * i for i in range(40)]
    book = _book(rows)
    decisions = np.array([_ms(60_000), _ms(60_500), _ms(61_100), _ms(61_200)])
    day = execute_day(book, decisions, np.array([1, -1, -1, 1]), NS_PER_S)
    assert day.trades["decision_ns"].tolist() == [_ms(60_000), _ms(61_100)]
    assert day.trades["exit_snapshot_ns"].iloc[0] == _ms(61_100)
    assert day.counters["entries_busy"] == 2
    assert day.counters["entries"] == 2


def test_an_entry_beyond_the_visible_depth_is_skipped_and_an_exit_is_charged_for_it() -> None:
    rows = [60_000.0 + 100 * i for i in range(30)]
    deep = _book(rows, size=1.0)
    one = np.array([_ms(60_000)])
    skipped = execute_day(deep, one, np.array([1]), NS_PER_S, TakerSettings(clip=2.5))
    assert skipped.trades.empty
    assert skipped.counters["entries_skipped"] == 1
    # The bids thin out after the entry: the exit takes 1 + 0.5 and charges the
    # last 0.5 at the deepest bid.
    thin = TakerBook(
        deep.day,
        deep.ts,
        deep.bid_px,
        np.where(np.arange(30)[:, None] >= 11, 0.5, 1.0) * np.ones((30, 2)),
        deep.ask_px,
        deep.ask_sz,
    )
    day = execute_day(thin, one, np.array([1]), NS_PER_S, TakerSettings(clip=1.5))
    (trade,) = day.trades.itertuples()
    assert trade.exit_beyond == pytest.approx(0.5)
    assert trade.exit_price == pytest.approx((0.5 * 100 + 0.5 * 99.5 + 0.5 * 99.5) / 1.5)
    assert day.counters["flatten_beyond_visible_book"] == 1
    assert walk.walk_book is taker.walk_book


def test_late_snapshots_are_counted_and_exclude_nothing() -> None:
    """A 1.5 s gap after the entry's arrival, and another after the exit's."""
    rows = [60_000.0, 61_510.0, 61_600.0, 62_000.0, 63_600.0, 63_700.0]
    day = execute_day(_book(rows), np.array([_ms(60_000)]), np.array([-1]), 2 * NS_PER_S)
    (trade,) = day.trades.itertuples()
    assert trade.entry_late and trade.exit_late
    assert trade.exit_decision_ns == _ms(62_000)
    assert trade.exit_snapshot_ns == _ms(63_600)
    assert day.counters["entry_late_snapshot"] == 1
    assert day.counters["exit_late_snapshot"] == 1
    assert not day.excluded


def test_entry_hours_and_the_day_end_flatten() -> None:
    late = 23 * 3_600_000.0 + 58 * 60_000.0
    rows = [59_000.0, 60_000.0, 60_100.0, late - 100, late, late + 100]
    rows += [late + 60_000.0 + 100 * i for i in range(3)]
    decisions = np.array([_ms(59_000), _ms(late - 100), _ms(late + 100)])
    day = execute_day(_book(rows), decisions, np.array([1, 1, 1]), 120 * NS_PER_S)
    assert day.counters["entries_outside_hours"] == 2
    (trade,) = day.trades.itertuples()
    assert trade.exit_reason == "day_end"
    assert trade.exit_decision_ns == _ms(late + 60_000)
    assert trade.exit_snapshot_ns == _ms(late + 60_100)


def test_an_exit_after_the_book_ends_walks_the_last_snapshot_and_may_flag_the_day() -> None:
    rows = [60_000.0 + 100 * i for i in range(5)]
    fresh = execute_day(_book(rows), np.array([_ms(60_000)]), np.array([1]), 4 * NS_PER_S)
    assert fresh.trades["exit_snapshot_ns"].iloc[0] == _ms(60_400)
    assert not fresh.excluded
    stale = execute_day(_book(rows), np.array([_ms(60_000)]), np.array([1]), 6 * NS_PER_S)
    assert stale.flags == ("flatten_stale",)
    assert stale.excluded


def test_funding_is_charged_on_the_position_held_at_the_settlement() -> None:
    eight = 8 * 3_600_000.0
    rows = [eight - 1_000.0 + 100 * i for i in range(30)]
    bids = [100.0 + i for i in range(30)]
    book = _book(rows, bid=bids, ask=[b + 0.5 for b in bids])
    planes = (book.bid_px, book.bid_sz, book.ask_px, book.ask_sz)
    book = TakerBook(book.day, book.ts, *planes, np.array([_ms(eight)]), np.array([1e-3]))
    held = execute_day(book, np.array([_ms(eight - 1_000)]), np.array([1]), NS_PER_S)
    # Long across 08:00: pays the rate on the clip at the mid of 07:59:59.900.
    mark = 100.0 + 9 + 0.25
    assert held.trades["funding"].iloc[0] == pytest.approx(-0.010 * mark * 1e-3)
    assert held.counters["funding_settlements_with_position"] == 1
    # Entry filled at 08:00:00.000 exactly: not yet held at the settlement.
    after = execute_day(book, np.array([_ms(eight - 100)]), np.array([1]), NS_PER_S)
    assert after.trades["entry_snapshot_ns"].iloc[0] == _ms(eight)
    assert after.trades["funding"].iloc[0] == 0.0
    # Exit filled at 08:00:00.000 exactly: still held.
    short = np.array([_ms(eight - 1_000)])
    exit_there = execute_day(book, short, np.array([-1]), 900 * MS)
    assert exit_there.trades["exit_snapshot_ns"].iloc[0] == _ms(eight)
    assert exit_there.trades["funding"].iloc[0] == pytest.approx(0.010 * 109.25 * 1e-3)


def test_books_from_a_frame_and_bad_input() -> None:
    from trading_research.data.schema import (
        ask_price_col,
        ask_size_col,
        bid_price_col,
        bid_size_col,
    )

    stamps = pd.to_datetime([_ms(60_000), _ms(60_100)], utc=True)
    frame = pd.DataFrame({"timestamp": stamps})
    for level in range(2):
        frame[bid_price_col(level)] = 100.0 - level
        frame[bid_size_col(level)] = 1.0
        frame[ask_price_col(level)] = 101.0 + level
        frame[ask_size_col(level)] = 1.0
    book = TakerBook.from_frame(frame, SYNTHETIC_DAY, depth=2)
    assert book.ts.tolist() == [_ms(60_000), _ms(60_100)]
    assert book.mid.tolist() == [100.5, 100.5]
    with pytest.raises(ValueError, match="time order"):
        execute_day(book, np.array([_ms(60_100), _ms(60_000)]), np.array([1, 1]), NS_PER_S)
    with pytest.raises(ValueError, match="side"):
        execute_day(book, np.array([_ms(60_000)]), np.array([0]), NS_PER_S)
    with pytest.raises(ValueError, match="hold"):
        execute_day(book, np.array([_ms(60_000)]), np.array([1]), 0)
