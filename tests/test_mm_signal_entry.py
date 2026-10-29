"""The signal-entry quoter: timing, one position, first price, exits, pauses and flags."""

from __future__ import annotations

import numpy as np
import pytest

from trading_research.market_making.quoters import FLAG_SIGNAL, MarketView, Quotes
from trading_research.market_making.signal_entry import (
    ENTRY_SIGNAL,
    SignalEntryQuoter,
    entry_tapes,
)
from trading_research.market_making.signals import SignalTape
from trading_research.market_making.simulator import SimConfig, simulate_day
from trading_research.market_making.synthetic import hand_built_market, random_market

MS = 1_000_000
CONFIG = SimConfig(clip_btc=1.0, warmup_s=0.0, soft_limit_clips=1.0, flatten_slippage_bp=0.0)


def _book(best_bid: int, best_ask: int, size: float = 10.0):
    return (
        [(best_bid - i, size) for i in range(3)],
        [(best_ask + i, size) for i in range(3)],
    )


def _market(rows: int, prints=(), *, bids=None, gap_after: int | None = None, lot: float = 1.0):
    """Snapshots every 100 ms (a gap of 5.9 s after row ``gap_after``), bid 100 / ask 102."""
    stamps = [100.0 * i for i in range(rows)]
    if gap_after is not None:
        stamps = [t if i <= gap_after else t + 5_900.0 for i, t in enumerate(stamps)]
    bids = bids or [100] * rows
    snapshots = [(t, *_book(b, b + 2)) for t, b in zip(stamps, bids, strict=True)]
    prints = list(prints) or [(stamps[-1] - 1.0, 120, 1.0, 1)]
    return hand_built_market(snapshots, prints, lot=lot)


def _tapes(market, decisions_ms: list[float], sides: list[int]):
    start = market.day_start_ns
    stamps = np.array([start + round(t * MS) for t in decisions_ms], dtype=np.int64)
    return entry_tapes(stamps, np.array(sides))


class Spy(SignalEntryQuoter):
    """Records what each decision could see of the entry tape."""

    def __init__(self, hold_s: float) -> None:
        super().__init__(hold_s)
        self.seen: list[tuple[int, float]] = []

    def quotes(self, view: MarketView, position: float) -> Quotes:
        self.seen.append((view.ts, view.signals.get(ENTRY_SIGNAL, float("nan"))))
        return super().quotes(view, position)


def test_no_event_at_or_before_the_decision_sees_it_and_every_post_is_live_after_the_latency():
    for seed in range(4):
        market = random_market(seed, n_snapshots=600, prints_per_snapshot=1.5)
        stamps = market.book_ts[20::40]
        sides = np.where(np.arange(len(stamps)) % 2 == 0, 1, -1)
        quoter = Spy(1.0)
        result = simulate_day(market, quoter, CONFIG, entry_tapes(stamps, sides))
        for ts, seen in quoter.seen:
            assert not (stamps[stamps >= ts] / 1e9 == seen).any()
        assert quoter.counts["attempts"] >= 5
        orders = result.orders.set_index("decided_ns")
        for decision_s, _, posted, _, _ in quoter.log:
            decision = round(decision_s * 1e9)
            assert posted > decision - 1_000  # seen only after its stamp
            assert orders.loc[posted, "live_ns"] >= decision + 10 * MS - 1_000


def test_an_order_whose_cancel_is_in_flight_counts_as_resting() -> None:
    """Hold 0.3 s: the entry posted at 100 ms is cancelled at 400 ms and the
    cancel takes 250 ms. The decision seen at 600 ms finds it still out and is
    skipped; the one seen at 800 ms is posted."""
    market = _market(20)
    quoter = SignalEntryQuoter(0.3)
    config = CONFIG.with_(cancel_latency_ns=250 * MS)
    result = simulate_day(market, quoter, config, _tapes(market, [0, 500, 700], [1, 1, -1]))
    orders = result.orders
    start = market.day_start_ns
    assert (orders["decided_ns"] - start).tolist() == [100 * MS, 800 * MS]
    assert orders["cancel_ns"].iloc[0] - start == 650 * MS
    assert orders["side"].tolist() == [1, -1]
    assert quoter.counts["skipped_busy"] == 1
    assert quoter.counts["entries_expired"] == 2


def test_a_fill_while_the_cancel_flies_is_exited_and_the_position_stays_one_clip() -> None:
    market = _market(20, [(500.0, 99, 5.0, -1), (520.0, 99, 5.0, -1)])
    quoter = SignalEntryQuoter(0.3)
    config = CONFIG.with_(cancel_latency_ns=250 * MS)
    result = simulate_day(market, quoter, config, _tapes(market, [0, 500], [1, 1]))
    assert result.counters["max_abs_position"] == 1.0
    assert quoter.counts["skipped_busy"] == 1
    fills = result.fills
    assert fills["path"].tolist() == ["through", "flatten"]
    # Opened at 500 ms; the exit is decided at 800 ms and walks the 900 ms snapshot.
    assert fills["ts"].iloc[1] - market.day_start_ns == 900 * MS


def test_the_entry_is_kept_at_its_first_price() -> None:
    market = _market(20, bids=[100] * 3 + [101] * 3 + [102] * 14)
    quoter = SignalEntryQuoter(1.5)
    result = simulate_day(market, quoter, CONFIG, _tapes(market, [0], [1]))
    assert result.orders["price"].tolist() == [100]
    assert result.counters.get("replaced", 0.0) == 0.0


def test_a_filled_entry_is_crossed_out_by_a_taker_hold_after_the_fill() -> None:
    market = _market(20, [(150.0, 99, 5.0, -1)], bids=[100] * 8 + [103] * 12)
    quoter = SignalEntryQuoter(0.5)
    result = simulate_day(market, quoter, CONFIG, _tapes(market, [0], [1]))
    fills = result.fills
    start = market.day_start_ns
    assert (fills["ts"] - start).tolist() == [150 * MS, 800 * MS]
    assert fills["price"].tolist() == pytest.approx([1.00, 1.03])  # the walk of the 800 ms bids
    assert fills["maker"].tolist() == [True, False]
    table = quoter.attempts(fills)
    assert table["filled"].tolist() == [True]
    assert table["gross"].iloc[0] == pytest.approx(0.03)
    assert table["net"].iloc[0] == pytest.approx(0.03 - fills["fee"].sum())
    assert quoter.counts["round_trips"] == 1


def test_a_partial_fill_keeps_the_rest_at_its_price_and_exits_what_was_filled() -> None:
    prints = [(150.0, 99, 0.4, -1), (250.0, 99, 0.3, -1)]
    market = _market(20, prints, lot=0.1)
    quoter = SignalEntryQuoter(1.0)
    result = simulate_day(market, quoter, CONFIG, _tapes(market, [0], [1]))
    assert len(result.orders) == 1
    fills = result.fills
    assert fills["size"].tolist() == pytest.approx([0.4, 0.3, 0.7])
    assert fills["ts"].iloc[2] - market.day_start_ns == 1_300 * MS  # opened 150 ms + 1 s
    assert result.counters["max_abs_position"] == pytest.approx(0.7)


def test_an_entry_pulled_by_a_feed_pause_is_not_reposted() -> None:
    market = _market(10, gap_after=1)
    quoter = SignalEntryQuoter(30.0)
    result = simulate_day(market, quoter, CONFIG, _tapes(market, [0], [1]))
    assert result.counters["feed_pauses"] == 1
    assert len(result.orders) == 1
    assert result.orders["outcome"].tolist() == ["cancelled"]
    assert quoter.counts["entries_unfilled"] == 1


def test_a_regime_flag_cancels_a_resting_entry_at_the_first_event_after_it() -> None:
    market = _market(20)
    flag_s = (market.day_start_ns + 350 * MS) / 1e9
    tapes = _tapes(market, [0], [1])
    tapes[FLAG_SIGNAL] = SignalTape(
        FLAG_SIGNAL, np.array([round(flag_s * 1e9)]), np.array([flag_s])
    )
    guarded = SignalEntryQuoter(1.5)
    config = CONFIG.with_(cancel_entries_on_flag=True)
    result = simulate_day(market, guarded, config, tapes)
    assert result.orders["cancel_ns"].iloc[0] - market.day_start_ns == 410 * MS
    assert result.counters["entries_cancelled_by_flag"] == 1
    assert guarded.counts["entries_unfilled"] == 1 and len(result.orders) == 1
    unguarded = simulate_day(market, SignalEntryQuoter(1.5), CONFIG, tapes)
    assert unguarded.orders["cancel_ns"].iloc[0] - market.day_start_ns == 1_610 * MS


def test_the_passive_exit_is_posted_at_the_touch_and_crossed_after_a_further_hold() -> None:
    market = _market(20, [(150.0, 99, 5.0, -1)])
    quoter = SignalEntryQuoter(0.5, passive_exit=True)
    result = simulate_day(market, quoter, CONFIG, _tapes(market, [0], [1]))
    orders = result.orders
    assert orders["side"].tolist() == [1, -1]
    assert orders["price"].tolist() == [100, 102]
    assert orders["decided_ns"].iloc[1] - market.day_start_ns == 700 * MS
    fills = result.fills
    assert fills["path"].tolist() == ["through", "flatten"]
    assert fills["ts"].iloc[1] - market.day_start_ns == 1_400 * MS
    assert quoter.counts["passive_exits_posted"] == 1


def test_a_decision_seen_after_its_hold_is_not_posted_and_bad_tapes_are_refused() -> None:
    market = _market(10, gap_after=1)
    quoter = SignalEntryQuoter(1.0)
    result = simulate_day(market, quoter, CONFIG, _tapes(market, [150], [1]))
    assert result.orders.empty
    assert quoter.counts["expired_unseen"] == 1
    with pytest.raises(ValueError, match="side"):
        entry_tapes(np.array([1, 2]), np.array([1, 0]))
    with pytest.raises(ValueError, match="hold"):
        SignalEntryQuoter(0.0)


def test_an_exit_walked_where_the_next_attempt_posts_stays_with_its_own_attempt() -> None:
    """The exit is decided at 700 ms and walks the 800 ms snapshot, where the
    decision of 750 ms is posted: the exit's fill shares the new post's stamp
    and belongs to the first attempt, which is flat again only after it."""
    market = _market(20, [(150.0, 99, 5.0, -1)], bids=[100] * 8 + [103] * 12)
    quoter = SignalEntryQuoter(0.5)
    result = simulate_day(market, quoter, CONFIG, _tapes(market, [0, 750], [1, 1]))
    fills = result.fills
    start = market.day_start_ns
    assert (fills["ts"] - start).tolist() == [150 * MS, 800 * MS]
    assert [posted - start for _, _, posted, _, _ in quoter.log] == [100 * MS, 800 * MS]
    table = quoter.attempts(fills)
    assert table["filled"].tolist() == [True, False]
    assert table["gross"].tolist() == pytest.approx([0.03, 0.0])
    assert table["net"].tolist() == pytest.approx([0.03 - fills["fee"].sum(), 0.0])
    d = result.decomposition
    assert table["net"].sum() == pytest.approx(d["gross"] - d["fees"])
    assert quoter.counts["round_trips"] == 1 and quoter.counts["attempts"] == 2


def test_an_exit_due_between_the_stop_and_the_flatten_is_crossed_when_due() -> None:
    """Filled at 23:57:20 with a 50 s hold: quoting stops at 23:58, but the exit
    due at 23:58:10 is crossed then, not held to the 23:59 flatten; flat, the
    quoter is not asked after the stop, so a decision then posts nothing."""
    late = (23 * 3600 + 57 * 60) * 1_000.0  # 23:57:00 in ms
    stamps = [late + 1_000.0 * i for i in range(150)]  # a snapshot a second to 23:59:29
    snapshots = [(t, *_book(100, 102)) for t in stamps]
    market = hand_built_market(snapshots, [(late + 20_500.0, 99, 5.0, -1)], lot=1.0)
    quoter = SignalEntryQuoter(50.0)
    tapes = _tapes(market, [late + 10_000.0, late + 75_000.0], [1, 1])
    result = simulate_day(market, quoter, CONFIG, tapes)
    fills = result.fills
    moments = ((fills["ts"] - market.day_start_ns) / MS - late).tolist()
    assert fills["maker"].tolist() == [True, False]
    assert moments == [
        20_500.0,
        72_000.0,
    ]  # 23:57:20.5 filled; exit asked at 23:58:11, walked at :12
    assert result.counters["flattens_strategy"] == 1 and "flattens_day_end" not in result.counters
    assert result.counters["decisions_after_stop"] > 0
    assert len(result.orders) == 1 and quoter.counts["attempts"] == 1
