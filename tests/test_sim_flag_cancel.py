"""The simulator's regime guard on resting entries: ``regime.cancel_times`` in event time.

A resting entry is cancelled at the first event (a snapshot, a print or a
funding settlement) after a flag at or after its decision; an order decided
after the flag, and an exit, are left alone.
"""

from __future__ import annotations

import numpy as np

from trading_research.market_making.events import DayEvents
from trading_research.market_making.quoters import FLAG_SIGNAL
from trading_research.market_making.signal_entry import SignalEntryQuoter, entry_tapes
from trading_research.market_making.signals import SignalTape
from trading_research.market_making.simulator import SimConfig, simulate_day
from trading_research.market_making.synthetic import hand_built_market
from trading_research.pipeline.regime import RegimePolicy, cancel_times

MS = 1_000_000
PLAIN = SimConfig(clip_btc=1.0, warmup_s=0.0, soft_limit_clips=1.0, flatten_slippage_bp=0.0)
GUARDED = PLAIN.with_(cancel_entries_on_flag=True)


def market(rows: int = 20, prints: tuple = ()) -> DayEvents:
    """Snapshots every 100 ms at bid 100 / ask 102; ``prints`` as (ms, price, size, side)."""
    book = ([(100 - i, 10.0) for i in range(3)], [(102 + i, 10.0) for i in range(3)])
    snapshots = [(100.0 * i, *book) for i in range(rows)]
    return hand_built_market(snapshots, list(prints) or [(100.0 * rows - 101, 120, 1.0, 1)])


def tapes(events: DayEvents, flags_ms: list[float]) -> dict[str, SignalTape]:
    start = events.day_start_ns
    out = entry_tapes(np.array([start], dtype=np.int64), np.array([1]))
    flags = np.array([start + round(f * MS) for f in flags_ms], dtype=np.int64)
    out[FLAG_SIGNAL] = SignalTape(FLAG_SIGNAL, flags, flags / 1e9)
    return out


def test_the_first_event_after_the_flag_may_be_a_print() -> None:
    events = market(prints=((370.0, 120, 1.0, 1), (1_899.0, 120, 1.0, 1)))
    result = simulate_day(events, SignalEntryQuoter(1.5), GUARDED, tapes(events, [350.0]))
    assert result.orders["cancel_ns"].iloc[0] - events.day_start_ns == 380 * MS
    assert result.counters["entries_cancelled_by_flag"] == 1
    assert len(result.orders) == 1


def test_the_cancel_is_the_one_cancel_times_gives() -> None:
    events = market()
    start = events.day_start_ns
    flags = [350.0, 1_700.0]
    plain = simulate_day(events, SignalEntryQuoter(1.5), PLAIN, tapes(events, flags)).orders
    assert plain["cancel_ns"].iloc[0] - start == 1_610 * MS  # the flag tape alone does nothing
    flag_ns = np.array([start + round(f * MS) for f in flags], dtype=np.int64)
    predicted = cancel_times(
        plain["decided_ns"].to_numpy(),
        plain["done_ns"].to_numpy(),
        flag_ns,
        RegimePolicy.parse("pull_4h"),
    )
    assert predicted.tolist() == [flag_ns[0]]
    guarded = simulate_day(events, SignalEntryQuoter(1.5), GUARDED, tapes(events, flags)).orders
    first_event_after = 400 * MS
    assert guarded["cancel_ns"].iloc[0] - start == first_event_after + GUARDED.cancel_latency_ns
    assert guarded["decided_ns"].iloc[0] <= predicted[0] < guarded["cancel_ns"].iloc[0]


def test_an_order_decided_after_the_flag_is_not_cancelled_by_it() -> None:
    events = market()
    result = simulate_day(events, SignalEntryQuoter(1.5), GUARDED, tapes(events, [50.0]))
    assert result.counters.get("entries_cancelled_by_flag", 0.0) == 0.0
    assert result.orders["cancel_ns"].iloc[0] - events.day_start_ns == 1_610 * MS
    policy = RegimePolicy.parse("pull_24h")
    orders = result.orders
    flag = np.array([events.day_start_ns + 50 * MS])
    late = cancel_times(orders["decided_ns"].to_numpy(), orders["done_ns"].to_numpy(), flag, policy)
    assert late.tolist() == [-1]


def test_an_exit_resting_at_the_touch_is_left_alone() -> None:
    events = market(prints=((150.0, 99, 5.0, -1), (1_899.0, 120, 1.0, 1)))
    flags = [800.0]
    plain = simulate_day(
        events, SignalEntryQuoter(0.5, passive_exit=True), PLAIN, tapes(events, flags)
    )
    guarded = simulate_day(
        events, SignalEntryQuoter(0.5, passive_exit=True), GUARDED, tapes(events, flags)
    )
    assert guarded.orders["side"].tolist() == [1, -1]
    assert guarded.counters.get("entries_cancelled_by_flag", 0.0) == 0.0
    assert guarded.orders.equals(plain.orders)
    assert guarded.fills["path"].tolist() == plain.fills["path"].tolist()
