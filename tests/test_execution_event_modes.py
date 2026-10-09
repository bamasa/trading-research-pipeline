"""The event-time taker and the signal-entry quoter as execution modes, in the attempts' shape."""

from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd
import pytest

from trading_research.backtest.taker import TakerBook, TakerSettings, execute_day
from trading_research.market_making.events import NS_PER_S, DayEvents
from trading_research.market_making.signal_entry import SignalEntryQuoter, entry_tapes
from trading_research.market_making.simulator import SimConfig, simulate_day
from trading_research.market_making.synthetic import random_market
from trading_research.pipeline import regime
from trading_research.pipeline.execution import (
    ATTEMPT_COLUMNS,
    EVENT_EXTRA_COLUMNS,
    VALID_PAIRS,
    ExecutionMode,
    SignalStream,
    check_pair,
    execute,
    guarded_rows,
    signal_entry_attempts,
)

SETTINGS = TakerSettings(clip=1.0, no_entry_before=time(0, 0))
CONFIG = SimConfig(clip_btc=1.0, warmup_s=0.0, soft_limit_clips=1.0, flatten_slippage_bp=0.0)


def day_and_stream(seed: int = 3, every: int = 40) -> tuple[DayEvents, SignalStream]:
    events = random_market(seed, n_snapshots=900, prints_per_snapshot=1.5)
    ts = np.asarray(events.book_ts[5::every], dtype=np.int64)
    side = np.where(np.arange(len(ts)) % 3 == 0, -1, 1).astype(np.int8)
    stream = SignalStream("BTCUSDT", ts, side, np.ones(len(ts)), 2 * NS_PER_S)
    return events, stream


def test_the_event_modes_pair_with_regime_policies_as_their_family() -> None:
    assert len(VALID_PAIRS) == 7
    assert ExecutionMode.EVENT_TAKER.family == "taker"
    assert ExecutionMode.SIGNAL_ENTRY.family == "passive_entry"
    assert ExecutionMode.MARKET_MAKER.family == "market_maker"
    assert check_pair("event_taker", "guard_pull") is ExecutionMode.EVENT_TAKER
    for mode in ("event_taker", "signal_entry"):
        with pytest.raises(ValueError, match="no spread of its own"):
            check_pair(mode, "guard_widen")
    names = [p.name for p in regime.policies_for("signal_entry")]
    assert names == [p.name for p in regime.policies_for("passive_entry")]
    assert regime.SEARCH_POLICIES == ("none", "pull_4h", "pull_24h", "pull_120h")


def test_the_event_taker_is_the_executor_in_the_attempts_shape() -> None:
    events, stream = day_and_stream()
    book = TakerBook.from_events(events)
    result = execute(stream, "event_taker", books=[book], taker_settings=SETTINGS)
    day = execute_day(book, stream.ts, stream.direction, stream.hold_ns, SETTINGS)
    attempts = result.attempts
    assert list(attempts.columns) == [*ATTEMPT_COLUMNS, *EVENT_EXTRA_COLUMNS]
    assert len(attempts) == len(day.trades) > 0
    assert attempts["ts"].tolist() == day.trades["decision_ns"].tolist()
    assert np.allclose(attempts["net_bp"], attempts["gross_bp"] - attempts["cost_bp"])
    assert attempts["filled"].all() and set(attempts["path"]) <= {"hold", "day_end"}
    assert result.mode is ExecutionMode.EVENT_TAKER and result.days is not None
    row = result.days.iloc[0]
    assert row["trades"] == len(day.trades) and row["net_usdt"] == pytest.approx(day.net)
    assert row["entries_busy"] == day.counters["entries_busy"]
    assert result.summary["attempts"] == len(day.trades)


def test_a_pull_guard_drops_the_taker_decisions_inside_its_window() -> None:
    events, stream = day_and_stream()
    book = TakerBook.from_events(events)
    flags = np.array([int(stream.ts[3]), int(stream.ts[10])], dtype=np.int64)
    window_min = 5 / 60  # five seconds
    guarded = execute(
        stream, "event_taker", books=[book], taker_settings=SETTINGS,
        regime_policy="guard_pull", flags_ns=flags, guard_window_min=window_min,
    )  # fmt: skip
    inside = guarded_rows(stream.ts, flags, 5 * NS_PER_S)
    assert inside.any()
    keep = ~inside
    day = execute_day(book, stream.ts[keep], stream.direction[keep], stream.hold_ns, SETTINGS)
    assert guarded.attempts["ts"].tolist() == day.trades["decision_ns"].tolist()
    with pytest.raises(ValueError, match="flag times"):
        execute(stream, "event_taker", books=[book], regime_policy="guard_pull")


def test_the_signal_entry_mode_is_the_quoter_in_the_attempts_shape() -> None:
    events, stream = day_and_stream()
    result = execute(stream, "signal_entry", events=[events], sim_config=CONFIG)
    quoter = SignalEntryQuoter(2.0)
    simulated = simulate_day(events, quoter, CONFIG, entry_tapes(stream.ts, stream.direction))
    table = quoter.attempts(simulated.fills)
    attempts = result.attempts
    assert len(attempts) == len(table) > 0
    assert set(attempts["ts"]) <= set(stream.ts.tolist())
    unfilled = attempts[~attempts["filled"]]
    assert (unfilled[["gross_bp", "cost_bp", "net_bp"]] == 0).all().all()
    assert set(unfilled["path"]) <= {"unfilled"}
    assert np.allclose(attempts["net_bp"], attempts["gross_bp"] - attempts["cost_bp"])
    assert result.days is not None and len(result.days) == 1
    assert result.days["quoter_attempts"].iloc[0] == quoter.counts["attempts"]
    expected = signal_entry_attempts("BTCUSDT", events.day, table, stream.ts)
    pd.testing.assert_frame_equal(attempts, expected)
    with pytest.raises(ValueError, match="simulator's settings"):
        execute(stream, "signal_entry", events=[events])


def test_a_pull_guard_on_the_quoter_also_cancels_resting_entries() -> None:
    events, stream = day_and_stream(every=60)
    flags = np.asarray(events.book_ts[[200, 500]], dtype=np.int64) + 1
    guarded = execute(
        stream, "signal_entry", events=[events], sim_config=CONFIG,
        regime_policy="guard_pull", flags_ns=flags, guard_window_min=1 / 60,
    )  # fmt: skip
    plain = execute(stream, "signal_entry", events=[events], sim_config=CONFIG)
    assert guarded.days is not None and plain.days is not None
    assert len(guarded.attempts) <= len(plain.attempts)
    assert guarded.days["entries_cancelled_by_flag"].iloc[0] == 1
    assert plain.days["entries_cancelled_by_flag"].iloc[0] == 0
    with pytest.raises(ValueError, match="give signals"):
        execute(None, "signal_entry", events=[events], sim_config=CONFIG)
