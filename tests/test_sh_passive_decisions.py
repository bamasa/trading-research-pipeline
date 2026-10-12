"""A passive cell's decisions: the entry hours, and one entry per EV-gate event for the quoter."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from trading_research.market_making.signal_entry import SignalEntryQuoter, entry_tapes
from trading_research.market_making.simulator import SimConfig, simulate_day
from trading_research.market_making.synthetic import hand_built_market
from trading_research.pipeline.short_horizon.passive import (
    Decisions,
    entry_hours,
    first_of_events,
    tape_decisions,
)
from trading_research.search_engine.catalogue import parse_key
from trading_research.search_engine.tapes import ev_gate_tape

DAY = date(2025, 3, 4)
R_CONFIG = "BTCUSDT|R|bound_ridge|micro|hold_s=5"
R_KEY = "BTCUSDT|R|bound_ridge|micro|open|hold_s=5|k=0|beta_bp=0|regime=none"
MS = 1_000_000
S = 1_000_000_000


def test_decisions_keep_the_entry_hours_and_every_row_of_an_event_with_its_id() -> None:
    """Rows at 00:00:30, 00:01, 12:00 (three open rows, one event), 23:58 and 23:58:30:
    the first and the last are outside [00:01, 23:58]; the noon event passes all
    three rows with one id."""
    midnight = pd.Timestamp(DAY, tz="UTC").value
    seconds = [30, 60, 43_200, 43_201, 43_202, 43_203, 86_280, 86_310]
    stamps = midnight + np.array(seconds, dtype=np.int64) * S
    mu = np.array([40.0, 40.0, 40.0, 40.0, 40.0, 0.0, -40.0, 40.0])
    n = len(stamps)
    gates = {"quiet_out": np.ones(n, bool), "tight_only": np.ones(n, bool)}
    tape = ev_gate_tape(DAY, R_CONFIG, stamps, mu, np.ones(n), np.zeros(n), gates, np.zeros(0))
    found = tape_decisions(tape, parse_key(R_KEY), None, 0.0, DAY)
    assert entry_hours(DAY) == (midnight + 60 * S, midnight + 86_280 * S)
    assert ((found.stamps_ns - midnight) // S).tolist() == [60, 43_200, 43_201, 43_202, 86_280]
    assert found.side.tolist() == [1, 1, 1, 1, -1]
    assert found.event is not None
    events = found.event.tolist()
    assert events[0] == events[1] == events[2] == events[3] and len(set(events)) == 2
    first = first_of_events(found)
    assert ((first.stamps_ns - midnight) // S).tolist() == [60, 86_280]


def _market():
    """A snapshot every 100 ms for 3 s, bid 100 / ask 102; a sale at 99 at 150 ms
    fills a resting bid, and a sale at 99 at 1,650 ms fills the next one."""
    book = ([(100 - i, 10.0) for i in range(3)], [(102 + i, 10.0) for i in range(3)])
    snapshots = [(100.0 * i, *book) for i in range(30)]
    return hand_built_market(snapshots, [(150.0, 99, 5.0, -1), (1_650.0, 99, 5.0, -1)], lot=1.0)


def test_an_event_that_opens_while_busy_enters_at_its_first_free_row_once() -> None:
    """Event 1 enters at 0 ms and holds to 650 ms (a 0.5 s hold after the 150 ms
    fill). Event 2 opens at 300 ms, busy: it enters at its first row after the exit
    (700 ms) and never again; event 2's later rows give no other entry."""
    market = _market()
    config = SimConfig(clip_btc=1.0, warmup_s=0.0, soft_limit_clips=1.0, flatten_slippage_bp=0.0)
    times = [0.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 1_500.0, 2_000.0]
    events = [1, 2, 2, 2, 2, 2, 2, 2, 2]
    stamps = np.array([market.day_start_ns + round(t * MS) for t in times], dtype=np.int64)
    sides = np.ones(len(times))
    quoter = SignalEntryQuoter(0.5)
    result = simulate_day(market, quoter, config, entry_tapes(stamps, sides, np.array(events)))
    posted = ((result.orders["decided_ns"] - market.day_start_ns) / MS).tolist()
    assert posted == [100.0, 800.0]  # each decision is first seen at the next snapshot
    assert quoter.counts["attempts"] == 2 and quoter.counts["skipped_busy"] == 1
    plain = SignalEntryQuoter(0.5)
    simulate_day(market, plain, config, entry_tapes(stamps, sides))
    assert plain.counts["attempts"] > 2  # without the event tape every free row enters


def test_placebo_decisions_carry_the_events() -> None:
    from trading_research.pipeline.short_horizon.heldout_passive import variants
    from trading_research.pipeline.short_horizon.passive import SimCell

    own = Decisions(np.array([1, 2, 3]), np.array([1, 1, -1]), np.array([5.0, 6.0, -7.0]),
                    np.array([4, 4, 9]))  # fmt: skip
    rngs = [np.random.default_rng(0)]
    runs = variants("S-P", SimCell("S-P", "k", "entry", 5.0), own, own, 0, 0, [], 1, rngs)
    assert all(r.decisions.event is not None for r in runs)
    assert all(r.decisions.event.tolist() == [4, 4, 9] for r in runs)
