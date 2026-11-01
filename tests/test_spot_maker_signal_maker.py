"""MK-S's executor: entries at the proxy, trade-through fills, the exit, flips and the flatten."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from trading_research.spot.fills import PrintTape
from trading_research.spot_maker import signal_maker as sm
from trading_research.spot_maker.executor import settings_for
from trading_research.spot_maker.touch import Proxies

DAY = date(2026, 5, 4)
T0 = int(np.datetime64("2026-05-04T01:00:00", "ns").astype(np.int64))
END = int(np.datetime64("2026-05-04T23:59:30", "ns").astype(np.int64))
S = 1_000_000_000

# (seconds after 01:00, price, size, aggressor)
PRINTS = [(0, 100.02, 0.01, 1), (1, 100.00, 0.01, -1), (5, 99.99, 0.004, -1),
          (6, 99.98, 0.010, -1), (56, 100.01, 0.01, 1), (57, 100.03, 0.02, 1)]  # fmt: skip


def _tape(rows: list[tuple[float, float, float, int]], tail: bool = True) -> PrintTape:
    ts = [T0 + round(r[0] * S) for r in rows] + ([END] if tail else [])
    price = [r[1] for r in rows] + ([100.0] if tail else [])
    size = [r[2] for r in rows] + ([0.05] if tail else [])
    side = [r[3] for r in rows] + ([-1] if tail else [])
    return PrintTape(DAY, np.array(ts), np.array(price), np.array(size), np.array(side))


def _run(tape: PrintTape, decisions: list[tuple[float, int]], **changes: object) -> sm.DayRun:
    grid = np.unique(tape.ts)
    ns = np.array([T0 + round(t * S) for t, _ in decisions], dtype=np.int64)
    side = np.array([s for _, s in decisions], dtype=np.int64)
    return sm.run_signal_maker(tape, Proxies.from_tape(tape), grid, ns, side,
                               settings_for("MK-S", **changes))  # fmt: skip


def test_a_long_fills_by_trade_through_and_exits_at_the_ask_proxy() -> None:
    run = _run(_tape(PRINTS), [(2, 1)])
    rows = run.log.rows
    assert [(r[1], r[2], round(r[3], 6), r[5]) for r in rows] == [
        (1, 100.00, 0.004, "entry"),
        (1, 100.00, 0.006, "entry"),
        (-1, 100.01, 0.01, "exit"),
    ]
    assert run.counters["entries_filled"] == 1 and run.holds_s == [pytest.approx(52.0)]


def test_touch_fills_earlier_and_delayed_later() -> None:
    touch = _run(_tape(PRINTS), [(2, 1)], rule="touch")
    assert touch.log.rows[0][0] == T0 + 5 * S  # the print at 1 s precedes the order
    delayed = _run(_tape(PRINTS), [(2, 1)], rule="delayed")
    assert [round(r[3], 6) for r in delayed.log.rows if r[5] == "entry"] == [0.01]


def test_unfilled_entry_cancelled_and_flip_reenters() -> None:
    rows = [(0, 100.02, 0.01, 1), (1, 100.00, 0.01, -1), (30, 100.05, 0.01, 1),
            (31, 99.99, 0.01, -1), (70, 99.90, 0.02, -1), (71, 100.10, 0.02, 1)]  # fmt: skip
    run = _run(_tape(rows), [(2, -1), (10, 1)])  # the short is cancelled by the long at 10 s
    kinds = [(r[1], r[5]) for r in run.log.rows]
    assert run.counters["entries_unfilled"] == 1 and kinds[0] == (1, "entry")
    assert run.counters["entries_posted"] == 2


def test_an_exit_that_never_fills_is_flattened_at_ten_bp_side() -> None:
    rows = [(0, 100.02, 0.01, 1), (1, 100.00, 0.01, -1), (5, 99.90, 0.02, -1)]
    run = _run(_tape(rows), [(2, 1)])
    last = run.log.rows[-1]
    assert last[5] == "flatten" and last[1] == -1 and last[4] is False
    assert run.counters["exits_flattened"] == 1 and run.counters["flattens"] == 1
    assert sm.next_opposite(np.array([1, 1, -1, 1])).tolist() == [2, 2, 3, -1]


def test_the_exit_skews_by_the_position_known_at_its_decision() -> None:
    # the long fills 0.004 at 5 s; the short at 20 s cancels it and starts the exit, and
    # 0.006 more fill at 20.005 s inside the cancel's latency: the exit is sized 0.010
    # at its arrival but skewed by the 0.004 known at 20 s (a 1 bp a clip cell)
    rows = [(0, 100.02, 0.01, 1), (1, 100.00, 0.01, -1), (5, 99.99, 0.004, -1),
            (20.005, 99.98, 0.01, -1), (30, 100.03, 0.02, 1)]  # fmt: skip
    run = _run(_tape(rows), [(2, 1), (20, -1)], skew_bp=1.0)
    exits = [r for r in run.log.rows if r[5] == "exit"]
    assert sum(r[3] for r in run.log.rows if r[5] == "entry") == pytest.approx(0.01)
    assert exits[0][2] == pytest.approx(100.02) and exits[0][3] == pytest.approx(0.01)
    known_all = _run(_tape([*rows[:3], (19.0, 99.98, 0.01, -1), rows[4]]), [(2, 1), (20, -1)],
                     skew_bp=1.0)  # fmt: skip
    assert next(r[2] for r in known_all.log.rows if r[5] == "exit") == pytest.approx(100.01)


def test_an_audited_run_notes_every_resting_order() -> None:
    run = _run(_tape(PRINTS), [(2, 1)], audit=True)
    assert run.orders is not None and len(run.orders) == 2
    side, price, size, start, end, stamps, sizes = run.orders[0]
    assert (side, price, size) == (1, 100.00, 0.01) and start == T0 + 2 * S + 10_000_000
    assert sizes.sum() == pytest.approx(0.01) and ((stamps >= start) & (stamps < end)).all()
    assert _run(_tape(PRINTS), [(2, 1)]).orders is None
