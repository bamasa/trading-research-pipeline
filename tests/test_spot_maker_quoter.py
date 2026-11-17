"""The quoters: requote instants, the signal's doubled side, the limit, the flatten."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from trading_research.spot.fills import PrintTape
from trading_research.spot_maker import quoter as qt
from trading_research.spot_maker.executor import settings_for
from trading_research.spot_maker.touch import Proxies

DAY = date(2026, 5, 4)
MID = int(np.datetime64("2026-05-04T00:00:00", "ns").astype(np.int64))
S = 1_000_000_000


def _tape(rows: list[tuple[float, float, float, int]]) -> PrintTape:
    rows = [*sorted(rows), (86370.0, 100.0, 0.05, -1), (86371.0, 100.02, 0.05, 1)]
    return PrintTape(DAY, np.array([MID + round(r[0] * S) for r in rows]),
                     np.array([r[1] for r in rows]), np.array([r[2] for r in rows]),
                     np.array([r[3] for r in rows]))  # fmt: skip


BASE = [(30.0, 100.02, 0.01, 1), (31.0, 100.00, 0.01, -1),
        (70.0, 99.99, 0.03, -1), (130.0, 100.03, 0.005, 1)]  # fmt: skip


def test_instants_on_the_clock_and_at_signals() -> None:
    tape = _tape(BASE)
    found = qt.instants(tape, settings_for("MK-Q"), None)
    assert len(found) == 1438 and found[0] == MID + 60 * S and found[-1] == MID + 86_280 * S
    assert qt.instants(tape, settings_for("MK-Q", requote_s=120.0), None)[0] == MID + 120 * S
    extra = qt.instants(tape, settings_for("MK-SQ"), np.array([MID + 75 * S]))
    assert MID + 75 * S in extra and MID + 125 * S in extra
    sides = qt.signal_sides(np.array([MID + 74 * S, MID + 80 * S, MID + 126 * S]),
                            np.array([MID + 75 * S]), np.array([1]), 50 * S)  # fmt: skip
    assert sides.tolist() == [0, 1, 0]


def test_mk_q_fills_the_bid_through_and_flattens() -> None:
    run = qt.run_quoter(_tape(BASE), Proxies.from_tape(_tape(BASE)), settings_for("MK-Q"))
    bids = [r for r in run.log.rows if r[5] == "bid"]
    asks = [r for r in run.log.rows if r[5] == "ask"]
    assert bids[0][:4] == (MID + 70 * S, 1, 100.00, 0.01)
    # long one clip at 120 s: no bid, the ask lowered 1 bp by the skew, rounded up
    assert asks[0][2] == pytest.approx(100.01) and asks[0][3] == pytest.approx(0.005)
    assert run.log.rows[-1][5] == "flatten" and run.log.rows[-1][1] == -1
    position = sum(r[1] * r[3] for r in run.log.rows)
    assert position == pytest.approx(0.0)


def test_mk_sq_doubles_the_signal_side_within_its_limit() -> None:
    tape = _tape(BASE)
    run = qt.run_quoter(tape, Proxies.from_tape(tape), settings_for("MK-SQ"),
                        np.array([MID + 65 * S]), np.array([1]))  # fmt: skip
    first = next(r for r in run.log.rows if r[5] == "bid")
    assert first[0] == MID + 70 * S and first[3] == pytest.approx(0.02)


def test_one_sided_modes() -> None:
    tape = _tape(BASE)
    long_only = qt.run_quoter(tape, Proxies.from_tape(tape), settings_for("MK-Q"), mode="bid_only")
    assert {r[5] for r in long_only.log.rows} <= {"bid", "flatten"}
    with pytest.raises(ValueError):
        qt.run_quoter(tape, Proxies.from_tape(tape), settings_for("MK-Q"), mode="lean")


def test_the_skew_reads_the_position_known_at_the_instant() -> None:
    # the bid of 60 s fills at 120.005 s, inside the latency after the instant at 120 s:
    # the ask of 120 s is priced unskewed (nothing known filled) but sized at its arrival
    rows = [(30.0, 100.02, 0.01, 1), (31.0, 100.00, 0.01, -1), (120.005, 99.99, 0.03, -1),
            (130.0, 100.03, 0.02, 1)]  # fmt: skip
    tape = _tape(rows)
    run = qt.run_quoter(tape, Proxies.from_tape(tape), settings_for("MK-Q"))
    bids = [r for r in run.log.rows if r[5] == "bid"]
    asks = [r for r in run.log.rows if r[5] == "ask"]
    assert bids[0][:4] == (MID + 120_005_000_000, 1, 100.00, 0.01)
    assert asks[0][:3] == (MID + 130 * S, -1, 100.02) and asks[0][3] == pytest.approx(0.01)
    assert len(bids) == 1 or bids[1][0] > MID + 130 * S  # no bid at 120 s: the limit was full
