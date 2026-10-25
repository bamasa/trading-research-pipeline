"""The side-correct print fill rule and its brackets, against hand cases and a slow loop."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from test_spot_synthetic_prints import DAY, no_network
from trading_research.spot.fills import (
    BRACKETS,
    BUY,
    SELL,
    PrintFillSettings,
    PrintTape,
    fill_orders,
)

S = 1_000_000_000
MS = 1_000_000
T0 = pd.Timestamp(DAY, tz="UTC").value + 3600 * S


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    no_network(monkeypatch)


def tape_of(rows: list[tuple[int, float, float, int, bool]], day: date = DAY) -> PrintTape:
    """A tape from ``(offset_ns, price, size, side, rpi)`` rows, offsets from T0."""
    frame = pd.DataFrame(rows, columns=["offset", "price", "size", "aggressor", "rpi"])
    frame.insert(0, "timestamp", pd.to_datetime(T0 + frame.pop("offset"), unit="ns", utc=True))
    return PrintTape.from_frame(frame, day)


def one(tape: PrintTape, side: int, arrival: int, decision: int | None = None, **kw: object):
    until = kw.pop("until_ns", tape.end_ns)
    fills = fill_orders(
        tape,
        side,
        np.array([T0 + arrival]),
        np.array([T0 + (arrival - 10 * MS if decision is None else decision)]),
        until_ns=until,  # type: ignore[arg-type]
        **kw,  # type: ignore[arg-type]
    )
    return {name: value[0] for name, value in fills._asdict().items()}


def test_a_buy_fills_at_the_first_buyer_print_at_or_after_arrival() -> None:
    tape = tape_of(
        [
            (0, 100.0, 1.0, BUY, False),
            (5 * MS, 99.0, 1.0, SELL, False),  # wrong side
            (7 * MS, 98.0, 1.0, BUY, True),  # unreachable
            (9 * MS, 101.0, 1.0, BUY, False),
            (20 * MS, 102.0, 1.0, BUY, False),
        ]
    )
    fill = one(tape, BUY, 6 * MS)
    assert fill["found"] and fill["print_index"] == 3 and fill["rule"] == 101.0
    assert fill["wait_ns"] == 3 * MS and not fill["late"]
    # A print stamped exactly at the arrival is reachable.
    assert one(tape, BUY, 9 * MS)["print_index"] == 3
    # A sell takes the seller-initiated print.
    assert one(tape, SELL, 1 * MS)["rule"] == 99.0


def test_the_clip_sweeps_same_stamp_prints_in_archive_order() -> None:
    tape = tape_of(
        [
            (0, 100.0, 0.004, BUY, False),
            (0, 99.0, 0.5, SELL, False),
            (0, 100.5, 0.003, BUY, False),
            (0, 100.2, 0.002, BUY, True),  # flagged: not walked
            (1 * S, 103.0, 1.0, BUY, False),
            (30 * S, 104.0, 1.0, BUY, False),
            (61 * S, 110.0, 1.0, BUY, False),  # outside the 60 s window
        ]
    )
    fill = one(tape, BUY, 0)
    walked = 0.004 * 100.0 + 0.003 * 100.5
    assert fill["sweep"] == pytest.approx(0.007)
    assert fill["beyond"] == pytest.approx(0.003)
    assert fill["rule"] == pytest.approx((walked + 0.003 * 100.5) / 0.010)
    assert fill["optimistic"] == 100.0
    # Pessimistic depth: the beyond part at the worst buyer print of (s, s + 60 s].
    assert fill["pessimistic"] == pytest.approx((walked + 0.003 * 104.0) / 0.010)
    covered = one(tape, BUY, 0, settings=PrintFillSettings(clip=0.005))
    assert covered["beyond"] == 0.0
    assert covered["rule"] == pytest.approx((0.004 * 100.0 + 0.001 * 100.5) / 0.005)
    assert covered["pessimistic"] == covered["rule"]


def test_pessimistic_depth_without_a_later_print_uses_the_sweeps_last_price() -> None:
    flatten = (23 * 3600 + 59 * 60) * S - 3600 * S  # 23:59 from T0
    tape = tape_of(
        [
            (0, 100.0, 0.001, SELL, False),
            (0, 99.5, 0.001, SELL, False),
            (flatten + S, 50.0, 1.0, SELL, False),  # after 23:59: outside the window
        ]
    )
    fill = one(tape, SELL, 0, settings=PrintFillSettings(depth_window_ns=86_400 * S))
    walked = 0.001 * 100.0 + 0.001 * 99.5
    assert fill["pessimistic"] == pytest.approx((walked + 0.008 * 99.5) / 0.010)
    assert fill["pessimistic"] == fill["rule"]


def test_stale_and_worse_of_two() -> None:
    tape = tape_of(
        [
            (0, 100.0, 1.0, BUY, False),
            (1 * MS, 101.0, 1.0, BUY, True),  # flagged: never a stale price either
            (2 * S, 102.0, 1.0, BUY, False),
            (3 * S, 98.0, 1.0, SELL, False),
            (4 * S, 99.0, 1.0, SELL, False),
        ]
    )
    buy = one(tape, BUY, 1 * S, decision=1 * S - 10 * MS)
    assert buy["stale"] == 100.0 and buy["rule"] == 102.0 and buy["worse_of_two"] == 102.0
    sell = one(tape, SELL, 4 * S, decision=3 * S)
    assert sell["stale"] == 98.0 and sell["rule"] == 99.0 and sell["worse_of_two"] == 98.0
    # No print of the side at or before the decision: the stale bracket takes the
    # rule's price, and worse of two the pessimistic one.
    early = one(tape, SELL, 0, decision=0)
    assert early["stale"] == early["rule"] and early["worse_of_two"] == early["pessimistic"] == 98.0


def test_bounds_fallback_and_late_prints() -> None:
    tape = tape_of(
        [
            (0, 100.0, 1.0, BUY, False),
            (2 * S, 99.0, 1.0, SELL, False),
            (5 * S, 101.0, 1.0, BUY, False),
        ]
    )
    late = one(tape, BUY, 1 * S)
    assert late["late"] and late["wait_ns"] == 4 * S
    missed = one(tape, BUY, 1 * S, until_ns=T0 + 5 * S)
    assert not missed["found"] and np.isnan(missed["rule"]) and missed["print_index"] == -1
    fallback = one(tape, SELL, 3 * S, fallback=True)
    assert not fallback["found"] and fallback["fallback"]
    assert fallback["rule"] == 99.0 and fallback["age_ns"] == 1 * S
    # The exit never fills on a print stamped before not_before.
    held = one(tape, BUY, 0, not_before_ns=np.array([T0 + 1 * S]))
    assert held["print_index"] == 2 and held["wait_ns"] == 5 * S
    # A side with no print before the arrival falls back on either side's last.
    only_buys = tape_of([(0, 100.0, 1.0, BUY, False)])
    either = one(only_buys, SELL, 1 * S, fallback=True)
    assert either["fallback"] and either["rule"] == 100.0


def test_the_tape_refuses_what_the_contract_forbids() -> None:
    with pytest.raises(ValueError, match="stamped on"):
        tape_of([(-3600 * S - 1, 100.0, 1.0, BUY, False)])
    with pytest.raises(ValueError, match="non-decreasing"):
        tape_of([(1, 100.0, 1.0, BUY, False), (0, 100.0, 1.0, BUY, False)])
    with pytest.raises(ValueError, match="aggressor"):
        tape_of([(0, 100.0, 1.0, 0, False)])
    with pytest.raises(ValueError, match="positive"):
        tape_of([(0, 100.0, 0.0, BUY, False)])
    with pytest.raises(ValueError, match="bracket"):
        fill_orders(tape_of([(0, 1.0, 1.0, BUY, False)]), BUY, np.array([T0]), np.array([T0]),
                    until_ns=T0 + S).price("best")  # fmt: skip
    assert BRACKETS == ("rule", "optimistic", "pessimistic", "stale", "worse_of_two")
