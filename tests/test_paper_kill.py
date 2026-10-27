"""The kill switch: max loss, max position, stale data and a sequence gap each latch it,
nothing new is sent afterwards, and what is held is closed by a taker order."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests._paper_events import T0, in_time, row, rows_every, schedule, session, trade
from trading_research.live.events import Gap
from trading_research.live.ledger import Limits, PaperLedger


def test_max_loss_latches_on_a_mark_and_the_position_is_flattened(tmp_path: Path) -> None:
    paper = session(tmp_path, mode="taker", clip=1.0, hold_s=60.0, max_position=2.0, max_loss=5.0)
    events = rows_every(0, 1_000)
    events += [trade(20, 100.1, 2.0, +1)]  # long one unit at 100.1
    events += rows_every(1_000, 2_000, bid=93.9, ask=94.1)  # marked at 94: -6.1 - fees
    events += [trade(1_150, 94.0, 5.0, -1)]
    summary = paper.run(in_time(events))
    pnl = summary["pnl"]
    assert pnl["killed"] == "max_loss"
    assert pnl["killed_ns"] == T0 + 1_000 * 1_000_000
    fills = paper.ledger.fills_frame()
    assert fills["reason"].tolist() == ["entry", "flatten:max_loss"]
    assert fills["price"].iloc[-1] == 94.0
    assert pnl["position"] == 0.0
    # The session stops once flat: nothing after the flatten fill is processed.
    assert summary["rows"] < 20


def test_an_order_beyond_the_position_limit_is_refused(tmp_path: Path) -> None:
    ledger = PaperLedger(schedule(), Limits(max_position=0.015, max_loss=1e6))
    assert ledger.allows(+1, 0.01)
    ledger.fill(1, +1, 100.0, 0.01, maker=False, oid=1, reason="entry", path="print")
    assert not ledger.allows(+1, 0.01)
    assert ledger.allows(-1, 0.02)
    assert ledger.refused == 1
    ledger.fill(2, +1, 100.0, 0.01, maker=False, oid=2, reason="entry", path="print")
    assert ledger.killed == "max_position"
    assert not ledger.allows(-1, 0.01)  # once latched, nothing is sent


def test_stale_data_latches_and_flattens(tmp_path: Path) -> None:
    paper = session(tmp_path, mode="taker", clip=0.01, hold_s=60.0, stale_after_s=5.0)
    events = [*rows_every(0, 500), trade(20, 100.1, 1.0, +1)]
    events += rows_every(10_000, 11_000)  # ten seconds without a row
    events += [trade(10_050, 99.9, 1.0, -1)]
    summary = paper.run(in_time(events))
    assert summary["pnl"]["killed"] == "stale_data"
    assert summary["counts"]["stale_pauses"] == 1
    assert paper.ledger.fills_frame()["reason"].tolist() == ["entry", "flatten:stale_data"]
    assert summary["pnl"]["position"] == 0.0


def test_a_sequence_gap_latches_and_no_order_follows(tmp_path: Path) -> None:
    paper = session(tmp_path, mode="taker", clip=0.01, every=1)
    gap = Gap(T0 + 50_000_000, T0 + 50_000_000, "sequence", "book update 9 after 7")
    events = [row(0), gap, *rows_every(100, 1_000), trade(20, 100.1, 1.0, +1)]
    summary = paper.run(in_time(events))
    assert summary["pnl"]["killed"] == "sequence_gap"
    outcomes = paper.decisions_frame()["outcome"].tolist()
    assert outcomes[0] == "entered" and set(outcomes[1:]) <= {"killed"}
    assert summary["gap_kinds"] == {"sequence": 1}


def test_without_a_print_the_flatten_is_priced_at_the_last_print_of_its_side(
    tmp_path: Path,
) -> None:
    paper = session(tmp_path, mode="taker", clip=0.01, hold_s=60.0, max_loss=0.01,
                    flatten_wait_s=1.0)  # fmt: skip
    events = [*rows_every(0, 200), trade(20, 100.1, 1.0, +1), trade(30, 99.5, 1.0, -1)]
    events += rows_every(200, 2_000, bid=97.9, ask=98.1)
    summary = paper.run(in_time(events))
    fills = paper.ledger.fills_frame()
    assert summary["pnl"]["killed"] == "max_loss"
    assert fills["path"].iloc[-1] == "no_print_last_same_side"
    assert fills["price"].iloc[-1] == pytest.approx(99.5)
    assert summary["execution_counts"]["exits_no_print"] == 1
