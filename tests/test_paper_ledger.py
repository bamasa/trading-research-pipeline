"""The paper account against fills and fees computed by hand, and the dated fee table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tests._paper_events import in_time, row, rows_every, schedule, session, trade
from trading_research.live.fees import FeeNotSourced, schedule_for
from trading_research.live.ledger import Limits, PaperLedger


def ns(stamp: str) -> int:
    return int(pd.Timestamp(stamp, tz="UTC").value)


def test_fills_fees_and_marks_by_hand() -> None:
    ledger = PaperLedger(schedule(maker=2.0, taker=5.5), Limits(max_position=1.0, max_loss=1e6))
    fee = ledger.fill(1, +1, 30_000.0, 0.01, maker=False, oid=1, reason="entry", path="print")
    assert fee == pytest.approx(300.0 * 5.5e-4)  # 0.165
    assert ledger.mark(2, 30_005.0) == pytest.approx(-300.0 - 0.165 + 300.05)
    fee = ledger.fill(3, -1, 30_010.0, 0.01, maker=True, oid=2, reason="exit", path="through")
    assert fee == pytest.approx(300.10 * 2e-4)  # 0.06002
    ledger.mark(4, 30_000.0)
    summary = ledger.summary()
    assert summary["position"] == 0.0
    assert summary["fees"] == pytest.approx(0.165 + 0.06002)
    assert summary["gross"] == pytest.approx(0.01 * (30_010.0 - 30_000.0))
    assert summary["net"] == pytest.approx(0.1 - 0.22502)
    assert summary["turnover"] == pytest.approx(600.10)
    assert summary["maker_turnover"] == pytest.approx(300.10)
    assert summary["net_bp_turnover"] == pytest.approx((0.1 - 0.22502) / 600.10 * 1e4)
    fills = ledger.fills_frame()
    assert fills["fee_bp"].tolist() == [5.5, 2.0]
    assert fills["fee_status"].tolist() == ["given", "given"]
    marks = ledger.marks_frame()
    assert marks["gross"].iloc[0] == pytest.approx(marks["net"].iloc[0] + 0.165)


def test_the_fee_is_the_dated_tables_at_the_fills_instant() -> None:
    spot = schedule_for("bybit", "spot", "BTCUSDT")
    assert spot.tier == "non_vip"
    inside = spot.quote(ns("2026-03-01"))
    assert (inside.maker_bp, inside.taker_bp, inside.status) == (10.0, 10.0, "sourced")
    after = spot.quote(ns("2026-10-27T12:00"))
    assert after.status == "carried" and after.taker_bp == 10.0
    vip = schedule_for("bybit", "spot", "BTCUSDT", "vip1").quote(ns("2026-03-01"))
    assert (vip.maker_bp, vip.taker_bp) == (6.75, 8.0)
    strict = schedule_for("bybit", "spot", "BTCUSDT", carry_forward=False)
    with pytest.raises(FeeNotSourced):
        strict.quote(ns("2026-10-27T12:00"))
    with pytest.raises(FeeNotSourced):
        spot.quote(ns("2024-01-01"))  # before the first interval: never carried backwards
    linear = schedule_for("bybit", "linear", "BTCUSDT")
    assert linear.quote(ns("2026-09-15")).taker_bp == 5.5
    binance = schedule_for("binance", "spot", "BTCUSDT")
    assert binance.quote(ns("2026-10-27T21:00")).status == "sourced"
    with pytest.raises(FeeNotSourced):
        binance.quote(ns("2026-10-01"))
    as_of = schedule_for("bybit", "linear", "DEMOUSDT", as_of_ns=ns("2026-09-15"))
    assert as_of.quote(ns("2026-01-01")).status.startswith("sourced as of")


def test_an_instrument_without_a_table_is_refused_unless_a_fee_is_given() -> None:
    with pytest.raises(FeeNotSourced):
        schedule_for("bybit", "spot", "ETHUSDT")
    given = schedule_for("bybit", "spot", "ETHUSDT", given=(1.0, 2.0))
    assert given.quote(0).status == "given"


def test_a_taker_round_trip_on_prints_by_hand(tmp_path: Path) -> None:
    """Buy at the first buyer print after the arrival, sweeping its stamp; the rest at
    the sweep's worst price; exit at the first seller print after the hold."""
    paper = session(tmp_path, mode="taker", clip=0.01, hold_s=1.0)
    events = rows_every(0, 1_500)
    events += [
        trade(5, 100.1, 1.0, +1),  # before the arrival (10 ms): not reachable
        trade(20, 99.9, 1.0, -1),  # the wrong side
        trade(20, 100.1, 0.004, +1),
        trade(20, 100.2, 0.004, +1),
        trade(30, 100.3, 1.0, +1),  # a later stamp: outside the sweep
        trade(1_150, 100.0, 1.0, -1),  # before the exit's arrival at 1,110? no: after it
        trade(1_300, 99.0, 1.0, -1),
    ]
    summary = paper.run(in_time(events))
    fills = paper.ledger.fills_frame()
    assert fills["side"].tolist() == [1, -1]
    entry = (0.004 * 100.1 + 0.004 * 100.2 + 0.002 * 100.2) / 0.01
    assert fills["price"].tolist() == pytest.approx([entry, 100.0])
    assert fills["path"].tolist() == ["print_beyond_sweep", "print"]
    # The exit is decided at the first row at or after the fill (20 ms) + 1 s: 1,100 ms.
    orders = paper.execution.orders_frame()
    assert orders["decided_ns"].iloc[1] - orders["decided_ns"].iloc[0] == 1_100_000_000
    fees = 0.01 * entry * 1e-3 + 0.01 * 100.0 * 1e-3
    assert summary["pnl"]["fees"] == pytest.approx(fees)
    assert summary["pnl"]["gross"] == pytest.approx(0.01 * (100.0 - entry))
    assert summary["pnl"]["net"] == pytest.approx(0.01 * (100.0 - entry) - fees)
    assert orders["book_walk_price"].iloc[0] == pytest.approx(100.1)


def test_a_passive_entry_fills_only_through_its_price(tmp_path: Path) -> None:
    paper = session(tmp_path, mode="passive", clip=0.01, hold_s=1.0)
    events = rows_every(0, 1_500)
    events += [
        trade(20, 99.9, 5.0, -1),  # at the bid: no fill on the trade-through rule
        trade(30, 99.8, 0.004, -1),  # through: 0.004 at the order's own price
        trade(40, 100.0, 1.0, +1),  # a buyer: never fills a bid
        trade(50, 99.7, 1.0, -1),  # through: the remaining 0.006
        trade(1_200, 99.5, 1.0, -1),  # the taker exit's print
    ]
    summary = paper.run(in_time(events))
    fills = paper.ledger.fills_frame()
    assert fills["size"].tolist() == pytest.approx([0.004, 0.006, 0.01])
    assert fills["price"].tolist() == pytest.approx([99.9, 99.9, 99.5])
    assert fills["maker"].tolist() == [True, True, False]
    assert fills["path"].tolist() == ["through", "through", "print"]
    assert summary["fill_rate"]["limit_fill_rate"] == 1.0
    assert summary["pnl"]["fees"] == pytest.approx(0.01 * 99.5 * 1e-3)  # maker at 0 bp
    assert summary["pnl"]["gross"] == pytest.approx(0.01 * (99.5 - 99.9))


def test_an_unfilled_passive_entry_is_cancelled_after_the_hold(tmp_path: Path) -> None:
    paper = session(tmp_path, mode="passive", clip=0.01, hold_s=1.0)
    paper.run(in_time([*rows_every(0, 2_000), trade(500, 99.9, 9.0, -1)]))
    orders = paper.execution.orders_frame()
    assert len(orders) == 1 and orders["outcome"].iloc[0] == "cancelled"
    assert orders["cancel_ns"].iloc[0] - orders["decided_ns"].iloc[0] == 1_000_000_000 + 10_000_000
    assert paper.ledger.fills_frame().empty


def test_the_maker_quotes_both_touches_and_stops_at_the_hard_limit(tmp_path: Path) -> None:
    paper = session(tmp_path, mode="maker", clip=0.01, max_position=1.0, every=1)
    events = rows_every(0, 3_000)
    events += [trade(ms, 99.8, 1.0, -1) for ms in range(150, 3_000, 200)]
    summary = paper.run(in_time(events))
    quotes = paper.execution.quotes_frame()
    assert len(quotes) == 30
    assert quotes["ask"].dropna().eq(100.1).all()
    fills = paper.ledger.fills_frame()
    assert (fills[fills["maker"]]["price"] == 99.9).all()
    # Soft limit six clips, hard limit seven: never more than seven clips long.
    assert fills["position_after"].max() == pytest.approx(0.07)
    assert summary["pnl"]["position"] == 0.0  # flattened at the end of the run


def test_marks_follow_every_row(tmp_path: Path) -> None:
    paper = session(tmp_path, mode="taker", proba=(0.2, 0.6, 0.2))
    paper.run(in_time([row(0), row(100, bid=100.9, ask=101.1)]))
    marks = paper.ledger.marks_frame()
    assert marks["mid"].tolist() == [100.0, 101.0]
    assert paper.decisions_frame()["outcome"].tolist() == ["no_signal"]
