"""The fill rule's calibration: the slots' rules in the simulator, the prints rule, the test."""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_research.market_making.quoters import MarketView
from trading_research.market_making.synthetic import random_market
from trading_research.spot.calibration import thin_prints
from trading_research.spot_maker import book_quoters as bq
from trading_research.spot_maker import calibration as cb
from trading_research.spot_maker import registration as rg
from trading_research.spot_maker.executor import settings_for
from trading_research.spot_maker.signal import DayDecisions

DAY = date(2026, 5, 4)
MID = int(np.datetime64("2026-05-04T00:00:00", "ns").astype(np.int64))
S = 1_000_000_000


def _view(ts: float, bid: int, ask: int) -> MarketView:
    return MarketView(MID + round(ts * S), 0.1, np.array([bid]), np.array([1.0]), np.array([ask]),
                      np.array([1.0]), bid, ask, (bid + ask) * 0.05, 1.0, 1.0, 0.01, 0.01, {})  # fmt: skip


def _quoter(slot: str, signal: tuple[list[int], list[int]] = ([], [])) -> bq.BookQuoter:
    instants = np.array([MID + 60 * S, MID + 120 * S], dtype=np.int64)
    return bq.BookQuoter(settings_for(slot, tick=0.1), instants, np.array(signal[0], dtype=np.int64),
                         np.array(signal[1], dtype=np.int64), name=slot)  # fmt: skip


def test_the_book_quoter_holds_its_quotes_between_instants_and_never_replenishes() -> None:
    quoter = _quoter("MK-Q")
    assert quoter.quotes(_view(30, 1000, 1001), 0.0).bid.price is None
    first = quoter.quotes(_view(60.05, 1000, 1001), 0.0)
    assert (first.bid.price, first.ask.price) == (1000, 1001) and first.bid.size == 0.01
    moved = quoter.quotes(_view(70, 999, 1000), 0.004)  # a partial fill of the bid
    assert moved.bid == (1000, pytest.approx(0.006)) and moved.ask.price == 1001
    full = quoter.quotes(_view(80, 999, 1000), 0.010)
    assert full.bid.price is None and full.ask.size == pytest.approx(0.01)
    later = quoter.quotes(_view(120.05, 1000, 1002), 0.010)  # long a clip: no bid
    assert later.bid.price is None and later.ask.price == 1002
    doubled = _quoter("MK-SQ", ([MID + 55 * S], [1])).quotes(_view(60.05, 1000, 1001), 0.0)
    assert doubled.bid.size == pytest.approx(0.02) and doubled.ask.size == pytest.approx(0.01)


def test_the_book_signal_maker_enters_cancels_holds_and_exits() -> None:
    settings = settings_for("MK-S", tick=0.1)
    maker = bq.BookSignalMaker(settings, np.array([MID + 100 * S, MID + 300 * S]), np.array([1, 1]),
                               MID + 60 * S, MID + 86_280 * S, MID + 86_340 * S)  # fmt: skip
    assert maker.quotes(_view(90, 1000, 1001), 0.0).bid.price is None
    entry = maker.quotes(_view(100.05, 1000, 1001), 0.0)
    assert entry.bid == (1000, 0.01) and entry.ask.price is None
    assert maker.quotes(_view(120, 999, 1000), 0.01).bid.price is None  # filled whole
    assert maker.state == "entry" and maker.quotes(_view(160, 999, 1000), 0.01).bid.price is None
    assert maker.state == "held"  # the entry cancelled at 150 s
    exit_ = maker.quotes(_view(170.1, 1003, 1004), 0.01)  # the fill seen at 120 s + 50 s
    assert maker.state == "exit" and exit_.ask == (1004, 0.01)
    assert maker.quotes(_view(200, 1003, 1004), 0.0).ask.price is None and maker.state == "idle"
    again = maker.quotes(_view(300.05, 1005, 1006), 0.0)
    assert again.bid.price == 1005
    assert maker.quotes(_view(350.1, 1005, 1006), 0.0).bid.price is None  # cancelled unfilled


@pytest.fixture(scope="module")
def market():  # type: ignore[no-untyped-def]
    return random_market(3, n_snapshots=3_000, snapshot_ms=400, prints_per_snapshot=1.5,
                         tick=0.1, lot=0.001, start_ticks=500_000, day=DAY)  # fmt: skip


def test_every_slot_runs_under_the_books_rules_and_the_prints_rule(market) -> None:  # type: ignore[no-untyped-def]
    n = len(market.trade_ts)
    frame = pd.DataFrame({
        "timestamp": pd.to_datetime(np.asarray(market.trade_ts, dtype=np.int64), utc=True),
        "price": np.asarray(market.trade_px, dtype=np.float64) * market.spec.tick,
        "size": np.asarray(market.trade_sz, dtype=np.float64),
        "aggressor": np.asarray(market.trade_aggressor, dtype=np.int8),
        "trade_id": np.arange(n, dtype=np.int64), "rpi": np.zeros(n, dtype=bool),
        "source": pd.Series(["synthetic"] * n, dtype="string")})  # fmt: skip
    frames = [thin_prints(frame, n // 2, seed).reset_index(drop=True) for seed in (0, 1)]
    ns = np.array([MID + 100 * S, MID + 400 * S], dtype=np.int64)
    decisions = DayDecisions(np.unique(np.asarray(market.trade_ts)), ns, np.array([1, -1]), 10)
    for slot in rg.CALIBRATION_SLOTS:
        book = cb.book_runs(market, slot, decisions.decision_ns, decisions.decision_side)
        assert set(book) == set(rg.CALIBRATION_RULES)
        assert book["trade_through"]["maker_btc"] <= book["touch"]["maker_btc"] + 1e-12
        prints = cb.prints_runs(frames, DAY, slot, 0.1, decisions)
        assert prints["maker_fills"] >= 0
        assert prints["making_n"] == 0 or math.isfinite(prints["making_part"])


def _record(fills: tuple[int, int, int], making: tuple[float, float, float], prints: float) -> dict:
    book = {r: {"maker_fills": n, "maker_btc": n * 0.001, "making_part": m, "excluded": []}
            for r, n, m in zip(rg.CALIBRATION_RULES, fills, making, strict=True)}  # fmt: skip
    slot = {"book": book, "prints": {"maker_fills": 5, "making_part": prints}}
    return {"usable": True, "slots": dict.fromkeys(rg.CALIBRATION_SLOTS, slot)}


def test_the_registered_test_and_its_label(tmp_path: Path) -> None:
    good = [_record((2, 5, 9), (-3.0, -1.0, 0.5), -1.5)] * 9 + [
        _record((5, 2, 9), (-3, -1, 0.5), 2.0)
    ]
    found = cb.verdict(good)
    assert found["passes"] and found["label"] is None and found["MK-Q"]["ordering_share"] == 0.9
    bad = good[:7] + [_record((5, 2, 9), (-3, -1, 0.5), 2.0)] * 3
    found = cb.verdict(bad)
    assert not found["passes"] and found["label"] == "uncalibrated fill rule"
    assert found["MK-S"]["bracket_share"] == pytest.approx(0.7)
    skipped = cb.verdict([{"usable": False}, _record((0, 1, 2), (0, 0, 0), 0.0)])
    assert skipped["MK-Q"]["days"] == 0 and not skipped["passes"]
    none = cb.run_calibration(tmp_path, None, {"2026-05-01": 1000}, None, tmp_path / "c")
    assert (
        none["status"] == "not run (no book day on disk)" and none["label"] == rg.CALIBRATION_LABEL
    )
    assert len(cb.calibration_days()) == 92
    away = cb.calibration_day(tmp_path, date(2026, 5, 1), 1000, None)
    assert away["usable"] is False
