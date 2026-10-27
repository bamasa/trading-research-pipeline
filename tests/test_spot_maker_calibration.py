"""The fill rule's calibration: the quoting rule in the simulator, the prints rule, the test."""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from trading_research.market_making.quoters import MarketView
from trading_research.market_making.synthetic import random_market
from trading_research.spot.calibration import thin_prints
from trading_research.spot_maker import calibration as cb
from trading_research.spot_maker import registration as rg
from trading_research.spot_maker.executor import settings_for

DAY = date(2026, 5, 4)
MID = int(np.datetime64("2026-05-04T00:00:00", "ns").astype(np.int64))
S = 1_000_000_000


def _view(ts: float, bid: int, ask: int) -> MarketView:
    return MarketView(MID + round(ts * S), 0.1, np.array([bid]), np.array([1.0]), np.array([ask]),
                      np.array([1.0]), bid, ask, (bid + ask) * 0.05, 1.0, 1.0, 0.01, 0.01, {})  # fmt: skip


def _quoter(slot: str, signal: tuple[list[int], list[int]] = ([], [])) -> cb.BookQuoter:
    settings = settings_for(slot, tick=0.1)
    instants = np.array([MID + 60 * S, MID + 120 * S], dtype=np.int64)
    return cb.BookQuoter(settings, instants, np.array(signal[0], dtype=np.int64),
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
    later = quoter.quotes(_view(120.05, 1000, 1002), 0.010)  # long a clip: no bid, the ask skewed
    assert later.bid.price is None and later.ask.price == 1002


def test_the_signal_side_is_doubled_within_the_limit() -> None:
    quoter = _quoter("MK-SQ", ([MID + 55 * S], [1]))
    found = quoter.quotes(_view(60.05, 1000, 1001), 0.0)
    assert found.bid.size == pytest.approx(0.02) and found.ask.size == pytest.approx(0.01)


@pytest.fixture(scope="module")
def market():  # type: ignore[no-untyped-def]
    return random_market(3, n_snapshots=3_000, snapshot_ms=400, prints_per_snapshot=1.5,
                         tick=0.1, lot=0.001, start_ticks=500_000, day=DAY)  # fmt: skip


def test_both_runs_on_a_synthetic_book(market) -> None:  # type: ignore[no-untyped-def]
    full = cb.prints_frame(market)
    frames = [thin_prints(full, len(full) // 2, seed).reset_index(drop=True) for seed in (0, 1)]
    none = np.zeros(0, dtype=np.int64)
    book = cb.book_runs(market, "MK-Q", none, none)
    assert set(book) == set(rg.CALIBRATION_RULES)
    assert book["trade_through"]["maker_btc"] <= book["touch"]["maker_btc"] + 1e-12
    prints = cb.prints_runs(frames, DAY, "MK-Q", none, none)
    assert prints["maker_fills"] >= 0 and (prints["making_n"] == 0 or math.isfinite(prints["making_part"]))
    signal = cb.book_runs(market, "MK-SQ", np.array([MID + 100 * S]), np.array([1]))
    assert signal["queue"]["maker_fills"] >= 0


def _record(fills: tuple[int, int, int], making: tuple[float, float, float], prints: float) -> dict:
    book = {r: {"maker_fills": n, "maker_btc": n * 0.001, "making_part": m, "excluded": []}
            for r, n, m in zip(rg.CALIBRATION_RULES, fills, making, strict=True)}  # fmt: skip
    slot = {"book": book, "prints": {"maker_fills": 5, "making_part": prints}}
    return {"usable": True, "slots": dict.fromkeys(rg.CALIBRATION_SLOTS, slot)}


def test_the_registered_test_and_its_label(tmp_path: Path) -> None:
    good = [_record((2, 5, 9), (-3.0, -1.0, 0.5), -1.5)] * 9 + [_record((5, 2, 9), (-3, -1, 0.5), 2.0)]
    found = cb.verdict(good)
    assert found["passes"] and found["label"] is None and found["MK-Q"]["ordering_share"] == 0.9
    bad = good[:7] + [_record((5, 2, 9), (-3, -1, 0.5), 2.0)] * 3
    found = cb.verdict(bad)
    assert not found["passes"] and found["label"] == "uncalibrated fill rule"
    assert found["MK-SQ"]["bracket_share"] == pytest.approx(0.7)
    skipped = cb.verdict([{"usable": False}, _record((0, 1, 2), (0, 0, 0), 0.0)])
    assert skipped["MK-Q"]["days"] == 0 and not skipped["passes"]
    none = cb.run_calibration(tmp_path, None, 1000, None, tmp_path / "c")
    assert none["status"] == "not run (no book day on disk)" and none["label"] == rg.CALIBRATION_LABEL
    assert len(cb.calibration_days()) == 25
