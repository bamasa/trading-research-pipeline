"""Block summaries, kills in their registered order, the precedence and Holm."""

from __future__ import annotations

import math

import numpy as np
import pytest

from trading_research.spot_maker import registration as rg
from trading_research.spot_maker import statuses as st


def _day(value: float, *, making: float = 1.0, fills: float = 10.0, venue: float = 1.0,
         void: tuple[str, ...] = (), signal: bool = True) -> dict:  # fmt: skip
    runs = {"touch": value + 1, "delayed": value - 0.1, "latency": value - 0.1,
            "cells": [value] * 9, "all_long": 0.0, "all_short": 0.0}  # fmt: skip
    if signal:
        runs.update(flipped=-value, random=list(np.linspace(-1, 0.1, rg.RANDOM_DRAWS)))
    true = {"fills": fills, "maker_fills": fills, "taker_fills": 0.0, "net_bp_clip_day": value,
            "gross_bp_fill": value, "markout_5s_sum": making, "markout_5s_n": 1.0,
            "markout_1s_sum": 0.0, "markout_1s_n": 1.0, "markout_30s_sum": 0.0, "markout_30s_n": 1.0,
            "spread_sum": 0.0, "spread_n": 1.0, "capacity_slot_btc": 0.01, "capacity_venue_btc": venue,
            "posted_btc": 0.02, "filled_maker_btc": 0.01, "net_quote": value,
            "tiers": {"zero_fees": value, "vip3": value, "vip9": value}}  # fmt: skip
    return {"value": value, "true": true, "runs": runs, "void": list(void)}


def _records(values: dict[str, list[float]], **kw: object) -> list[dict]:
    n = len(next(iter(values.values())))
    return [{"usable": True, "slots": {s: _day(values[s][i], signal=rg.SLOTS[s].signal, **kw)
                                       for s in rg.FAMILY}} for i in range(n)]  # fmt: skip


def test_a_strong_positive_family_becomes_candidates() -> None:
    rng = np.random.default_rng(0)
    values = {s: list(2.0 + 0.3 * rng.standard_normal(30)) for s in rg.FAMILY}
    records = _records(values)
    summaries = {s: st.summarise(records, s) for s in rg.FAMILY}
    assert summaries["MK-S"]["days"] == 30 and summaries["MK-S"]["capacity"] == pytest.approx(0.01)
    found = st.assess(summaries, "H", dict.fromkeys(rg.FAMILY, 0.1))
    assert {f["status"] for f in found.values()} == {"candidate"}
    assert st.after_p("candidate", None) == "candidate (P not yet read)"
    assert st.after_p("candidate", "passes") == "confirmed"
    assert st.placebo_percentile(summaries["MK-S"]) == 100.0


def test_negative_void_and_too_few_fills() -> None:
    values = {s: [-1.0, -2.0, 0.5] * 10 for s in rg.FAMILY}
    records = _records(values)
    summaries = {s: st.summarise(records, s) for s in rg.FAMILY}
    found = st.assess(summaries, "H", {})
    assert found["MK-S"]["status"] == "killed" and found["MK-S"]["first_kill"]["name"] == "K1"
    assert [k["name"] for k in found["MK-Q"]["kills"]] == list(rg.SLOTS["MK-Q"].kills)
    few = {s: st.summarise(_records({k: [1.0] * 3 for k in rg.FAMILY}, fills=1.0), s)
           for s in rg.FAMILY}  # fmt: skip
    assert st.assess(few, "H", {})["MK-Q"]["status"] == "inconclusive (too few fills)"
    void = {s: st.summarise(_records(values, void=("K-acct",)), s) for s in rg.FAMILY}
    assert st.assess(void, "H", {})["MK-S"]["status"] == "void"


def test_making_part_capacity_and_headlines() -> None:
    values = {s: [1.0, 2.0, 1.5] * 10 for s in rg.FAMILY}
    records = _records(values, making=-0.5, venue=0.05)
    summaries = {s: st.summarise(records, s) for s in rg.FAMILY}
    checks = {k["name"]: k for k in st.kills(summaries["MK-S"], "MK-S")}
    assert checks["K-dir"]["fired"] and checks["K-cap"]["fired"] and not checks["K1"]["fired"]
    found = st.assess(summaries, "H", {})
    lines = st.headlines(summaries, found)
    assert lines[0].startswith("MK-S on H, Binance spot BTC/FDUSD at the fees charged: killed;")
    assert "first kill K-dir at -0.50 bp against 0.00 bp" in lines[0]
    assert math.isfinite(summaries["MK-Q"]["fill_ratio"])
