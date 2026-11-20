"""The cross-venue funding rule on planted venues, and the Bybit kline pager."""

from __future__ import annotations

import io
import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from trading_research.data import bybit_klines
from trading_research.xvenue import engine

HOURS = pd.date_range("2021-10-01", "2022-07-01", freq="1h", tz="UTC", inclusive="left")
SETTLE = HOURS[HOURS.hour % 8 == 0]


def venues(spread_bp_a_day: float, seed: int, gap_bp: float = 0.0) -> engine.Venues:
    """Same prices on both venues (plus a fixed gap); Binance pays ``spread`` more a day."""
    rng = np.random.default_rng(seed)
    price = pd.Series(10 * np.exp(np.cumsum(rng.normal(0, 0.003, len(HOURS)))), index=HOURS)
    base = 1.0 / 3 / 1e4  # 1 bp a day on Bybit
    return engine.Venues(
        binance=price,
        binance_funding=pd.Series(base + spread_bp_a_day / 3 / 1e4, index=SETTLE),
        bybit=price * (1 + gap_bp / 1e4),
        bybit_funding=pd.Series(base, index=SETTLE),
    )


MONTHS = pd.period_range("2022-01", "2022-06", freq="M")


def test_spread_reads_the_months_before_only() -> None:
    v = venues(3.0, 0)
    month = pd.Period("2022-03", "M")
    assert engine.spread(v, month) == pytest.approx(3.0)
    later = engine.Venues(v.binance, v.binance_funding.where(v.binance_funding.index < pd.Timestamp("2022-03-01", tz="UTC"), 1.0), v.bybit, v.bybit_funding)  # fmt: skip
    assert engine.spread(later, month) == pytest.approx(3.0)


def test_the_rule_takes_wide_spreads_in_their_direction_and_skips_narrow_ones() -> None:
    vs = {"WIDE": venues(4.0, 1), "BACK": venues(-3.0, 2), "NARROW": venues(0.5, 3)}
    chosen = engine.choose(vs, pd.Period("2022-02", "M"))
    assert chosen == [("WIDE", 1), ("BACK", -1)]


def test_a_held_pair_earns_the_spread_and_pays_one_round_trip() -> None:
    vs = {"WIDE": venues(4.0, 1)}
    out = engine.replay(vs, MONTHS)
    days = sum(m.days_in_month for m in MONTHS)
    slot = 1 / engine.SLOTS / 2  # one leg's share of the account
    assert out["price_pct"].abs().max() < 1e-9  # same prices: no gap
    expected_funding = 4.0e-4 * days * slot * 100
    assert out["funding_pct"].sum() == pytest.approx(expected_funding, rel=0.05)
    assert out["cost_pct"].iloc[0] < 0 and (out["cost_pct"].iloc[1:] == 0).all()
    reversed_ = engine.replay(vs, MONTHS, chooser=lambda _m: [("WIDE", -1)])
    assert reversed_["return_pct"].sum() < 0 < out["return_pct"].sum()


def test_the_price_gap_is_the_long_leg_minus_the_short() -> None:
    v = venues(4.0, 1)
    start, end = pd.Timestamp("2022-02-01", tz="UTC"), pd.Timestamp("2022-02-28 23:00", tz="UTC")
    wider = engine.Venues(v.binance, v.binance_funding, v.bybit * (1 + np.linspace(0, 0.01, len(HOURS))), v.bybit_funding)  # fmt: skip
    h = engine.hourly(wider, 1.0, 1, start, end)  # long Bybit, whose price drifts up
    assert h["price"].sum() > 0
    assert engine.hourly(wider, 1.0, -1, start, end)["price"].sum() == pytest.approx(
        -h["price"].sum()
    )


class _Response(io.BytesIO):
    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_: Any) -> None:
        return None


def test_the_kline_pager_covers_the_span_once(tmp_path: Path) -> None:
    asked = []

    def opener(url: str, timeout: float) -> _Response:
        q = dict(p.split("=") for p in url.split("?")[1].split("&"))
        a, b = int(q["start"]), int(q["end"])
        asked.append((a, b))
        rows = [
            [str(t), "1", "1", "1", str(t / 1e12), "1", "1"] for t in range(b, a - 1, -3_600_000)
        ]
        return _Response(json.dumps({"retCode": 0, "result": {"list": rows}}).encode())

    bybit_klines.download_range(
        "XUSDT", date(2022, 1, 1), date(2022, 3, 31), tmp_path, opener=opener
    )
    s = bybit_klines.load("XUSDT", date(2022, 1, 1), date(2022, 3, 31), tmp_path)
    assert len(s) == 90 * 24 and s.index.is_unique and s.index.is_monotonic_increasing
    assert len(asked) == 3  # 2,160 hours at 1,000 a page
    bybit_klines.download_range(
        "XUSDT", date(2022, 1, 1), date(2022, 3, 31), tmp_path, opener=opener
    )
    assert len(asked) == 3  # every day on disk: nothing fetched again


def test_a_pair_carried_across_months_is_marked_without_a_gap() -> None:
    flat_price = pd.Series(10.0, index=HOURS)
    v0 = venues(4.0, 1)
    v = engine.Venues(flat_price, v0.binance_funding, flat_price, v0.bybit_funding)
    step = pd.Series(1.0, index=HOURS).where(pd.Timestamp("2022-03-01", tz="UTC") > HOURS, 1.05)
    jumped = engine.Venues(v.binance, v.binance_funding, v.bybit * step, v.bybit_funding)
    months = pd.period_range("2022-02", "2022-03", freq="M")
    out = engine.replay({"X": jumped}, months, chooser=lambda _m: [("X", 1)])
    flat = engine.replay({"X": v}, months, chooser=lambda _m: [("X", 1)])
    # Long Bybit, whose price steps up 5% at the month's first hour: the step is earned.
    assert out["price_pct"].sum() - flat["price_pct"].sum() == pytest.approx(
        5.0 / engine.SLOTS / 2, rel=0.02
    )


def test_a_pair_closed_at_a_month_start_is_marked_to_that_hour() -> None:
    flat_price = pd.Series(10.0, index=HOURS)
    v0 = venues(4.0, 1)
    v = engine.Venues(flat_price, v0.binance_funding, flat_price, v0.bybit_funding)
    step = pd.Series(1.0, index=HOURS).where(pd.Timestamp("2022-03-01", tz="UTC") > HOURS, 1.05)
    jumped = engine.Venues(v.binance, v.binance_funding, v.bybit * step, v.bybit_funding)
    picks = {pd.Period("2022-02", "M"): [("X", 1)], pd.Period("2022-03", "M"): []}
    out = engine.replay({"X": jumped}, list(picks), chooser=lambda m: picks[m])
    assert out["price_pct"].sum() == pytest.approx(5.0 / engine.SLOTS / 2, rel=0.02)
