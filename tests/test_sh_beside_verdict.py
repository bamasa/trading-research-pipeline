"""What is reported beside every verdict: fee break-evens, tiers, sides, drawdown and counters."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from trading_research.backtest.costs import BYBIT_BASE, best_maker_tier, tiers_for
from trading_research.pipeline.short_horizon import beside_verdict as bv
from trading_research.pipeline.short_horizon import statuses as st
from trading_research.pipeline.short_horizon.page_numbers import slot_numbers
from trading_research.reporting.finalists import FinalistInputs, finalist_row

DAYS = [date(2025, 7, 8) + timedelta(days=i) for i in range(24)]  # dates only; no data is read
N = len(DAYS)


def _passive(slot: str, kind: str, value: np.ndarray, sums: dict[str, float]) -> st.SlotDays:
    zeros = np.zeros(N)
    extra = {f"day/{name}": np.full(N, v) for name, v in sums.items()}
    return st.SlotDays(slot, kind, list(DAYS), value, np.full(N, 10), gross=extra["day/net_usdt"],
                       turnover=extra["day/passive_notional"], ic=np.full(N, 0.1),
                       random=np.zeros((1, N)), flipped=zeros, shifted=zeros, long_only=zeros,
                       short_only=zeros, neighbours=np.zeros((1, N)), latency=value,
                       passive_notional=extra["day/passive_notional"], extra=extra)  # fmt: skip


def _s25z() -> st.SlotDays:
    """Ten filled entries a day at ~48,000 x 0.010, gross +3 bp of entry notional
    at 0/0, the taker exit's notional equal to the entry's (the reviewer's trace)."""
    maker = 10 * 480.0
    net00 = maker * 3.0 * 1e-4
    s = _passive("S25-Z", "zero", np.full(N, 3.0),
                 {"net_usdt": net00, "passive_notional": maker, "taker_notional": maker,
                  "entry_notional": maker})  # fmt: skip
    s.second = np.full(N, 3.0)
    return s


def test_s25z_break_even_is_the_maker_fee_with_the_taker_exit_at_the_base_fee() -> None:
    """Registered: (net00 - 5.5e-4 N_taker) / N_maker x 1e4 = 3 - 5.5 = -2.5 bp (the
    page showed 3.0 bp, its gross over the maker side, and 13 tiers paying). A tier
    pays when its own maker and taker fees leave a positive net: 3 - m - x > 0."""
    s = _s25z()
    mask = np.ones(N, dtype=bool)
    assert st.fee_check(s, mask).value == pytest.approx(-2.5)
    assert bv.passive_break_even(s, mask, 1.5) == pytest.approx(1.5)
    paying = sum(t.maker_bp + t.taker_bp < 3.0 for t in tiers_for("g1"))
    assert paying == 4 and bv.tiers_paying(s, mask) == paying
    numbers = slot_numbers(s, list(DAYS))
    assert numbers["break_even_per_side"] == pytest.approx(-2.5)
    assert numbers["tiers_paying"] == paying
    assert numbers["net_base_tier_mean"] == pytest.approx(3.0 - 2.0 - 5.5)
    cheapest = best_maker_tier("g1")
    assert numbers["net_cheapest_tier_mean"] == pytest.approx(
        3.0 - cheapest.maker_bp - cheapest.taker_bp
    )
    row = finalist_row(FinalistInputs("S25-Z", "H", "inconclusive", np.full(N, 3.0), 240,
                                      break_even_per_side=numbers["break_even_per_side"],
                                      tiers_paying=numbers["tiers_paying"]))  # fmt: skip
    assert row["tiers_paying"] == paying
    assert st._series(s, s.value, mask).size == N  # a taker slot's days with a trade


def test_a_tier_pays_when_its_own_maker_and_taker_fees_leave_a_positive_net() -> None:
    """S-P earned 0.6 USDT a day at the base tier on 100 of passive and 100 of
    taker notional: +60 bp over the maker side before... at (m, x) its net is
    0.6 - (m - 2) 1e-4 x 100 - (x - 5.5) 1e-4 x 100, positive for every tier."""
    s = _passive("S-P", "entry", np.full(N, 0.6),
                 {"net_usdt": 0.6, "passive_notional": 100.0, "taker_notional": 100.0,
                  "per100_maker_weight": 100.0, "per100_taker_weight": 100.0})  # fmt: skip
    mask = np.ones(N, dtype=bool)
    rows = bv.tier_results(s, mask)
    for row, tier in zip(rows, tiers_for("g1"), strict=True):
        shift = (tier.maker_bp - 2.0 + tier.taker_bp - 5.5) * 1e-4 * 100.0
        assert row["net_usdt"] == pytest.approx(N * (0.6 - shift))
        assert row["mean"] == pytest.approx(0.6 - shift)
    assert bv.tiers_paying(s, mask) == len(rows)
    assert st.fee_check(s, mask).value == pytest.approx(2.0 + 0.6 / 100.0 * 1e4)
    found = bv.verdict_measures(s, DAYS)
    assert found["break_even_maker_bp"]["exit_1.5"] == pytest.approx(
        2.0 + (0.6 + 4e-4 * 100) / 100 * 1e4
    )
    assert found["maker_fees"]["maker_0"]["mean"] == pytest.approx(0.6 + 2e-4 * 100)
    assert found["maker_fees"]["maker_-1"]["mean"] == pytest.approx(0.6 + 3e-4 * 100)


def test_k_dir_reads_the_mean_over_the_passive_fills_in_bp_per_fill() -> None:
    """One day 1 fill at -10 bp on 1,000 USDT, the others 9 fills at +1 bp on 10 USDT
    each: the mean per fill is (-10 + 9 x 23 days ...) / fills, not notional-weighted."""
    value = np.full(N, 0.5)
    s = _passive("S-M", "maker", value, {"net_usdt": 0.1, "passive_notional": 90.0,
                                          "taker_notional": 0.0, "making_bp_sum": 9.0,
                                          "making_fills": 9.0})  # fmt: skip
    s.extra["day/making_bp_sum"][0], s.extra["day/making_fills"][0] = -10.0, 1.0
    s.making = np.full(N, 9.0 * 10.0 * 1e-4)
    s.making[0] = -10.0 * 1_000.0 * 1e-4
    s.passive_notional = np.full(N, 90.0)
    s.passive_notional[0] = 1_000.0
    mask = np.ones(N, dtype=bool)
    per_fill = (9.0 * (N - 1) - 10.0) / (9 * (N - 1) + 1)
    assert bv.making_bp_per_fill(s, mask) == pytest.approx(per_fill) and per_fill > 0
    weighted = float(s.making.sum() / s.passive_notional.sum() * 1e4)
    assert weighted < 0  # the notional-weighted figure would have fired K-dir
    k_dir = next(k for k in st.kills(s, mask) if k.name == "K-dir")
    assert k_dir.value == pytest.approx(per_fill) and not k_dir.fired


def test_a_taker_day_is_kept_as_sums_and_read_back_by_side_and_tier() -> None:
    from _engine_books import decisions, make_book
    from trading_research.backtest.taker import execute_day

    book = make_book(3)
    rows, side = decisions(book, 40, seed=2)
    day = execute_day(book, book.ts[rows], side, 5 * 10**9)
    sums = bv.taker_day_sums(day)
    trades = day.trades
    assert sums["day/net_long_usdt"] + sums["day/net_short_usdt"] == pytest.approx(
        trades["net"].sum()
    )
    assert sums["day/turnover"] == pytest.approx(day.turnover)
    assert sums["day/entries_busy"] == day.counters["entries_busy"]
    extra = {name: np.full(N, v) for name, v in sums.items()}
    value = np.full(N, day.day_bp("net"))
    zeros = np.zeros(N)
    s = st.SlotDays("S-C", "taker", list(DAYS), value, np.full(N, len(trades)),
                    np.full(N, sums["day/gross_usdt"]), np.full(N, day.turnover), np.zeros(N),
                    np.zeros((1, N)), zeros, zeros, zeros, zeros, np.zeros((1, N)), value, extra=extra)  # fmt: skip
    found = bv.verdict_measures(s, DAYS)
    base = next(r for r in found["tiers"] if r["tier"] == BYBIT_BASE.name)
    assert base["mean"] == pytest.approx(value[0]) and base["net_usdt"] == pytest.approx(
        N * trades["net"].sum()
    )
    assert found["by_side"]["long"]["trades"] == N * int((trades["side"] > 0).sum())
    assert found["counters"]["entries_busy"] == N * day.counters["entries_busy"]


def test_max_drawdown_is_the_largest_fall_from_a_running_peak() -> None:
    assert bv.max_drawdown(np.array([1.0, -2.0, 0.5, -1.0, 3.0])) == pytest.approx(2.5)
    assert bv.max_drawdown(np.array([np.nan, 1.0, 2.0])) == 0.0
    assert np.isnan(bv.max_drawdown(np.array([np.nan])))


def test_a0s_crossing_is_reported_at_the_base_and_every_published_tier() -> None:
    crossing = [{"net": 2.0, "turnover": 1_000.0, "notional": 500.0},
                {"net": -1.0, "turnover": 2_000.0, "notional": 1_000.0}]  # fmt: skip
    rows = bv.crossing_tiers(crossing)
    assert [r["tier"] for r in rows] == [t.name for t in tiers_for("g1")]
    base = rows[0]
    assert base["tier"] == "base" and base["net_usdt"] == pytest.approx(1.0 - 5.5e-4 * 3_000.0)
    assert base["mean_net_bp"] == pytest.approx(
        ((2.0 - 0.55) / 500 + (-1.0 - 1.1) / 1_000) / 2 * 1e4
    )
