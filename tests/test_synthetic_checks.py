"""The checks read planted data right: they find the effect, nothing on the null, and a leak."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_research.market_making.simulator import SimConfig, simulate_day
from trading_research.synthetic import null, planted_edge, recover
from trading_research.synthetic.book_days import BookDaysSpec
from trading_research.synthetic.cross_section import CrossSectionSpec
from trading_research.synthetic.cross_section import generate as generate_panel
from trading_research.synthetic.order_flow import OrderFlowSpec
from trading_research.synthetic.order_flow import generate as generate_flow
from trading_research.synthetic.reversion_check import relative_trades, summary
from trading_research.synthetic.simulator_check import ProbeQuoter, probe_orders, share
from trading_research.synthetic.verify import CheckResult

SPEC = BookDaysSpec(days=12, session_s=1800)


@pytest.fixture(scope="module")
def planted():
    return planted_edge(SPEC)


@pytest.fixture(scope="module")
def nothing():
    return null(SPEC)


def test_the_imbalance_rule_recovers_the_planted_edge(planted) -> None:
    report = recover("order_flow", planted, placebo=True)
    s = report.summary
    assert s["recovered_t"] >= 3
    assert s["planted_bp"] >= 0.8 * SPEC.size_bp
    assert abs(s["residual_t"]) < 3
    assert abs(report.placebo["recovered_t"]) < 3


def test_the_imbalance_rule_finds_nothing_on_the_null(nothing) -> None:
    s = recover("order_flow", nothing, placebo=False).summary
    assert abs(s["recovered_t"]) < 3
    assert abs(s["residual_t"]) < 3


def test_a_strategy_that_reads_the_future_is_caught_by_the_residual(nothing) -> None:
    """On the null the planted drift owes nothing, yet a peeking rule earns: the
    residual, recovered minus planted, is what gives it away."""
    mids = {}
    for day in nothing.days:
        book = nothing.book[day]
        mid = ((book["bid_price_0"] + book["ask_price_0"]) / 2).to_numpy()
        stamps = book["timestamp"].to_numpy(dtype="datetime64[ns]").view(np.int64)
        mids[day.isoformat()] = (stamps, mid)

    def peeking(frame: pd.DataFrame) -> np.ndarray:
        _, mid = mids[str(frame["day"].iloc[0])]
        ahead = np.roll(mid, -300) - mid
        ahead[-300:] = 0.0
        out = np.sign(ahead)
        out[np.abs(ahead) < 0.8] = 0.0
        return out

    s = recover(peeking, nothing, placebo=False).summary
    assert s["recovered_t"] > 3
    assert s["residual_t"] > 3


def test_a_rule_never_sees_the_forward_move(planted) -> None:
    seen: list[set[str]] = []

    def looking(frame: pd.DataFrame) -> np.ndarray:
        seen.append(set(frame.columns))
        return np.zeros(len(frame))

    recover(looking, planted, placebo=False)
    assert seen and all("forward_bp" not in columns for columns in seen)


def test_the_relative_rule_recovers_reversion_and_nothing_on_the_null() -> None:
    planted_panel = generate_panel(CrossSectionSpec(days=6))
    null_panel = generate_panel(CrossSectionSpec(days=6, null=True))
    found = summary(relative_trades(planted_panel))
    empty = summary(relative_trades(null_panel))
    assert found["recovered_t"] >= 3 and abs(found["residual_t"]) < 3
    assert abs(empty["recovered_t"]) < 3


def test_the_probe_holds_each_order_for_its_lifetime_and_fills_match_the_replay() -> None:
    data = generate_flow(OrderFlowSpec(days=1, session_s=1200, adverse_bp=0.0, noise_rate=0.02))
    orders = probe_orders(data, data.days[0], hold_s=5.0)
    assert len(orders) > 100
    lifetime = (orders["end_ns"] - orders["decided_ns"]) / 1e9
    # Every order lives its five seconds, but those the day's stop cancelled early.
    assert lifetime.between(5.0, 5.2).mean() > 0.95
    sim, _ = share(orders["sim_hit"])
    true, _ = share(orders["true_hit"])
    assert abs(sim - true) < 0.05


def test_the_probe_is_an_ordinary_quoter_to_the_simulator() -> None:
    data = generate_flow(OrderFlowSpec(days=1, session_s=900))
    config = SimConfig(clip_btc=1.0, warmup_s=60.0, soft_limit_clips=1e9)
    result = simulate_day(data.events(data.days[0]), ProbeQuoter(), config)  # type: ignore[arg-type]
    assert result.counters["placed"] > 50
    # Post-only refuses an order the book moved through while it travelled.
    assert result.counters.get("rejected", 0) < 0.05 * result.counters["placed"]


def test_a_check_page_says_what_failed(tmp_path) -> None:
    result = CheckResult(
        "demo", "a demonstration", "nothing", "p", "q", "trading-research verify demo"
    )
    result.add("finds it", "1 bp", "0 bp", "t >= 3", False)
    result.add("measured", "x", "y", "measured", True, binding=False)
    page = result.write(tmp_path).read_text()
    assert not result.passed
    assert "**FAIL**: 0 of 1 binding criteria passed" in page
    assert "| finds it | 1 bp | 0 bp | t >= 3 | FAIL |" in page
    assert "| measured | x | y | measured | info |" in page
