"""The Bybit 2021 engine on hand-built days: the fill rule, its brackets, latency,
post-only, the limits, the inventory rule, funding, the flatten and the day's numbers."""

from __future__ import annotations

import numpy as np
import pytest

from _mm_bybit_days import CLOSING, HOUR, INPUTS, OPENING, settings, tape_of
from trading_research.mm_bybit import engine, metrics
from trading_research.mm_bybit import quoters as q

# A bid at 100.0 goes live at 601.05; the print at its arrival does not fill it,
# two later prints below it do, one at its price does not.
SWEEP = (
    *OPENING,
    (601.0, 100.0, 0.001, -1),
    (601.05, 99.5, 0.004, -1),
    (601.08, 99.5, 0.004, -1),
    (601.09, 100.0, 0.5, -1),
    (601.095, 99.0, 0.02, -1),
    *CLOSING,
)


def run(prints, *, cell: q.Cell = q.S0, rates=(0.0, 0.0, 0.0), **changes) -> engine.DayRun:
    return engine.run_day(tape_of(prints, rates=rates), cell, settings(**changes), INPUTS)


def maker_fills(result: engine.DayRun) -> list[tuple[int, float, float]]:
    m = result.fill_maker
    return [
        (int(s), float(p), round(float(z), 9))
        for s, p, z in zip(
            result.fill_side[m], result.fill_price[m], result.fill_size[m], strict=True
        )
    ]


def test_trade_through_fills_only_strictly_through_at_the_order_price_by_print_size() -> None:
    result = run(SWEEP)
    assert maker_fills(result) == [(1, 100.0, 0.004), (1, 100.0, 0.006)]
    stamps = (result.fill_ts[result.fill_maker] - tape_of(SWEEP).start_ns) / 1e9
    assert np.allclose(stamps, [601.08, 601.095])
    assert result.identity_ok
    assert result.counters.get("limit_breach", 0) == 0


def test_the_touch_bracket_fills_the_whole_order_at_the_first_print_at_or_through() -> None:
    assert maker_fills(run(SWEEP, rule="touch")) == [(1, 100.0, 0.01)]


def test_the_delay_bracket_waits_for_a_print_stamped_after_the_first_trade_through() -> None:
    result = run(SWEEP, rule="delay")
    assert maker_fills(result) == [(1, 100.0, 0.01)]
    stamp = (result.fill_ts[result.fill_maker][0] - tape_of(SWEEP).start_ns) / 1e9
    assert stamp == pytest.approx(601.095)


def test_a_print_stamped_at_the_arrival_never_fills() -> None:
    prints = (*OPENING, (601.0, 100.0, 0.001, -1), (601.05, 99.0, 1.0, -1), *CLOSING)
    assert maker_fills(run(prints)) == []


def test_a_print_stamped_at_a_cancels_landing_still_fills() -> None:
    # The sell at 100.5 locks the proxies; the quotes are pulled apart (R4) and the
    # ask moves from 100.5 to 101.0: its cancel lands at 601.15, with the buy at 101.0.
    prints = (
        *OPENING,
        (601.0, 100.0, 0.001, -1),
        (601.1, 100.5, 0.001, -1),
        (601.15, 101.0, 0.5, 1),
        *CLOSING,
    )
    slow = run(prints)
    assert maker_fills(slow) == [(-1, 100.5, 0.01)]
    assert slow.counters["self_cross_clamped"] >= 1
    fast = run(prints, latency_ns=10_000_000)  # the cancel lands at 601.11, before the print
    assert maker_fills(fast) == []


def test_post_only_rejects_a_bid_at_or_above_the_ask_proxy_on_arrival() -> None:
    prints = (*OPENING, (601.0, 100.0, 0.001, -1), (601.02, 100.0, 0.001, 1), *CLOSING)
    result = run(prints)
    assert result.counters["rejected_post_only"] == 1
    assert maker_fills(result) == []


def staircase(steps: int = 10) -> tuple[tuple[float, float, float, int], ...]:
    """Each 100 ms a sell sets the bid proxy and a later sell trades through it."""
    prints = list(OPENING)
    for k in range(steps):
        price = 100.0 - 0.5 * k
        prints += [(601.0 + 0.1 * k, price, 0.001, -1), (601.08 + 0.1 * k, price - 0.5, 1.0, -1)]
    return (*prints, *CLOSING)


def test_the_soft_limit_is_never_exceeded_and_no_limit_taker_order_exists() -> None:
    result = run(staircase())
    assert len(maker_fills(result)) == 6
    assert result.max_abs_position == pytest.approx(0.06)
    assert result.counters.get("not_quoted_limit", 0) >= 1
    assert result.counters.get("limit_breach", 0) == 0
    taker = ~result.fill_maker
    assert taker.sum() == 1  # the 23:59 flatten, and nothing else
    assert result.fill_size[taker][0] == pytest.approx(0.06)


@pytest.mark.parametrize(("lam", "fills"), [(0.0, 6), (0.5, 6), (1.0, 3), (2.0, 2)])
def test_the_inventory_rule_withdraws_the_adding_side(lam: float, fills: int) -> None:
    # sigma_ref = 0 makes sigma / sigma_ref read as 1: the bid stops at lam |q| / Q >= 1/2.
    assert len(maker_fills(run(staircase(), cell=q.Cell(lam)))) == fills


def test_nothing_is_quoted_before_00_10_or_from_23_58() -> None:
    late = 23 * HOUR + 58 * 60
    prints = (
        *OPENING,
        (300.0, 100.0, 0.001, -1),
        (300.08, 99.0, 1.0, -1),
        (late + 10.0, 100.0, 0.001, -1),
        (late + 10.08, 99.0, 1.0, -1),
        *CLOSING,
    )
    result = run(prints)
    assert "placed" not in result.counters
    assert len(result.fill_ts) == 0
    assert result.net == 0.0


def test_funding_pays_minus_position_times_mark_times_rate_at_the_mid_proxy() -> None:
    # Long 0.01 from 601.095 to 23:59; at 08:00 the proxies are 100.5 (buy) and 99.0 (sell).
    result = run(SWEEP, rates=(5e-4, 1e-4, 0.0))
    assert result.funding == pytest.approx(-0.01 * 99.75 * 1e-4)
    assert result.counters["funding_settlements_with_position"] == 2  # 08:00 and 16:00


def test_the_flatten_closes_at_the_first_print_of_its_side_after_arrival_paying_the_taker_fee() -> (
    None
):
    result = run(SWEEP)
    taker = ~result.fill_maker
    assert result.fill_side[taker].tolist() == [-1]
    assert result.fill_price[taker][0] == 100.0
    assert result.fill_fee[taker][0] == pytest.approx(7.5e-4 * 100.0 * 0.01)
    assert result.net == pytest.approx(0.01 * 100.0 * (2.5e-4 - 7.5e-4))


def test_the_flatten_falls_back_to_the_last_print_of_its_side_when_none_follows() -> None:
    prints = (*SWEEP[:-2], (23 * HOUR + 58 * 60 + 30.0, 100.5, 0.1, 1))
    result = run(prints)
    assert result.counters["flatten_fallback"] == 1
    assert result.fill_price[~result.fill_maker][0] == 99.0  # the last seller-initiated print


def test_the_day_summary_decomposes_a_known_day() -> None:
    tape = tape_of(SWEEP)
    s = settings()
    result = engine.run_day(tape, q.S0, s, INPUTS)
    row = metrics.day_summary(tape, result, cell=q.S0, inputs=INPUTS, settings=s)
    scale = 100.0 / (0.01 * 100.25)  # the mid proxy at 00:10 is 100.25
    assert row["spread_y"] == pytest.approx(0.0015 * scale)
    assert row["making_y"] == pytest.approx(-0.0025 * scale)
    assert row["rebate_y"] == pytest.approx(2.5e-4 * 1.0 * scale)
    assert row["taker_fees_y"] == pytest.approx(7.5e-4 * 1.0 * scale)
    parts = row["making_y"] + row["rebate_y"] - row["taker_fees_y"] + row["funding_y"]
    assert row["y"] == pytest.approx(parts + row["inventory_y"])
    assert sum(row["hourly_y"]) == pytest.approx(row["y"])
    assert row["passive_fills"] == 2
    assert row["turnover_ok"] and row["identity_ok"]
    assert row["capacity"] == pytest.approx(0.01 / (0.1 + 0.001 + 0.004 * 2 + 0.5 + 0.02 + 1.0))


def test_a_repricing_at_another_maker_fee_equals_a_run_at_that_fee() -> None:
    tape = tape_of(staircase())
    rows = {}
    for maker_bp in (-2.5, 0.0, 7.5, -10.0):
        s = settings(maker_bp=maker_bp)
        result = engine.run_day(tape, q.S0, s, INPUTS)
        rows[maker_bp] = metrics.day_summary(tape, result, cell=q.S0, inputs=INPUTS, settings=s)
    assert rows[-2.5]["ex_rebate_y"] == pytest.approx(rows[0.0]["y"])
    assert rows[-2.5]["negative_control_y"] == pytest.approx(rows[7.5]["y"])
    assert rows[-2.5]["positive_control_y"] == pytest.approx(rows[-10.0]["y"])
    assert rows[-2.5]["negative_control_y"] < rows[-2.5]["y"] < rows[-2.5]["positive_control_y"]


def test_a_shifted_state_replaces_only_what_the_gate_reads() -> None:
    tape = tape_of(staircase())
    cell = q.Cell(0.0, "spread", 5.0)  # the proxy spread is 1 + k ticks at step k: open from k = 4
    assert len(maker_fills(engine.run_day(tape, cell, settings(), INPUTS))) == 6
    narrow = (np.zeros(86_401), np.full(86_401, np.nan))
    assert len(maker_fills(engine.run_day(tape, cell, settings(), INPUTS, narrow))) == 0
    wide = (np.full(86_401, 9.0), np.full(86_401, np.nan))
    assert len(maker_fills(engine.run_day(tape, cell, settings(), INPUTS, wide))) == 6
