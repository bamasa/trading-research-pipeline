"""The print rule's brackets where a review found them weaker than their names.

Each test is a case a review built to show a fault; each now asserts the
registered rule: the part beyond a sweep at the sweep's worst price, an exit's
pessimistic window to the day's end, worse of two as the worst of the rule's,
the pessimistic and the stale price (and, for an exit priced without a later
print, of the other side's last print), and no bracket price for an order the
rule could not fill.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from test_spot_synthetic_prints import no_network
from trading_research.market_making.events import NS_PER_S, day_start_ns
from trading_research.spot.calibration import k_fill_charges, price_orders, side_summary
from trading_research.spot.costs import FeeTable
from trading_research.spot.fills import PrintTape, fill_orders, round_trips
from trading_research.spot.print_taker import execute_print_day, print_settings

DAY = date(2026, 8, 10)
T0 = day_start_ns(DAY)
MS = 1_000_000
S = NS_PER_S
CLIP = 0.010
TABLE = FeeTable.from_config(
    {
        "verdict_tier": {"binance_spot_BTCFDUSD": "regular", "bybit_spot_BTCUSDT": "non_vip"},
        "intervals": {
            "binance_spot_BTCFDUSD": [
                {"start": "2026-01-29T00:00:00Z", "end": "2026-10-22T00:00:00Z",
                 "sources": ["B4"], "tiers": {"regular": [0.0, 10.0]}},
            ],
            "bybit_spot_BTCUSDT": [
                {"start": "2025-01-31T00:00:00Z", "end": "2026-10-22T00:00:00Z",
                 "sources": ["Y1"], "tiers": {"non_vip": [10.0, 10.0]}},
            ],
        },
    }
)  # fmt: skip


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    no_network(monkeypatch)


def tape(rows: list[tuple[float, float, float, int, bool]]) -> PrintTape:
    """(seconds after midnight, price, size, side, rpi) per print, archive order."""
    ts = np.array([T0 + round(r[0] * S) for r in rows], dtype=np.int64)
    return PrintTape(DAY, ts, np.array([r[1] for r in rows]), np.array([r[2] for r in rows]),
                     np.array([r[3] for r in rows]), np.array([r[4] for r in rows]))  # fmt: skip


def run(p: PrintTape, decision: np.ndarray, side: np.ndarray, hold_s: int, bracket: str):
    grid = np.unique(p.ts)
    settings = print_settings(TABLE, "binance-spot:BTCFDUSD", bracket=bracket)
    return execute_print_day(p, grid, decision, side, hold_s * S, settings)


def test_an_unfilled_entry_has_no_stale_price_and_leaves_fill_s() -> None:
    p = tape([(3600.0 + k, 100.0, 0.05, 1, False) for k in range(100)]
             + [(23 * 3600 + 58 * 60 + 30.0, 100.0, 0.05, 1, False)])  # fmt: skip
    late = T0 + (23 * 3600 + 58 * 60 + 40) * S
    orders = pd.DataFrame({
        "kind": ["entry", "entry"], "side": [1, 1], "size": [CLIP, CLIP],
        "decision_ns": [T0 + 3650 * S, late], "arrival_ns": [T0 + 3650 * S + 10 * MS, late + 10 * MS],
        "book_price": [100.5, 100.5], "book_mid": [100.25, 100.25],
    })  # fmt: skip
    priced = price_orders(p, orders)
    assert not priced["found"].iloc[1]
    for bracket in ("rule", "stale", "worse_of_two"):
        assert np.isnan(priced[bracket].iloc[1])
        assert side_summary(priced, bracket).loc["buy", "orders"] == 1
    assert k_fill_charges([priced])["a_buy"] == priced["adv_worse_of_two_bp"].iloc[0]


def test_a_no_print_exit_before_the_entry_is_held_to_the_other_sides_last_print() -> None:
    """A long entered at 23:58 after a fall: the rule exits at the morning's seller
    print (110), worse of two at the last buyer print at or before the arrival (100)."""
    rows = [(3600.0, 110.0, 0.05, -1, False)]
    rows += [(23 * 3600 + 58 * 60 + k * 0.5, 100.0, 0.05, 1, False) for k in range(200)]
    p = tape(rows)
    decision = np.array([np.unique(p.ts)[1]])
    rule = run(p, decision, np.array([1]), 50, "rule").trades.iloc[0]
    assert rule.exit_no_print and rule.exit_price == 110.0
    worse = run(p, decision, np.array([1]), 50, "worse_of_two").trades.iloc[0]
    assert worse.exit_price == 100.0 and worse.net_bp < 0


def test_an_either_side_exit_pays_no_spread_but_gains_nothing() -> None:
    """No seller print all day: the fallback and the other side's last print are the
    same buyer print, so worse of two exits a long at its entry price."""
    rows = [(60.0 + k, 100.0, 0.02, 1, False) for k in range(200)]
    rows += [(90.0 + k, 99.0, 0.02, -1, True) for k in range(5)]  # sellers, rpi only
    rows.sort(key=lambda r: r[0])
    p = tape(rows)
    decision = np.array([T0 + 61 * S])
    day = run(p, decision, np.array([1]), 50, "worse_of_two")
    assert day.counters["exits_no_print_either_side"] == 1
    assert day.trades.iloc[0].exit_price == day.trades.iloc[0].entry_price == 100.0


def test_worse_of_two_is_never_better_than_the_rule() -> None:
    rows = [
        (100.0, 99.0, 0.05, 1, False),  # stale print, better than the rule
        (200.0, 100.0, 0.001, 1, False),
        (200.0, 105.0, 0.001, 1, False),  # the sweep's worst: 105
        (210.0, 101.0, 0.05, 1, False),  # the window's worst (only print): 101
    ]
    p = tape(rows)
    d = np.array([T0 + 150 * S])
    f = fill_orders(p, 1, d + 10 * MS, d, until_ns=p.end_ns)
    assert f.rule[0] == pytest.approx((0.1 + 0.105 + 0.008 * 105) / CLIP)
    assert f.pessimistic[0] < f.rule[0] == f.worse_of_two[0]


def test_the_beyond_part_is_charged_at_the_sweeps_worst() -> None:
    """Two prints in one millisecond out of price order: the beyond part at 101."""
    t = 7200.0
    p = tape([(t, 101.0, 0.001, 1, False), (t, 100.0, 0.001, 1, False)])
    d = np.array([T0 + round((t - 1) * S)])
    f = fill_orders(p, 1, d + 10 * MS, d, until_ns=p.end_ns)
    assert f.rule[0] == pytest.approx((0.101 + 0.100 + 0.008 * 101) / CLIP)
    # The optimistic bracket is, as registered, the whole clip at the first print's
    # price: with same-stamp prints out of price order it can be the worse one.
    assert f.optimistic[0] == 101.0 > f.rule[0]


def test_a_flatten_exit_reads_its_window_to_the_day_end() -> None:
    rows = [(23 * 3600 + 58 * 60.0, 100.0, 0.05, 1, False),
            (23 * 3600 + 58 * 60 + 10.0, 100.0, 0.05, 1, False),
            (23 * 3600 + 59 * 60 + 1.0, 99.0, 0.001, -1, False)]  # fmt: skip
    rows += [(23 * 3600 + 59 * 60 + 5.0 + k, 90.0, 0.05, -1, False) for k in range(10)]
    p = tape(rows)
    grid = np.unique(p.ts)
    trips = round_trips(p, grid, np.array([grid[0]]), 1, 120 * S)
    assert trips.exit_reason[0] == "day_end" and trips.exit.beyond[0] > 0
    assert trips.exit.rule[0] == pytest.approx(99.0)
    assert trips.exit.pessimistic[0] == pytest.approx((0.001 * 99.0 + 0.009 * 90.0) / CLIP)
    assert trips.exit.worse_of_two[0] == trips.exit.pessimistic[0]


def test_fallback_ages_are_reported_apart_from_the_waits() -> None:
    rows = [(3600.0, 110.0, 0.05, -1, False)]
    rows += [(23 * 3600 + 58 * 60 + k * 0.5, 100.0, 0.05, 1, False) for k in range(200)]
    p = tape(rows)
    day = run(p, np.array([np.unique(p.ts)[1]]), np.array([1]), 50, "rule")
    trade = day.trades.iloc[0]
    assert trade.exit_fallback_age_ns > 22 * 3600 * S
    assert trade.exit_wait_ns == -trade.exit_fallback_age_ns
