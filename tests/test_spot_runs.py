"""The shared pieces of a row's day: decisions, draws, IC, tiers and the shift's rows."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd

from test_spot_day_fixture import TABLE
from trading_research.spot import access as sa
from trading_research.spot.runs import (
    decisions,
    hour_flips,
    random_sides,
    rank_ic,
    shift_rows,
    signal,
    table_bp,
    tier_values,
)
from trading_research.strategies.hold_first import HoldFirstRule

DAY = date(2026, 6, 3)


def test_decisions_skip_unscored_rows_and_follow_the_rule() -> None:
    proba = np.array([[0.1, 0.2, 0.7], [0.7, 0.2, 0.1], [np.nan] * 3, [0.3, 0.4, 0.3]])
    rows, side = decisions(proba, HoldFirstRule(0.5, 0.5))
    assert rows.tolist() == [0, 1] and side.tolist() == [1, -1]
    assert np.isnan(signal(proba)[2]) and signal(proba)[0] > 0


def test_draws_are_fixed_by_seed_and_day_alone() -> None:
    a, b = random_sides(50, 3, DAY), random_sides(50, 3, DAY)
    assert np.array_equal(a, b) and not np.array_equal(a, random_sides(50, 4, DAY))
    start = 1_000
    flips = hour_flips(np.array([start, start + 10, start + 3_600_000_000_000]), start, 5, DAY)
    assert flips[0] == flips[1] and set(flips.tolist()) <= {-1, 1}


def test_rank_ic_is_undefined_without_rows_or_spread() -> None:
    assert math.isnan(rank_ic(np.array([1.0, 2.0]), np.array([1.0, 2.0])))
    assert math.isnan(rank_ic(np.ones(5), np.arange(5.0)))
    assert abs(rank_ic(np.arange(5.0), np.arange(5.0)) - 1.0) < 1e-12


def test_a_day_without_a_trade_has_no_value() -> None:
    class NoTrade:
        trades = 0

        def day_values(self) -> dict[str, float]:
            return {"net_bp": 0.0}

    assert math.isnan(table_bp(NoTrade()))


def test_every_tier_is_priced_and_a_tier_without_a_source_is_refused() -> None:
    trades = pd.DataFrame({"size": [0.01], "entry_price": [100.0], "exit_price": [101.0],
                           "gross": [0.01], "notional": [1.0]})  # fmt: skip
    found = tier_values(trades, TABLE, sa.BINANCE, lambda _: (np.array([10.0]), np.array([10.0])))
    assert found["zero_fees"] == 100.0 and abs(found["regular"] - (0.01 - 0.00201) * 1e4) < 1e-9

    def unsourced(tier: str) -> tuple[np.ndarray, np.ndarray]:
        from trading_research.spot.costs import FeeNotSourced

        raise FeeNotSourced(tier)

    assert all(math.isnan(v) for k, v in tier_values(trades, TABLE, sa.BINANCE, unsourced).items()
               if k != "zero_fees")  # fmt: skip
    assert math.isnan(tier_values(trades.iloc[:0], TABLE, sa.BINANCE, unsourced)["regular"])


def test_a_shifted_decision_takes_the_first_row_at_or_after_its_time_of_day() -> None:
    grid = np.array([10, 20, 30])
    assert shift_rows(grid, 0, np.array([5, 20, 31])).tolist() == [0, 1, 3]
