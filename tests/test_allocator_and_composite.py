"""Tests for the daily strategy selector, the composites and the new metrics.

The selector has exactly one dangerous property: if the choice for a day can see
that day's result, it produces a beautiful equity curve on any data at all. That
is checked directly on the sequence of choices rather than inferred from a
result, because a leak of one day is invisible in any summary.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from trading_research.backtest.metrics import drawdown_bp, summarise
from trading_research.models.base import CLASSES, Model
from trading_research.models.composite import Agreement, Gated
from trading_research.pipeline.stages import build_model
from trading_research.validation.allocator import (
    STAND_ASIDE,
    AllocationError,
    AllocatorConfig,
    allocate,
    default_configs,
    per_day_results,
)


def days(n: int, start: date = date(2024, 2, 1)) -> list[date]:
    return [start + timedelta(days=i) for i in range(n)]


def daily_frame(rows: list[tuple]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["day", "strategy", "trades", "net_bp"])
    frame["net_per_trade_bp"] = frame["net_bp"] / frame["trades"]
    return frame


class Fixed(Model):
    """A member that always points one way with a set confidence."""

    def __init__(self, direction: int, confidence: float) -> None:
        super().__init__(name=f"fixed{direction:+d}@{confidence:g}")
        self.direction, self.confidence = direction, confidence

    def fit(self, x: pd.DataFrame, y: pd.Series) -> Fixed:
        self.fitted_ = True
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        proba = np.zeros((len(x), len(CLASSES)))
        column = CLASSES.index(1) if self.direction > 0 else CLASSES.index(-1)
        proba[:, column] = self.confidence
        proba[:, CLASSES.index(0)] = 1.0 - self.confidence
        return proba


@pytest.fixture
def x() -> pd.DataFrame:
    return pd.DataFrame({"anything": np.zeros(50)})


@pytest.fixture
def y() -> pd.Series:
    return pd.Series(np.zeros(50))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_drawdown_measures_peak_to_trough() -> None:
    assert drawdown_bp(np.array([5.0, 5.0, -3.0, -3.0, 2.0]))[0] == pytest.approx(6.0)


def test_a_curve_that_only_rises_has_no_drawdown() -> None:
    assert drawdown_bp(np.array([1.0, 2.0, 3.0]))[0] == 0.0


def test_drawdown_length_counts_trades_from_peak_to_trough() -> None:
    _, length = drawdown_bp(np.array([5.0, -1.0, -1.0, -1.0, 4.0]))
    assert length == 3


def test_metrics_report_trades_and_profit_per_day() -> None:
    result = summarise(np.array([2.0, -1.0, 3.0, -4.0]), days=2.0)
    assert result["trades_per_day"] == 2.0
    assert result["net_bp_per_day"] == 0.0


def test_profit_factor_compares_winners_with_losers() -> None:
    assert summarise(np.array([6.0, -2.0, -1.0]), days=1.0)["profit_factor"] == pytest.approx(2.0)


def test_no_trades_reports_zeroes_rather_than_failing() -> None:
    result = summarise(np.array([]), days=5.0)
    assert result["trades"] == 0.0
    assert result["max_drawdown_bp"] == 0.0


def test_standing_aside_for_a_week_still_counts_as_a_week() -> None:
    """Inferring the span from the trades would delete the idle days."""
    busy = summarise(np.array([1.0, 1.0]), days=2.0)
    idle = summarise(np.array([1.0, 1.0]), days=10.0)
    assert busy["trades_per_day"] > idle["trades_per_day"]


# ---------------------------------------------------------------------------
# Composites
# ---------------------------------------------------------------------------


def test_agreement_trades_only_when_members_agree(x, y) -> None:
    proba = Agreement([Fixed(1, 0.8), Fixed(1, 0.6)]).fit(x, y).predict_proba(x)
    assert proba[:, CLASSES.index(1)].max() == pytest.approx(0.6)


def test_agreement_stands_aside_when_members_disagree(x, y) -> None:
    proba = Agreement([Fixed(1, 0.9), Fixed(-1, 0.9)]).fit(x, y).predict_proba(x)
    assert proba[:, CLASSES.index(0)].min() == pytest.approx(1.0)


def test_agreement_takes_the_weakest_member_not_the_average(x, y) -> None:
    """Averaging would let one confident member drag a doubtful one over the
    threshold, which is what this class exists to prevent."""
    proba = Agreement([Fixed(1, 0.95), Fixed(1, 0.4)]).fit(x, y).predict_proba(x)
    assert proba[:, CLASSES.index(1)].max() == pytest.approx(0.4)


def test_a_gate_cannot_flip_the_direction(x, y) -> None:
    proba = Gated(Fixed(1, 0.5), Fixed(-1, 0.99)).fit(x, y).predict_proba(x)
    assert proba[:, CLASSES.index(-1)].max() == 0.0
    assert proba[:, CLASSES.index(0)].min() == pytest.approx(1.0)


def test_a_gate_supplies_the_confidence(x, y) -> None:
    proba = Gated(Fixed(1, 0.4), Fixed(1, 0.85)).fit(x, y).predict_proba(x)
    assert proba[:, CLASSES.index(1)].max() == pytest.approx(0.85)


@pytest.mark.parametrize("name", ["agree", "rule_gated_by_model", "model_gated_by_rule", "blend"])
def test_composites_are_addressable_by_name(name) -> None:
    assert build_model(name).members


def test_a_composite_needs_two_members() -> None:
    with pytest.raises(ValueError, match="at least two"):
        Agreement([Fixed(1, 0.5)])


@pytest.mark.parametrize(
    "build",
    [
        lambda: Agreement([Fixed(1, 0.7), Fixed(1, 0.6)]),
        lambda: Gated(Fixed(1, 0.7), Fixed(1, 0.6)),
    ],
)
def test_composite_probabilities_sum_to_one(build, x, y) -> None:
    assert np.allclose(build().fit(x, y).predict_proba(x).sum(axis=1), 1.0)


# ---------------------------------------------------------------------------
# Allocator
# ---------------------------------------------------------------------------


@pytest.fixture
def two_strategies() -> pd.DataFrame:
    """'early' works for the first half, 'late' for the second."""
    rows = []
    for i, day in enumerate(days(20)):
        rows.append((day, "early", 100, 200.0 if i < 10 else -200.0))
        rows.append((day, "late", 100, -200.0 if i < 10 else 200.0))
    return daily_frame(rows)


def test_the_choice_for_a_day_never_reads_that_day() -> None:
    """The one property that makes the sequence of choices meaningful."""
    rows = []
    for i, day in enumerate(days(12)):
        rows.append((day, "oracle", 100, 5000.0 if i == 11 else -500.0))
        rows.append((day, "steady", 100, 50.0))
    result = allocate(daily_frame(rows), AllocatorConfig(lookback_days=3))
    assert all(a.chosen != "oracle" for a in result.allocations)


def test_the_selector_follows_which_strategy_is_working(two_strategies) -> None:
    result = allocate(two_strategies, AllocatorConfig(lookback_days=3))
    chosen = [a.chosen for a in result.allocations]
    assert chosen[0] == "early"
    assert chosen[-1] == "late"
    assert result.switches >= 1


def test_it_stands_aside_when_nothing_clears_the_floor() -> None:
    rows = [(day, "bad", 100, -300.0) for day in days(12)]
    result = allocate(daily_frame(rows), AllocatorConfig(lookback_days=3, minimum_edge_bp=1.0))
    assert result.days_aside == len(result.allocations)
    assert all(a.chosen == STAND_ASIDE for a in result.allocations)
    assert result.metrics.get("trades", 0.0) == 0.0


def test_a_zero_floor_refuses_a_losing_strategy() -> None:
    """Zero is a floor, not an absence of one: a strategy that has been losing
    does not get run just because it is the least bad."""
    rows = [(day, "bad", 100, -300.0) for day in days(12)]
    result = allocate(daily_frame(rows), AllocatorConfig(lookback_days=3, minimum_edge_bp=0.0))
    assert result.days_aside == len(result.allocations)


def test_no_floor_forces_a_choice_every_day() -> None:
    """The control: without a floor the selector must run its least bad option,
    which is how a small edge does most of its losing."""
    rows = [(day, "bad", 100, -300.0) for day in days(12)]
    result = allocate(daily_frame(rows), AllocatorConfig(lookback_days=3, minimum_edge_bp=None))
    assert result.days_aside == 0
    assert result.metrics["net_bp"] < 0


def test_a_strategy_that_barely_traded_is_not_chosen() -> None:
    rows = []
    for day in days(12):
        rows.append((day, "lucky", 2, 400.0))
        rows.append((day, "solid", 100, 50.0))
    result = allocate(daily_frame(rows), AllocatorConfig(lookback_days=3, minimum_trades=20))
    assert all(a.chosen == "solid" for a in result.allocations)


def test_metrics_include_the_things_a_desk_asks_for(two_strategies) -> None:
    result = allocate(two_strategies, AllocatorConfig(lookback_days=3))
    for key in ("trades_per_day", "max_drawdown_bp", "net_bp", "net_per_trade_bp"):
        assert key in result.metrics


def test_too_few_days_for_the_lookback_is_refused() -> None:
    rows = [(day, "a", 100, 10.0) for day in days(3)]
    with pytest.raises(AllocationError, match="lookback"):
        allocate(daily_frame(rows), AllocatorConfig(lookback_days=5))


def test_missing_columns_are_named() -> None:
    with pytest.raises(AllocationError, match="net_bp"):
        allocate(pd.DataFrame({"day": days(9), "strategy": "a", "trades": 1}))


def test_per_day_results_collapses_trades_into_days() -> None:
    two = days(2)
    trades = pd.DataFrame(
        {"day": [two[0]] * 3 + [two[1]] * 2, "net_bp": [1.0, -2.0, 3.0, 4.0, -1.0]}
    )
    collapsed = per_day_results(trades, "demo")
    assert list(collapsed["trades"]) == [3, 2]
    assert collapsed["net_bp"].tolist() == [2.0, 3.0]


def test_the_default_configs_include_the_no_floor_control() -> None:
    assert {c.minimum_edge_bp for c in default_configs()} == {None, 0.0, 1.0}
