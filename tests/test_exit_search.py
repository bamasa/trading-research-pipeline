"""Tests for the exit rules that read a path or a signal, and for their search.

Exit rules are tested against hand-built price and probability paths because
every one of them is about *when* something happens, and an off-by-one changes
the outcome of every trade while leaving the summary looking reasonable.

The search tests pin the property that makes the result mean anything: the
policy is chosen on validation and the test block is scored once, with the
winner and nothing else.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_research.backtest.costs import TakerCosts
from trading_research.backtest.execution import ThinningRules, score, thin
from trading_research.labels.directional import BUY, HOLD, SELL
from trading_research.pipeline.exits import PROBA_COLUMNS, policy_evaluator
from trading_research.validation.exits import (
    ExitPolicy,
    ExitSearchError,
    default_grid,
    expand_grid,
    search,
)

COSTS = TakerCosts(fee_bp_per_side=5.0, slippage_bp=0.5)


def path(prices: list[float]) -> np.ndarray:
    return np.array(prices, dtype="float64")


def constant(n: int, value: float) -> np.ndarray:
    return np.full(n, value)


# ---------------------------------------------------------------------------
# Trailing stop
# ---------------------------------------------------------------------------


def test_trailing_stop_closes_after_giving_back_the_gap() -> None:
    # Rises to +200 bp, then falls back 100. A 50 bp trail closes on the way
    # down, not at the peak, and not at entry.
    mid = path([100.0, 101.0, 102.0, 101.5, 101.0])
    trades = thin(
        np.array([BUY, HOLD, HOLD, HOLD, HOLD]),
        constant(5, 100.0),
        constant(5, 2.0),
        ThinningRules(hold_periods=4, trailing_stop_bp=50.0),
        mid=mid,
    )
    assert trades[0].exit_reason == "trailing_stop"
    assert trades[0].exit_index == 3
    assert trades[0].move_bp == pytest.approx(150.0, abs=1.0)


def test_a_trailing_stop_does_not_fire_before_the_position_is_ahead() -> None:
    """Armed from entry it would just be a stop-loss under another name."""
    mid = path([100.0, 99.0, 98.0])
    trades = thin(
        np.array([BUY, HOLD, HOLD]),
        constant(3, -200.0),
        constant(3, 2.0),
        ThinningRules(hold_periods=2, trailing_stop_bp=10.0),
        mid=mid,
    )
    assert trades[0].exit_reason == "clock"


def test_a_trailing_stop_leaves_a_rising_position_alone() -> None:
    mid = path([100.0, 101.0, 102.0, 103.0])
    trades = thin(
        np.array([BUY, HOLD, HOLD, HOLD]),
        constant(4, 300.0),
        constant(4, 2.0),
        ThinningRules(hold_periods=3, trailing_stop_bp=50.0),
        mid=mid,
    )
    assert trades[0].exit_reason == "clock"
    assert trades[0].exit_index == 3


def test_the_trailing_high_water_mark_resets_between_trades() -> None:
    """A peak carried from a previous trade would close the next one instantly."""
    mid = path([100.0, 105.0, 100.0, 100.0, 100.1, 100.2])
    decision = np.array([BUY, HOLD, HOLD, BUY, HOLD, HOLD])
    trades = thin(
        decision,
        constant(6, 10.0),
        constant(6, 2.0),
        ThinningRules(hold_periods=2, trailing_stop_bp=100.0),
        mid=mid,
    )
    assert len(trades) == 2
    assert trades[1].exit_reason != "trailing_stop"


# ---------------------------------------------------------------------------
# Signal exits
# ---------------------------------------------------------------------------


def test_a_position_closes_when_the_model_stops_believing_it() -> None:
    p_buy = np.array([0.8, 0.7, 0.2, 0.2])
    p_sell = np.array([0.05, 0.05, 0.05, 0.05])
    trades = thin(
        np.array([BUY, HOLD, HOLD, HOLD]),
        constant(4, 30.0),
        constant(4, 2.0),
        ThinningRules(hold_periods=3, exit_below_confidence=0.5),
        mid=path([100.0, 100.1, 100.2, 100.3]),
        p_buy=p_buy,
        p_sell=p_sell,
    )
    assert trades[0].exit_reason == "signal_decay"
    assert trades[0].exit_index == 2


def test_a_position_closes_when_the_model_flips() -> None:
    p_buy = np.array([0.8, 0.6, 0.2])
    p_sell = np.array([0.05, 0.1, 0.7])
    trades = thin(
        np.array([BUY, HOLD, HOLD]),
        constant(3, 30.0),
        constant(3, 2.0),
        ThinningRules(hold_periods=2, exit_on_flip=True),
        mid=path([100.0, 100.1, 100.2]),
        p_buy=p_buy,
        p_sell=p_sell,
    )
    assert trades[0].exit_reason == "flip"
    assert trades[0].exit_index == 2


def test_a_flip_exit_goes_flat_rather_than_reversing() -> None:
    """Distinct from allow_reversal, which pays immediately to enter the other side."""
    p_buy = np.array([0.8, 0.1, 0.1])
    p_sell = np.array([0.05, 0.8, 0.8])
    trades = thin(
        np.array([BUY, SELL, SELL]),
        constant(3, 30.0),
        constant(3, 2.0),
        ThinningRules(hold_periods=2, cooldown_periods=5, exit_on_flip=True),
        mid=path([100.0, 100.1, 100.2]),
        p_buy=p_buy,
        p_sell=p_sell,
    )
    assert len(trades) == 1
    assert trades[0].exit_reason == "flip"


def test_a_short_reads_its_own_side_of_the_book() -> None:
    """A long's confidence is p_buy; a short's is p_sell. Swapping them would
    close every short immediately and look like a working rule."""
    p_buy = np.array([0.05, 0.05, 0.05])
    p_sell = np.array([0.8, 0.7, 0.6])
    trades = thin(
        np.array([SELL, HOLD, HOLD]),
        constant(3, -30.0),
        constant(3, 2.0),
        ThinningRules(hold_periods=2, exit_below_confidence=0.5),
        mid=path([100.0, 99.9, 99.8]),
        p_buy=p_buy,
        p_sell=p_sell,
    )
    assert trades[0].exit_reason == "clock"


def test_a_signal_exit_without_probabilities_is_refused() -> None:
    """Silently ignoring it would report a backtest that never applied the rule."""
    with pytest.raises(ValueError, match="model's opinion"):
        thin(
            np.array([BUY, HOLD]),
            constant(2, 10.0),
            constant(2, 2.0),
            ThinningRules(hold_periods=1, exit_on_flip=True),
            mid=path([100.0, 100.1]),
        )


def test_an_impossible_confidence_threshold_is_rejected() -> None:
    with pytest.raises(ValueError, match="exit_below_confidence"):
        ThinningRules(hold_periods=1, exit_below_confidence=1.5)
    with pytest.raises(ValueError, match="trailing_stop_bp"):
        ThinningRules(hold_periods=1, trailing_stop_bp=0.0)


def test_the_new_rules_change_nothing_when_unset() -> None:
    """Adding options must not alter results for anyone not using them."""
    decision = np.array([BUY, HOLD, HOLD, SELL, HOLD, HOLD])
    forward, spread = constant(6, 20.0), constant(6, 2.0)
    rules = ThinningRules(hold_periods=2, cooldown_periods=1)
    without = thin(decision, forward, spread, rules)
    with_path = thin(decision, forward, spread, rules, mid=path([100.0] * 6))
    assert [t.move_bp for t in without] == [t.move_bp for t in with_path]
    assert [t.exit_reason for t in without] == [t.exit_reason for t in with_path]


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


def test_a_policy_label_names_only_what_is_switched_on() -> None:
    assert ExitPolicy(hold_periods=24).label == "hold24"
    assert ExitPolicy(hold_periods=24, take_profit_bp=11.0).label == "hold24/tp11"
    assert ExitPolicy(hold_periods=24, exit_on_flip=True).label == "hold24/flip"


def test_a_policy_reports_which_rule_it_is_testing() -> None:
    assert ExitPolicy(hold_periods=24).family == "clock only"
    assert ExitPolicy(hold_periods=24, trailing_stop_bp=5.0).family == "trailing stop"
    assert (
        ExitPolicy(hold_periods=24, take_profit_bp=5.0, stop_loss_bp=5.0).family
        == "take-profit + stop-loss"
    )


def test_the_default_grid_covers_every_rule() -> None:
    families = {p.family for p in default_grid(24)}
    assert families == {
        "clock only",
        "take-profit",
        "stop-loss",
        "trailing stop",
        "take-profit + stop-loss",
        "confidence decay",
        "prediction flip",
    }


def test_the_default_grid_includes_the_untouched_baseline() -> None:
    """Without it, the table cannot say whether any rule helped."""
    assert ExitPolicy(hold_periods=24) in default_grid(24)


def test_expand_grid_builds_the_cross_product() -> None:
    grid = expand_grid([12, 24], take_profit_bp=[None, 11.0])
    assert len(grid) == 4


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def test_the_best_policy_on_validation_is_the_one_chosen() -> None:
    def favour_take_profit(policy: ExitPolicy) -> dict[str, float]:
        return {
            "net_per_trade_bp": 5.0 if policy.take_profit_bp else -1.0,
            "trades": 500.0,
        }

    outcome = search(favour_take_profit, grid=default_grid(24))
    assert outcome.chosen.take_profit_bp is not None


def test_a_policy_that_barely_traded_is_not_eligible() -> None:
    """Every rule here can be tightened until it selects a handful of trades."""

    def rewards_inactivity(policy: ExitPolicy) -> dict[str, float]:
        if policy.take_profit_bp == 4.0:
            return {"net_per_trade_bp": 99.0, "trades": 3.0}
        return {"net_per_trade_bp": 1.0, "trades": 500.0}

    outcome = search(rewards_inactivity, grid=default_grid(24), minimum_trades=50)
    assert outcome.chosen.take_profit_bp != 4.0
    row = outcome.validation.loc[outcome.validation["policy"] == "hold24/tp4"]
    assert "fewer than 50 trades" in str(row["skipped"].iloc[0])


def test_the_test_block_is_scored_once_with_the_winner() -> None:
    seen: list[ExitPolicy] = []

    def on_validation(policy: ExitPolicy) -> dict[str, float]:
        return {"net_per_trade_bp": float(policy.hold_periods), "trades": 500.0}

    def on_test(policy: ExitPolicy) -> dict[str, float]:
        seen.append(policy)
        return {"net_per_trade_bp": -2.0, "trades": 400.0}

    outcome = search(on_validation, grid=default_grid(24), on_test=on_test)
    assert seen == [outcome.chosen]
    assert outcome.test is not None
    assert "test net_per_trade_bp = -2.000" in outcome.summary()


def test_a_policy_that_raises_is_recorded_not_fatal() -> None:
    def flaky(policy: ExitPolicy) -> dict[str, float]:
        if policy.exit_on_flip:
            raise RuntimeError("no probabilities")
        return {"net_per_trade_bp": 1.0, "trades": 500.0}

    outcome = search(flaky, grid=default_grid(24))
    skipped = outcome.validation["skipped"].dropna()
    assert any("no probabilities" in str(s) for s in skipped)


def test_a_grid_where_nothing_scores_is_an_error() -> None:
    def never_trades(policy: ExitPolicy) -> dict[str, float]:
        return {"net_per_trade_bp": float("nan"), "trades": 0.0}

    with pytest.raises(ExitSearchError, match="net_per_trade_bp"):
        search(never_trades, grid=default_grid(24))


# ---------------------------------------------------------------------------
# The evaluator built from predictions
# ---------------------------------------------------------------------------


@pytest.fixture
def predictions() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 600
    p_buy = rng.uniform(0.2, 0.9, n)
    frame = pd.DataFrame(
        {
            "fold": np.repeat([0, 1], n // 2),
            "forward_bp": rng.normal(0, 20, n),
            "spread_bp_now": np.full(n, 2.0),
            "mid": 100 * np.exp(np.cumsum(rng.normal(0, 1e-4, n))),
        }
    )
    frame[PROBA_COLUMNS[0]] = 1 - p_buy - 0.05
    frame[PROBA_COLUMNS[1]] = 0.05
    frame[PROBA_COLUMNS[2]] = p_buy
    return frame


def test_the_evaluator_scores_a_policy_end_to_end(predictions) -> None:
    evaluate = policy_evaluator(predictions, COSTS, min_confidence=0.4)
    result = evaluate(ExitPolicy(hold_periods=5, take_profit_bp=10.0))
    assert result["trades"] > 0
    assert "share_take_profit" in result or "share_clock" in result


def test_positions_do_not_survive_a_fold_boundary(predictions) -> None:
    """A trade opened at the end of one fold and closed in the next is not a
    trade anyone could have taken."""
    evaluate = policy_evaluator(predictions, COSTS, min_confidence=0.4)
    both = evaluate(ExitPolicy(hold_periods=5))

    one_fold = predictions[predictions["fold"] == 0].reset_index(drop=True)
    first = policy_evaluator(one_fold, COSTS, min_confidence=0.4)(ExitPolicy(hold_periods=5))
    assert both["trades"] > first["trades"]


def test_exit_reason_shares_are_reported(predictions) -> None:
    """A take-profit that never fires is the baseline under another name."""
    evaluate = policy_evaluator(predictions, COSTS, min_confidence=0.4)
    result = evaluate(ExitPolicy(hold_periods=5, take_profit_bp=1e6))
    assert result.get("share_clock", 0.0) + result.get("share_end_of_data", 0.0) == pytest.approx(
        1.0
    )


def test_the_baseline_is_scored_on_test_beside_the_winner() -> None:
    """Without it the winner's test figure has nothing to be read against."""
    seen: list[str] = []

    def on_validation(policy: ExitPolicy) -> dict[str, float]:
        return {"net_per_trade_bp": 9.0 if policy.stop_loss_bp else 1.0, "trades": 500.0}

    def on_test(policy: ExitPolicy) -> dict[str, float]:
        seen.append(policy.label)
        return {"net_per_trade_bp": -3.0 if policy.stop_loss_bp else -6.0, "trades": 400.0}

    grid = default_grid(24)
    outcome = search(on_validation, grid=grid, baseline=grid[0], on_test=on_test)
    assert outcome.test_baseline is not None
    assert seen == [outcome.chosen.label, grid[0].label]
    assert "against -6.000 for the baseline" in outcome.summary()


def test_the_baseline_is_not_scored_twice_when_it_wins() -> None:
    def prefers_the_baseline(policy: ExitPolicy) -> dict[str, float]:
        return {"net_per_trade_bp": 9.0 if policy.family == "clock only" else 1.0, "trades": 500.0}

    calls = []
    grid = default_grid(24)
    outcome = search(
        prefers_the_baseline,
        grid=grid,
        baseline=grid[0],
        on_test=lambda p: (calls.append(p), {"net_per_trade_bp": 0.0, "trades": 9.0})[1],
    )
    assert outcome.chosen == grid[0]
    assert len(calls) == 1
    assert outcome.test_baseline is None


# ---------------------------------------------------------------------------
# The sign of a short's profit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rules",
    [
        ThinningRules(hold_periods=2),
        ThinningRules(hold_periods=2, take_profit_bp=50.0),
        ThinningRules(hold_periods=2, stop_loss_bp=500.0),
        ThinningRules(hold_periods=2, trailing_stop_bp=30.0),
    ],
    ids=["clock", "take_profit", "stop_loss", "trailing"],
)
def test_a_profitable_short_is_profitable_however_it_exits(rules) -> None:
    """The regression test for a real bug, and the shape of it is worth keeping.

    `Trade.move_bp` is the price change; `score` applies the direction. An
    earlier version stored a value that *already* had the direction applied on
    every path exit, so it was applied twice — inverting the profit of every
    short that exited early, while leaving every long correct. Totals stayed
    plausible and the error was invisible in any summary.
    """
    free = TakerCosts(fee_bp_per_side=0.0, slippage_bp=0.0, half_spread_multiplier=0.0)
    mid = path([100.0, 99.0, 98.0])  # falling: a short is in profit throughout
    trades = thin(
        np.array([SELL, HOLD, HOLD]),
        np.array([-200.0, -100.0, 0.0]),
        constant(3, 0.0),
        rules,
        mid=mid,
    )
    assert score(trades, free)["gross_bp"] > 0


def test_a_losing_long_and_a_winning_short_are_not_the_same_trade() -> None:
    """Same price path, opposite positions, opposite outcomes."""
    free = TakerCosts(fee_bp_per_side=0.0, slippage_bp=0.0, half_spread_multiplier=0.0)
    mid = path([100.0, 99.0, 98.0])
    rules = ThinningRules(hold_periods=2, take_profit_bp=50.0, stop_loss_bp=50.0)

    short = thin(np.array([SELL, HOLD, HOLD]), np.full(3, -200.0), constant(3, 0.0), rules, mid=mid)
    long = thin(np.array([BUY, HOLD, HOLD]), np.full(3, -200.0), constant(3, 0.0), rules, mid=mid)

    assert score(short, free)["gross_bp"] > 0
    assert score(long, free)["gross_bp"] < 0
