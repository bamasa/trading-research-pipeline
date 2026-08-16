"""Tests for the cost model, the decision rule and the baselines.

The cost tests matter more than they look. A round trip that charges the spread
twice overstates costs about as badly as one that ignores it understates them,
and on this horizon the whole result turns on that arithmetic.

The decision-rule tests pin down one rule in particular: a model may be
confident that the market will move while being nearly indifferent about which
way. Acting on that is a coin flip that still pays full costs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_research.backtest.costs import BINANCE_UM_TAKER_BP, TakerCosts, breakeven_share
from trading_research.backtest.evaluate import choose_confidence, decide, evaluate
from trading_research.labels.directional import BUY, HOLD, SELL
from trading_research.models.base import CLASSES, AlwaysHold, ClassPrior, clean
from trading_research.models.linear import LogisticBaseline

COSTS = TakerCosts(fee_bp_per_side=5.0, slippage_bp=0.5)


# ---------------------------------------------------------------------------
# Costs
# ---------------------------------------------------------------------------


def test_published_taker_rate_is_five_basis_points() -> None:
    """Binance USD-M standard taker fee is 0.05%."""
    assert BINANCE_UM_TAKER_BP == 5.0


def test_round_trip_charges_two_fees_one_spread_two_slippages() -> None:
    # 2*5 fees + 1*2 spread + 2*0.5 slippage
    assert COSTS.round_trip_bp(2.0) == pytest.approx(13.0)


def test_spread_is_charged_once_not_twice() -> None:
    """Entry pays half against mid, exit pays the other half."""
    wide, narrow = COSTS.round_trip_bp(4.0), COSTS.round_trip_bp(0.0)
    assert wide - narrow == pytest.approx(4.0)


def test_entry_leg_is_half_the_spread_plus_one_fee() -> None:
    assert COSTS.entry_bp(2.0) == pytest.approx(5.0 + 1.0 + 0.5)


def test_two_entry_legs_equal_one_round_trip() -> None:
    """The two accounting routes must agree, or PnL depends on which is used."""
    assert 2 * COSTS.entry_bp(2.0) == pytest.approx(COSTS.round_trip_bp(2.0))


def test_costs_scale_with_a_series_of_spreads() -> None:
    spreads = pd.Series([0.0, 2.0, 4.0])
    got = COSTS.round_trip_bp(spreads)
    assert list(got) == pytest.approx([11.0, 13.0, 15.0])


def test_negative_cost_parameters_are_rejected() -> None:
    with pytest.raises(ValueError, match="fee_bp_per_side"):
        TakerCosts(fee_bp_per_side=-1.0)


def test_breakeven_share_is_an_upper_bound_under_perfect_foresight() -> None:
    move = pd.Series([1.0, -20.0, 30.0, -2.0])
    assert breakeven_share(move, 11.0) == pytest.approx(0.5)


def test_cost_model_describes_itself_as_taker() -> None:
    """The manifest must not leave the execution assumption implicit."""
    described = COSTS.describe()
    assert described["execution"] == "taker"
    assert "adverse selection" in str(described["note"])


# ---------------------------------------------------------------------------
# Decision rule
# ---------------------------------------------------------------------------


def proba(sell: float, hold: float, buy: float) -> np.ndarray:
    row = {SELL: sell, HOLD: hold, BUY: buy}
    return np.array([[row[c] for c in CLASSES]])


def test_confident_buy_is_taken() -> None:
    assert decide(proba(0.1, 0.2, 0.7), min_confidence=0.5)[0] == BUY


def test_confident_sell_is_taken() -> None:
    assert decide(proba(0.7, 0.2, 0.1), min_confidence=0.5)[0] == SELL


def test_below_the_threshold_nothing_is_traded() -> None:
    assert decide(proba(0.1, 0.5, 0.4), min_confidence=0.5)[0] == HOLD


def test_a_move_of_unknown_direction_is_not_traded() -> None:
    """Confident the market moves, indifferent which way: a coin flip at full cost."""
    assert decide(proba(0.45, 0.10, 0.45), min_confidence=0.4)[0] == HOLD


def test_threshold_of_one_never_trades() -> None:
    assert decide(proba(0.4, 0.2, 0.4), min_confidence=1.01)[0] == HOLD


def test_wrong_shaped_probabilities_are_rejected() -> None:
    with pytest.raises(ValueError, match="probability columns"):
        decide(np.zeros((3, 2)), min_confidence=0.5)


# ---------------------------------------------------------------------------
# Accounting
# ---------------------------------------------------------------------------


def test_profit_is_direction_times_move_minus_cost() -> None:
    decision = np.array([BUY])
    move = pd.Series([20.0])
    spread = pd.Series([2.0])
    result = evaluate(decision, move, spread, COSTS)
    assert result.gross_bp == pytest.approx(20.0)
    assert result.cost_bp == pytest.approx(13.0)
    assert result.net_bp == pytest.approx(7.0)


def test_a_correct_short_earns_on_a_falling_price() -> None:
    result = evaluate(np.array([SELL]), pd.Series([-20.0]), pd.Series([2.0]), COSTS)
    assert result.gross_bp == pytest.approx(20.0)


def test_a_right_call_can_still_lose_money() -> None:
    """The point of the whole cost model, in one test."""
    result = evaluate(np.array([BUY]), pd.Series([5.0]), pd.Series([2.0]), COSTS)
    assert result.gross_bp > 0
    assert result.net_bp < 0
    assert result.hit_rate == pytest.approx(1.0)


def test_holding_costs_nothing_and_earns_nothing() -> None:
    result = evaluate(
        np.array([HOLD, HOLD]), pd.Series([50.0, -50.0]), pd.Series([2.0, 2.0]), COSTS
    )
    assert result.n_trades == 0
    assert result.net_bp == 0.0


def test_rows_without_an_outcome_are_not_traded() -> None:
    """A decision at the very end of the sample has no outcome to score."""
    result = evaluate(
        np.array([BUY, BUY]),
        pd.Series([20.0, np.nan]),
        pd.Series([2.0, 2.0]),
        COSTS,
    )
    assert result.n_trades == 1


def test_hit_rate_counts_gross_wins_not_net_ones() -> None:
    """Kept separate so a losing-but-accurate model is visible as such."""
    result = evaluate(
        np.array([BUY, BUY]),
        pd.Series([1.0, -1.0]),
        pd.Series([2.0, 2.0]),
        COSTS,
    )
    assert result.hit_rate == pytest.approx(0.5)
    assert result.net_bp < 0


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------


def test_threshold_is_chosen_on_the_data_it_is_given() -> None:
    rng = np.random.default_rng(0)
    n = 2000
    p_buy = rng.uniform(0.2, 0.9, n)
    matrix = np.column_stack([1 - p_buy - 0.1, np.full(n, 0.1), p_buy])
    # Confident predictions are right; unconfident ones are noise.
    move = np.where(p_buy > 0.6, 30.0, rng.normal(0, 5, n))

    chosen, curve = choose_confidence(matrix, pd.Series(move), pd.Series(np.full(n, 2.0)), COSTS)
    assert chosen >= 0.5
    assert len(curve) > 5
    assert {"min_confidence", "net_bp", "trades"} <= set(curve.columns)


def test_a_threshold_that_never_trades_enough_is_not_chosen() -> None:
    """With too few trades the average is one lucky move, not an edge."""
    n = 500
    matrix = np.column_stack([np.full(n, 0.33), np.full(n, 0.34), np.full(n, 0.33)])
    chosen, _ = choose_confidence(
        matrix, pd.Series(np.zeros(n)), pd.Series(np.full(n, 2.0)), COSTS, min_trades=10_000
    )
    # Nothing qualified, so the rule stands aside rather than picking noise.
    assert chosen > 1.0


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


@pytest.fixture
def toy() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(1)
    x = pd.DataFrame({"a": rng.normal(size=400), "b": rng.normal(size=400)})
    y = pd.Series(rng.choice([SELL, HOLD, BUY], size=400, p=[0.1, 0.8, 0.1]), dtype="float64")
    return x, y


def test_always_hold_never_trades(toy: tuple[pd.DataFrame, pd.Series]) -> None:
    x, y = toy
    p = AlwaysHold().fit(x, y).predict_proba(x)
    assert np.allclose(p[:, CLASSES.index(HOLD)], 1.0)
    assert (decide(p, min_confidence=0.3) == HOLD).all()


def test_class_prior_reproduces_the_training_balance(toy: tuple[pd.DataFrame, pd.Series]) -> None:
    x, y = toy
    model = ClassPrior().fit(x, y)
    p = model.predict_proba(x)[0]
    expected = [float((y == c).mean()) for c in CLASSES]
    assert p == pytest.approx(expected)


def test_probabilities_sum_to_one(toy: tuple[pd.DataFrame, pd.Series]) -> None:
    x, y = toy
    for model in (AlwaysHold(), ClassPrior(), LogisticBaseline()):
        p = model.fit(x, y).predict_proba(x)
        assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)
        assert p.shape == (len(x), 3)


def test_unfitted_model_refuses_to_predict(toy: tuple[pd.DataFrame, pd.Series]) -> None:
    x, _ = toy
    with pytest.raises(RuntimeError, match="not been fitted"):
        LogisticBaseline().predict_proba(x)


def test_reordered_columns_are_refused(toy: tuple[pd.DataFrame, pd.Series]) -> None:
    """A silently reordered matrix changes what every coefficient means."""
    x, y = toy
    model = LogisticBaseline().fit(x, y)
    with pytest.raises(ValueError, match="columns differ"):
        model.predict_proba(x[["b", "a"]])


def test_probability_columns_follow_the_declared_class_order(
    toy: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Guards the mapping that would otherwise swap buy and sell."""
    x = pd.DataFrame({"a": np.linspace(-3, 3, 400)})
    y = pd.Series(np.where(x["a"] > 0, BUY, SELL), dtype="float64")
    p = LogisticBaseline(class_weight=None).fit(x, y).predict_proba(x)
    # Large positive feature must push probability towards BUY, not SELL.
    assert p[-1, CLASSES.index(BUY)] > p[-1, CLASSES.index(SELL)]
    assert p[0, CLASSES.index(SELL)] > p[0, CLASSES.index(BUY)]


def test_missing_class_leaves_a_zero_column_not_a_shift() -> None:
    """A block with no BUY must not shift SELL into BUY's column."""
    x = pd.DataFrame({"a": np.linspace(-1, 1, 200)})
    y = pd.Series(np.where(x["a"] > 0, HOLD, SELL), dtype="float64")
    p = LogisticBaseline(class_weight=None).fit(x, y).predict_proba(x)
    assert np.allclose(p[:, CLASSES.index(BUY)], 0.0)


def test_clean_drops_warmup_and_unlabelled_rows() -> None:
    x = pd.DataFrame({"a": [np.nan, 1.0, 2.0, 3.0]})
    y = pd.Series([0.0, 1.0, np.nan, -1.0])
    xc, yc = clean(x, y)
    assert len(xc) == 2
    assert list(yc) == [1.0, -1.0]


def test_empty_training_block_is_refused() -> None:
    empty_x = pd.DataFrame({"a": []})
    empty_y = pd.Series([], dtype="float64")
    with pytest.raises(ValueError, match="empty training block"):
        ClassPrior().fit(empty_x, empty_y)
