"""Tests for the heteroscedastic model and the confidence-bound rule.

The dangerous mistake here is fitting the variance model on the mean model's
in-sample residuals. It then learns that the mean model is far better than it
is, the predicted uncertainty comes out too small everywhere, and the gate opens
on everything — a gate reporting confidence it has not earned, which is worse
than no gate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_research.models.base import CLASSES
from trading_research.models.regression import RidgeBaseline
from trading_research.models.uncertainty import ConfidenceBound, HeteroscedasticRegressor
from trading_research.pipeline.stages import build_model

BUY, SELL, HOLD = CLASSES.index(1), CLASSES.index(-1), CLASSES.index(0)


@pytest.fixture
def easy() -> tuple[pd.DataFrame, pd.Series]:
    """A target the model can predict well: small residual, large signal."""
    rng = np.random.default_rng(0)
    n = 4000
    signal = rng.normal(size=n)
    return (
        pd.DataFrame({"signal": signal, "noise": rng.normal(size=n)}),
        pd.Series(signal * 30.0 + rng.normal(0, 2.0, n)),
    )


@pytest.fixture
def hard() -> tuple[pd.DataFrame, pd.Series]:
    """A target it cannot: the residual dwarfs anything it can explain."""
    rng = np.random.default_rng(1)
    n = 4000
    signal = rng.normal(size=n)
    return (
        pd.DataFrame({"signal": signal, "noise": rng.normal(size=n)}),
        pd.Series(signal * 0.5 + rng.normal(0, 20.0, n)),
    )


def build(cost_bp: float = 11.0, k: float = 1.0) -> ConfidenceBound:
    return ConfidenceBound(
        HeteroscedasticRegressor(RidgeBaseline(scale_bp=cost_bp), RidgeBaseline(scale_bp=cost_bp)),
        cost_bp=cost_bp,
        k=k,
    )


# ---------------------------------------------------------------------------
# The variance model
# ---------------------------------------------------------------------------


def test_uncertainty_is_larger_where_the_model_is_worse(easy, hard) -> None:
    """The whole point: sigma has to track how wrong the mean model actually is."""
    confident = HeteroscedasticRegressor(RidgeBaseline(), RidgeBaseline()).fit(*easy)
    unsure = HeteroscedasticRegressor(RidgeBaseline(), RidgeBaseline()).fit(*hard)
    assert np.median(unsure.predict_sigma_bp(hard[0])) > np.median(
        confident.predict_sigma_bp(easy[0])
    )


def test_sigma_is_measured_on_data_the_mean_model_did_not_fit(easy) -> None:
    """In-sample residuals would report an uncertainty the model has not earned."""
    x, y = easy
    model = HeteroscedasticRegressor(RidgeBaseline(), RidgeBaseline(), holdout_fraction=0.3).fit(
        x, y
    )
    sigma = float(np.median(model.predict_sigma_bp(x)))
    # The true residual standard deviation is 2.0 by construction. An in-sample
    # fit would come out well under it; an honest one lands near it.
    assert 1.0 < sigma < 4.0


def test_sigma_never_falls_below_the_floor(easy) -> None:
    x, y = easy
    model = HeteroscedasticRegressor(RidgeBaseline(), RidgeBaseline(), floor_bp=5.0).fit(x, y)
    assert model.predict_sigma_bp(x).min() >= 5.0


def test_a_negative_predicted_variance_is_clipped_not_propagated(hard) -> None:
    """A regressor on squared residuals can predict a negative number."""
    x, y = hard
    model = HeteroscedasticRegressor(RidgeBaseline(), RidgeBaseline()).fit(x, y)
    assert np.isfinite(model.predict_sigma_bp(x)).all()
    assert (model.predict_sigma_bp(x) > 0).all()


def test_too_little_data_to_split_is_refused(easy) -> None:
    x, y = easy
    with pytest.raises(ValueError, match="residual split"):
        HeteroscedasticRegressor(RidgeBaseline(), RidgeBaseline()).fit(x.head(300), y.head(300))


def test_an_impossible_holdout_is_rejected() -> None:
    with pytest.raises(ValueError, match="holdout_fraction"):
        HeteroscedasticRegressor(RidgeBaseline(), RidgeBaseline(), holdout_fraction=0.9)


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------


def test_it_rejects_unreliable_predictions_not_merely_small_ones(easy) -> None:
    """A size threshold discards small predictions; this discards unsure ones,
    and the two disagree on most observations."""
    x, y = easy
    model = build(cost_bp=5.0, k=1.0).fit(x, y)
    mean = model.model.predict_mean_bp(x)
    lower, upper = model.bounds(x)

    by_bound = (lower > 5.0) | (upper < -5.0)
    by_size = np.abs(mean) > 5.0
    assert by_bound.sum() < by_size.sum()
    assert (by_size & ~by_bound).sum() > 0


def test_a_larger_k_asks_for_more_margin_and_trades_less(easy) -> None:
    x, y = easy
    counts = []
    for k in (0.0, 1.0, 2.0):
        model = build(cost_bp=5.0, k=k).fit(x, y)
        lower, upper = model.bounds(x)
        counts.append(int(((lower > 5.0) | (upper < -5.0)).sum()))
    assert counts == sorted(counts, reverse=True)


def test_at_k_zero_the_rule_is_expected_value_against_cost(easy) -> None:
    x, y = easy
    model = build(cost_bp=5.0, k=0.0).fit(x, y)
    lower, upper = model.bounds(x)
    mean = model.model.predict_mean_bp(x)
    assert np.allclose(lower, mean)
    assert np.allclose(upper, mean)


def test_a_model_whose_uncertainty_exceeds_its_signal_never_trades(hard) -> None:
    """The finding this class produced on real data, as a test.

    When sigma dwarfs mu no bound can clear any cost, and the honest output is
    to stand aside rather than to lower the bar until something passes.
    """
    x, y = hard
    model = build(cost_bp=11.0, k=1.0).fit(x, y)
    proba = model.predict_proba(x)
    assert proba[:, HOLD].min() == pytest.approx(1.0)


def test_nothing_trades_when_the_cost_is_out_of_reach(easy) -> None:
    x, y = easy
    model = build(cost_bp=1e6, k=1.0).fit(x, y)
    assert model.predict_proba(x)[:, HOLD].min() == pytest.approx(1.0)


def test_probabilities_sum_to_one(easy) -> None:
    x, y = easy
    assert np.allclose(build(cost_bp=5.0).fit(x, y).predict_proba(x).sum(axis=1), 1.0)


def test_confidence_grows_with_the_surplus_over_the_cost(easy) -> None:
    x, y = easy
    model = build(cost_bp=3.0, k=0.5).fit(x, y)
    lower, _ = model.bounds(x)
    proba = model.predict_proba(x)
    acted = lower > 3.0
    if acted.sum() > 10:
        order = np.argsort(lower[acted])
        assert np.all(np.diff(proba[acted, BUY][order]) >= -1e-9)


def test_invalid_parameters_are_rejected() -> None:
    hetero = HeteroscedasticRegressor(RidgeBaseline(), RidgeBaseline())
    with pytest.raises(ValueError, match="cost_bp"):
        ConfidenceBound(hetero, cost_bp=0.0)
    with pytest.raises(ValueError, match="k must be"):
        ConfidenceBound(hetero, cost_bp=11.0, k=-1.0)


@pytest.mark.parametrize("name", ["bound_ridge", "bound_mixed"])
def test_bound_models_are_addressable_by_name(name) -> None:
    model = build_model(name, cost_bp=11.02, k=1.0)
    assert model.cost_bp == pytest.approx(11.02)
    assert model.k == pytest.approx(1.0)
