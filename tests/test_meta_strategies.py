"""Tests for the two-stage and expected-value strategies.

The property that makes meta-labelling worth the second model is that the
secondary is fitted only on the moments the primary wanted. Fitting it
everywhere would teach it about rows the strategy will never be at, and the
arrangement would collapse into two models answering the same question.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_research.models.base import CLASSES, Model
from trading_research.models.meta import ExpectedValue, MetaLabelled, direction_of
from trading_research.models.regression import RidgeBaseline
from trading_research.pipeline.stages import build_model

BUY, SELL, HOLD = CLASSES.index(1), CLASSES.index(-1), CLASSES.index(0)


class AlwaysLong(Model):
    """A primary that points the same way everywhere."""

    def __init__(self) -> None:
        super().__init__(name="always_long")

    def fit(self, x: pd.DataFrame, y: pd.Series) -> AlwaysLong:
        self.fitted_ = True
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        proba = np.zeros((len(x), len(CLASSES)))
        proba[:, BUY] = 0.7
        proba[:, HOLD] = 0.3
        return proba


class Recorder(Model):
    """A secondary that records how many rows it was fitted on."""

    def __init__(self) -> None:
        super().__init__(name="recorder")
        self.rows_seen = 0
        self.positive_share = float("nan")

    def fit(self, x: pd.DataFrame, y: pd.Series) -> Recorder:
        self.rows_seen = len(x)
        self.positive_share = float(y.mean())
        self.fitted_ = True
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        proba = np.zeros((len(x), len(CLASSES)))
        proba[:, BUY] = 0.8
        proba[:, HOLD] = 0.2
        return proba


@pytest.fixture
def data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    n = 3000
    signal = rng.normal(size=n)
    label = np.where(signal > 1.0, 1.0, np.where(signal < -1.0, -1.0, 0.0))
    return pd.DataFrame({"signal": signal, "noise": rng.normal(size=n)}), pd.Series(label)


# ---------------------------------------------------------------------------
# Meta-labelling
# ---------------------------------------------------------------------------


def test_the_secondary_sees_only_the_moments_the_primary_wanted(data) -> None:
    """The property that makes the second model worth having."""
    x, y = data
    recorder = Recorder()
    MetaLabelled(AlwaysLong(), recorder).fit(x, y)
    assert recorder.rows_seen == int(np.isfinite(y).sum())

    fewer = Recorder()
    x_gapped = x.copy()
    y_gapped = y.copy()
    y_gapped.iloc[:1000] = np.nan
    MetaLabelled(AlwaysLong(), fewer).fit(x_gapped, y_gapped)
    assert fewer.rows_seen < recorder.rows_seen


def test_the_secondary_learns_whether_the_primary_was_right(data) -> None:
    """Its target is the primary's outcome, not the market's direction."""
    x, y = data
    recorder = Recorder()
    MetaLabelled(AlwaysLong(), recorder).fit(x, y)
    # The primary is always long, so it wins exactly where the target is +1.
    assert recorder.positive_share == pytest.approx(float((y == 1).mean()), abs=1e-9)


def test_a_target_of_zero_counts_as_a_loss(data) -> None:
    """The trade was taken and paid its costs; nothing happening is not neutral."""
    x = pd.DataFrame({"signal": np.zeros(600), "noise": np.zeros(600)})
    y = pd.Series([1.0] * 300 + [0.0] * 300)
    recorder = Recorder()
    MetaLabelled(AlwaysLong(), recorder).fit(x, y)
    assert recorder.positive_share == pytest.approx(0.5)


def test_the_direction_comes_from_the_primary_only(data) -> None:
    """The secondary cannot flip the side — it only says whether to act."""
    x, y = data
    model = MetaLabelled(AlwaysLong(), Recorder()).fit(x, y)
    proba = model.predict_proba(x)
    assert (proba[:, SELL] == 0).all()
    assert proba[:, BUY].max() > 0


def test_the_secondary_supplies_the_confidence(data) -> None:
    x, y = data
    model = MetaLabelled(AlwaysLong(), Recorder()).fit(x, y)
    # Recorder always returns 0.8 for the winning class.
    assert model.predict_proba(x)[:, BUY].max() == pytest.approx(0.8)


def test_a_primary_that_never_wins_is_refused(data) -> None:
    """One outcome class gives the secondary nothing to separate."""
    x, _ = data
    always_lost = pd.Series(np.full(len(x), -1.0))
    with pytest.raises(ValueError, match="outcome classes"):
        MetaLabelled(AlwaysLong(), Recorder()).fit(x, always_lost)


def test_meta_probabilities_sum_to_one(data) -> None:
    x, y = data
    proba = MetaLabelled(AlwaysLong(), Recorder()).fit(x, y).predict_proba(x)
    assert np.allclose(proba.sum(axis=1), 1.0)


@pytest.mark.parametrize("name", ["meta_rule_model", "meta_rule_tree", "meta_model_model"])
def test_meta_strategies_are_addressable_by_name(name) -> None:
    model = build_model(name)
    assert model.primary is not None and model.secondary is not None


def test_direction_of_reads_the_stronger_side() -> None:
    proba = np.zeros((2, len(CLASSES)))
    proba[0, BUY], proba[1, SELL] = 0.6, 0.6
    assert direction_of(proba).tolist() == [1, -1]


# ---------------------------------------------------------------------------
# Expected value
# ---------------------------------------------------------------------------


@pytest.fixture
def regression_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(1)
    n = 2000
    signal = rng.normal(size=n)
    return (
        pd.DataFrame({"signal": signal, "noise": rng.normal(size=n)}),
        pd.Series(signal * 30.0 + rng.normal(0, 3, n)),
    )


def test_it_trades_only_when_the_prediction_beats_the_cost(regression_data) -> None:
    x, y = regression_data
    model = ExpectedValue(RidgeBaseline(scale_bp=11.0), cost_bp=11.0).fit(x, y)
    edge = model.predict_edge_bp(x)
    proba = model.predict_proba(x)
    acted = proba[:, [SELL, BUY]].max(axis=1) > 1.0 / 3.0 + 1e-9
    assert (np.abs(edge[acted]) > model.hurdle_bp).all()
    assert not acted[np.abs(edge) <= model.hurdle_bp].any()


def test_a_margin_above_one_asks_for_a_buffer(regression_data) -> None:
    x, y = regression_data
    plain = ExpectedValue(RidgeBaseline(scale_bp=11.0), cost_bp=11.0, margin=1.0).fit(x, y)
    strict = ExpectedValue(RidgeBaseline(scale_bp=11.0), cost_bp=11.0, margin=2.0).fit(x, y)
    assert strict.hurdle_bp == 2 * plain.hurdle_bp

    def acted(model):
        return (model.predict_proba(x)[:, [SELL, BUY]].max(axis=1) > 1.0 / 3.0 + 1e-9).sum()

    assert acted(strict) < acted(plain)


def test_size_grows_with_the_surplus_over_the_hurdle(regression_data) -> None:
    x, y = regression_data
    model = ExpectedValue(RidgeBaseline(scale_bp=11.0), cost_bp=11.0).fit(x, y)
    edge = np.abs(model.predict_edge_bp(x))
    size = model.position_size(x)
    order = np.argsort(edge)
    assert np.all(np.diff(size[order]) >= -1e-9)


def test_size_is_capped(regression_data) -> None:
    """The largest predictions are where the model extrapolates, so an
    uncapped rule puts the most weight exactly where it is least reliable."""
    x, y = regression_data
    model = ExpectedValue(RidgeBaseline(scale_bp=11.0), cost_bp=11.0).fit(x, y)
    assert model.position_size(x, cap=2.0).max() <= 2.0


def test_nothing_below_the_hurdle_is_sized(regression_data) -> None:
    x, y = regression_data
    model = ExpectedValue(RidgeBaseline(scale_bp=11.0), cost_bp=11.0).fit(x, y)
    edge = np.abs(model.predict_edge_bp(x))
    assert (model.position_size(x)[edge <= model.hurdle_bp] == 0).all()


def test_a_non_regression_model_is_refused() -> None:
    from trading_research.models.linear import LogisticBaseline

    model = ExpectedValue(LogisticBaseline(), cost_bp=11.0)
    model.fitted_ = True
    with pytest.raises(TypeError, match="basis points"):
        model.predict_edge_bp(pd.DataFrame({"a": [1.0]}))


def test_impossible_parameters_are_rejected() -> None:
    with pytest.raises(ValueError, match="cost_bp"):
        ExpectedValue(RidgeBaseline(), cost_bp=0.0)
    with pytest.raises(ValueError, match="margin"):
        ExpectedValue(RidgeBaseline(), cost_bp=11.0, margin=0.0)
