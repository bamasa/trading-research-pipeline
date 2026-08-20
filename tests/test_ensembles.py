"""Forests, stacking, and the bias-variance instrument that chooses between them."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from trading_research.evaluation.bias_variance import (
    DecompositionError,
    compare,
    decompose,
)
from trading_research.models.base import CLASSES
from trading_research.models.stacking import (
    Bagged,
    Stacked,
    StackingError,
    Voting,
    forward_chaining_folds,
)
from trading_research.pipeline.stages import build_model


def _data(n: int = 4_000, strength: float = 0.4, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(rng.normal(0, 1, (n, 5)), columns=list("abcde"))
    signal = strength * x["a"] + rng.normal(0, 1, n)
    y = pd.Series(np.sign(signal).astype(int), name="label")
    return x, y


# ---------------------------------------------------------------------------
# Registered models
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["random_forest", "extra_trees", "lightgbm"])
def test_new_models_honour_the_project_contract(name: str) -> None:
    x, y = _data()
    proba = build_model(name).fit(x, y).predict_proba(x)
    assert proba.shape == (len(x), len(CLASSES))
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (proba >= 0).all()


@pytest.mark.parametrize("name", ["random_forest", "extra_trees", "lightgbm"])
def test_class_columns_are_mapped_by_label_not_position(name: str) -> None:
    """A block with no sells must not shift every column left.

    The estimator's ``classes_`` is whatever it happened to see; the project's
    column order is fixed. Mapping by position silently mislabels every
    probability whenever a training block is missing a class.
    """
    x, y = _data()
    y = y.replace(-1, 0)  # buys and holds only
    proba = build_model(name).fit(x, y).predict_proba(x)
    assert np.allclose(proba[:, CLASSES.index(-1)], 0.0)
    assert np.allclose(proba.sum(axis=1), 1.0)


@pytest.mark.parametrize("name", ["random_forest", "extra_trees", "lightgbm"])
def test_a_single_class_block_is_refused(name: str) -> None:
    x, _ = _data()
    with pytest.raises(ValueError, match="one class"):
        build_model(name).fit(x, pd.Series(np.ones(len(x), dtype=int)))


@pytest.mark.parametrize("name", ["random_forest", "extra_trees", "lightgbm"])
def test_predicting_before_fitting_raises(name: str) -> None:
    x, _ = _data()
    with pytest.raises(RuntimeError, match="not been fitted"):
        build_model(name).predict_proba(x)


# ---------------------------------------------------------------------------
# Forward-chaining folds — where stacking usually leaks
# ---------------------------------------------------------------------------


def test_folds_never_train_on_the_future() -> None:
    for train, predict in forward_chaining_folds(10_000, folds=4):
        assert train.stop <= predict.start, "training must end before prediction begins"


def test_the_purge_separates_training_from_the_predicted_block() -> None:
    for train, predict in forward_chaining_folds(10_000, folds=4, purge=100):
        assert predict.start - train.stop >= 100


def test_folds_cover_distinct_blocks() -> None:
    folds = forward_chaining_folds(10_000, folds=4)
    starts = [p.start for _, p in folds]
    assert len(set(starts)) == len(starts)
    for (_, a), (_, b) in pairwise(folds):
        assert a.stop <= b.start


def test_too_few_folds_or_too_large_a_purge_is_refused() -> None:
    with pytest.raises(StackingError, match="two folds"):
        forward_chaining_folds(1_000, folds=1)
    with pytest.raises(StackingError, match="purge"):
        forward_chaining_folds(1_000, folds=4, purge=500)


# ---------------------------------------------------------------------------
# Bagging and voting
# ---------------------------------------------------------------------------


def test_bagging_members_disagree_when_fitted_on_different_history() -> None:
    """Zero disagreement means the averaging is doing nothing."""
    x, y = _data()
    bagged = Bagged(lambda: build_model("logistic"), members=5, block_fraction=0.4).fit(x, y)
    assert bagged.member_disagreement(x) > 0


def test_bagging_needs_at_least_two_members() -> None:
    with pytest.raises(StackingError, match="two members"):
        Bagged(lambda: build_model("logistic"), members=1)


def test_voting_reports_how_similar_its_members_are() -> None:
    """Two members correlating at 1.0 are one member with extra steps."""
    x, y = _data()
    same = Voting([build_model("logistic"), build_model("logistic")]).fit(x, y)
    assert same.diversity(x) == pytest.approx(1.0, abs=1e-6)

    different = Voting([build_model("logistic"), build_model("extra_trees")]).fit(x, y)
    assert different.diversity(x) < same.diversity(x)


def test_voting_weights_must_match_the_members() -> None:
    with pytest.raises(StackingError, match="weights"):
        Voting([build_model("logistic"), build_model("logistic")], weights=[1.0])


def test_equal_weights_are_the_plain_average() -> None:
    x, y = _data()
    members = [build_model("logistic"), build_model("extra_trees")]
    plain = Voting(members).fit(x, y).predict_proba(x)
    weighted = Voting(members, weights=[1.0, 1.0]).fit(x, y).predict_proba(x)
    assert np.allclose(plain, weighted)


# ---------------------------------------------------------------------------
# Stacking
# ---------------------------------------------------------------------------


def test_stacking_produces_valid_probabilities() -> None:
    x, y = _data()
    stack = Stacked(
        [lambda: build_model("logistic"), lambda: build_model("extra_trees")],
        lambda: build_model("logistic"),
        folds=3,
    ).fit(x, y)
    proba = stack.predict_proba(x)
    assert proba.shape == (len(x), len(CLASSES))
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_the_meta_model_never_sees_in_sample_base_predictions() -> None:
    """The defect stacking exists to avoid, asserted on the fold geometry.

    Trained on in-sample predictions, the second stage learns that the most
    flexible base model is always right -- because on its own training rows it
    is -- and the stack then fails on everything else.
    """
    n = 12_000
    for train, predict in forward_chaining_folds(n, folds=4, purge=50):
        overlap = set(range(train.start, train.stop)) & set(range(predict.start, predict.stop))
        assert not overlap


def test_stacking_needs_two_base_models() -> None:
    with pytest.raises(StackingError, match="two base models"):
        Stacked([lambda: build_model("logistic")], lambda: build_model("logistic"))


# ---------------------------------------------------------------------------
# Bias-variance
# ---------------------------------------------------------------------------


def _fit_predict(name: str, **params):
    def run(x: pd.DataFrame, y: pd.Series, test: pd.DataFrame) -> np.ndarray:
        model = build_model(name, **params).fit(x, y)
        proba = model.predict_proba(test)
        return proba[:, CLASSES.index(1)] - proba[:, CLASSES.index(-1)]

    return run


def test_an_unregularised_model_carries_more_variance_than_a_regularised_one() -> None:
    """The instrument must tell the two apart, or it says nothing.

    A single unpruned randomised tree against four hundred of them with large
    leaves. Note which comparison this is *not*: a default ExtraTrees has lower
    prediction variance than logistic regression, because averaging four
    hundred trees is itself a variance-reduction method. Flexibility of the
    model class and variance of its predictions are different things, which is
    the whole reason this is measured rather than assumed.
    """
    x, y = _data(n=6_000)
    test_x, test_y = _data(n=2_000, seed=99)

    wild = decompose(
        _fit_predict("extra_trees", n_estimators=1, max_depth=None, min_samples_leaf=1),
        x,
        y,
        test_x,
        test_y.astype(float),
        name="single_tree",
        fits=6,
    )
    tame = decompose(
        _fit_predict("extra_trees"), x, y, test_x, test_y.astype(float), name="extra_trees", fits=6
    )
    assert wild.variance > tame.variance
    assert wild.variance_share > tame.variance_share


def test_a_model_that_explains_nothing_says_so_rather_than_blaming_bias() -> None:
    """The reading that matters when the target is mostly noise.

    All the error lands in the "bias" term by arithmetic when a model explains
    nothing, which reads as "needs a better model class" and is misleading. The
    verdict has to distinguish the two.
    """
    rng = np.random.default_rng(21)
    x = pd.DataFrame(rng.normal(0, 1, (6_000, 4)), columns=list("abcd"))
    y = pd.Series(rng.choice([-1, 0, 1], 6_000))  # nothing to learn
    test_x = pd.DataFrame(rng.normal(0, 1, (2_000, 4)), columns=list("abcd"))
    test_y = pd.Series(rng.normal(0, 40, 2_000))  # pure noise target

    result = decompose(_fit_predict("logistic"), x, y, test_x, test_y, fits=5)
    assert result.explained_share < 0.01
    assert "noise" in result.verdict


def test_variance_share_and_verdict_agree() -> None:
    x, y = _data(n=6_000)
    test_x, test_y = _data(n=2_000, seed=7)
    result = decompose(_fit_predict("extra_trees"), x, y, test_x, test_y.astype(float), fits=5)
    assert 0.0 <= result.variance_share <= 1.0
    if result.variance_share > 0.6:
        assert "averaging" in result.verdict
    elif result.variance_share < 0.3:
        assert "different model class" in result.verdict


def test_comparison_ranks_by_what_averaging_buys() -> None:
    x, y = _data(n=6_000)
    test_x, test_y = _data(n=2_000, seed=11)
    table = compare(
        [
            decompose(_fit_predict(n), x, y, test_x, test_y.astype(float), name=n, fits=4)
            for n in ("logistic", "extra_trees")
        ]
    )
    assert list(table.columns[:3]) == ["model", "fits", "bias_squared"]
    assert table["ic_gain_from_averaging"].is_monotonic_decreasing


def test_one_fit_cannot_measure_variance() -> None:
    x, y = _data(n=2_000)
    with pytest.raises(DecompositionError, match="two fits"):
        decompose(_fit_predict("logistic"), x, y, x, y.astype(float), fits=1)


def test_too_little_data_is_refused_rather_than_guessed() -> None:
    x, y = _data(n=400)
    with pytest.raises(DecompositionError, match="cannot support"):
        decompose(_fit_predict("logistic"), x, y, x, y.astype(float), fits=4)
