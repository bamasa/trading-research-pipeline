"""Tests for successive halving, configuration sampling and the rule strategies.

The halving tests pin the property that makes it worth having: a candidate is
only promoted on evidence, and the winner has been measured at the full budget.
Getting that wrong produces a search that is fast and meaningless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_research.models.base import CLASSES
from trading_research.models.classical import MAX_CONFIDENCE, NEUTRAL, Breakout, Momentum
from trading_research.pipeline.stages import build_model
from trading_research.validation.search import (
    SearchError,
    sample_configurations,
    successive_halving,
)


def scorer(quality: dict[str, float], noise: float = 0.0, trades: float = 500.0):
    """A candidate's true quality, optionally obscured at small budgets."""
    rng = np.random.default_rng(0)

    def score(candidate, budget: int) -> dict[str, float]:
        blur = noise / max(1, budget)
        return {
            "net_per_trade_bp": quality[candidate] + rng.normal(0, blur),
            "trades": trades,
            "budget": float(budget),
        }

    return score


# ---------------------------------------------------------------------------
# Successive halving
# ---------------------------------------------------------------------------


def test_the_best_candidate_wins() -> None:
    quality = {"a": 1.0, "b": 5.0, "c": 2.0, "d": -1.0, "e": 0.0, "f": 3.0}
    outcome = successive_halving(list(quality), scorer(quality), budgets=(2, 4, 8))
    assert outcome.best == "b"


def test_the_winner_was_measured_at_the_full_budget() -> None:
    """A winner promoted on cheap evidence and never re-scored is not a winner."""
    quality = {"a": 1.0, "b": 5.0, "c": 2.0, "d": -1.0, "e": 0.0, "f": 3.0}
    outcome = successive_halving(list(quality), scorer(quality), budgets=(2, 4, 8))
    assert outcome.best_metrics["budget"] == 8.0


def test_later_rungs_carry_fewer_candidates() -> None:
    quality = {c: float(i) for i, c in enumerate("abcdefghij")}
    outcome = successive_halving(list(quality), scorer(quality), budgets=(2, 4, 8))
    counts = [rung.candidates for rung in outcome.rungs]
    assert counts == sorted(counts, reverse=True)
    assert counts[-1] < counts[0]


def test_halving_is_cheaper_than_scoring_everything() -> None:
    quality = {c: float(i) for i, c in enumerate("abcdefghijkl")}
    outcome = successive_halving(list(quality), scorer(quality), budgets=(2, 6, 20))
    spent = sum(len(r.results) * r.budget for r in outcome.rungs)
    exhaustive = len(quality) * 20
    assert spent < exhaustive / 2


def test_a_candidate_that_barely_traded_is_not_promoted() -> None:
    """Otherwise the early rounds promote whoever took three lucky trades."""

    def score(candidate, budget: int) -> dict[str, float]:
        if candidate == "lucky":
            return {"net_per_trade_bp": 99.0, "trades": 3.0}
        return {"net_per_trade_bp": float(len(candidate)), "trades": 500.0}

    outcome = successive_halving(
        ["lucky", "aa", "bbb", "cccc"], score, budgets=(2, 4), minimum_trades=50
    )
    assert outcome.best != "lucky"


def test_a_candidate_that_raises_is_recorded_not_fatal() -> None:
    def score(candidate, budget: int) -> dict[str, float]:
        if candidate == "broken":
            raise RuntimeError("no data")
        return {"net_per_trade_bp": 1.0, "trades": 500.0}

    outcome = successive_halving(["broken", "fine", "also_fine"], score, budgets=(2, 4))
    assert outcome.best in {"fine", "also_fine"}
    assert "no data" in str(outcome.table["skipped"].dropna().iloc[0])


def test_a_search_where_nothing_scores_is_an_error() -> None:
    def score(candidate, budget: int) -> dict[str, float]:
        return {"net_per_trade_bp": float("nan"), "trades": 0.0}

    with pytest.raises(SearchError, match="net_per_trade_bp"):
        successive_halving(["a", "b"], score, budgets=(2,))


def test_an_empty_candidate_list_is_refused() -> None:
    with pytest.raises(SearchError, match="no candidates"):
        successive_halving([], scorer({}), budgets=(2,))


def test_the_summary_reports_the_saving() -> None:
    quality = {c: float(i) for i, c in enumerate("abcdefgh")}
    outcome = successive_halving(list(quality), scorer(quality), budgets=(2, 4, 16))
    assert "cheaper than scoring every candidate" in outcome.summary()
    assert outcome.evaluations > 0


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


SPACE = {"a": [1, 2, 3, 4], "b": ["x", "y", "z"], "c": [True, False]}


def test_sampling_returns_distinct_configurations() -> None:
    drawn = sample_configurations(SPACE, 10, seed=1)
    assert len(drawn) == 10
    assert len({tuple(sorted(d.items())) for d in drawn}) == 10


def test_sampling_is_deterministic_given_a_seed() -> None:
    """A search nobody can repeat is not evidence."""
    assert sample_configurations(SPACE, 8, seed=3) == sample_configurations(SPACE, 8, seed=3)


def test_a_space_smaller_than_the_request_is_enumerated() -> None:
    drawn = sample_configurations(SPACE, 100)
    assert len(drawn) == 4 * 3 * 2


def test_every_key_appears_in_every_configuration() -> None:
    assert all(set(d) == set(SPACE) for d in sample_configurations(SPACE, 5, seed=2))


# ---------------------------------------------------------------------------
# Rule strategies
# ---------------------------------------------------------------------------


@pytest.fixture
def book() -> pd.DataFrame:
    rng = np.random.default_rng(4)
    n = 3000
    return pd.DataFrame(
        {
            "log_mid_ret20": rng.normal(0, 5, n),
            "log_mid_ret50": rng.normal(0, 8, n),
            "log_mid_vol50": np.abs(rng.normal(4, 1, n)),
            "queue_imbalance": rng.uniform(-1, 1, n),
            "spread_bp": np.abs(rng.normal(2, 0.5, n)),
        }
    )


RULES = ("momentum", "mean_reversion", "order_flow", "breakout", "spread_capture")


@pytest.mark.parametrize("name", RULES)
def test_a_rule_produces_valid_probabilities(name, book) -> None:
    model = build_model(name).fit(book, pd.Series(np.zeros(len(book))))
    proba = model.predict_proba(book)
    assert proba.shape == (len(book), len(CLASSES))
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (proba >= 0).all()


@pytest.mark.parametrize("name", RULES)
def test_a_rule_can_be_made_selective(name, book) -> None:
    """Its strongest signal must sit inside the threshold sweep's range, or the
    rule could never be excluded and would not be comparable with a model."""
    model = build_model(name).fit(book, pd.Series(np.zeros(len(book))))
    strongest = model.predict_proba(book)[:, [CLASSES.index(-1), CLASSES.index(1)]].max()
    assert strongest <= MAX_CONFIDENCE + 1e-9
    assert strongest < 0.95


def test_no_opinion_means_the_class_prior(book) -> None:
    flat = book.assign(log_mid_ret20=0.0)
    model = Momentum().fit(book, pd.Series(np.zeros(len(book))))
    proba = model.predict_proba(flat)
    assert np.allclose(proba, NEUTRAL, atol=1e-9)


def test_mean_reversion_is_the_opposite_of_momentum(book) -> None:
    y = pd.Series(np.zeros(len(book)))
    up = Momentum().fit(book, y).predict_proba(book)
    down = build_model("mean_reversion").fit(book, y).predict_proba(book)
    assert np.allclose(up[:, CLASSES.index(1)], down[:, CLASSES.index(-1)])


def test_a_deadzone_keeps_a_rule_quiet_in_the_middle(book) -> None:
    model = Breakout().fit(book, pd.Series(np.zeros(len(book))))
    proba = model.predict_proba(book)
    silent = np.isclose(proba, NEUTRAL).all(axis=1)
    assert silent.mean() > 0.3


def test_a_rule_ignores_the_label(book) -> None:
    """None of these fit anything; handing them a different target must not move
    a single prediction."""
    rng = np.random.default_rng(5)
    a = Momentum().fit(book, pd.Series(np.zeros(len(book)))).predict_proba(book)
    b = Momentum().fit(book, pd.Series(rng.normal(size=len(book)))).predict_proba(book)
    assert np.array_equal(a, b)


def test_a_missing_column_is_reported_by_name(book) -> None:
    with pytest.raises(KeyError, match="log_mid_ret20"):
        Momentum().fit(book.drop(columns="log_mid_ret20"), pd.Series(np.zeros(len(book))))
