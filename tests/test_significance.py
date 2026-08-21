"""The unit of observation, which this project got wrong once and expensively."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_research.evaluation.significance import (
    SignificanceError,
    assess,
    effective_count,
    mean_pairwise_correlation,
)


def _panel(days: int, symbols: int, shared: float, seed: int = 0) -> pd.DataFrame:
    """Trades whose per-day results share a common component of size ``shared``."""
    rng = np.random.default_rng(seed)
    rows = []
    for day in range(days):
        common = rng.normal(0.0, shared)
        for s in range(symbols):
            rows.append(
                {
                    "day": day,
                    "symbol": f"S{s}",
                    "net_bp": common + rng.normal(0.0, 1.0 - shared),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# The effective count
# ---------------------------------------------------------------------------


def test_uncorrelated_observations_all_count() -> None:
    assert effective_count(26, 0.0) == pytest.approx(26.0)


def test_perfectly_correlated_observations_count_once() -> None:
    assert effective_count(26, 0.999) == pytest.approx(1.0, abs=0.05)


def test_the_real_case_lands_where_it_did() -> None:
    """26 instruments correlating at 0.47 are about two bets, not twenty-six."""
    assert effective_count(26, 0.47) == pytest.approx(2.05, abs=0.1)


def test_negative_correlation_does_not_inflate_the_count() -> None:
    """A noisy estimate below zero must not be read as extra evidence."""
    assert effective_count(10, -0.5) == pytest.approx(10.0)


def test_an_unknown_correlation_falls_back_to_the_raw_count() -> None:
    assert effective_count(10, float("nan")) == 10.0


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------


def test_correlation_recovers_a_known_common_component() -> None:
    panel = _panel(days=200, symbols=8, shared=0.7).pivot_table(
        index="day", columns="symbol", values="net_bp"
    )
    assert mean_pairwise_correlation(panel) > 0.5


def test_independent_series_correlate_at_about_zero() -> None:
    panel = _panel(days=200, symbols=8, shared=0.0).pivot_table(
        index="day", columns="symbol", values="net_bp"
    )
    assert abs(mean_pairwise_correlation(panel)) < 0.15


def test_a_single_series_has_no_correlation_to_report() -> None:
    panel = pd.DataFrame({"only": [1.0, 2.0, 3.0]})
    assert np.isnan(mean_pairwise_correlation(panel))


def test_constant_series_are_excluded_rather_than_crashing() -> None:
    panel = pd.DataFrame({"a": [1.0, 2.0, 3.0], "flat": [5.0, 5.0, 5.0]})
    assert np.isnan(mean_pairwise_correlation(panel))


# ---------------------------------------------------------------------------
# The assessment
# ---------------------------------------------------------------------------


def test_correlated_series_inflate_the_trade_level_statistic() -> None:
    """The mistake, reproduced deliberately.

    With a strong common component the trade-level *t* is several times the
    cluster-level one, and the trade-level number is the wrong one.
    """
    trades = _panel(days=14, symbols=26, shared=0.8, seed=1)
    trades["net_bp"] += 6.0  # a real but small edge
    result = assess(trades)

    assert result.trades == 14 * 26
    assert result.clusters == 14
    assert abs(result.trade_t) > abs(result.cluster_t)
    assert result.inflation > 2.0


def test_independent_series_agree_between_the_two_views() -> None:
    """The control: when bets really are independent, both figures match."""
    trades = _panel(days=60, symbols=20, shared=0.0, seed=2)
    trades["net_bp"] += 0.5
    result = assess(trades)
    assert result.inflation == pytest.approx(1.0, abs=0.6)


def test_the_verdict_refuses_to_call_two_sigma_settled() -> None:
    """This project reached a *t* of 2 and then lost money for six weeks."""
    # Constructed rather than sampled, so the statistic is exactly where the
    # verdict boundary is being tested and not wherever a seed happened to land.
    values = np.zeros(20)
    values[:] = np.linspace(-1.0, 1.0, 20)
    values = values - values.mean()
    values = values / values.std(ddof=1)
    target_t = 2.4
    values = values + target_t / np.sqrt(len(values))
    trades = pd.DataFrame({"day": np.arange(20), "symbol": "ONE", "net_bp": values})
    result = assess(trades, series=None)
    assert 2.0 <= abs(result.cluster_t) < 3.0
    assert "been wrong at exactly this level" in result.verdict


def test_a_weak_result_is_named_as_chance() -> None:
    trades = _panel(days=14, symbols=26, shared=0.8, seed=4)
    result = assess(trades)
    assert abs(result.cluster_t) < 2.0
    assert "chance" in result.verdict
    assert f"/{result.clusters}" in result.verdict


def test_one_cluster_cannot_support_a_statistic() -> None:
    trades = _panel(days=1, symbols=26, shared=0.5)
    with pytest.raises(SignificanceError, match="single one"):
        assess(trades)


def test_a_missing_column_is_named() -> None:
    trades = _panel(days=5, symbols=3, shared=0.2)
    with pytest.raises(SignificanceError, match="episode"):
        assess(trades, cluster="episode")


def test_an_empty_frame_is_refused() -> None:
    with pytest.raises(SignificanceError, match="no trades"):
        assess(pd.DataFrame({"day": [], "net_bp": [], "symbol": []}))


def test_the_summary_carries_both_figures() -> None:
    trades = _panel(days=30, symbols=10, shared=0.5, seed=5)
    out = assess(trades).to_dict()
    for key in ("trade_t", "cluster_t", "correlation_within_cluster", "inflation"):
        assert key in out
