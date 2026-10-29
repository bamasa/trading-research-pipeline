"""The hold-first rule and its per-side thresholds from a trade rate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_research.strategies.hold_first import (
    HoldFirstRule,
    count_days,
    fit_hold_first,
    side_threshold,
)


def _proba(n: int, seed: int = 0) -> np.ndarray:
    # Columns in the models' order: sell, hold, buy.
    return np.random.default_rng(seed).dirichlet([1.0, 6.0, 1.0], size=n)


def test_the_rule_needs_its_side_to_clear_and_to_lead() -> None:
    rule = HoldFirstRule(theta_long=0.4, theta_short=0.5)
    p_long = np.array([0.45, 0.45, 0.39, 0.30, 0.5])
    p_short = np.array([0.10, 0.46, 0.10, 0.55, 0.5])
    assert rule.decide(p_long, p_short).tolist() == [1, 0, 0, -1, 0]


def test_thresholds_hit_half_the_rate_on_each_side() -> None:
    days, rate = 4, 240
    proba = _proba(40_000)
    rule = fit_hold_first(proba, days=days, trades_per_day=rate)
    long_rate, short_rate = rule.entries_per_day(proba, days)
    assert long_rate == pytest.approx(rate / 2)
    assert short_rate == pytest.approx(rate / 2)
    looser = fit_hold_first(proba, days=days, trades_per_day=2 * rate)
    assert looser.theta_long < rule.theta_long


def test_a_missing_class_never_trades() -> None:
    proba = _proba(1_000)
    proba[:, 2] = 0.0  # no buy class in this fit
    rule = fit_hold_first(proba, days=1, trades_per_day=50)
    assert rule.theta_long == float("inf")
    assert (rule.decide_proba(proba) != 1).all()
    assert (rule.decide_proba(proba) == -1).sum() == 25


def test_fewer_candidates_than_entries_takes_them_all() -> None:
    theta = side_threshold(np.array([0.9, 0.2, 0.6]), np.array([0.0, 0.5, 0.1]), 10)
    assert theta == pytest.approx(0.6)


def test_days_are_counted_by_utc_calendar_day() -> None:
    stamps = pd.to_datetime(
        ["2025-06-24 23:59:59", "2025-06-25 00:00:01", "2025-06-30 12:00:00"], utc=True
    )
    assert count_days(pd.Series(stamps)) == 3
    with pytest.raises(ValueError, match="positive"):
        fit_hold_first(_proba(10), days=0, trades_per_day=25)
