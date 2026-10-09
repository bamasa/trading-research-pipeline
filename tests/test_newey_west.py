"""The Newey-West t of a daily series, on series whose answer is known."""

from __future__ import annotations

import math

import numpy as np
import pytest

from trading_research.evaluation.significance import (
    NEWEY_WEST_LAG,
    SignificanceError,
    newey_west_t,
)
from trading_research.market_making.verdicts import day_t


def test_the_registered_lag_is_five_days() -> None:
    assert NEWEY_WEST_LAG == 5
    assert newey_west_t(np.arange(30.0)).lag == 5


def test_a_hand_computed_series() -> None:
    # mean 3, gamma_0 = 2, gamma_1 = 0.8, Bartlett weight 1/2: variance 2.8.
    out = newey_west_t([1.0, 2.0, 3.0, 4.0, 5.0], lag=1)
    assert out.days == 5
    assert out.mean == pytest.approx(3.0)
    assert out.se == pytest.approx(math.sqrt(2.8 / 5))
    assert out.t == pytest.approx(3.0 / math.sqrt(0.56))


def test_lag_zero_is_the_plain_t_with_population_variance() -> None:
    values = np.random.default_rng(0).normal(0.3, 1.0, 40)
    plain = day_t(values)
    out = newey_west_t(values, lag=0)
    assert out.t == pytest.approx(plain.t * math.sqrt(40 / 39))


def test_runs_of_equal_days_widen_the_error() -> None:
    # Each value repeated over five consecutive days: strongly dependent days.
    base = np.random.default_rng(1).normal(0.5, 1.0, 12)
    values = np.repeat(base, 5)
    assert newey_west_t(values).se > newey_west_t(values, lag=0).se * 1.5


def test_alternating_days_narrow_the_error() -> None:
    values = np.tile([2.0, -1.0], 20)
    assert newey_west_t(values).se < newey_west_t(values, lag=0).se


def test_a_constant_shift_moves_the_mean_and_not_the_error() -> None:
    values = np.random.default_rng(2).normal(0.0, 1.0, 50)
    a, b = newey_west_t(values), newey_west_t(values + 3.0)
    assert b.mean == pytest.approx(a.mean + 3.0)
    assert b.se == pytest.approx(a.se)


def test_non_finite_days_are_dropped() -> None:
    out = newey_west_t([1.0, np.nan, 2.0, np.inf, 3.0], lag=1)
    assert out.days == 3
    assert out.mean == pytest.approx(2.0)


def test_a_short_series_has_no_t() -> None:
    assert math.isnan(newey_west_t([1.0]).t)
    assert math.isnan(newey_west_t([]).mean)
    assert math.isnan(newey_west_t([2.0, 2.0, 2.0]).t)


def test_the_lag_is_cut_to_the_series() -> None:
    assert newey_west_t([1.0, 2.0, 4.0], lag=5).lag == 2


def test_a_negative_lag_is_refused() -> None:
    with pytest.raises(SignificanceError):
        newey_west_t([1.0, 2.0], lag=-1)
