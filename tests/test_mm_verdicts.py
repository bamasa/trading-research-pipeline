"""The market-making study's registered statistics and its status rule."""

from __future__ import annotations

import math

import numpy as np
import pytest

from trading_research.market_making.verdicts import (
    INCONCLUSIVE,
    KILLED,
    ROWS_PER_DAY,
    ROWS_PER_HOUR,
    KillCheck,
    breakeven_on_grid,
    day_t,
    exceeds_placebos,
    holm,
    k_dir,
    k_pess,
    roll_values,
    state_shift_rows,
    status,
    t_cdf_two_sided,
    t_sf,
)


@pytest.mark.parametrize(
    ("t", "df", "p"),
    [
        (1.7823, 12, 0.05),  # the 95% one-sided critical value at 12 df
        (2.6810, 12, 0.01),
        (3.0545, 12, 0.005),
        (1.6973, 30, 0.05),
        (6.3138, 1, 0.05),
        (2.3534, 3, 0.05),
        (2.0150, 5, 0.05),
    ],
)
def test_the_t_tail_matches_published_critical_values(t: float, df: int, p: float) -> None:
    assert t_sf(t, df) == pytest.approx(p, abs=2e-5)
    assert t_sf(-t, df) == pytest.approx(1 - p, abs=2e-5)


def test_the_t_distribution_is_a_distribution() -> None:
    for df in (1, 2, 7, 12):
        assert t_cdf_two_sided(0.0, df) == 0.0
        assert t_sf(0.0, df) == pytest.approx(0.5)
        assert t_cdf_two_sided(1e6, df) == pytest.approx(1.0, abs=1e-5)
        values = [t_sf(x, df) for x in np.linspace(-5, 5, 41)]
        assert all(np.diff(values) <= 0)
    assert t_sf(math.inf, 12) == 0.0
    assert math.isnan(t_sf(math.nan, 12))


def test_day_t_is_the_mean_over_its_standard_error() -> None:
    values = [1.0, 2.0, -0.5, 0.75, 1.25]
    out = day_t(values)
    array = np.asarray(values)
    assert out.days == 5 and out.positive_days == 4
    assert out.t == pytest.approx(array.mean() / (array.std(ddof=1) / math.sqrt(5)))
    assert out.p_one_sided == pytest.approx(t_sf(out.t, 4))
    assert day_t([1.0, math.nan, 2.0]).days == 2
    assert math.isnan(day_t([1.0]).t)


def test_holm_steps_down_and_stops_at_the_first_failure() -> None:
    assert holm({"a": 0.01, "b": 0.015, "c": 0.02, "d": 0.04}) == dict.fromkeys("abcd", True)
    # 0.012 > 0.05 / 3: b fails, and c, smaller than its own bar, fails with it.
    assert holm({"a": 0.001, "b": 0.02, "c": 0.024, "d": 0.5}) == {
        "a": True,
        "b": False,
        "c": False,
        "d": False,
    }
    assert holm({"a": 0.03, "b": math.nan}) == {"a": False, "b": False}
    assert holm({"a": 0.02, "b": math.nan}) == {"a": True, "b": False}
    assert holm({"a": 0.012, "b": 0.9, "c": 0.9, "d": 0.9})["a"] is True


def test_a_true_value_must_be_strictly_above_the_placebo_percentile() -> None:
    placebos = np.arange(100, dtype=float)
    level, above = exceeds_placebos(95.0, placebos)
    assert level == pytest.approx(94.05) and above
    assert not exceeds_placebos(94.05, placebos)[1]
    assert not exceeds_placebos(1.0, [math.nan])[1]


def test_the_state_shift_is_between_25_hours_and_13_days_less_an_hour() -> None:
    shifts = [state_shift_rows(seed) for seed in range(50)]
    assert shifts == [state_shift_rows(seed) for seed in range(50)]
    assert min(shifts) >= ROWS_PER_DAY + ROWS_PER_HOUR
    assert max(shifts) <= 12 * ROWS_PER_DAY + 23 * ROWS_PER_HOUR
    assert len(set(shifts)) > 45
    assert roll_values(np.arange(5), 2).tolist() == [3, 4, 0, 1, 2]


def test_the_breakeven_is_the_crossing_below_the_highest_positive_fee() -> None:
    fees = [-1.0, 0.0, 1.0, 2.0]
    assert breakeven_on_grid(fees, [3.0, 1.0, -1.0, -3.0]) == pytest.approx(0.5)
    assert breakeven_on_grid(fees, [3.0, 2.0, 1.0, 0.5]) == math.inf
    assert breakeven_on_grid(fees, [-3.0, -2.0, -1.0, -0.5]) == -math.inf
    assert math.isnan(breakeven_on_grid(fees, [math.nan] * 4))
    # Unsorted input is sorted first; a positive island below a loss is read from the top.
    assert breakeven_on_grid([2.0, 0.0, 1.0, -1.0], [-3.0, 1.0, -1.0, 3.0]) == pytest.approx(0.5)


def test_the_common_kill_conditions() -> None:
    assert k_pess([True, True], [True, False])
    assert not k_pess([True, True], [True, True])
    assert not k_pess([True, False], [False, False])  # nothing positive to kill
    assert k_dir(making=-1.0, net=2.0) and k_dir(making=0.0, net=2.0)
    assert not k_dir(making=-1.0, net=-2.0) and not k_dir(making=1.0, net=2.0)


def check(name: str, fired: bool, outcome: str = KILLED) -> KillCheck:
    return KillCheck("H", name, "test", 0.0, 0.0, fired, outcome)


def test_the_status_rule() -> None:
    assert status([check("K1", False), check("K2", False)], True) == ("candidate", [])
    assert status([check("K1", False)], False) == (INCONCLUSIVE, ["Holm"])
    assert status([check("K1", True), check("K5", True, INCONCLUSIVE)], True) == (KILLED, ["K1"])
    assert status([check("K1", False), check("K5", True, INCONCLUSIVE)], True) == (
        INCONCLUSIVE,
        ["K5"],
    )
