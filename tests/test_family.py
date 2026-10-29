"""The family of five, Holm at m = 5, and the bar on a daily series."""

from __future__ import annotations

import numpy as np
import pytest

from trading_research.evaluation.family import (
    FAMILY,
    MIN_DAYS,
    bar,
    family_holm,
    holm_first_step_t,
    k_placebo,
    passes_alone,
    slot_p_value,
)
from trading_research.market_making.verdicts import day_t, t_sf


def test_the_family_is_fixed() -> None:
    assert FAMILY == ("S25", "S-C", "S-R", "S-P", "S-M")
    assert MIN_DAYS == {"H": 20, "F": 10, "P": 10}


@pytest.mark.parametrize(("df", "t"), [(43, 2.42), (19, 2.54), (9, 2.82)])
def test_holm_s_first_step_at_m_5_is_below_the_t_bar(df: int, t: float) -> None:
    assert holm_first_step_t(df) == pytest.approx(t, abs=0.005)
    assert holm_first_step_t(df) < 3.0


def test_ten_days_at_t_3() -> None:
    assert t_sf(3.0, 9) == pytest.approx(0.0075, abs=0.0001)


def test_the_p_value_is_the_day_level_one_sided_p() -> None:
    values = np.random.default_rng(0).normal(0.5, 1.0, 30)
    assert slot_p_value(values) == pytest.approx(day_t(values).p_one_sided)
    assert slot_p_value([2.0]) == 1.0
    assert slot_p_value([]) == 1.0
    assert slot_p_value([0.0, 0.0]) == 1.0  # no defined t
    assert slot_p_value([1.0, 1.0]) == 0.0  # as day_t: an infinite t


def test_missing_members_enter_with_p_1_and_keep_m_at_5() -> None:
    # 0.011 passes Holm's first step only at m <= 4; with m = 5 it fails.
    alone = family_holm({"S25": 0.011})
    assert set(alone) == set(FAMILY) and not any(alone.values())
    out = family_holm({"S25": 0.009, "S-C": None, "S-R": float("nan")})
    assert out["S25"] and not out["S-C"] and not out["S-R"]
    with pytest.raises(ValueError):
        family_holm({"S25-Z": 0.001})


def test_the_bar_names_each_condition() -> None:
    rng = np.random.default_rng(1)
    strong = rng.normal(1.0, 1.0, 44)
    out = bar(strong, passes_holm=True, min_days=MIN_DAYS["H"])
    assert out.passes and out.t >= 3 and out.newey_west_t >= 2 and out.median > 0
    short = bar(strong[:12], passes_holm=True, min_days=MIN_DAYS["H"])
    assert not short.days_ok and not short.passes
    assert not bar(strong, passes_holm=False, min_days=20).passes


def test_a_positive_mean_with_a_negative_median_fails() -> None:
    values = np.array([-0.1] * 25 + [5.0] * 19)
    out = bar(values, passes_holm=True, min_days=20)
    assert out.mean > 0 and out.median < 0 and not out.median_ok


def test_k_placebo_fires_at_or_below_the_95th_percentile() -> None:
    placebos = np.arange(100.0)
    level = float(np.percentile(placebos, 95))
    assert k_placebo(level, placebos)
    assert not k_placebo(level + 0.01, placebos)
    assert k_placebo(5.0, [])


def test_s25_z_is_tested_alone_at_alpha_over_6() -> None:
    assert passes_alone([0.008, 0.001])
    assert not passes_alone([0.009, 0.001])
    assert not passes_alone([float("nan")])
