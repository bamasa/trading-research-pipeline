"""N_eff, the deflated Sharpe ratio and CSCV PBO, on series with known answers."""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from trading_research.evaluation.overfitting import (
    CSCV_BLOCKS,
    cscv_pbo,
    cscv_pbo_of_means,
    cscv_splits,
    deflated_sharpe_ratio,
    effective_trials,
    expected_maximum_sharpe,
    relative_rank_logit,
    sharpe_ratios,
    sharpe_variance,
)
from trading_research.evaluation.significance import minimum_detectable_effect, power_at_bar


def test_n_eff_is_one_for_copies_and_n_for_independent_trials() -> None:
    rng = np.random.default_rng(0)
    common = rng.normal(size=150)
    copies = pd.DataFrame({f"c{i}": common * (i + 1) for i in range(20)})
    assert effective_trials(20, copies) == pytest.approx(1.0, abs=0.05)
    independent = pd.DataFrame(rng.normal(size=(150, 20)))
    assert effective_trials(20, independent) == pytest.approx(20.0, rel=0.25)


def test_the_expected_maximum_matches_simulation() -> None:
    rng = np.random.default_rng(1)
    for n in (10, 100, 1000):
        simulated = rng.normal(0.0, 0.5, (4000, n)).max(axis=1).mean()
        assert expected_maximum_sharpe(n, 0.25) == pytest.approx(simulated, rel=0.05)
    assert expected_maximum_sharpe(1, 0.25) == 0.0
    assert expected_maximum_sharpe(100, 0.0) == 0.0


def test_a_single_trial_is_the_probabilistic_sharpe_ratio() -> None:
    values = np.random.default_rng(2).normal(0.2, 1.0, 200)
    out = deflated_sharpe_ratio(values, 1, 0.01)
    centred = values - values.mean()
    skew = np.mean(centred**3) / values.std() ** 3
    kurt = np.mean(centred**4) / values.std() ** 4
    sharpe = values.mean() / values.std(ddof=1)
    z = sharpe * math.sqrt(199) / math.sqrt(1 - skew * sharpe + (kurt - 1) / 4 * sharpe**2)
    assert out.benchmark == 0.0
    assert out.dsr == pytest.approx(NormalDist().cdf(z))
    assert out.kurtosis == pytest.approx(kurt) and out.days == 200


def test_more_trials_deflate_more() -> None:
    values = np.random.default_rng(3).normal(0.15, 1.0, 150)
    dsr = [deflated_sharpe_ratio(values, n, 0.01).dsr for n in (1, 10, 100, 2400)]
    assert all(np.diff(dsr) < 0)


def test_a_short_or_flat_series_has_no_dsr() -> None:
    assert math.isnan(deflated_sharpe_ratio([1.0, 2.0], 10, 0.01).dsr)
    assert math.isnan(deflated_sharpe_ratio([1.0, 1.0, 1.0], 10, 0.01).dsr)


def test_sharpe_ratios_per_column() -> None:
    panel = np.array([[1.0, 2.0, np.nan], [3.0, 2.0, 1.0], [2.0, 2.0, np.nan]])
    out = sharpe_ratios(panel)
    assert out[0] == pytest.approx(2.0)
    assert np.isnan(out[1]) and np.isnan(out[2])
    assert sharpe_variance(np.array([[1.0, 0.0, 2.0], [3.0, 1.0, 2.5], [2.0, 2.0, 2.0]])) > 0


def test_ten_blocks_give_252_splits() -> None:
    splits = cscv_splits(150, CSCV_BLOCKS)
    assert len(splits) == 252
    inside, outside = splits[0]
    assert len(inside) == len(outside) == 75
    assert sorted(np.concatenate([inside, outside]).tolist()) == list(range(150))
    with pytest.raises(ValueError):
        cscv_splits(150, 7)


def test_relative_rank_logit_signs() -> None:
    assert relative_rank_logit(2, np.array([1.0, 2.0, 3.0])) > 0
    assert relative_rank_logit(0, np.array([1.0, 2.0, 3.0])) < 0
    assert relative_rank_logit(1, np.array([1.0, 2.0, 3.0])) == pytest.approx(0.0)


def test_noise_overfits_half_the_time_and_an_edge_never() -> None:
    rng = np.random.default_rng(4)
    noise = rng.normal(0.0, 1.0, (150, 50))
    assert 0.3 < cscv_pbo_of_means(noise).pbo < 0.7
    edge = noise.copy()
    edge[:, 7] += 1.0
    out = cscv_pbo_of_means(edge)
    assert out.pbo == 0.0 and out.splits == 252


def test_a_reversing_trial_is_flagged() -> None:
    # Trial 0 wins on whichever half it is chosen on and loses on the other.
    values = np.zeros((20, 5))
    values[:, 0] = np.where(np.arange(20) % 4 < 2, 3.0, -3.0)
    values[:, 1:] = np.random.default_rng(5).normal(0.0, 0.1, (20, 4))
    assert cscv_pbo_of_means(values, blocks=10).pbo > 0.4


def test_a_split_without_a_choice_is_skipped() -> None:
    out = cscv_pbo(20, lambda _days: (None, np.arange(3)), lambda _days: np.zeros(3))
    assert out.skipped == 252 and out.splits == 0 and math.isnan(out.pbo)


def test_mde_has_80_percent_power_at_the_bar() -> None:
    effect = minimum_detectable_effect(2.0, 44)
    assert effect == pytest.approx(3.84 * 2.0 / math.sqrt(44))
    assert power_at_bar(effect, 2.0, 44) == pytest.approx(0.8, abs=0.001)
    assert power_at_bar(0.0, 2.0, 44) < 0.01
    assert math.isnan(minimum_detectable_effect(2.0, 0))


def test_min_track_record_is_where_the_psr_reaches_the_confidence() -> None:
    from trading_research.evaluation.overfitting import minimum_track_record_length

    rng = np.random.default_rng(7)
    series = rng.normal(0.2, 1.0, 400)
    record = minimum_track_record_length(series, threshold=0.0, confidence=0.95)
    # at exactly MinTRL periods the probabilistic Sharpe ratio is the confidence
    n = record.minimum
    spread = 1 - record.skewness * record.sharpe + (record.kurtosis - 1) / 4 * record.sharpe**2
    z = record.sharpe * math.sqrt(n - 1) / math.sqrt(spread)
    assert NormalDist().cdf(z) == pytest.approx(0.95)
    assert record.enough
    # normal returns: about (1.645 / SR)^2 periods
    assert n == pytest.approx(
        1 + (1.6449 / record.sharpe) ** 2 * (1 + record.sharpe**2 / 2), rel=0.1
    )


def test_min_track_record_is_undefined_below_the_threshold() -> None:
    from trading_research.evaluation.overfitting import minimum_track_record_length

    series = np.random.default_rng(8).normal(-0.1, 1.0, 200)
    assert math.isnan(minimum_track_record_length(series).minimum)
    assert not minimum_track_record_length(series).enough
    assert math.isnan(minimum_track_record_length([1.0, 1.0, 1.0]).minimum)


def test_adjusted_p_values_match_the_textbook() -> None:
    from trading_research.evaluation.overfitting import adjusted_p_values

    p = [0.01, 0.04, 0.03, 0.005]
    assert adjusted_p_values(p, "bonferroni") == pytest.approx([0.04, 0.16, 0.12, 0.02])
    # Holm: sorted 0.005, 0.01, 0.03, 0.04 times 4, 3, 2, 1, made monotone
    assert adjusted_p_values(p, "holm") == pytest.approx([0.03, 0.06, 0.06, 0.02])
    c = 1 + 1 / 2 + 1 / 3 + 1 / 4
    bhy = adjusted_p_values(p, "bhy")
    assert bhy[1] == pytest.approx(0.04 * c)
    assert bhy[3] == pytest.approx(min(0.005 * 4 * c, 0.01 * 2 * c))
    with pytest.raises(ValueError):
        adjusted_p_values(p, "sidak")


def test_one_trial_has_no_haircut_and_more_trials_cut_more() -> None:
    from trading_research.evaluation.overfitting import haircut_sharpe

    one = haircut_sharpe([0.3], periods=60, chosen=0, method="bonferroni")
    assert one.adjusted_sharpe == pytest.approx(0.3)
    assert one.haircut == pytest.approx(0.0, abs=1e-12)
    many = haircut_sharpe([0.3] + [0.0] * 19, periods=60, chosen=0, method="bonferroni")
    assert many.adjusted_p_value == pytest.approx(min(one.p_value * 20, 1.0))
    assert 0 < many.haircut < 1
    holm = haircut_sharpe([0.3] + [0.0] * 19, periods=60, chosen=0, method="holm")
    assert holm.adjusted_sharpe == pytest.approx(
        many.adjusted_sharpe
    )  # the best trial: same as Bonferroni
    weak = haircut_sharpe([0.05] * 20, periods=60, chosen=0, method="bonferroni")
    assert weak.adjusted_sharpe == 0.0 and weak.haircut == 1.0
