"""The placebo search on synthetic trades whose information is known."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_research.validation.choice import (
    CellDays,
    ChoiceError,
    StratumOutcome,
    neighbourhood_index,
    rerun,
)
from trading_research.validation.placebo_search import (
    NULL_A_DRAWS,
    HourCells,
    NullDraws,
    admission_iv,
    exceedances,
    global_flip,
    max_passing_count,
    null_a,
    null_a_signs,
    null_b,
    null_b_pairs,
    null_b_shifts,
    p_search,
    passes_iv,
    placebo_p,
    refit_replication,
    selection_premium,
    shift_labels_circularly,
)

CELLS, DAYS, HOURS_PER_DAY, NOTIONAL = 12, 30, 24, 1000.0


def _hours(skill: np.ndarray, seed: int = 0, cost_bp: float = 1.0) -> HourCells:
    """One trade per cell and hour. Every cell sees the hour's move ``m``; a cell
    takes the side of ``m`` with probability ``0.5 + skill`` and the other side
    otherwise. The long walk is ``m - cost`` and the short ``-m - cost``."""
    rng = np.random.default_rng(seed)
    hours = DAYS * HOURS_PER_DAY
    move = rng.normal(0.0, 10.0, hours)
    right = rng.random((len(skill), hours)) < 0.5 + skill[:, None]
    side = np.where(right, np.sign(move), -np.sign(move))
    long_bp, short_bp = move - cost_bp, -move - cost_bp
    kept = np.where(side > 0, long_bp, short_bp)
    flipped = np.where(side > 0, short_bp, long_bp)
    g = side * (long_bp - short_bp) / 2
    return HourCells(
        hour_day=np.repeat(np.arange(DAYS), HOURS_PER_DAY),
        days=DAYS,
        trades=np.ones((len(skill), hours), dtype=np.int64),
        notional=np.full((len(skill), hours), NOTIONAL),
        net_kept=kept * NOTIONAL / 1e4,
        net_flipped=flipped * NOTIONAL / 1e4,
        g=g,
    )


def _grid() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    table = pd.DataFrame({"x": np.tile(np.arange(6.0), 2), "s": ["C"] * 6 + ["R"] * 6})
    return neighbourhood_index(table, ["x"], by=["s"]), {"C": np.arange(6), "R": np.arange(6, 12)}


def test_hour_sums_become_daily_values() -> None:
    hours = _hours(np.zeros(CELLS))
    cells = hours.kept()
    assert cells.trades.shape == (CELLS, DAYS) and (cells.trades == 24).all()
    first = slice(0, 24)
    expected = hours.net_kept[3, first].sum() / (24 * NOTIONAL) * 1e4
    assert cells.net_bp[3, 0] == pytest.approx(expected)
    assert cells.g_bp[3, 0] == pytest.approx(hours.g[3, first].mean())


def test_flipping_every_hour_swaps_the_walks_and_negates_g() -> None:
    hours = _hours(np.full(CELLS, 0.2))
    kept, flipped = hours.kept(), hours.under_signs(-np.ones(hours.hours, dtype=int))
    np.testing.assert_allclose(flipped.g_bp, -kept.g_bp)
    expected = hours.net_flipped[:, :24].sum(axis=1) / (24 * NOTIONAL) * 1e4
    np.testing.assert_allclose(flipped.net_bp[:, 0], expected)


def test_a_day_without_hours_has_no_trade() -> None:
    hours = HourCells(
        hour_day=np.array([0, 0, 2]),
        days=3,
        trades=np.ones((1, 3), dtype=np.int64),
        notional=np.full((1, 3), 100.0),
        net_kept=np.array([[1.0, 2.0, 3.0]]),
        net_flipped=np.zeros((1, 3)),
        g=np.array([[1.0, 3.0, 5.0]]),
    )
    cells = hours.kept()
    assert cells.trades.tolist() == [[2, 0, 1]]
    assert np.isnan(cells.net_bp[0, 1])
    assert cells.net_bp[0, 0] == pytest.approx(3.0 / 200.0 * 1e4)
    assert cells.g_bp[0, 2] == 5.0


def test_bad_hour_inputs_are_refused() -> None:
    ok = _hours(np.zeros(2))
    with pytest.raises(ChoiceError):
        ok.under_signs(np.zeros(ok.hours))
    with pytest.raises(ChoiceError):
        HourCells(np.array([1, 0]), 2, *[np.ones((1, 2))] * 5)


def test_the_signs_are_seeded_per_draw() -> None:
    a, b = null_a_signs(0, 500), null_a_signs(1, 500)
    assert set(np.unique(a)) == {-1, 1}
    assert np.array_equal(a, null_a_signs(0, 500)) and not np.array_equal(a, b)
    assert abs(a.mean()) < 0.15
    assert NULL_A_DRAWS == 999


def test_a_planted_edge_beats_null_a_and_noise_does_not() -> None:
    index, strata = _grid()
    skill = np.zeros(CELLS)
    skill[2:5] = 0.15  # an informative region in stratum C
    hours = _hours(skill, seed=1)
    found = rerun(hours.kept(), strata, index)
    assert found["C"].cell in (2, 3, 4) and found["C"].a > 3
    draws = null_a(hours, strata, index, draws=99)
    assert draws.a.shape == (99, 2) and draws.strata == ("C", "R")
    assert exceedances(draws.maximum, found["C"].a) == 0

    noise = _hours(np.zeros(CELLS), seed=2)
    found_noise = rerun(noise.kept(), strata, index)
    draws_noise = null_a(noise, strata, index, draws=99)
    assert exceedances(draws_noise.maximum, max(o.a for o in found_noise.values())) > 5


def test_a_fixed_cell_makes_no_choice_under_the_null() -> None:
    index, strata = _grid()
    hours = _hours(np.zeros(CELLS), seed=3)
    draws = null_a(hours, strata, index, draws=5, fixed={"C": 0, "R": 6})
    for b in range(5):
        cells = hours.under_signs(null_a_signs(b, hours.hours))
        assert draws.a[b, 0] == pytest.approx(rerun(cells, strata, index, {"C": 0, "R": 6})["C"].a)


def test_the_global_flip_turns_an_edge_into_an_anti_signal() -> None:
    index, strata = _grid()
    skill = np.zeros(CELLS)
    skill[2:5] = 0.15
    hours = _hours(skill, seed=1)
    flipped = global_flip(hours, strata, index)
    assert flipped["C"].cell not in (2, 3, 4)  # the region now loses net
    assert flipped["C"].a < rerun(hours.kept(), strata, index)["C"].a - 5


def test_137_shifts_on_150_days() -> None:
    shifts = null_b_shifts()
    assert len(shifts) == 137 and shifts[0] == 7 and shifts[-1] == 143
    with pytest.raises(ChoiceError):
        null_b_shifts(days=100)


def test_shift_pairs_follow_the_circle_and_count_exclusions() -> None:
    usable = np.ones(10, dtype=bool)
    pairs = null_b_pairs(usable, 3)
    assert pairs.target.tolist() == list(range(10))
    assert pairs.source.tolist() == [7, 8, 9, 0, 1, 2, 3, 4, 5, 6]
    assert pairs.excluded == 0
    usable[4] = False  # excluded as a source (to day 7) and as a target (from day 1)
    pairs = null_b_pairs(usable, 3)
    assert pairs.excluded == 2
    assert 4 not in pairs.source and 4 not in pairs.target


def test_null_b_reruns_the_choice_on_every_shift() -> None:
    index, strata = _grid()
    base = _hours(np.zeros(CELLS), seed=4).kept()
    seen: list[int] = []

    def shifted(s: int) -> CellDays:
        seen.append(s)
        roll = s % base.days
        return CellDays(base.trades, base.net_bp, np.roll(base.g_bp, roll, axis=0))

    draws = null_b(shifted, strata, index)
    assert seen == list(range(7, 144))
    assert draws.replications == 137
    assert draws.a[0, 0] == pytest.approx(rerun(shifted(7), strata, index)["C"].a)


def test_the_registered_boundary_at_137_shifts() -> None:
    # At R_D = 1: five shifts at or above A_j pass, six fail.
    assert max_passing_count(137) == 5
    assert passes_iv(0, 999, 5, 137)
    assert not passes_iv(0, 999, 6, 137)
    assert placebo_p(5, 137) == pytest.approx(6 / 138)
    assert placebo_p(6, 137) > 0.05


def test_the_boundary_at_999_draws_is_exact() -> None:
    # (1 + 49) / 1000 is 0.05 exactly and passes; 50 fails.
    assert max_passing_count(999) == 49
    assert passes_iv(49, 999, 0, 137)
    assert not passes_iv(50, 999, 0, 137)


def test_more_d_runs_tighten_the_bar() -> None:
    assert max_passing_count(137, runs=2) == 2
    assert max_passing_count(999, runs=2) == 24
    assert not passes_iv(3, 999, 3, 137, runs=2)
    with pytest.raises(ChoiceError):
        passes_iv(0, 999, 0, 137, runs=0)


def test_exceedances_count_ties_and_refuse_nan() -> None:
    assert exceedances(np.array([1.0, 2.0, 2.0, 3.0]), 2.0) == 3
    with pytest.raises(ChoiceError):
        exceedances(np.array([1.0, np.nan]), 0.5)


def _null(values: list[float]) -> NullDraws:
    return NullDraws(("C", "R"), np.column_stack([values, np.zeros(len(values))]))


def test_admission_counts_against_the_maximum_over_strata() -> None:
    a_null = _null([0.5] * 990 + [3.0] * 9)
    b_null = _null([0.5] * 132 + [3.0] * 5)
    observed = {"C": StratumOutcome(4, 2.9), "R": StratumOutcome(None, 0.0)}
    out = admission_iv(observed, a_null, b_null)
    assert out["C"].count_a == 9 and out["C"].count_b == 5 and out["C"].passes
    assert out["C"].p == pytest.approx(6 / 138)
    assert not out["R"].passes and out["R"].p == 1.0
    refit = admission_iv(observed, a_null, b_null, refit={"C": 2.9})
    assert not refit["C"].passes  # A must beat the refit strictly
    assert admission_iv(observed, a_null, b_null, refit={"C": 2.0})["C"].passes
    assert not admission_iv(observed, a_null, b_null, runs=2)["C"].passes
    assert p_search(out) == pytest.approx(6 / 138)
    assert p_search(out, runs=30) == 1.0


def test_the_selection_premium() -> None:
    median, premium = selection_premium(2.5, np.array([1.0, 2.0, 3.0]))
    assert median == 2.0 and premium == 0.5


def test_labels_move_28_days_later_around_the_span() -> None:
    stamps = pd.date_range("2025-01-01", periods=60, freq="D")
    labels = pd.Series(np.arange(60.0), index=stamps, name="y")
    shifted = shift_labels_circularly(labels, "2025-01-01", "2025-03-02")
    assert shifted.name == "y" and shifted.index.equals(stamps)
    assert shifted.iloc[28] == 0.0  # 2025-01-29 takes 2025-01-01's label
    assert shifted.iloc[0] == 32.0  # 2025-01-01 wraps to 2025-02-02 (60 - 28)
    assert sorted(shifted.tolist()) == labels.tolist()


def test_a_missing_source_stamp_gives_no_label() -> None:
    stamps = pd.date_range("2025-01-01", periods=60, freq="D").delete(5)
    labels = pd.Series(1.0, index=stamps)
    shifted = shift_labels_circularly(labels, "2025-01-01", "2025-03-02")
    assert shifted.isna().sum() == 1
    assert np.isnan(shifted.loc["2025-02-03"])  # its source, 2025-01-06, is missing


def test_intraday_stamps_keep_their_time_of_day() -> None:
    stamps = pd.date_range("2025-01-01", "2025-02-28 23:00", freq="h")
    labels = pd.Series(np.arange(len(stamps), dtype=float), index=stamps)
    shifted = shift_labels_circularly(labels, "2025-01-01", "2025-03-01")
    when = pd.Timestamp("2025-02-10 13:00")
    assert shifted.loc[when] == labels.loc[pd.Timestamp("2025-01-13 13:00")]


def test_bad_label_spans_are_refused() -> None:
    labels = pd.Series(1.0, index=pd.date_range("2025-01-01", periods=10, freq="D"))
    with pytest.raises(ChoiceError):
        shift_labels_circularly(labels, "2025-01-01", "2025-01-11")  # 28 days > span
    with pytest.raises(ChoiceError):
        shift_labels_circularly(labels, "2025-01-02", "2025-03-01")  # a label before start
    with pytest.raises(ChoiceError):
        shift_labels_circularly(pd.Series([1.0, 2.0]), "2025-01-01", "2025-03-01")


def test_the_refit_replication_reruns_stratum_c() -> None:
    index, strata = _grid()
    refitted = _hours(np.zeros(CELLS), seed=5).kept()
    asked: list[int] = []

    def refit(days: int) -> CellDays:
        asked.append(days)
        return refitted

    out = refit_replication(refit, strata["C"], index)
    assert asked == [28]
    assert out == rerun(refitted, {"C": strata["C"]}, index)["C"]
