"""Tests for the retraining schedule.

The property that matters most is the boundary: a schedule that trains on any
day it later trades produces good numbers and means nothing. That is checked
directly on the windows rather than through a result, because a leak of one day
in twenty moves a metric by less than its noise and no summary would show it.

The rest pin the arithmetic — how many refits a schedule implies, what happens
when a window cannot be scored, and that the choice is made on validation and
the test span is touched once.
"""

from __future__ import annotations

from datetime import date, timedelta
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from trading_research.backtest.costs import TakerCosts
from trading_research.pipeline.retraining import (
    DayCache,
    positive_window_share,
    prepared_evaluator,
    summarise,
    trade_weighted,
)
from trading_research.validation.retrain import (
    DEFAULT_GRID,
    RetrainSchedule,
    ScheduleError,
    ScheduleResult,
    days_between,
    describe_grid,
    expand_grid,
    mean_of_windows,
    run_schedule,
    search,
    split_span,
)


def days(n: int, start: date = date(2024, 2, 1)) -> list[date]:
    return [start + timedelta(days=i) for i in range(n)]


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------


def test_training_always_ends_before_the_apply_window_opens() -> None:
    """The one property a leak would hide: no overlap, not even a single day."""
    for schedule in DEFAULT_GRID:
        for (train_start, train_end), (apply_start, apply_end) in schedule.windows(days(40)):
            assert train_start <= train_end < apply_start <= apply_end


def test_window_lengths_are_what_the_schedule_says() -> None:
    schedule = RetrainSchedule(train_days=7, apply_days=3)
    for (ts, te), (as_, ae) in schedule.windows(days(30)):
        assert (te - ts).days + 1 == 7
        assert (ae - as_).days + 1 == 3


def test_step_defaults_to_the_apply_length_so_coverage_has_no_gaps() -> None:
    """Contiguous apply windows are what a live system produces; check it."""
    schedule = RetrainSchedule(train_days=5, apply_days=2)
    covered = [w[1] for w in schedule.windows(days(20))]
    for (_, previous_end), (next_start, _) in pairwise(covered):
        assert next_start == previous_end + timedelta(days=1)


def test_a_smaller_step_overlaps_rather_than_skipping() -> None:
    overlapping = RetrainSchedule(train_days=5, apply_days=3, step_days=1)
    contiguous = RetrainSchedule(train_days=5, apply_days=3)
    assert overlapping.count(days(20)) > contiguous.count(days(20))


def test_refit_count_matches_the_windows_produced() -> None:
    schedule = RetrainSchedule(train_days=7, apply_days=1)
    assert schedule.count(days(15)) == len(list(schedule.windows(days(15))))
    # 15 days, 7 to train and 1 to act: the first apply day is the 8th, the last
    # is the 15th, so eight refits.
    assert schedule.count(days(15)) == 8


def test_too_few_days_is_refused_rather_than_silently_empty() -> None:
    with pytest.raises(ScheduleError, match="needs 15 days"):
        list(RetrainSchedule(train_days=14, apply_days=1).windows(days(10)))
    assert RetrainSchedule(train_days=14).count(days(10)) == 0


def test_days_are_sorted_before_use() -> None:
    shuffled = list(reversed(days(12)))
    (train_start, _), _ = next(RetrainSchedule(train_days=3).windows(shuffled))
    assert train_start == date(2024, 2, 1)


def test_invalid_schedules_are_rejected() -> None:
    with pytest.raises(ValueError, match="train_days"):
        RetrainSchedule(train_days=0)
    with pytest.raises(ValueError, match="apply_days"):
        RetrainSchedule(train_days=5, apply_days=0)
    with pytest.raises(ValueError, match="step_days"):
        RetrainSchedule(train_days=5, step_days=0)


def test_label_names_the_three_numbers() -> None:
    assert RetrainSchedule(train_days=7, apply_days=1).label == "train7/apply1/step1"
    assert RetrainSchedule(train_days=7, apply_days=3, step_days=1).label == "train7/apply3/step1"


# ---------------------------------------------------------------------------
# Running a schedule
# ---------------------------------------------------------------------------


def counting_evaluator(value: float = 1.0):
    calls: list[tuple] = []

    def evaluate(train, apply):
        calls.append((train, apply))
        return {"net_per_trade_bp": value, "trades": 10.0}

    return evaluate, calls


def test_a_schedule_scores_every_window_it_produces() -> None:
    evaluate, calls = counting_evaluator()
    result = run_schedule(RetrainSchedule(train_days=5, apply_days=1), days(12), evaluate)
    assert len(calls) == 7
    assert result.metrics["windows"] == 7.0
    assert result.metrics["net_per_trade_bp"] == pytest.approx(1.0)


def test_a_window_that_cannot_be_scored_is_skipped_not_fatal() -> None:
    """A schedule failing half its windows is a finding; an exception hides it."""

    def flaky(train, apply):
        if apply[0].day % 2 == 0:
            raise RuntimeError("not enough rows")
        return {"net_per_trade_bp": 2.0}

    result = run_schedule(RetrainSchedule(train_days=3, apply_days=1), days(13), flaky)
    assert result.metrics["windows_skipped"] > 0
    assert result.metrics["windows"] > 0
    assert result.metrics["windows"] + result.metrics["windows_skipped"] == result.refits


def test_a_schedule_that_fails_everywhere_is_an_error() -> None:
    def always_fails(train, apply):
        raise RuntimeError("no")

    with pytest.raises(ScheduleError, match="every window failed"):
        run_schedule(RetrainSchedule(train_days=3), days(10), always_fails)


def test_metrics_are_averaged_over_windows() -> None:
    values = iter([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

    def varying(train, apply):
        return {"net_per_trade_bp": next(values)}

    result = run_schedule(RetrainSchedule(train_days=5, apply_days=1), days(12), varying)
    assert result.metrics["net_per_trade_bp"] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def test_the_best_schedule_on_validation_is_the_one_chosen() -> None:
    def by_train_days(train, apply):
        length = (train[1] - train[0]).days + 1
        return {"net_per_trade_bp": float(length)}

    outcome = search(days(60), by_train_days, minimum_windows=3)
    assert outcome.chosen.train_days == 21


def test_a_schedule_with_too_few_windows_is_not_eligible() -> None:
    """Best-of-two windows is one lucky window with a decimal point."""

    def favour_the_longest(train, apply):
        return {"net_per_trade_bp": float((train[1] - train[0]).days)}

    # 24 days: train21/apply1 yields 3 windows, so requiring 5 rules it out.
    outcome = search(days(24), favour_the_longest, minimum_windows=5)
    assert outcome.chosen.train_days < 21
    skipped = outcome.validation.loc[outcome.validation["schedule"] == "train21/apply1/step1"]
    assert "fewer than 5 windows" in str(skipped["skipped"].iloc[0])


def test_the_test_span_is_scored_only_with_the_chosen_schedule() -> None:
    seen: list[tuple] = []

    def record(train, apply):
        seen.append((train, apply))
        return {"net_per_trade_bp": float(train[1].day)}

    validation, test = split_span(days(80))
    outcome = search(validation, record, test_days=test)
    assert outcome.test is not None

    test_set = set(test)
    on_test = [w for w in seen if w[1][0] in test_set]
    lengths = {(w[0][1] - w[0][0]).days + 1 for w in on_test}
    assert lengths == {outcome.chosen.train_days}


def test_the_test_span_is_traded_from_its_first_day() -> None:
    """Found by running it: the test span used to lose its first `train_days`.

    Worse than wasteful — the amount of test data would depend on which schedule
    won, so two schedules would be compared over different periods.
    """
    traded: list[date] = []

    def record(train, apply):
        traded.append(apply[0])
        return {"net_per_trade_bp": 1.0}

    validation, test = split_span(days(40))
    outcome = search(validation, record, minimum_windows=3, test_days=test)
    assert outcome.test is not None
    assert min(w["apply_start"] for w in outcome.test.per_window) == min(test)


def test_warm_up_days_are_used_for_training_only_never_scored() -> None:
    """The training window may reach into validation; the scored days may not."""
    scored: list[date] = []

    def record(train, apply):
        scored.append(apply[0])
        return {"net_per_trade_bp": 1.0}

    validation, test = split_span(days(40))
    outcome = search(validation, record, minimum_windows=3, test_days=test)
    assert outcome.test is not None
    assert all(w["apply_start"] in set(test) for w in outcome.test.per_window)
    assert all(w["apply_end"] in set(test) for w in outcome.test.per_window)


def test_windows_outside_the_scored_set_are_not_counted_as_failures() -> None:
    """A window skipped for being out of range is not evidence about a schedule."""
    schedule = RetrainSchedule(train_days=3, apply_days=1)
    inside = set(days(10)[5:])
    evaluate, _ = counting_evaluator()
    result = run_schedule(schedule, days(10), evaluate, apply_within=inside)
    assert result.metrics["windows_skipped"] == 0.0
    assert result.metrics["windows"] == len(inside)


def test_history_lets_a_long_window_trade_the_whole_validation_span() -> None:
    """Otherwise long and short schedules are compared over different periods,
    and the comparison measures span length instead of staleness."""
    scored: list[date] = []

    def record(train, apply):
        scored.append(apply[0])
        return {"net_per_trade_bp": 1.0}

    validation = days(20, start=date(2024, 3, 1))
    history = days(21, start=date(2024, 2, 9))
    search(
        validation,
        record,
        grid=(RetrainSchedule(train_days=21, apply_days=1),),
        history=history,
        minimum_windows=3,
    )
    assert min(scored) == min(validation)
    assert len(scored) == len(validation)


def test_history_days_are_never_scored() -> None:
    scored: list[date] = []

    def record(train, apply):
        scored.append(apply[0])
        return {"net_per_trade_bp": 1.0}

    validation = days(20, start=date(2024, 3, 1))
    history = days(21, start=date(2024, 2, 9))
    search(
        validation,
        record,
        grid=(RetrainSchedule(train_days=7, apply_days=1),),
        history=history,
        minimum_windows=3,
    )
    assert not set(scored) & set(history)


def test_a_schedule_too_long_for_the_history_available_is_still_ruled_out() -> None:
    def record(train, apply):
        return {"net_per_trade_bp": 1.0}

    outcome = search(
        days(10, start=date(2024, 3, 1)),
        record,
        grid=(RetrainSchedule(train_days=21), RetrainSchedule(train_days=3)),
        minimum_windows=3,
    )
    assert outcome.chosen.train_days == 3


def test_test_days_are_untouched_when_not_supplied() -> None:
    evaluate, calls = counting_evaluator()
    outcome = search(days(60), evaluate)
    assert outcome.test is None
    assert all(apply[0] in set(days(60)) for _, apply in calls)


def test_an_objective_nothing_reports_is_an_error() -> None:
    evaluate, _ = counting_evaluator()
    with pytest.raises(ScheduleError, match="sharpe"):
        search(days(60), evaluate, objective="sharpe")


def test_summary_states_the_choice_and_the_test_score() -> None:
    evaluate, _ = counting_evaluator(value=3.0)
    validation, test = split_span(days(80))
    outcome = search(validation, evaluate, test_days=test)
    text = outcome.summary()
    assert outcome.chosen.label in text
    assert "test net_per_trade_bp = 3.000" in text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_split_is_chronological_and_disjoint() -> None:
    validation, test = split_span(days(40), validation_fraction=0.5)
    assert len(validation) == 20
    assert max(validation) < min(test)
    assert not set(validation) & set(test)


def test_a_span_too_short_to_split_is_refused() -> None:
    with pytest.raises(ScheduleError, match="cannot be split"):
        split_span(days(3))


def test_an_impossible_fraction_is_rejected() -> None:
    with pytest.raises(ValueError, match="validation_fraction"):
        split_span(days(40), validation_fraction=1.5)


def test_days_between_is_inclusive() -> None:
    assert len(days_between(date(2024, 2, 1), date(2024, 2, 5))) == 5
    with pytest.raises(ValueError, match="before start"):
        days_between(date(2024, 2, 5), date(2024, 2, 1))


def test_expand_grid_builds_every_combination() -> None:
    grid = expand_grid([3, 7], [1, 3])
    assert len(grid) == 4
    assert {s.label for s in grid} == {
        "train3/apply1/step1",
        "train3/apply3/step3",
        "train7/apply1/step1",
        "train7/apply3/step3",
    }


def test_grid_description_lists_the_minimum_days_each_needs() -> None:
    table = describe_grid()
    assert set(table.columns) == {
        "label",
        "train_days",
        "apply_days",
        "step_days",
        "minimum_days",
    }
    assert (table["minimum_days"] == table["train_days"] + table["apply_days"]).all()


# ---------------------------------------------------------------------------
# The evaluator built from prepared data
# ---------------------------------------------------------------------------


def write_prepared(directory, n_days: int = 12, rows: int = 900) -> None:
    """Prepared days with a feature that genuinely predicts the forward move."""
    rng = np.random.default_rng(7)
    directory.mkdir(parents=True, exist_ok=True)
    for d in range(n_days):
        day = date(2024, 2, 1) + timedelta(days=d)
        signal = rng.normal(size=rows)
        forward = signal * 8.0 + rng.normal(0, 6.0, rows)
        pd.DataFrame(
            {
                "timestamp": pd.date_range(f"{day}T00:00", periods=rows, freq="1min", tz="UTC"),
                "signal": signal,
                "noise": rng.normal(size=rows),
                "forward_bp": forward,
                "spread_bp_now": np.full(rows, 1.0),
            }
        ).to_parquet(directory / f"{day}.parquet", index=False)


@pytest.fixture
def prepared(tmp_path):
    directory = tmp_path / "prepared"
    write_prepared(directory)
    return directory


def test_the_evaluator_reads_only_the_days_it_was_given(prepared) -> None:
    """The guarantee the whole file layout exists for, checked at this level too."""
    cache = DayCache(prepared)
    frame = cache.span(date(2024, 2, 1), date(2024, 2, 3))
    assert set(frame["timestamp"].dt.date) == {
        date(2024, 2, 1),
        date(2024, 2, 2),
        date(2024, 2, 3),
    }


def test_a_full_sweep_runs_end_to_end_on_prepared_data(prepared) -> None:
    evaluate = prepared_evaluator(
        prepared,
        ["signal", "noise"],
        threshold_bp=6.0,
        hold_periods=1,
        costs=TakerCosts(fee_bp_per_side=0.1, slippage_bp=0.0),
        model="logistic",
    )
    grid = expand_grid([3, 5], [1])
    outcome = search(days(12), evaluate, grid=grid, minimum_windows=3)
    assert outcome.chosen in grid
    assert {"trades", "hit_rate"} <= set(outcome.validation.columns)


def test_the_threshold_is_swept_inside_the_training_block(prepared) -> None:
    """Not on the days being traded: that would be choosing on the result."""
    evaluate = prepared_evaluator(
        prepared,
        ["signal"],
        threshold_bp=6.0,
        hold_periods=1,
        costs=TakerCosts(fee_bp_per_side=0.1, slippage_bp=0.0),
        inner_validation_fraction=0.25,
    )
    metrics = evaluate((date(2024, 2, 1), date(2024, 2, 5)), (date(2024, 2, 6), date(2024, 2, 6)))
    # Five days of 900 rows, a quarter reserved, and rows without an outcome
    # dropped: the fit sees roughly three quarters and never the whole block.
    assert metrics["train_rows"] < 5 * 900 * 0.8
    assert "confidence" in metrics


def test_a_training_block_with_one_class_is_refused(prepared) -> None:
    evaluate = prepared_evaluator(
        prepared,
        ["signal"],
        threshold_bp=1e6,  # nothing clears it, so every row is HOLD
        hold_periods=1,
        costs=TakerCosts(),
    )
    with pytest.raises(Exception, match="one class"):
        evaluate((date(2024, 2, 1), date(2024, 2, 5)), (date(2024, 2, 6), date(2024, 2, 6)))


def test_a_per_trade_figure_is_weighted_by_trades_not_by_windows() -> None:
    """The bug this caught: +15 bp per trade reported alongside a -37 bp total.

    One window taking a single lucky trade must not outweigh one taking thirty
    losing ones — and since the schedules being compared differ precisely in how
    much they trade, averaging by window made the objective reward inactivity.
    """
    windows = [
        {"trades": 1.0, "net_bp": 50.0, "net_per_trade_bp": 50.0},
        {"trades": 30.0, "net_bp": -300.0, "net_per_trade_bp": -10.0},
    ]
    naive = mean_of_windows(windows)
    weighted = trade_weighted(windows)

    assert naive["net_per_trade_bp"] == pytest.approx(20.0)  # what looked true
    assert weighted["net_per_trade_bp"] == pytest.approx(-250.0 / 31)  # what is true
    assert weighted["net_bp"] == pytest.approx(-250.0)  # totals add, not average
    assert weighted["net_per_trade_bp_window_mean"] == pytest.approx(20.0)  # kept, visibly


def test_the_hit_rate_is_weighted_by_trades_too() -> None:
    windows = [
        {"trades": 1.0, "hit_rate": 1.0},
        {"trades": 99.0, "hit_rate": 0.0},
    ]
    assert trade_weighted(windows)["hit_rate"] == pytest.approx(0.01)


def test_a_schedule_that_took_no_trades_does_not_divide_by_zero() -> None:
    result = trade_weighted([{"trades": 0.0, "net_bp": 0.0, "net_per_trade_bp": float("nan")}])
    assert np.isnan(result["net_per_trade_bp"])


def test_windows_that_never_traded_do_not_count_against_a_schedule() -> None:
    """Ten idle windows of twenty-five turned 47% positive into 28%."""
    result = ScheduleResult(
        schedule=RetrainSchedule(train_days=7),
        refits=4,
        metrics={},
        per_window=[
            {"net_bp": 10.0, "trades": 3.0},
            {"net_bp": -5.0, "trades": 2.0},
            {"net_bp": 0.0, "trades": 0.0},
            {"net_bp": 0.0, "trades": 0.0},
        ],
    )
    assert positive_window_share(result) == pytest.approx(0.5)


def test_a_schedule_that_never_traded_at_all_reports_nothing_rather_than_zero() -> None:
    result = ScheduleResult(
        schedule=RetrainSchedule(train_days=7),
        refits=2,
        metrics={},
        per_window=[{"net_bp": 0.0, "trades": 0.0}],
    )
    assert np.isnan(positive_window_share(result))


def test_summarise_orders_by_the_objective(prepared) -> None:
    table = pd.DataFrame(
        {
            "schedule": ["a", "b", "c"],
            "refits": [5, 5, 5],
            "net_per_trade_bp": [-1.0, 3.0, 1.0],
            "trades": [10.0, 20.0, 30.0],
        }
    )
    ordered = summarise(table)
    assert list(ordered["schedule"]) == ["b", "c", "a"]
