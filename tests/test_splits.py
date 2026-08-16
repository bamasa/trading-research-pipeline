"""Tests for chronological splits and labels.

The tests that matter most are the ones about purging. A split that merely
looks chronological — blocks in order, no shuffling — still leaks if the label
horizon reaches across a boundary, and nothing about the resulting numbers
looks wrong. So the property is checked directly: no row used for training may
have a label determined by a price inside validation or test.
"""

from __future__ import annotations

from datetime import date, timedelta
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from trading_research.data.schema import mid_price
from trading_research.labels.directional import (
    BUY,
    HOLD,
    SELL,
    DirectionalLabel,
    class_balance,
    round_trip_cost_bp,
    threshold_for_share,
)
from trading_research.validation.splits import (
    Fold,
    SplitError,
    WalkForwardSpec,
    available_days,
    describe,
    fold_masks,
    walk_forward,
)

SPEC = WalkForwardSpec(train_days=14, validation_days=7, test_days=7, step_days=1, n_folds=7)


def days_from(start: date, n: int) -> list[date]:
    return [start + timedelta(days=i) for i in range(n)]


@pytest.fixture
def schedule() -> list[Fold]:
    return walk_forward(days_from(date(2024, 2, 1), 38), SPEC)


# ---------------------------------------------------------------------------
# Schedule shape
# ---------------------------------------------------------------------------


def test_produces_the_requested_number_of_folds(schedule: list[Fold]) -> None:
    assert len(schedule) == 7


def test_block_lengths_match_the_specification(schedule: list[Fold]) -> None:
    for fold in schedule:
        assert (fold.train[1] - fold.train[0]).days + 1 == 14
        assert (fold.validation[1] - fold.validation[0]).days + 1 == 7
        assert (fold.test[1] - fold.test[0]).days + 1 == 7


def test_blocks_are_in_order_and_adjacent(schedule: list[Fold]) -> None:
    """Validation follows training, test follows validation, with no overlap."""
    for fold in schedule:
        assert fold.validation[0] == fold.train[1] + timedelta(days=1)
        assert fold.test[0] == fold.validation[1] + timedelta(days=1)


def test_each_fold_starts_one_day_later(schedule: list[Fold]) -> None:
    for earlier, later in pairwise(schedule):
        assert later.train[0] == earlier.train[0] + timedelta(days=1)
        assert later.test[0] == earlier.test[0] + timedelta(days=1)


def test_first_and_last_fold_cover_the_expected_span(schedule: list[Fold]) -> None:
    assert schedule[0].train[0] == date(2024, 2, 1)
    assert schedule[0].test[1] == date(2024, 2, 28)
    assert schedule[-1].test[1] == date(2024, 3, 5)


def test_required_days_is_reported_correctly() -> None:
    # 28-day window, six one-day steps after the first fold.
    assert SPEC.window_days == 28
    assert SPEC.required_days == 34


def test_too_few_days_is_refused_with_the_shortfall(schedule: list[Fold]) -> None:
    with pytest.raises(SplitError, match="need 34 contiguous days"):
        walk_forward(days_from(date(2024, 2, 1), 33), SPEC)


def test_a_gap_in_the_days_is_refused(schedule: list[Fold]) -> None:
    """A fold spanning a gap covers less market than its label claims."""
    days = days_from(date(2024, 2, 1), 40)
    del days[20]
    with pytest.raises(SplitError, match="not contiguous"):
        walk_forward(days, SPEC)


def test_invalid_specification_is_rejected() -> None:
    with pytest.raises(ValueError, match="train_days"):
        WalkForwardSpec(train_days=0)
    with pytest.raises(ValueError, match="step_days"):
        WalkForwardSpec(step_days=0)


def test_schedule_tabulates_for_the_manifest(schedule: list[Fold]) -> None:
    table = describe(schedule)
    assert len(table) == 7
    assert list(table.columns)[:3] == ["fold", "train_start", "train_end"]


# ---------------------------------------------------------------------------
# Masks and purging
# ---------------------------------------------------------------------------


@pytest.fixture
def timestamps() -> pd.Series:
    """38 days at one observation a minute, which keeps the tests quick."""
    return pd.Series(
        pd.date_range("2024-02-01", periods=38 * 24 * 60, freq="1min", tz="UTC"),
    )


def test_masks_select_the_right_days(schedule: list[Fold], timestamps: pd.Series) -> None:
    masks = fold_masks(schedule[0], timestamps, purge=0)
    train_days = set(timestamps[masks.train].dt.date)
    assert min(train_days) == date(2024, 2, 1)
    assert max(train_days) == date(2024, 2, 14)
    assert len(train_days) == 14


def test_blocks_do_not_overlap(schedule: list[Fold], timestamps: pd.Series) -> None:
    masks = fold_masks(schedule[2], timestamps, purge=30)
    assert not (masks.train & masks.validation).any()
    assert not (masks.validation & masks.test).any()
    assert not (masks.train & masks.test).any()


def test_purging_removes_exactly_the_tail(schedule: list[Fold], timestamps: pd.Series) -> None:
    unpurged = fold_masks(schedule[0], timestamps, purge=0)
    purged = fold_masks(schedule[0], timestamps, purge=30)
    assert purged.sizes["train"] == unpurged.sizes["train"] - 30
    assert purged.sizes["validation"] == unpurged.sizes["validation"] - 30
    # Nothing follows test inside the fold, so its tail is left alone.
    assert purged.sizes["test"] == unpurged.sizes["test"]


def test_purged_rows_are_the_last_ones_before_the_boundary(
    schedule: list[Fold], timestamps: pd.Series
) -> None:
    purge = 30
    unpurged = fold_masks(schedule[0], timestamps, purge=0)
    purged = fold_masks(schedule[0], timestamps, purge=purge)
    removed = np.flatnonzero(unpurged.train & ~purged.train)
    expected = np.flatnonzero(unpurged.train)[-purge:]
    assert removed.tolist() == expected.tolist()


def test_no_training_label_is_decided_inside_validation(
    schedule: list[Fold], timestamps: pd.Series
) -> None:
    """The property purging exists for, stated directly.

    A label at row t reads the price at t + horizon. After purging, the largest
    row index used for training plus the horizon must still fall before the
    first validation row — otherwise the model was shown validation prices.
    """
    horizon = 30
    masks = fold_masks(schedule[3], timestamps, purge=horizon)
    last_train = int(np.flatnonzero(masks.train)[-1])
    first_validation = int(np.flatnonzero(masks.validation)[0])
    assert last_train + horizon < first_validation


def test_without_purging_the_leak_is_present(schedule: list[Fold], timestamps: pd.Series) -> None:
    """The same measurement without purging, to show the check has teeth."""
    horizon = 30
    masks = fold_masks(schedule[3], timestamps, purge=0)
    last_train = int(np.flatnonzero(masks.train)[-1])
    first_validation = int(np.flatnonzero(masks.validation)[0])
    assert last_train + horizon >= first_validation


def test_purge_larger_than_a_block_empties_it(schedule: list[Fold], timestamps: pd.Series) -> None:
    """Better an empty block that fails loudly than a silently tiny one."""
    masks = fold_masks(schedule[0], timestamps, purge=10_000_000)
    with pytest.raises(SplitError, match="empty after purging"):
        masks.check_non_empty()


def test_naive_timestamps_are_refused(schedule: list[Fold]) -> None:
    naive = pd.Series(pd.date_range("2024-02-01", periods=1000, freq="1min"))
    with pytest.raises(SplitError, match="timezone-aware"):
        fold_masks(schedule[0], naive, purge=0)


def test_available_days_lists_what_is_present(timestamps: pd.Series) -> None:
    days = available_days(timestamps)
    assert len(days) == 38
    assert days[0] == date(2024, 2, 1)


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


def make_book(mids: list[float]) -> pd.DataFrame:
    n = len(mids)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-02-01", periods=n, freq="100ms", tz="UTC"),
            "bid_price_0": [m - 0.5 for m in mids],
            "ask_price_0": [m + 0.5 for m in mids],
            "bid_size_0": [1.0] * n,
            "ask_size_0": [1.0] * n,
        }
    )


def test_forward_return_looks_the_declared_distance_ahead() -> None:
    book = make_book([100.0, 100.0, 101.0, 101.0])
    label = DirectionalLabel(horizon=2, threshold_bp=0.0)
    got = label.forward_return_bp(book)
    assert got.iloc[0] == pytest.approx(1e4 * np.log(101.0 / 100.0))


def test_last_rows_have_no_label_rather_than_a_neutral_one() -> None:
    """Filling them would train the model on 'the price did not move'."""
    book = make_book([100.0] * 10)
    label = DirectionalLabel(horizon=3, threshold_bp=1.0)
    values = label(book)
    assert values.iloc[-3:].isna().all()
    assert values.iloc[:-3].notna().all()


def test_threshold_decides_the_three_classes() -> None:
    # +100 bp, -100 bp, and flat.
    book = make_book([100.0, 100.0, 100.0, 101.005, 98.995, 100.0])
    label = DirectionalLabel(horizon=3, threshold_bp=50.0)
    values = label(book)
    assert values.iloc[0] == BUY
    assert values.iloc[1] == SELL
    assert values.iloc[2] == HOLD


def test_a_move_exactly_at_the_threshold_is_hold() -> None:
    """The boundary has to go somewhere; not trading is the safe side."""
    move_bp = 50.0
    later = 100.0 * np.exp(move_bp / 1e4)
    book = make_book([100.0, later])
    label = DirectionalLabel(horizon=1, threshold_bp=move_bp)
    assert label(book).iloc[0] == HOLD


def test_purge_equals_the_horizon() -> None:
    """The split takes its purge from here, so the two cannot drift apart."""
    assert DirectionalLabel(horizon=37, threshold_bp=1.0).purge == 37


def test_invalid_label_parameters_are_rejected() -> None:
    with pytest.raises(ValueError, match="horizon"):
        DirectionalLabel(horizon=0, threshold_bp=1.0)
    with pytest.raises(ValueError, match="threshold_bp"):
        DirectionalLabel(horizon=1, threshold_bp=-1.0)


def test_round_trip_cost_counts_both_sides() -> None:
    # Two sides of fees, the spread once, slippage on each side.
    assert round_trip_cost_bp(fee_bp_per_side=2.0, slippage_bp=0.5, spread_bp=2.0) == pytest.approx(
        7.0
    )


def test_class_balance_names_the_classes(book: pd.DataFrame) -> None:
    label = DirectionalLabel(horizon=20, threshold_bp=0.5)
    balance = class_balance(label(book))
    assert set(balance.index) <= {"sell", "hold", "buy"}
    assert balance.sum() == pytest.approx(1.0)


def test_threshold_for_share_hits_the_requested_frequency(book: pd.DataFrame) -> None:
    """Fixing how often to trade, then letting the threshold follow."""
    horizon = 20
    threshold = threshold_for_share(book, horizon=horizon, target_share=0.2)
    label = DirectionalLabel(horizon=horizon, threshold_bp=threshold)
    values = label(book).dropna()
    non_hold = (values != HOLD).mean()
    assert non_hold == pytest.approx(0.2, abs=0.03)


def test_threshold_for_share_rejects_impossible_targets(book: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="target_share"):
        threshold_for_share(book, horizon=10, target_share=1.5)


# ---------------------------------------------------------------------------
# Label and split together
# ---------------------------------------------------------------------------


def test_split_purge_removes_every_label_that_reads_across_the_boundary(
    schedule: list[Fold], timestamps: pd.Series
) -> None:
    """End to end: the label sets the purge, the split applies it, nothing leaks."""
    label = DirectionalLabel(horizon=45, threshold_bp=1.0)
    masks = fold_masks(schedule[1], timestamps, purge=label.purge)

    train_rows = np.flatnonzero(masks.train)
    validation_rows = np.flatnonzero(masks.validation)
    # Every training label is decided strictly before validation begins.
    assert (train_rows + label.horizon).max() < validation_rows.min()


def test_labels_and_features_agree_on_row_alignment(book: pd.DataFrame) -> None:
    label = DirectionalLabel(horizon=10, threshold_bp=1.0)
    values = label(book)
    assert len(values) == len(book)
    assert values.index.equals(book.index)


def test_label_direction_matches_the_price_move(book: pd.DataFrame) -> None:
    """Sanity: BUY rows really are followed by a rise."""
    horizon = 20
    label = DirectionalLabel(horizon=horizon, threshold_bp=0.5)
    values = label(book)
    mid = mid_price(book)
    forward = mid.shift(-horizon) / mid - 1.0

    assert forward[values == BUY].mean() > 0
    assert forward[values == SELL].mean() < 0
