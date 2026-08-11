"""Tests for the dataset quality checks.

Each test breaks one invariant in an otherwise valid frame and asserts that the
right check fires at the right severity. Constructing the failures from a good
dataset — rather than from hand-written frames — keeps the tests honest about
what the validator will actually meet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lobml.data.schema import ask_price_col, bid_price_col, bid_size_col
from lobml.data.validate import (
    DataQualityError,
    Severity,
    validate,
    validate_book,
    validate_trades,
)


def _checks(report: object, severity: Severity | None = None) -> set[str]:
    findings = report.findings if severity is None else report.of(severity)  # type: ignore[attr-defined]
    return {f.check for f in findings}


# ---------------------------------------------------------------------------
# Clean data
# ---------------------------------------------------------------------------


def test_generated_book_is_clean(book: pd.DataFrame) -> None:
    report = validate_book(book)
    assert report.ok, report.summary()
    assert not report.errors


def test_generated_trades_are_clean(trades: pd.DataFrame) -> None:
    report = validate_trades(trades)
    assert report.ok, report.summary()


def test_empty_dataset_is_an_error() -> None:
    report = validate_trades(pd.DataFrame())
    assert not report.ok
    assert "non_empty" in _checks(report, Severity.ERROR)


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------


def test_non_monotonic_timestamps_are_an_error(book: pd.DataFrame) -> None:
    broken = book.copy()
    # Swap two adjacent stamps: the realistic failure is rows arriving out of
    # order, not an impossible time appearing from nowhere.
    stamps = broken["timestamp"].tolist()
    stamps[10], stamps[11] = stamps[11], stamps[10]
    broken["timestamp"] = pd.DatetimeIndex(stamps)
    report = validate_book(broken)
    assert "timestamp_monotonic" in _checks(report, Severity.ERROR)


def test_equal_timestamps_are_allowed(trades: pd.DataFrame) -> None:
    """Several trades can share a millisecond; that is not a defect."""
    tied = trades.copy()
    tied.loc[tied.index[5], "timestamp"] = tied["timestamp"].iloc[4]
    report = validate_trades(tied)
    assert "timestamp_monotonic" not in _checks(report, Severity.ERROR)


def test_long_pause_is_reported_as_a_warning(book: pd.DataFrame) -> None:
    """Rolling windows silently span gaps, so a gap has to be visible."""
    gapped = book.copy()
    seconds = np.where(np.arange(len(gapped)) >= 500, 2 * 3600, 0)
    gapped["timestamp"] = gapped["timestamp"] + pd.to_timedelta(seconds, unit="s")
    report = validate_book(gapped)
    assert "time_gaps" in _checks(report, Severity.WARNING)


# ---------------------------------------------------------------------------
# Book invariants
# ---------------------------------------------------------------------------


def test_crossed_book_is_an_error(book: pd.DataFrame) -> None:
    crossed = book.copy()
    row = crossed.index[3]
    crossed.loc[row, bid_price_col(0)] = crossed.loc[row, ask_price_col(0)] + 1.0
    report = validate_book(crossed)
    assert "not_crossed" in _checks(report, Severity.ERROR)


def test_locked_book_is_a_warning_not_an_error(book: pd.DataFrame) -> None:
    locked = book.copy()
    row = locked.index[3]
    locked.loc[row, bid_price_col(0)] = locked.loc[row, ask_price_col(0)]
    report = validate_book(locked)
    assert "not_locked" in _checks(report, Severity.WARNING)
    assert "not_crossed" not in _checks(report, Severity.ERROR)


def test_out_of_order_bid_levels_are_an_error(book: pd.DataFrame) -> None:
    """A level 1 bid above level 0 means the book was assembled wrongly."""
    scrambled = book.copy()
    row = scrambled.index[7]
    scrambled.loc[row, bid_price_col(1)] = scrambled.loc[row, bid_price_col(0)] + 5.0
    report = validate_book(scrambled)
    assert "bid_level_ordering" in _checks(report, Severity.ERROR)


def test_negative_size_is_an_error(book: pd.DataFrame) -> None:
    negative = book.copy()
    negative.loc[negative.index[2], bid_size_col(0)] = -1.0
    report = validate_book(negative)
    assert "non_negative_sizes" in _checks(report, Severity.ERROR)


def test_zero_size_is_allowed(book: pd.DataFrame) -> None:
    """An empty level is normal; only a negative size is impossible."""
    empty_level = book.copy()
    empty_level.loc[empty_level.index[2], bid_size_col(4)] = 0.0
    report = validate_book(empty_level)
    assert "non_negative_sizes" not in _checks(report, Severity.ERROR)


def test_non_positive_price_is_an_error(book: pd.DataFrame) -> None:
    zero_price = book.copy()
    zero_price.loc[zero_price.index[2], bid_price_col(3)] = 0.0
    report = validate_book(zero_price)
    assert "positive_prices" in _checks(report, Severity.ERROR)


# ---------------------------------------------------------------------------
# Identifiers
# ---------------------------------------------------------------------------


def test_sequence_gap_is_a_warning(book: pd.DataFrame) -> None:
    """A missed update corrupts every later snapshot until resynchronisation."""
    gapped = book.copy()
    idx = gapped.index[100:]
    gapped.loc[idx, "sequence_id"] = gapped.loc[idx, "sequence_id"] + 500
    report = validate_book(gapped)
    assert "sequence_gaps" in _checks(report, Severity.WARNING)


def test_repeated_sequence_id_is_an_error(book: pd.DataFrame) -> None:
    repeated = book.copy()
    repeated.loc[repeated.index[8], "sequence_id"] = repeated["sequence_id"].iloc[7]
    report = validate_book(repeated)
    assert "sequence_unique" in _checks(report, Severity.ERROR)


def test_duplicate_trade_id_is_an_error(trades: pd.DataFrame) -> None:
    duplicated = trades.copy()
    duplicated.loc[duplicated.index[4], "trade_id"] = duplicated["trade_id"].iloc[3]
    report = validate_trades(duplicated)
    assert "trade_id_unique" in _checks(report, Severity.ERROR)


# ---------------------------------------------------------------------------
# Report behaviour
# ---------------------------------------------------------------------------


def test_warnings_do_not_fail_a_dataset(book: pd.DataFrame) -> None:
    locked = book.copy()
    row = locked.index[3]
    locked.loc[row, bid_price_col(0)] = locked.loc[row, ask_price_col(0)]
    report = validate_book(locked)
    assert report.warnings
    assert report.ok
    report.raise_if_failed()


def test_raise_if_failed_names_the_broken_checks(book: pd.DataFrame) -> None:
    crossed = book.copy()
    row = crossed.index[3]
    crossed.loc[row, bid_price_col(0)] = crossed.loc[row, ask_price_col(0)] + 1.0
    with pytest.raises(DataQualityError, match="not_crossed"):
        validate_book(crossed).raise_if_failed()


def test_report_serialises_for_the_run_manifest(book: pd.DataFrame) -> None:
    payload = validate_book(book).to_dict()
    assert payload["plane"] == "book"
    assert payload["ok"] is True
    assert isinstance(payload["findings"], list)


def test_dispatch_rejects_an_unknown_plane(book: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="unknown plane"):
        validate(book, "orders")
