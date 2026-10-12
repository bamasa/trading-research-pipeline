"""Stage A's fits: the refit's circular label shift and the thinning of training rows."""

from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd

from trading_research.pipeline.short_horizon import features as ft
from trading_research.pipeline.short_horizon.fits import FoldData, thin_positions, window_days
from trading_research.pipeline.short_horizon.plan import (
    a0_fold,
    frozen_fold,
    midnight_ns,
    purge_ns,
    walk_forward,
)

NS = 1_000_000_000


def _fold_data(stamps: np.ndarray, labels: np.ndarray) -> FoldData:
    targets = pd.DataFrame({ft.TIME: stamps, "y": labels})
    return FoldData(pd.DataFrame(index=targets.index), targets)


def test_the_refit_shift_is_circular_on_a_span_shorter_than_its_lag() -> None:
    # February's fold: training 2025-01-01 to 01-24 (about 24 days, less than 28)
    start = midnight_ns(date(2025, 1, 1))
    end = midnight_ns(date(2025, 1, 25)) - purge_ns(20.0)
    stamps = np.arange(start, end, 10 * NS, dtype=np.int64)
    data = _fold_data(stamps, np.arange(len(stamps), dtype=float))
    shifted = data.shifted("y", start, end, 28)
    assert np.isfinite(shifted).all()  # every row keeps a label: nothing falls off the span
    source = start + np.mod(stamps - start - 28 * 86_400 * NS, end - start)
    expected = np.searchsorted(stamps, source, side="right") - 1
    assert (shifted == expected.astype(float)).all()
    assert sorted(shifted.tolist()) == list(range(len(stamps)))  # a permutation of the labels


def test_the_refit_shift_moves_labels_by_the_lag_on_a_long_span() -> None:
    start = midnight_ns(date(2025, 1, 1))
    end = start + 60 * 86_400 * NS
    stamps = np.arange(start, end, 600 * NS, dtype=np.int64)
    data = _fold_data(stamps, np.arange(len(stamps), dtype=float))
    shifted = data.shifted("y", start, end, 28)
    later = stamps >= start + 28 * 86_400 * NS
    lag_rows = 28 * 86_400 // 600
    assert (shifted[later] == np.flatnonzero(later) - lag_rows).all()
    assert (shifted[~later] == np.flatnonzero(~later) + len(stamps) - lag_rows).all()


def test_thinning_is_integer_floor_over_every_row_of_the_span() -> None:
    assert thin_positions(10, 20).tolist() == list(range(10))
    total, kept = 3 * 1_562_000 + 7, 4_000_000  # floor(i * total / kept), exactly
    positions = thin_positions(total, kept)
    i = np.array([0, 1, 2_999_999, 3_999_999], dtype=object)
    assert positions[[0, 1, 2_999_999, 3_999_999]].tolist() == [int(v * total // kept) for v in i]
    # FoldData.rows thins over every row before the cut, labelled or not
    stamps = np.arange(100, dtype=np.int64) * NS
    labels = np.where(np.arange(100) % 3 == 0, np.nan, 1.0)
    rows = _fold_data(stamps, labels).rows(before_ns=80 * NS, max_rows=8)
    assert rows.tolist() == [i * 80 // 8 for i in range(8)]  # 0, 10, ..., 70


def test_the_networks_rows_are_thinned_before_the_unlabelled_ones_are_dropped() -> None:
    days = [date(2025, 1, 2), date(2025, 1, 3)]
    frames = {}
    for k, day in enumerate(days):
        rows = np.arange(0, 400, 10, dtype=np.int64)
        target = np.where((rows // 10) % 8 == 0, np.nan, rows / 10.0 + 100 * k)
        frames[day] = pd.DataFrame({"row": rows, ft.TIME: midnight_ns(day) + rows * NS,
                                    "forward_20": target})  # fmt: skip
    inputs = SimpleNamespace(stores={"every_row": None}, targets=lambda d: frames[d],
                             cold=lambda _: False)  # fmt: skip
    found = window_days(inputs, days, ["f"], "forward_20", None, max_rows=20)  # type: ignore[arg-type]
    flat = np.concatenate([frames[d]["row"].to_numpy() for d in days])
    every = flat[[i * 80 // 20 for i in range(20)]]  # 80 rows in all, every fourth kept
    kept = np.concatenate([d.rows for d in found])
    assert kept.tolist() == [r for r in every.tolist() if (r // 10) % 8 != 0]
    assert len(kept) == 10  # thinning first, labelled rows only, would keep 20
    assert all(np.isfinite(d.targets).all() for d in found)


def test_the_inner_validation_is_the_seven_calendar_days_before_the_month() -> None:
    days = [date(2025, 1, 1) + timedelta(days=i) for i in range(59)]  # to 02-28
    days.remove(date(2025, 1, 28))  # an excluded day inside the window
    fold = walk_forward(days, months=("2025-02",))[0]
    assert fold.inner == tuple(date(2025, 1, d) for d in (25, 26, 27, 29, 30, 31))
    assert fold.train[-1] == date(2025, 1, 24)  # never filled from earlier days


def test_the_frozen_fits_and_a0_purge_both_ends_of_their_threshold_window() -> None:
    days = [date(2025, 1, 1) + timedelta(days=i) for i in range(181)]
    frozen = frozen_fold(days)
    assert frozen.inner[0] == date(2025, 6, 24) and frozen.inner[-1] == date(2025, 6, 30)
    assert frozen.train_end_ns(50.0) == midnight_ns(date(2025, 6, 24)) - purge_ns(50.0)
    assert frozen.inner_end_ns(50.0) == midnight_ns(date(2025, 7, 1)) - purge_ns(50.0)
    a0 = a0_fold(days)
    assert a0.train_end_ns(50.0) == midnight_ns(date(2025, 1, 28)) - purge_ns(50.0)
    assert a0.inner_end_ns(50.0) == midnight_ns(date(2025, 2, 4)) - purge_ns(50.0)
