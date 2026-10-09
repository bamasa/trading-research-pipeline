"""Tests for the lazy window dataset.

The windows must be the same rows in training and inference, never reach past
their decision row, cross midnight only where the days are contiguous, and be
built per batch from loaders that are read one day at a time.
"""

from __future__ import annotations

import numpy as np
import pytest

from trading_research.models.windows import (
    Day,
    WindowDataset,
    complete_windows,
    every_nth_row,
    gather_windows,
    thin_evenly,
)


def counter_day(key: str, start: int, n: int, *, follows: bool = False, rows=None) -> Day:
    """A day whose single input is the global row number, so windows are readable."""
    values = np.arange(start, start + n, dtype=np.float32).reshape(-1, 1)
    picked = np.arange(n) if rows is None else np.asarray(rows)
    return Day(key, lambda: values, picked, picked.astype(np.float32), follows)


def test_a_window_ends_at_its_row_and_never_reaches_past_it() -> None:
    values = np.arange(20, dtype=np.float32).reshape(-1, 1)
    windows = gather_windows(values, np.array([4, 19]), window=5)
    assert windows.shape == (2, 1, 5)
    assert windows[0, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert windows[1, 0, -1] == 19.0


def test_incomplete_and_non_finite_windows_are_flagged() -> None:
    values = np.arange(10, dtype=np.float32).reshape(-1, 1)
    values[6] = np.nan
    ok = complete_windows(values, np.arange(10), window=3)
    # rows 0-1 lack history; rows 6-8 contain the NaN row
    assert ok.tolist() == [False, False, True, True, True, True, False, False, False, True]


def test_a_contiguous_day_takes_its_history_from_the_day_before() -> None:
    days = [counter_day("d1", 0, 10), counter_day("d2", 10, 10, follows=True)]
    dataset = WindowDataset(days, window=4)
    seen = {}
    for i, positions, windows in dataset.inference_batches(batch_size=64):
        for p, w in zip(positions, windows, strict=True):
            seen[(i, int(p))] = w[0].tolist()
    assert (0, 2) not in seen  # cold start: no full window yet
    assert seen[(1, 0)] == [7.0, 8.0, 9.0, 10.0]  # crosses midnight
    assert len(seen) == 7 + 10


def test_a_cold_start_day_does_not_borrow_history() -> None:
    days = [counter_day("d1", 0, 10), counter_day("d3", 100, 10, follows=False)]
    dataset = WindowDataset(days, window=4)
    positions = [(i, int(p)) for i, ps, _ in dataset.inference_batches(64) for p in ps]
    assert (1, 2) not in positions
    assert (1, 3) in positions


def test_training_covers_every_complete_window_once_with_its_own_target() -> None:
    days = [counter_day(f"d{k}", 10 * k, 10, follows=k > 0) for k in range(5)]
    dataset = WindowDataset(days, window=3)
    windows, targets = [], []
    for w, t in dataset.training_batches(4, np.random.default_rng(0), days_per_shard=2):
        windows.append(w)
        targets.append(t)
    w, t = np.concatenate(windows), np.concatenate(targets)
    # 50 rows, only the first two of the first day lack history
    assert len(t) == 48
    # each window's last value is its own row, which the counter target equals
    # modulo the day: the target is the row within its day
    assert np.array_equal(w[:, 0, -1] % 10, t)
    assert sorted(w[:, 0, -1].tolist()) == list(range(2, 50))


def test_the_shuffle_is_reproducible_from_the_seed() -> None:
    days = [counter_day(f"d{k}", 10 * k, 10, follows=k > 0) for k in range(4)]

    def run(seed: int) -> list[float]:
        dataset = WindowDataset(days, window=3)
        rng = np.random.default_rng(seed)
        return [float(x) for w, _ in dataset.training_batches(5, rng) for x in w[:, 0, -1]]

    assert run(1) == run(1)
    assert run(1) != run(2)


def test_days_are_loaded_lazily_one_shard_at_a_time() -> None:
    loads: list[str] = []

    def day(k: int) -> Day:
        def load() -> np.ndarray:
            loads.append(f"d{k}")
            return np.ones((8, 2), dtype=np.float16)

        rows = np.arange(8)
        return Day(f"d{k}", load, rows, np.zeros(8, dtype=np.float32), k > 0)

    dataset = WindowDataset([day(k) for k in range(4)], window=3)
    assert loads == []
    batches = dataset.training_batches(100, np.random.default_rng(0), days_per_shard=1)
    first, _ = next(batches)
    assert first.dtype == np.float32
    assert len(loads) <= 2  # its own day, and its predecessor's tail at most


def test_targets_must_match_rows_and_rows_must_lie_inside_the_day() -> None:
    with pytest.raises(ValueError, match="one per row"):
        Day("d", lambda: np.zeros((3, 1)), np.arange(3), np.zeros(2))
    dataset = WindowDataset([Day("d", lambda: np.zeros((3, 1)), np.array([5]))], window=2)
    with pytest.raises(ValueError, match="past the day's last row"):
        list(dataset.inference_batches(8))


def test_rows_are_thinned_evenly_across_days() -> None:
    rows = [every_nth_row(100, 10) for _ in range(4)]  # 40 rows in all
    kept = thin_evenly(rows, 20)
    assert sum(len(r) for r in kept) == 20
    assert [len(r) for r in kept] == [5, 5, 5, 5]
    assert thin_evenly(rows, 1000)[0].tolist() == rows[0].tolist()
