"""Windows of consecutive rows, built per batch and never stored.

The network sees the 64 consecutive 100 ms rows ending at its decision row, in
training and in inference alike. Materialising those windows is not an
option at this sampling rate: a fit reads up to 4 million training rows of about
70 inputs, and as windows that is 4e6 x 64 x 70 x 4 bytes, about 72 GB.
``models/tcn._windowise`` copies its strided view into exactly such an array.

So the windows here are gathered per batch from the rows themselves. The rows
come one day at a time from the every-row feature store through a loader, so
neither a fit nor a day ever holds its windows at once: a day holds its rows,
and a batch holds its windows.

Day boundaries
--------------
A day whose predecessor in the sequence is the calendar day before, with no gap,
takes the history of its first rows from that day's last ``window - 1`` rows.
Any other day is a cold start: its rows before ``window - 1`` have no window and
are neither trained on nor scored. A window that contains a non-finite input is
treated the same way. The predecessor's tail is kept after its first load, so a
day whose neighbour sits in another shard costs one extra load at most once.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np

#: Rows in the network's window: 6.4 s at 100 ms, ending at the decision row.
WINDOW_ROWS = 64


@dataclass(frozen=True, eq=False)
class Day:
    """One day of the every-row store, read only when its windows are needed.

    ``load`` returns the day's inputs as ``(rows, features)`` in row order; it
    may return a memory map. ``rows`` are the row indices at which windows end:
    the training targets' rows in a fit, every row at inference. ``targets``
    holds one value per entry of ``rows``, and is None at inference.
    ``follows_previous`` marks a day whose predecessor in the sequence is the
    calendar day before it, with no gap.
    """

    key: str
    load: Callable[[], np.ndarray]
    rows: np.ndarray
    targets: np.ndarray | None = None
    follows_previous: bool = False

    def __post_init__(self) -> None:
        rows = np.asarray(self.rows)
        if rows.ndim != 1 or (len(rows) and (rows.min() < 0 or np.any(np.diff(rows) < 0))):
            raise ValueError(f"{self.key}: rows must be non-negative and sorted")
        if self.targets is not None and len(self.targets) != len(rows):
            raise ValueError(
                f"{self.key}: {len(self.targets)} targets for {len(rows)} rows; one per row"
            )


def every_nth_row(n_rows: int, step: int = 10) -> np.ndarray:
    """Rows 0, ``step``, 2 ``step``, ...: the rows training targets are taken on."""
    if step < 1:
        raise ValueError(f"step must be at least 1, got {step}")
    return np.arange(0, n_rows, step, dtype=np.int64)


def thin_evenly(rows_per_day: Sequence[np.ndarray], max_rows: int) -> list[np.ndarray]:
    """Keep at most ``max_rows`` of the rows across all days, evenly spaced in time.

    The days are taken in sequence order and the kept rows are spread over their
    concatenation, so a fit over many days thins every day alike rather than
    dropping its last days.
    """
    if max_rows < 1:
        raise ValueError(f"max_rows must be at least 1, got {max_rows}")
    sizes = np.array([len(r) for r in rows_per_day], dtype=np.int64)
    total = int(sizes.sum())
    if total <= max_rows:
        return [np.asarray(r, dtype=np.int64) for r in rows_per_day]
    keep = np.floor(np.arange(max_rows) * (total / max_rows)).astype(np.int64)
    bounds = np.concatenate([[0], np.cumsum(sizes)])
    out = []
    for day, rows in enumerate(rows_per_day):
        mine = keep[(keep >= bounds[day]) & (keep < bounds[day + 1])] - bounds[day]
        out.append(np.asarray(rows, dtype=np.int64)[mine])
    return out


def gather_windows(values: np.ndarray, ends: np.ndarray, window: int) -> np.ndarray:
    """``(len(ends), features, window)`` float32 windows ending at ``ends``.

    A copy of this batch's windows only, in the layout a ``Conv1d`` reads:
    (batch, channels, time). The window for end ``e`` covers rows
    ``e - window + 1`` to ``e`` and never a row after ``e``.
    """
    index = np.asarray(ends, dtype=np.int64)[:, None] + np.arange(1 - window, 1)
    return np.ascontiguousarray(values[index].transpose(0, 2, 1), dtype=np.float32)


def complete_windows(values: np.ndarray, ends: np.ndarray, window: int) -> np.ndarray:
    """Whether each window ending at ``ends`` lies inside ``values`` and is finite."""
    ends = np.asarray(ends, dtype=np.int64)
    bad = ~np.isfinite(values).all(axis=1)
    count = np.concatenate([[0], np.cumsum(bad)])
    starts = ends - (window - 1)
    inside = (starts >= 0) & (ends < len(values))
    lo = np.clip(starts, 0, len(values))
    hi = np.clip(ends + 1, 0, len(values))
    return inside & (count[hi] - count[lo] == 0)


class WindowDataset:
    """Batches of windows over a sequence of days, gathered from rows per batch.

    Training iterates over shards of ``days_per_shard`` days in a random order,
    each shard's window ends mixed together, so a batch is not one moment of one
    day. Inference walks the days in sequence order. Both read every day through
    its loader and keep only the shard (or day) being read, plus each day's last
    ``window - 1`` rows for its successor.
    """

    def __init__(self, days: Sequence[Day], *, window: int = WINDOW_ROWS) -> None:
        if window < 1:
            raise ValueError(f"window must be at least 1, got {window}")
        self.days = list(days)
        self.window = window
        self._tails: dict[int, np.ndarray] = {}
        self.n_features: int | None = None

    def __len__(self) -> int:
        """Window ends requested over all days, complete or not."""
        return int(sum(len(day.rows) for day in self.days))

    def _load(self, i: int) -> np.ndarray:
        values = np.asarray(self.days[i].load())
        if values.ndim != 2:
            raise ValueError(f"{self.days[i].key}: inputs must be (rows, features)")
        if self.n_features is None:
            self.n_features = int(values.shape[1])
        elif values.shape[1] != self.n_features:
            raise ValueError(
                f"{self.days[i].key}: {values.shape[1]} inputs, earlier days had {self.n_features}"
            )
        self._tails[i] = np.array(values[len(values) - min(len(values), self.window - 1) :])
        return values

    def extended(self, i: int) -> tuple[np.ndarray, int]:
        """Day ``i``'s rows with its predecessor's tail in front, and the tail's length."""
        values = self._load(i)
        if not (i > 0 and self.days[i].follows_previous and self.window > 1):
            return values, 0
        if i - 1 not in self._tails:
            self._load(i - 1)
        tail = self._tails[i - 1]
        return np.concatenate([tail.astype(values.dtype, copy=False), values]), len(tail)

    def complete(self, i: int, values: np.ndarray, offset: int) -> np.ndarray:
        """Which of day ``i``'s rows have a complete window (and a finite target)."""
        day = self.days[i]
        if len(day.rows) and int(np.max(day.rows)) >= len(values) - offset:
            raise ValueError(f"{day.key}: a row index lies past the day's last row")
        ok = complete_windows(values, np.asarray(day.rows) + offset, self.window)
        if day.targets is not None:
            ok &= np.isfinite(np.asarray(day.targets, dtype=np.float64))
        return ok

    def training_batches(
        self, batch_size: int, rng: np.random.Generator, days_per_shard: int = 4
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """``(windows, targets)`` batches over every complete window, in a shuffled order."""
        order = rng.permutation(len(self.days))
        for start in range(0, len(order), days_per_shard):
            blocks, ends, targets = [], [], []
            base = 0
            for i in sorted(int(d) for d in order[start : start + days_per_shard]):
                day = self.days[i]
                if day.targets is None:
                    raise ValueError(f"{day.key}: a training day needs targets")
                values, offset = self.extended(i)
                ok = self.complete(i, values, offset)
                blocks.append(values)
                ends.append(np.asarray(day.rows)[ok] + offset + base)
                targets.append(np.asarray(day.targets, dtype=np.float32)[ok])
                base += len(values)
            if not blocks:
                continue
            shard = np.concatenate(blocks) if len(blocks) > 1 else blocks[0]
            shard_ends, shard_targets = np.concatenate(ends), np.concatenate(targets)
            mix = rng.permutation(len(shard_ends))
            for b in range(0, len(mix), batch_size):
                pick = mix[b : b + batch_size]
                yield gather_windows(shard, shard_ends[pick], self.window), shard_targets[pick]

    def inference_batches(self, batch_size: int) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
        """``(day, positions in day.rows, windows)`` over complete windows, in day order."""
        for i, day in enumerate(self.days):
            values, offset = self.extended(i)
            positions = np.flatnonzero(self.complete(i, values, offset))
            ends = np.asarray(day.rows)[positions] + offset
            for b in range(0, len(positions), batch_size):
                chunk = slice(b, b + batch_size)
                yield i, positions[chunk], gather_windows(values, ends[chunk], self.window)
