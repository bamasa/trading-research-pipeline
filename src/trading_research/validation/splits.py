"""Chronological splits with purging.

The layout is a rolling walk-forward over whole days::

    fold 0:  [--- train 14d ---][- val 7d -][- test 7d -]
    fold 1:   [--- train 14d ---][- val 7d -][- test 7d -]
    fold 2:    [--- train 14d ---][- val 7d -][- test 7d -]
    ...

Each fold shifts one day later, so every fold is trained, tuned and evaluated
on a different stretch of market. Seven of them over a 28-day window needs 34
days of data.

Why days rather than a fraction of rows: a day is a real boundary in the data —
activity, spread and volatility all have a daily shape — so a split on day
boundaries produces blocks that are comparable to each other. A split at row
60% lands in the middle of some afternoon, and the resulting "periods" differ
in ways that have nothing to do with the model.

Purging
-------
The part that is easy to get wrong. A label at time *t* is computed from prices
up to *t + H*. So the last *H* observations of the training block have labels
that were determined by prices inside the validation block: train those and the
model has been shown validation prices, however carefully the blocks were cut.

Every boundary is therefore purged: the final *H* rows of the earlier block are
dropped. *H* is taken from the label itself rather than configured by hand,
because a purge chosen by hand is chosen too small, and because a change to the
label horizon must not silently invalidate the split.

Only forward overlap needs purging here. A later block reading *earlier* rows
through a feature's lookback is not leakage — that information genuinely
existed at the time. The reverse never happens, because the blocks are strictly
ordered in time and never shuffled.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import pairwise

import numpy as np
import pandas as pd


class SplitError(ValueError):
    """A split cannot be constructed from the data or specification given."""


@dataclass(frozen=True)
class WalkForwardSpec:
    """Shape of the rolling walk-forward.

    Defaults are the layout this project reports on: two weeks to fit, one week
    to choose thresholds, one week to evaluate, stepped a day at a time.
    """

    train_days: int = 14
    validation_days: int = 7
    test_days: int = 7
    step_days: int = 1
    n_folds: int = 7

    def __post_init__(self) -> None:
        for name in ("train_days", "validation_days", "test_days", "n_folds"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1, got {getattr(self, name)}")
        if self.step_days < 1:
            raise ValueError(f"step_days must be at least 1, got {self.step_days}")

    @property
    def window_days(self) -> int:
        """Days spanned by one fold."""
        return self.train_days + self.validation_days + self.test_days

    @property
    def required_days(self) -> int:
        """Days of contiguous data needed for the whole schedule."""
        return self.window_days + self.step_days * (self.n_folds - 1)


@dataclass(frozen=True)
class Fold:
    """One fold, as inclusive day ranges.

    Days rather than row indices, so a fold can be described, logged and
    reproduced without reference to a particular frame — and so the same fold
    means the same calendar period across instruments whose row counts differ.
    """

    index: int
    train: tuple[date, date]
    validation: tuple[date, date]
    test: tuple[date, date]

    def __str__(self) -> str:
        return (
            f"fold {self.index}: "
            f"train {self.train[0]}..{self.train[1]}, "
            f"val {self.validation[0]}..{self.validation[1]}, "
            f"test {self.test[0]}..{self.test[1]}"
        )

    def days(self) -> list[date]:
        """Every day the fold touches, in order."""
        start, end = self.train[0], self.test[1]
        return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def walk_forward(days: Sequence[date], spec: WalkForwardSpec | None = None) -> list[Fold]:
    """Build the fold schedule over a contiguous run of days.

    ``days`` must be sorted and have no gaps. A missing day is refused rather
    than skipped over: silently treating 1 March as following 27 February would
    make one fold cover a different amount of market than the others while
    still being labelled fourteen days.
    """
    spec = spec or WalkForwardSpec()
    ordered = sorted(days)
    if not ordered:
        raise SplitError("no days given")

    for earlier, later in pairwise(ordered):
        if (later - earlier).days != 1:
            raise SplitError(
                f"days are not contiguous: {earlier} is followed by {later}. "
                f"Fill or trim the range; a fold must not silently span a gap."
            )

    if len(ordered) < spec.required_days:
        raise SplitError(
            f"{spec.n_folds} fold(s) of {spec.window_days} day(s) stepping {spec.step_days} "
            f"need {spec.required_days} contiguous days, but only {len(ordered)} are available "
            f"({ordered[0]}..{ordered[-1]})"
        )

    folds: list[Fold] = []
    for i in range(spec.n_folds):
        offset = i * spec.step_days
        train_start = offset
        val_start = train_start + spec.train_days
        test_start = val_start + spec.validation_days
        test_end = test_start + spec.test_days - 1
        folds.append(
            Fold(
                index=i,
                train=(ordered[train_start], ordered[val_start - 1]),
                validation=(ordered[val_start], ordered[test_start - 1]),
                test=(ordered[test_start], ordered[test_end]),
            )
        )
    return folds


@dataclass(frozen=True)
class FoldMasks:
    """Row masks for one fold over one frame, after purging."""

    index: int
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    purged: int

    @property
    def sizes(self) -> dict[str, int]:
        return {
            "train": int(self.train.sum()),
            "validation": int(self.validation.sum()),
            "test": int(self.test.sum()),
        }

    def check_non_empty(self) -> None:
        empty = [name for name, n in self.sizes.items() if n == 0]
        if empty:
            raise SplitError(f"fold {self.index}: {', '.join(empty)} is empty after purging")


def fold_masks(
    fold: Fold,
    timestamps: pd.Series,
    *,
    purge: int,
    groups: pd.Series | None = None,
) -> FoldMasks:
    """Turn a fold into row masks over ``timestamps``, purging block tails.

    ``purge`` is the number of rows to drop from the end of the training and
    validation blocks — the label horizon, in observations. Pass 0 only for a
    target that does not look forward at all.

    ``groups`` is required whenever the frame holds **more than one time
    series** — several instruments stacked and sorted by time. The purge counts
    rows, and in an interleaved frame the last *n* rows are spread across every
    series, so a purge of 24 removes only 8 rows from each of three
    instruments. The tail of each series then overlaps the next block by
    two-thirds of the label horizon.

    That is not hypothetical: it produced the only positive result this project
    has seen, +11.69 bp per trade on a pooled fit, which vanished once the
    purge was applied per series. Passing ``groups`` purges each series
    separately, which is what "drop the last H observations" was always
    supposed to mean.

    The test block is not purged at its tail: nothing follows it in this fold,
    so there is nothing for its labels to leak into. Its final labels are
    incomplete for a different reason — they need prices past the end of the
    data — and those rows drop out as ``NaN`` when the label is computed.
    """
    if purge < 0:
        raise ValueError(f"purge must be non-negative, got {purge}")
    if timestamps.dt.tz is None:
        raise SplitError("timestamps must be timezone-aware; naive times cannot be split by day")
    if groups is not None and len(groups) != len(timestamps):
        raise SplitError(f"{len(groups)} groups for {len(timestamps)} rows")

    day = timestamps.dt.tz_convert("UTC").dt.date.to_numpy()

    train = _range_mask(day, *fold.train)
    validation = _range_mask(day, *fold.validation)
    test = _range_mask(day, *fold.test)

    if purge:
        keys = groups.to_numpy() if groups is not None else None
        train = _purge_tail(train, purge, keys)
        validation = _purge_tail(validation, purge, keys)

    return FoldMasks(index=fold.index, train=train, validation=validation, test=test, purged=purge)


def _range_mask(day: np.ndarray, start: date, end: date) -> np.ndarray:
    return (day >= start) & (day <= end)


def _purge_tail(mask: np.ndarray, purge: int, groups: np.ndarray | None = None) -> np.ndarray:
    """Drop the last ``purge`` selected rows of each series in the block.

    With no ``groups`` the whole block is one series and the last ``purge``
    rows go. With groups, each series loses its own last ``purge`` rows —
    otherwise an interleaved frame is purged by a fraction of what was asked.
    """
    if groups is None:
        selected = np.flatnonzero(mask)
        if len(selected) <= purge:
            return np.zeros_like(mask)
        out = mask.copy()
        out[selected[-purge:]] = False
        return out

    out = mask.copy()
    for key in np.unique(groups[mask]):
        selected = np.flatnonzero(mask & (groups == key))
        if len(selected) <= purge:
            out[selected] = False
        else:
            out[selected[-purge:]] = False
    return out


def available_days(timestamps: pd.Series) -> list[date]:
    """The distinct UTC days present, in order."""
    if timestamps.dt.tz is None:
        raise SplitError("timestamps must be timezone-aware")
    return sorted(set(timestamps.dt.tz_convert("UTC").dt.date))


def describe(folds: Sequence[Fold]) -> pd.DataFrame:
    """Tabulate a schedule, for the run manifest and the report."""
    return pd.DataFrame(
        [
            {
                "fold": f.index,
                "train_start": f.train[0],
                "train_end": f.train[1],
                "val_start": f.validation[0],
                "val_end": f.validation[1],
                "test_start": f.test[0],
                "test_end": f.test[1],
            }
            for f in folds
        ]
    )
