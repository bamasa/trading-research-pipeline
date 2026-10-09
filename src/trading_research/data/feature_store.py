"""Two float16 feature stores: model inputs only, one zstd parquet file per day.

A day of BTCUSDT at 100 ms is about 864,000 rows. Seventy inputs at float32
over the development period would not fit the disk budget next to the books,
so the inputs are stored at float16, in two stores:

- **every row** (stride 1): the network's inputs, read in windows of
  consecutive rows; built once and kept through the development period;
- **every tenth row** (stride 10): the rows the other fits train on.

Only model inputs go here. Labels, costs, the gate's inputs and every fill are
computed from the book's float64 prices and never read from a store, so the
half-precision cast cannot move a price, a fee or a P&L.

**Float16 overflows at 65,504.** A feature in raw size units (order-flow
imbalance, depth or print-volume sums) is in BTC and can exceed that on a busy
day, where the cast would turn it into infinity. Such columns are divided by the
median touch size on the warm-up block before the cast (:func:`median_touch_size`,
:func:`scale_raw_sizes`). The finiteness test (:func:`finiteness_failures`) then
runs on the warm-up block: a column that still overflows is stored at float32
on every block. The scale, the raw-size columns and the float32 columns form
the :class:`StoreLayout`, fitted once and written beside each store, so a block
can never be stored under a different layout from the one it is read with.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

#: The largest finite float16.
FLOAT16_MAX = float(np.finfo(np.float16).max)

#: The column that orders a day's rows; kept as it is, never cast.
TIME_COLUMN = "timestamp"

EVERY_ROW = "every_row"
EVERY_TENTH_ROW = "every_tenth_row"
STRIDES: dict[str, int] = {EVERY_ROW: 1, EVERY_TENTH_ROW: 10}

LAYOUT_FILE = "layout.json"


def median_touch_size(books: Iterable[pd.DataFrame]) -> float:
    """The median size at the touch, over both sides and every row of ``books``.

    ``books`` are the warm-up block's replayed days (``bid_size_0`` and
    ``ask_size_0`` are all that is read). Pooling the two sides gives one scale
    for every raw-size column, whichever side it measures.
    """
    sizes = [
        np.concatenate(
            [
                book["bid_size_0"].to_numpy(dtype=np.float64),
                book["ask_size_0"].to_numpy(dtype=np.float64),
            ]
        )
        for book in books
    ]
    if not sizes or not sum(len(s) for s in sizes):
        raise ValueError("no book rows to take the touch size from")
    median = float(np.median(np.concatenate(sizes)))
    if not median > 0:
        raise ValueError(f"the median touch size is {median}, not a positive scale")
    return median


@dataclass(frozen=True)
class StoreLayout:
    """How stored columns are scaled and cast; fitted on the warm-up block."""

    touch_size: float
    raw_size_columns: tuple[str, ...]
    float32_columns: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if not self.touch_size > 0:
            raise ValueError(f"touch size must be positive, got {self.touch_size}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> StoreLayout:
        return cls(
            touch_size=float(raw["touch_size"]),
            raw_size_columns=tuple(raw["raw_size_columns"]),
            float32_columns=tuple(raw.get("float32_columns", ())),
        )


def _inputs(frame: pd.DataFrame) -> list[str]:
    columns = [c for c in frame.columns if c != TIME_COLUMN]
    other = [c for c in columns if not pd.api.types.is_numeric_dtype(frame[c])]
    if other:
        raise TypeError(f"a store holds numeric model inputs only, not {other}")
    return columns


def scale_raw_sizes(
    frame: pd.DataFrame, raw_size_columns: Sequence[str], touch_size: float
) -> pd.DataFrame:
    """A float64 copy of ``frame`` with each raw-size column divided by ``touch_size``."""
    absent = sorted(set(raw_size_columns) - set(frame.columns))
    if absent:
        raise KeyError(f"raw-size columns absent from the frame: {absent}")
    out = frame.copy()
    for column in raw_size_columns:
        out[column] = frame[column].to_numpy(dtype=np.float64) / touch_size
    return out


@dataclass(frozen=True)
class Finiteness:
    """The finiteness test's findings for one frame."""

    #: Finite before the cast and infinite after it: stored at float32 instead.
    overflow: tuple[str, ...]
    #: Infinite before any cast: a feature fault no dtype repairs, reported.
    infinite_at_source: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.overflow and not self.infinite_at_source


def finiteness_failures(
    frame: pd.DataFrame, raw_size_columns: Sequence[str], touch_size: float
) -> Finiteness:
    """Which columns would not be finite once scaled and cast to float16.

    A missing value (NaN, as in a lookback's first rows) is not a failure; an
    infinity is, whether the cast made it or the feature already held it.
    """
    scaled = scale_raw_sizes(frame, raw_size_columns, touch_size)
    overflow, infinite = [], []
    for column in _inputs(scaled):
        values = scaled[column].to_numpy(dtype=np.float64)
        if np.isinf(values).any():
            infinite.append(column)
        elif (np.abs(values[np.isfinite(values)]) > FLOAT16_MAX).any():
            overflow.append(column)
    return Finiteness(tuple(overflow), tuple(infinite))


def fit_layout(
    warm_up: Iterable[pd.DataFrame], *, touch_size: float, raw_size_columns: Sequence[str]
) -> tuple[StoreLayout, Finiteness]:
    """Run the finiteness test over every warm-up day and fix the layout.

    Every column that overflows on any warm-up day is stored at float32 on
    every block. Returns the layout and the union of the findings.
    """
    overflow: set[str] = set()
    infinite: set[str] = set()
    for frame in warm_up:
        found = finiteness_failures(frame, raw_size_columns, touch_size)
        overflow.update(found.overflow)
        infinite.update(found.infinite_at_source)
    layout = StoreLayout(touch_size, tuple(raw_size_columns), tuple(sorted(overflow)))
    return layout, Finiteness(tuple(sorted(overflow)), tuple(sorted(infinite)))


def to_stored(frame: pd.DataFrame, layout: StoreLayout) -> pd.DataFrame:
    """Scale the raw-size columns and cast every input to its stored dtype."""
    scaled = scale_raw_sizes(frame, layout.raw_size_columns, layout.touch_size)
    wide = set(layout.float32_columns)
    columns: dict[str, Any] = {}
    if TIME_COLUMN in scaled.columns:
        columns[TIME_COLUMN] = scaled[TIME_COLUMN].reset_index(drop=True)
    for column in _inputs(scaled):
        dtype = np.float32 if column in wide else np.float16
        with np.errstate(over="ignore"):  # an overflow is reported by the caller
            columns[column] = scaled[column].to_numpy(dtype=np.float64).astype(dtype)
    return pd.DataFrame(columns)


@dataclass(frozen=True)
class Written:
    """One stored day: its rows, its file size and any column that overflowed."""

    rows: int
    size: int
    #: Columns stored at float16 that were not finite after the cast on this
    #: day although finite before it. Empty on the warm-up block by
    #: construction; on a later block it is reported, never repaired.
    overflow: tuple[str, ...]


class FeatureStore:
    """One store: ``<root>/<name>/<day>.parquet``, with its layout beside the days.

    ``stride`` keeps every ``stride``-th row of a day by position, the first row
    included, so the every-tenth-row store holds rows 0, 10, 20, ... of each
    day's frame. The layout is written on first use and checked on every later
    open: a store is never written under two layouts.
    """

    def __init__(self, root: Path | str, name: str, layout: StoreLayout) -> None:
        if name not in STRIDES:
            raise ValueError(f"unknown store {name!r}; expected one of {sorted(STRIDES)}")
        self.directory = Path(root) / name
        self.name = name
        self.stride = STRIDES[name]
        self.layout = layout
        path = self.directory / LAYOUT_FILE
        if path.exists():
            existing = StoreLayout.from_dict(json.loads(path.read_text(encoding="utf-8")))
            if existing != layout:
                raise ValueError(f"{self.directory} was written under another layout: {existing}")
        else:
            self.directory.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(layout.to_dict(), indent=1) + "\n", encoding="utf-8")

    def path(self, day: date) -> Path:
        return self.directory / f"{day.isoformat()}.parquet"

    def write_day(self, day: date, frame: pd.DataFrame) -> Written:
        """Store one day's inputs, every row or every tenth, at their stored dtypes."""
        rows = frame.iloc[:: self.stride]
        stored = to_stored(rows, self.layout)
        overflow = tuple(
            column
            for column in stored.columns
            if column != TIME_COLUMN
            and stored[column].dtype == np.float16
            and bool(np.isinf(stored[column].to_numpy()).any())
            and not bool(np.isinf(rows[column].to_numpy(dtype=np.float64)).any())
        )
        target = self.path(day)
        partial = target.with_name(f"{target.name}.part")
        stored.to_parquet(partial, compression="zstd", index=False)
        partial.replace(target)
        return Written(rows=len(stored), size=target.stat().st_size, overflow=overflow)

    def read_day(self, day: date, columns: Sequence[str] | None = None) -> pd.DataFrame:
        path = self.path(day)
        if not path.exists():
            raise FileNotFoundError(f"no {self.name} features for {day} under {self.directory}")
        return pd.read_parquet(path, columns=list(columns) if columns is not None else None)

    def days(self) -> list[date]:
        return sorted(date.fromisoformat(p.stem) for p in self.directory.glob("*.parquet"))

    def size_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.directory.glob("*.parquet"))

    def delete_span(self, start: date, end: date) -> int:
        """Delete the stored days from ``start`` to ``end`` inclusive; bytes freed."""
        from trading_research.data.disk import delete_days

        return delete_days(self.directory, [d for d in self.days() if start <= d <= end])


def open_stores(root: Path | str, layout: StoreLayout) -> dict[str, FeatureStore]:
    """Both stores of a study under ``root``, sharing one layout."""
    return {name: FeatureStore(root, name, layout) for name in STRIDES}


def delete_features(stores: Iterable[FeatureStore], start: date, end: date) -> int:
    """Delete every store's days in ``[start, end]``; the warm-up and development
    features go this way once the held-out results are committed."""
    return sum(store.delete_span(start, end) for store in stores)
