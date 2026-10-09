"""Features whose windows are measured in seconds on row stamps.

:mod:`trading_research.features.registry` declares lookbacks in rows. That is
exact on a regular grid and wrong on the replayed book, where a row is emitted
only for a 100 ms interval that saw an update: twenty rows are two seconds in a
busy minute and a minute in a quiet one. A window meant as "the last 60 s" has
to be declared and computed in time.

A timed feature therefore makes a two-part claim. Its value at row *t* depends
only on

- the rows stamped at or after ``s_t - lookback_s``, and
- the ``lookback_rows`` rows before the first of them,

and, on the trade plane, on the prints that belong to those rows. A print
belongs to the first row stamped at or after it (the backward rule of
:mod:`trading_research.features.prints`), so the rows above own the prints
stamped after the row before them, up to ``s_t``.

The row part exists for quantities that read one row further than their window:
a row-to-row change at the window's first row needs the row before it. Wrapped
registry features, whose windows are in rows, declare ``lookback_s=0`` and
their rows. :mod:`trading_research.validation.leakage_timed` tests both parts.

Two numbers come out of the declarations. ``N``, the minutes a cold-start day
leaves unscored, is the longest ``lookback_s`` rounded up to whole minutes;
the row parts are covered separately by the rows since midnight. And on a grid
of ``grid_ms``, where at most one row falls in each interval, a time window of
``L`` seconds spans at most ``ceil(L / grid) + 1`` rows, which bounds the
purge in rows.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pandas.core.window.rolling import Rolling

Plane = Literal["book", "trades"]

#: A timed feature reads the book rows and, on the trade plane, the prints.
TimedFn = Callable[[pd.DataFrame, pd.DataFrame | None], "pd.Series[float]"]

#: A frame builder computes several columns in one pass.
FrameBuilder = Callable[[pd.DataFrame, pd.DataFrame | None], pd.DataFrame]

NS_PER_S = 1_000_000_000


def stamps_ns(rows: pd.DataFrame) -> np.ndarray:
    """Row stamps as int64 nanoseconds since the epoch, checked increasing."""
    index = pd.DatetimeIndex(rows["timestamp"]).as_unit("ns")
    stamps = index.to_numpy(dtype="datetime64[ns]").view(np.int64)
    if len(stamps) > 1 and bool(np.any(np.diff(stamps) < 0)):
        raise ValueError("row stamps must be non-decreasing")
    return np.asarray(stamps, dtype=np.int64)


def seconds_ns(seconds: float) -> int:
    """A duration in seconds, as whole nanoseconds."""
    return round(seconds * NS_PER_S)


def at_or_after(stamps: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Index of the first row stamped at or after each target.

    ``len(stamps)`` where no row follows: the caller decides what an absent
    snapshot means, rather than receiving the last one in its place.
    """
    return np.searchsorted(stamps, targets, side="left").astype(np.int64)


def at_or_before(stamps: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Index of the last row stamped at or before each target, -1 if none."""
    return np.searchsorted(stamps, targets, side="right").astype(np.int64) - 1


def trailing_start(stamps: np.ndarray, seconds: float) -> np.ndarray:
    """First row of each row's trailing window ``(s_t - seconds, s_t]``."""
    return np.searchsorted(stamps, stamps - seconds_ns(seconds), side="right").astype(np.int64)


def timed_series(values: np.ndarray | pd.Series, stamps: np.ndarray) -> pd.Series:
    """Values on a DatetimeIndex of the stamps, for pandas' time windows."""
    data = values.to_numpy(dtype="float64") if isinstance(values, pd.Series) else values
    return pd.Series(np.asarray(data, dtype="float64"), index=pd.DatetimeIndex(stamps))


def rolling_time(
    values: np.ndarray | pd.Series, stamps: np.ndarray, seconds: float
) -> Rolling[pd.Series]:
    """A pandas rolling object over ``(s_t - seconds, s_t]`` on the row stamps."""
    return timed_series(values, stamps).rolling(pd.Timedelta(seconds_ns(seconds), unit="ns"))


@dataclass(frozen=True)
class TimedFeature:
    """One feature and the claim it makes about the rows and prints it reads.

    ``builder`` and ``column`` are set when the feature is one column of a
    frame builder; :func:`build_timed` then runs each builder once.
    """

    name: str
    plane: Plane
    lookback_s: float
    lookback_rows: int
    fn: TimedFn
    description: str = ""
    builder: FrameBuilder | None = None
    column: str | None = None

    def __post_init__(self) -> None:
        if self.lookback_s < 0 or self.lookback_rows < 0:
            raise ValueError(f"{self.name}: lookbacks must be non-negative")

    def __call__(self, rows: pd.DataFrame, prints: pd.DataFrame | None = None) -> pd.Series:
        if self.plane == "trades" and prints is None:
            raise ValueError(f"feature {self.name!r} reads prints; none were given")
        result = self.fn(rows, prints)
        if len(result) != len(rows):
            raise ValueError(
                f"feature {self.name!r} returned {len(result)} values for {len(rows)} rows"
            )
        return pd.Series(np.asarray(result, dtype="float64"), index=rows.index, name=self.name)

    def max_rows(self, grid_ms: int = 100) -> int:
        """Rows the value can depend on before its own, on a grid of ``grid_ms``."""
        window = math.ceil(self.lookback_s * 1000 / grid_ms) + 1 if self.lookback_s > 0 else 0
        return window + self.lookback_rows


@dataclass
class TimedRegistry:
    """A named collection of timed features, kept as an object for tests."""

    features: dict[str, TimedFeature] = field(default_factory=dict)

    def add(self, feature: TimedFeature) -> TimedFeature:
        if feature.name in self.features:
            raise ValueError(f"feature {feature.name!r} is already registered")
        self.features[feature.name] = feature
        return feature

    def get(self, name: str) -> TimedFeature:
        try:
            return self.features[name]
        except KeyError as exc:
            raise KeyError(f"unknown timed feature {name!r}") from exc

    def names(self, plane: Plane | None = None) -> list[str]:
        return sorted(n for n, f in self.features.items() if plane is None or f.plane == plane)

    def longest_lookback_s(self, names: Iterable[str] | None = None) -> float:
        chosen = self.names() if names is None else list(names)
        return max((self.get(n).lookback_s for n in chosen), default=0.0)

    def longest_lookback_minutes(self, names: Iterable[str] | None = None) -> int:
        """``N``: the longest lookback in seconds, rounded up to whole minutes."""
        return math.ceil(self.longest_lookback_s(names) / 60.0)

    def longest_lookback_rows(self, names: Iterable[str] | None = None) -> int:
        chosen = self.names() if names is None else list(names)
        return max((self.get(n).lookback_rows for n in chosen), default=0)

    def max_rows(self, names: Iterable[str] | None = None, *, grid_ms: int = 100) -> int:
        """A purge in rows that covers every declared window on the grid."""
        chosen = self.names() if names is None else list(names)
        return max((self.get(n).max_rows(grid_ms) for n in chosen), default=0)

    def describe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "name": f.name,
                    "plane": f.plane,
                    "lookback_s": f.lookback_s,
                    "lookback_rows": f.lookback_rows,
                    "description": f.description,
                }
                for f in sorted(self.features.values(), key=lambda f: (f.plane, f.name))
            ]
        )

    def __contains__(self, name: object) -> bool:
        return name in self.features

    def __len__(self) -> int:
        return len(self.features)


#: The registry the study's feature modules populate on import.
TIMED_REGISTRY = TimedRegistry()


def timed_feature(
    *,
    name: str,
    plane: Plane,
    lookback_s: float,
    lookback_rows: int = 0,
    description: str = "",
    registry: TimedRegistry | None = None,
) -> Callable[[TimedFn], TimedFeature]:
    """Register a timed feature function ``fn(rows, prints) -> Series``."""
    target = registry if registry is not None else TIMED_REGISTRY

    def register(fn: TimedFn) -> TimedFeature:
        text = description or (fn.__doc__ or "").strip().split("\n")[0]
        return target.add(TimedFeature(name, plane, lookback_s, lookback_rows, fn, text))

    return register


def register_builder(
    builder: FrameBuilder,
    columns: Sequence[str],
    *,
    plane: Plane,
    lookback_s: float,
    lookback_rows: int,
    description: str,
    prefix: str = "",
    registry: TimedRegistry | None = None,
) -> list[TimedFeature]:
    """Register each column of a frame builder as a feature of its own.

    The declaration is the builder's: every column claims the builder's longest
    window, which is what the checker then holds each of them to.
    """
    target = registry if registry is not None else TIMED_REGISTRY
    added = []
    for column in columns:

        def one(
            rows: pd.DataFrame, prints: pd.DataFrame | None, _column: str = column
        ) -> pd.Series:
            return builder(rows, prints)[_column]

        feature = TimedFeature(
            prefix + column,
            plane,
            lookback_s,
            lookback_rows,
            one,
            f"{description} ({column})",
            builder=builder,
            column=column,
        )
        added.append(target.add(feature))
    return added


def build_timed(
    rows: pd.DataFrame,
    names: Sequence[str],
    *,
    prints: pd.DataFrame | None = None,
    registry: TimedRegistry | None = None,
) -> pd.DataFrame:
    """Compute named features on ``rows`` (and ``prints``), one builder pass each."""
    reg = registry if registry is not None else TIMED_REGISTRY
    if not names:
        raise ValueError("no features requested")
    cache: dict[int, pd.DataFrame] = {}
    columns: dict[str, np.ndarray] = {}
    for name in names:
        feature = reg.get(name)
        if feature.builder is not None and feature.column is not None:
            key = id(feature.builder)
            if key not in cache:
                if feature.plane == "trades" and prints is None:
                    raise ValueError(f"feature {name!r} reads prints; none were given")
                cache[key] = feature.builder(rows, prints)
            columns[name] = cache[key][feature.column].to_numpy(dtype="float64")
        else:
            columns[name] = feature(rows, prints).to_numpy()
    out = pd.DataFrame(columns, index=rows.index)
    if "timestamp" in rows.columns:
        out.insert(0, "timestamp", rows["timestamp"])
    return out
