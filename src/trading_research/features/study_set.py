"""The short-horizon study's microstructure set, as one registry of timed features.

The set is the registry's book features, the depth builder, the flow builder
and the 2025 additions, all over ten levels. The two builders produce frames
outside the registry; here each column becomes a registered feature with the
builder's declared lookback, so that the leakage checks, the purge and ``N``
cover them:

- the **depth builder** (:func:`trading_research.features.depth.build_depth_features`)
  reads the row alone: ``lookback_s=0``, no rows;
- the **flow builder** (:func:`trading_research.features.flow.build_flow_features`)
  runs on prints bucketed by the backward rule
  (:func:`trading_research.features.prints.bucket_prints`), never on
  ``flow.align_to_grid``. Its windows are in rows, the longest 120 (a median
  of 120 rows), so it declares 120 rows;
- the **registry's book features** keep their row lookbacks.

Row-declared windows are not covered by ``N``; on a cold-start day they are
covered by the rows since midnight.

Winsorising and raw sizes
-------------------------
Each feature of the set is clipped at the 0.1% and 99.9% quantiles of its own
trailing 3,600 s on row stamps. That adds 3,600 s to its declared lookback,
which is what makes ``N`` 61 minutes for this set. A feature in raw size units
is then divided by the median touch size on W
(:func:`fit_touch_size_scale`, written to the fitted-values file).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from trading_research.features import REGISTRY
from trading_research.features.depth import build_depth_features
from trading_research.features.flow import build_flow_features
from trading_research.features.micro2025 import REGISTRY_2025
from trading_research.features.prints import bucket_prints
from trading_research.features.registry import Feature
from trading_research.features.timed import (
    FrameBuilder,
    TimedFeature,
    TimedRegistry,
    register_builder,
    rolling_time,
    stamps_ns,
)

WINSOR_WINDOW_S: float = 3600.0
WINSOR_QUANTILES: tuple[float, float] = (0.001, 0.999)

DEPTH_COLUMNS: tuple[str, ...] = (
    "depth_imbalance",
    "weighted_depth_imbalance",
    "bid_concentration",
    "ask_concentration",
    "concentration_imbalance",
    "bid_slope_bp",
    "ask_slope_bp",
    "slope_imbalance",
    "bid_impact_bp",
    "ask_impact_bp",
    "impact_imbalance_bp",
    "log_visible_depth",
    "book_span_bp",
)

FLOW_COLUMNS: tuple[str, ...] = (
    "flow_imbalance_6",
    "flow_imbalance_24",
    "flow_imbalance_120",
    "trade_intensity_6",
    "trade_intensity_24",
    "trade_intensity_120",
    "volume_surge",
    "trade_concentration",
    "trade_size_shock",
    "vwap_deviation_bp",
    "flow_persistence_24",
)

#: The flow builder's longest window, in rows.
FLOW_LOOKBACK_ROWS: int = 120

#: Columns in raw size units, divided by W's median touch size before storage.
RAW_SIZE_COLUMNS: tuple[str, ...] = ("ofi_1", "ofi_20")

#: The 2025 additions, by name.
ADDED_2025: tuple[str, ...] = tuple(REGISTRY_2025.names())


def depth_frame(rows: pd.DataFrame, _: pd.DataFrame | None = None) -> pd.DataFrame:
    """The depth builder on ten levels."""
    return build_depth_features(rows, depth=10)


def flow_frame(rows: pd.DataFrame, prints: pd.DataFrame | None) -> pd.DataFrame:
    """The flow builder on prints aligned by the backward rule."""
    if prints is None:
        raise ValueError("the flow features read prints; none were given")
    return build_flow_features(bucket_prints(prints, rows))


def winsorise_trailing(
    values: np.ndarray,
    stamps: np.ndarray,
    window_s: float = WINSOR_WINDOW_S,
    quantiles: tuple[float, float] = WINSOR_QUANTILES,
) -> np.ndarray:
    """Clip at quantiles of the trailing ``window_s`` on row stamps, row included."""
    rolling = rolling_time(values, stamps, window_s)
    low = rolling.quantile(quantiles[0]).to_numpy()
    high = rolling.quantile(quantiles[1]).to_numpy()
    return np.clip(np.asarray(values, dtype="float64"), low, high)


def _winsorised_builder(builder: FrameBuilder) -> FrameBuilder:
    def wrapped(rows: pd.DataFrame, prints: pd.DataFrame | None) -> pd.DataFrame:
        frame = builder(rows, prints)
        stamps = stamps_ns(rows)
        return frame.apply(lambda column: winsorise_trailing(column.to_numpy(), stamps))

    return wrapped


def _winsorised(feature: TimedFeature, builders: dict[int, FrameBuilder]) -> TimedFeature:
    """The same feature clipped on its trailing quantiles, its lookback extended."""
    lookback_s = feature.lookback_s + WINSOR_WINDOW_S
    if feature.builder is not None and feature.column is not None:
        key = id(feature.builder)
        if key not in builders:
            builders[key] = _winsorised_builder(feature.builder)
        builder = builders[key]
        column = feature.column

        def from_builder(rows: pd.DataFrame, prints: pd.DataFrame | None) -> pd.Series:
            return builder(rows, prints)[column]

        return TimedFeature(
            feature.name,
            feature.plane,
            lookback_s,
            feature.lookback_rows,
            from_builder,
            feature.description,
            builder=builder,
            column=column,
        )

    def clipped(rows: pd.DataFrame, prints: pd.DataFrame | None) -> pd.Series:
        values = feature(rows, prints).to_numpy()
        return pd.Series(winsorise_trailing(values, stamps_ns(rows)), index=rows.index)

    return TimedFeature(
        feature.name, feature.plane, lookback_s, feature.lookback_rows, clipped, feature.description
    )


def _from_row_registry(feature: Feature) -> TimedFeature:
    def fn(rows: pd.DataFrame, _: pd.DataFrame | None) -> pd.Series:
        return feature(rows)

    return TimedFeature(feature.name, "book", 0.0, feature.lookback, fn, feature.description)


def microstructure_registry(*, winsorise: bool = True) -> TimedRegistry:
    """A fresh registry holding the whole microstructure set."""
    raw = TimedRegistry()
    for feature in REGISTRY.of_plane("book"):
        raw.add(_from_row_registry(feature))
    for name in REGISTRY_2025.names():
        raw.add(REGISTRY_2025.get(name))
    register_builder(
        depth_frame,
        DEPTH_COLUMNS,
        plane="book",
        lookback_s=0.0,
        lookback_rows=0,
        description="depth builder",
        registry=raw,
    )
    register_builder(
        flow_frame,
        FLOW_COLUMNS,
        plane="trades",
        lookback_s=0.0,
        lookback_rows=FLOW_LOOKBACK_ROWS,
        description="flow builder on backward-aligned prints",
        registry=raw,
    )
    if not winsorise:
        return raw
    out = TimedRegistry()
    builders: dict[int, FrameBuilder] = {}
    for name in raw.names():
        out.add(_winsorised(raw.get(name), builders))
    return out


def fit_touch_size_scale(days: Iterable[pd.DataFrame], decimals: int = 8) -> float:
    """The median touch size on W, both sides pooled, exact from counts."""
    counts: dict[float, int] = {}
    for rows in days:
        sizes = np.concatenate([rows["bid_size_0"].to_numpy(), rows["ask_size_0"].to_numpy()])
        values, found = np.unique(np.round(sizes.astype("float64"), decimals), return_counts=True)
        for value, count in zip(values.tolist(), found.tolist(), strict=True):
            counts[value] = counts.get(value, 0) + int(count)
    if not counts:
        raise ValueError("no rows were given")
    values = np.array(sorted(counts))
    cumulative = np.cumsum([counts[v] for v in values])
    total = int(cumulative[-1])
    lower = values[np.searchsorted(cumulative, (total - 1) // 2 + 1)]
    upper = values[np.searchsorted(cumulative, total // 2 + 1)]
    scale = float((lower + upper) / 2)
    if not scale > 0:
        raise ValueError(f"the median touch size is {scale}; it cannot scale anything")
    return scale


def scale_raw_sizes(
    frame: pd.DataFrame, scale: float, columns: Sequence[str] = RAW_SIZE_COLUMNS
) -> pd.DataFrame:
    """Divide the raw-size columns present in ``frame`` by W's median touch size."""
    out = frame.copy()
    for column in columns:
        if column in out.columns:
            out[column] = out[column] / scale
    return out
