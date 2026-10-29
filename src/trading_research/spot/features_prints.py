"""S25's nine print members and S25-PFT's twelve touch-proxy members, on the print grid.

All are built in a registry of this study's own (:func:`print_registry`), never
in the global ones, from the 2025 feature code, imported unchanged:

- **the eight flow members** (``flow_imbalance_{6,24,120}``,
  ``trade_intensity_{6,24,120}``, ``flow_persistence_24``,
  ``vwap_deviation_bp``) are columns of the 2025 flow builder
  (``features/study_set.flow_frame``: ``bucket_prints`` then
  ``build_flow_features``), run on the print grid's rows. Their 6-, 24- and
  120-row windows count print-grid rows, a registered change of meaning, and
  they keep the builder's 2025 declaration of 120 rows. ``vwap_deviation_bp``
  divides a row's print VWAP by the **previous** row's mid proxy, the two
  registered changes: the builder reads the mid from ``bid_price_0`` and
  ``ask_price_0``, so it is given the mid proxy shifted by one row in both;
- **``trade_run_length_60s``** is the 2025 timed feature, evaluated at
  print-grid stamps;
- **the twelve touch-proxy members** (S25-PFT only) are the 2025 functions of
  those names applied to the touch proxies (``spot/grid.py``) in place of the
  touch: the ask proxy as ``ask_price_0``, the bid proxy as ``bid_price_0``,
  sizes absent (NaN), each keeping its 2025 lookback.

Each is winsorised at the 0.1% and 99.9% quantiles of its own trailing
3,600 s on row stamps (``study_set.winsorise_trailing``), which adds 3,600 s
to its declared lookback, so ``N`` is 61 minutes. None is in raw size units,
so none is scaled. Every one is a ``trades``-plane feature, so
``validation/leakage_timed.check_timed_causal`` cuts the prints at a time
strictly inside a row interval; ``check_streaming_parity`` holds each to its
declaration. The proxies are row state (``spot/grid.py``); a test rebuilds the
grid from cut prints to check that they, too, read nothing after a row.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from trading_research.features import REGISTRY
from trading_research.features.micro2025 import REGISTRY_2025
from trading_research.features.study_set import (
    FLOW_LOOKBACK_ROWS,
    WINSOR_WINDOW_S,
    flow_frame,
    winsorise_trailing,
)
from trading_research.features.timed import (
    TimedFeature,
    TimedRegistry,
    build_timed,
    register_builder,
    seconds_ns,
    stamps_ns,
)
from trading_research.spot.grid import grid_stamps_ns, print_grid

FLOW_MEMBERS: tuple[str, ...] = (
    "flow_imbalance_6",
    "flow_imbalance_24",
    "flow_imbalance_120",
    "trade_intensity_6",
    "trade_intensity_24",
    "trade_intensity_120",
    "flow_persistence_24",
    "vwap_deviation_bp",
)
RUN_LENGTH = "trade_run_length_60s"
#: S25's nine print members, the order of ``configs/spot_2026.yaml``.
PRINT_MEMBERS: tuple[str, ...] = (*FLOW_MEMBERS, RUN_LENGTH)
#: S25-PFT's twelve touch-price members, recomputed from the touch proxies.
TOUCH_PROXY_MEMBERS: tuple[str, ...] = (
    "spread_bp",
    "quote_intensity_20",
    "mid_return_1_bp",
    "mid_return_20_bp",
    "realized_vol_50_bp",
    "realised_vol_20",
    "volatility_1s_bp",
    "turn_count_2s",
    "turn_count_5s",
    "turn_count_10s",
    "ask0_log_change_bp_1s",
    "ask0_log_change_bp_5s",
)


def _winsorise(values: pd.Series | np.ndarray, rows: pd.DataFrame) -> np.ndarray:
    return winsorise_trailing(np.asarray(values, dtype=np.float64), stamps_ns(rows))


def previous_mid_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """The grid's rows as the flow builder reads them: the previous row's mid proxy as touch."""
    previous = rows["mid_proxy"].to_numpy(dtype=np.float64)
    shifted = np.r_[np.nan, previous[:-1]] if len(previous) else previous
    return pd.DataFrame(
        {"timestamp": rows["timestamp"], "bid_price_0": shifted, "ask_price_0": shifted},
        index=rows.index,
    )


def proxy_touch_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """The grid's rows as a touch: the proxies for prices, sizes absent."""
    missing = np.full(len(rows), np.nan)
    return pd.DataFrame(
        {
            "timestamp": rows["timestamp"],
            "bid_price_0": rows["bid_proxy"].to_numpy(dtype=np.float64),
            "ask_price_0": rows["ask_proxy"].to_numpy(dtype=np.float64),
            "bid_size_0": missing,
            "ask_size_0": missing,
        },
        index=rows.index,
    )


def flow_members_frame(rows: pd.DataFrame, prints: pd.DataFrame | None) -> pd.DataFrame:
    """The eight flow members on the print grid, winsorised."""
    frame = flow_frame(previous_mid_rows(rows), prints)
    return pd.DataFrame(
        {name: _winsorise(frame[name], rows) for name in FLOW_MEMBERS}, index=rows.index
    )


def _run_length(rows: pd.DataFrame, prints: pd.DataFrame | None) -> pd.Series:
    values = REGISTRY_2025.get(RUN_LENGTH)(rows, prints)
    return pd.Series(_winsorise(values, rows), index=rows.index)


def _touch_member(name: str) -> TimedFeature:
    """The 2025 feature ``name`` on the proxies, winsorised, its lookback kept."""
    if name in REGISTRY_2025:
        timed = REGISTRY_2025.get(name)
        lookback_s, lookback_rows = timed.lookback_s, timed.lookback_rows

        def raw(touch: pd.DataFrame) -> pd.Series:
            return timed(touch, None)
    else:
        feature = REGISTRY.get(name)
        lookback_s, lookback_rows = 0.0, feature.lookback

        def raw(touch: pd.DataFrame) -> pd.Series:
            return feature(touch)

    def fn(rows: pd.DataFrame, _: pd.DataFrame | None) -> pd.Series:
        return pd.Series(_winsorise(raw(proxy_touch_rows(rows)), rows), index=rows.index)

    return TimedFeature(
        name,
        "trades",
        lookback_s + WINSOR_WINDOW_S,
        lookback_rows,
        fn,
        f"{name} of the touch proxies, winsorised",
    )


def print_registry() -> TimedRegistry:
    """A fresh registry holding the nine print members and the twelve touch-proxy members."""
    registry = TimedRegistry()
    register_builder(
        flow_members_frame,
        FLOW_MEMBERS,
        plane="trades",
        lookback_s=WINSOR_WINDOW_S,
        lookback_rows=FLOW_LOOKBACK_ROWS,
        description="2025 flow builder on the print grid, winsorised",
        registry=registry,
    )
    run = REGISTRY_2025.get(RUN_LENGTH)
    registry.add(
        TimedFeature(
            RUN_LENGTH,
            "trades",
            run.lookback_s + WINSOR_WINDOW_S,
            run.lookback_rows,
            _run_length,
            "2025 run length at print-grid stamps, winsorised",
        )
    )
    for name in TOUCH_PROXY_MEMBERS:
        registry.add(_touch_member(name))
    return registry


def history_tail(
    history: pd.DataFrame,
    first_ns: int,
    registry: TimedRegistry,
    names: Sequence[str],
) -> pd.DataFrame:
    """The earlier day's prints a day's first rows can depend on.

    Every print of the rows stamped after ``first_ns`` less the longest
    declared time window, of the longest row window (plus one) of rows before
    those, and the last print of each side before them, which carries the
    touch proxies across midnight.
    """
    if history.empty:
        return history
    stamps = grid_stamps_ns(history)
    since = first_ns - seconds_ns(registry.longest_lookback_s(names))
    first = int(np.searchsorted(stamps, since, side="left"))
    start = first - registry.longest_lookback_rows(names) - 1
    if start <= 0:
        return history
    times, side = stamps_ns(history), history["aggressor"].to_numpy()
    earlier = np.flatnonzero(times < stamps[start])
    carried = [earlier[side[earlier] == s][-1:] for s in (1, -1)]
    keep = np.sort(np.concatenate([*carried, np.flatnonzero(times >= stamps[start])]))
    return history.iloc[keep]


def day_frame(
    prints: pd.DataFrame,
    *,
    names: Sequence[str] = PRINT_MEMBERS,
    history: pd.DataFrame | None = None,
    registry: TimedRegistry | None = None,
) -> pd.DataFrame:
    """``timestamp`` and the named features at every print-grid row of a day.

    ``prints`` are the day's; ``history``, when the day is warm, the previous
    usable day's, of which :func:`history_tail` is read. A cold day passes
    none, and its proxies and windows start at its own first print.
    """
    reg = registry if registry is not None else print_registry()
    if not len(prints):
        raise ValueError("a day without a print has no grid")
    first_ns = int(stamps_ns(prints)[0])
    if history is not None:
        prints = pd.concat([history_tail(history, first_ns, reg, names), prints], ignore_index=True)
    rows = print_grid(prints)
    frame = build_timed(rows, list(names), prints=prints, registry=reg)
    return frame[stamps_ns(rows) >= first_ns].reset_index(drop=True)
