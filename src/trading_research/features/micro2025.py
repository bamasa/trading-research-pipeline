"""The 2025 microstructure set: what the registry lacked, with windows in seconds.

Six quantities the registry did not have, each re-specified so that it is
causal, bounded in time and in units a float16 store keeps:

- **one-second volatility**: the standard deviation of the mid's row-to-row
  simple returns over the trailing 1 s, in bp;
- **return-sign autocorrelation**: the mean of ``sign(r_j) * sign(r_{j-1})``
  over the trailing 1 s, in [-1, 1];
- **turn count**: reversals of the mid's direction inside the trailing window,
  a turn being a move opposite to the previous non-zero move, both inside the
  window. Counting every change of ``sign(r)`` instead, zeros included, mostly
  counts whether the mid moved at all on a 100 ms grid;
- **run length**: the length of the current run of prints on one aggressor
  side, signed by that side, counting only prints in the trailing 60 s;
- **inter-trade time**: the seconds between the last two distinct print stamps,
  both in the trailing 60 s, and the time since the last print, capped at 60 s;
- **lagged level-0 logs**: the log change of the best ask, the best bid and the
  mid since the last row stamped at or before ``s_t - L``, in bp. The lag is a
  change, never the lagged log level, which is a price and not a feature.

Every window is declared in seconds on row stamps. Prints follow the backward
rule of :mod:`trading_research.features.prints`. Nothing is winsorised or
filled here; :mod:`trading_research.features.study_set` winsorises on trailing
quantiles.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from trading_research.features.prints import print_stamps_ns
from trading_research.features.timed import (
    TimedRegistry,
    at_or_before,
    rolling_time,
    seconds_ns,
    stamps_ns,
    timed_feature,
    trailing_start,
)

BP: float = 1e4

#: Windows of the turn count, in seconds.
TURN_WINDOWS_S: tuple[float, ...] = (2.0, 5.0, 10.0)

#: Lags of the level-0 log changes, in seconds.
LAGS_S: tuple[float, ...] = (1.0, 5.0)

#: The trailing span the trade-plane state is read over, in seconds.
PRINT_SPAN_S: float = 60.0

#: The 2025 additions, kept apart from any other timed feature.
REGISTRY_2025 = TimedRegistry()


def _mid(rows: pd.DataFrame) -> np.ndarray:
    return (rows["bid_price_0"].to_numpy("float64") + rows["ask_price_0"].to_numpy("float64")) / 2


def _returns_bp(rows: pd.DataFrame) -> np.ndarray:
    mid = _mid(rows)
    out = np.full(len(mid), np.nan)
    out[1:] = (mid[1:] / mid[:-1] - 1.0) * BP
    return out


@timed_feature(
    name="volatility_1s_bp",
    plane="book",
    lookback_s=1.0,
    lookback_rows=1,
    registry=REGISTRY_2025,
)
def volatility_1s_bp(rows: pd.DataFrame, _: pd.DataFrame | None) -> pd.Series:
    """Std of the mid's row-to-row simple returns over the trailing 1 s, in bp."""
    return rolling_time(_returns_bp(rows), stamps_ns(rows), 1.0).std()


@timed_feature(
    name="return_sign_autocorr_1s",
    plane="book",
    lookback_s=1.0,
    lookback_rows=2,
    registry=REGISTRY_2025,
)
def return_sign_autocorr_1s(rows: pd.DataFrame, _: pd.DataFrame | None) -> pd.Series:
    """Mean of consecutive return-sign products over the trailing 1 s."""
    sign = np.sign(_returns_bp(rows))
    product = np.full(len(sign), np.nan)
    product[1:] = sign[1:] * sign[:-1]
    return rolling_time(product, stamps_ns(rows), 1.0).mean()


def turn_count(rows: pd.DataFrame, window_s: float) -> np.ndarray:
    """Reversals of the mid's direction with both moves inside the window."""
    stamps = stamps_ns(rows)
    direction = np.sign(np.diff(_mid(rows)))
    moved = np.flatnonzero(direction != 0) + 1
    signs = direction[moved - 1]
    turned = signs[1:] == -signs[:-1]
    turn_rows = moved[1:][turned]
    turn_from = moved[:-1][turned]
    t = np.arange(len(stamps))
    upto = np.searchsorted(turn_rows, t, side="right")
    since = np.searchsorted(turn_from, trailing_start(stamps, window_s), side="left")
    return np.maximum(upto - since, 0).astype("float64")


def _register_turns() -> None:
    for window in TURN_WINDOWS_S:

        def fn(rows: pd.DataFrame, _: pd.DataFrame | None, w: float = window) -> pd.Series:
            return pd.Series(turn_count(rows, w), index=rows.index)

        timed_feature(
            name=f"turn_count_{window:g}s",
            plane="book",
            lookback_s=window,
            lookback_rows=1,
            description=f"Reversals of the mid's direction over the trailing {window:g} s.",
            registry=REGISTRY_2025,
        )(fn)


def lag_log_change_bp(values: np.ndarray, stamps: np.ndarray, lag_s: float) -> np.ndarray:
    """``log(x_t / x_lag) * 1e4``, ``x_lag`` the last row at or before ``s_t - lag``."""
    before = at_or_before(stamps, stamps - seconds_ns(lag_s))
    out = np.full(len(values), np.nan)
    known = before >= 0
    out[known] = np.log(values[known] / values[before[known]]) * BP
    return out


def _register_lags() -> None:
    sources: dict[str, Callable[[pd.DataFrame], np.ndarray]] = {
        "ask0": lambda rows: rows["ask_price_0"].to_numpy("float64"),
        "bid0": lambda rows: rows["bid_price_0"].to_numpy("float64"),
        "mid": _mid,
    }
    for prefix, source in sources.items():
        for lag in LAGS_S:

            def fn(
                rows: pd.DataFrame,
                _: pd.DataFrame | None,
                s: Callable[[pd.DataFrame], np.ndarray] = source,
                lag_s: float = lag,
            ) -> pd.Series:
                return pd.Series(lag_log_change_bp(s(rows), stamps_ns(rows), lag_s), rows.index)

            timed_feature(
                name=f"{prefix}_log_change_bp_{lag:g}s",
                plane="book",
                lookback_s=lag,
                lookback_rows=1,
                description=f"Log change of {prefix} over {lag:g} s on row stamps, in bp.",
                registry=REGISTRY_2025,
            )(fn)


def _print_window(rows: pd.DataFrame, prints: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per row: the last print at or before its stamp, and the first in its span."""
    stamps = stamps_ns(rows)
    times = print_stamps_ns(prints)
    last = np.searchsorted(times, stamps, side="right") - 1
    first = np.searchsorted(times, stamps - seconds_ns(PRINT_SPAN_S), side="right")
    return last, first


@timed_feature(
    name="trade_run_length_60s",
    plane="trades",
    lookback_s=PRINT_SPAN_S,
    registry=REGISTRY_2025,
)
def trade_run_length(rows: pd.DataFrame, prints: pd.DataFrame | None) -> pd.Series:
    """Signed length of the current one-sided run of prints in the trailing 60 s."""
    assert prints is not None
    side = prints["aggressor"].to_numpy(dtype="int64")
    starts = np.ones(len(side), dtype=bool)
    starts[1:] = side[1:] != side[:-1]
    run_start = np.maximum.accumulate(np.where(starts, np.arange(len(side)), 0))
    last, first = _print_window(rows, prints)
    out = np.zeros(len(rows))
    seen = last >= first
    begin = np.maximum(run_start[last[seen]], first[seen])
    out[seen] = (last[seen] - begin + 1) * side[last[seen]]
    return pd.Series(out, index=rows.index)


@timed_feature(
    name="inter_trade_time_s",
    plane="trades",
    lookback_s=PRINT_SPAN_S,
    registry=REGISTRY_2025,
)
def inter_trade_time(rows: pd.DataFrame, prints: pd.DataFrame | None) -> pd.Series:
    """Seconds between the last two distinct print stamps in the trailing 60 s."""
    assert prints is not None
    stamps = stamps_ns(rows)
    distinct = np.unique(print_stamps_ns(prints))
    last = np.searchsorted(distinct, stamps, side="right") - 1
    out = np.full(len(rows), np.nan)
    if len(distinct) < 2:
        return pd.Series(out, index=rows.index)
    known = last >= 1
    gap_from = distinct[np.maximum(last - 1, 0)]
    known &= gap_from > stamps - seconds_ns(PRINT_SPAN_S)
    out[known] = (distinct[last[known]] - gap_from[known]) / 1e9
    return pd.Series(out, index=rows.index)


@timed_feature(
    name="time_since_trade_s",
    plane="trades",
    lookback_s=PRINT_SPAN_S,
    registry=REGISTRY_2025,
)
def time_since_trade(rows: pd.DataFrame, prints: pd.DataFrame | None) -> pd.Series:
    """Seconds since the last print at or before the stamp, capped at 60 s."""
    assert prints is not None
    stamps = stamps_ns(rows)
    times = print_stamps_ns(prints)
    last = np.searchsorted(times, stamps, side="right") - 1
    out = np.full(len(rows), PRINT_SPAN_S)
    known = last >= 0
    out[known] = np.minimum((stamps[known] - times[last[known]]) / 1e9, PRINT_SPAN_S)
    return pd.Series(out, index=rows.index)


_register_turns()
_register_lags()
