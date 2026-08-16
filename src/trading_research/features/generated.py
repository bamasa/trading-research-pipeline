"""A wide generated feature set, and the machinery that produces it.

The hand-written features in :mod:`trading_research.features.book` are a small set chosen
for being individually checkable. This module takes the other approach: start
from a handful of base quantities and expand them mechanically across
transforms and windows, producing a few hundred columns, then let selection cut
them back down.

Both approaches are in the project on purpose. The small set is what a person
can reason about; the wide set is what actually gets used in practice, and it
brings its own failure mode — with enough columns something always looks
predictive on the training block. That is why the selection in
:mod:`trading_research.features.selection` is fitted strictly inside the training window,
and why the wide set is never used without it.

The expansion is deliberately generic: differences, log ratios, lags,
exponentially weighted means and standard deviations, rolling z-scores and
rank transforms, over several window lengths. Nothing here is instrument
specific, and every transform is causal by construction — a property the
existing look-ahead checks verify rather than assume.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from trading_research.data.schema import (
    ask_price_col,
    ask_size_col,
    bid_price_col,
    bid_size_col,
    mid_price,
)

#: Window lengths in observations. On a 100 ms grid: 0.5 s to 100 s.
DEFAULT_WINDOWS: tuple[int, ...] = (5, 20, 50, 200, 1000)

#: Lags in observations.
DEFAULT_LAGS: tuple[int, ...] = (1, 5, 20, 50)

BP = 1e4


def base_quantities(book: pd.DataFrame) -> pd.DataFrame:
    """The raw series everything else is expanded from.

    Kept small. Every one is a directly observable property of the touch, and
    every derived column below is a transform of one of these — so a surprising
    feature can always be traced back to something physical.
    """
    bid_price = book[bid_price_col(0)]
    ask_price = book[ask_price_col(0)]
    bid_size = book[bid_size_col(0)]
    ask_size = book[ask_size_col(0)]
    mid = mid_price(book)

    total = bid_size + ask_size
    out = pd.DataFrame(index=book.index)
    out["mid"] = mid
    out["log_mid"] = np.log(mid.to_numpy())
    out["spread_bp"] = BP * (ask_price - bid_price) / mid
    out["queue_imbalance"] = (bid_size - ask_size) / total.where(total > 0)
    out["log_bid_size"] = np.log1p(bid_size.to_numpy())
    out["log_ask_size"] = np.log1p(ask_size.to_numpy())
    out["log_total_size"] = np.log1p(total.to_numpy())
    out["log_size_ratio"] = np.log((bid_size / ask_size.where(ask_size > 0)).to_numpy())
    micro = (bid_price * ask_size + ask_price * bid_size) / total.where(total > 0)
    out["microprice_dev_bp"] = BP * (micro - mid) / mid
    return out


def expand(
    base: pd.DataFrame,
    *,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    lags: Sequence[int] = DEFAULT_LAGS,
    level_columns: Sequence[str] = ("spread_bp", "queue_imbalance", "microprice_dev_bp"),
    diff_columns: Sequence[str] = ("log_mid", "log_bid_size", "log_ask_size", "log_total_size"),
) -> pd.DataFrame:
    """Expand base quantities across transforms and windows.

    Two families, because the base quantities are two kinds of thing:

    ``level_columns`` are already stationary and comparable — a spread in basis
    points means the same today as last week — so they get windowed statistics
    and lags directly.

    ``diff_columns`` are levels that drift, so a rolling mean of one says more
    about where the price was than about the market. They are differenced
    first, and the differences get the same treatment.
    """
    # Accumulated in a dict and concatenated once. Assigning a few hundred
    # columns one at a time repeatedly reallocates the frame, which pandas
    # warns about and which dominates the runtime of this function.
    columns: dict[str, pd.Series] = {}

    for name in level_columns:
        if name not in base:
            continue
        series = base[name]
        columns[name] = series
        for lag in lags:
            columns[f"{name}_lag{lag}"] = series.shift(lag)
            columns[f"{name}_chg{lag}"] = series - series.shift(lag)
        for window in windows:
            roll = series.rolling(window, min_periods=window)
            mean = roll.mean()
            std = roll.std()
            columns[f"{name}_mean{window}"] = mean
            columns[f"{name}_std{window}"] = std
            columns[f"{name}_ewm{window}"] = series.ewm(
                span=window, adjust=False, min_periods=window
            ).mean()
            # A z-score against the recent past says "unusual for now", which
            # transfers across instruments and regimes where a level does not.
            columns[f"{name}_z{window}"] = _z_score(series, mean, std)

    for name in diff_columns:
        if name not in base:
            continue
        series = base[name]
        scale = BP if name == "log_mid" else 1.0
        step = series.diff()
        for lag in lags:
            columns[f"{name}_ret{lag}"] = scale * (series - series.shift(lag))
        for window in windows:
            roll = step.rolling(window, min_periods=window)
            columns[f"{name}_vol{window}"] = scale * roll.std()
            columns[f"{name}_drift{window}"] = scale * roll.mean()
            columns[f"{name}_absmean{window}"] = (
                scale * step.abs().rolling(window, min_periods=window).mean()
            )

    return pd.DataFrame(columns, index=base.index)


def _z_score(series: pd.Series, mean: pd.Series, std: pd.Series) -> pd.Series:
    """Standardise against a rolling mean, treating a flat window as zero.

    A zero standard deviation is not missing information — it says the quantity
    did not move, so its deviation from its own mean is exactly zero. Dividing
    and yielding NaN instead throws the row away, and that is not a rare edge
    case here: BTCUSDT futures sit at a one-tick spread almost permanently, so
    the naive version left 80% of rows unusable on the shortest window and
    would have silently discarded most of the sample.

    Only genuinely absent inputs — the warm-up rows — stay NaN.
    """
    deviation = series - mean
    scaled = deviation / std.where(std > 0)
    flat = (std == 0) & mean.notna()
    return scaled.mask(flat, 0.0)


def time_of_day(book: pd.DataFrame) -> pd.DataFrame:
    """Cyclical time-of-day and day-of-week terms.

    Encoded as sine and cosine pairs rather than as raw hour numbers, so that
    23:59 and 00:01 are adjacent. A raw hour column puts them maximally far
    apart, which is not a property of the market.

    These are genuinely available in advance — the clock is not a forecast — so
    they carry no look-ahead. They can still be spurious on a short sample: with
    thirty-odd days there are only about five of each weekday, so a "Tuesday
    effect" is five observations.
    """
    stamp = book["timestamp"].dt.tz_convert("UTC")
    seconds = stamp.dt.hour * 3600 + stamp.dt.minute * 60 + stamp.dt.second
    day_fraction = seconds / 86_400.0
    weekday = stamp.dt.dayofweek

    out = pd.DataFrame(index=book.index)
    out["tod_sin"] = np.sin(2 * np.pi * day_fraction.to_numpy())
    out["tod_cos"] = np.cos(2 * np.pi * day_fraction.to_numpy())
    out["dow_sin"] = np.sin(2 * np.pi * weekday.to_numpy() / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * weekday.to_numpy() / 7.0)
    return out


def generate(
    book: pd.DataFrame,
    trades: pd.DataFrame | None = None,
    *,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    lags: Sequence[int] = DEFAULT_LAGS,
    include_time: bool = True,
) -> pd.DataFrame:
    """Build the full generated feature frame for one book series.

    ``mid`` and ``log_mid`` are dropped from the output: they are price levels,
    and a model given the price level of a rising market will happily learn
    "high price means keep buying", which is a statement about February 2024
    rather than about markets.
    """
    base = base_quantities(book)
    parts = [expand(base, windows=windows, lags=lags)]

    if trades is not None and not trades.empty:
        from trading_research.features.align import build_trade_features

        parts.append(build_trade_features(book, trades))

    if include_time:
        parts.append(time_of_day(book))

    out = pd.concat(parts, axis=1)
    return out.drop(columns=[c for c in ("mid", "log_mid") if c in out.columns])
