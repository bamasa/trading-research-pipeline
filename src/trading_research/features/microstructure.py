"""Features for the very short end, where a trade lasts seconds.

The book features in :mod:`trading_research.features.book` describe the state
of the touch. Over a two-minute horizon that is most of what is knowable from a
free feed. Over ten seconds it is not: what matters there is how the book is
*changing* — how fast quotes arrive, whether the imbalance is building or
decaying, whether the last few updates all pushed the same way.

Everything here is computed from ``bookTicker`` alone, so it runs on the same
free archives as the rest. Depth-dependent quantities still need a collector.

Two conventions carried over from the book module: prices in basis points of
mid, and positive means upward pressure.

A caution about lookbacks at this end. A twenty-row window is two minutes on
the coarse grid and thirty seconds on the fine one, so a feature's declared
lookback is in rows and its meaning in seconds depends on the sampling. That is
deliberate — the purge is in rows too — but it means a feature set tuned at one
resolution is not automatically tuned at another.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_research.data.schema import (
    ask_price_col,
    ask_size_col,
    bid_price_col,
    bid_size_col,
    mid_price,
)
from trading_research.features.registry import feature

BP: float = 1e4


def _log(values: pd.Series) -> pd.Series:
    """``np.log`` that keeps the Series. Losing it turns the next ``.rolling``
    into an attribute error, and the same wrapper exists in the book module for
    the same reason."""
    return pd.Series(np.log(values.to_numpy(dtype="float64")), index=values.index)


def _touch(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    bid = df[bid_price_col(0)]
    ask = df[ask_price_col(0)]
    return bid, ask, df[bid_size_col(0)], df[ask_size_col(0)], mid_price(df)


@feature(
    name="quote_intensity_20",
    plane="book",
    lookback=20,
    description="How often the touch actually changed over the last 20 rows.",
)
def quote_intensity_20(df: pd.DataFrame) -> pd.Series:
    """Share of recent rows where the quoted price moved.

    A proxy for how contested the touch is. A book being repriced on most
    updates is one where someone is working an order; a still one is not. At
    ten seconds this separates moments worth predicting from moments where
    nothing is happening and the model is fitting the clock.
    """
    bid, ask, _, _, _ = _touch(df)
    changed = ((bid.diff() != 0) | (ask.diff() != 0)).astype("float64")
    return changed.rolling(20, min_periods=1).mean()


@feature(
    name="imbalance_slope_10",
    plane="book",
    lookback=10,
    description="Whether queue imbalance is building or decaying.",
)
def imbalance_slope_10(df: pd.DataFrame) -> pd.Series:
    """Change in queue imbalance over ten rows.

    The level says which side is heavier; the slope says whether it is getting
    heavier. Over a two-minute horizon the level dominates, because the
    imbalance mean-reverts long before the horizon is up. Over ten seconds the
    direction of travel is the more informative half.
    """
    _, _, bid_size, ask_size, _ = _touch(df)
    total = bid_size + ask_size
    imbalance = (bid_size - ask_size) / total.where(total > 0)
    return imbalance - imbalance.shift(10)


@feature(
    name="touch_persistence_20",
    plane="book",
    lookback=20,
    description="How long the current best bid and ask have survived.",
)
def touch_persistence_20(df: pd.DataFrame) -> pd.Series:
    """Rows since either side of the touch last moved, capped at the window.

    A quote that has stood for a while is one nobody wants to trade through.
    Capped rather than unbounded so that a quiet overnight stretch does not
    produce a value hundreds of times larger than anything in the training
    window and drag every normalisation with it.
    """
    bid, ask, _, _, _ = _touch(df)
    moved = (bid.diff() != 0) | (ask.diff() != 0)
    groups = moved.cumsum()
    since = moved.groupby(groups).cumcount().astype("float64")
    return since.clip(upper=20)


@feature(
    name="mid_reversal_10",
    plane="book",
    lookback=10,
    description="Whether the last move undid the one before it.",
)
def mid_reversal_10(df: pd.DataFrame) -> pd.Series:
    """Product of two consecutive five-row returns, negated.

    Positive when the recent path reversed, negative when it continued. At this
    horizon the mid oscillates between bid and ask, and a great deal of what
    looks like momentum is that oscillation; this measures it directly instead
    of leaving a model to infer it from a lag.
    """
    mid = mid_price(df)
    recent = _log(mid / mid.shift(5)) * BP
    previous = _log(mid.shift(5) / mid.shift(10)) * BP
    return -(recent * previous)


@feature(
    name="spread_pressure_20",
    plane="book",
    lookback=20,
    description="Current spread against its recent typical value.",
)
def spread_pressure_20(df: pd.DataFrame) -> pd.Series:
    """Spread relative to its own twenty-row median.

    Scale-free by construction, so it means the same thing on an instrument
    quoted at 40,000 and one quoted at 0.5 — which the raw spread in basis
    points does not, since one of them is pinned at a single tick and the other
    is not.
    """
    bid, ask, _, _, mid = _touch(df)
    spread = (ask - bid) / mid * BP
    typical = spread.rolling(20, min_periods=5).median()
    return spread / typical.where(typical > 0)


@feature(
    name="size_shock_10",
    plane="book",
    lookback=10,
    description="Sudden change in resting size at the touch.",
)
def size_shock_10(df: pd.DataFrame) -> pd.Series:
    """Change in total touch size against its recent level.

    Size disappearing from the touch is the visible half of an order being
    filled or pulled, and either way it precedes a move more often than a full
    book does. Signed so that bid-side depletion is negative.
    """
    _, _, bid_size, ask_size, _ = _touch(df)
    signed = bid_size - ask_size
    total = (bid_size + ask_size).rolling(10, min_periods=3).mean()
    return (signed - signed.shift(1)) / total.where(total > 0)


@feature(
    name="micro_drift_5",
    plane="book",
    lookback=5,
    description="Microprice movement over five rows, in basis points.",
)
def micro_drift_5(df: pd.DataFrame) -> pd.Series:
    """Return of the microprice rather than the mid.

    The microprice weights each side by the *opposite* queue, so it leans
    towards the side likely to be hit. Its short-run drift is a cleaner
    momentum measure than the mid's, which spends most of its variance
    bouncing between the two quotes.
    """
    bid, ask, bid_size, ask_size, _ = _touch(df)
    total = bid_size + ask_size
    micro = (bid * ask_size + ask * bid_size) / total.where(total > 0)
    return _log(micro / micro.shift(5)) * BP


@feature(
    name="realised_vol_20",
    plane="book",
    lookback=20,
    description="Short-window realised volatility of the mid, in basis points.",
)
def realised_vol_20(df: pd.DataFrame) -> pd.Series:
    """Standard deviation of one-row mid returns over twenty rows.

    The denominator of the edge identity — expected edge is the information
    coefficient times this. Present as a feature as well as a diagnostic
    because the model has to know when a move is large *for now* rather than
    large in absolute terms.
    """
    mid = mid_price(df)
    returns = _log(mid / mid.shift(1)) * BP
    return returns.rolling(20, min_periods=5).std()
