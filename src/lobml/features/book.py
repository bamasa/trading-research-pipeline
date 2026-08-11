"""Order-book features.

Standard definitions from the open microstructure literature — Cont, Kukanov
and Stoikov on order flow imbalance; Gould et al.'s survey; the microprice as
used by Stoikov. Implemented from the papers rather than adapted from anywhere,
and kept deliberately small: a handful of well-understood quantities that can
each be checked by hand beats a wide set nobody can reason about.

Everything here works at the touch, so it runs on ``bookTicker`` data, which is
all any exchange publishes for free. Depth-dependent features — level
imbalance, book slope, concentration — need the collector and are added when
there is real data to test them against, not before.

Two conventions hold throughout:

**Scale-free units.** Prices differences are expressed in basis points of mid,
never in quote currency. A spread of 0.1 means nothing without knowing whether
the instrument trades at 40,000 or at 0.5, and a model trained on BTCUSDT
should be able to say something about XRPUSDT.

**Sign means direction.** Positive is bullish: buying pressure, upward
pressure on price. Every imbalance is bid minus ask, in that order, so a
positive value always points the same way.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lobml.data.schema import (
    ask_price_col,
    ask_size_col,
    bid_price_col,
    bid_size_col,
    mid_price,
)
from lobml.features.registry import feature

BP: float = 1e4


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide, returning NaN where the denominator vanishes.

    A zero denominator here means a degenerate book — no resting size on either
    side. Returning NaN says "unknown", which a split can drop. Returning zero
    would say "balanced", which is a statement about the market that the data
    does not support.
    """
    return numerator / denominator.where(denominator != 0)


def _log(values: pd.Series) -> pd.Series:
    """Natural log, preserving the Series and its index.

    ``np.log`` on a Series does return a Series at runtime, but its type stubs
    say ndarray, which loses the index for anything downstream that then wants
    to ``.rolling()`` or align. Wrapping once here keeps every call site plain.
    """
    return pd.Series(np.log(values.to_numpy(dtype="float64")), index=values.index)


# ---------------------------------------------------------------------------
# Static: functions of the current snapshot alone
# ---------------------------------------------------------------------------


@feature(
    name="spread_bp",
    plane="book",
    lookback=0,
    description="Quoted spread in basis points of mid.",
)
def spread_bp(df: pd.DataFrame) -> pd.Series:
    """Quoted spread, in basis points of mid.

    The most direct cost of taking liquidity, and the first thing any strategy
    on this horizon has to clear.
    """
    return BP * _safe_ratio(df[ask_price_col(0)] - df[bid_price_col(0)], mid_price(df))


@feature(
    name="queue_imbalance",
    plane="book",
    lookback=0,
    description="(bid size - ask size) / (bid size + ask size) at the touch, in [-1, 1].",
)
def queue_imbalance(df: pd.DataFrame) -> pd.Series:
    """Relative size imbalance at the touch.

    Bounded in [-1, 1], positive when the bid queue is larger. Among the
    best-documented short-horizon predictors of the next price move: the
    smaller queue tends to be consumed first, so the mid drifts towards the
    thinner side.
    """
    bid = df[bid_size_col(0)]
    ask = df[ask_size_col(0)]
    return _safe_ratio(bid - ask, bid + ask)


@feature(
    name="microprice_dev_bp",
    plane="book",
    lookback=0,
    description="Size-weighted mid minus arithmetic mid, in basis points.",
)
def microprice_dev_bp(df: pd.DataFrame) -> pd.Series:
    """Deviation of the microprice from the mid, in basis points.

    The microprice weights each side by the size resting on the *other* side,
    so it sits nearer the side more likely to be hit. Its deviation from the
    arithmetic mid is the imbalance expressed as a price rather than a ratio,
    which makes it directly comparable to the spread and to costs.
    """
    bid_price = df[bid_price_col(0)]
    ask_price = df[ask_price_col(0)]
    bid_size = df[bid_size_col(0)]
    ask_size = df[ask_size_col(0)]

    total = bid_size + ask_size
    micro = _safe_ratio(bid_price * ask_size + ask_price * bid_size, total)
    return BP * _safe_ratio(micro - mid_price(df), mid_price(df))


@feature(
    name="log_depth_ratio",
    plane="book",
    lookback=0,
    description="log(bid size / ask size) at the touch.",
)
def log_depth_ratio(df: pd.DataFrame) -> pd.Series:
    """Log ratio of resting sizes at the touch.

    Carries the same information as the queue imbalance but is unbounded and
    roughly symmetric, which suits linear models: doubling the bid side moves
    it by the same amount wherever it starts, whereas the bounded ratio
    saturates.
    """
    bid = df[bid_size_col(0)].where(df[bid_size_col(0)] > 0)
    ask = df[ask_size_col(0)].where(df[ask_size_col(0)] > 0)
    return _log(bid / ask)


@feature(
    name="log_total_depth",
    plane="book",
    lookback=0,
    description="log of total size resting at the touch.",
)
def log_total_depth(df: pd.DataFrame) -> pd.Series:
    """Log of the total size at the touch.

    A liquidity level rather than a direction: it says how much can be traded
    before the price moves, which is what decides whether an edge measured in
    basis points survives being acted on.
    """
    total = df[bid_size_col(0)] + df[ask_size_col(0)]
    return _log(total.where(total > 0))


# ---------------------------------------------------------------------------
# Dynamic: functions of the current snapshot and recent history
# ---------------------------------------------------------------------------


def _ofi(df: pd.DataFrame) -> pd.Series:
    """Order flow imbalance at the touch, per Cont-Kukanov-Stoikov.

    The contribution of one update depends on how the price moved:

    - bid price up, or unchanged: bid size added is buying pressure;
      bid price down means the whole queue left, so the contribution is minus
      the *previous* size.
    - the mirror image on the ask side, with the sign flipped.

    Encoding the price move is what distinguishes this from a naive size
    difference: a queue that disappears because the price improved means the
    opposite of one that disappears because it was consumed.
    """
    bid_price = df[bid_price_col(0)]
    ask_price = df[ask_price_col(0)]
    bid_size = df[bid_size_col(0)]
    ask_size = df[ask_size_col(0)]

    prev_bid_price = bid_price.shift(1)
    prev_ask_price = ask_price.shift(1)
    prev_bid_size = bid_size.shift(1)
    prev_ask_size = ask_size.shift(1)

    bid_flow = pd.Series(np.nan, index=df.index, dtype="float64")
    bid_flow = bid_flow.mask(bid_price > prev_bid_price, bid_size)
    bid_flow = bid_flow.mask(bid_price == prev_bid_price, bid_size - prev_bid_size)
    bid_flow = bid_flow.mask(bid_price < prev_bid_price, -prev_bid_size)

    ask_flow = pd.Series(np.nan, index=df.index, dtype="float64")
    ask_flow = ask_flow.mask(ask_price < prev_ask_price, ask_size)
    ask_flow = ask_flow.mask(ask_price == prev_ask_price, ask_size - prev_ask_size)
    ask_flow = ask_flow.mask(ask_price > prev_ask_price, -prev_ask_size)

    return bid_flow - ask_flow


@feature(
    name="ofi_1",
    plane="book",
    lookback=1,
    description="Order flow imbalance at the touch over one update.",
)
def ofi_1(df: pd.DataFrame) -> pd.Series:
    """Single-step order flow imbalance."""
    return _ofi(df)


@feature(
    name="ofi_20",
    plane="book",
    lookback=20,
    description="Order flow imbalance summed over the last 20 updates, in size units.",
)
def ofi_20(df: pd.DataFrame) -> pd.Series:
    """Order flow imbalance accumulated over a short window.

    A single update is mostly noise. Summing recent flow is what turns it into
    a usable signal, and the window length is the main thing worth varying.

    In raw size units, as originally defined. Prefer :func:`ofi_20_norm` for
    anything that has to hold across instruments.
    """
    return _ofi(df).rolling(20, min_periods=20).sum()


@feature(
    name="ofi_20_norm",
    plane="book",
    lookback=70,
    description="Order flow imbalance over 20 updates, divided by typical resting depth.",
)
def ofi_20_norm(df: pd.DataFrame) -> pd.Series:
    """Order flow imbalance as a fraction of the depth it moved against.

    Raw OFI is in size units, and size units are not comparable across
    instruments: on real February 2024 data, XRPUSDT flow has a standard
    deviation around 2x104 while BTCUSDT rests single-digit quantities at the
    touch. A model fitted on one would read the other as permanently extreme,
    which would make the BTC-to-XRP transfer experiment measure the unit
    mismatch rather than anything about markets.

    Dividing by recent average depth gives "how much of the resting book this
    flow represents" — dimensionless, comparable, and closer to the quantity
    that actually predicts a move. The 50-observation average is deliberately
    longer than the 20-observation flow window, so the denominator describes the
    prevailing state rather than tracking the numerator.
    """
    flow = _ofi(df).rolling(20, min_periods=20).sum()
    depth = (df[bid_size_col(0)] + df[ask_size_col(0)]).rolling(50, min_periods=50).mean()
    return _safe_ratio(flow, depth)


@feature(
    name="mid_return_1_bp",
    plane="book",
    lookback=1,
    description="Log return of mid over one observation, in basis points.",
)
def mid_return_1_bp(df: pd.DataFrame) -> pd.Series:
    """One-step log return of the mid, in basis points."""
    mid = mid_price(df)
    return BP * _log(mid / mid.shift(1))


@feature(
    name="mid_return_20_bp",
    plane="book",
    lookback=20,
    description="Log return of mid over 20 observations, in basis points.",
)
def mid_return_20_bp(df: pd.DataFrame) -> pd.Series:
    """Twenty-step log return of the mid, in basis points.

    Recent direction. Included as much to be controlled for as to predict:
    short-horizon returns mean-revert at some scales and trend at others, and a
    model that cannot see recent direction may attribute it to something else.
    """
    mid = mid_price(df)
    return BP * _log(mid / mid.shift(20))


@feature(
    name="realized_vol_50_bp",
    plane="book",
    lookback=51,
    description="Standard deviation of one-step mid returns over 50 observations, in basis points.",
)
def realized_vol_50_bp(df: pd.DataFrame) -> pd.Series:
    """Realized volatility of the mid over a short window, in basis points.

    Sets the scale everything else is judged against. A one-basis-point edge is
    substantial in a calm market and noise in a violent one, so a threshold
    that ignores volatility is really a different threshold in every regime.

    The lookback is 51, not 50: the window covers 50 one-step returns, and the
    first of those needs the observation before it.
    """
    mid = mid_price(df)
    step = _log(mid / mid.shift(1))
    return BP * step.rolling(50, min_periods=50).std()


@feature(
    name="spread_bp_ratio_50",
    plane="book",
    lookback=50,
    description="Current spread divided by its mean over the last 50 observations.",
)
def spread_bp_ratio_50(df: pd.DataFrame) -> pd.Series:
    """Current spread relative to its own recent average.

    Whether the book is unusually wide *for this instrument, now*. A ratio
    rather than a level, so it transfers across instruments whose typical
    spreads differ by orders of magnitude — which is exactly what the
    BTC-to-XRP experiment tests.
    """
    spread = spread_bp(df)
    return _safe_ratio(spread, spread.rolling(50, min_periods=50).mean())
