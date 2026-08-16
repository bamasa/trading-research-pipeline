"""Aligning the trade plane onto the book grid.

Trades and book snapshots arrive on different clocks: a trade happens when
someone crosses, a snapshot when the touch changes. To use both in one model
they have to be put on one grid, and the join that does it is the single
easiest place in this project to introduce look-ahead.

The rule is that a row stamped *t* may only aggregate trades that happened at
or before *t*. That is a **backward** as-of join, and pandas will happily do a
forward or nearest one instead if asked — both of which read trades that had
not occurred yet, produce entirely plausible numbers, and inflate every result
downstream.

Why aggregate rather than join row to row: several trades can fall between two
snapshots and none may fall between another two. What a model can use at time
*t* is a summary of recent flow — how much traded, in which direction, how
fast — so the aggregation is the feature, not a preliminary to it.

Windows are expressed in time rather than in trade counts. A hundred trades
span a second in a busy market and a minute in a quiet one, so a count-based
window silently changes what it measures with activity; a time window keeps its
meaning and lets the *rate* become a feature in its own right.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from lobml.data.schema import TRADE_SCHEMA, signed_quantity

#: Windows used for trade-flow aggregation, as pandas offset strings.
DEFAULT_WINDOWS: tuple[str, ...] = ("1s", "5s", "30s", "300s")


def prepare_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Add the derived columns the aggregations need, once."""
    TRADE_SCHEMA.validate(trades)
    out = trades[["timestamp", "price", "quantity"]].copy()
    out["signed"] = signed_quantity(trades)
    out["notional"] = trades["price"] * trades["quantity"]
    out["is_buy"] = ~trades["is_buyer_maker"]
    out["count"] = 1.0
    return out.sort_values("timestamp", kind="stable").reset_index(drop=True)


def rolling_trade_stats(
    trades: pd.DataFrame,
    windows: Sequence[str] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """Aggregate trade flow over trailing time windows, indexed by trade time.

    Every window is closed on the right and trailing, so the value at a trade's
    timestamp summarises that trade and everything before it — never anything
    after.
    """
    frame = trades.set_index("timestamp")
    out = pd.DataFrame(index=frame.index)

    for window in windows:
        roll = frame.rolling(window, closed="right")
        volume = roll["quantity"].sum()
        signed = roll["signed"].sum()
        count = roll["count"].sum()
        notional = roll["notional"].sum()
        buys = roll["is_buy"].sum()

        out[f"trade_count_{window}"] = count
        out[f"trade_volume_{window}"] = volume
        out[f"signed_volume_{window}"] = signed
        # Bounded in [-1, 1]: the share of volume that was buyer-initiated,
        # centred at zero. Comparable across instruments, unlike raw signed size.
        out[f"trade_imbalance_{window}"] = signed / volume.where(volume > 0)
        out[f"buy_share_{window}"] = buys / count.where(count > 0)
        out[f"trade_vwap_{window}"] = notional / volume.where(volume > 0)
        out[f"trade_price_std_{window}"] = roll["price"].std()

    return out


def attach_to_book(
    book: pd.DataFrame,
    trade_stats: pd.DataFrame,
    *,
    tolerance: pd.Timedelta | None = None,
) -> pd.DataFrame:
    """Carry the latest trade summary onto each book row, backwards only.

    ``direction="backward"`` is the whole point of this function. The
    alternatives read trades from the future, and nothing about the result
    looks wrong afterwards.

    Rows with no preceding trade inside ``tolerance`` are left ``NaN`` rather
    than filled. A stale summary from an hour ago is not a description of
    current flow, and presenting it as one would let a model treat a dead market
    as an active one.
    """
    left = book[["timestamp"]].reset_index(drop=True)
    right = trade_stats.reset_index().rename(columns={"index": "timestamp"})

    merged = pd.merge_asof(
        left,
        right,
        on="timestamp",
        direction="backward",
        tolerance=tolerance,
        allow_exact_matches=True,
    )
    merged.index = book.index
    return merged.drop(columns=["timestamp"])


def build_trade_features(
    book: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    windows: Sequence[str] = DEFAULT_WINDOWS,
    tolerance: pd.Timedelta | None = None,
) -> pd.DataFrame:
    """Compute trade-flow features and align them to the book grid.

    Returns a frame on the book's index. Columns that depend on the instrument's
    units — raw volume, VWAP in quote currency — are converted to scale-free
    forms here, so that a model fitted on one instrument can be applied to
    another without reading it as permanently extreme.
    """
    if tolerance is None:
        tolerance = pd.Timedelta(60, unit="s")

    prepared = prepare_trades(trades)
    stats = rolling_trade_stats(prepared, windows)
    aligned = attach_to_book(book, stats, tolerance=tolerance)

    mid = (book["bid_price_0"] + book["ask_price_0"]) / 2.0
    out = pd.DataFrame(index=book.index)

    for window in windows:
        out[f"trade_imbalance_{window}"] = aligned[f"trade_imbalance_{window}"]
        out[f"buy_share_{window}"] = aligned[f"buy_share_{window}"]
        # Counts and volumes are unit-dependent, so they enter as logs: a log
        # difference is a ratio, which transfers where a level does not.
        out[f"log_trade_count_{window}"] = np.log1p(aligned[f"trade_count_{window}"])
        out[f"log_trade_volume_{window}"] = np.log1p(aligned[f"trade_volume_{window}"])
        # Where trades have printed relative to the current mid, in bp. Positive
        # means recent prints were above the mid.
        out[f"vwap_dev_bp_{window}"] = 1e4 * (aligned[f"trade_vwap_{window}"] - mid) / mid
        out[f"trade_disp_bp_{window}"] = 1e4 * aligned[f"trade_price_std_{window}"] / mid

    # Price impact per unit of flow: how far the mid moved over the window for
    # the signed volume that arrived. A liquidity measure that says what a trade
    # of a given size is likely to cost.
    for window, steps in (("1s", 10), ("5s", 50), ("30s", 300)):
        if f"signed_volume_{window}" not in aligned:
            continue
        move_bp = 1e4 * np.log(mid / mid.shift(steps))
        flow = aligned[f"signed_volume_{window}"]
        volume = aligned[f"trade_volume_{window}"]
        normalised_flow = flow / volume.where(volume > 0)
        out[f"impact_{window}"] = move_bp / normalised_flow.where(normalised_flow.abs() > 1e-9)

    return out
