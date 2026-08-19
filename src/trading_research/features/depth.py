"""Features that need more of the book than the touch.

These were defined in prose in :mod:`trading_research.features.book` and left
unimplemented, because Binance publishes one level and inventing the rest would
have been worse than going without. Bybit publishes five hundred, so they can be
written now.

What the extra levels are supposed to add
-----------------------------------------
The touch answers one question: which side is heavier right now. Everything
past it answers a different one — how much it would *cost* to move the price,
and how quickly resistance builds. Two books with identical touches can be
completely different: one with a wall three ticks out and one with nothing for
fifty.

That distinction is what the whole project has been missing. A model given only
the touch cannot tell a price that is pinned from one that is free to move, and
the difference decides whether a predicted move is realisable.

The three shapes worth measuring
--------------------------------
**Slope** — how fast size accumulates away from the mid. A steep book resists
movement; a flat one does not.

**Concentration** — what share of the visible size sits at the very front. A
book whose depth is all at the touch is one large order away from being empty.

**Weighted imbalance** — the touch imbalance generalised. Levels further out
count less, because they are further from being hit and more likely to be
cancelled before they are.

Every one is scale-free: a ratio, a share or a slope in basis points, so a model
fitted on one instrument means the same thing on another.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BP: float = 1e4


def _levels(frame: pd.DataFrame, side: str, depth: int) -> tuple[np.ndarray, np.ndarray]:
    prices = np.column_stack([frame[f"{side}_price_{i}"].to_numpy() for i in range(depth)])
    sizes = np.column_stack([frame[f"{side}_size_{i}"].to_numpy() for i in range(depth)])
    return prices, sizes


def available_depth(frame: pd.DataFrame) -> int:
    """How many levels a frame actually carries, per side."""
    depth = 0
    while all(f"{side}_price_{depth}" in frame.columns for side in ("bid", "ask")):
        depth += 1
    return depth


def build_depth_features(frame: pd.DataFrame, *, depth: int | None = None) -> pd.DataFrame:
    """Compute the depth-dependent features on a multi-level book frame.

    ``depth`` defaults to whatever the frame carries. Asking for more than it
    has fails rather than filling with zeros: a level that is not there is not a
    level with no size, and a feature that quietly treats it as one reports a
    thin book wherever the data is merely absent.
    """
    present = available_depth(frame)
    if present < 2:
        raise ValueError(f"depth features need at least two levels a side, found {present}")
    depth = present if depth is None else depth
    if depth > present:
        raise ValueError(f"asked for {depth} levels, frame carries {present}")

    bid_price, bid_size = _levels(frame, "bid", depth)
    ask_price, ask_size = _levels(frame, "ask", depth)
    mid = (bid_price[:, 0] + ask_price[:, 0]) / 2.0
    out = pd.DataFrame(index=frame.index)

    # Distance of each level from the mid, in basis points. Scale-free, so a
    # book on a $40,000 instrument and a $0.50 one are described alike.
    bid_distance = (mid[:, None] - bid_price) / mid[:, None] * BP
    ask_distance = (ask_price - mid[:, None]) / mid[:, None] * BP

    bid_total = bid_size.sum(axis=1)
    ask_total = ask_size.sum(axis=1)
    total = bid_total + ask_total

    with np.errstate(divide="ignore", invalid="ignore"):
        # The touch imbalance generalised across levels. Weighted by the inverse
        # distance, so a level ten basis points out counts a tenth of one at the
        # touch — further orders are both less likely to be hit and more likely
        # to be pulled first.
        bid_weight = np.where(bid_distance > 0, 1.0 / bid_distance, 0.0)
        ask_weight = np.where(ask_distance > 0, 1.0 / ask_distance, 0.0)
        weighted_bid = (bid_size * bid_weight).sum(axis=1)
        weighted_ask = (ask_size * ask_weight).sum(axis=1)
        weighted_total = weighted_bid + weighted_ask

        out["depth_imbalance"] = np.divide(
            bid_total - ask_total, total, out=np.zeros_like(total), where=total > 0
        )
        out["weighted_depth_imbalance"] = np.divide(
            weighted_bid - weighted_ask,
            weighted_total,
            out=np.zeros_like(weighted_total),
            where=weighted_total > 0,
        )

        # Concentration: the share of visible size sitting at the front. A book
        # whose depth is all at the touch is one order away from being empty,
        # and looks identical to a deep one from the touch alone.
        out["bid_concentration"] = np.divide(
            bid_size[:, 0], bid_total, out=np.zeros_like(bid_total), where=bid_total > 0
        )
        out["ask_concentration"] = np.divide(
            ask_size[:, 0], ask_total, out=np.zeros_like(ask_total), where=ask_total > 0
        )
        out["concentration_imbalance"] = out["bid_concentration"] - out["ask_concentration"]

        # Slope: basis points of distance bought per unit of cumulative size.
        # Small means a steep book that resists movement; large means a thin one
        # where a modest order walks a long way.
        bid_cumulative = np.cumsum(bid_size, axis=1)
        ask_cumulative = np.cumsum(ask_size, axis=1)
        out["bid_slope_bp"] = np.divide(
            bid_distance[:, -1],
            bid_cumulative[:, -1],
            out=np.zeros_like(bid_total),
            where=bid_cumulative[:, -1] > 0,
        )
        out["ask_slope_bp"] = np.divide(
            ask_distance[:, -1],
            ask_cumulative[:, -1],
            out=np.zeros_like(ask_total),
            where=ask_cumulative[:, -1] > 0,
        )
        out["slope_imbalance"] = out["ask_slope_bp"] - out["bid_slope_bp"]

        # Cost of moving the price: the size-weighted average distance a market
        # order would pay to consume the visible book. This is the quantity the
        # touch cannot express at all.
        out["bid_impact_bp"] = np.divide(
            (bid_size * bid_distance).sum(axis=1),
            bid_total,
            out=np.zeros_like(bid_total),
            where=bid_total > 0,
        )
        out["ask_impact_bp"] = np.divide(
            (ask_size * ask_distance).sum(axis=1),
            ask_total,
            out=np.zeros_like(ask_total),
            where=ask_total > 0,
        )
        out["impact_imbalance_bp"] = out["ask_impact_bp"] - out["bid_impact_bp"]

        # Total visible size, logged: a level rather than a shape, but the shapes
        # above mean different things in a thick book and a thin one.
        out["log_visible_depth"] = np.log(np.where(total > 0, total, np.nan))

        # How far the book reaches. Two books with the same size can span very
        # different distances, and the span is what a large order actually pays.
        out["book_span_bp"] = (bid_distance[:, -1] + ask_distance[:, -1]) / 2.0

    return out
