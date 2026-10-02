"""What a quoting rule may see, what it returns, and the baseline rule.

A quoter is called at book events only, with a :class:`MarketView` of that
snapshot and the position, and returns one :class:`Quote` per side. It sees no
fill, no markout and nothing after the snapshot; the arrays it is handed are
read-only views. Whether a quote becomes an order — the clip, the inventory
limits, the depth of the visible book, post-only on arrival — is the
simulator's decision, not the quoter's, so every quoter is held to the same
rules.

Only the baseline lives here for now: :class:`TouchQuoter` (S0), bid at the best
bid and ask at the best ask with the simulator's clip. The strategies the
pre-registration compares (S1-S4, X1) are added with the development-period
search.

The value types are named tuples rather than dataclasses because one of each is
built at every snapshot — several hundred thousand a day — and a tuple is the
cheapest immutable record Python has.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import NamedTuple, Protocol

import numpy as np


class MarketView(NamedTuple):
    """Everything a quoter may read at a decision. Prices in ticks."""

    #: The snapshot's timestamp, ns.
    ts: int
    tick: float
    #: The snapshot's levels, read-only, best first.
    bid_px: np.ndarray
    bid_sz: np.ndarray
    ask_px: np.ndarray
    ask_sz: np.ndarray
    best_bid: int
    best_ask: int
    #: Mid of the snapshot, in price units.
    mid: float
    #: EWMA of one-second mid returns, half-life 60 s, scaled to one minute, bp.
    vol_bp_1m: float
    #: Median touch size over the trailing hour, base units.
    touch_median_1h: float
    #: The simulator's clip for this decision, base units, already on the lot.
    clip: float
    #: The soft inventory limit for this decision, base units.
    soft_limit: float
    #: External signals, as of strictly before ``ts``.
    signals: Mapping[str, float]


class Quote(NamedTuple):
    """One side's wish. ``price`` None means do not quote this side."""

    price: int | None
    size: float


class Quotes(NamedTuple):
    bid: Quote
    ask: Quote


NO_QUOTE = Quote(None, 0.0)


class Quoter(Protocol):
    """A quoting rule."""

    #: Short label carried into results.
    name: str
    #: True only for rules that read the future (the ladder's forecast rungs).
    uses_future: bool

    def quotes(self, view: MarketView, position: float) -> Quotes: ...


@dataclass(frozen=True)
class TouchQuoter:
    """S0: join both touches with the clip, and nothing else.

    The baseline every other rule is measured against, and rung 0 of the
    advantage ladder. Its decisions read neither the fee nor the volatility, so
    its fee break-even is exact without a re-run.
    """

    name: str = "S0"
    uses_future: bool = False

    def quotes(self, view: MarketView, position: float) -> Quotes:  # noqa: ARG002
        return Quotes(Quote(view.best_bid, view.clip), Quote(view.best_ask, view.clip))
