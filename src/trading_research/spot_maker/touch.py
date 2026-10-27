"""The touch proxies at any instant, and a quote's price: post-only and skewed, on the tick.

The proxies are ``spot/grid.py``'s (the registration, "Common settings"): the
ask proxy is the price of the last buyer-initiated print stamped at or before
an instant, the bid proxy the last seller-initiated one's, the mid proxy their
mean; every print counts. On a warm day each side starts from the previous
usable day's last print of that side (``carried``); on a cold day a side is
undefined (NaN) until it prints.

A quote's price (:func:`quote_price`): the own side's proxy (the bid proxy for
a buy, the ask proxy for a sell), moved by the skew to ``price x (1 - skew_bp
x 1e-4 x position / clip)``, then held off the other side's proxy by a tick
(post-only: a buy at most ``ask - tick``, a sell at least ``bid + tick``, when
that proxy is defined), a buy rounded down to the tick and a sell up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from trading_research.spot.fills import PrintTape

BUY, SELL = 1, -1
#: Prices are compared on the tick within this share of a tick.
TICK_EPS = 1e-6


@dataclass(frozen=True)
class Proxies:
    """Each side's prints of one day, in archive order, and the carried prices."""

    ask_ts: np.ndarray
    ask_price: np.ndarray
    bid_ts: np.ndarray
    bid_price: np.ndarray
    carried_ask: float = math.nan
    carried_bid: float = math.nan

    @classmethod
    def from_tape(cls, tape: PrintTape, carried: tuple[float, float] | None = None) -> Proxies:
        """``carried`` is the previous usable day's last (buyer, seller) print prices."""
        buys, sells = tape.side == BUY, tape.side == SELL
        ask0, bid0 = carried if carried is not None else (math.nan, math.nan)
        return cls(tape.ts[buys], tape.price[buys], tape.ts[sells], tape.price[sells],
                   float(ask0), float(bid0))  # fmt: skip

    @staticmethod
    def _last(ts: np.ndarray, price: np.ndarray, at: np.ndarray, carried: float,
              strict: bool) -> np.ndarray:  # fmt: skip
        j = np.searchsorted(ts, at, side="left" if strict else "right") - 1
        out = np.full(len(at), carried, dtype=np.float64)
        known = j >= 0
        out[known] = price[j[known]]
        return out

    def at(self, ns: np.ndarray | int, *, strict: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """``(ask, bid)`` after every print stamped at (or, ``strict``, before) each instant."""
        at = np.atleast_1d(np.asarray(ns, dtype=np.int64))
        ask = self._last(self.ask_ts, self.ask_price, at, self.carried_ask, strict)
        bid = self._last(self.bid_ts, self.bid_price, at, self.carried_bid, strict)
        return ask, bid

    def mid(self, ns: np.ndarray | int, *, strict: bool = False) -> np.ndarray:
        ask, bid = self.at(ns, strict=strict)
        return (ask + bid) / 2.0

    def one(self, ns: int) -> tuple[float, float]:
        ask, bid = self.at(ns)
        return float(ask[0]), float(bid[0])


def last_prices(tape: PrintTape) -> tuple[float, float]:
    """The day's last buyer-initiated and seller-initiated print prices (NaN if none):
    what the next day carries."""
    out = []
    for side in (BUY, SELL):
        mine = np.flatnonzero(tape.side == side)
        out.append(float(tape.price[mine[-1]]) if len(mine) else math.nan)
    return out[0], out[1]


def on_tick(price: float, tick: float, side: int) -> float:
    """A buy's price rounded down to the tick, a sell's up."""
    steps = price / tick
    rounded = math.floor(steps + TICK_EPS) if side > 0 else math.ceil(steps - TICK_EPS)
    return round(rounded * tick, 10)


def quote_price(
    side: int, ask: float, bid: float, position: float, *, skew_bp: float, clip: float, tick: float
) -> float:
    """The price a quote of ``side`` is posted at; NaN when its own proxy is undefined."""
    own = bid if side > 0 else ask
    if not math.isfinite(own):
        return math.nan
    target = own * (1.0 - skew_bp * 1e-4 * position / clip)
    if side > 0 and math.isfinite(ask):
        target = min(target, ask - tick)
    if side < 0 and math.isfinite(bid):
        target = max(target, bid + tick)
    return on_tick(target, tick, side)
