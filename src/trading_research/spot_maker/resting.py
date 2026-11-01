"""Resting limit orders filled from prints, and the taker flatten: the study's only fills.

Registered under "Fills". An order of side ``s`` (+1 buy, -1 sell), limit
price ``p`` and size ``q`` is live on ``[start_ns, end_ns)`` (its decision +
latency to its cancel's decision + latency). It fills on the reachable prints
of the **other** aggressor side (a buy on seller-initiated prints), stamped
while it is live, in archive order, each print filling ``min(remaining, the
print's size)`` at ``p``:

* ``trade_through`` (the primary rule): prints strictly beyond ``p`` (below
  ``p`` for a buy, above for a sell); the queue at ``p`` is taken as exhausted;
* ``touch`` (the optimistic bracket): prints at ``p`` or beyond;
* ``delayed`` (K-fill's run): ``trade_through`` with the first qualifying
  print filling nothing, so every fill is one further print later.

A fill's time is its print's stamp. A remainder of at most
:data:`~trading_research.spot_maker.registration.REMAINDER_BTC` is none.

**The flatten** (:func:`flatten`) is the spot study's side-correct taker rule,
``spot/fills.fill_orders``, unchanged: one order of the position's size at
23:59 + latency, bounded by the day's end, falling back to the last print of
its side (or either side) when none follows; its time is its print's stamp, or
its arrival when it fell back.
"""

from __future__ import annotations

import math

import numpy as np

from trading_research.spot.fills import PrintFillSettings, PrintTape, fill_orders
from trading_research.spot_maker.registration import REMAINDER_BTC

RULES: tuple[str, ...] = ("trade_through", "touch", "delayed")
EMPTY = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64))


def bounds(tape: PrintTape, side: int, start_ns: np.ndarray, end_ns: np.ndarray
           ) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """For orders of ``side`` live on ``[start, end)``: the range of the other side's
    reachable prints each may fill on."""
    other = tape.side_prints(-side)
    return (np.searchsorted(other.ts, start_ns, side="left").astype(np.int64),
            np.searchsorted(other.ts, end_ns, side="left").astype(np.int64))  # fmt: skip


def fill_range(tape: PrintTape, side: int, price: float, size: float, lo: int, hi: int,
               rule: str) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """``(tape positions, filled sizes)`` of an order whose live prints of the other side
    are ``[lo, hi)``, in time order."""
    if rule not in RULES:
        raise ValueError(f"unknown fill rule {rule!r}; expected one of {RULES}")
    if hi <= lo or size <= REMAINDER_BTC or not math.isfinite(price):
        return EMPTY
    other = tape.side_prints(-side)
    seen = other.price[lo:hi]
    if rule == "touch":
        beyond = (seen <= price) if side > 0 else (seen >= price)
    else:
        beyond = (seen < price) if side > 0 else (seen > price)
    at = np.flatnonzero(beyond) + lo
    if rule == "delayed":
        at = at[1:]
    if not len(at):
        return EMPTY
    sizes = other.size[at]
    before = np.cumsum(sizes) - sizes
    take = np.minimum(np.maximum(size - before, 0.0), sizes)
    keep = take > REMAINDER_BTC
    return other.index[at[keep]], take[keep]


def fill(
    tape: PrintTape, side: int, price: float, size: float, start_ns: int, end_ns: int, rule: str
) -> tuple[np.ndarray, np.ndarray]:
    """``(tape positions, filled sizes)`` of a resting order live on ``[start, end)``."""
    if end_ns <= start_ns:
        if rule not in RULES:
            raise ValueError(f"unknown fill rule {rule!r}; expected one of {RULES}")
        return EMPTY
    lo, hi = bounds(tape, side, np.array([start_ns]), np.array([end_ns]))
    return fill_range(tape, side, price, size, int(lo[0]), int(hi[0]), rule)


def flatten(tape: PrintTape, side: int, size: float, decision_ns: int,
            latency_ns: int) -> tuple[int, float, bool]:  # fmt: skip
    """``(time, price, fell_back)`` of the taker order closing a position of ``size``
    by a ``side`` order decided at ``decision_ns``."""
    settings = PrintFillSettings(clip=size, latency_ns=latency_ns)
    arrival = decision_ns + latency_ns
    found = fill_orders(tape, side, np.array([arrival]), np.array([decision_ns]),
                        until_ns=tape.end_ns, fallback=True, settings=settings)  # fmt: skip
    price = float(found.rule[0])
    if not math.isfinite(price):
        raise ValueError(f"{tape.day}: the flatten found no print to be priced at")
    fell_back = bool(found.fallback[0])
    return (arrival if fell_back else int(found.print_ns[0])), price, fell_back
