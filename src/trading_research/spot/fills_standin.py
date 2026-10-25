"""A stand-in for ``spot/fills.py``: the side-correct print rule and its brackets.

``spot/fills.py`` is written on another branch; until it is merged, the print
executor (:mod:`.print_taker`) reaches the rule through this module, by one
import there, and the print walk tables reach it only through the executor.
At the merge that import names ``spot.fills`` and this file is deleted. The
interface assumed of ``spot/fills.py`` is exactly this module's public names.

The rule, as registered ("Fills", "On prints, the side-correct rule"):

* An order of side +1 (a buy) fills at the first **buyer-initiated** print
  stamped at or after its arrival, a sell (-1) at the first **seller-initiated**
  one. Prints flagged unreachable (RPI or an equivalent) are never filled
  against. A fill's print must be stamped before ``until_ns`` (23:59 for an
  entry, the day's end for an exit), or there is no fill.
* The clip walks that print and the other reachable prints of the same side
  with the same stamp, in archive order, at their volume-weighted price, with
  :func:`trading_research.backtest.walk.walk_book`'s arithmetic. Because the
  first print at or after an arrival opens its stamp's group, the sweep depends
  only on the group, and is computed once per group (:func:`sweeps`).
* The part beyond the sweep is charged at the sweep's last price (``rule``).

The brackets change the price only, never which print fills or when:

* ``optimistic``: the whole clip at the first print's price;
* ``pessimistic_depth``: the part beyond the sweep at the worst price (highest
  for a buy, lowest for a sell) of a reachable print of the same side stamped
  after the fill's print, at most ``window_ns`` (60 s) after it and before
  ``window_end_ns`` (23:59); the sweep's last price if there is none;
* ``stale``: the whole clip at the last reachable print of the same side stamped
  at or before the decision; the rule's price if the day has none;
* ``worse_of_two``: the worse of ``pessimistic_depth`` and ``stale``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import NamedTuple

import numpy as np

from trading_research.market_making.events import NS_PER_S

RULES: tuple[str, ...] = ("rule", "optimistic", "pessimistic_depth", "stale", "worse_of_two")
WINDOW_NS = 60 * NS_PER_S


@dataclass(frozen=True)
class DayPrints:
    """One day's prints in archive order: what ``spot/loader.py`` is assumed to give.

    ``side`` is +1 for a buyer-initiated print (the taker bought) and -1 for a
    seller-initiated one; ``reachable`` is False for a print no order can fill
    against. Prices are raw float64 quote prices, sizes base units.
    """

    day: date
    ts: np.ndarray
    price: np.ndarray
    size: np.ndarray
    side: np.ndarray
    reachable: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.ts)
        if any(len(a) != n for a in (self.price, self.size, self.side, self.reachable)):
            raise ValueError("one price, size, side and reachable flag per print")
        if n > 1 and np.any(np.diff(self.ts) < 0):
            raise ValueError("print stamps must be in time order")
        if not np.isin(self.side, (-1, 1)).all():
            raise ValueError("a print's side is +1 (buyer-initiated) or -1 (seller-initiated)")
        if not (np.isfinite(self.price).all() and (self.price > 0).all()):
            raise ValueError("raw print prices only: positive and finite")
        if not (np.isfinite(self.size).all() and (self.size > 0).all()):
            raise ValueError("a print's size must be positive and finite")


@dataclass(frozen=True)
class Sweeps:
    """One side's reachable prints, grouped by stamp, walked for the clip once."""

    side: int
    clip: float
    #: Archive positions, stamps and prices of the side's reachable prints.
    position: np.ndarray
    ts: np.ndarray
    price: np.ndarray
    #: Per print of the side: its stamp group's index.
    group: np.ndarray
    #: Per group: the first print's index in the side's arrays, and the sweep.
    first: np.ndarray
    rule_price: np.ndarray
    pessimistic_price: np.ndarray
    beyond: np.ndarray
    sweep: np.ndarray


class PrintFills(NamedTuple):
    """One fill per order. Where ``found`` is False the order has no print to fill
    at: ``position`` and ``print_ns`` are -1 and the money is NaN."""

    found: np.ndarray
    #: The fill's first print, as a position in the day's archive order.
    position: np.ndarray
    print_ns: np.ndarray
    #: The clip's price per unit under the rule asked for.
    price: np.ndarray
    #: The part of the clip beyond the sweep, and the sweep's size.
    beyond: np.ndarray
    sweep: np.ndarray


def _range_extreme(values: np.ndarray, a: np.ndarray, b: np.ndarray, side: int) -> np.ndarray:
    """``max`` (``side`` +1) or ``min`` (-1) of ``values[a:b]`` per pair; NaN when empty."""
    out = np.full(len(a), np.nan)
    ok = np.flatnonzero(a < b)
    if not len(ok):
        return out
    op = np.maximum if side > 0 else np.minimum
    levels = [values]
    while 2 ** len(levels) <= len(values):
        span = 2 ** (len(levels) - 1)
        levels.append(op(levels[-1][:-span], levels[-1][span:]))
    length = b[ok] - a[ok]
    k = np.floor(np.log2(length)).astype(np.int64)
    k = np.where(2 ** (k + 1) <= length, k + 1, np.where(2**k > length, k - 1, k))
    for level in np.unique(k):
        at = ok[k == level]
        left, right = a[at], b[at] - 2 ** int(level)
        out[at] = op(levels[level][left], levels[level][right])
    return out


def sweeps(
    prints: DayPrints, side: int, clip: float, *, window_end_ns: int, window_ns: int = WINDOW_NS
) -> Sweeps:
    """Group ``side``'s reachable prints by stamp and walk each group for ``clip``."""
    if side not in (-1, 1):
        raise ValueError("a side is +1 (buy) or -1 (sell)")
    if not clip > 0:
        raise ValueError(f"the clip must be positive, got {clip}")
    position = np.flatnonzero((prints.side == side) & prints.reachable).astype(np.int64)
    ts = np.asarray(prints.ts, dtype=np.int64)[position]
    px = np.asarray(prints.price, dtype=np.float64)[position]
    sz = np.asarray(prints.size, dtype=np.float64)[position]
    n = len(ts)
    new = np.r_[True, ts[1:] != ts[:-1]] if n else np.zeros(0, dtype=bool)
    first = np.flatnonzero(new).astype(np.int64)
    lengths = np.diff(np.r_[first, n]).astype(np.int64)
    left, cost = np.full(len(first), float(clip)), np.zeros(len(first))
    for level in range(int(lengths.max()) if n else 0):
        active = np.flatnonzero(lengths > level)
        i = first[active] + level
        take = np.minimum(left[active], sz[i])
        cost[active] += take * px[i]
        left[active] -= take
    beyond = np.where(left > 0.0, left, 0.0)
    last = px[first + lengths - 1] if n else np.zeros(0)
    after = first + lengths
    stamp = ts[first]
    stop = np.minimum(
        np.searchsorted(ts, stamp + window_ns, side="right"),
        np.searchsorted(ts, window_end_ns, side="left"),
    )
    worst = _range_extreme(px, after, stop, side)
    worst = np.where(np.isnan(worst), last, worst)
    return Sweeps(
        side,
        float(clip),
        position,
        ts,
        px,
        (np.cumsum(new) - 1).astype(np.int64),
        first,
        (cost + beyond * last) / clip,
        (cost + beyond * worst) / clip,
        beyond,
        np.add.reduceat(sz, first) if n else np.zeros(0),
    )


def worse(side: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The price worse for the account: the higher for a buy, the lower for a sell."""
    return np.maximum(a, b) if side > 0 else np.minimum(a, b)


def fill(
    sw: Sweeps,
    arrival_ns: np.ndarray,
    *,
    until_ns: int | np.ndarray,
    decision_ns: np.ndarray,
    rule: str = "rule",
) -> PrintFills:
    """Fill orders of ``sw.side`` arriving at ``arrival_ns`` under ``rule``.

    Each order's result depends on its own arrival, bound and decision only, never
    on the other orders of the call, so one order filled alone and the same order
    filled among many give the same values to the last bit.
    """
    if rule not in RULES:
        raise ValueError(f"unknown fill rule {rule!r}; registered: {RULES}")
    arrival = np.atleast_1d(np.asarray(arrival_ns, dtype=np.int64))
    decision = np.broadcast_to(np.asarray(decision_ns, dtype=np.int64), arrival.shape)
    n = len(sw.ts)
    if n == 0:
        none = np.full(arrival.shape, -1, dtype=np.int64)
        nan = np.full(arrival.shape, np.nan)
        return PrintFills(np.zeros(arrival.shape, dtype=bool), none, none.copy(), nan, nan, nan)
    j = np.searchsorted(sw.ts, arrival, side="left")
    at = np.minimum(j, n - 1)
    found = (j < n) & (sw.ts[at] < np.asarray(until_ns, dtype=np.int64))
    g = sw.group[at]
    price = sw.rule_price[g]
    if rule == "optimistic":
        price = sw.price[at]
    elif rule == "pessimistic_depth":
        price = sw.pessimistic_price[g]
    elif rule in ("stale", "worse_of_two"):
        last = np.searchsorted(sw.ts, decision, side="right") - 1
        stale = np.where(last >= 0, sw.price[np.maximum(last, 0)], sw.rule_price[g])
        price = stale if rule == "stale" else worse(sw.side, sw.pessimistic_price[g], stale)
    return PrintFills(
        found,
        np.where(found, sw.position[at], -1),
        np.where(found, sw.ts[at], -1),
        np.where(found, price, np.nan),
        np.where(found, sw.beyond[g], np.nan),
        np.where(found, sw.sweep[g], np.nan),
    )
