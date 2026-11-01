"""MK-S: S25-PF's decisions as resting orders at the touch proxies, one position at a time.

As registered under "The slots", MK-S, step by step:

1. **Entry.** At a decision row ``t`` of side ``s``, flat and with no resting
   order, inside the posting hours: one clip on side ``s`` at its own proxy
   (a buy at the bid proxy), post-only, live from ``t`` + latency; its
   unfilled part is cancelled at the earliest of ``t + hold``, the first later
   decision of the other side and 23:59 (each + latency). An entry whose own
   proxy is undefined is not posted (``entries_no_proxy``).
2. **No fill.** The next decision may be taken at a row stamped at or after the
   cancel's decision, so an opposite decision that cancelled it is a new entry.
3. **Exit.** After a fill, decided at the earliest of the first grid row
   stamped at or after the entry's first fill + hold, the first decision of
   the other side after ``t`` and 23:59: a resting order of the other side for
   the position at its own proxy, skewed by the position known at its
   decision, post-only, sized at its arrival; requoted every ``requote_s``
   after its decision at the proxy then (an undefined proxy keeps the last
   price); flattened at 23:59 as a taker if anything remains
   (``exits_flattened``).
4. **One position.** After the exit, the next decision may be taken at a row
   stamped at or after the position's last fill. Other decisions meanwhile are
   skipped (``skipped_busy``), except the opposite one that ends an entry or
   starts an exit.
"""

from __future__ import annotations

import math

import numpy as np

from trading_research.spot.fills import PrintTape
from trading_research.spot_maker import resting
from trading_research.spot_maker.executor import DayRun, MakerSettings, close_position, known_after
from trading_research.spot_maker.registration import REMAINDER_BTC
from trading_research.spot_maker.touch import Proxies, quote_price


def next_opposite(side: np.ndarray) -> np.ndarray:
    """For each decision, the index of the first later decision of the other side, or -1."""
    out = np.full(len(side), -1, dtype=np.int64)
    for s in (1, -1):
        mine, theirs = np.flatnonzero(side == s), np.flatnonzero(side == -s)
        j = np.searchsorted(theirs, mine, side="right")
        ok = j < len(theirs)
        out[mine[ok]] = theirs[j[ok]]
    return out


def _record(run: DayRun, tape: PrintTape, side: int, price: float,
            found: tuple[np.ndarray, np.ndarray], kind: str) -> float:  # fmt: skip
    idx, sizes = found
    for i, q in zip(idx, sizes, strict=True):
        run.log.add(int(tape.ts[i]), side, price, float(q), True, kind)
    return float(sizes.sum())


def _exit(run: DayRun, tape: PrintTape, proxies: Proxies, side: int, size: float, decided: int,
          settings: MakerSettings, entry_unknown: float = 0.0) -> int:  # fmt: skip
    """The exit of a position of ``size`` held on ``side``, decided at ``decided``; the
    time of its last fill. ``entry_unknown``: the part of ``size`` filled after
    ``decided`` (inside the entry's cancel latency), not yet known when the exit is
    priced; each order is sized at its arrival and skewed by the position known at
    its decision."""
    flat, lat = settings.flatten_ns(tape), settings.latency_ns
    remaining, tau, price, last = size, decided, math.nan, decided
    known = size - entry_unknown  # the position's size known at the decision
    while remaining > REMAINDER_BTC and tau < flat:
        ask, bid = proxies.one(tau)
        quoted = quote_price(-side, ask, bid, side * known, skew_bp=settings.skew_bp,
                             clip=settings.clip, tick=settings.tick)  # fmt: skip
        price = quoted if math.isfinite(quoted) else price
        end = min(tau + settings.requote_ns, flat)
        if math.isfinite(price):
            run.counters["exit_quotes"] += 1
            run.posted_btc += remaining
            found = resting.fill(tape, -side, price, remaining, tau + lat, end + lat, settings.rule)
            run.note(tape, -side, price, remaining, tau + lat, end + lat, found)
            remaining -= _record(run, tape, -side, price, found, "exit")
            if len(found[0]):
                last = int(tape.ts[found[0][-1]])
            known = remaining + known_after(tape, found, end)
        else:
            known = remaining
        tau = end
    if remaining > REMAINDER_BTC:
        run.counters["exits_flattened"] += 1
        last = close_position(run, tape, side * remaining, settings)
    return last


def run_signal_maker(
    tape: PrintTape,
    proxies: Proxies,
    grid_ns: np.ndarray,
    decision_ns: np.ndarray,
    decision_side: np.ndarray,
    settings: MakerSettings,
) -> DayRun:
    """One day of MK-S on the decisions given (the true ones, or a placebo's sides)."""
    run = DayRun.start(settings)
    sides = np.asarray(decision_side, dtype=np.int64)
    stamps = np.asarray(decision_ns, dtype=np.int64)
    opposite = next_opposite(sides)
    first, last_post, flat = settings.first_ns(tape), settings.last_ns(tape), settings.flatten_ns(tape)
    lat, busy = settings.latency_ns, -1
    run.counters["decisions"] = len(stamps)
    for k in range(len(stamps)):
        t, s = int(stamps[k]), int(sides[k])
        if t < busy:
            run.counters["skipped_busy"] += 1
            continue
        if t < first or t > last_post:
            run.counters["skipped_hours"] += 1
            continue
        ask, bid = proxies.one(t)
        price = quote_price(s, ask, bid, 0.0, skew_bp=settings.skew_bp, clip=settings.clip,
                            tick=settings.tick)  # fmt: skip
        if not math.isfinite(price):
            run.counters["entries_no_proxy"] += 1
            continue
        run.counters["entries_posted"] += 1
        run.posted_btc += settings.clip
        flip = int(stamps[opposite[k]]) if opposite[k] >= 0 else math.inf
        cancel = int(min(t + settings.hold_ns, flip, flat))
        found = resting.fill(tape, s, price, settings.clip, t + lat, cancel + lat, settings.rule)
        run.note(tape, s, price, settings.clip, t + lat, cancel + lat, found)
        if not len(found[0]):
            run.counters["entries_unfilled"] += 1
            busy = cancel
            continue
        size = _record(run, tape, s, price, found, "entry")
        first_fill = int(tape.ts[found[0][0]])
        run.counters["entries_filled"] += 1
        run.counters["entries_partial"] += int(size < settings.clip - REMAINDER_BTC)
        if cancel == flip:
            decided = cancel
        else:
            row = int(np.searchsorted(grid_ns, first_fill + settings.hold_ns, side="left"))
            after = int(grid_ns[row]) if row < len(grid_ns) else math.inf
            decided = int(min(after, flip, flat))
        if decided >= flat:
            done = close_position(run, tape, s * size, settings)
            run.counters["exits_flattened"] += 1
        else:
            done = _exit(run, tape, proxies, s, size, decided, settings,
                         known_after(tape, found, decided))  # fmt: skip
        run.holds_s.append((done - first_fill) / 1e9)
        busy = done
    return run
