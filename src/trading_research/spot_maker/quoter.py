"""MK-Q and MK-SQ: two-sided quoting at the touch proxies, requoted on a clock.

As registered under "The slots": at every requote instant both resting orders
are cancelled and reposted (each change taking effect after the latency), a
buy at the bid proxy and a sell at the ask proxy, skewed by the position known
at the instant (the fills stamped at or before it), post-only, the buy cut to
``limit - position`` and the sell to ``limit + position`` at their arrival; a side whose own proxy is undefined is not
quoted; orders rest until the next instant or 23:59, with no replenishment in
between; at 23:59 the position is flattened as a taker.

* **The instants** are the multiples of ``requote_s`` from midnight between
  00:01 and 23:58; for MK-SQ also every signal decision inside those hours and
  each such decision + hold inside them.
* **The signal's side** (MK-SQ) at an instant is that of the latest decision
  stamped at or before it and less than ``hold`` before it, or none; that
  side's order is two clips and the other's one, both cut by the limit.
* ``mode``: ``both`` (the slots), or K-drift's one-sided quoting for MK-Q,
  ``bid_only`` (all-long) and ``ask_only`` (all-short).
"""

from __future__ import annotations

import math

import numpy as np

from trading_research.spot.fills import PrintTape
from trading_research.spot_maker import resting
from trading_research.spot_maker.executor import DayRun, MakerSettings, close_position, known_after
from trading_research.spot_maker.registration import REMAINDER_BTC
from trading_research.spot_maker.touch import Proxies, quote_price

MODES: tuple[str, ...] = ("both", "bid_only", "ask_only")


def instants(tape: PrintTape, settings: MakerSettings, signal_ns: np.ndarray | None) -> np.ndarray:
    """The day's requote instants, in order, without repeats."""
    first, last = settings.first_ns(tape), settings.last_ns(tape)
    step = settings.requote_ns
    clock = np.arange(-(-(first - tape.start_ns) // step) * step, last - tape.start_ns + 1, step)
    found = [tape.start_ns + clock.astype(np.int64)]
    if signal_ns is not None and len(signal_ns):
        for extra in (signal_ns, signal_ns + settings.hold_ns):
            found.append(extra[(extra >= first) & (extra <= last)])
    return np.unique(np.concatenate(found)).astype(np.int64)


def signal_sides(at: np.ndarray, signal_ns: np.ndarray, signal_side: np.ndarray,
                 hold_ns: int) -> np.ndarray:  # fmt: skip
    """The live signal's side at each instant (0 for none)."""
    if not len(signal_ns):
        return np.zeros(len(at), dtype=np.int64)
    j = np.searchsorted(signal_ns, at, side="right") - 1
    live = (j >= 0) & (at < signal_ns[np.maximum(j, 0)] + hold_ns)
    return np.where(live, signal_side[np.maximum(j, 0)], 0).astype(np.int64)


def _sizes(mode: str, sigma: int, clip: float) -> tuple[float, float]:
    bid = 0.0 if mode == "ask_only" else clip * (2 if sigma > 0 else 1)
    ask = 0.0 if mode == "bid_only" else clip * (2 if sigma < 0 else 1)
    return bid, ask


def run_quoter(
    tape: PrintTape,
    proxies: Proxies,
    settings: MakerSettings,
    signal_ns: np.ndarray | None = None,
    signal_side: np.ndarray | None = None,
    mode: str = "both",
) -> DayRun:
    """One day of quoting; ``signal_ns``/``signal_side`` the decisions inside the hours."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    run = DayRun.start(settings)
    stamps = np.zeros(0, dtype=np.int64) if signal_ns is None else np.asarray(signal_ns, np.int64)
    sides = (
        np.zeros(0, dtype=np.int64) if signal_side is None else np.asarray(signal_side, np.int64)
    )
    at = instants(tape, settings, stamps if len(stamps) else None)
    sigma = signal_sides(at, stamps, sides, settings.hold_ns)
    flat, lat, limit = settings.flatten_ns(tape), settings.latency_ns, settings.limit
    ends = np.append(at[1:], flat).astype(np.int64)
    asks, bids = proxies.at(at)
    ranges = {side: resting.bounds(tape, side, at + lat, ends + lat) for side in (1, -1)}
    position, unknown = 0.0, 0.0
    run.counters["instants"] = len(at)
    for j in range(len(at)):
        ask, bid = float(asks[j]), float(bids[j])
        if math.isfinite(ask) and math.isfinite(bid):
            run.widths_bp.append((ask - bid) / ((ask + bid) / 2) * 1e4)
        want_bid, want_ask = _sizes(mode, int(sigma[j]), settings.clip)
        orders = ((1, min(want_bid, limit - position)), (-1, min(want_ask, limit + position)))
        known = position - unknown  # the previous orders' fills stamped after this instant
        moved, unknown = 0.0, 0.0
        for side, size in orders:
            if size <= REMAINDER_BTC:
                continue
            price = quote_price(side, ask, bid, known, skew_bp=settings.skew_bp,
                                clip=settings.clip, tick=settings.tick)  # fmt: skip
            if not math.isfinite(price):
                continue
            run.counters["quotes_bid" if side > 0 else "quotes_ask"] += 1
            run.posted_btc += size
            lo, hi = ranges[side]
            found = resting.fill_range(tape, side, price, size, int(lo[j]), int(hi[j]),
                                       settings.rule)  # fmt: skip
            run.note(tape, side, price, size, int(at[j]) + lat, int(ends[j]) + lat, found)
            for i, q in zip(*found, strict=True):
                run.log.add(
                    int(tape.ts[i]), side, price, float(q), True, "bid" if side > 0 else "ask"
                )
            moved += side * float(found[1].sum())
            unknown += side * known_after(tape, found, int(ends[j]))
        position += moved
    if abs(position) > REMAINDER_BTC:
        close_position(run, tape, position, settings)
    return run
