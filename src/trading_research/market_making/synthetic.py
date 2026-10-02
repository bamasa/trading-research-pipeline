"""Markets with a known answer, and random ones, for testing the simulator.

The repository's other tests inject a signal of known strength and check the
pipeline finds it, and switch it off and check it finds nothing. These are the
market-making counterpart.

:func:`known_answer_market` is a constant spread with prints that each consume
the whole touch, so a quoter at the touch is filled by every print on its side.
After each print the mid moves ``move_ticks`` in the print's direction:

* ``move_ticks = 0`` is uninformed flow. The mid never moves, so a touch quoter
  earns exactly the half-spread on every unit it is filled, and nothing else;
* ``move_ticks = spread / 2`` is fully informed flow in the sense of Glosten and
  Milgrom: the mid moves to the price just traded, so the half-spread earned is
  given back at once and spread plus adverse selection is exactly zero;
* ``move_ticks = spread`` is worse than informed: every fill loses, and its
  markout has the sign of the move.

Prints are at least ``min_gap_s`` apart, longer than the five-second
decomposition horizon, so each fill's markout sees exactly one move.

:func:`random_market` is a random walk of the book with random level sizes,
gaps between levels, and prints at, through and away from the touch on both
sides, for the invariant tests.

Nothing here reads data.
"""

from __future__ import annotations

from datetime import date

import numpy as np

from trading_research.market_making.events import (
    NS_PER_S,
    DayEvents,
    InstrumentSpec,
    assemble,
    day_start_ns,
)

#: The day synthetic markets are stamped with. Arbitrary, and nowhere near the
#: blocks of the study.
SYNTHETIC_DAY = date(2020, 1, 6)


def known_answer_market(
    *,
    move_ticks: int,
    spread_ticks: int = 4,
    touch_size: float = 10.0,
    print_excess: float = 1.0,
    n_prints: int = 300,
    min_gap_s: float = 6.0,
    mean_extra_gap_s: float = 4.0,
    snapshot_s: float = 1.0,
    first_print_s: float = 30.0,
    depth: int = 10,
    tick: float = 0.01,
    start_ticks: int = 10_000,
    seed: int = 0,
    day: date = SYNTHETIC_DAY,
) -> DayEvents:
    """A constant-spread market whose prints each sweep the touch exactly.

    Every print is ``touch_size + print_excess``: it consumes the visible touch
    and then ``print_excess`` more, which is what fills an own order joined at
    the tail of that touch.
    """
    if spread_ticks < 1 or depth < 1:
        raise ValueError("spread and depth must be at least one tick and one level")
    rng = np.random.default_rng(seed)
    start = day_start_ns(day)
    gaps = min_gap_s + rng.exponential(mean_extra_gap_s, n_prints)
    times_s = first_print_s + np.concatenate([[0.0], np.cumsum(gaps[:-1])])
    # Millisecond prints, half a millisecond off any whole-second snapshot.
    print_ns = start + (np.round(times_s * 1000.0).astype(np.int64) * 1_000_000) + 500_000
    aggressor = rng.choice(np.array([-1, 1], dtype=np.int8), size=n_prints)

    bid_after = np.empty(n_prints, dtype=np.int64)
    trade_px = np.empty(n_prints, dtype=np.int64)
    bid = start_ticks
    for i in range(n_prints):
        trade_px[i] = bid if aggressor[i] < 0 else bid + spread_ticks
        bid += int(aggressor[i]) * move_ticks
        bid_after[i] = bid

    duration_s = float(times_s[-1]) + 60.0
    snapshot_ns = start + np.round(np.arange(0.0, duration_s, snapshot_s) * NS_PER_S).astype(
        np.int64
    )
    applied = np.searchsorted(print_ns, snapshot_ns, side="right") - 1
    best_bid = np.where(applied >= 0, bid_after[np.maximum(applied, 0)], start_ticks)
    offsets = np.arange(depth, dtype=np.int64)
    bid_px = best_bid[:, None] - offsets[None, :]
    ask_px = best_bid[:, None] + spread_ticks + offsets[None, :]
    sizes = np.full(bid_px.shape, touch_size)

    spec = InstrumentSpec("SYNTHUSDT", tick, 1.0)
    return assemble(
        spec,
        day,
        book_ts=snapshot_ns,
        bid_px=bid_px,
        bid_sz=sizes,
        ask_px=ask_px,
        ask_sz=sizes.copy(),
        trade_ts=print_ns,
        trade_px=trade_px,
        trade_sz=np.full(n_prints, touch_size + print_excess),
        trade_aggressor=aggressor,
    )


def random_market(
    seed: int,
    *,
    n_snapshots: int = 600,
    snapshot_ms: int = 100,
    prints_per_snapshot: float = 0.6,
    depth: int = 10,
    tick: float = 0.01,
    start_ticks: int = 10_000,
    lot: float = 1.0,
    max_step: int = 1,
    through_share: float = 0.2,
    away_share: float = 0.2,
    funding: bool = False,
    day: date = SYNTHETIC_DAY,
) -> DayEvents:
    """A random book and tape: the stream the invariant tests fuzz over.

    The touch walks by up to ``max_step`` ticks a snapshot (zero holds it
    still), the spread is one to three ticks, levels are one or two ticks
    apart with random sizes, and prints land at the touch, through it
    (``through_share``), or short of it (``away_share``), from either side,
    several sometimes sharing a timestamp.
    """
    rng = np.random.default_rng(seed)
    start = day_start_ns(day)
    snapshot_ns = (
        start + NS_PER_S + np.arange(n_snapshots, dtype=np.int64) * (snapshot_ms * 1_000_000)
    )
    best_bid = start_ticks + np.cumsum(rng.integers(-max_step, max_step + 1, n_snapshots))
    spread = rng.integers(1, 4, n_snapshots) if max_step else np.full(n_snapshots, 2)
    steps_bid = rng.integers(1, 3, (n_snapshots, depth))
    steps_ask = rng.integers(1, 3, (n_snapshots, depth))
    steps_bid[:, 0] = 0
    steps_ask[:, 0] = 0
    bid_px = best_bid[:, None] - np.cumsum(steps_bid, axis=1)
    ask_px = (best_bid + spread)[:, None] + np.cumsum(steps_ask, axis=1)
    bid_sz = rng.integers(1, 15, (n_snapshots, depth)).astype(np.float64) * lot
    ask_sz = rng.integers(1, 15, (n_snapshots, depth)).astype(np.float64) * lot

    counts = rng.poisson(prints_per_snapshot, n_snapshots)
    trade_ts: list[int] = []
    trade_px: list[int] = []
    trade_sz: list[float] = []
    trade_aggressor: list[int] = []
    for row in range(n_snapshots):
        for _ in range(int(counts[row])):
            offset = int(rng.integers(0, snapshot_ms)) * 1_000_000
            if rng.random() < 0.15:
                offset = 0  # a tie with the snapshot itself
            aggressor = int(rng.choice([-1, 1]))
            touch = int(bid_px[row, 0]) if aggressor < 0 else int(ask_px[row, 0])
            where = rng.random()
            if where < through_share:
                price = touch + aggressor * int(rng.integers(1, 3))
            elif where < through_share + away_share:
                price = touch - aggressor * int(rng.integers(1, 3))
            else:
                price = touch
            trade_ts.append(int(snapshot_ns[row]) + offset)
            trade_px.append(price)
            trade_sz.append(float(rng.integers(1, 12)) * lot)
            trade_aggressor.append(aggressor)
            if through_share and rng.random() < 0.2:  # a second print in the same sweep
                trade_ts.append(trade_ts[-1])
                trade_px.append(price + aggressor)
                trade_sz.append(float(rng.integers(1, 6)) * lot)
                trade_aggressor.append(aggressor)
    if not trade_ts:
        trade_ts, trade_px, trade_sz, trade_aggressor = (
            [int(snapshot_ns[-1])],
            [int(bid_px[-1, 0])],
            [lot],
            [-1],
        )

    funding_ts = funding_rate = None
    if funding:
        funding_ts = np.array([start, start + 8 * 3600 * NS_PER_S], dtype=np.int64)
        funding_rate = np.array([1e-4, -2e-4])

    return assemble(
        InstrumentSpec("FUZZUSDT", tick, lot),
        day,
        book_ts=snapshot_ns,
        bid_px=bid_px,
        bid_sz=bid_sz,
        ask_px=ask_px,
        ask_sz=ask_sz,
        trade_ts=np.array(trade_ts, dtype=np.int64),
        trade_px=np.array(trade_px, dtype=np.int64),
        trade_sz=np.array(trade_sz),
        trade_aggressor=np.array(trade_aggressor, dtype=np.int8),
        funding_ts=funding_ts,
        funding_rate=funding_rate,
    )


#: One snapshot for :func:`hand_built_market`: milliseconds after midnight, then
#: the bid levels and the ask levels as (price in ticks, size), best first.
Snapshot = tuple[float, list[tuple[int, float]], list[tuple[int, float]]]

#: One print: milliseconds after midnight, price in ticks, size, aggressor.
Print = tuple[float, int, float, int]


def hand_built_market(
    snapshots: list[Snapshot],
    prints: list[Print],
    *,
    funding: list[tuple[float, float]] | None = None,
    tick: float = 0.01,
    lot: float = 1.0,
    day: date = SYNTHETIC_DAY,
) -> DayEvents:
    """A stream written out by hand, for tests that need one exact situation.

    Every snapshot must list the same number of levels a side. ``funding`` is
    a list of (milliseconds after midnight, rate).
    """
    if not snapshots:
        raise ValueError("a market needs at least one snapshot")
    depth = len(snapshots[0][1])
    if any(len(b) != depth or len(a) != depth for _, b, a in snapshots):
        raise ValueError("every snapshot must list the same number of levels a side")
    start = day_start_ns(day)

    def ns(milliseconds: float) -> int:
        return start + round(milliseconds * 1_000_000)

    return assemble(
        InstrumentSpec("HANDUSDT", tick, lot),
        day,
        book_ts=np.array([ns(t) for t, _, _ in snapshots], dtype=np.int64),
        bid_px=np.array([[p for p, _ in b] for _, b, _ in snapshots], dtype=np.int64),
        bid_sz=np.array([[s for _, s in b] for _, b, _ in snapshots], dtype=np.float64),
        ask_px=np.array([[p for p, _ in a] for _, _, a in snapshots], dtype=np.int64),
        ask_sz=np.array([[s for _, s in a] for _, _, a in snapshots], dtype=np.float64),
        trade_ts=np.array([ns(t) for t, _, _, _ in prints], dtype=np.int64),
        trade_px=np.array([p for _, p, _, _ in prints], dtype=np.int64),
        trade_sz=np.array([s for _, _, s, _ in prints], dtype=np.float64),
        trade_aggressor=np.array([a for _, _, _, a in prints], dtype=np.int8),
        funding_ts=None
        if funding is None
        else np.array([ns(t) for t, _ in funding], dtype=np.int64),
        funding_rate=None if funding is None else np.array([r for _, r in funding]),
    )
