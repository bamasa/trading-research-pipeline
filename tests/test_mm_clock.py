"""The book's clock against the prints' clock.

A book row carries the time the venue generated it, a print the time it
matched. If the book lags, a print just before a snapshot is not in it yet, and
a queue model that forgot the print at that snapshot would count it again, one
snapshot later, as a cancellation. If the book leads, a snapshot shows prints
stamped after it, which is look-ahead. The diagnostic below measures the
offset; the queue model carries unreflected prints forward.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from trading_research.market_making.analysis import best_clock_offset_ms, clock_offset_profile
from trading_research.market_making.events import (
    DayEvents,
    InstrumentSpec,
    assemble,
    day_start_ns,
)
from trading_research.market_making.queue import CancelAttribution
from trading_research.market_making.quoters import MarketView, Quote, Quotes
from trading_research.market_making.simulator import SimConfig, simulate_day
from trading_research.market_making.synthetic import SYNTHETIC_DAY, hand_built_market


def _lagged_market(lag_ms: int, seed: int = 0, n: int = 3000, level0: float = 60.0) -> DayEvents:
    """Touch prices fixed at 100 / 102; the touch sizes move by additions,
    cancellations and prints, and the snapshot at ``ts`` shows the state as of
    ``ts - lag`` (a negative lag: the snapshot leads the prints)."""
    rng = np.random.default_rng(seed)
    start = day_start_ns(SYNTHETIC_DAY) + 10**9
    book_ts = start + np.arange(n, dtype=np.int64) * 100_000_000
    count = n * 6
    times = np.sort(start + rng.integers(0, n * 100, count) * 1_000_000 + 500_000)
    kind = rng.choice([0, 1, 2], size=count, p=[0.35, 0.3, 0.35])  # add, cancel, print
    side = rng.choice([1, -1], size=count)
    size = rng.integers(1, 6, count).astype(float)
    level = {1: level0, -1: level0}
    states: list[tuple[int, float, float]] = []
    trades: list[tuple[int, int, float, int]] = []
    for i in range(count):
        s = int(side[i])
        if kind[i] == 0:
            level[s] += size[i]
        elif kind[i] == 1:
            level[s] = max(5.0, level[s] - size[i])
        else:
            q = min(size[i], level[s] - 1.0)
            if q <= 0:
                continue
            level[s] -= q
            trades.append((int(times[i]), 100 if s == 1 else 102, q, -s))
        states.append((int(times[i]), level[1], level[-1]))
    state_ts = np.array([x[0] for x in states])
    bid_level = np.array([x[1] for x in states])
    ask_level = np.array([x[2] for x in states])
    seen = np.searchsorted(state_ts, book_ts - lag_ms * 1_000_000, side="right") - 1
    depth = 3
    bid_sz = np.full((n, depth), 50.0)
    ask_sz = np.full((n, depth), 50.0)
    bid_sz[:, 0] = np.where(seen >= 0, bid_level[np.maximum(seen, 0)], level0)
    ask_sz[:, 0] = np.where(seen >= 0, ask_level[np.maximum(seen, 0)], level0)
    prints = np.array(trades)
    return assemble(
        InstrumentSpec("LAGUSDT", 0.01, 1.0),
        SYNTHETIC_DAY,
        book_ts=book_ts,
        bid_px=np.tile(100 - np.arange(depth), (n, 1)),
        bid_sz=bid_sz,
        ask_px=np.tile(102 + np.arange(depth), (n, 1)),
        ask_sz=ask_sz,
        trade_ts=prints[:, 0].astype(np.int64),
        trade_px=prints[:, 1].astype(np.int64),
        trade_sz=prints[:, 2],
        trade_aggressor=prints[:, 3].astype(np.int8),
    )


@pytest.mark.parametrize("lag", [0, 20, -20])
def test_the_offset_diagnostic_recovers_a_known_lag(lag: int) -> None:
    """A book lagging by 20 ms reads +20, one leading by 20 ms reads -20."""
    profile = clock_offset_profile(_lagged_market(lag))
    assert abs(best_clock_offset_ms(profile) - lag) <= 5


@dataclass(frozen=True)
class Fixed:
    bid: int | None
    ask: int | None = None
    name: str = "fixed"
    uses_future: bool = False

    def quotes(self, view: MarketView, position: float) -> Quotes:
        return Quotes(Quote(self.bid, view.clip), Quote(self.ask, view.clip))


def _book(best_bid: int, best_ask: int, bid_top: float = 10.0):
    bids = [(best_bid - i, 10.0) for i in range(3)]
    bids[0] = (best_bid, bid_top)
    return bids, [(best_ask + i, 10.0) for i in range(3)]


def _lag_market(lagged: bool) -> DayEvents:
    """Ten at the bid; our bid joins behind them at 10 ms. A sale of 4 at 150 ms.
    The consistent book shows 6 from 200 ms; the lagged one still shows 10 at
    200 ms and 6 from 300 ms. A sale of 5 at 350 ms leaves one ahead of us:
    nothing can fill."""
    full, after = _book(100, 102), _book(100, 102, bid_top=6.0)
    snapshots = [
        (0.0, *full),
        (100.0, *full),
        (200.0, *(full if lagged else after)),
        (300.0, *after),
        (400.0, *after),
    ]
    return hand_built_market(snapshots, [(150.0, 100, 4.0, -1), (350.0, 100, 5.0, -1)])


@pytest.mark.parametrize("rule", list(CancelAttribution))
def test_a_lagged_snapshot_does_not_count_a_print_twice(rule: CancelAttribution) -> None:
    config = SimConfig(
        clip_notional=5.0, clip_touch_share=1e9, warmup_s=0.0, cancel_attribution=rule
    )
    for lagged in (False, True):
        fills = simulate_day(_lag_market(lagged), Fixed(bid=100), config).fills
        assert fills.loc[fills["maker"], "size"].sum() == 0.0, (rule, lagged)
