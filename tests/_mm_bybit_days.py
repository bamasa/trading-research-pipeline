"""Hand-built print days for the Bybit 2021 market-making tests.

A day is a list of prints ``(seconds after midnight, price, size, aggressor)``
with the three funding settlements on the 8-hour grid. Prices sit on a 0.5 USDT
grid, so a tick is 0.5 and 100.0 USDT is 200 ticks.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import numpy as np

from trading_research.market_making.events import NS_PER_S, day_start_ns
from trading_research.mm_bybit import engine, synthetic
from trading_research.mm_bybit.tape import DayTape, build

DAY = date(2021, 1, 4)  # a day of D
HOUR = 3600.0
#: The two prints that set both proxies before quoting starts: ask 100.5, bid 100.0.
OPENING: tuple[tuple[float, float, float, int], ...] = ((1.0, 100.5, 0.1, 1), (2.0, 100.0, 0.1, -1))
#: Prints after 23:59 for the flatten to fill at, either side.
CLOSING: tuple[tuple[float, float, float, int], ...] = (
    (23 * HOUR + 59 * 60 + 1.0, 100.5, 1.0, 1),
    (23 * HOUR + 59 * 60 + 1.0, 100.0, 1.0, -1),
)
INPUTS = engine.QuoterInputs(sigma_ref=0.0, quantiles={0.33: 1.0, 0.67: 2.0, 0.9: 3.0})


def tape_of(
    prints: Sequence[tuple[float, float, float, int]],
    *,
    day: date = DAY,
    rates: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> DayTape:
    """A day's tape from prints given in time order (ties in the order given)."""
    start = day_start_ns(day)
    seconds = np.array([p[0] for p in prints])
    stamps = start + np.round(seconds * NS_PER_S).astype(np.int64)
    frame = synthetic.frame(
        day,
        stamps,
        np.array([p[1] for p in prints]),
        np.array([p[2] for p in prints]),
        np.array([p[3] for p in prints]),
    )
    settlements = synthetic.funding(day, rates)
    funding_ts = start + np.arange(3, dtype=np.int64) * 8 * 3600 * NS_PER_S
    return build(day, frame, funding_ts, settlements["rate"].to_numpy(dtype=float))


def settings(**changes: object) -> engine.RunSettings:
    """The primary settings at the 2021 fees, with ``changes`` applied."""
    base: dict[str, object] = {"maker_bp": -2.5, "taker_bp": 7.5}
    base.update(changes)
    return engine.RunSettings(**base)  # type: ignore[arg-type]
