"""A kill switch on a strategy's own results: stop when its mean return has dropped.

A regime change that breaks a strategy is often not forecastable (§27 of
``docs/results.md`` found the reversion rule's market state measurable but
not predictable a day ahead). What can be done is to notice, from the
strategy's own profit and loss and nothing later, that the edge it was
earning is gone, and stop trading. This module does that with Page's CUSUM,
the sequential test for a drop in a mean:

    S_t = max(0, S_{t-1} - z_t - k),   alarm when S_t > h,

where ``z_t`` is the period's return divided by the standard deviation of
the returns before it. ``k`` = 0.5 and ``h`` = 5 are the textbook values for
a drop of one standard deviation (about 930 periods between false alarms on
Gaussian noise). The first ``warm_up`` periods only estimate the deviation.
Measured here on Gaussian returns (``tests/test_kill_switch.py``): with an
edge of +0.2 standard deviations, about one run in ten of 528 monitored
periods raises a false alarm; after the mean falls to -1, the median alarm
comes 8 periods later. A more sensitive switch (smaller ``k`` or ``h``) stops
sooner and more often by mistake.

Two policies after an alarm:

- ``restart=False``: stop and stay stopped; a person decides what happens next.
- ``restart=True``: stop but keep trading on paper, and go live again when the
  same test pointed upwards sees the paper result recover. In the one case the
  repository has measured (``posts/btc_ath_2024``) this gave most of what the
  stop had kept back on two bounces, which is why it is not the default.

The switch is causal: the decision at a period reads that period and the ones
before it, never a later one (``tests/test_kill_switch.py`` mutates the future
and checks every earlier decision stays as it was).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: Page's reference value and threshold for a one-standard-deviation drop.
TEXTBOOK_K = 0.5
TEXTBOOK_H = 5.0


@dataclass(frozen=True)
class Event:
    """A stop or a restart, at the period whose return triggered it."""

    kind: str  # "stop" or "restart"
    at: object
    position: int
    equity: float


@dataclass
class KillSwitchResult:
    """The equity the strategy keeps under the switch, and every decision."""

    equity: pd.Series
    live: pd.Series
    events: list[Event] = field(default_factory=list)

    @property
    def first_stop(self) -> Event | None:
        return next((e for e in self.events if e.kind == "stop"), None)


def kill_switch(
    returns: pd.Series,
    *,
    k: float = TEXTBOOK_K,
    h: float = TEXTBOOK_H,
    warm_up: int = 72,
    restart: bool = False,
    booked: pd.Series | None = None,
) -> KillSwitchResult:
    """Run the switch over a strategy's per-period returns (any unit; they are summed).

    ``returns`` is what the strategy's trades realised in each period, every
    trade counted in the period it closed: what was known at the period's end,
    and all the test reads. A decision taken on period ``i`` applies from
    period ``i + 1``.

    ``booked``, on the same periods, is the result of the trades *opened* in
    each period, whenever they close. The returned ``equity`` sums it over the
    periods the strategy was live: a stop forbids new entries, and the
    positions already open run to their exits and count. Without ``booked``,
    ``returns`` stands in for it, which is exact only for trades that open and
    close within one period.
    """
    if warm_up < 2:
        raise ValueError("warm_up must leave at least two periods to estimate a deviation")
    if k < 0 or h <= 0:
        raise ValueError(f"k must be non-negative and h positive, got k={k}, h={h}")
    x = returns.to_numpy(dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("returns must be finite; mark a period without trading as 0")
    credit = x if booked is None else booked.to_numpy(dtype=np.float64)
    if len(credit) != len(x) or not np.isfinite(credit).all():
        raise ValueError("booked must be finite and as long as returns")
    live, down, up, total = True, 0.0, 0.0, 0.0
    equity, is_live = np.empty(len(x)), np.empty(len(x), dtype=bool)
    events: list[Event] = []
    for i, value in enumerate(x):
        is_live[i] = live
        if live:
            total += credit[i]
        equity[i] = total
        if i < warm_up:
            continue
        sd = float(np.std(x[:i], ddof=1))
        z = value / sd if sd > 0 else 0.0
        if live:
            down = max(0.0, down - z - k)
            if down > h:
                live, down, up = False, 0.0, 0.0
                events.append(Event("stop", returns.index[i], i, total))
        elif restart:
            up = max(0.0, up + z - k)
            if up > h:
                live, down, up = True, 0.0, 0.0
                events.append(Event("restart", returns.index[i], i, total))
    return KillSwitchResult(
        pd.Series(equity, index=returns.index), pd.Series(is_live, index=returns.index), events
    )


__all__ = ["TEXTBOOK_H", "TEXTBOOK_K", "Event", "KillSwitchResult", "kill_switch"]
