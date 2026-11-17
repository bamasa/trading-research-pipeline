"""The reversion rule with the kill switch inside it: it stops itself when its own results say the edge is gone.

:class:`~trading_research.plugins.builtins.strategies.Reversion` decides every
row from the index's last ten minutes; this strategy takes those decisions and
keeps a paper account of them as it goes. Each paper trade enters where the
rule would, holds the rule's ten minutes, pays the touch's half-spread on both
legs, Bybit's base taker fee and 0.5 bp of slippage a side, and is counted in
the hour it closes, when its result is known. The hourly series runs through
:func:`trading_research.validation.kill_switch.kill_switch` at its textbook
values; from the hour after the switch fires, the strategy opens nothing.

It lives in its own module so the frozen rule's source, which the benchmark's
forward track pins by its sha256, is not touched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_research.plugins.builtins.strategies import Reversion
from trading_research.plugins.registry import register_strategy

TAKER_BP, SLIPPAGE_BP = 5.5, 0.5


def paper_hourly(
    decisions: np.ndarray, frame: pd.DataFrame, hold: int, cooldown: int
) -> tuple[pd.Series, pd.Series]:
    """The paper results of the decisions, by the hour each trade closed and the hour it opened."""
    mid = frame["mid"].to_numpy(dtype=np.float64)
    spread = frame["spread_bp"].to_numpy(dtype=np.float64)
    stamps = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True))
    entries, exits, results = [], [], []
    i, n = 0, len(decisions)
    while i < n - hold:
        if decisions[i] == 0:
            i += 1
            continue
        j = i + hold
        move = np.sign(decisions[i]) * (mid[j] / mid[i] - 1.0) * 1e4
        cost = (spread[i] + spread[j]) / 2 + 2 * (TAKER_BP + SLIPPAGE_BP)
        entries.append(stamps[i])
        exits.append(stamps[j])
        results.append(move - cost)
        i = j + cooldown + 1
    hours = pd.date_range(stamps[0].ceil("1h"), stamps[-1].ceil("1h"), freq="1h")
    pnl = pd.Series(results, dtype=np.float64)

    def by_hour(when: list[pd.Timestamp]) -> pd.Series:
        if not when:
            return pd.Series(0.0, index=hours)
        series = pd.Series(pnl.to_numpy(), index=pd.DatetimeIndex(when))
        return (
            series.resample("1h", label="right", closed="right")
            .sum()
            .reindex(hours, fill_value=0.0)
        )

    return by_hour(exits), by_hour(entries)


@register_strategy("reversion_guarded")
class ReversionGuarded(Reversion):
    """The frozen reversion rule, stopped by the kill switch on its own paper results."""

    live_from: pd.Timestamp | None = None

    def fit(self, history: pd.DataFrame) -> ReversionGuarded:
        super().fit(history)
        # the switch watches the strategy from the first row after its fit, never the history
        self.live_from = pd.Timestamp(pd.to_datetime(history["timestamp"], utc=True).iloc[-1])
        return self

    def decide(self, frame: pd.DataFrame) -> np.ndarray:
        from trading_research.validation.kill_switch import kill_switch

        raw = np.asarray(super().decide(frame), dtype=np.int64).copy()
        stamps = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True))
        live = stamps > self.live_from if self.live_from is not None else np.ones(len(raw), bool)
        frame, watched = frame[live], np.where(live, raw, 0)[live]
        if not live.any():
            return raw
        closed, opened = paper_hourly(watched, frame, int(self.horizon), int(self.cooldown))
        if len(closed) <= 72:
            return raw
        switch = kill_switch(closed, warm_up=72, booked=opened)
        stop = switch.first_stop
        self.stopped_at = None if stop is None else closed.index[stop.position]
        if self.stopped_at is not None:
            raw[stamps > self.stopped_at] = 0
        return raw
