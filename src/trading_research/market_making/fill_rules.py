"""The fill-rule comparison: one passive order stream under three rules for when it fills.

The same quoter and the same signal tapes are simulated once per rule
(:class:`~trading_research.market_making.queue.FillRule`):

=================  ===========================================================  ===================
rule               a resting order fills                                        bias
=================  ===========================================================  ===================
``touch``          in full, at the first print at or through its price from     optimistic bound
                   the side that hits it, whatever is queued ahead
``trade_through``  only on prints strictly through its price, up to their size  pessimistic
``queue``          after the visible queue ahead is consumed by prints at its   the simulator's
                   price (R5, R6), or on prints through it (R7)                 default
=================  ===========================================================  ===================

Everything else (latency, limits, pauses, the exits) is the configuration's.
The streams are the same decisions, not the same orders: a quoter that holds
one position at a time posts again only when it is flat, so which later
decisions become orders depends on what filled earlier. That is the comparison
the question asks: what the same strategy would have been credited with.

Reported per rule, over the days not excluded from verdicts: the **fill ratio**
(the share of posted orders with any fill, the convention of ``fill_rate`` in
:mod:`trading_research.backtest.maker`), filled volume over posted volume,
gross and net with the filled notional they are a share of, and
volume-weighted markouts by fill path at each of the configuration's horizons.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from trading_research.market_making import analysis
from trading_research.market_making.events import DayEvents
from trading_research.market_making.queue import FillRule
from trading_research.market_making.quoters import Quoter
from trading_research.market_making.signals import SignalTape
from trading_research.market_making.simulator import DayResult, SimConfig, day_tapes, simulate_day

#: The rules, in the order of the comparison's table, with the bias of each.
RULES: tuple[FillRule, ...] = (FillRule.TOUCH, FillRule.THROUGH, FillRule.QUEUE)
BIAS: dict[FillRule, str] = {
    FillRule.TOUCH: "optimistic bound",
    FillRule.THROUGH: "pessimistic",
    FillRule.QUEUE: "simulator default",
}

#: The registered prediction: at least this under ``touch``, at most this under ``queue``.
TOUCH_AT_LEAST, QUEUE_AT_MOST = 0.75, 0.50


@dataclass(frozen=True)
class FillComparison:
    """The comparison's three tables."""

    #: One row per rule.
    summary: pd.DataFrame
    #: One row per rule and fill path: fills, volume and markouts.
    markouts: pd.DataFrame
    #: One row per rule and day, excluded days included and marked.
    days: pd.DataFrame

    def prediction(self) -> dict[str, float | bool]:
        """The registered prediction on the share of orders, and whether it held."""
        ratio = self.summary.set_index("rule")["fill_ratio"]
        touch, queue = float(ratio[FillRule.TOUCH.value]), float(ratio[FillRule.QUEUE.value])
        return {
            "touch_fill_ratio": touch,
            "queue_fill_ratio": queue,
            "holds": touch >= TOUCH_AT_LEAST and queue <= QUEUE_AT_MOST,
        }


def _day_row(rule: FillRule, result: DayResult) -> dict[str, object]:
    orders = result.orders
    decomposition = result.decomposition
    return {
        "rule": rule.value,
        "day": result.day.isoformat(),
        "excluded": result.excluded,
        "orders_posted": float(len(orders)),
        "orders_filled": float((orders["filled"] > 0).sum()),
        "volume_posted": float(orders["size"].sum()),
        "volume_filled": float(orders["filled"].sum()),
        "notional_filled": float((orders["filled"] * orders["price"]).sum()),
        "gross": float(decomposition["gross"]),
        "fees": float(decomposition["fees"]),
        "net": float(decomposition["net"]),
    }


def compare_fill_rules(
    days: Sequence[DayEvents],
    make_quoter: Callable[[], Quoter],
    config: SimConfig,
    tapes: Mapping[str, SignalTape] | None = None,
) -> FillComparison:
    """Simulate every day under each rule with a fresh quoter; the three tables."""
    day_rows: list[dict[str, object]] = []
    fills: list[pd.DataFrame] = []
    for rule in RULES:
        ruled = config.with_(fill_rule=rule)
        for events in days:
            result = simulate_day(events, make_quoter(), ruled, day_tapes(tapes, events.day))
            day_rows.append(_day_row(rule, result))
            if not result.excluded:
                maker = result.fills[result.fills["maker"]]
                fills.append(maker.assign(rule=rule.value))
    frame = pd.DataFrame(day_rows)
    usable = frame[~frame["excluded"].astype(bool)]
    totals = ["orders_posted", "orders_filled", "volume_posted", "volume_filled", "notional_filled"]
    totals += ["gross", "fees", "net"]
    summary = usable.groupby("rule", sort=False)[totals].sum()
    summary = summary.reindex([r.value for r in RULES]).fillna(0.0)
    summary.insert(0, "days", usable.groupby("rule").size().reindex(summary.index).fillna(0))
    summary["fill_ratio"] = summary["orders_filled"] / summary["orders_posted"]
    summary["volume_ratio"] = summary["volume_filled"] / summary["volume_posted"]
    summary["bias"] = [BIAS[FillRule(r)] for r in summary.index]
    paths = []
    horizons = config.markout_horizons_s
    for rule in RULES:
        mine = [f for f in fills if len(f) and f["rule"].iloc[0] == rule.value]
        if mine:
            table = analysis.summarise_markouts(pd.concat(mine), horizons)
            paths.append(table.assign(rule=rule.value))
    markouts = pd.concat(paths, ignore_index=True) if paths else pd.DataFrame()
    return FillComparison(summary.reset_index(), markouts, frame)
