"""The facts a strategy review needs, gathered in one place.

Half the rubric in ``docs/evaluation/rubric.md`` is a matter of judgement and
half is arithmetic. This computes the arithmetic half so a reviewer — a person
or an agent — spends its attention on the judgement rather than on deriving
numbers from a trade log.

It deliberately does not score anything. A module that both produced the
evidence and graded it would be marking its own work, which is the failure the
rubric exists to catch. What comes out is a set of measurements and the
questions they leave open; the verdict belongs to whoever reads it.

Three things it insists on
--------------------------
**Net, never gross alone.** Every per-trade figure is reported after costs, and
gross is shown beside it. A number that does not say which it is gets read as
profit, which is how a losing strategy gets published.

**Configurations tried, not configurations shown.** A search over forty
candidates reported as one result is forty chances to fit the validation block.
If the caller does not supply the count, the scorecard says it is unknown rather
than omitting the line.

**The path, not only the average.** Drawdown, its length, and the share of
periods that lost, because a strategy is run by someone who sees the path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_research.backtest.metrics import summarise


@dataclass
class Scorecard:
    """Measured facts about one strategy, and what is not known about it."""

    name: str
    metrics: dict[str, float]
    per_period: pd.DataFrame
    context: dict[str, Any] = field(default_factory=dict)
    unknowns: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        """A document a reviewer can read without opening the repository."""
        lines = [f"# Scorecard: {self.name}", ""]

        lines += ["## Result", "", "| Measure | Value |", "|---|---:|"]
        for key in (
            "trades",
            "trades_per_day",
            "gross_per_trade_bp",
            "net_per_trade_bp",
            "net_bp",
            "net_bp_per_day",
            "hit_rate",
            "profit_factor",
            "max_drawdown_bp",
            "drawdown_trades",
            "return_over_drawdown",
            "days",
        ):
            if key in self.metrics:
                value = self.metrics[key]
                lines.append(f"| {key.replace('_', ' ')} | {value:,.4g} |")

        if "net_per_trade_bp" in self.metrics and "cost_bp_per_trade" in self.context:
            net = self.metrics["net_per_trade_bp"]
            cost = float(self.context["cost_bp_per_trade"])
            lines += [
                "",
                f"Net is **{net:+.2f} bp per trade** against a round trip of "
                f"{cost:.2f} bp. Gross figures elsewhere in this document are "
                f"before that cost and are not profits.",
            ]

        if not self.per_period.empty:
            lines += ["", "## Dispersion", ""]
            column = "net_bp" if "net_bp" in self.per_period else self.per_period.columns[-1]
            values = self.per_period[column].to_numpy(dtype="float64")
            lines += [
                f"- periods: {len(values)}",
                f"- positive: {int((values > 0).sum())} of {len(values)}",
                f"- best {values.max():,.1f} bp, worst {values.min():,.1f} bp",
                f"- standard deviation across periods: {values.std(ddof=1):,.1f} bp"
                if len(values) > 1
                else "- one period only",
            ]
            if "per_trade_dispersion_bp" in self.metrics:
                lines.append(
                    f"- per-trade dispersion: {self.metrics['per_trade_dispersion_bp']:.2f} bp"
                )

        if self.context:
            lines += ["", "## How it was produced", "", "| | |", "|---|---|"]
            for key, value in self.context.items():
                lines.append(f"| {key.replace('_', ' ')} | {value} |")

        lines += ["", "## Not established by this scorecard", ""]
        for item in self.unknowns:
            lines.append(f"- {item}")
        return "\n".join(lines) + "\n"

    def write(self, path: Path | str) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.to_markdown(), encoding="utf-8")
        (out.parent / f"{out.stem}.json").write_text(
            json.dumps(
                {
                    "name": self.name,
                    "metrics": self.metrics,
                    "context": self.context,
                    "unknowns": self.unknowns,
                },
                indent=2,
                default=float,
            )
            + "\n",
            encoding="utf-8",
        )
        return out


#: Questions the arithmetic cannot answer, carried into every scorecard so a
#: reviewer is told what is missing rather than left to notice.
STANDING_UNKNOWNS = (
    "Whether the features leak — this is a property of the code, not of the trade log.",
    "Whether the reported configuration was chosen before or after seeing this period.",
    "Whether fills are achievable at size: no queue position, partial fills or impact "
    "are modelled.",
    "Whether the result holds outside the period and instrument measured here.",
)


def build_scorecard(
    name: str,
    net_per_trade_bp: np.ndarray,
    *,
    days: float,
    gross_per_trade_bp: np.ndarray | None = None,
    per_period: pd.DataFrame | None = None,
    context: dict[str, Any] | None = None,
    configurations_tried: int | None = None,
) -> Scorecard:
    """Measure a strategy from its trade-level results.

    ``configurations_tried`` is asked for explicitly and recorded as unknown
    when absent. A search over many candidates reported as one result is the
    single most common way a backtest overstates itself, and a scorecard that
    quietly omits the number would be helping.
    """
    metrics = summarise(np.asarray(net_per_trade_bp, dtype="float64"), days=days)
    if gross_per_trade_bp is not None:
        gross = np.asarray(gross_per_trade_bp, dtype="float64")
        metrics["gross_per_trade_bp"] = float(gross.mean()) if gross.size else float("nan")
    if len(net_per_trade_bp) > 1:
        metrics["per_trade_dispersion_bp"] = float(np.std(net_per_trade_bp, ddof=1))

    unknowns = list(STANDING_UNKNOWNS)
    full_context = dict(context or {})
    if configurations_tried is None:
        unknowns.insert(
            1,
            "How many configurations were tried in total — not supplied, so the "
            "selection dimension cannot be scored above 1.",
        )
    else:
        full_context["configurations_tried"] = configurations_tried

    # Power, stated rather than left to the reader. An edge this small against
    # this dispersion needs a sample this large before it is distinguishable
    # from nothing.
    edge = metrics.get("net_per_trade_bp", float("nan"))
    dispersion = metrics.get("per_trade_dispersion_bp", float("nan"))
    if np.isfinite(edge) and np.isfinite(dispersion) and edge != 0:
        needed = (2.0 * dispersion / abs(edge)) ** 2
        full_context["trades_for_two_sigma"] = f"{needed:,.0f}"
        full_context["trades_available"] = f"{metrics.get('trades', 0):,.0f}"

    return Scorecard(
        name=name,
        metrics=metrics,
        per_period=per_period if per_period is not None else pd.DataFrame(),
        context=full_context,
        unknowns=unknowns,
    )
