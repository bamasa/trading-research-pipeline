"""Turning fold results into a report.

The report is written to be read by someone who did not run it, and who is
entitled to be sceptical. So it leads with the assumptions and the cost floor
rather than with the headline number, and it always shows the per-fold spread
rather than only the mean.

That last choice matters more than it sounds. A mean across seven overlapping
folds looks like seven observations and is not: the windows here shift one day
against a twenty-eight-day span, so they share about 96% of their data.
Reporting the mean alone invites a reader to treat it as an average over
independent trials. Showing every fold, and how many were positive, makes the
dependence visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ReportContext:
    """Everything a reader needs in order to interpret the numbers."""

    symbol: str
    source: str
    period: str
    grid: str
    horizon_obs: int
    horizon_seconds: float
    cost_round_trip_bp: float
    cost_description: dict[str, Any]
    label_threshold_bp: float
    class_balance: dict[str, float]
    perfect_foresight_share: float
    n_folds: int
    fold_layout: str
    features: list[str]
    synthetic: bool = False


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-fold results by model."""
    return (
        results.groupby("model")
        .agg(
            folds=("fold", "count"),
            folds_positive=("net_bp", lambda s: int((s > 0).sum())),
            trades=("trades", "mean"),
            trade_rate=("trade_rate", "mean"),
            hit_rate=("hit_rate", "mean"),
            gross_bp=("gross_bp", "mean"),
            cost_bp=("cost_bp", "mean"),
            net_bp=("net_bp", "mean"),
            net_bp_worst=("net_bp", "min"),
            net_bp_best=("net_bp", "max"),
            net_per_trade_bp=("net_per_trade_bp", "mean"),
        )
        .sort_values("net_bp", ascending=False)
    )


def to_markdown(results: pd.DataFrame, context: ReportContext) -> str:
    """Render a complete report."""
    summary = summarise(results)
    stamp = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = []
    add = lines.append

    add(f"# Walk-forward results — {context.symbol}")
    add("")
    if context.synthetic:
        add("> **SYNTHETIC DATA.** Generated, not observed. These numbers say")
        add("> nothing about real markets.")
        add("")
    add(f"Generated {stamp}.")
    add("")

    # ---- assumptions first -------------------------------------------------
    add("## What was assumed")
    add("")
    add("Read these before the numbers. Every one of them moves the result.")
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| Instrument | {context.symbol} |")
    add(f"| Data | {context.source} |")
    add(f"| Period | {context.period} |")
    add(f"| Sampling | {context.grid} |")
    add(f"| Horizon | {context.horizon_obs} obs = {context.horizon_seconds:.0f} s |")
    add(f"| Execution | {context.cost_description.get('execution', 'taker')} on both legs |")
    add(f"| Fee | {context.cost_description.get('fee_bp_per_side')} bp per side |")
    add(f"| Slippage | {context.cost_description.get('slippage_bp')} bp per side |")
    add(f"| **Round trip** | **{context.cost_round_trip_bp:.2f} bp** |")
    add(f"| Label threshold | {context.label_threshold_bp:.2f} bp |")
    add(f"| Validation | {context.fold_layout}, {context.n_folds} folds |")
    add("")

    # ---- the ceiling -------------------------------------------------------
    add("## The ceiling, before any model")
    add("")
    add(
        f"Only **{context.perfect_foresight_share:.2%}** of moments have a future move "
        f"larger than the {context.cost_round_trip_bp:.2f} bp it costs to trade — "
        f"and that figure already assumes the direction is predicted perfectly."
    )
    add("")
    add("This is arithmetic, not modelling. It bounds what any classifier on this")
    add("horizon can achieve, and no amount of model capacity changes it.")
    add("")
    balance = ", ".join(f"{k} {v:.2%}" for k, v in sorted(context.class_balance.items()))
    add(f"Class balance at that threshold: {balance}.")
    add("")

    # ---- results -----------------------------------------------------------
    add("## Test results")
    add("")
    add("Net basis points summed over each test block, after costs. The threshold")
    add("was chosen on validation and applied unchanged to test.")
    add("")
    add(_frame_to_markdown(summary.round(3)))
    add("")

    add("### Per fold")
    add("")
    add("Shown in full because the folds overlap: each shifts one day against a")
    add("28-day window, so they share about 96% of their data. Seven folds are not")
    add("seven independent trials, and a mean across them should not be read as one.")
    add("")
    per_fold = results.pivot_table(index="fold", columns="model", values="net_bp", aggfunc="sum")
    add(_frame_to_markdown(per_fold.round(1)))
    add("")

    # ---- reading it --------------------------------------------------------
    add("## How to read this")
    add("")
    net = summary["net_bp"].astype("float64")
    best = str(summary.index[0])
    hold_net = float(net["always_hold"]) if "always_hold" in net.index else 0.0
    best_net = float(net[best])

    if best == "always_hold" or best_net <= hold_net:
        add("**No model beat not trading.** Under these costs, at this horizon, on")
        add("this period, the best available action was to stand aside. That is a")
        add("result about the cost floor rather than about the models: the fee is")
        add("large relative to the moves being predicted.")
    else:
        add(f"`{best}` produced the highest mean net figure ({best_net:.1f} bp).")
        add("")
        add("Before treating that as an edge, check three things in the table above:")
        add("whether it is positive in most folds rather than rescued by one;")
        add("whether net per trade is comfortably above zero rather than marginal;")
        add("and whether the trade count is large enough for the mean to mean anything.")
    add("")
    add("`always_hold` is in the comparison deliberately. Trading is expensive, so")
    add("doing nothing has a real expected value, and a model that does not beat it")
    add("net of costs has not earned its complexity.")
    add("")

    add("## Limitations")
    add("")
    add("This is a backtest computed with hindsight under the assumptions above. It")
    add("is not evidence that any strategy is or was profitable. Execution is")
    add("simplified — no queue position, no partial fills, no market impact, no")
    add("limit on concurrent exposure — and every one of those simplifications")
    add("flatters the result. See [limitations.md](../docs/limitations.md).")
    add("")
    add(f"Features used: {', '.join(context.features)}.")
    add("")

    return "\n".join(lines)


def _frame_to_markdown(frame: pd.DataFrame) -> str:
    """Render a frame as a markdown table without needing an optional dependency."""
    index_name = frame.index.name or ""
    header = f"| {index_name} | " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "|---" * (len(frame.columns) + 1) + "|"
    rows = [
        f"| {idx} | " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
        for idx, row in zip(frame.index, frame.to_numpy(), strict=True)
    ]
    return "\n".join([header, rule, *rows])


def write_report(results: pd.DataFrame, context: ReportContext, path: Path | str) -> Path:
    """Write the report to disk and return where it went."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(to_markdown(results, context), encoding="utf-8")
    return target
