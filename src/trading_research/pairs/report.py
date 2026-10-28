"""The pages ``trp pairs screen`` and ``trp pairs backtest`` write: markdown, csv and json."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from trading_research.pairs.backtest import PairsResult
from trading_research.pairs.costs import LegCosts

FOOTER = (
    "bp are of the first leg's notional; the second leg's notional is |beta| of it. "
    "Decisions on a bar's close, fills at the next bar's open, four taker fills a trade. "
    "Borrow on a short spot leg and funding on perpetuals are not charged."
)


def _cell(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if math.isnan(value):
            return "—"
        if math.isinf(value):
            return "inf"
        return f"{value:,.4g}" if abs(value) < 1e5 else f"{value:,.0f}"
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M")
    return str(value)


def markdown(table: pd.DataFrame, limit: int | None = None) -> str:
    """A frame as a markdown table, at most ``limit`` rows."""
    shown = table if limit is None else table.head(limit)
    lines = [
        "| " + " | ".join(map(str, shown.columns)) + " |",
        "|" + "---|" * len(shown.columns),
    ]
    for row in shown.itertuples(index=False):
        lines.append("| " + " | ".join(_cell(v) for v in row) + " |")
    if limit is not None and len(table) > limit:
        lines.append(f"\n{len(table) - limit} more row(s) in the csv.")
    return "\n".join(lines)


def _json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def write_screen(
    table: pd.DataFrame, out: Path, *, source: str, window: str, costs: LegCosts, alpha: float
) -> Path:
    """``page.md``, ``screen.csv`` and ``summary.json`` under ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "screen.csv", index=False)
    passing = int(table["passes"].sum())
    lines = [
        "# pairs screen",
        "",
        f"{len(table)} pairs from {source}, formation window {window}; "
        f"{passing} pass(es) Holm's correction at {alpha:g} across all {len(table)}.",
        "",
        "Each pair `a` on `b`: hedge by least squares of log closes with an intercept, the "
        "Engle-Granger t of its residual (no constant, the stated lags) with a p-value from "
        "the null simulated at this length, the half-life of the residual's AR(1) in bars, "
        "the correlation of bar log returns, and the residual's standard deviation against "
        "the round trip of both legs (four taker fills, "
        f"{costs.per_fill_bp:g} bp a fill, {costs.source}). `critical_5` is MacKinnon's "
        "(2010) 5% value for two series, for reference.",
        "",
        markdown(table),
        "",
        "A pair that passes on its formation window is a candidate for `trp pairs backtest`, "
        "which trades it on the bars after; it is not a result. " + FOOTER,
        "",
    ]
    (out / "page.md").write_text("\n".join(lines), encoding="utf-8")
    _json(
        out / "summary.json",
        {
            "source": source,
            "window": window,
            "alpha": alpha,
            "pairs": len(table),
            "passing": passing,
            **costs.describe(),
        },
    )
    return out / "page.md"


SUMMARY_ROWS = (
    ("trades", "trades"),
    ("windows (traded)", None),
    ("days with trades", "days_with_trades"),
    ("gross per trade, bp", "gross_per_trade_bp"),
    ("cost per trade, bp", "cost_per_trade_bp"),
    ("net per trade, bp", "net_per_trade_bp"),
    ("net total, bp", "net_total_bp"),
    ("gross, day-clustered t", "gross_day_t"),
    ("net, day-clustered t", "net_day_t"),
    ("share of days net positive", "net_days_positive_share"),
    ("mean bars held", "mean_bars_held"),
    ("exits", "exits"),
    ("cost per trade at which net is zero, bp", "cost_to_break_even_bp_per_trade"),
    ("cost per fill at which net is zero, bp", "cost_to_break_even_bp_per_fill"),
    ("cost cut per trade to break even, bp", "cost_cut_to_break_even_bp_per_trade"),
)


def summary_table(summary: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for label, key in SUMMARY_ROWS:
        if key is None:
            value: Any = f"{summary['windows']} ({summary['windows_traded']})"
        else:
            value = summary.get(key, math.nan)
        rows.append({"measure": label, "value": value})
    return pd.DataFrame(rows)


def write_backtest(result: PairsResult, out: Path, *, source: str) -> Path:
    """``page.md``, ``trades.csv``, ``days.csv``, ``windows.csv`` and ``summary.json``."""
    out.mkdir(parents=True, exist_ok=True)
    result.trades.to_csv(out / "trades.csv", index=False)
    result.days.to_csv(out / "days.csv", index=False)
    result.windows.to_csv(out / "windows.csv", index=False)
    rule, s = result.rule, result.summary
    gate = "none" if rule.gate_p is None else f"formation p-value <= {rule.gate_p:g}"
    lines = [
        f"# pairs backtest: {result.a} on {result.b}",
        "",
        f"Bars from {source}. Walk-forward: formation {rule.formation} bars, trading "
        f"{rule.trading}, stepped by {rule.trading}; hedge, residual mean and sd fitted on "
        f"formation only. Bands: entry |z| >= {rule.entry_z:g}, exit |z| <= {rule.exit_z:g}, "
        f"stop |z| >= {rule.stop_z:g}, max hold {rule.max_hold} bars; gate {gate}. "
        f"Costs: {result.costs.per_fill_bp:g} bp a fill on each leg "
        f"(fee {result.costs.taker.fee_bp_per_side:g}, slippage "
        f"{result.costs.taker.slippage_bp:g}; {result.costs.source}).",
        "",
        "## Result",
        "",
        markdown(summary_table(s)),
        "",
        "The day-clustered t is the mean net (or gross) per trade within each UTC day of "
        "exit, then the t of those day means (`evaluation.significance.assess`). The cost "
        "at which net is zero restates the gross (`walkthrough.verdict.headroom`); it is "
        "not a result at any other cost a strategy would have traded differently at.",
        "",
        "## Windows",
        "",
        markdown(result.windows),
        "",
        "## Days",
        "",
        markdown(result.days, limit=60) if len(result.days) else "No trade.",
        "",
        FOOTER,
        "",
    ]
    (out / "page.md").write_text("\n".join(lines), encoding="utf-8")
    _json(out / "summary.json", {"source": source, **s})
    return out / "page.md"
