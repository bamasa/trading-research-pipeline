"""The replication's results page: its tables, from the fits, the pooled numbers and the days."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from trading_research.replication.fdusd_windows import WINDOWS

FEE_TITLES: dict[str, str] = {
    "zero": "zero fees: 0 maker, 0 taker (the 2025 backtest's charge; VIP 2-9 in 2025)",
    "regular_10bp": "the Regular account: 0 maker, 10 bp taker",
}
POOLED_COLUMNS: dict[str, str] = {
    "window": "Window", "variant": "Variant", "days": "Test days", "trades_per_day": "Trades a day",
    "net_bp_trade": "Net bp a trade (pooled)", "net_quote_per_trade": "Net FDUSD a trade (0.010 BTC)",
    "net_quote_per_day": "Net FDUSD a day", "share_days_positive": "Days positive",
    "net_bp_day_mean": "Mean of days, bp", "day_t": "Day t",
    "gross_bp_turnover": "Gross bp of turnover (break-even fee a side)",
}  # fmt: skip


def cell(value: Any, digits: int = 2) -> str:
    if isinstance(value, str):
        return value
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def table(frame: pd.DataFrame, columns: Mapping[str, str], digits: Mapping[str, int]) -> str:
    head = "| " + " | ".join(columns.values()) + " |"
    rule = "|" + "---|" * len(columns)
    body = ["| " + " | ".join(cell(r[c], digits.get(c, 2)) for c in columns) + " |"
            for r in frame.to_dict("records")]  # fmt: skip
    return "\n".join([head, rule, *body])


def pooled_table(summary: pd.DataFrame, fees: str) -> str:
    found = summary[summary["fees"] == fees].copy()
    found["variant"] = found["variant"].str.replace("_", "-")
    found["share_days_positive"] = found["share_days_positive"].map(
        lambda v: f"{v:.0%}" if math.isfinite(v) else "n/a"
    )
    return table(found, POOLED_COLUMNS, {"trades_per_day": 1, "net_quote_per_day": 1})


def windows_table(fits: Sequence[Mapping[str, Any]]) -> str:
    rows = []
    for fit in fits:
        w = WINDOWS[fit["window"]]
        classes = fit["classes"]
        rows.append({
            "window": w.name, "train": f"{w.train[0]} to {w.train[1]}",
            "inner": f"{w.inner[0]} to {w.inner[1]}", "test": f"{w.test[0]} to {w.test[1]}",
            "rows": int(fit["rows"]),
            "classes": " / ".join(str(classes.get(k, 0)) for k in ("1", "0", "-1")),
            "theta": f"{fit['theta']['theta_long']:.4f} / {fit['theta']['theta_short']:.4f}",
        })  # fmt: skip
    columns = {"window": "Window", "train": "Training days", "inner": "Thresholds on",
               "test": "Test days", "rows": "Training rows",
               "classes": "Long / hold / short labels", "theta": "θ long / θ short"}  # fmt: skip
    return table(pd.DataFrame(rows), columns, {})


def days_table(days: pd.DataFrame, window: str) -> str:
    """The two-sided slot's days at both fees."""
    mine = days[(days["window"] == window) & (days["variant"] == "two_sided")]
    wide = mine.pivot_table(index="day", columns="fees", values=["net_bp", "net_quote_per_trade"])
    zero = mine[mine["fees"] == "zero"].set_index("day")
    at = zero.index

    def col(name: str, fees: str) -> list[float]:
        if (name, fees) not in wide:
            return [math.nan] * len(at)
        return list(wide[(name, fees)].reindex(at).to_numpy())

    rows = pd.DataFrame({"day": at, "trades": zero["trades"].astype(int).to_numpy(),
                         "zero_bp": col("net_bp", "zero"),
                         "zero_q": col("net_quote_per_trade", "zero"),
                         "reg_bp": col("net_bp", "regular_10bp"),
                         "reg_q": col("net_quote_per_trade", "regular_10bp"),
                         "market": zero["market_bp"].to_numpy()})  # fmt: skip
    columns = {"day": "Day", "trades": "Trades", "zero_bp": "Net bp a trade, zero fees",
               "zero_q": "Net FDUSD a trade, zero fees", "reg_bp": "Net bp a trade, 10 bp taker",
               "reg_q": "Net FDUSD a trade, 10 bp taker", "market": "BTC's move, bp"}  # fmt: skip
    return table(rows, columns, {"market": 0, "zero_q": 3, "reg_q": 3})


def signal_table(summary: pd.DataFrame) -> str:
    """The two-sided slot's rate and signal per window (the same at both fees)."""
    found = summary[(summary["fees"] == "zero") & (summary["variant"] == "two_sided")].copy()
    found["long_share"] = found["long_share"].map(
        lambda v: f"{v:.0%}" if math.isfinite(v) else "n/a"
    )
    columns = {"window": "Window", "days": "Test days", "decisions_per_day": "Decisions a day",
               "trades_per_day": "Trades a day", "long_share": "Long trades",
               "ic_mean": "Mean rank IC", "gross_bp_trade": "Gross bp a trade",
               "market_bp_day_mean": "BTC's mean move a day, bp"}  # fmt: skip
    return table(found, columns, {"decisions_per_day": 1, "trades_per_day": 1, "ic_mean": 4,
                                  "market_bp_day_mean": 0})  # fmt: skip
