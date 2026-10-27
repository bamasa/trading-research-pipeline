"""H's runs, the primaries {F, S0} assembled from them, and the results record.

The jobs on H: S0 and F under the primary rule, the touch and one-further-print
brackets and at 150 ms and 10 ms; every cell of F's and of S0's neighbourhoods
under the primary rule; for a gated F, its shifted-state run.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trading_research.mm_bybit import quoters as q
from trading_research.mm_bybit import statuses as st
from trading_research.mm_bybit.access import STRESS_DAYS

FINALIST_VARIANTS = ("primary", "touch", "delay", "latency150", "latency10")


def jobs(chosen: str | None) -> list[tuple[str, str]]:
    """Every (cell, variant) H runs, without the shifted-state run."""
    keys = [q.S0.key] + ([chosen] if chosen and chosen != q.S0.key else [])
    out = [(key, v) for key in keys for v in FINALIST_VARIANTS]
    for key in keys:
        out += [(n.key, "primary") for n in q.neighbourhood(q.from_key(key))]
    return sorted(set(out))


def _series(
    rows: pd.DataFrame, cell: str, variant: str, column: str, days: list[str]
) -> np.ndarray:
    part = rows[(rows["cell"] == cell) & (rows["variant"] == variant)].set_index("day")
    return part.loc[days, column].to_numpy(dtype=np.float64)


def primary(
    name: str, cell: str, rows: pd.DataFrame, days: list[str], *, holm_one: str = "",
    mde: float = float("nan"),
) -> st.Primary:  # fmt: skip
    base = rows[(rows["cell"] == cell) & (rows["variant"] == "primary")].set_index("day")
    base = base.loc[days]
    void = []
    if not bool(rows.loc[rows["cell"] == cell, "identity_ok"].all()) or not bool(
        rows.loc[rows["cell"] == cell, "turnover_ok"].all()
    ):
        void.append("K-acct")
    if float(rows.loc[rows["cell"] == cell, "limit_breach"].sum()) > 0:
        void.append("K-limit")
    neighbours = [n.key for n in q.neighbourhood(q.from_key(cell))]
    shifted = rows[(rows["cell"] == cell) & (rows["variant"] == "shifted")]
    return st.Primary(
        name=name, cell=cell, y=base["y"].to_numpy(float), making=base["making_y"].to_numpy(float),
        inventory=base["inventory_y"].to_numpy(float), funding=base["funding_y"].to_numpy(float),
        ex_rebate=base["ex_rebate_y"].to_numpy(float),
        touch=_series(rows, cell, "touch", "y", days),
        delay=_series(rows, cell, "delay", "y", days),
        latency150=_series(rows, cell, "latency150", "y", days),
        hourly=np.array([list(v) for v in base["hourly_y"]], dtype=float),
        fills=int(base["passive_fills"].sum()), capacity=base["capacity"].to_numpy(float),
        neighbour_means=[float(_series(rows, n, "primary", "y", days).mean()) for n in neighbours],
        shifted_mean=float(shifted["y"].mean()) if len(shifted) else None,
        void=void, holm_one=holm_one, mde=mde,
    )  # fmt: skip


def measurements(rows: pd.DataFrame, cell: str, days: list[str]) -> dict[str, Any]:
    """Every measurement beside a primary: brackets, latencies, re-pricings, stress day."""
    base = rows[(rows["cell"] == cell) & (rows["variant"] == "primary")].set_index("day")
    base = base.loc[days]
    turnover = float(base["passive_turnover"].sum())

    def bp(total: float) -> float | None:
        return total / turnover * 1e4 if turnover else None

    net, rebate = float(base["net_usdt"].sum()), float(base["rebate_usdt"].sum())
    out: dict[str, Any] = {
        "days": len(days),
        "net_bp_per_fill": bp(net),
        "ex_rebate_bp_per_fill": bp(net - rebate),
        "making_bp_per_fill": bp(float(base["making_usdt"].sum())),
        "rebate_bp_per_fill": bp(rebate),
        "fills_per_day": float(base["passive_fills"].mean()),
        "fills_total": int(base["passive_fills"].sum()),
        "passive_turnover_per_day": turnover / len(days) if days else None,
        "share_days_positive": float((base["y"] > 0).mean()),
        "median_y": float(base["y"].median()),
        "max_abs_position_max": float(base["max_abs_position"].max()),
        "capacity_median": float(base["capacity"].median()),
    }
    for column in ("y", "making_y", "spread_y", "adverse_y", "rebate_y", "taker_fees_y",
                   "funding_y", "inventory_y", "flatten_cost_y", "ex_rebate_y",
                   "negative_control_y", "positive_control_y", "max_abs_position",
                   "soft_limit_share", "gate_open_share"):  # fmt: skip
        out[f"mean_{column}"] = float(base[column].mean())
    for variant in ("touch", "delay", "latency150", "latency10"):
        out[f"{variant}_mean_y"] = float(_series(rows, cell, variant, "y", days).mean())
    marks = {h: sum(float(v[h]) for v in base["markout_usdt"]) for h in ("1", "5", "30")}
    out["markout_bp"] = {h: bp(v) for h, v in marks.items()}
    market = list(base["market_markout_bp"])
    out["market_markout_bp"] = {h: float(np.mean([m[h] for m in market])) for h in ("1", "5", "30")}
    for stress in STRESS_DAYS:
        key = stress.isoformat()
        if key in base.index:
            row = base.loc[key].to_dict()
            stress_day: dict[str, Any] = {
                c: float(row[c]) for c in ("y", "making_y", "inventory_y")
            }
            stress_day["passive_fills"] = int(row["passive_fills"])
            out[f"stress_{key}"] = stress_day
            out["mean_y_without_stress"] = float(base.drop(index=key)["y"].mean())
        else:
            out[f"stress_{key}"] = "excluded"
    months = pd.Series(base["y"].to_numpy(float), index=[d[:7] for d in days])
    grouped = months.groupby(level=0).mean()
    out["y_by_month"] = {str(k): float(v) for k, v in grouped.items()}
    out["daily"] = {
        "day": list(days),
        **{c: [float(v) for v in base[c]] for c in ("y", "making_y", "inventory_y")},
        "passive_fills": [int(v) for v in base["passive_fills"]],
    }
    return out
