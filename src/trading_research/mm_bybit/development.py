"""D: every cell on every usable day, the choice of F, the nulls, sigma_D and the MDEs.

Everything here is development: selected, biased upward, and never quoted as a
result. The output is the D record the amendment freezes from: the choice and
its nulls, and, gating nothing, every cell's daily series and summary, F's and
S0's decomposition, the controls and the markouts.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trading_research.mm_bybit import quoters as q
from trading_research.mm_bybit import selection as sel

H_DAYS = 84
MDE_Z = 3.84
SERIES = ("y", "passive_fills", "making_y", "spread_y", "adverse_y", "rebate_y",
          "taker_fees_y", "funding_y", "inventory_y", "flatten_cost_y", "ex_rebate_y",
          "negative_control_y", "positive_control_y", "capacity", "passive_turnover",
          "max_abs_position", "soft_limit_share", "gate_open_share")  # fmt: skip
DAILY = ("y", "passive_fills", "making_y")
DECOMPOSITION = ("y", "making_y", "spread_y", "adverse_y", "rebate_y", "taker_fees_y",
                 "funding_y", "inventory_y", "flatten_cost_y")  # fmt: skip


def matrices(rows: pd.DataFrame, cells: list[q.Cell]) -> tuple[list[str], dict[str, np.ndarray]]:
    """The usable days and the cells x days matrices of every series (primary rows)."""
    primary = rows[rows["variant"] == "primary"]
    day_list = sorted(primary["day"].unique())
    keys = [c.key for c in cells]
    out: dict[str, np.ndarray] = {}
    for name in SERIES:
        table = primary.pivot(index="cell", columns="day", values=name)  # noqa: PD010
        out[name] = table.loc[keys, day_list].to_numpy(dtype=np.float64)
    hours = primary.pivot(index="cell", columns="day", values="hourly_y")  # noqa: PD010
    hours = hours.loc[keys, day_list]
    out["hourly"] = np.array([[list(v) for v in row] for row in hours.to_numpy()], dtype=float)
    return day_list, out


def _markouts(rows: pd.DataFrame, key: str) -> dict[str, float | None]:
    """A cell's passive markouts over D, bp of its passive turnover."""
    part = rows[(rows["variant"] == "primary") & (rows["cell"] == key)]
    turnover = float(part["passive_turnover"].sum())
    out: dict[str, float | None] = {}
    for h in ("1", "5", "30"):
        total = sum(float(v[h]) for v in part["markout_usdt"])
        out[h] = total / turnover * 1e4 if turnover else None
    return out


def _market_markouts(rows: pd.DataFrame) -> dict[str, float]:
    per_day = rows.drop_duplicates("day")["market_markout_bp"]
    return {h: float(np.mean([float(v[h]) for v in per_day])) for h in ("1", "5", "30")}


def _by_month(values: np.ndarray, day_list: list[str]) -> dict[str, float]:
    series = pd.Series(values, index=[d[:7] for d in day_list])
    grouped = series.groupby(level=0).mean()
    return {str(k): float(v) for k, v in grouped.items()}


def develop(rows: pd.DataFrame, *, null_a_draws: int = 999) -> dict[str, Any]:
    """The choice, the nulls and the frozen values, from D's primary rows."""
    cells = q.grid()
    keys = [c.key for c in cells]
    day_list, m = matrices(rows, cells)
    nbhd = sel.neighbour_index(cells)
    admissible = m["passive_fills"].mean(axis=1) >= sel.MIN_FILLS_PER_DAY
    choice = sel.choose(m["y"], admissible, nbhd, keys)
    record: dict[str, Any] = {
        "label": "development, selected, biased upward",
        "usable_days": len(day_list), "first_day": day_list[0], "last_day": day_list[-1],
        "days": day_list,
    }  # fmt: skip
    means = m["y"].mean(axis=1)
    s_values = [float(np.median(means[idx])) for idx in nbhd]
    t_values = sel.t_rows(m["y"])
    record["cells"] = {}
    for i, k in enumerate(keys):
        summary: dict[str, Any] = {
            "mean_y": float(means[i]), "S": s_values[i], "t": float(t_values[i]),
            "fills_per_day": float(m["passive_fills"][i].mean()),
            "median_day": float(np.median(m["y"][i])),
            "share_days_positive": float(np.mean(m["y"][i] > 0)),
            "capacity_median": float(np.median(m["capacity"][i])),
            "admissible": bool(admissible[i]),
            "markout_bp": _markouts(rows, k),
            "y_by_month": _by_month(m["y"][i], day_list),
        }  # fmt: skip
        for name in SERIES:
            if name not in ("y", "passive_fills", "capacity"):
                summary[f"mean_{name}"] = float(m[name][i].mean())
        turnover = float(m["passive_turnover"][i].sum())
        clip_notional = rows[(rows["variant"] == "primary") & (rows["cell"] == k)]
        scale = clip_notional.set_index("day").loc[day_list, "clip_notional"].to_numpy(float)
        net = float((m["y"][i] * scale / 100.0).sum())
        rebate = float((m["rebate_y"][i] * scale / 100.0).sum())
        making = float((m["making_y"][i] * scale / 100.0).sum())
        summary["net_bp_per_fill"] = net / turnover * 1e4 if turnover else None
        summary["ex_rebate_bp_per_fill"] = (net - rebate) / turnover * 1e4 if turnover else None
        summary["making_bp_per_fill"] = making / turnover * 1e4 if turnover else None
        summary["daily"] = {name: [float(v) for v in m[name][i]] for name in DAILY}
        for variant in ("touch", "latency10"):
            part = rows[(rows["variant"] == variant) & (rows["cell"] == k)]
            if len(part):
                summary[f"{variant}_mean_y"] = float(
                    part.set_index("day").loc[day_list, "y"].mean()
                )
        record["cells"][k] = summary
    record["market_markout_bp"] = _market_markouts(rows)
    record["outright_peak"] = keys[choice.peak]
    if choice.index < 0:
        record.update(chosen_cell=None, admission={"admitted": False,
                      "reasons": "not run (no admissible cell on D)"})  # fmt: skip
        s0 = keys.index(q.S0.key)
        sigma_s0 = float(np.std(m["y"][s0], ddof=1))
        record["sigma_d"] = {"S0": sigma_s0}
        record["mde"] = {"S0": MDE_Z * sigma_s0 / np.sqrt(H_DAYS)}
        record["decomposition"] = {"S0": _decomposition(m, s0)}
        return record
    chosen = keys[choice.index]
    null_a = sel.null_a(m["hourly"], admissible, nbhd, keys, choice.a, null_a_draws)
    null_b = sel.null_b(m["y"], admissible, nbhd, keys, choice.a)
    reasons = [r for r, failed in (("i", not choice.s > 0),
                                   ("iv", not max(null_a.p, null_b.p) <= 0.05)) if failed]  # fmt: skip
    record["chosen_cell"] = chosen
    record["neighbourhood"] = [c.key for c in q.neighbourhood(cells[choice.index])]
    record["admission"] = {
        "S": choice.s, "A": choice.a, "p_A": null_a.p, "p_B": null_b.p,
        "count_A": null_a.count, "draws_A": len(null_a.a), "count_B": null_b.count,
        "shifts_B": len(null_b.a), "null_A_q95": _q95(null_a.a),
        "null_B_q95": _q95(null_b.a),
        "admitted": not reasons, "reasons": ", ".join(reasons),
    }  # fmt: skip
    sigma: dict[str, float] = {}
    for name, key in (("F", chosen), ("S0", q.S0.key)):
        sigma[name] = float(np.std(m["y"][keys.index(key)], ddof=1))
    record["sigma_d"] = sigma
    record["mde"] = {k: MDE_Z * v / np.sqrt(H_DAYS) for k, v in sigma.items()}
    record["decomposition"] = {
        "F": _decomposition(m, choice.index),
        "S0": _decomposition(m, keys.index(q.S0.key)),
    }
    return record


def _q95(values: np.ndarray) -> float | None:
    return float(np.quantile(values, 0.95)) if len(values) else None


def _decomposition(m: dict[str, np.ndarray], i: int) -> dict[str, float]:
    """A cell's mean daily parts over D, per 100 USDT of clip."""
    return {name: float(m[name][i].mean()) for name in DECOMPOSITION}
