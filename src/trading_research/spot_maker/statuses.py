"""A block's summaries, kills and statuses, from its day records, as registered.

:func:`summarise` reads one slot's day records of a block: the daily series of
``net_bp_fill`` (one value per day with a fill) and its statistics
(``market_making/verdicts.day_t``, ``evaluation/significance.newey_west_t``),
every run's block mean (the mean of its daily values over its days with a fill,
0 when it has none), the making part and the markouts pooled over the block's
maker fills, the capacity ratio, the fills and the void conditions.
:func:`kills` evaluates the slot's kill conditions in its registered order and
:func:`assess` applies the precedence and Holm with m = 3.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from trading_research.evaluation.significance import newey_west_t
from trading_research.market_making.verdicts import day_t, holm
from trading_research.spot_maker import registration as rg


def _mean(values: Sequence[float]) -> float:
    found = np.asarray([v for v in values if v is not None], dtype=np.float64)
    found = found[np.isfinite(found)]
    return float(found.mean()) if len(found) else 0.0


def _finite_mean(values: Sequence[float | None]) -> float:
    """A measurement's mean over the days it is defined on; NaN when it never is."""
    found = np.asarray([v for v in values if v is not None], dtype=np.float64)
    found = found[np.isfinite(found)]
    return float(found.mean()) if len(found) else math.nan


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator * 1e4 if denominator > 0 else math.nan


TOTALS: tuple[str, ...] = ("net_quote", "gross_quote", "fees_quote", "turnover", "entry_notional",
                           "taker_notional")  # fmt: skip


def measurements(days: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The block's measurements beside the verdict: pooled money, fills by kind, the
    flattens, the executors' counters, the realised hold and the proxies' width."""
    trues = [d["true"] for d in days]
    totals = {k: float(sum(float(t.get(k, 0.0)) for t in trues)) for k in TOTALS}
    counters: Counter[str] = Counter()
    for t in trues:
        counters.update({k: int(v) for k, v in t.get("counters", {}).items()})
    n = len(days)
    widths = [t.get("width_bp_median") for t in trues]
    widths = [float(w) for w in widths if w is not None and math.isfinite(float(w))]
    medians = [t["hold_s"][1] for t in trues if t.get("hold_s") and t["hold_s"][1] is not None]
    held = sum(float(t.get("hold_s_n", 0.0)) for t in trues)
    return {
        "totals": totals,
        "net_bp_fill_pooled": _ratio(totals["net_quote"], totals["turnover"]),
        "gross_bp_fill_pooled": _ratio(totals["gross_quote"], totals["turnover"]),
        "net_bp_round_trip": _finite_mean([t.get("net_bp_round_trip") for t in trues]),
        "net_bp_round_trip_pooled": _ratio(totals["net_quote"], totals["entry_notional"]),
        "maker_fills_a_day": sum(float(t["maker_fills"]) for t in trues) / n if n else math.nan,
        "taker_fills_a_day": sum(float(t["taker_fills"]) for t in trues) / n if n else math.nan,
        "flattens": int(counters.get("flattens", 0)),
        "flatten_notional": totals["taker_notional"],
        "counters": dict(sorted(counters.items())),
        "hold_s_mean": sum(float(t.get("hold_s_sum", 0.0)) for t in trues) / held if held else math.nan,
        "hold_s_median_of_days": float(np.median(medians)) if medians else math.nan,
        "width_bp_median": float(np.median(widths)) if widths else math.nan,
        "width_bp_range": [min(widths), max(widths)] if widths else [math.nan, math.nan],
    }  # fmt: skip


def _pooled(days: Sequence[Mapping[str, Any]], name: str) -> float:
    total = sum(float(d["true"].get(f"{name}_sum", 0.0)) for d in days)
    count = sum(float(d["true"].get(f"{name}_n", 0.0)) for d in days)
    return total / count if count > 0 else math.nan


def summarise(records: Sequence[Mapping[str, Any]], slot: str) -> dict[str, Any]:
    """One slot's block summary from the block's day records (usable days only)."""
    days = [r["slots"][slot] for r in records if r.get("usable")]
    series = np.asarray([d["value"] for d in days if d["value"] is not None], dtype=np.float64)
    series = series[np.isfinite(series)]
    t, nw = day_t(series), newey_west_t(series, rg.NW_LAG)
    runs = [d["runs"] for d in days]
    out: dict[str, Any] = {
        "usable_days": len(days), "days": len(series), "mean": t.mean, "t": t.t,
        "p_one_sided": t.p_one_sided, "newey_west_t": nw.t,
        "median": float(np.median(series)) if len(series) else math.nan,
        "share_positive": float((series > 0).mean()) if len(series) else math.nan,
        "fills": float(sum(d["true"]["fills"] for d in days)),
        "maker_fills": float(sum(d["true"]["maker_fills"] for d in days)),
        "taker_fills": float(sum(d["true"]["taker_fills"] for d in days)),
        "net_bp_clip_day": _mean([d["true"]["net_bp_clip_day"] for d in days]),
        "gross_bp_fill": _mean([d["true"]["gross_bp_fill"] for d in days]),
        "making_part": _pooled(days, "markout_5s"),
        "markouts": {h: _pooled(days, f"markout_{h:g}s") for h in rg.MARKOUTS_S},
        "spread_part": _pooled(days, "spread"),
        "void": sorted({v for d in days for v in d["void"]}),
    }  # fmt: skip
    out["fills_a_day"] = out["fills"] / len(days) if days else math.nan
    out["days_positive"] = int((series > 0).sum())
    out["adverse_part"] = out["making_part"] - out["spread_part"]
    out.update(measurements(days))
    slot_btc = sum(float(d["true"]["capacity_slot_btc"]) for d in days)
    venue_btc = sum(float(d["true"]["capacity_venue_btc"]) for d in days)
    out["capacity"] = slot_btc / venue_btc if venue_btc > 0 else (math.inf if slot_btc else 0.0)
    posted = sum(float(d["true"]["posted_btc"]) for d in days)
    out["fill_ratio"] = sum(float(d["true"]["filled_maker_btc"]) for d in days) / posted if posted else math.nan
    for name in ("touch", "delayed", "latency", "all_long", "all_short", "flipped"):
        if runs and name in runs[0]:
            out[name] = _mean([r[name] for r in runs])
    if runs:
        out["cells"] = [_mean([r["cells"][k] for r in runs]) for k in range(len(runs[0]["cells"]))]
        if "random" in runs[0]:
            out["random"] = [_mean([r["random"][k] for r in runs]) for k in range(rg.RANDOM_DRAWS)]
    for tier in ("zero_fees", "vip3", "vip9"):
        out[f"tier_{tier}"] = _mean([d["true"]["tiers"].get(tier) for d in days])
    cum = np.cumsum([float(d["true"]["net_quote"]) for d in days])
    out["max_drawdown_quote"] = float((np.maximum.accumulate(cum) - cum).max()) if len(cum) else 0.0
    return out


def kills(summary: Mapping[str, Any], slot: str) -> list[dict[str, Any]]:
    """Each kill of the slot, in its registered order: value, reference, fired."""
    mean = summary["mean"]
    positive = math.isfinite(mean) and mean > 0
    found: dict[str, tuple[float, float, bool]] = {
        "K1": (mean, 0.0, not positive),
        "K-dir": (summary["making_part"], 0.0, not summary["making_part"] > 0),
        "K-days": (summary["share_positive"], 0.5, not summary["share_positive"] > 0.5),
        "K-latency": (summary.get("latency", 0.0), 0.0, positive and summary.get("latency", 0.0) <= 0),
        "K-fill": (summary.get("delayed", 0.0), 0.0, positive and summary.get("delayed", 0.0) <= 0),
        "K-cap": (summary["capacity"], rg.CAPACITY_LIMIT, not summary["capacity"] <= rg.CAPACITY_LIMIT),
    }  # fmt: skip
    drift = max(summary.get("all_long", 0.0), summary.get("all_short", 0.0))
    found["K-drift"] = (mean, drift, not mean > drift)
    cells = summary.get("cells", [])
    median = float(np.median(cells)) if cells else math.nan
    found["K-nbhd"] = (median, 0.0, not median > 0)
    if "random" in summary:
        level = float(np.percentile(summary["random"], rg.PLACEBO_PERCENTILE))
        reference = max(level, summary["flipped"])
        found["K-placebo"] = (mean, reference, not mean > reference)
    return [{"name": k, "value": found[k][0], "reference": found[k][1], "fired": bool(found[k][2])}
            for k in rg.SLOTS[slot].kills]  # fmt: skip


def placebo_percentile(summary: Mapping[str, Any]) -> float:
    """The share of random draws' means below the true mean, in percent (NaN without)."""
    if "random" not in summary or not math.isfinite(summary["mean"]):
        return math.nan
    return float((np.asarray(summary["random"]) < summary["mean"]).mean() * 100.0)


def first_kill(checks: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    return next((c for c in checks if c["fired"]), None)


def bar_met(summary: Mapping[str, Any], block: str) -> bool:
    """Every condition of the bar but Holm."""
    return bool(
        summary["t"] >= rg.T_BAR
        and summary["newey_west_t"] >= rg.NW_BAR
        and summary["median"] > 0
        and summary["days"] >= rg.MIN_DAYS[block]
    )


def assess(summaries: Mapping[str, Mapping[str, Any]], block: str,
           mde: Mapping[str, float]) -> dict[str, dict[str, Any]]:  # fmt: skip
    """Every slot's status on ``block`` (H or P), the registered precedence and Holm."""
    out: dict[str, dict[str, Any]] = {}
    p_values: dict[str, float] = {}
    for slot in rg.FAMILY:
        summary = summaries[slot]
        checks = kills(summary, slot)
        found: dict[str, Any] = {"kills": checks, "first_kill": first_kill(checks)}
        if summary["void"]:
            found["status"] = "void"
        elif summary["fills"] < rg.MIN_FILLS:
            found["status"] = "inconclusive (too few fills)"
        elif found["first_kill"] is not None:
            found["status"] = "killed"
        p = summary["p_one_sided"]
        p_values[slot] = 1.0 if "status" in found or not math.isfinite(p) else float(p)
        out[slot] = found
    passed = holm(p_values, rg.ALPHA)
    for slot, found in out.items():
        found["holm_p"], found["passes_holm"] = p_values[slot], passed[slot]
        if "status" in found:
            continue
        summary = summaries[slot]
        if passed[slot] and bar_met(summary, block):
            found["status"] = "candidate" if block == "H" else "passes"
        elif abs(summary["mean"]) < mde.get(slot, math.nan):
            found["status"] = "inconclusive, below the minimum detectable effect"
        else:
            found["status"] = "inconclusive"
    return out


P_TABLE: dict[str, str] = {
    "passes": "confirmed",
    "inconclusive": "candidate, not confirmed",
    "inconclusive, below the minimum detectable effect": "candidate, not confirmed",
    "inconclusive (too few fills)": "candidate, not confirmed",
    "killed": "killed on P",
    "void": "void",
    "not assessed (P unreadable on Binance)": "candidate, not confirmed (P unreadable)",
    "not assessed (P read failed)": "candidate, not confirmed (P read failed)",
    "not read by its deadline": "candidate, not confirmed (P not read by its deadline)",
}


def after_p(h_status: str, p_status: str | None) -> str:
    """The final status from H's and P's (None: P not read, or P's preset status)."""
    if h_status != "candidate":
        return h_status
    if p_status is None:
        return "candidate (P not yet read)"
    return P_TABLE.get(p_status, p_status)


def _bp(value: float, unit: str = "bp") -> str:
    return "n/a" if value is None or not math.isfinite(value) else f"{value:.2f} {unit}"


KILL_UNITS = {"K-days": "(share of days positive)", "K-cap": "(share of printed volume)"}


def headlines(summaries: Mapping[str, Mapping[str, Any]],
              statuses: Mapping[str, Mapping[str, Any]]) -> list[str]:  # fmt: skip
    """The registered headline sentences on H, brackets filled."""
    out = []
    for slot in rg.FAMILY:
        kill = statuses[slot].get("first_kill")
        name, value, ref = ("none", "n/a", "n/a") if kill is None else (
            kill["name"], _bp(kill["value"], KILL_UNITS.get(kill["name"], "bp")),
            _bp(kill["reference"], KILL_UNITS.get(kill["name"], "bp")))  # fmt: skip
        out.append(f"{slot} on H, Binance spot BTC/FDUSD at the fees charged: "
                   f"{statuses[slot]['status']}; first kill {name} at {value} against {ref}.")  # fmt: skip
    s, q, sq = (summaries[k] for k in rg.FAMILY)
    out.append(f"S25-PF's signal as a maker on Binance BTC/FDUSD, August to October 2026: "
               f"{_bp(s['mean'])} per fill at a 0 bp maker fee, {s['fills_a_day']:.1f} fills a day, "
               f"making part {_bp(s['making_part'])} per fill; {statuses['MK-S']['status']}.")  # fmt: skip
    out.append(f"Two-sided quoting at the touch proxies on Binance BTC/FDUSD, August to October "
               f"2026: {_bp(q['mean'])} per fill, making part {_bp(q['making_part'])} per fill; "
               f"{statuses['MK-Q']['status']}.")  # fmt: skip
    out.append(f"Quoting with S25-PF's side doubled on Binance BTC/FDUSD, August to October 2026: "
               f"{_bp(sq['mean'])} per fill, making part {_bp(sq['making_part'])} per fill; "
               f"{statuses['MK-SQ']['status']}.")  # fmt: skip
    out.append(f"Under the touch rule MK-S's net was {_bp(s.get('touch', math.nan))} per fill "
               f"against {_bp(s['mean'])} under trade-through.")  # fmt: skip
    return out
