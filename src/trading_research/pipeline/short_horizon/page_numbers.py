"""What the pages show beside each status, taken from a block's per-day series in its read.

A block's read (:mod:`.reads`) computes every slot's per-day values once
(:class:`~.statuses.SlotDays`) and assesses each slot on the days its status is
taken on: on H, a slot on S25's signal on the days to 2025-07-31 and every
other slot on all of H; on F, F- when F was split and all of F otherwise; on P,
all of P. :func:`block_numbers` adds, for every slot, on those same days, what
the finalists page and the S25 page show beside the status: the daily series,
its trades, the fee break-even per side, the placebo percentile, the K-nbhd
median, S25-Z's net at 0/0, S-M's making part, and, on H, the mean before
2025-08-01. It keeps the frozen detector's alarms over the whole stream,
summarises S-P's and S25-Z's orders under the three fill rules (FILL on the
block, from the passive runs' day files), and writes the hourly close, read
under the session's access, for the break-detector view. Nothing here decides
a status.

:func:`d_daily` and :func:`d_alarms` give the development period's part of the
regime map: each slot's walk-forward daily series on D's usable test days, cut
at each test month's fold-wise alarms.
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_research.pipeline.short_horizon import beside_verdict as bv
from trading_research.pipeline.short_horizon.context import Study
from trading_research.pipeline.short_horizon.statuses import (
    S25_WINDOW_END,
    SlotDays,
    _mean,
    _series,
    fee_check,
    trade_count,
)

NAN = math.nan
#: Where a read keeps the hourly close for the break-detector view (under the run's work).
CLOSE_FILE = "hourly_close.parquet"


def status_days(block: str, s: SlotDays, split: dict[str, Any] | None = None) -> list[date]:
    """The days a slot's status on ``block`` is taken on."""
    if block == "H":
        return [d for d in s.days if d <= S25_WINDOW_END] if s.on_s25_signal else list(s.days)
    if block == "F" and split and split.get("b") and not split.get("on_all_of_F", True):
        b = date.fromisoformat(str(split["b"]))
        return [d for d in s.days if d < b]
    return list(s.days)


def _percentile(value: float, draws: np.ndarray) -> float:
    from trading_research.reporting.headlines import percentile_of

    finite = draws[np.isfinite(draws)]
    if not len(finite) or not math.isfinite(value):
        return NAN
    return float(percentile_of(value, finite))


def slot_numbers(s: SlotDays, days: list[date]) -> dict[str, Any]:
    """One slot's numbers on ``days``, computed as the status rules compute them:
    ``status_days`` counts the days with a value (a taker slot's days with a
    trade), as the assessment's ``days`` does; ``window_days`` the window's days."""
    mask = s.mask(days)
    values = _series(s, s.value, mask)
    mean = float(values.mean()) if len(values) else NAN
    draws = np.array([_mean(_series(s, row, mask)) for row in s.random], dtype=np.float64)
    near = [np.nan_to_num(_mean(_series(s, row, mask)), nan=0.0) for row in s.neighbours]
    out: dict[str, Any] = {
        "status_days": len(values),
        "window_days": int(mask.sum()),
        "values": [float(v) for v in values],
        "mean": mean,
        "trades": trade_count(s, mask),
        "break_even_per_side": float(fee_check(s, mask).value),
        "placebo_percentile": _percentile(mean, draws),
        "k_nbhd_median": float(np.median(near)) if near else NAN,
    }
    if s.second is not None:
        second = _series(s, s.second, mask)
        out["net_zero_fee_mean"] = float(second.mean()) if len(second) else NAN
    if s.making is not None and s.passive_notional is not None:
        out["making_bp_per_fill"] = bv.making_bp_per_fill(s, mask)
    return out | fee_numbers(s, mask)


def fee_numbers(s: SlotDays, mask: np.ndarray) -> dict[str, Any]:
    """Beside the break-even: how many published tiers pay at their own maker and
    taker fees, a passive slot's maker break-even with the taker exit at 1.5 bp, and
    S25-Z's net on the same trades at the base tier and at the cheapest published
    tier (it is never reported without them). Only from a read that kept the day sums."""
    from trading_research.backtest.costs import BYBIT_BASE, best_maker_tier

    if "day/net_usdt" not in s.extra:
        return {}
    out: dict[str, Any] = {"tiers_paying": bv.tiers_paying(s, mask)}
    if s.kind != "taker":
        out["break_even_maker_bp_exit_1_5"] = bv.passive_break_even(s, mask, 1.5)
    if s.kind == "zero":
        for name, tier in (("base_tier", BYBIT_BASE), ("cheapest_tier", best_maker_tier("g1"))):
            values = bv.series_at(s, mask, tier.maker_bp, tier.taker_bp)
            finite = values[np.isfinite(values)]
            out[f"net_{name}_mean"] = float(finite.mean()) if len(finite) else NAN
            out[f"net_{name}_usdt"] = bv.net_usdt_at(s, mask, tier.maker_bp, tier.taker_bp)
            out[f"net_{name}_fees"] = [tier.maker_bp, tier.taker_bp]
    return out


def delta_b(s: SlotDays, b: date) -> dict[str, Any]:
    """R's statistic for one slot as a measurement: Delta(b), the mean of the daily
    series on F's usable days before ``b`` less its mean after, with its percentile
    among Delta(d) at every admissible split day d != b (at least 20 usable days
    before and 10 after), as R's own test forms them; reported whether or not the
    slot passes on F-, the registration's measurement when it does not."""
    days = list(s.days)

    def mean_on(chosen: list[date]) -> float:
        picked = _series(s, s.value, s.mask(chosen))
        return float(picked.mean()) if len(picked) else 0.0

    def delta(d: date) -> float:
        return mean_on([x for x in days if x < d]) - mean_on([x for x in days if x > d])

    admissible = [d for i, d in enumerate(days) if d != b and i >= 20 and len(days) - i - 1 >= 10]
    found = delta(b)
    others = np.array([delta(d) for d in admissible], dtype=np.float64)
    return {"b": b.isoformat(), "delta_b": found, "admissible_days": len(admissible),
            "percentile_among_admissible": _percentile(found, others)}  # fmt: skip


def close_path(study: Study, label: str) -> Path:
    """The hourly close a read (or the rehearsal, under its label) kept."""
    return study.work / "reads" / label / CLOSE_FILE


def save_close(block: Any) -> Path:
    """The hourly close from the stream's start to the block's end, under the
    session's access, kept for the break-detector view."""
    from trading_research.data import study_access as sa
    from trading_research.pipeline.short_horizon import flags

    end = block.end or sa.BLOCKS[block.name][1]
    close = flags.hourly_close(block.study.symbol, block.study.roots.bars, end, block.access)
    path = block.work / CLOSE_FILE
    close.rename("close").to_frame().to_parquet(path)
    return path


def fill_rule_summary(block: Any) -> dict[str, list[dict[str, Any]]]:
    """FILL on the block: S-P's and S25-Z's orders under the three rules, every
    usable day pooled, one table per slot in the form of
    ``fill_rules.FillComparison.summary`` (the fill ratio is the share of posted
    orders with a fill). Read from the passive runs' day files; nothing when the
    block ran no passive slot, and nothing on a block without the other rules'
    runs (the registration measures FILL on H's streams alone)."""
    from trading_research.market_making.fill_rules import BIAS, RULES
    from trading_research.market_making.queue import FillRule

    files = sorted((block.work / "passive").glob("*.parquet"))
    if not files:
        return {}
    rows = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    if "orders_posted" not in rows:
        return {}
    totals = ["orders_posted", "orders_filled", "volume_posted", "volume_filled", "notional_filled",
              "gross_usdt", "fees_usdt", "net_usdt"]  # fmt: skip
    out: dict[str, list[dict[str, Any]]] = {}
    for slot in ("S-P", "S25-Z"):
        mine = rows[(rows["run_slot"] == slot) & ~rows["excluded"].astype(bool)]
        summary = []
        for rule in RULES:
            variant = "true" if rule is FillRule.QUEUE else f"fill/{rule.value}"
            part = mine[mine["variant"] == variant]
            sums = {k.removesuffix("_usdt"): float(part[k].sum()) for k in totals}
            posted, volume = sums["orders_posted"], sums["volume_posted"]
            summary.append({"rule": rule.value, "days": len(part), **sums,
                            "fill_ratio": sums["orders_filled"] / posted if posted else NAN,
                            "volume_ratio": sums["volume_filled"] / volume if volume else NAN,
                            "bias": BIAS[rule]})  # fmt: skip
        if all(s["days"] for s in summary):
            out[slot] = summary
    return out


def block_numbers(block: Any, series: dict[str, SlotDays], found: dict[str, Any]) -> dict[str, Any]:
    """Every slot's numbers on its status days, its whole daily series, the
    frozen detector's alarms and its mean channel, the second detector's span,
    FILL for S-P's and S25-Z's streams, and (written beside, not returned) the
    hourly close."""
    split = found.get("split")
    taken = found.get("after_F")  # on F: the slots taken there; the others on all of F
    slots: dict[str, Any] = {}
    for slot, s in series.items():
        cut = split if taken is None or slot in taken else None
        numbers = slot_numbers(s, status_days(block.name, s, cut))
        numbers["daily"] = {"days": [d.isoformat() for d in s.days],
                            "value": [float(v) for v in s.value]}  # fmt: skip
        numbers["window_h"] = int(s.window_h)
        if block.name == "H":
            early = [d for d in s.days if d <= S25_WINDOW_END]
            numbers["h_before_0801_mean"] = slot_numbers(s, early)["mean"]
        tested = (found.get("R") or {}).get(slot, {}).get("outcome", "not tested") != "not tested"
        if block.name == "F" and split and split.get("b") and tested:
            passes = (found.get("assessments") or {}).get(slot, {}).get("status") == "passes"
            b = date.fromisoformat(str(split["b"]))
            numbers["R_delta_b"] = delta_b(s, b) | {"passes_on_F_minus": passes}
        slots[slot] = numbers
    from trading_research.pipeline.short_horizon import second_detector

    save_close(block)
    return {"slots": slots, "alarms": list(block.detector["alarms"]),
            "fill_rules": fill_rule_summary(block)} | second_detector.beside_r(block)  # fmt: skip


def _d_days(study: Study) -> list[date]:
    return [date.fromisoformat(d) for d in study.record("D")["usable_test_days"]]


def d_daily(study: Study) -> dict[str, pd.Series]:
    """Each slot's walk-forward daily net on D's usable test days: the taker slots'
    ``net_bp_trade`` from the trial ledger (NaN without a trade), S-P's and S-M's
    ``net_per_100_clip_day`` and S25-Z's net at 0/0 (``net_bp_trade``) from stage C,
    on the days no simulated run flagged."""
    from trading_research.pipeline.short_horizon.freeze import d_series
    from trading_research.pipeline.short_horizon.passive import kept_rows
    from trading_research.reporting.search_keys import S25_KEY

    record_d, record_c = study.record("D"), study.record("C")
    days = _d_days(study)
    out: dict[str, pd.Series] = {}
    taker = {"S25": S25_KEY, "S-C": record_d["choice"]["C*"]["cell"],
             "S-R": record_d["choice"]["R*"]["cell"]}  # fmt: skip
    for slot, key in taker.items():
        if key is not None and key in study.keys:
            out[slot] = d_series(study, key, days)
    rows = kept_rows(pd.read_parquet(study.work / "C" / "cells.parquet"))
    passive = {"S-P": record_c["admission"]["S-P"]["cell"], "S-M": record_c["admission"]["S-M"]["cell"],
               "S25-Z": f"S25-Z|{S25_KEY}|entry|fees=zero"}  # fmt: skip
    for slot, label in passive.items():
        mine = rows[rows["label"] == label]
        if mine.empty:
            continue
        column = "net_bp_trade" if slot == "S25-Z" else "net_per_100"
        index = [date.fromisoformat(str(d)) for d in mine["day"]]
        out[slot] = pd.Series(mine[column].to_numpy(np.float64), index=index).reindex(days)
    return out


def d_alarms(study: Study) -> list[pd.Timestamp]:
    """Each test month's fold-wise alarms inside the month: where D's map is cut."""
    from trading_research.pipeline.short_horizon import flags
    from trading_research.pipeline.short_horizon.plan import TEST_MONTHS

    out: list[pd.Timestamp] = []
    for month in TEST_MONTHS:
        path = study.work / "A" / "flags" / f"{month}.json"
        if not path.exists():
            continue
        fold = flags.month_fold(month)
        for alarm in flags.saved_fold_flags(month, path.parent)["alarms"]:
            stamp = pd.Timestamp(alarm["timestamp"])
            if fold.start <= stamp < fold.end:
                out.append(stamp)
    return sorted(out)


def d_regime_map(study: Study) -> pd.DataFrame:
    """D's segments, every slot's series cut at the fold-wise alarms."""
    from trading_research.reporting.regime_map import regime_maps

    series = d_daily(study)
    days = _d_days(study)
    return regime_maps(series, d_alarms(study), blocks=dict.fromkeys(days, "D"))
