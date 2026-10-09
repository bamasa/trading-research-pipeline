"""What the pages show beside each status, taken from a block's per-day series in its read.

A block's read (:mod:`.reads`) computes every slot's per-day values once
(:class:`~.statuses.SlotDays`) and assesses each slot on the days its status is
taken on: on H, a slot on S25's signal on the days to 2025-07-31 and every
other slot on all of H; on F, F- when F was split and all of F otherwise; on P,
all of P. :func:`block_numbers` adds, for every slot, on those same days, what
the finalists page and the S25 page show beside the status: the daily series,
its trades, the fee break-even per side, the placebo percentile, the K-nbhd
median, S25-Z's net at 0/0, S-M's making part, and, on H, the mean before
2025-08-01. It keeps the frozen detector's alarms over the whole stream, and
writes the hourly close, read under the session's access, for the
break-detector view. Nothing here decides a status.

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
    """One slot's numbers on ``days``, computed as the status rules compute them."""
    mask = s.mask(days)
    values = _series(s, s.value, mask)
    mean = float(values.mean()) if len(values) else NAN
    draws = np.array([_mean(_series(s, row, mask)) for row in s.random], dtype=np.float64)
    near = [np.nan_to_num(_mean(_series(s, row, mask)), nan=0.0) for row in s.neighbours]
    out: dict[str, Any] = {
        "status_days": int(mask.sum()),
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
        notional = float(np.nansum(s.passive_notional[mask]))
        making = float(np.nansum(s.making[mask]))
        out["making_bp_per_fill"] = making / notional * 1e4 if notional > 0 else NAN
    return out


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


def block_numbers(block: Any, series: dict[str, SlotDays], found: dict[str, Any]) -> dict[str, Any]:
    """Every slot's numbers on its status days, its whole daily series, the
    frozen detector's alarms, and (written beside, not returned) the hourly close."""
    split = found.get("split")
    slots: dict[str, Any] = {}
    for slot, s in series.items():
        numbers = slot_numbers(s, status_days(block.name, s, split))
        numbers["daily"] = {"days": [d.isoformat() for d in s.days],
                            "value": [float(v) for v in s.value]}  # fmt: skip
        numbers["window_h"] = int(s.window_h)
        if block.name == "H":
            early = [d for d in s.days if d <= S25_WINDOW_END]
            numbers["h_before_0801_mean"] = slot_numbers(s, early)["mean"]
        slots[slot] = numbers
    save_close(block)
    return {"slots": slots, "alarms": list(block.detector["alarms"])}


def _d_days(study: Study) -> list[date]:
    return [date.fromisoformat(d) for d in study.record("D")["usable_test_days"]]


def d_daily(study: Study) -> dict[str, pd.Series]:
    """Each slot's walk-forward daily series on D's usable test days: the taker
    slots' ``net_bp_trade`` from the trial ledger (NaN without a trade), S-P's and
    S-M's ``net_per_100_clip_day`` and S25-Z's ``gross_bp_trade`` from stage C."""
    from trading_research.pipeline.short_horizon.freeze import d_series
    from trading_research.reporting.search_keys import S25_KEY

    record_d, record_c = study.record("D"), study.record("C")
    days = _d_days(study)
    out: dict[str, pd.Series] = {}
    taker = {"S25": S25_KEY, "S-C": record_d["choice"]["C*"]["cell"],
             "S-R": record_d["choice"]["R*"]["cell"]}  # fmt: skip
    for slot, key in taker.items():
        if key is not None and key in study.keys:
            out[slot] = d_series(study, key, days)
    rows = pd.read_parquet(study.work / "C" / "cells.parquet")
    passive = {"S-P": record_c["admission"]["S-P"]["cell"], "S-M": record_c["admission"]["S-M"]["cell"],
               "S25-Z": f"S25-Z|{S25_KEY}|entry|fees=zero"}  # fmt: skip
    for slot, label in passive.items():
        mine = rows[(rows["label"] == label) & ~rows["excluded"].astype(bool)]
        if mine.empty:
            continue
        column = "gross_bp_trade" if slot == "S25-Z" else "net_per_100"
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
