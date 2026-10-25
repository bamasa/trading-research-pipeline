"""A read's day records as the status rules read them, and the measurements beside them.

:func:`slot_days` turns the day records of :mod:`.heldout` into one
:class:`~trading_research.spot.statuses.SlotDays` per row over its venue's kept
days (usable, and on Bybit not stale), **a day without a trade as NaN** in every
series that has no value then (``value``, ``gross_bp_turnover``, the placebos,
the cells, the 110 ms re-run), and 0 trades and 0 capacity. :func:`excluded`
counts each venue's excluded days by reason; :func:`calibration` reads FILL-S
and FILL-S thinned (``a_buy`` and ``a_sell``); :func:`measurements` writes what
the page and the results file report beside each status.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_research.spot import statuses as st
from trading_research.spot.calibration import capacity_median, k_fill_charges, side_summary
from trading_research.spot.fills import BRACKETS
from trading_research.spot.plan import ROWS
from trading_research.spot.registration import FAMILY, PRINT_ROWS
from trading_research.spot.runs_prints import RANDOM_DRAWS


def kept_days(records: Mapping[str, Mapping[str, Any]], venue: str) -> list[str]:
    return sorted(d for d, r in records.items() if r["kept"].get(venue))


def excluded(
    records: Mapping[str, Mapping[str, Any]], venue: str, planned: Sequence[date]
) -> dict[date, tuple[str, ...]]:
    """Each planned day of the venue that is not kept, with its reasons (a day the
    read never reached counts as not read)."""
    out: dict[date, tuple[str, ...]] = {}
    for day in planned:
        record = records.get(day.isoformat())
        if record is None:
            out[day] = ("not read",)
        elif not record["kept"].get(venue):
            out[day] = tuple(record["venues"][venue]["reasons"]) or ("not kept",)
    return out


def _values(rows: Sequence[Mapping[str, Any]], key: str, width: int | None = None) -> np.ndarray:
    if width is None:
        return np.array([np.nan if r.get(key) is None else r[key] for r in rows], dtype=float)
    filled = [[np.nan if v is None else v for v in (r.get(key) or [np.nan] * width)] for r in rows]
    return np.array(filled, dtype=float).reshape(len(rows), width).T


def slot_days(
    records: Mapping[str, Mapping[str, Any]], row: str, mde: float | None = None
) -> st.SlotDays:
    """One row's SlotDays over its venue's kept days."""
    venue = ROWS[row].instrument
    days = kept_days(records, venue)
    rows = [records[d]["rows"].get(row, {}) for d in days]
    worse = None
    if row in PRINT_ROWS:
        sums = [r.get("worse") or {} for r in rows]
        worse = st.FillDays(*(np.array([s.get(k, 0.0) for s in sums], dtype=float) for k in
                              ("net_quote", "entry_notional", "buy_notional", "sell_notional")))  # fmt: skip
    trades = np.nan_to_num(_values(rows, "trades"), nan=0.0)
    return st.SlotDays(
        slot=row,
        days=[date.fromisoformat(d) for d in days],
        value=_values(rows, "value"),
        trades=trades,
        gross_bp_turnover=_values(rows, "gross_bp_turnover"),
        ic=_values(rows, "ic"),
        random=_values(rows, "random", RANDOM_DRAWS),
        flipped=_values(rows, "flipped"),
        shifted=_values(rows, "shifted"),
        all_long=_values(rows, "all_long"),
        all_short=_values(rows, "all_short"),
        latency=_values(rows, "latency"),
        neighbourhood=_values(rows, "neighbourhood", st.NEIGHBOURHOOD_CELLS)
        if row in FAMILY else None,
        worse_of_two=worse,
        fills=_values(rows, "fills") if row in PRINT_ROWS else None,
        beyond_sweep=_values(rows, "beyond_sweep") if row in PRINT_ROWS else None,
        capacity=np.nan_to_num(_values(rows, "capacity"), nan=0.0),
        void=tuple(sorted({v for r in rows for v in r.get("void", [])},
                          key=st.VOID_CONDITIONS.index)),
        mde=mde,
    )  # fmt: skip


def _frames(folders: Sequence[Path], name: str) -> pd.DataFrame:
    parts = [
        p for f in folders if (f / name).exists() for p in sorted((f / name).glob("*.parquet"))
    ]
    return (
        pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        if parts
        else pd.DataFrame()
    )


def calibration(folders: Sequence[Path], kept: Sequence[str] | None = None) -> dict[str, Any]:
    """FILL-S per slot and bracket, FILL-S thinned per seed and bracket, and
    ``a_buy`` / ``a_sell``, on the read's kept Bybit days (``kept``)."""
    out: dict[str, Any] = {"fill_s": {}, "thinned": {}, "a_buy": math.nan, "a_sell": math.nan}
    full = _frames(folders, "fill_s")
    thinned = _frames(folders, "thinned")
    if kept is not None:
        full = full[full["day"].isin(kept)] if len(full) else full
        thinned = thinned[thinned["day"].isin(kept)] if len(thinned) else thinned
    for slot in ("S25-PB", "S25-BB"):
        mine = full[full["slot"] == slot] if len(full) else full
        if len(mine):
            out["fill_s"][slot] = {b: side_summary(mine, b).reset_index().to_dict("records")
                                   for b in BRACKETS}  # fmt: skip
    if len(thinned):
        for seed, part in thinned.groupby("seed"):
            out["thinned"][str(seed)] = {b: side_summary(part, b).reset_index().to_dict("records")
                                         for b in BRACKETS}  # fmt: skip
        out.update(k_fill_charges([thinned]))
    return out


def rule_advantage(found: Mapping[str, Any]) -> tuple[float, float] | None:
    """The calibration sentence's numbers: S25-PB's FILL-S under the primary rule, the
    mean advantage per side in bp of mid; None when FILL-S was not measured."""
    rows = found.get("fill_s", {}).get("S25-PB", {}).get("rule")
    if not rows:
        return None
    by = {r["side"]: r["mean_advantage_bp"] for r in rows}
    buy, sell = by.get("buy"), by.get("sell")
    if buy is None or sell is None or not (math.isfinite(buy) and math.isfinite(sell)):
        return None
    return float(buy), float(sell)


def _mean(values: np.ndarray) -> float | None:
    found = np.asarray(values, dtype=float)
    found = found[np.isfinite(found)]
    return float(found.mean()) if len(found) else None


def _max_drawdown(net_quote: np.ndarray) -> float:
    """The largest fall of the cumulative daily net, in quote."""
    path = np.cumsum(np.nan_to_num(np.asarray(net_quote, dtype=float)))
    if not len(path):
        return 0.0
    peaks = np.maximum.accumulate(np.concatenate([[0.0], path]))[1:]
    return float((peaks - path).max())


def measurements(records: Mapping[str, Mapping[str, Any]], s: st.SlotDays) -> dict[str, Any]:
    """Everything reported beside a row's status: the series' statistics, the placebos,
    the brackets, every tier, by side, the long-only variant, the inventory's own P&L,
    the slot's volume, the counters, the waits and the share of grid rows traded."""
    days = [d.isoformat() for d in s.days]
    rows = [records[d]["rows"].get(s.slot, {}) for d in days]
    stats = st.day_statistics(s.value, s.days)
    v = st.series(s.value)
    tiers = sorted({t for r in rows for t in (r.get("tiers") or {})})
    counters: dict[str, float] = {}
    for r in rows:
        for k, n in (r.get("counters") or {}).items():
            counters[k] = counters.get(k, 0.0) + float(n)
    turnover = _values(rows, "turnover")
    grid = float(np.nansum(_values(rows, "grid_rows")))
    out: dict[str, Any] = {
        "days": len(days), "days_with_trade": stats.days, "mean": stats.mean, "t": stats.t,
        "newey_west_t": stats.newey_west_t, "median": stats.median, "p_one_sided": stats.p_one_sided,
        "trades": int(np.nansum(s.trades)), "trades_per_day": st.trades_per_day(s),
        "share_days_positive": float((v > 0).mean()) if len(v) else None,
        "max_drawdown_quote": _max_drawdown(_values(rows, "net_quote")),
        "net_quote_a_day": _mean(_values(rows, "net_quote")),
        "net_per_100_clip_a_day": _mean(_values(rows, "net_per_100_clip")),
        "gross_bp_trade": _mean(_values(rows, "gross_bp")),
        "fee_break_even_bp": st.fee_break_even(s), "ic": _mean(s.ic),
        "placebos": st.placebo_means(s), "placebo_percentile": st.placebo_percentile(s),
        "tiers": {t: _mean(np.array([(r.get("tiers") or {}).get(t, np.nan) for r in rows], float))
                  for t in tiers},
        "by_side": {"long": _mean(_values(rows, "net_bp_long")),
                    "short": _mean(_values(rows, "net_bp_short"))},
        "long_only": _mean(_values(rows, "long_only")),
        "inventory_pnl_quote": float(np.nansum(_values(rows, "inventory_pnl"))),
        "inventory_pnl_bp_a_day": _mean(_values(rows, "inventory_pnl_bp")),
        "volume_30_day_quote": 30.0 * float(np.nanmean(turnover)) if len(turnover) else None,
        "capacity_median": capacity_median(np.nan_to_num(s.capacity, nan=0.0)) if s.capacity is not None else None,
        "counters": counters, "grid_row_share_traded": float(np.nansum(s.trades)) / grid if grid else None,
        "skips": {k: float(np.nansum([float((r.get("skips") or {}).get(k, 0)) for r in rows]))
                  for k in ("flipped", "all_long", "all_short", "random")},
        "void": list(s.void),
    }  # fmt: skip
    if s.neighbourhood is not None:
        out["neighbourhood_median"] = st.neighbourhood_median(s)
        out["cells"] = [st.series_mean(c) for c in np.asarray(s.neighbourhood)]
    if s.slot in PRINT_ROWS:
        out["brackets"] = {b: _mean(np.array([(r.get("brackets") or {}).get(b, np.nan) for r in rows],
                                             float)) for b in ("optimistic", "pessimistic", "stale",
                                                               "worse_of_two")}  # fmt: skip
        for name in ("entry_wait_s", "exit_wait_s", "hold_realised_s"):
            medians = np.array([(r.get(name) or [None, None, None])[1] for r in rows], dtype=float)
            out[f"{name}_median_of_days"] = _mean(medians)
        out["caps"] = st.caps(s) if len(days) else None
    return out
