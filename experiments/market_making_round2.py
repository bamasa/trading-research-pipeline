"""The second market-making round on its development block (D), and the freeze.

Runs everything ``docs/preregistration/market_making_round2.md`` assigns to
block D, 2024-02-01 to 2024-02-25, and nothing on any later day:

1. the admission screen on the eight candidates (A1 or A2), H's coverage from
   the file system alone; the clip cap and ``sigma_ref`` per admitted
   instrument;
2. every D print of the admitted instruments and of BICOUSDT scored from its
   resting side at five seconds, kept; the hour masks per instrument;
3. S1's successive-halving search per admitted instrument (144 cells, budgets
   of 5, 12 and 25 days on a seeded nested subset, keep 0.34), each choice by
   the neighbourhood of the cells measured at the full budget;
4. the 42 gate cells on every D day and admitted instrument, on each
   instrument's chosen S1 or its S1-touch cell, and the one gate cell for the
   basket;
5. B3's strategy at each professional tier (each instrument's S1 or the chosen
   G(base), every value frozen);
6. B4's 42 cells on BICOUSDT's D, on round one's frozen S1 and S1-touch;
7. the daily spread of every primary's metric and its minimum detectable
   effect; with ``--freeze``, the chosen values written into the placeholders
   of ``configs/mm_prereg_round2.yaml``, and nothing else.

Every reader goes through :class:`trading_research.market_making.round2.Access`;
this script holds only the development access, which permits D for the eight
candidates and BICOUSDT, so any other day raises before a file of it is opened.

Usage::

    .venv/bin/python -m experiments.market_making_round2 --workers 8
    .venv/bin/python -m experiments.market_making_round2 --workers 8 --freeze

Results land in ``experiments/results/mm_round2_*_D.csv``; per-day rows and the
scored prints in the gitignored ``artifacts/mm/round2``, from which a re-run is
answered.
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
import warnings
from collections.abc import Sequence
from datetime import date
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from experiments import _round2 as r2
from experiments._common import RESULTS
from trading_research.market_making import gate, round2, round2_run, round2_stats
from trading_research.market_making.heldout import Spec
from trading_research.validation.search import neighbourhood_scores, successive_halving

#: The amendment's date.
AMENDMENT_DATE = date(2026, 10, 4)

D_OUT = r2.OUT_ROOT / "D"


def peak_mb(who: int = resource.RUSAGE_SELF) -> float:
    peak = resource.getrusage(who).ru_maxrss
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3


class Runner:
    """Runs plans on D under the development access, and keeps the clock."""

    def __init__(self, workers: int, access: round2.Access) -> None:
        self.workers, self.access = workers, access
        self.stages: dict[str, float] = {}
        self.worker_peak_mb: dict[str, float] = {}

    def run(
        self, stage: str, plans: Sequence[tuple[str, Sequence[date], Sequence[Spec], int]]
    ) -> pd.DataFrame:
        started = time.perf_counter()
        rows, _ = round2_run.run_plans(
            plans,
            access=self.access,
            book_roots=r2.BOOK_ROOTS,
            trades_root=r2.TRADES_ROOT,
            funding_root=r2.FUNDING_ROOT,
            out_root=D_OUT / stage,
            workers=self.workers,
            on_progress=r2.report,
        )
        self.stages[stage] = self.stages.get(stage, 0.0) + time.perf_counter() - started
        if len(rows) and "worker_peak_rss_mb" in rows:
            for symbol, part in rows.groupby("symbol"):
                peak = float(part["worker_peak_rss_mb"].max())
                self.worker_peak_mb[symbol] = max(self.worker_peak_mb.get(symbol, 0.0), peak)
        return rows


def per_job(prints_per_day: float) -> int:
    """Specs per job: fewer on the busiest instruments, so a worker's peak
    stays inside its budget."""
    if prints_per_day > 200_000:
        return 3
    if prints_per_day > 60_000:
        return 5
    return 8


def instrument_metrics(rows: pd.DataFrame) -> dict[str, float]:
    """S1's objective and its companions over one instrument's days."""
    ok = round2_stats.usable(rows)
    if ok.empty:
        return {"net_usd_day": math.nan, "trades": 0.0, "days": 0.0}
    return {
        "net_usd_day": float(ok["net"].mean()),
        "trades": float((ok["fills"] - ok["flattens"]).mean()),
        "days": float(len(ok)),
        "making_usd_day": float(ok["making"].mean()),
        "excluded_days": float(len(rows) - len(ok)),
    }


# ---------------------------------------------------------------------------
# S1, per instrument
# ---------------------------------------------------------------------------


def s1_specs(instrument: r2.Instrument, alive: Sequence[dict[str, float]]) -> list[Spec]:
    return [
        r2.spec(
            r2.s1_label(cell),
            r2.s1(instrument, cell),
            r2.config(instrument, cell["soft_limit_clips"]),
            "s1_search",
        )
        for cell in alive
    ]


def s1_candidates() -> list[dict[str, float]]:
    grid = round2.S1_GRID
    return [
        dict(zip(r2.S1_AXES, (float(v) for v in values), strict=True))
        for values in product(*grid.values())
    ]


def search_s1(
    instrument: r2.Instrument,
    subsets: dict[int, list[date]],
    runner: Runner,
    *,
    keep_fraction: float,
    jobs: int,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, Any]]:
    """One instrument's successive-halving run and its neighbourhood choice.

    Returns every evaluation, the chosen cell, and the record the amendment
    freezes: the chosen cell, its neighbours with their values, the median,
    the cell's own value and the outright peak at the full budget.
    """
    symbol = instrument.symbol

    def score_batch(alive: Sequence[dict[str, float]], budget: int) -> list[dict[str, float]]:
        plan = (symbol, subsets[budget], s1_specs(instrument, alive), jobs)
        rows = runner.run(f"s1_b{budget}", [plan])
        r2.report(f"  {symbol} S1: {len(alive)} cells x {budget} days scored")
        return [instrument_metrics(rows[rows["label"] == r2.s1_label(c)]) for c in alive]

    outcome = successive_halving(
        s1_candidates(),
        score_batch=score_batch,
        objective="net_usd_day",
        budgets=sorted(subsets),
        keep_fraction=keep_fraction,
        minimum_trades=r2.MIN_FILLS_S1,
        label=r2.s1_label,
    )
    table = outcome.table.copy()
    parsed = pd.DataFrame(
        [
            dict(item.split("=", 1) for item in str(c).removeprefix("S1|").split("/"))
            for c in table["candidate"]
        ]
    )
    axes = list(r2.S1_AXES)
    for axis in axes:
        table[axis] = pd.to_numeric(parsed[axis])
    table.insert(0, "symbol", symbol)
    top = max(subsets)
    full = table[(table["budget"] == top) & (table["trades"] >= r2.MIN_FILLS_S1)].copy()
    full = full.dropna(subset=["net_usd_day"]).reset_index(drop=True)
    if full.empty:
        raise SystemExit(f"{symbol}: no S1 cell reached {r2.MIN_FILLS_S1} fills a day")
    full["neighbourhood_usd_day"] = neighbourhood_scores(full, axes, "net_usd_day")
    ranked = full.sort_values(["neighbourhood_usd_day", "net_usd_day"], kind="stable")
    chosen_row = ranked.iloc[-1]
    peak_row = full.loc[full["net_usd_day"].idxmax()]
    chosen = {axis: float(chosen_row[axis]) for axis in axes}
    positions = {
        axis: full[axis].map({v: i for i, v in enumerate(sorted(full[axis].unique()))})
        for axis in axes
    }
    near = np.ones(len(full), dtype=bool)
    for axis in axes:
        near &= (positions[axis] - positions[axis][chosen_row.name]).abs().to_numpy() <= 1
    record = {
        "axes": axes,
        "chosen": [r2.plain(chosen_row[a]) for a in axes],
        "median": r2.sig(float(chosen_row["neighbourhood_usd_day"])),
        "own": r2.sig(float(chosen_row["net_usd_day"])),
        "cells": [
            [*(r2.plain(r[a]) for a in axes), r2.sig(float(r["net_usd_day"]))]
            for _, r in full[near].iterrows()
        ],
        "peak": [*(r2.plain(peak_row[a]) for a in axes), r2.sig(float(peak_row["net_usd_day"]))],
        "full_budget_cells": len(full),
    }
    table = table.merge(full[[*axes, "neighbourhood_usd_day"]], on=axes, how="left")
    return table, chosen, record


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def gate_specs(
    instrument: r2.Instrument,
    s1_cell: dict[str, float],
    cells: Sequence[gate.GateCell],
    markouts: dict[date, gate.PrintMarkouts],
    hourly: pd.DataFrame,
    tier: str = "base",
) -> list[Spec]:
    """G(base) for each cell on one instrument, the gate reading the fee paid."""
    fees = r2.tier(tier)
    settings = r2.config(instrument, s1_cell["soft_limit_clips"], fees)
    tag = "" if tier == "base" else f"|tier={tier}"
    out = []
    for cell in cells:
        tape = r2.state(markouts, cell, fees.maker_bp, hourly)
        quoter = r2.gated(cell, instrument, s1_cell, fees.maker_bp)
        out.append(r2.spec(f"G[{cell.label}]{tag}", quoter, settings, "gate", {cell.signal: tape}))
    return out


def base_specs(
    instrument: r2.Instrument, s1_cell: dict[str, float], tier: str = "base"
) -> list[Spec]:
    """The chosen S1 and its S1-touch cell, at a tier."""
    fees = r2.tier(tier)
    settings = r2.config(instrument, s1_cell["soft_limit_clips"], fees)
    tag = "" if tier == "base" else f"|tier={tier}"
    return [
        r2.spec(f"S1{tag}", r2.s1(instrument, s1_cell, fees.maker_bp), settings, "base"),
        r2.spec(
            f"S1-touch{tag}", r2.s1_touch(instrument, s1_cell, fees.maker_bp), settings, "base"
        ),
    ]


def gate_scores(
    rows: pd.DataFrame,
    cells: Sequence[gate.GateCell],
    clips: dict[str, float],
    tag: str = "",
) -> tuple[dict[gate.GateCell, float], dict[gate.GateCell, float], pd.DataFrame]:
    """Each cell's objective (the mean over D of the pooled daily metric) and
    its mean over instruments of passive fills a day."""
    values, fills, table = {}, {}, []
    for cell in cells:
        mine = rows[rows["label"] == f"G[{cell.label}]{tag}"]
        daily = round2_stats.pooled_daily(mine, clips)
        values[cell] = float(daily.mean()) if len(daily) else math.nan
        per_symbol = [float(r2.passive_fills(part).mean()) for _, part in mine.groupby("symbol")]
        fills[cell] = float(np.mean(per_symbol)) if per_symbol else 0.0
        ok = round2_stats.usable(mine)
        table.append(
            {
                **cell.record(),
                "label": cell.label,
                "per_100_clip_day": values[cell],
                "passive_fills_day": fills[cell],
                "net_usd_day_pooled_sum": float(round2_stats.usdt_daily(mine).mean()),
                "making_usd_day": float(ok["making"].mean()) if len(ok) else math.nan,
                "days": float(daily.size),
            }
        )
    return values, fills, pd.DataFrame(table)


def choose(
    values: dict[gate.GateCell, float], fills: dict[gate.GateCell, float], table: pd.DataFrame
) -> tuple[gate.GateCell, dict[str, Any], pd.DataFrame]:
    """The registered choice, and the table with each cell's neighbourhood median."""
    cell, record = round2_stats.choose_gate_cell(values, fills, min_fills=r2.MIN_FILLS_GATE)
    taking = {c: v for c, v in values.items() if fills[c] >= r2.MIN_FILLS_GATE}
    medians = round2_stats.neighbourhood_medians(taking)
    table = table.copy()
    table["takes_part"] = [fills[c] >= r2.MIN_FILLS_GATE for c in values]
    table["neighbourhood_median"] = [medians.get(c, math.nan) for c in values]
    table["chosen"] = [c == cell for c in values]
    return cell, record, table


# ---------------------------------------------------------------------------
# The stages before the searches
# ---------------------------------------------------------------------------


def check_funding(symbols: Sequence[str], days: Sequence[date]) -> None:
    for symbol in symbols:
        missing = [
            d for d in days if not (r2.FUNDING_ROOT / symbol / f"{d.isoformat()}.parquet").is_file()
        ]
        if missing:
            raise SystemExit(f"{symbol}: no funding file for D days {missing}; fetch stage 1 first")


def admission(access: round2.Access, workers: int) -> pd.DataFrame:
    table = round2_run.screen(
        round2.BASKET,
        access=access,
        book_roots=r2.BOOK_ROOTS,
        trades_root=r2.TRADES_ROOT,
        workers=workers,
        on_progress=r2.report,
    )
    return round2.admit(table)


def admission_rows(table: pd.DataFrame) -> list[dict[str, Any]]:
    """The admission table as the amendment freezes it."""
    rows = []
    for _, row in table.iterrows():
        rows.append(
            {
                "symbol": row["symbol"],
                "tick_bp": r2.sig(row["tick_bp"], 4),
                "spread_bp": r2.sig(row["spread_bp_time_weighted"], 4),
                "share_two_ticks": r2.sig(row["share_at_two_ticks_or_more"], 4),
                "touch_over_print": r2.sig(row["median_touch_over_median_print"], 4),
                "prints_per_day": round(float(row["prints_per_day"])),
                "markout_1s_bp": r2.sig(row["market_wide_passive_markout_bp_1s"], 4),
                "markout_5s_bp": r2.sig(row["market_wide_passive_markout_bp_5s"], 4),
                "markout_30s_bp": r2.sig(row["market_wide_passive_markout_bp_30s"], 4),
                "coverage_D": r2.sig(row["coverage_D"], 4),
                "coverage_H": r2.sig(row["coverage_H"], 4),
                "sequence_gap_days_D": int(row["sequence_gap_days"]),
                "criterion": str(row["criterion"]),
                "admitted": bool(row["admitted"]),
            }
        )
    return rows


def hour_tables(
    symbols: Sequence[str], access: round2.Access, d_days: Sequence[date], workers: int
) -> tuple[dict[str, dict[date, gate.PrintMarkouts]], dict[str, pd.DataFrame], pd.DataFrame]:
    """Every D print scored, per instrument; the hourly markouts; one table."""
    markouts, hourly, frames = {}, {}, []
    for symbol in symbols:
        days, missing = round2_run.markouts(
            symbol,
            d_days,
            access=access,
            book_roots=r2.BOOK_ROOTS,
            trades_root=r2.TRADES_ROOT,
            out=r2.MARKOUTS_ROOT,
            workers=workers,
        )
        if missing:
            r2.report(f"  {symbol}: {len(missing)} D days not scored: {sorted(missing)}")
        markouts[symbol] = days
        hourly[symbol] = gate.hourly_markouts(days.values())
        frame = hourly[symbol].copy()
        frame.insert(0, "symbol", symbol)
        frame["room_base_bp"] = frame["markout_bp"] - r2.tier("base").maker_bp
        for margin in gate.MARGINS_BP:
            frame[f"in_mask_mu{margin:g}"] = frame["hour"].isin(
                gate.hour_mask(hourly[symbol], r2.tier("base").maker_bp, margin)
            )
        frames.append(frame)
        r2.report(f"  {symbol}: {sum(len(d.ts) for d in days.values()):,} prints scored")
    return markouts, hourly, pd.concat(frames, ignore_index=True)


def mask_record(hourly: pd.DataFrame) -> dict[int, list[int]]:
    """The hour masks at the base fee, as the amendment freezes them."""
    return {int(m): list(hours) for m, hours in r2.masks(hourly, r2.tier("base").maker_bp).items()}


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def labelled(rows: pd.DataFrame, label: str) -> pd.DataFrame:
    return rows[rows["label"] == label]


def b3_stage(
    runner: Runner,
    order: Sequence[str],
    instruments: dict[str, r2.Instrument],
    s1_chosen: dict[str, dict[str, float]],
    cell: gate.GateCell,
    markouts: dict[str, dict[date, gate.PrintMarkouts]],
    hourly: dict[str, pd.DataFrame],
    jobs: dict[str, int],
    d_days: Sequence[date],
) -> tuple[dict[str, Any], dict[str, pd.Series], pd.DataFrame]:
    """Each instrument's S1 and the chosen G(base) at each professional tier,
    every value frozen; the one with the higher pooled mean on D is B3's."""
    clips = {s: instruments[s].clip for s in order}
    choice: dict[str, Any] = {}
    daily: dict[str, pd.Series] = {}
    frames = []
    for name in ("pro1_altcoin", "programme"):
        plans = []
        for s in order:
            inst = instruments[s]
            specs = base_specs(inst, s1_chosen[s], name)[:1]
            specs += gate_specs(inst, s1_chosen[s], [cell], markouts[s], hourly[s], name)
            plans.append((s, d_days, specs, jobs[s]))
        rows = runner.run(f"b3_{name}", plans)
        frames.append(rows)
        s1_daily = round2_stats.pooled_daily(labelled(rows, f"S1|tier={name}"), clips)
        g_daily = round2_stats.pooled_daily(labelled(rows, f"G[{cell.label}]|tier={name}"), clips)
        pick = "G" if float(g_daily.mean()) > float(s1_daily.mean()) else "S1"
        choice[name] = {
            "strategy": pick,
            "S1_per_100_clip_day": r2.sig(float(s1_daily.mean())),
            "G_per_100_clip_day": r2.sig(float(g_daily.mean())),
        }
        daily[name] = g_daily if pick == "G" else s1_daily
        r2.report(f"  B3 at {name}: S1 {s1_daily.mean():+.4f}, G {g_daily.mean():+.4f}: {pick}")
    return choice, daily, pd.concat(frames, ignore_index=True)


def power_row(name: str, primary: str, claim: str, daily: pd.Series) -> dict[str, Any]:
    stats = round2_stats.mde(daily, round2.PRIMARIES[primary])
    return {
        "quantity": name,
        "primary": primary,
        "claim": claim,
        "unit": "USDT per 100 USDT of clip a day",
        "days": int(stats["days"]),
        "mean_D": float(np.nanmean(daily)) if len(daily) else math.nan,
        "sigma_D": stats["sigma_d"],
        "n_heldout": round2.PRIMARIES[primary],
        "mde": stats["mde"],
    }


def mde_values(power: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    """The minimum detectable effects as the amendment freezes them."""
    sigma: dict[str, Any] = {}
    value: dict[str, Any] = {}
    for primary, part in power.groupby("primary", sort=False):
        if len(part) == 1:
            sigma[primary] = r2.sig(float(part["sigma_D"].iloc[0]))
            value[primary] = r2.sig(float(part["mde"].iloc[0]))
        else:
            sigma[primary] = {
                c: r2.sig(float(s)) for c, s in zip(part["claim"], part["sigma_D"], strict=True)
            }
            value[primary] = {
                c: r2.sig(float(m)) for c, m in zip(part["claim"], part["mde"], strict=True)
            }
    return sigma, value


def daily_table(frames: dict[str, pd.DataFrame], clips: dict[str, float]) -> pd.DataFrame:
    """The chosen strategies' rows on every D day, with the per-clip metric."""
    keep = [
        "symbol", "day", "label", "status", "excluded", "flags", "net", "making", "spread",
        "adverse", "inventory", "fees", "funding", "fills", "flattens", "max_abs_position",
        "maker_turnover", "taker_turnover", "clip_over_touch", "bid_quoted_s", "ask_quoted_s",
    ]  # fmt: skip
    out = []
    for strategy, rows in frames.items():
        frame = rows[[c for c in keep if c in rows.columns]].copy()
        frame.insert(0, "strategy", strategy)
        frame["per_100_clip"] = frame["net"] * 100.0 / frame["symbol"].map(clips)
        out.append(frame)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--freeze", action="store_true", help="write the frozen values")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise SystemExit("between 1 and 8 workers")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    started = time.perf_counter()
    registration = round2.Registration.load().require_consistent()
    if registration.frozen:
        raise SystemExit("the registration is already frozen; the development block is done")
    access = round2.development_access()
    d_days = round2.block_days("D")
    search = registration.value(("d_search", "S1"))
    budgets = [int(b) for b in search["budgets_days"]]
    runner = Runner(args.workers, access)
    RESULTS.mkdir(parents=True, exist_ok=True)
    bico = round2.B4_SYMBOL

    # 1. admission ------------------------------------------------------------
    t = time.perf_counter()
    r2.report("screen: D statistics of the eight candidates, H coverage from the file system")
    table = admission(access, args.workers)
    table.to_csv(RESULTS / "mm_round2_admission_D.csv", index=False)
    runner.stages["screen"] = time.perf_counter() - t
    busy = dict(zip(table["symbol"], table["prints_per_day"], strict=True))
    admitted = [s for s, ok in zip(table["symbol"], table["admitted"], strict=True) if ok]
    order = sorted(admitted, key=lambda s: -float(busy[s]))
    r2.report(f"  admitted: {order or 'none'}")
    instruments = {
        row["symbol"]: r2.Instrument(
            row["symbol"], r2.sig(row["clip_notional_usdt"]), r2.sig(row["sigma_ref_bp_1m"])
        )
        for _, row in table.iterrows()
        if row["admitted"]
    }
    instruments[bico] = r2.Instrument(bico, round2.BICO_CLIP_USDT, round2.BICO_SIGMA_REF)
    clips = {s: i.clip for s, i in instruments.items()}
    jobs = {s: per_job(float(busy[s])) for s in order} | {bico: 8}
    check_funding([*order, bico], d_days)

    # 2. the prints, scored, and the hour masks -------------------------------------
    t = time.perf_counter()
    r2.report("prints: every D print scored from its resting side at 5 s")
    markouts, hourly, hours = hour_tables([*order, bico], access, d_days, args.workers)
    hours.to_csv(RESULTS / "mm_round2_hours_D.csv", index=False)
    runner.stages["prints"] = time.perf_counter() - t
    cells = gate.gate_cells()
    frames: dict[str, pd.DataFrame] = {}
    power: list[dict[str, Any]] = []
    s1_chosen: dict[str, dict[str, float]] = {}
    s1_records: dict[str, Any] = {}
    gate_cell: gate.GateCell | None = None
    gate_record: Any = round2.NOT_TESTED
    b3_choice: Any = round2.NOT_TESTED

    if order:
        # 3. S1, per instrument -------------------------------------------------
        subsets = r2.nested_days(d_days, budgets, int(search["day_subset_seed"]))
        pd.DataFrame(
            [
                {"budget": b, "days": " ".join(d.isoformat() for d in ds)}
                for b, ds in subsets.items()
            ]
        ).to_csv(RESULTS / "mm_round2_search_days_D.csv", index=False)
        first = budgets[0]
        r2.report(f"search S1: the first rung on every instrument ({first} days)")
        runner.run(
            f"s1_b{first}",
            [
                (s, subsets[first], s1_specs(instruments[s], s1_candidates()), jobs[s])
                for s in order
            ],
        )
        tables = []
        for s in order:
            found, s1_chosen[s], s1_records[s] = search_s1(
                instruments[s],
                subsets,
                runner,
                keep_fraction=float(search["keep_fraction"]),
                jobs=jobs[s],
            )
            tables.append(found)
            r2.report(f"  {s} S1 chosen {s1_records[s]['chosen']}, peak {s1_records[s]['peak']}")
        pd.concat(tables, ignore_index=True).to_csv(
            RESULTS / "mm_round2_search_S1_D.csv", index=False
        )

        # 4. the gate, and its bases, on every D day --------------------------------
        r2.report(f"gate: {len(cells)} cells and both bases on {len(d_days)} days")
        plans = []
        for s in order:
            inst = instruments[s]
            specs = base_specs(inst, s1_chosen[s])
            specs += gate_specs(inst, s1_chosen[s], cells, markouts[s], hourly[s])
            plans.append((s, d_days, specs, jobs[s]))
        rows = runner.run("gate", plans)
        values, fills, gate_table = gate_scores(rows, cells, clips)
        gate_cell, gate_record, gate_table = choose(values, fills, gate_table)
        gate_table.to_csv(RESULTS / "mm_round2_gate_D.csv", index=False)
        r2.report(f"  gate chosen {gate_cell.label}: median {gate_record['median']}")
        base_label = "S1" if gate_cell.base == "S1" else "S1-touch"
        frames["S1"] = labelled(rows, "S1")
        frames["S1-touch"] = labelled(rows, "S1-touch")
        frames["G"] = labelled(rows, f"G[{gate_cell.label}]")
        s1_daily = round2_stats.pooled_daily(frames["S1"], clips)
        g_daily = round2_stats.pooled_daily(frames["G"], clips)
        paired = round2_stats.paired_daily(frames["G"], labelled(rows, base_label), clips)
        power.append(power_row("S1 pooled daily net", "B1", "net", s1_daily))
        power.append(power_row("G(base) pooled daily net", "B2", "net", g_daily))
        power.append(power_row(f"G(base) - {base_label}, paired", "B2", "minus_base", paired))

        # 5. B3's strategy at each tier -----------------------------------------------
        r2.report("B3: S1 and the chosen G(base) at the professional tiers")
        b3_choice, b3_daily, b3_rows = b3_stage(
            runner, order, instruments, s1_chosen, gate_cell, markouts, hourly, jobs, d_days
        )
        for name, primary in (("pro1_altcoin", "B3a"), ("programme", "B3b")):
            pick = b3_choice[name]["strategy"]
            label = f"S1|tier={name}" if pick == "S1" else f"G[{gate_cell.label}]|tier={name}"
            frames[f"B3 {name}"] = labelled(b3_rows, label)
            power.append(
                power_row(f"{pick} at {name}, pooled daily net", primary, "net", b3_daily[name])
            )

    # 6. B4 on BICOUSDT's D -------------------------------------------------------
    r2.report(f"B4: {len(cells)} cells on BICOUSDT's D")
    inst = instruments[bico]
    bases = r2.s1_cell(round2.BICO_BASES["S1"])
    specs = base_specs(inst, bases) + gate_specs(inst, bases, cells, markouts[bico], hourly[bico])
    rows_b4 = runner.run("b4", [(bico, d_days, specs, jobs[bico])])
    values, fills, b4_table = gate_scores(rows_b4, cells, {bico: inst.clip})
    b4_cell, b4_record, b4_table = choose(values, fills, b4_table)
    b4_table.to_csv(RESULTS / "mm_round2_b4_gate_D.csv", index=False)
    r2.report(f"  B4 chosen {b4_cell.label}: median {b4_record['median']}")
    b4_base = "S1" if b4_cell.base == "S1" else "S1-touch"
    frames["B4 G"] = labelled(rows_b4, f"G[{b4_cell.label}]")
    frames["B4 base"] = labelled(rows_b4, b4_base)
    power.append(
        power_row(
            "B4 G(base) daily net", "B4", "net", round2_stats.pooled_daily(frames["B4 G"], clips)
        )
    )
    power.append(
        power_row(
            f"B4 G(base) - {b4_base}, paired",
            "B4",
            "minus_base",
            round2_stats.paired_daily(frames["B4 G"], frames["B4 base"], clips),
        )
    )

    # 7. power, the tables and the frozen values ------------------------------------
    power_table = pd.DataFrame(power)
    power_table.to_csv(RESULTS / "mm_round2_power_D.csv", index=False)
    daily = daily_table(frames, clips)
    daily.to_csv(RESULTS / "mm_round2_daily_D.csv", index=False)
    exclusions = (
        daily.groupby(["strategy", "symbol"])
        .agg(days=("day", "size"), excluded=("excluded", "sum"))
        .reset_index()
    )
    exclusions.to_csv(RESULTS / "mm_round2_exclusions_D.csv", index=False)
    sigma, mde = mde_values(power_table)
    values_out: dict[tuple[str, ...], Any] = {
        ("admission", "table"): admission_rows(table),
        ("simulator", "clip", "notional_usdt"): {s: instruments[s].clip for s in order},
        ("simulator", "volatility", "sigma_ref"): {s: instruments[s].sigma_ref for s in order},
        ("strategies", "S1", "chosen"): {s: s1_records[s]["chosen"] for s in order},
        ("d_search", "S1", "neighbourhood"): {s: s1_records[s] for s in order},
        ("d_search", "gate", "chosen"): gate_cell.record() if gate_cell else round2.NOT_TESTED,
        ("d_search", "gate", "chosen_neighbourhood"): gate_record,
        ("d_search", "gate", "hours_masks"): {s: mask_record(hourly[s]) for s in order},
        ("d_search", "b3_choice", "chosen"): b3_choice,
        ("d_search", "b4", "chosen"): b4_cell.record(),
        ("d_search", "b4", "chosen_neighbourhood"): b4_record,
        ("d_search", "b4", "hours_masks"): mask_record(hourly[bico]),
        ("metrics", "minimum_detectable_effect", "sigma_d"): sigma,
        ("metrics", "minimum_detectable_effect", "value"): mde,
        ("amendment", "date"): AMENDMENT_DATE,
    }
    runner.stages["total"] = time.perf_counter() - started
    compute = [{"stage": k, "value": round(v, 1), "unit": "s"} for k, v in runner.stages.items()]
    compute += [
        {"stage": f"peak RSS, largest worker, {s}", "value": round(v), "unit": "MB"}
        for s, v in sorted(runner.worker_peak_mb.items())
    ]
    compute += [
        {
            "stage": "peak RSS, any worker process",
            "value": round(peak_mb(resource.RUSAGE_CHILDREN)),
            "unit": "MB",
        },
        {"stage": "peak RSS, driver", "value": round(peak_mb()), "unit": "MB"},
        {"stage": "workers", "value": args.workers, "unit": ""},
    ]
    pd.DataFrame(compute).to_csv(RESULTS / "mm_round2_compute_D.csv", index=False)
    frozen = {".".join(k): v for k, v in values_out.items()}
    D_OUT.mkdir(parents=True, exist_ok=True)
    (D_OUT / "freeze_D.json").write_text(json.dumps(frozen, indent=2, default=str) + "\n")
    r2.report(json.dumps(frozen, indent=1, default=str))
    if args.freeze:
        path = round2.CONFIG_PATH
        text = round2.freeze_text(path.read_text(encoding="utf-8"), values_out)
        path.write_text(text, encoding="utf-8")
        r2.report(f"frozen: {path} sha256 {round2.Registration.from_text(text).sha256}")


if __name__ == "__main__":
    main()
