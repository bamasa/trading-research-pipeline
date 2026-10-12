"""Stage D: the choice, the placebo search with the refit, carry-forward and admission.

On the run's rows of the trial ledger, over D's usable test days (stage 1's
usable days that stage B scored without a stale walk), in the catalogue's
byte order so that every tie goes to the key first in it:

* **the choice** per stratum (``validation.choice.choose``): S, the five-trade
  filter, the outright peak; A of the chosen cell;
* **Null A** (hour-block sign flips) on the trades stage B kept, and **Null B**
  (day re-pairing over every shift) through the walk-table route, its target
  days filled one at a time on disk so a stopped run resumes;
* **the refit replication**: every boosted and logistic fit of every fold redone
  on training labels shifted circularly, stratum C re-scored and re-chosen,
  giving the A that C* must beat;
* **admission**: carry-forward (iii), the placebo search (iv) at alpha / R_D,
  and the slot conditions (i) and (ii) for S-C and S-R; S-P's and S-M's wait
  for the simulator (stage C);
* **reported, gating nothing**: p_search, the global flip, the selection
  premium, PBO per stratum, the deflated Sharpe ratios, N_trials and N_eff,
  and S25 against both nulls with its cell fixed;
* the **leaderboard**, **marginals** and **nulls** files.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd

from trading_research.pipeline.short_horizon.context import Study
from trading_research.pipeline.short_horizon.inputs import Inputs
from trading_research.reporting.search_keys import S25_KEY
from trading_research.search_engine.catalogue import (
    GROUP_COLUMNS,
    ORDERED_AXES,
    axes_table,
    parse_key,
)
from trading_research.search_engine.ledger import TrialLedger
from trading_research.validation.choice import CellDays, neighbourhood_index

SLOT_OF = {"C": "S-C", "R": "S-R"}


@dataclass
class Grid:
    """The cells scored, as the choice reads them: their table, neighbourhoods and strata."""

    keys: list[str]

    @cached_property
    def table(self) -> pd.DataFrame:
        return axes_table(self.keys)

    @cached_property
    def index(self) -> np.ndarray:
        return neighbourhood_index(self.table, ORDERED_AXES, by=GROUP_COLUMNS)

    @cached_property
    def strata(self) -> dict[str, np.ndarray]:
        stratum = self.table["stratum"].to_numpy()
        return {name: np.flatnonzero(stratum == name) for name in ("C", "R")}

    @cached_property
    def holds_ns(self) -> np.ndarray:
        return np.array([round(parse_key(k).hold_s * 1e9) for k in self.keys], dtype=np.int64)

    def position(self, key: str) -> int | None:
        return self.keys.index(key) if key in self.keys else None

    def neighbours(self, cell: int) -> list[str]:
        return [self.keys[i] for i in self.index[cell] if i >= 0]


@dataclass
class Calendar:
    """D's test days in calendar order (Null B's circle) and which are usable."""

    days: list[date]
    usable: np.ndarray

    @property
    def usable_days(self) -> list[date]:
        return [d for d, ok in zip(self.days, self.usable, strict=True) if ok]

    @classmethod
    def of(cls, study: Study, inputs: Inputs) -> Calendar:
        planned = [d for d in study.scale.days("D") if d.month >= 2]
        scored = set(study.record("B")["test_days"])
        stale = set(study.record("B")["stale_days"])
        good = set(inputs.usable("D"))
        usable = np.array(
            [d in good and d.isoformat() in scored and d.isoformat() not in stale for d in planned]
        )
        return cls(planned, usable)


def observed_cells(study: Study, grid: Grid, calendar: Calendar) -> CellDays:
    """The run's ledger rows on the usable test days, as the choice reads them."""
    rows = TrialLedger.in_repository(study.repo, study.catalogue()).run_rows(study.run)
    days = calendar.usable_days
    rows = rows[rows["day"].isin(days) & rows["key"].isin(grid.keys)].reset_index(drop=True)
    return CellDays.from_ledger(rows, grid.keys, days)


def choice_record(grid: Grid, cells: CellDays) -> dict[str, Any]:
    """Each stratum's chosen cell, its S and neighbourhood with their D values, and the peak."""
    from trading_research.validation.choice import cell_means, choose, statistic_a

    own = cell_means(cells.net_bp)
    out: dict[str, Any] = {}
    for name, members in grid.strata.items():
        found = choose(cells, members, grid.index)
        entry: dict[str, Any] = {"cell": None, "peak": None}
        if found.peak is not None:
            entry["peak"] = {"key": grid.keys[found.peak], "S": float(found.s[found.peak])}
        if found.cell is not None:
            near = [int(i) for i in grid.index[found.cell] if i >= 0]
            entry["cell"] = grid.keys[found.cell]
            entry["S"] = float(found.s[found.cell])
            entry["A"] = statistic_a(cells, grid.index, found.cell)
            entry["neighbourhood"] = {grid.keys[i]: {"mean_net_bp": float(own[i]),
                                      "trades_per_day": float(cells.trades[i].sum() / max(cells.days, 1))}
                                      for i in near}  # fmt: skip
        out[f"{name}*"] = entry
    return out


def hour_cells(study: Study, grid: Grid, calendar: Calendar) -> Any:
    """Null A's hour sums from the trades stage B kept on the usable test days."""
    from trading_research.pipeline.short_horizon.scoring import day_frames
    from trading_research.validation.placebo_search import HourCells

    trades = day_frames(study.work / "B" / "days", calendar.usable_days, "trades.parquet")
    trades = trades[trades["key"].isin(grid.keys)].reset_index(drop=True)
    return HourCells.from_trades(trades, grid.keys, calendar.usable_days)


def null_b_route(study: Study, inputs: Inputs, grid: Grid, calendar: Calendar) -> Any:
    """Null B's route, every usable target day filled (resumable on disk)."""
    from trading_research.pipeline.short_horizon.scoring import day_tables
    from trading_research.search_engine.null_b import Entries, NullBRoute

    shifts = study.scale.null_b_shifts(len(calendar.days))
    route = NullBRoute(grid.keys, grid.holds_ns.tolist(), calendar.usable, shifts=shifts,
                       directory=study.dir("D", "null_b"))  # fmt: skip
    positions = {key: i for i, key in enumerate(grid.keys)}
    scored_keys = list(study.keys)
    for at, day in enumerate(calendar.days):
        if not calendar.usable[at]:
            continue
        frame = pd.read_parquet(study.work / "B" / "days" / day.isoformat() / "entries.parquet")
        mapping = {}
        for cell, rows in frame.groupby("cell", sort=True):
            key = scored_keys[int(str(cell))]
            if key in positions:
                mapping[positions[key]] = Entries(
                    rows["tod_ns"].to_numpy(np.int64), rows["side"].to_numpy(np.int64)
                )
        route.add_source_day(at, mapping)
    holds = sorted({parse_key(k).hold_s for k in grid.keys})
    for at, day in enumerate(calendar.days):
        if calendar.usable[at] and not route.done[at]:
            route.add_target_day(at, day_tables(inputs, day, holds))
    return route


def refit_cells(study: Study, inputs: Inputs, grid: Grid, calendar: Calendar) -> CellDays:
    """Stratum C re-scored after every boosted and logistic fit is redone on labels
    shifted circularly within each fold's training span (all of C's cells)."""
    from trading_research.data import study_access as sa
    from trading_research.pipeline.short_horizon import flags
    from trading_research.pipeline.short_horizon.fits import configurations, run_fold
    from trading_research.pipeline.short_horizon.plan import walk_forward
    from trading_research.pipeline.short_horizon.scoring import day_frames, score_days

    keys_c = [k for k in grid.keys if parse_key(k).stratum == "C"]
    root = study.dir("D", "refit")
    close = flags.hourly_close(
        study.symbol, study.roots.bars, sa.BLOCKS["D"][1], sa.development_access()
    )
    configs = configurations(keys_c)
    for fold in walk_forward(inputs.usable("D")):
        run_fold(inputs, fold, configs, close, root, shift_days=study.scale.refit_shift_days)
    score_days(inputs, keys_c, study.run, root / "tapes", root / "folds", root / "days")
    rows = day_frames(root / "days", calendar.usable_days, "ledger.parquet")
    return CellDays.from_ledger(rows, grid.keys, calendar.usable_days)


def daily_ic(study: Study, grid: Grid) -> np.ndarray:
    """Each cell's mean daily IC: its configuration's, from stage A's folds."""
    import json

    by_configuration: dict[str, list[float]] = {}
    for path in sorted((study.work / "A" / "folds").glob("*.json")):
        for name, days in json.loads(path.read_text(encoding="utf-8"))["ic"].items():
            by_configuration.setdefault(name, []).extend(v for v in days.values() if v == v)
    means = {
        name: float(np.mean(values)) if values else float("nan")
        for name, values in by_configuration.items()
    }
    return np.array([means.get(parse_key(k).configuration, float("nan")) for k in grid.keys])


def d_status(
    stratum: str,
    choice: dict[str, Any],
    conditions: Any,
    carry: Any,
    nulls: Any,
    hand_runs: list[dict[str, Any]],
) -> str:
    """A searched slot's status set on D, or ``admitted``."""
    from trading_research.validation.choice import not_admitted

    if hand_runs:
        return "not admitted on D (hand run)"
    if choice[f"{stratum}*"]["cell"] is None:
        return "not run (no admissible cell on D)"
    if choice[f"{stratum}*"]["cell"] == S25_KEY:
        return "same as S25"
    failed = [name for name, ok in (("i", conditions.i), ("ii", conditions.ii),
              ("iii", carry.passes), ("iv", nulls.passes)) if not ok]  # fmt: skip
    return not_admitted(failed) if failed else "admitted"


def overfitting(
    grid: Grid, cells: CellDays, chosen: dict[str, int | None], n_trials: int
) -> dict[str, Any]:
    """PBO per stratum, N_eff, and the deflated Sharpe ratios of the chosen cells and
    of S25's fixed cell (S-P's and S-M's are stage C's)."""
    from trading_research.evaluation.overfitting import (
        deflated_sharpe_ratio,
        effective_trials,
        sharpe_variance,
    )
    from trading_research.validation.choice import ChoiceError, pbo_of_choice

    panel = pd.DataFrame(cells.g_bp.T)
    out: dict[str, Any] = {"N_eff": effective_trials(n_trials, panel), "pbo": {}, "dsr": {}}
    variance = sharpe_variance(panel)
    for name, members in grid.strata.items():
        try:
            out["pbo"][name] = pbo_of_choice(cells, members, grid.index).pbo
        except (ChoiceError, ValueError) as exc:
            out["pbo"][name] = f"not computed: {exc}"
        cell = chosen.get(name)
        if cell is None:
            continue
        for series, values in (("G", cells.g_bp[cell]), ("net", cells.net_bp[cell])):
            finite = values[np.isfinite(values)]
            if len(finite) >= 3 and np.isfinite(variance):
                out["dsr"][f"{name}*/{series}"] = deflated_sharpe_ratio(
                    finite, n_trials, variance
                ).dsr
    s25 = grid.position(S25_KEY)
    for series, values in (
        () if s25 is None else (("G", cells.g_bp[s25]), ("net", cells.net_bp[s25]))
    ):
        finite = values[np.isfinite(values)]
        if len(finite) >= 3 and np.isfinite(variance):
            out["dsr"][f"S25/{series}"] = deflated_sharpe_ratio(finite, n_trials, variance).dsr
    return out


def power(
    cells: CellDays, choice: dict[str, Any], carry: dict[str, Any], grid: Grid
) -> dict[str, Any]:
    """For each searched slot's chosen cell: sigma_D of its daily net on D's usable
    test days, n_eff (44 x the share of them with a trade), the MDE, and the power
    of the t >= 3 bar against the carry-forward estimate of its net."""
    from trading_research.evaluation.significance import minimum_detectable_effect, power_at_bar

    out: dict[str, Any] = {}
    for name in grid.strata:
        key = choice[f"{name}*"]["cell"]
        if key is None:
            continue
        values = cells.net_bp[grid.keys.index(key)]
        traded = cells.trades[grid.keys.index(key)] > 0
        finite = values[np.isfinite(values) & traded]
        sigma = float(finite.std(ddof=1)) if len(finite) >= 2 else float("nan")
        n_eff = 44.0 * float(traded.mean()) if len(traded) else 0.0
        effect = float(carry[name].mean_net)
        out[SLOT_OF[name]] = {"cell": key, "sigma_D": sigma, "n_eff": n_eff,
                              "mde": minimum_detectable_effect(sigma, n_eff),
                              "carry_forward_net": effect, "power": power_at_bar(effect, sigma, n_eff)}  # fmt: skip
    return out


def lacking_a_class(study: Study) -> list[dict[str, Any]]:
    """Every stage-A fit (fold, configuration) whose training labels lacked a class,
    with the side it lacked (sell: short, buy: long), or that had too few to fit."""
    import json

    from trading_research.labels.directional import CLASS_NAMES

    sides = {-1: "short", 0: "hold", 1: "long"}
    out: list[dict[str, Any]] = []
    for path in sorted((study.work / "A" / "folds").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        for name, entry in sorted(record.get("fits", {}).items()):
            if "|C|" not in name:  # stratum C's classifiers only
                continue
            present = entry.get("classes_present")
            if present is None:  # not fitted: too few rows or classes, which is not recorded
                out.append({"fold": path.stem, "configuration": name, "note": entry.get("note", ""),
                            "missing": None, "side": None})  # fmt: skip
                continue
            missing = sorted(set(sides) - set(present))
            if missing:
                out.append({"fold": path.stem, "configuration": name, "note": "",
                            "missing": [CLASS_NAMES[c] for c in missing],
                            "side": [sides[c] for c in missing]})  # fmt: skip
    return out


def write_reports(
    study: Study,
    grid: Grid,
    cells: CellDays,
    tables: list[pd.DataFrame],
    n_trials: int,
    days: list[date],
) -> dict[str, str]:
    """The leaderboard (with each cell's gross over the usable test days ``days``),
    marginals and nulls files; their sha256s."""
    from trading_research.pipeline.short_horizon.models import file_sha256
    from trading_research.pipeline.short_horizon.scoring import gross_bp
    from trading_research.reporting.leaderboard import leaderboard, write_leaderboard
    from trading_research.reporting.marginals import marginals, write_marginals
    from trading_research.reporting.nulls import write_nulls
    from trading_research.reporting.search_keys import catalogue_table

    table = catalogue_table(grid.keys)
    gross = gross_bp(study.work / "B" / "days", days, grid.keys)
    board = leaderboard(table, cells, grid.index, gross_bp=gross, ic=daily_ic(study, grid))
    planned = len(study.catalogue())
    paths = {
        "leaderboard": write_leaderboard(board, study.results("leaderboard.csv"), planned=planned,
                                         scored=len(grid.keys), skipped=planned - len(grid.keys),
                                         n_trials=n_trials),
        "marginals": write_marginals(marginals(board, table), study.results("marginals.csv")),
        "nulls": write_nulls(tables, study.results("nulls.csv")),
    }  # fmt: skip
    return {name: file_sha256(path) for name, path in paths.items()}


def run(study: Study) -> dict[str, Any]:
    """Stage D's record: the choice, both nulls, the refit, admission and what is reported."""
    from trading_research.reporting.nulls import nulls_table, s25_nulls_table
    from trading_research.validation import choice as ch
    from trading_research.validation import placebo_search as ps

    inputs = Inputs(study)
    grid = Grid(list(study.keys))
    calendar = Calendar.of(study, inputs)
    cells = observed_cells(study, grid, calendar)
    days = calendar.usable_days
    choice = choice_record(grid, cells)
    observed = ch.rerun(cells, grid.strata, grid.index)
    hours = hour_cells(study, grid, calendar)
    draws = study.scale.null_a_draws
    a_null = ps.null_a(hours, grid.strata, grid.index, draws=draws)
    route = null_b_route(study, inputs, grid, calendar)
    b_null = ps.null_b(route.shifted, grid.strata, grid.index, shifts=route.shifts)
    refitted = refit_cells(study, inputs, grid, calendar)
    shift = study.scale.refit_shift_days
    a_refit = ps.refit_replication(lambda _: refitted, grid.strata["C"], grid.index, days=shift)
    runs = study.run
    results = ps.admission_iv(observed, a_null, b_null, runs=runs, refit={"C": a_refit.a})
    carry = {s: ch.carry_forward(cells, days, m, grid.index) for s, m in grid.strata.items()}
    conditions: dict[str, Any] = {}
    for s, outcome in observed.items():
        found = outcome.cell
        conditions[s] = (
            None if found is None else ch.slot_conditions(cells, days, grid.index, found)
        )
    flip = ps.global_flip(hours, grid.strata, grid.index)
    ledger = TrialLedger.in_repository(study.repo, study.catalogue())
    manifest, hand = ledger.record(), ledger.hand_runs
    n_trials = int(manifest["n_trials"])
    tables = [nulls_table(results, a_null, b_null, grid.keys, runs=runs, shifts=list(route.shifts),
                          global_flip=flip)]  # fmt: skip
    s25 = grid.position(S25_KEY)
    s25_record: dict[str, Any] = {"in_scored_cells": s25 is not None}
    if s25 is not None:
        fixed = {"C": grid.strata["C"]}
        a_s25 = ps.null_a(hours, fixed, grid.index, draws=draws, fixed={"C": s25})
        b_s25 = ps.null_b(route.shifted, fixed, grid.index, shifts=route.shifts, fixed={"C": s25})
        a_obs = ch.statistic_a(cells, grid.index, s25)
        tables.append(s25_nulls_table(a_obs, a_s25, b_s25, shifts=list(route.shifts)))
        s25_record |= {"A": a_obs,
                       "p_A": ps.placebo_p(ps.exceedances(a_s25.maximum, a_obs), a_s25.replications),
                       "p_B": ps.placebo_p(ps.exceedances(b_s25.maximum, a_obs), b_s25.replications)}  # fmt: skip
    files = write_reports(study, grid, cells, tables, n_trials, days)
    statuses = {
        SLOT_OF[s]: d_status(s, choice, conditions[s], carry[s], results[s], hand)
        for s in grid.strata
    }
    return {
        "commit": study.commit(), "R_D": runs, "test_days": [d.isoformat() for d in calendar.days],
        "usable_test_days": [d.isoformat() for d in days], "choice": choice,
        "nulls": {s: {"A": r.a, "count_A": r.count_a, "count_B": r.count_b, "p_A": r.p_a, "p_B": r.p_b,
                      "A_refit": r.a_refit, "passes_iv": r.passes} for s, r in results.items()},
        "p_search": ps.p_search(results, runs),
        "null_A": {"draws": a_null.replications, "seeds": [0, a_null.replications - 1]},
        "null_B": {"shifts": [route.shifts[0], route.shifts[-1]], "replications": b_null.replications,
                   "busy_skipped": int(np.asarray(route.busy).sum()),
                   "excluded_pairs": {str(s): route.excluded(s) for s in route.shifts}},
        "refit": {"A": a_refit.a, "cell": None if a_refit.cell is None else grid.keys[a_refit.cell],
                  "shift_days": shift},
        "carry_forward": {s: {"months": list(c.months), "cells": [None if i is None else grid.keys[i] for i in c.cells],
                              "g": list(c.g), "net": list(c.net), "mean_g": c.mean_g, "mean_net": c.mean_net,
                              "passes_iii": c.passes} for s, c in carry.items()},
        "conditions": {s: None if c is None else {"S": c.s, "monthly": list(c.monthly), "months_positive":
                       c.months_positive, "i": c.i, "ii": c.ii} for s, c in conditions.items()},
        "signal_admitted": {s: bool(carry[s].passes and results[s].passes and not hand) for s in grid.strata},
        "statuses_on_D": statuses,
        "global_flip": {s: {"cell": None if o.cell is None else grid.keys[o.cell], "A": o.a} for s, o in flip.items()},
        "selection_premium": {s: {"A": list(ps.selection_premium(r.a, a_null.maximum)),
                                  "B": list(ps.selection_premium(r.a, b_null.maximum))} for s, r in results.items()},
        "overfitting": overfitting(grid, cells, {s: o.cell for s, o in observed.items()}, n_trials),
        "power_at_bar": power(cells, choice, carry, grid), "fits_lacking_a_class": lacking_a_class(study),
        "N_trials": n_trials, "ledger_sha256": manifest["sha256"], "ledger_rows": int(manifest["rows"]),
        "hand_runs": hand, "S25": s25_record, "files_sha256": files,
    }  # fmt: skip
