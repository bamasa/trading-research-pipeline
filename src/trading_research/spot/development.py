"""The D procedure: the fold-level leakage test, the fits, scoring on June and July, sigma_D.

Registered (steps 2 to 5 of the development-period procedure; "Nothing is
searched"):

* **The fits** of each fold (:func:`trading_research.spot.fits.fit_fold`):
  June, July and the frozen fits, every hold and label fee of S25-PF's and
  S25-BB's neighbourhoods, S25-PB's and S25-PFT's own cell; S25-T is the 2025
  study's frozen fit, its files checked by sha256 here as at the H and P gates.
* **The fold-level leakage test** (K-leak) on June and July for S25-PF and
  S25-BB: the fit path (training rows, purge, fit, thresholds) run on the stores
  of the fold's days and again with every input and label after the fold's cutoff
  mutated (``validation/fold_leakage``); weights and theta must keep their sha256.
  The frozen fits' cutoff is D's end, after which no day is loaded.
* **Scoring**: each fold's fits on its test month, every row and cell through
  :mod:`.heldout`'s read (placebos, neighbours, latency, brackets, calibration),
  with Null A's hour-block sign flips on the own cell.
* **sigma_D and the MDE** of each slot of the family over the 61 test days:
  ``(3 + 0.84) x sigma_D / sqrt(n_eff)``, ``n_eff`` = 75 x the share of the test
  days with a trade; and each slot's series against the two nulls, described
  ("development"), gating nothing: Null A (999 draws) and Null B (the day
  re-pairing over the shifts 7 to 54).

Nothing is chosen on an outcome. Every computation on a test day is entered in
the run log.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_research.evaluation.significance import minimum_detectable_effect
from trading_research.features.timed import stamps_ns
from trading_research.spot import access as sa
from trading_research.spot import assemble as asm
from trading_research.spot import days as dy
from trading_research.spot import heldout as ho
from trading_research.spot import runlog
from trading_research.spot import statuses as st
from trading_research.spot.context import SpotStudy
from trading_research.spot.fits import (
    fit_fold,
    fit_model,
    frozen_2025,
    row_models,
    rules_at,
    store_rows,
    training_set,
)
from trading_research.spot.plan import ROWS, WALK_FORWARD, Fold, folds
from trading_research.spot.print_label import label_column
from trading_research.spot.registration import FAMILY
from trading_research.spot.runs import RowModels
from trading_research.spot.stage1 import LeakError, spec_of
from trading_research.spot.state import write_json

LEAK_ROWS: tuple[str, ...] = ("S25-PF", "S25-BB")


def fold_leak(study: SpotStudy, fold: Fold, name: str) -> dict[str, Any]:
    """The fold-level test on one row's own cell (module docstring)."""
    from trading_research.validation.fold_leakage import check_fold

    row = ROWS[name]
    cell = row.own
    frame = store_rows(study, row.store, [*fold.train, *fold.inner, *fold.test], row.inputs)
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    target = label_column(cell.hold_s, cell.fee_bp)

    def fit(data: pd.DataFrame) -> dict[str, Any]:
        x, y = training_set(data, fold, cell.hold_s, cell.fee_bp, row.inputs,
                            study.scale.max_training_rows)  # fmt: skip
        model = fit_model(x, y, study.scale.model_params)
        if model is None:
            return {"weights": None, "theta": None}
        stamps = stamps_ns(data)
        keep = fold.inner_keep(stamps, cell.hold_s) & data["scored"].to_numpy(dtype=bool)
        inputs = data.loc[keep, list(model.columns_)].astype(np.float64)
        complete = inputs.notna().all(axis=1).to_numpy()
        proba = np.asarray(model.predict_proba(inputs[complete]))
        rule = rules_at(proba, len(fold.inner), [cell.rate])[cell.rate]
        return {"weights": model, "theta": asdict(rule)}

    cutoff = pd.Timestamp(fold.cutoff_ns, unit="ns", tz="UTC")
    found = check_fold(fit, frame, cutoff, price_columns=list(row.inputs), label_columns=[target],
                       time_column="timestamp", fold=f"{name}/{fold.name}")  # fmt: skip
    return {"row": name, "fold": fold.name, "cutoff": str(cutoff), "rows": len(frame),
            "changed": list(found.changed), "leaks": found.leaks}  # fmt: skip


def usable_days(
    record1: Mapping[str, Any], days: Sequence[date], venue: str | None = None
) -> list[date]:
    """D's days stage 1 found usable on ``venue`` (on either venue when None)."""
    names = dy.VENUE_PLANES if venue is None else (venue,)
    found = record1["days"]
    return [d for d in days if any(not found[d.isoformat()][n]["reasons"] for n in names)]


def models_for(
    study: SpotStudy, record: Mapping[str, Any], touch: float, s25_t: RowModels
) -> dict[str, RowModels]:
    out = row_models(study, record, touch)
    out["S25-T"] = s25_t
    return out


def plan_for(
    study: SpotStudy, fold: Fold, models: Mapping[str, RowModels], record1: Mapping[str, Any]
) -> ho.ReadPlan:
    return ho.ReadPlan(label=f"D_{fold.name}", block="D", days=fold.test,
                       access=sa.development_access(), models=models,
                       max_gaps=int(record1["max_sequence_gaps"]), depth=int(record1["walk"]),
                       spec=spec_of(record1), null_a=study.scale.null_a_draws)  # fmt: skip


def fold_records(study: SpotStudy, fold: Fold, record1: Mapping[str, Any]) -> dict[str, Any]:
    """The fold's fits, written once; a fold whose record exists is not fitted again."""
    path = study.work / "D" / "fits" / fold.name / "record.json"
    if path.exists():
        loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return loaded
    load = ho.loader(study, plan_for(study, fold, {}, record1))
    found = fit_fold(study, fold, load, float(record1["touch_size"]))
    write_json(path, found)
    return json.loads(path.read_text(encoding="utf-8"))


def sigma_and_mde(s: st.SlotDays) -> dict[str, Any]:
    """sigma_D over the test days with a trade, and the MDE at n_eff = 75 x their share
    of the venue's kept test days; an undefined sigma is recorded as such."""
    v = st.series(s.value)
    share = len(v) / len(s.days) if s.days else 0.0
    sigma = float(np.std(v, ddof=1)) if len(v) >= 2 else math.nan
    mde = (
        minimum_detectable_effect(sigma, st.MDE_DAYS_H * share)
        if math.isfinite(sigma)
        else math.nan
    )
    return {"sigma_d": sigma, "share_days_with_trade": share, "n_eff": st.MDE_DAYS_H * share,
            "mde": mde, "test_days": len(s.days), "days_with_trade": len(v)}  # fmt: skip


def null_a(records: Mapping[str, Any], s: st.SlotDays, draws: int) -> dict[str, Any]:
    """Null A on the own cell: each draw's mean of the daily series against the true mean."""
    true = st.series_mean(s.value)
    rows = [records[d.isoformat()]["rows"].get(s.slot, {}) for d in s.days]
    table = np.array([[np.nan if v is None else v for v in (r.get("null_a") or [None] * draws)]
                      for r in rows], dtype=float).reshape(len(rows), draws).T  # fmt: skip
    means = [st.series_mean(row) for row in table]
    above = sum(m >= true for m in means)
    return {"draws": draws, "true_mean": true, "p": (1 + above) / (1 + draws) if draws else None,
            "median_draw": float(np.median(means)) if means else None}  # fmt: skip


def null_b(study: SpotStudy, plan: ho.ReadPlan, records: Mapping[str, Any],
           rows: Mapping[str, st.SlotDays]) -> dict[str, Any]:  # fmt: skip
    """Null B on the own cell: every test day takes the signal tape of the test day
    ``s`` before it, circularly over the kept test days, for every registered shift."""
    load = ho.loader(study, plan)
    table = study.fee_table
    out: dict[str, Any] = {}
    for venue, names in ho.VENUE_ROWS.items():
        days = asm.kept_days(records, venue)
        shifts = list(study.scale.null_b_shifts(len(days)))
        values = {name: np.full((len(shifts), len(days)), np.nan) for name in names}
        for j, day in enumerate(days):
            inputs = load(date.fromisoformat(day))
            tables = ho.own_tables(plan, inputs, table, venue)
            start = dy.midnight_ns(inputs.day)
            for k, shift in enumerate(shifts):
                source = records[days[(j - shift) % len(days)]]
                for name in names:
                    if name in source["rows"] and "signal" in source["rows"][name]:
                        values[name][k, j] = ho._shift(name, tables[name], start, source)[0]
        for name in names:
            true = st.series_mean(rows[name].value)
            means = [st.series_mean(row) for row in values[name]]
            out[name] = {"shifts": [shifts[0], shifts[-1]] if shifts else None,
                         "true_mean": true,
                         "p": (1 + sum(m >= true for m in means)) / (1 + len(means)) if means else None,
                         "median_shift": float(np.median(means)) if means else None}  # fmt: skip
    return out


def ensure_2025(study: SpotStudy) -> None:
    """Outside the checkout that holds the 2025 study's frozen files, copy them in,
    each sha256 checked before and after the copy; the study then reads its own."""
    from dataclasses import replace

    from trading_research.spot.ledger import copy_frozen_2025

    root = study.binding.root
    if root is None or Path(root).resolve() == study.repo.resolve():
        return
    copy_frozen_2025(Path(root), study.repo, dict(study.binding.files))
    study.binding = replace(study.binding, root=None)


def run(study: SpotStudy) -> dict[str, Any]:
    """The D stage's record (module docstring)."""
    record1 = study.state.record("1")
    touch = float(record1["touch_size"])
    ensure_2025(study)
    s25_t = frozen_2025(study)
    all_folds = folds(usable_days(record1, list(study.scale.days("D"))))
    fitted = {name: fold_records(study, fold, record1) for name, fold in all_folds.items()}
    leaks = []
    for name in WALK_FORWARD:
        for row in LEAK_ROWS:
            path = study.work / "D" / "leak" / f"{name}_{row}.json"
            if not path.exists():
                write_json(path, fold_leak(study, all_folds[name], row))
            leaks.append(json.loads(path.read_text(encoding="utf-8")))
    if any(found["leaks"] for found in leaks):
        raise LeakError(f"K-leak: the fold-level test changed {[f for f in leaks if f['leaks']]}")
    records: dict[str, Any] = {}
    plans = {}
    for name in WALK_FORWARD:
        plans[name] = plan_for(study, all_folds[name], models_for(study, fitted[name], touch, s25_t),
                               record1)  # fmt: skip
        records.update(ho.read_days(study, plans[name]))
    rows = {r: asm.slot_days(records, r) for r in ROWS}
    read = sorted({d for f in all_folds.values() for d in (*f.inner, *f.test)})
    runlog.append(study.repo, run=f"D{study.run}", stage="D", days=read,
                  note="D: inner validations and the test months scored")  # fmt: skip
    kept = asm.kept_days(records, sa.BYBIT)
    folders = [study.work / "reads" / f"D_{name}" for name in WALK_FORWARD]
    return {
        "folds": {n: {"train": list(f.train), "inner": list(f.inner), "test": list(f.test)}
                  for n, f in all_folds.items()},
        "fits": {n: r["fits"] for n, r in fitted.items()},
        "thresholds": {n: r["thresholds"] for n, r in fitted.items()},
        "fold_leakage": leaks,
        "series": {r: asm.measurements(records, s) for r, s in rows.items()},
        "mde": {r: sigma_and_mde(rows[r]) for r in FAMILY},
        "null_a": {r: null_a(records, s, study.scale.null_a_draws) for r, s in rows.items()},
        "null_b": null_b(study, plans[WALK_FORWARD[0]], records, rows),
        "calibration": asm.calibration(folders, kept),
        "excluded_on_scoring": {name: {d: r["venues"][name]["reasons"] for d, r in records.items()
                                       if not r["kept"].get(name)} for name in ho.VENUE_ROWS},
        "s25_t_files": {str(k): v for k, v in study.binding.files.items()},
        "labelled": "development",
    }  # fmt: skip
