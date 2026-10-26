"""The D procedure: the fold-level leakage test, the fits, scoring on June and July, sigma_D.

Registered (steps 2 to 5 of the development-period procedure; "Nothing is
searched"):

* **The fits** of each fold (:func:`trading_research.spot.fits.fit_fold`):
  June, July and the frozen fits, every hold and label fee of S25-PF's and
  S25-BB's neighbourhoods, S25-PB's and S25-PFT's own cell; S25-T is the 2025
  study's frozen fit, its files checked by sha256 here as at the H and P gates.
* **The fold-level leakage test** (K-leak) on every fold, the frozen one
  included: the fold's own fit path (:func:`~trading_research.spot.fits.fit_fold`:
  training rows, purge, every fit, every cell's theta on the inner validation's
  loaded days) run again on stores and loaded days whose every input, label and
  price after the fold's cutoff is mutated (``validation/fold_leakage``); every
  fit's digest and every theta must be unchanged. The frozen fits' cutoff is D's
  end, after which no day is on disk.
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
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_research.evaluation.significance import minimum_detectable_effect
from trading_research.spot import access as sa
from trading_research.spot import assemble as asm
from trading_research.spot import days as dy
from trading_research.spot import heldout as ho
from trading_research.spot import runlog
from trading_research.spot import statuses as st
from trading_research.spot.context import SpotStudy
from trading_research.spot.fits import fit_fold, frozen_2025, load_model, row_models, store_rows
from trading_research.spot.plan import ROWS, WALK_FORWARD, Fold, folds
from trading_research.spot.registration import FAMILY
from trading_research.spot.runs import RowModels
from trading_research.spot.stage1 import LeakError, frozen_under, spec_of
from trading_research.spot.state import write_json


def mutated_after(frame: pd.DataFrame | None, cutoff_ns: int, prices: Sequence[str],
                  labels: Sequence[str] = (), seed: int = 0) -> pd.DataFrame | None:  # fmt: skip
    """``frame`` with every named price and label at a row stamped after the cutoff
    changed (``validation/fold_leakage.mutate_after_cutoff``)."""
    from trading_research.features.timed import stamps_ns
    from trading_research.validation.fold_leakage import mutate_after_cutoff

    if frame is None or not len(frame) or not (stamps_ns(frame) > cutoff_ns).any():
        return frame  # nothing after the cutoff: the frame itself, uncopied
    cutoff = pd.Timestamp(cutoff_ns, unit="ns", tz="UTC")
    return mutate_after_cutoff(frame, cutoff, price_columns=[c for c in prices if c in frame],
                               label_columns=[c for c in labels if c in frame], seed=seed)  # fmt: skip


def mutated_inputs(inputs: dy.DayInputs, cutoff_ns: int) -> dy.DayInputs:
    """A loaded day with every book and print price stamped after the cutoff changed."""
    from dataclasses import replace

    book = inputs.book.frame
    levels = (
        []
        if book is None
        else [c for c in book.columns if c.startswith(("bid_price_", "ask_price_"))]
    )
    return replace(
        inputs,
        book=replace(inputs.book, frame=mutated_after(book, cutoff_ns, levels)),
        binance=replace(inputs.binance, frame=mutated_after(inputs.binance.frame, cutoff_ns, ["price"])),
        bybit=replace(inputs.bybit, frame=mutated_after(inputs.bybit.frame, cutoff_ns, ["price"], seed=1)),
    )  # fmt: skip


def fold_leak(
    study: SpotStudy, fold: Fold, record1: Mapping[str, Any], fitted: Mapping[str, Any]
) -> dict[str, Any]:
    """The fold-level test (K-leak) through the fold's own fit path
    (:func:`trading_research.spot.fits.fit_fold`): every fit of the fold and every
    cell's theta again, on stores and loaded days whose every input, label and
    price stamped after the fold's cutoff is changed; each fit (by its model's
    digest) and each theta, in the frozen form, must equal the fold's own."""
    from trading_research.validation.fold_leakage import digest

    cutoff = fold.cutoff_ns
    load = ho.loader(study, plan_for(study, fold, {}, record1))

    def read(
        study_: SpotStudy, store: str, days: Sequence[Any], columns: Sequence[str]
    ) -> pd.DataFrame:
        frame = store_rows(study_, store, days, columns)
        labels = [c for c in frame.columns if c.startswith("label_")]
        found = mutated_after(frame, cutoff, list(columns), labels, seed=2)
        return frame if found is None else found

    folder = study.dir("D", "leak", fold.name, "fits")
    mutated = fit_fold(study, fold, lambda day: mutated_inputs(load(day), cutoff),
                       float(record1["touch_size"]), read=read, folder=folder)  # fmt: skip
    own = study.work / "D" / "fits" / fold.name

    def models(record: Mapping[str, Any], where: Path) -> dict[str, str | None]:
        return {key: None if not entry.get("sha256") else digest(load_model(where / key))
                for key, entry in record["fits"].items()}  # fmt: skip

    before, after = models(fitted, own), models(mutated, folder)
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    changed += sorted(f"theta {row}" for row in set(fitted["thresholds"]) | set(mutated["thresholds"])
                      if fitted["thresholds"].get(row) != mutated["thresholds"].get(row))  # fmt: skip
    return {"fold": fold.name, "cutoff": str(pd.Timestamp(cutoff, unit="ns", tz="UTC")),
            "fits": len(before), "thresholds": sorted(fitted["thresholds"]),
            "changed": changed, "leaks": bool(changed)}  # fmt: skip


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
    ``s`` before it, circularly over the kept test days, for every registered shift.

    Null B describes D and gates nothing, so it excludes no day: a shifted run that
    flags ``flatten_stale`` gives its day no value under that shift, and such runs
    are counted (``stale_runs``)."""
    load = ho.loader(study, plan)
    table = study.fee_table
    out: dict[str, Any] = {}
    for venue, names in ho.VENUE_ROWS.items():
        days = asm.kept_days(records, venue)
        shifts = list(study.scale.null_b_shifts(len(days)))
        values = {name: np.full((len(shifts), len(days)), np.nan) for name in names}
        stale = dict.fromkeys(names, 0)
        for j, day in enumerate(days):
            inputs = load(date.fromisoformat(day))
            tables = ho.own_tables(plan, inputs, table, venue)
            start = dy.midnight_ns(inputs.day)
            for k, shift in enumerate(shifts):
                source = records[days[(j - shift) % len(days)]]
                for name in names:
                    if name in source["rows"] and "signal" in source["rows"][name]:
                        value, flagged = ho._shift(name, tables[name], start, source)
                        stale[name] += int(flagged)
                        values[name][k, j] = np.nan if flagged else value
        for name in names:
            true = st.series_mean(rows[name].value)
            means = [st.series_mean(row) for row in values[name]]
            out[name] = {"shifts": [shifts[0], shifts[-1]] if shifts else None,
                         "true_mean": true, "stale_runs": stale[name],
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
    frozen_under(study)
    record1 = study.state.record("1")
    touch = float(record1["touch_size"])
    ensure_2025(study)
    s25_t = frozen_2025(study)
    all_folds = folds(usable_days(record1, list(study.scale.days("D"))))
    fitted = {name: fold_records(study, fold, record1) for name, fold in all_folds.items()}
    leaks = []
    for name, fold in all_folds.items():  # every fold, the frozen one included
        path = study.work / "D" / "leak" / f"{name}.json"
        if not path.exists():
            write_json(path, fold_leak(study, fold, record1, fitted[name]))
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
