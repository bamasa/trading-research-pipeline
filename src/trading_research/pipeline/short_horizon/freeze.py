"""Stage E: the frozen fits and files, sigma_D and the MDEs, the detector on D, and the D amendment.

* **Frozen fits**, trained on D before 2025-06-24 (purged before it), their
  thresholds and checkpoints set on 06-24 to 06-30: every configuration in the
  neighbourhood of C*, R* (admitted or not) and S25, and the control networks
  on R*'s holds when R* is the network. Weights go to
  ``models/short_horizon_2025/weights/<fit key>`` (ignored by git); their
  sha256s, every threshold and every checkpoint epoch go in the amendment.
* **Finalist files** ``models/short_horizon_2025/finalists/<slot>.json``: a
  slot's configuration, weight files, thresholds, gate levels, regime window
  and the fitted values' sha256, the set a later trial would load.
* **The fitted-values file** ``models/short_horizon_2025/fitted_values.json``:
  W's touch size, the tensor's calibration, the stores' layout and S1's
  ``sigma_ref``.
* **sigma_D and the minimum detectable effect** per slot, the frozen
  detector's hour profile, its alarms and alarm rate on D (R is declared
  untestable when the rate is above one per 14 days), and the fold-wise flags.
* **The amendment**: every placeholder of the YAML filled
  (``short_horizon_registration.freeze_text``) and a dated section appended
  to the registration with the line ``Frozen configuration sha256: `<hash>```.
  Both files are written for review; committing them is the author's act.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_research.pipeline.short_horizon.context import Study
from trading_research.pipeline.short_horizon.fits import Configuration, configurations
from trading_research.reporting.search_keys import S25_KEY
from trading_research.search_engine.catalogue import parse_key

#: Rows the frozen network fits and the control fits take their windows over.
CONTROL_PREFIX = "control"


def plain(value: Any) -> Any:
    """A value as YAML and JSON both write it back: numpy scalars as Python numbers,
    a non-finite float as None, mappings with string keys."""
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def frozen_configurations(keys: list[str], record_d: dict[str, Any]) -> list[Configuration]:
    """Every configuration in the neighbourhoods of C*, R* and S25 (catalogue order)."""
    from trading_research.pipeline.short_horizon.search import Grid

    grid = Grid(keys)
    wanted: list[str] = []
    for name in ("C*", "R*"):
        cell = record_d["choice"][name]["cell"]
        if cell is not None and grid.position(cell) is not None:
            wanted += grid.neighbours(grid.position(cell))  # type: ignore[arg-type]
    s25 = grid.position(S25_KEY)
    if s25 is not None:
        wanted += grid.neighbours(s25)
    return configurations([k for k in keys if k in set(wanted)])


def fit_key(cfg: Configuration, *, control: bool = False) -> str:
    """A frozen fit's name, the name of its weight file under ``weights/``: its
    configuration made a file name, with the file type the model is saved as."""
    suffix = ".pt" if cfg.model == "tcn" else ".pkl"
    return f"{CONTROL_PREFIX}__{cfg.folder}{suffix}" if control else f"{cfg.folder}{suffix}"


def save_json(path: Path, value: Any) -> str:
    """Write ``value`` as JSON and return the file's sha256."""
    from trading_research.pipeline.short_horizon.models import file_sha256

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain(value), indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return file_sha256(path)


def utc_today() -> date:
    return datetime.now(UTC).date()


def d_series(study: Study, key: str, days: list[date]) -> pd.Series:
    """A taker cell's daily ``net_bp`` over D's usable test days (NaN without a trade)."""
    from trading_research.search_engine.ledger import LEDGER_PATH, TrialLedger

    rows = TrialLedger(study.repo / LEDGER_PATH, study.catalogue()).run_rows(study.run)
    mine = rows[(rows["key"] == key) & rows["day"].isin(days)].set_index("day")
    values = mine["net_bp"].where(mine["trades"] > 0)
    return values.reindex(days)


def control_targets(inputs: Any, days: list[date], holds: list[float], out: Path) -> Path:
    """The control label at the every-tenth rows of ``days``, for the control networks."""
    from trading_research.labels.taker_vwap import control_move_bp
    from trading_research.pipeline.short_horizon import features as ft

    out.mkdir(parents=True, exist_ok=True)
    for day in days:
        target = out / f"{day}.parquet"
        if target.exists():
            continue
        book = inputs.book(day)
        frame = inputs.targets(day)[["row", ft.TIME]].copy()
        for hold in holds:
            frame[f"control_{hold:g}"] = control_move_bp(book, hold).to_numpy()[
                frame["row"].to_numpy()
            ]
        frame.to_parquet(target, index=False)
    return out


def frozen_fits(study: Study, inputs: Any, record_d: dict[str, Any]) -> dict[str, Any]:
    """Every frozen fit, its weights saved and hashed, thresholds, epochs and gate levels."""
    import shutil

    from trading_research.pipeline.short_horizon.fits import (
        FoldData,
        NetworkJob,
        fit_main_process,
        fit_networks,
    )
    from trading_research.pipeline.short_horizon.models import file_sha256, in_fresh_process, save
    from trading_research.pipeline.short_horizon.plan import frozen_fold
    from trading_research.pipeline.short_horizon_ledger import WEIGHTS_DIR

    fold = frozen_fold(inputs.usable("D"))
    if not fold.train or not fold.inner:
        raise RuntimeError("the frozen fits need usable days before 06-24 and from 06-24 to 06-30")
    configs = frozen_configurations(list(study.keys), record_d)
    weights = study.repo / WEIGHTS_DIR
    columns = sorted({c for cfg in configs for c in inputs.columns(cfg.inputs)})
    data = FoldData.load(inputs, fold.train, columns)
    out: dict[str, Any] = {
        "weights_sha256": {},
        "checkpoint_epoch": {},
        "thresholds": {},
        "gate_levels": {},
    }
    for cfg in configs:
        if cfg.model == "tcn":
            continue
        model, entry = fit_main_process(inputs, fold, cfg, data)
        if model is not None:  # a configuration with nothing to fit on has no weights
            out["weights_sha256"][fit_key(cfg)] = save(model, weights / fit_key(cfg))
        if cfg.stratum == "C":
            out["thresholds"][cfg.name] = entry["thresholds"]
    for hold in sorted({cfg.hold_s for cfg in configs}):
        out["gate_levels"][f"{hold:g}"] = data.gate_levels(fold.train_end_ns(hold)).to_dict()
    del data
    networks = tuple(cfg for cfg in configs if cfg.model == "tcn")
    if networks:
        holds = sorted({cfg.hold_s for cfg in networks})
        control = control_targets(
            inputs, [*fold.train, *fold.inner], holds, study.dir("E", "control_targets")
        )
        for kind, job in (
            ("network", NetworkJob(study, fold, networks, study.dir("E", "network"))),
            (
                "control",
                NetworkJob(study, fold, networks, study.dir("E", "control"), control_dir=control),
            ),
        ):
            found = in_fresh_process(fit_networks, job)
            for cfg in networks:
                if found[cfg.name]["weights_sha256"] is None:
                    continue
                key = fit_key(cfg, control=kind == "control")
                target = weights / key
                shutil.copyfile(job.out / cfg.folder / "weights.pt", target)
                out["weights_sha256"][key] = file_sha256(target)
                out["checkpoint_epoch"][key] = int(found[cfg.name]["checkpoint_epoch"])
    out["fold"] = {
        "train": [d.isoformat() for d in fold.train],
        "inner": [d.isoformat() for d in fold.inner],
    }
    return out


def theta(record: dict[str, Any], key: str) -> dict[str, float]:
    """A C cell's frozen thresholds at its rate; a side with no candidate row never
    trades, written as 2 (a probability never reaches it) so the value stays finite."""
    cell = parse_key(key)
    values = record["thresholds"][cell.configuration][f"{cell.trades_per_day:g}"]
    return {name: (float(v) if math.isfinite(float(v)) else 2.0) for name, v in values.items()}


def slot_cells(
    study: Study, record_d: dict[str, Any], record_c: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Each slot's frozen cell, its neighbourhood (the cells K-nbhd reruns) and settings."""
    from trading_research.pipeline.short_horizon.search import Grid

    grid = Grid(list(study.keys))
    out: dict[str, dict[str, Any]] = {}

    def entry(key: str, **extra: Any) -> dict[str, Any]:
        at = grid.position(key)
        near = [] if at is None else grid.neighbours(at)
        return {
            "key": key,
            "neighbourhood": near,
            "regime_window_h": parse_key(key).regime_h,
        } | extra

    out["S25"] = entry(S25_KEY)
    for stratum, slot in (("C", "S-C"), ("R", "S-R")):
        cell = record_d["choice"][f"{stratum}*"]["cell"]
        if cell is not None:
            out[slot] = entry(cell)
    passive = record_c["admission"]
    signal = passive["S-P"]["signal_key"]
    cells = record_c["cells"]
    out["S-P"] = entry(signal) | {"execution": "signal-entry quoter", "cells": [
        c["key"] for c in cells if c["slot"] == "S-P" and not c["measured"]]}  # fmt: skip
    out["S-M"] = entry(signal) | {"execution": "S1 with the lean", "leans": [0.5, 1.0, 2.0],
                                  "guard": parse_key(signal).regime}  # fmt: skip
    out["S25-Z"] = entry(S25_KEY) | {"execution": "signal-entry quoter at 0/0", "cells": [
        c["key"] for c in cells if c["slot"] == "S25-Z"]}  # fmt: skip
    return out


def finalist_files(study: Study, slots: dict[str, dict[str, Any]], fitted: dict[str, Any],
                   fitted_sha: str) -> dict[str, str]:  # fmt: skip
    """One file per slot: configuration, weights, thresholds, gate levels, fitted values."""
    from trading_research.pipeline.short_horizon_ledger import FINALISTS_DIR

    out = {}
    for slot, entry in slots.items():
        keys = sorted({entry["key"], *entry["neighbourhood"], *entry.get("cells", [])})
        configs = configurations([k for k in study.keys if k in set(keys)])
        record = entry | {
            "slot": slot,
            "weights_sha256": {k: v for k, v in fitted["weights_sha256"].items()
                               if k in {fit_key(c, control=x) for c in configs for x in (False, True)}},
            "thresholds": {k: theta(fitted, k) for k in keys if parse_key(k).stratum == "C"
                           and parse_key(k).configuration in fitted["thresholds"]},
            "gate_levels": {f"{c.hold_s:g}": fitted["gate_levels"][f"{c.hold_s:g}"] for c in configs},
            "fitted_values_sha256": fitted_sha,
        }  # fmt: skip
        out[slot] = save_json(study.repo / FINALISTS_DIR / f"{slot}.json", record)
    return out


def mde(series: pd.Series | np.ndarray, n: int, usable_days: int) -> tuple[float, float] | None:
    """sigma_D of a daily series and its MDE: ``(3 + 0.84) sigma / sqrt(n_eff)``,
    ``n_eff`` being ``n`` times the share of days with a value."""
    from trading_research.evaluation.significance import minimum_detectable_effect

    values = np.asarray(series, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 2 or usable_days < 1:
        return None
    sigma = float(np.std(values, ddof=1))
    n_eff = n * len(values) / usable_days
    return sigma, float(minimum_detectable_effect(sigma, n_eff))


def slot_mdes(
    study: Study, record_d: dict[str, Any], slots: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """sigma_D and the MDE of each slot's daily series on D (24 for a slot on S25's signal)."""
    days = [date.fromisoformat(d) for d in record_d["usable_test_days"]]
    rows = pd.read_parquet(study.work / "C" / "cells.parquet")
    on_s25 = slots["S-P"]["key"] == S25_KEY
    out: dict[str, Any] = {}
    for slot in ("S25", "S-C", "S-R"):
        if slot in slots:
            out[slot] = mde(
                d_series(study, slots[slot]["key"], days), 24 if slot == "S25" else 44, len(days)
            )
    for slot, n in (("S-P", 24 if on_s25 else 44), ("S-M", 24 if on_s25 else 44), ("S25-Z", 24)):
        mine = rows[(rows["slot"] == slot) & ~rows["excluded"]]
        if slot == "S-M":
            mine = mine[mine["label"].str.contains("lean=1|", regex=False) & mine["label"].str.contains(
                f"policy={slots['S-M']['guard']}", regex=False)]  # fmt: skip
        else:
            mine = mine[mine["key"] == slots[slot]["key"]]
        column = "gross_bp_trade" if slot == "S25-Z" else "net_per_100"
        out[slot] = mde(mine[column].to_numpy(), n, len(days))
    return {slot: value for slot, value in out.items() if value is not None}


def detector_record(study: Study) -> dict[str, Any]:
    """The frozen detector on D: its hour profile, alarms, alarm rate, and the fold-wise flags."""
    from trading_research.data import study_access as sa
    from trading_research.pipeline.short_horizon import flags
    from trading_research.pipeline.short_horizon.plan import TEST_MONTHS
    from trading_research.validation import regime_flags as rf

    d_start, d_end = sa.BLOCKS["D"]
    close = flags.hourly_close(study.symbol, study.roots.bars, d_end, sa.development_access())
    frozen = flags.frozen(close, d_end)
    alarms = flags.alarms_frame(frozen)
    end = pd.Timestamp(d_end, tz="UTC") + pd.Timedelta(days=1)
    rate = rf.alarm_rate(alarms, pd.Timestamp(d_start, tz="UTC"), end)
    on_d = [a for a in frozen["alarms"] if pd.Timestamp(a["timestamp"]) < end]
    foldwise = {}
    for month in TEST_MONTHS:
        record = flags.saved_fold_flags(month, study.work / "A" / "flags")
        foldwise[month] = [
            pd.Timestamp(int(t), unit="ns", tz="UTC").isoformat() for t in record["guard_flags_ns"]
        ]
    return {
        "hour_profile": frozen["profile"],
        "alarms_D": [
            {"timestamp": a["timestamp"], "channel": a["statistic"], "direction": a["direction"]}
            for a in on_d
        ],
        "alarm_rate_D": rate,
        "r_untestable_declared": bool(rate > 1.0 / 14.0),
        "flags_D_foldwise": foldwise,
    }


def compute_record(study: Study) -> dict[str, Any]:
    """The compute record: stage times, workers, the replay's memory, the stores' sizes."""
    entries = study.state.entries()
    record0, record1 = study.record("0"), study.record("1")
    return {
        "stages": {
            s: {k: entries[s].get(k) for k in ("started", "finished", "attempts")} for s in entries
        },
        "workers": study.workers,
        "replay": record0["replay_memory"],
        "store_bytes": record1["stores"]["store_bytes"],
        "network_device": study.scale.params("tcn").get("device") or "mps when available",
        "scale": study.scale.name,
    }


def placeholder_values(study: Study, parts: dict[str, Any]) -> dict[tuple[str, ...], Any]:
    """Every placeholder the D amendment fills, from the stages' records."""
    record1, record_b = study.record("1"), study.record("B")
    record_d, record_c = study.record("D"), study.record("C")
    excluded = [[date.fromisoformat(d), "; ".join(e["reasons"])] for d, e in record1["days"].items()
                if not e["usable"]]  # fmt: skip
    excluded += [[date.fromisoformat(d), "flatten_stale"] for d in record_b["stale_days"]]
    slots, fits, mdes = parts["slots"], parts["fits"], parts["mdes"]
    admission = {"S25": {"status_on_D": "fixed before D; no admission and no search-level null",
                         "against_its_own_nulls": record_d["S25"]},
                 "S25-Z": {"status_on_D": "fixed before D; outside the family"}}  # fmt: skip
    for stratum, slot in (("C", "S-C"), ("R", "S-R")):
        admission[slot] = {
            "status_on_D": record_d["statuses_on_D"][slot],
            "carry_forward": record_d["carry_forward"][stratum],
            "conditions": record_d["conditions"][stratum],
            "nulls": record_d["nulls"][stratum],
            "pbo": record_d["overfitting"]["pbo"].get(stratum),
            "selection_premium": record_d["selection_premium"][stratum],
        }
    for slot in ("S-P", "S-M"):
        admission[slot] = record_c["admission"][slot]
    detector = parts["detector"]
    s25_cells = [k for k in slots["S25"]["neighbourhood"] if parse_key(k).stratum == "C"]
    c_cells = slots["S-C"]["neighbourhood"] if "S-C" in slots else []
    values: dict[tuple[str, ...], Any] = {
        ("data", "max_sequence_gaps"): int(record1["max_sequence_gaps"]),
        ("data", "excluded_days_D"): sorted(excluded),
        ("data", "instrument_spec"): record1["instrument_spec"],
        ("features", "selected"): record1["selected"],
        ("features", "fitted_values_sha256"): parts["fitted_sha256"],
        ("models", "weights_sha256"): fits["weights_sha256"],
        ("models", "checkpoint_epoch"): fits["checkpoint_epoch"],
        ("search", "result_trials"): {
            "N_trials": record_d["N_trials"], "N_eff": record_d["overfitting"]["N_eff"],
            "R_D": record_d["R_D"], "ledger_sha256": record_d["ledger_sha256"],
            "ledger_rows": record_d["ledger_rows"], "hand_runs": record_d["hand_runs"]},
        ("search", "result_choice"): record_d["choice"],
        ("search", "result_admission"): admission,
        ("search", "result_nulls"): {"nulls_file_sha256": record_d["files_sha256"]["nulls"],
                                     "p_search": record_d["p_search"], "strata": record_d["nulls"],
                                     "null_A": record_d["null_A"], "null_B": record_d["null_B"],
                                     "refit": record_d["refit"]},
        ("slots", "S25", "thresholds"): {k: theta(fits, k) for k in s25_cells},
        ("slots", "S-C", "thresholds"): {k: theta(fits, k) for k in c_cells},
        ("metrics", "mde", "sigma_d"): {slot: v[0] for slot, v in mdes.items()},
        ("metrics", "mde", "value"): {slot: v[1] for slot, v in mdes.items()},
        ("measurements", "result_O"): record_c["a0_and_fill"],
        ("detector", "flags_D_foldwise"): detector["flags_D_foldwise"],
        ("detector", "hour_profile"): detector["hour_profile"],
        ("detector", "alarm_rate_D"): detector["alarm_rate_D"],
        ("detector", "alarms_D"): detector["alarms_D"],
        ("detector", "r_untestable_declared"): detector["r_untestable_declared"],
        ("compute", "record"): compute_record(study),
        ("amendment", "date"): utc_today(),
        ("artefacts", "frozen_finalist_files_sha256"): parts["finalists"],
    }  # fmt: skip
    return {path: plain(value) for path, value in values.items()}


def amendment_text(study: Study, frozen: str, values: dict[tuple[str, ...], Any]) -> str:
    """The dated section appended to the registration's amendments."""
    from trading_research.pipeline import short_horizon_registration as shr

    record_d = study.record("D")
    choice = record_d["choice"]
    lines = [
        "",
        f"### {values[('amendment', 'date')]}: the D amendment",
        "",
        "Written by the entry point (`experiments/short_horizon_2025.py`, stage E) and reviewed",
        "before it is committed. Every frozen value is in the configuration's placeholders.",
        "",
        shr.frozen_hash_line(frozen),
        "",
        f"- Complete D runs: R_D = {record_d['R_D']}; N_trials = {record_d['N_trials']}; "
        f"trial ledger sha256 `{record_d['ledger_sha256']}`, {record_d['ledger_rows']} rows.",
        f"- C*: `{choice['C*']['cell']}`; R*: `{choice['R*']['cell']}` (selected, biased upward).",
        f"- p_search = {record_d['p_search']:.4g} (on development data, selected).",
        f"- Statuses set on D: {record_d['statuses_on_D']}.",
        f"- The fitted-values file's sha256 `{values[('features', 'fitted_values_sha256')]}`.",
        "",
    ]
    return "\n".join(lines)


def write_amendment(study: Study, values: dict[tuple[str, ...], Any]) -> dict[str, str]:
    """Fill the YAML's placeholders and append the dated amendment (both reviewed and
    committed by the author); refused unless every placeholder is still null."""
    from trading_research.market_making.prereg import sha256_bytes
    from trading_research.pipeline import short_horizon_registration as shr

    config, document = study.repo / shr.CONFIG_PATH, study.repo / shr.DOCUMENT_PATH
    text, doc = config.read_text(encoding="utf-8"), document.read_text(encoding="utf-8")
    frozen = shr.freeze_text(text, values, null_sha256=shr.latest_null_sha256(doc),
                             catalogue_sha256=shr.latest_catalogue_sha256(doc))  # fmt: skip
    if shr.AMENDMENTS_HEADING not in doc:
        raise RuntimeError("the registration has no amendments section to append to")
    config.write_text(frozen, encoding="utf-8")
    document.write_text(
        doc.rstrip("\n") + "\n" + amendment_text(study, frozen, values), encoding="utf-8"
    )
    return {
        "frozen_config_sha256": sha256_bytes(frozen.encode()),
        "document": shr.DOCUMENT_PATH.as_posix(),
    }


def lean_scales(study: Study, inputs: Any) -> dict[str, float]:
    """S-M's lean on a classifier's signal: the median absolute forward move over each
    hold on the frozen fits' training rows (a fitted value, not refitted at a read)."""
    from trading_research.pipeline.short_horizon import features as ft
    from trading_research.pipeline.short_horizon.plan import frozen_fold

    fold = frozen_fold(inputs.usable("D"))
    out = {}
    for hold in sorted({parse_key(k).hold_s for k in study.keys}):
        moves = [inputs.targets(d)[ft.forward_column(hold)].to_numpy() for d in fold.train]
        values = np.abs(np.concatenate(moves)) if moves else np.zeros(0)
        values = values[np.isfinite(values)]
        out[f"{hold:g}"] = float(np.median(values)) if len(values) else 0.0
    return out


def run(study: Study) -> dict[str, Any]:
    """Stage E's record: the frozen fits and files, the MDEs, the detector, the amendment."""
    from trading_research.pipeline import short_horizon_registration as shr
    from trading_research.pipeline.short_horizon.inputs import Inputs
    from trading_research.pipeline.short_horizon_ledger import FITTED_VALUES_PATH

    registration = shr.Registration.load(repo=study.repo)
    if registration.frozen:
        raise RuntimeError(
            "the configuration is already frozen: stage E writes the D amendment once"
        )
    record1, record_d, record_c = study.record("1"), study.record("D"), study.record("C")
    inputs = Inputs(study)
    fitted_values = {"touch_size": record1["fitted"]["touch_size"], "tensor": record1["fitted"]["tensor"],
                     "sigma_ref": record1["fitted"]["sigma_ref"], "store_layout": record1["stores"]["layout"],
                     "lean_scale_bp": lean_scales(study, inputs)}  # fmt: skip
    fitted_sha = save_json(study.repo / FITTED_VALUES_PATH, fitted_values)
    fits = frozen_fits(study, inputs, record_d)
    slots = slot_cells(study, record_d, record_c)
    parts = {
        "fits": fits, "slots": slots, "fitted_sha256": fitted_sha,
        "finalists": finalist_files(study, slots, fits, fitted_sha),
        "mdes": slot_mdes(study, record_d, slots), "detector": detector_record(study),
    }  # fmt: skip
    values = placeholder_values(study, parts)
    written = write_amendment(study, values)
    return plain({"commit": study.commit(), "fits": len(fits["weights_sha256"]),
                  "weights_sha256": fits["weights_sha256"], "finalists": parts["finalists"],
                  "fitted_values_sha256": fitted_sha, "mde": parts["mdes"], "detector": parts["detector"],
                  "amendment": written, "next": "review and commit the frozen configuration, the amendment, "
                  "the trial ledger and the results files; then open H"})  # fmt: skip
