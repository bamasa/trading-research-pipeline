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
* **The pages** as far as D, C and E fill them (:mod:`.pages`): the search
  and payoff sentences, the passive sentence when S-P and S-M were not
  admitted, the S25 page with the ladder's steps 1 and 2, D's part of the
  regime map, and the figures of the search; a failure there is recorded,
  never raised after the amendment is written.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, date, datetime, timedelta
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
    from trading_research.search_engine.ledger import TrialLedger

    rows = TrialLedger.in_repository(study.repo, study.catalogue()).run_rows(study.run)
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
    start = pd.Timestamp(d_start, tz="UTC")
    end = pd.Timestamp(d_end + timedelta(days=1), tz="UTC")
    rate = rf.alarm_rate(alarms, start, end)
    # the run starts on 2024-12-01; alarms_D is the registered span, 2025-01-01..06-30
    on_d = [a for a in frozen["alarms"] if start <= pd.Timestamp(a["timestamp"]) < end]
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


def complete_trials(study: Study, inputs: Any, record_d: dict[str, Any]) -> dict[str, Any]:
    """The trial ledger after stage C: N_trials with the simulator trials counted, the
    ledger's sha256 and rows, and N_eff and the deflated Sharpe ratios at that N (stage
    D quoted them at the taker cells' count, before the simulator ran). The
    leaderboard's header is rewritten to quote the same N."""
    from trading_research.pipeline.short_horizon import search
    from trading_research.search_engine.ledger import TrialLedger

    manifest = TrialLedger.in_repository(study.repo, study.catalogue()).record()
    n = int(manifest["n_trials"])
    grid = search.Grid(list(study.keys))
    cells = search.observed_cells(study, grid, search.Calendar.of(study, inputs))
    chosen = {s: grid.position(record_d["choice"][f"{s}*"]["cell"] or "") for s in grid.strata}
    board = study.results("leaderboard.csv")
    if board.exists():
        text = board.read_text(encoding="utf-8")
        board.write_text(text.replace(f"# N trials: {record_d['N_trials']}\n", f"# N trials: {n}\n"),
                         encoding="utf-8")  # fmt: skip
    return {"N_trials": n, "N_trials_taker": int(manifest.get("n_trials_taker", n)),
            "N_trials_simulator": int(manifest.get("n_trials_simulator", 0)),
            "ledger_sha256": manifest["sha256"], "ledger_rows": int(manifest["rows"]),
            "overfitting": search.overfitting(grid, cells, chosen, n)}  # fmt: skip


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


#: What each deflated Sharpe ratio is deflated against (``evaluation/overfitting``).
TAKER_DSR_VARIANCE = ("N_trials with the simulator trials counted, and the variance of the Sharpe "
                      "ratios of every taker cell's daily G on D's usable test days")  # fmt: skip
PASSIVE_DSR_VARIANCE = ("N_trials with the simulator trials counted, and the variance of the Sharpe "
                        "ratios of every S-P and S-M cell's daily net_per_100_clip_day simulated on "
                        "D, over the days no run flagged, 0 on a day without a fill")  # fmt: skip
#: Stage C's second detector on D when the stage's record does not carry it.
SECOND_DETECTOR_NOT_RUN = {"span": "D", "alarms": [], "note": "not run by stage C at this commit"}


def slot_dsr(found: dict[str, Any], name: str, series: tuple[str, ...], against: str) -> dict[str, Any]:  # fmt: skip
    """A slot's deflated Sharpe ratios as a stage recorded them under ``<name>/<series>``
    (None where none was computed), with what they are deflated against."""
    return {s: found.get(f"{name}/{s}") for s in series} | {"deflated_against": against}


def slot_dsr_inputs(inputs: dict[str, Any], name: str, series: tuple[str, ...]) -> dict[str, Any]:
    """Beside each deflated Sharpe ratio: the variance it is deflated against, the
    benchmark SR0, the series' Sharpe ratio and its days (None where not recorded)."""
    return {s: inputs.get(f"{name}/{s}") for s in series}


def placeholder_values(study: Study, parts: dict[str, Any]) -> dict[tuple[str, ...], Any]:
    """Every placeholder the D amendment fills, from the stages' records."""
    record1, record_b = study.record("1"), study.record("B")
    record_d, record_c = study.record("D"), study.record("C")
    excluded = [[date.fromisoformat(d), "; ".join(e["reasons"])] for d, e in record1["days"].items()
                if not e["usable"]]  # fmt: skip
    excluded += [[date.fromisoformat(d), "flatten_stale"] for d in record_b["stale_days"]]
    slots, fits, mdes, trials = parts["slots"], parts["fits"], parts["mdes"], parts["trials"]
    dsr = trials["overfitting"].get("dsr", {})  # at N_trials with the simulator trials counted
    inputs = trials["overfitting"].get("dsr_inputs", {})
    power = record_d.get("power_at_bar", {})
    admission = {"S25": {"status_on_D": "fixed before D; no admission and no search-level null",
                         "against_its_own_nulls": record_d["S25"],
                         "dsr": slot_dsr(dsr, "S25", ("G", "net"), TAKER_DSR_VARIANCE),
                         "dsr_inputs": slot_dsr_inputs(inputs, "S25", ("G", "net"))},
                 "S25-Z": {"status_on_D": "fixed before D; outside the family"}}  # fmt: skip
    for stratum, slot in (("C", "S-C"), ("R", "S-R")):
        admission[slot] = {
            "status_on_D": record_d["statuses_on_D"][slot],
            "carry_forward": record_d["carry_forward"][stratum],
            "conditions": record_d["conditions"][stratum],
            "nulls": record_d["nulls"][stratum],
            "pbo": record_d["overfitting"]["pbo"].get(stratum),
            "selection_premium": record_d["selection_premium"][stratum],
            "dsr": slot_dsr(dsr, f"{stratum}*", ("G", "net"), TAKER_DSR_VARIANCE),
            "dsr_inputs": slot_dsr_inputs(inputs, f"{stratum}*", ("G", "net")),
            "power_at_bar": power.get(slot),
        }
    passive_dsr = record_c.get("overfitting", {}).get("dsr", {})
    for slot in ("S-P", "S-M"):
        admission[slot] = dict(record_c["admission"][slot]) | {
            "dsr": slot_dsr(passive_dsr, slot, ("net",), PASSIVE_DSR_VARIANCE),
            "dsr_inputs": slot_dsr_inputs(passive_dsr, f"inputs/{slot}", ("net",)),
        }
    detector = parts["detector"]
    s25_cells = [k for k in slots["S25"]["neighbourhood"] if parse_key(k).stratum == "C"]
    c_cells = slots["S-C"]["neighbourhood"] if "S-C" in slots else []
    values: dict[tuple[str, ...], Any] = {
        ("data", "max_sequence_gaps"): int(record1["max_sequence_gaps"]),
        ("data", "excluded_days_D"): sorted(excluded),
        ("data", "instrument_spec"): record1["instrument_spec"],
        ("data", "ob200_reader_test_result"): record1["reader_test"],
        ("features", "selected"): record1["selected"],
        ("features", "fitted_values_sha256"): parts["fitted_sha256"],
        ("models", "weights_sha256"): fits["weights_sha256"],
        ("models", "checkpoint_epoch"): fits["checkpoint_epoch"],
        ("models", "fits_lacking_a_class"): record_d.get("fits_lacking_a_class", []),
        ("search", "result_trials"): {
            "N_trials": trials["N_trials"], "N_eff": trials["overfitting"]["N_eff"],
            "R_D": record_d["R_D"], "ledger_sha256": trials["ledger_sha256"],
            "ledger_rows": trials["ledger_rows"], "hand_runs": record_d["hand_runs"]},
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
        ("detector", "second_detector_D"): record_c.get("second_detector_D")
        or SECOND_DETECTOR_NOT_RUN,
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
    result = values[("search", "result_trials")]
    simulator = int(study.record("C").get("trials", {}).get("n_trials_simulator", 0))
    trials = {"N": result["N_trials"], "N_taker": result["N_trials"] - simulator,
              "N_simulator": simulator, "sha256": result["ledger_sha256"], "rows": result["ledger_rows"]}  # fmt: skip
    lines = [
        "",
        f"### {values[('amendment', 'date')]}: the D amendment",
        "",
        "Written by the entry point (`experiments/short_horizon_2025.py`, stage E) and reviewed",
        "before it is committed. Every frozen value is in the configuration's placeholders.",
        "",
        shr.frozen_hash_line(frozen),
        "",
        f"- Complete D runs: R_D = {record_d['R_D']}; N_trials = {trials['N']} "
        f"({trials['N_taker']} taker cells and {trials['N_simulator']} simulator trials); "
        f"trial ledger sha256 `{trials['sha256']}`, {trials['rows']} rows.",
        f"- C*: `{choice['C*']['cell']}`; R*: `{choice['R*']['cell']}` (selected, biased upward).",
        f"- p_search = {record_d['p_search']:.4g} (on development data, selected).",
        f"- Statuses set on D: {record_d['statuses_on_D']}.",
        f"- The fitted-values file's sha256 `{values[('features', 'fitted_values_sha256')]}`.",
        *reader_test_lines(values[("data", "ob200_reader_test_result")]),
        *overfitting_lines(values[("search", "result_admission")]),
        lacking_a_class_line(values[("models", "fits_lacking_a_class")]),
        second_detector_line(values[("detector", "second_detector_D")]),
        *results_file_lines(study),
        "",
    ]
    return "\n".join(lines)


def results_file_lines(study: Study) -> list[str]:
    """The sha256 of D's results files (the leaderboard after its header names
    N_trials, the marginals, the nulls), which every open of a block checks."""
    from trading_research.data.study_fetch import file_sha256
    from trading_research.pipeline import short_horizon_registration as shr

    return [f"- {shr.results_file_line(p, file_sha256(study.repo / p))}"
            for p in shr.D_RESULTS_FILES if (study.repo / p).is_file()]  # fmt: skip


def _number(value: Any) -> str:
    return "not computed" if value is None else f"{float(value):.4g}"


def overfitting_lines(admission: dict[str, Any]) -> list[str]:
    """The deflated Sharpe ratios of every finalist (S25 among them) with what they
    are deflated against, and the power of the t >= 3 bar against each searched
    slot's carry-forward estimate, as frozen under ``search.result_admission``."""
    parts, against = [], set()
    for slot in ("S25", "S-C", "S-R", "S-P", "S-M"):
        dsr = admission.get(slot, {}).get("dsr")
        if dsr is None:
            continue
        found = {k: v for k, v in dsr.items() if k != "deflated_against"}
        parts.append(f"{slot} " + ", ".join(f"{k} {_number(v)}" for k, v in found.items()))
        against.add(str(dsr["deflated_against"]))
    lines = ["- Deflated Sharpe ratios: " + "; ".join(parts) + " (deflated against "
             + "; ".join(sorted(against)) + ")."]  # fmt: skip
    power = []
    for slot in ("S-C", "S-R"):
        found = admission.get(slot, {}).get("power_at_bar")
        if found is not None:
            power.append(f"{slot} {_number(found.get('power'))} against a carry-forward net of "
                         f"{_number(found.get('carry_forward_net'))} bp (sigma_D "
                         f"{_number(found.get('sigma_D'))}, n_eff {_number(found.get('n_eff'))})")  # fmt: skip
    lines.append(
        "- Power of the t >= 3 bar: " + ("; ".join(power) if power else "not computed") + "."
    )
    inputs = []
    for slot in ("S25", "S-C", "S-R", "S-P", "S-M"):
        for series, found in (admission.get(slot, {}).get("dsr_inputs") or {}).items():
            if found:
                inputs.append(f"{slot} {series}: variance {_number(found.get('variance'))}, "
                              f"SR0 {_number(found.get('benchmark_sr0'))}, Sharpe ratio "
                              f"{_number(found.get('sharpe'))}, {found.get('days')} days")  # fmt: skip
    if inputs:
        lines.append("- Beside each deflated Sharpe ratio: " + "; ".join(inputs) + ".")
    return lines


def lacking_a_class_line(fits: list[dict[str, Any]]) -> str:
    """The stage-A fits whose training labels lacked a class, with the side."""
    if not fits:
        return "- Fits that lacked a class: none."
    found = "; ".join(f"{f.get('fold')} {f.get('configuration')} ({', '.join(f.get('side') or [])}"
                      f"{' ' + f['note'] if f.get('note') else ''})" for f in fits)  # fmt: skip
    return f"- Fits that lacked a class, with the side: {len(fits)}: {found}."


def second_detector_line(found: dict[str, Any]) -> str:
    """The second detector's run over D (``validation/changepoint.detect`` on the 1 s
    grid), reported beside R and never defining ``b``."""
    alarms = list(found.get("alarms", []))
    first = f", the first at {alarms[0]['timestamp']}" if alarms else ""
    note = f" ({found['note']})" if found.get("note") else ""
    return (f"- The second detector on D: {len(alarms)} alarms over {found.get('days_read', 0)} "
            f"days read{first}{note}.")  # fmt: skip


def reader_test_lines(result: dict[str, Any]) -> list[str]:
    """The ``ob200`` reader test in the amendment: each day's outcome and stream
    statistics beside D's medians (every finding is in the frozen configuration)."""
    d = result["D"]["stream"]
    lines = [f"- The ob200 reader test against max_sequence_gaps = {result['max_sequence_gaps']} "
             f"(D's medians: {d['messages']['median']} messages, {d['median_interval_ms']['median']} "
             f"ms between messages, {d['sequence_gaps']['median']} sequence gaps):"]  # fmt: skip
    for day, entry in sorted(result["days"].items()):
        outcome = ("passes" if entry["passes"] else "fails on its gap count alone"
                   if entry["fails_on_gaps_alone"] else f"fails: {entry['exclusions']}")  # fmt: skip
        stream = entry["stream"]
        lines.append(f"  {day}: {outcome}; {stream.get('messages')} messages, "
                     f"{stream.get('median_interval_ms')} ms, {stream.get('sequence_gaps')} gaps.")  # fmt: skip
    return lines


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


#: The registered dry run's stand-in for H: D's last 44 days.
STAND_IN_H = (date(2025, 5, 18), date(2025, 6, 30))


class FrozenValues:
    """The values the amendment will freeze, read as the frozen registration gives them."""

    def __init__(self, values: dict[tuple[str, ...], Any]) -> None:
        self.values = values

    def value(self, path: tuple[str, ...]) -> Any:
        return self.values[path]


def rehearsal_read(block: Any, statuses: Any) -> dict[str, Any]:
    """One stand-in block's read as ``reads.compute_results`` computes it: the
    usable days under the frozen threshold (or the block unreadable), every
    slot's series with its kills and placebos, the gap days put back, the MDEs,
    the statuses from ``statuses(series, info, gaps)`` and the pages' numbers."""
    from trading_research.pipeline.short_horizon.heldout import compute_block
    from trading_research.pipeline.short_horizon.page_numbers import block_numbers

    if block.validated["unreadable"]:
        return {"unreadable": True, "excluded": block.validated["reasons"], "found": None}
    series, info = compute_block(block)
    gaps = (
        compute_block(block.put_back(), days=block.gap_days, kills=False)[0]
        if block.gap_days
        else {}
    )
    mdes = block.value(("metrics", "mde", "value")) or {}
    for slot, s in series.items():
        s.mde = mdes.get(slot)
    found = statuses(series, info, gaps)
    found["pages"] = block_numbers(block, series, found)
    return {"unreadable": False, "info": info, "found": found}


def rehearse(study: Study, inputs: Any, values: dict[tuple[str, ...], Any]) -> dict[str, Any]:
    """The registered dry run, before the amendment: the parity test repeated on the
    2024 days (a failure is a bug, and D is re-run in full), then the block reads'
    whole computation, under the development access, on D's last 44 days standing
    in for H and on the three ``ob200`` reader-test days standing in for F and then
    for P, each read on the stand-in before it (statuses, kills, placebos, R, P's
    reading and the pages' numbers). Its outputs are kept apart and never
    reported; the record says only that it ran."""
    from trading_research.data import study_access as sa
    from trading_research.pipeline.short_horizon import reads
    from trading_research.pipeline.short_horizon.heldout import Block
    from trading_research.pipeline.short_horizon.stage0 import load_timing_day, parity

    holds = sorted({parse_key(k).hold_s for k in study.keys})
    gate = parity([load_timing_day(study, d)[1] for d in study.scale.timing_days], holds)
    if not gate["ok"]:
        raise RuntimeError("the dry run's repeat of the walk-table parity test failed: a bug under "
                           "the amendment protocol, and D is re-run in full")  # fmt: skip
    access, end = sa.development_access(), sa.BLOCKS["D"][1]
    frozen = FrozenValues(values)
    days = tuple(d for d in inputs.usable("D") if STAND_IN_H[0] <= d <= STAND_IN_H[1])
    reader = study.scale.days(sa.READER_TEST)

    def block(name: str, planned: Any) -> Block:
        return Block(
            study, name, access, planned, values=values, end=end, label=f"rehearsal_{name}"
        )

    h_block, f_block, p_block = block("H", days), block("F", reader), block("P", reader)
    h = rehearsal_read(h_block, lambda s, i, g: reads.statuses_h(study, s, i, g, h_block.flags_ns))
    skipped: dict[str, Any] = {"unreadable": None, "found": None,
                               "skipped": "the stand-in before it was unreadable"}  # fmt: skip
    f, p = skipped, skipped
    if h["found"] is not None:
        f = rehearsal_read(f_block, lambda s, i, g: reads.statuses_f(
            study, s, f_block.detector["alarms"], frozen, i, g, f_block.flags_ns, h=h["found"]))  # fmt: skip
    if f["found"] is not None:
        p = rehearsal_read(p_block, lambda s, i, g: reads.statuses_p(
            study, s, i, g, p_block.flags_ns, f=f["found"]))  # fmt: skip
    outputs = {"stand_in_H": h, "stand_in_F": f, "stand_in_P": p}
    save_json(study.dir("E", "rehearsal") / "discarded_outputs.json", outputs)
    return {"ran": True, "parity_repeated_ok": True, "stand_in_H_days": len(days),
            "stand_in_F_and_P_days": len(reader),
            "read": {name: ("unreadable" if out["unreadable"] else "skipped" if out["unreadable"] is None
                            else "read") for name, out in outputs.items()},
            "outputs": "discarded, never reported"}  # fmt: skip


def run(study: Study) -> dict[str, Any]:
    """Stage E's record: the frozen fits and files, the MDEs, the detector, the amendment."""
    from trading_research.pipeline import short_horizon_registration as shr
    from trading_research.pipeline.short_horizon.inputs import Inputs
    from trading_research.pipeline.short_horizon.pages import publish_or_note
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
        "trials": complete_trials(study, inputs, record_d),
    }  # fmt: skip
    values = placeholder_values(study, parts)
    rehearsal = rehearse(study, inputs, values)
    written = write_amendment(study, values)
    pages = publish_or_note(study)
    return plain({"commit": study.commit(), "fits": len(fits["weights_sha256"]),
                  "weights_sha256": fits["weights_sha256"], "finalists": parts["finalists"],
                  "fitted_values_sha256": fitted_sha, "mde": parts["mdes"], "detector": parts["detector"],
                  "trials": parts["trials"], "amendment": written, "dry_run": rehearsal, "pages": pages,
                  "next": "review and commit the frozen configuration, the amendment, the trial ledger's "
                  "manifest, the results files and the pages; then open H"})  # fmt: skip
