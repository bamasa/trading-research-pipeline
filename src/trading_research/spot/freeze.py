"""The D amendment: the frozen files, the rehearsal, the run log's manifest, the frozen configuration.

Registered ("Amendment protocol", the procedure's steps 6 and 7):

* **The frozen files** under ``models/spot_2026/`` (ignored by git): every
  frozen fit (``weights/<fit key>``, copied from the frozen fold's fits at
  their sha256), the fitted-values file (W's touch size), and one slot file per
  row (``slots/<row>.json``: its configuration, weights, thresholds and fitted
  values); S25-T's records the 2025 files it loads, each checked by sha256.
* **The rehearsal**: the read's whole computation (:func:`trading_research.spot.blockread.compute`)
  on July's usable days standing in for H, under the development access, with
  the values about to be frozen; its outputs are kept apart and never reported.
* **The run log's manifest** (``experiments/results/spot_2026_trials.json``),
  its sha256 and row count frozen under ``compute.record.trials``.
* **The amendment**: every placeholder filled
  (:func:`trading_research.spot.registration.freeze_text`) and a dated section
  appended to the document with the line ``Frozen configuration sha256:
  `<hash>```. Both are written for review; committing them, the manifest and
  nothing else is the author's act.
"""

from __future__ import annotations

import math
import shutil
from collections.abc import Mapping
from datetime import UTC, date, datetime
from typing import Any

import yaml

from trading_research.market_making.prereg import sha256_bytes
from trading_research.spot import access as sa
from trading_research.spot import registration as sr
from trading_research.spot import runlog
from trading_research.spot.context import SpotStudy
from trading_research.spot.fits import KEY_2025, check_2025
from trading_research.spot.ledger import (
    FITTED_VALUES_PATH,
    SLOT_FILES_DIR,
    WEIGHTS_DIR,
    file_sha256,
)
from trading_research.spot.plan import ROWS
from trading_research.spot.registration import FAMILY
from trading_research.spot.state import plain, write_json


class FrozenValues:
    """The values the amendment is about to freeze, read as a frozen registration's."""

    def __init__(self, values: Mapping[tuple[str, ...], Any]) -> None:
        self.values = dict(values)

    def value(self, path: tuple[str, ...]) -> Any:
        return self.values[path]


def frozen_files(
    study: SpotStudy, record1: Mapping[str, Any], fitted: Mapping[str, Any]
) -> dict[str, Any]:
    """Write every frozen file; their sha256s."""
    weights: dict[str, str] = {}
    source = study.work / "D" / "fits" / "frozen"
    for key, entry in sorted(fitted["fits"].items()):
        if not entry.get("sha256"):
            continue
        target = study.repo / WEIGHTS_DIR / key
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / key, target)
        if file_sha256(target) != entry["sha256"]:
            raise RuntimeError(f"the frozen fit {key} differs from the one the D stage wrote")
        weights[key] = entry["sha256"]
    fitted_sha = write_json(study.repo / FITTED_VALUES_PATH, {
        "touch_size": float(record1["touch_size"]), "raw_size_columns": ["ofi_1", "ofi_20"],
        "source": "the median of bid_size_0 and ask_size_0 over every W row of Bybit's spot book",
    })  # fmt: skip
    slots: dict[str, str] = {}
    for name, row in ROWS.items():
        record: dict[str, Any] = {"row": name, "instrument": row.instrument, "grid": row.grid,
                                  "inputs": list(row.inputs), "walk": int(record1["walk"]),
                                  "max_sequence_gaps": int(record1["max_sequence_gaps"])}  # fmt: skip
        if row.fitted:
            keys = [k for k in weights if k.startswith(f"{name}__")]
            record |= {"cells": fitted["thresholds"][name],
                       "weights_sha256": {k: weights[k] for k in keys},
                       "fitted_values_sha256": fitted_sha}  # fmt: skip
        else:
            record |= s25_t_slot(study)
        slots[name] = write_json(study.repo / SLOT_FILES_DIR / f"{name}.json", record)
    return {"weights": weights, "fitted_values": fitted_sha, "slots": slots}


def s25_t_slot(study: SpotStudy) -> dict[str, Any]:
    """S25-T's slot record: the 2025 files it loads and their thresholds, unchanged."""
    check_2025(study)
    config = yaml.safe_load((study.repo / study.binding.config).read_text(encoding="utf-8"))
    thresholds = config["slots"]["S25"]["thresholds"]
    cells = {c.key: thresholds[KEY_2025.format(h=c.hold_s, r=c.rate, f=c.fee_bp)]
             for c in ROWS["S25-T"].cells}  # fmt: skip
    return {"cells": cells, "config_2025_sha256": study.binding.config_sha256,
            "files_2025_sha256": {str(k): v for k, v in study.binding.files.items()}}  # fmt: skip


def number(value: Any) -> float:
    """A frozen number: an undefined one (no D day with a trade) is frozen as 0."""
    return float(value) if value is not None and math.isfinite(float(value)) else 0.0


def excluded_days(
    record1: Mapping[str, Any], record_d: Mapping[str, Any]
) -> dict[str, list[list[Any]]]:
    """``excluded_days_D``: every W and D day excluded on a venue, with each reason."""
    development = set(sa.block_days("W")) | set(sa.block_days("D"))
    out: dict[str, set[tuple[date, str]]] = {i.key: set() for i in sa.INSTRUMENTS.values()}
    for key, items in record1["excluded"].items():
        out[key].update((date.fromisoformat(d), r) for d, r in items)
    for name, days in record_d["excluded_on_scoring"].items():
        for day, reasons in days.items():
            out[sa.instrument(name).key].update((date.fromisoformat(day), r) for r in reasons)
    return {k: [[d, r] for d, r in sorted(v) if d in development] for k, v in out.items()}


def today() -> date:
    return datetime.now(UTC).date()


def result_d(record_d: Mapping[str, Any]) -> dict[str, Any]:
    """``measurements.result_D``: each row's D series summary against the two nulls and
    its trade rate, and both calibrations on D, labelled development."""
    keep = ("days", "days_with_trade", "mean", "t", "newey_west_t", "median", "trades_per_day",
            "gross_bp_trade", "fee_break_even_bp", "ic", "placebo_percentile")  # fmt: skip
    out: dict[str, Any] = {"labelled": "development"}
    for row, found in record_d["series"].items():
        out[row] = {k: found.get(k) for k in keep} | {
            "null_a_p": record_d["null_a"][row]["p"], "null_b_p": record_d["null_b"][row]["p"],
            "registered_trades_per_day": 240}  # fmt: skip
    calib = record_d["calibration"]
    rule = calib.get("fill_s", {}).get("S25-PB", {}).get("rule", [])
    out["FILL-S"] = {"S25-PB_rule_mean_advantage_bp": {r["side"]: r["mean_advantage_bp"] for r in rule},
                     "a_buy": calib.get("a_buy"), "a_sell": calib.get("a_sell")}  # fmt: skip
    return plain(out)


def compute_record(study: SpotStudy, record0: Mapping[str, Any], record1: Mapping[str, Any],
                   trials: Mapping[str, Any]) -> dict[str, Any]:  # fmt: skip
    """``compute.record``: stage 0's timings and checks, free disk at each stage's start,
    the replay's peak memory, the worker count, the stores and the run log."""
    peaks = [
        r.stats.get("peak_rss_bytes") for s in (0, 1) for r in study.fetch_log(s).records.values()
    ]
    return plain({
        "stage_0": {"timings_s": record0["timings"], "walk": record0["walk"],
                    "print_rule": record0["print_rule"],
                    "message_interval_ms": record0["message_interval_ms"]},
        "disk": [{k: d.get(k) for k in ("stage", "at", "free_gb", "passed")} for d in record1["disk"]],
        "replay_peak_rss_bytes": max([int(p) for p in peaks if p] or [0]),
        "workers": study.workers, "stores": record1["stores"],
        "code_freeze": {"commit": str(record1["code_freeze"]["commit"])},
        "trials": {"sha256": trials["sha256"], "rows": int(trials["rows"])},
    })  # fmt: skip


def placeholder_values(study: SpotStudy, files: Mapping[str, Any], trials: Mapping[str, Any] | None
                       ) -> dict[tuple[str, ...], Any]:  # fmt: skip
    """Every placeholder's frozen value (``compute.record`` once the run log is closed)."""
    record0, record1, record_d = (study.state.record(s) for s in ("0", "1", "D"))
    frozen = record_d["thresholds"]["frozen"]
    mde = record_d["mde"]
    values: dict[tuple[str, ...], Any] = {
        ("data", "reader_test_result"): plain(record0["reader_test_result"]),
        ("data", "max_sequence_gaps"): int(record1["max_sequence_gaps"]),
        ("data", "excluded_days_D"): excluded_days(record1, record_d),
        ("data", "instrument_spec"): plain(record0["instrument_spec"]),
        ("common", "book_levels", "walk"): int(record1["walk"]),
        ("features", "fitted_values_sha256"): files["fitted_values"],
        ("models", "weights_sha256"): dict(files["weights"]),
        ("models", "thresholds"): {row: frozen[row] for row in ("S25-PF", "S25-BB", "S25-PB", "S25-PFT")},
        ("metrics", "mde", "sigma_d"): {s: number(mde[s]["sigma_d"]) for s in FAMILY},
        ("metrics", "mde", "value"): {s: number(mde[s]["mde"]) for s in FAMILY},
        ("measurements", "result_D"): result_d(record_d),
        ("amendment", "date"): today(),
        ("artefacts", "frozen_slot_files_sha256"): dict(files["slots"]),
    }  # fmt: skip
    if trials is not None:
        values[("compute", "record")] = compute_record(study, record0, record1, trials)
    return values


def rehearse(study: SpotStudy, values: Mapping[tuple[str, ...], Any]) -> dict[str, Any]:
    """The registered dry run before the amendment: the read's whole computation on
    July's usable days standing in for H, under the development access; its outputs
    are kept apart and never reported."""
    from trading_research.spot import blockread as br
    from trading_research.spot import heldout as ho
    from trading_research.spot.development import usable_days
    from trading_research.spot.stage1 import spec_of

    record1 = study.state.record("1")
    july = [d for d in usable_days(record1, list(study.scale.days("D"))) if d.month == 7]
    frozen = FrozenValues(values)
    plan = ho.ReadPlan(label="rehearsal", block="H", days=tuple(july), access=sa.development_access(),
                       models=br.frozen_models(study, frozen), max_gaps=int(record1["max_sequence_gaps"]),
                       depth=int(record1["walk"]), spec=spec_of(record1))  # fmt: skip
    found = br.compute(study, plan, blind=study.registration.h_blind,
                       mde=values[("metrics", "mde", "value")])  # fmt: skip
    write_json(study.dir("amend", "rehearsal") / "discarded_outputs.json",
               {"statuses": {s: a.record() for s, a in found["assessments"].items()},
                "measurements": found["measurements"]})  # fmt: skip
    runlog.append(study.repo, run=f"D{study.run}", stage="amend", days=july,
                  note="the rehearsal: July standing in for H, outputs discarded")  # fmt: skip
    return {"ran": True, "stand_in_days": len(july), "outputs": "discarded, never reported"}


def _f(value: Any) -> str:
    return (
        "undefined" if value is None or not math.isfinite(float(value)) else f"{float(value):.4g}"
    )


def amendment_text(study: SpotStudy, frozen: str, values: Mapping[tuple[str, ...], Any]) -> str:
    """The dated section appended to the document's amendments."""
    record_d = study.state.record("D")
    trials = values[("compute", "record")]["trials"]
    walk = values[("compute", "record")]["stage_0"]["walk"]
    excluded = values[("data", "excluded_days_D")]
    readers = values[("data", "reader_test_result")]
    lines = ["", f"### {values[('amendment', 'date')]}: the D amendment", "",
             "Written by the entry point (`experiments/spot_2026.py amend`) and reviewed before it",
             "is committed. Every frozen value is in the configuration's placeholders.", "",
             sr.frozen_hash_line(frozen), "",
             f"- Complete D run D{study.run}; the run log's manifest "
             f"`{sr.TRIALS_MANIFEST_PATH.as_posix()}`: sha256 `{trials['sha256']}`, "
             f"{trials['rows']} rows.",
             "- The code-freeze commit stage 1, D and this amendment ran under: "
             f"`{values[('compute', 'record')]['code_freeze']['commit']}`.",
             "- Days excluded on W and D: " + "; ".join(
                 f"{key} {len({d for d, _ in items})} ({', '.join(sorted({r for _, r in items})) or 'none'})"
                 for key, items in excluded.items()) + ".",
             f"- `max_sequence_gaps` = {values[('data', 'max_sequence_gaps')]}.",
             "- Tick and lot: " + "; ".join(f"{k} {v['tick']} and {v['lot']}"
                                           for k, v in values[("data", "instrument_spec")].items()) + ".",
             "- The reader-test days: " + "; ".join(
                 f"{key} {day} {'passes' if r['passes'] else 'fails'}"
                 for key, days in readers.items() for day, r in days.items()) + ".",
             f"- The walk: {_f(walk['share'])} of the reader-test rows beyond ten levels "
             f"(threshold {walk['threshold']}): the walk reads {walk['walk']} levels.",
             f"- Frozen fits: {len(values[('models', 'weights_sha256')])}; the fitted-values file's "
             f"sha256 `{values[('features', 'fitted_values_sha256')]}`.",
             "- sigma_D and the MDE (development): " + "; ".join(
                 f"{s} {_f(record_d['mde'][s]['sigma_d'])} and {_f(record_d['mde'][s]['mde'])} bp"
                 for s in FAMILY) + ".",
             "- D's series against the two nulls (development): " + "; ".join(
                 f"{r} Null A p {_f(record_d['null_a'][r]['p'])}, Null B p {_f(record_d['null_b'][r]['p'])}"
                 for r in ROWS) + ".",
             f"- The thinned calibration on D: a_buy {_f(record_d['calibration'].get('a_buy'))} bp, "
             f"a_sell {_f(record_d['calibration'].get('a_sell'))} bp.", ""]  # fmt: skip
    return "\n".join(lines)


def write_amendment(study: SpotStudy, values: Mapping[tuple[str, ...], Any]) -> dict[str, str]:
    """Fill the YAML's placeholders and append the dated amendment, for review."""
    config, document = study.repo / sr.CONFIG_PATH, study.repo / sr.DOCUMENT_PATH
    text, doc = config.read_text(encoding="utf-8"), document.read_text(encoding="utf-8")
    frozen = sr.freeze_text(text, values, null_sha256=sr.latest_null_sha256(doc))
    if sr.AMENDMENTS_HEADING not in doc:
        raise RuntimeError("the registration has no amendments section to append to")
    config.write_text(frozen, encoding="utf-8")
    document.write_text(doc.rstrip("\n") + "\n" + amendment_text(study, frozen, values),
                        encoding="utf-8")  # fmt: skip
    study.reload()
    return {"frozen_config_sha256": sha256_bytes(frozen.encode()),
            "document": sr.DOCUMENT_PATH.as_posix()}  # fmt: skip


def run(study: SpotStudy) -> dict[str, Any]:
    """The amendment stage's record (module docstring)."""
    from trading_research.spot.development import ensure_2025
    from trading_research.spot.stage1 import frozen_under

    if study.reload().frozen:
        raise RuntimeError("the configuration is already frozen: the D amendment is written once")
    freeze = frozen_under(study)
    ensure_2025(study)
    check_2025(study)
    files = frozen_files(study, study.state.record("1"), _frozen_fold(study))
    rehearsal = rehearse(study, placeholder_values(study, files, None))
    trials = runlog.write_manifest(study.repo)
    values = placeholder_values(study, files, trials)
    written = write_amendment(study, values)
    return {"files": files, "trials": trials, "rehearsal": rehearsal, "amendment": written,
            "commit": study.commit(), "code_freeze": freeze,
            "next": "review the frozen configuration, the amendment and the run log's manifest; "
                    "commit them; then open H"}  # fmt: skip


def _frozen_fold(study: SpotStudy) -> dict[str, Any]:
    import json

    path = study.work / "D" / "fits" / "frozen" / "record.json"
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded
