"""Stage D (every run of every slot on D, sigma_D, the MDEs, the rehearsal) and the D amendment.

**D**, under the code-freeze commit stage 1 ran under: a run-log row first, then
every usable D day's record (:func:`~trading_research.spot_maker.context.compute_days`),
each slot's development summary, sigma_D (ddof 1 over D's days with a fill), the
share of D's usable days with a fill and the MDE ``(3 + 0.84) sigma_D / sqrt(75 x
share)``; and **the rehearsal**, H's status rules applied to July with the MDEs
about to be frozen, labelled rehearsal. D is in-sample for the signal; nothing
is chosen.

**The amendment** writes every placeholder (:data:`~trading_research.spot_maker.registration.PLACEHOLDERS`)
with its value, the run log's manifest, and the amendment's entry appended to
the document with the frozen configuration's sha256; the author reviews and
commits them.
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np

from trading_research.evaluation.significance import minimum_detectable_effect
from trading_research.market_making.prereg import sha256_bytes
from trading_research.spot.state import write_json
from trading_research.spot_maker import blocks as bk
from trading_research.spot_maker import calibration
from trading_research.spot_maker import registration as rg
from trading_research.spot_maker import statuses as st
from trading_research.spot_maker.context import MakerStudy, compute_days
from trading_research.spot_maker.ledger import run_log_digest
from trading_research.spot_maker.stages import log_run, require_code_freeze, require_config_pinned

H_DAYS = bk.BLOCK_DAYS["H"]
SUMMARY_KEYS: tuple[str, ...] = (
    "days", "usable_days", "mean", "t", "newey_west_t", "median", "share_positive", "fills",
    "fills_a_day", "net_bp_clip_day", "gross_bp_fill", "making_part", "spread_part",
    "capacity", "fill_ratio", "touch", "delayed", "latency",
)  # fmt: skip


TIMINGS: tuple[str, ...] = ("load_s_a_day", "features_s_a_day", "signal_s_a_day", "runs_s_a_day",
                           "checks_s_a_day")  # fmt: skip


def _g(value: Any) -> str:
    return "n/a" if value is None or not math.isfinite(float(value)) else f"{float(value):.4g}"


def calibration_line(found: dict[str, Any]) -> str:
    if found.get("status") != "run":
        return f"{found.get('status')}; every verdict labelled \"{found.get('label')}\""
    parts = [f"{s}: fills ordered on {_g(found[s]['ordering_share'])} and the prints rule's making "
             f"part inside the book's bracket on {_g(found[s]['bracket_share'])} of {found[s]['days']} "
             f"days" for s in rg.CALIBRATION_SLOTS]  # fmt: skip
    verdict = "passes" if found["passes"] else f"does not pass; every verdict labelled \"{found['label']}\""
    return "; ".join(parts) + f"; thinned to {found['thinned_to']} prints a day; it {verdict}"


def sigma_and_mde(records: list[dict[str, Any]], slot: str) -> tuple[float, float, float]:
    """``(sigma_D, the share of usable days with a fill, the MDE)``; 0 where undefined."""
    usable = [r for r in records if r.get("usable")]
    values = [r["slots"][slot]["value"] for r in usable]
    series = np.asarray([v for v in values if v is not None], dtype=np.float64)
    series = series[np.isfinite(series)]
    sigma = float(series.std(ddof=1)) if len(series) > 1 else 0.0
    share = len(series) / len(usable) if usable else 0.0
    mde = minimum_detectable_effect(sigma, H_DAYS * share) if share > 0 else 0.0
    return sigma, share, (mde if math.isfinite(mde) else 0.0)


def short_summary(summary: dict[str, Any]) -> dict[str, Any]:
    out = {k: summary.get(k) for k in SUMMARY_KEYS}
    out["placebo_percentile"] = st.placebo_percentile(summary)
    return out


def stage_d(study: MakerStudy, stage_1: dict[str, Any]) -> dict[str, Any]:
    commit = require_code_freeze(study)
    if commit != stage_1["code_freeze"]:
        raise RuntimeError(f"stage 1 ran under {stage_1['code_freeze']}, the note now names {commit}")
    require_config_pinned(study, commit)
    disk = study.disk_check("D")
    days = list(study.days("D"))
    log_run(study.repo, f"D{study.run}", "D", days, "every run of every slot")
    records = compute_days(study, days, bk.development_access(),
                           study.work / f"run{study.run}" / "D")  # fmt: skip
    summaries = {s: st.summarise(records, s) for s in rg.FAMILY}
    mde = {s: sigma_and_mde(records, s) for s in rg.FAMILY}
    prints = [int(r["prints"]) for r in records if r.get("usable")]
    count = int(np.median(prints)) if prints else 0
    log_run(study.repo, f"C{study.run}", "D", calibration.calibration_days(),
            "the fill rule's calibration on the book days")  # fmt: skip
    calibrated = calibration.run_calibration(
        study.repo, study.book_data, count, study.binding,
        study.work / f"run{study.run}" / "calibration", study.workers)  # fmt: skip
    july = [r for r in records if date.fromisoformat(r["day"]).month == 7]
    rehearsal_summaries = {s: st.summarise(july, s) for s in rg.FAMILY}
    rehearsal = st.assess(rehearsal_summaries, "H", {s: v[2] for s, v in mde.items()})
    return {
        "run": f"D{study.run}", "code_freeze": commit, "disk": disk,
        "usable_days": sum(1 for r in records if r.get("usable")),
        "summaries": {s: short_summary(v) for s, v in summaries.items()},
        "sigma_d": {s: v[0] for s, v in mde.items()},
        "share_with_fill": {s: v[1] for s, v in mde.items()},
        "mde": {s: v[2] for s, v in mde.items()},
        "rehearsal": {"labelled": "rehearsal (July standing in for H); discarded",
                      "statuses": {s: v["status"] for s, v in rehearsal.items()}},
        "calibration": calibrated,
    }  # fmt: skip


def amendment_values(stage_0: dict[str, Any], stage_1: dict[str, Any], d: dict[str, Any],
                     repo: Path, today: date) -> dict[tuple[str, ...], Any]:  # fmt: skip
    """Every placeholder's value."""
    copies = (Path(repo) / rg.COPY_LOG_PATH).read_bytes()
    disk = [stage_0["disk"], stage_1["disk"], d["disk"]]
    record = {
        "stage_0": {"timings_s": stage_0["timings"], "passes": stage_0["passes"]},
        "disk": disk, "copies": {"sha256": sha256_bytes(copies), **stage_1["copies"]},
        "code_freeze": {"commit": d["code_freeze"]}, "trials": run_log_digest(repo),
        "d_run": d["run"],
    }  # fmt: skip
    result_d = {"labelled": "development; D is in-sample for the signal", **d["summaries"],
                "rehearsal": d["rehearsal"]["statuses"]}  # fmt: skip
    return {
        ("data", "reader_test_result"): stage_0["reader_test"],
        ("data", "excluded_days_D"): stage_1["excluded"],
        ("metrics", "mde", "sigma_d"): d["sigma_d"],
        ("metrics", "mde", "value"): d["mde"],
        ("measurements", "result_D"): result_d,
        ("calibration", "result"): d["calibration"],
        ("compute", "record"): record,
        ("amendment", "date"): today.isoformat(),
    }


def amend(study: MakerStudy, stage_0: dict[str, Any], stage_1: dict[str, Any],
          d: dict[str, Any]) -> dict[str, Any]:  # fmt: skip
    """Freeze the configuration, write the manifest and append the amendment."""
    commit = require_code_freeze(study)
    require_config_pinned(study, commit)
    today = datetime.now(UTC).date()
    values = amendment_values(stage_0, stage_1, d, study.repo, today)
    from trading_research.spot.state import plain

    values = {k: plain(v) for k, v in values.items()}
    config = Path(study.repo) / rg.CONFIG_PATH
    frozen = rg.freeze_text(config.read_text("utf-8"), values)
    config.write_text(frozen, encoding="utf-8")
    digest = sha256_bytes(frozen.encode())
    write_json(Path(study.repo) / rg.TRIALS_MANIFEST_PATH, values[("compute", "record")]["trials"])
    lines = [f"### {today.isoformat()}: the D amendment", "",
             "Written by the entry point (`experiments/spot_maker_2026.py amend`) and reviewed "
             "before it is committed. Every frozen value is in the configuration's placeholders.",
             "", f"Frozen configuration sha256: `{digest}`", "",
             f"- Complete D run {d['run']}; the run log's manifest "
             f"`{rg.TRIALS_MANIFEST_PATH.as_posix()}`: {values[('compute', 'record')]['trials']}.",
             f"- The code-freeze commit stage 1, D and this amendment ran under: `{commit}`.",
             f"- Days excluded on W and D: {stage_1['excluded'] or 'none'}.",
             f"- The reader-test days pass: {stage_0['passes']}.",
             "- Stage 0's timings a day: " + ", ".join(
                 f"{k[:-8]} {_g(stage_0['timings'].get(k))} s" for k in TIMINGS) + "; free disk "
             + ", ".join(f"{x['stage']} {_g(x['free_gb'])} GB" for x in values[("compute", "record")]["disk"]) + ".",
             "- sigma_D and the MDE (development, bp per fill): "
             + "; ".join(f"{s} {_g(d['sigma_d'][s])} and {_g(d['mde'][s])}" for s in rg.FAMILY) + ".",
             "- D's means (development, in-sample for the signal; never a result): "
             + "; ".join(f"{s} {_g(d['summaries'][s]['mean'])}" for s in rg.FAMILY) + ".",
             "- The fill rule's calibration on the book days: "
             + calibration_line(d["calibration"]) + ".",
             "- The rehearsal on July (H's rules with these values; discarded): "
             + "; ".join(f"{s} {d['rehearsal']['statuses'][s]}" for s in rg.FAMILY) + "."]  # fmt: skip
    document = Path(study.repo) / rg.DOCUMENT_PATH
    text = document.read_text("utf-8").rstrip("\n")
    document.write_text(text + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    return {"frozen_sha256": digest, "amendment": lines}
