"""Stage 0 (the reader-test days), the code-freeze check, stage 1 (W and D copied) and the run log.

**Stage 0**: the signal's files and the three reader-test days are copied; each
day is loaded through the loader and checked (the print contract, both
aggressor sides present, the tick 0.01); the features, the signal and every
slot's every run are computed and timed; the fill rule is checked on every
resting order of every slot's run under each of the three rules and on probe
orders (:mod:`~trading_research.spot_maker.checks`), and every run's void
conditions (the account's identity among them) must be empty. Only the checks
and the timings are kept; every other number is discarded.

**The code freeze** (:func:`require_code_freeze`): the note's latest commit is
an ancestor of HEAD and nothing of the study code changed since.

**Stage 1**: W and D copied inside the development access, each day loaded for
its exclusions, the signal computed on D's usable days (its decisions counted
under the work directory, nothing reported); a row of the run log.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from trading_research.market_making.prereg import HeldOutLocked
from trading_research.spot.state import write_json
from trading_research.spot_maker import blocks as bk
from trading_research.spot_maker import copies as cp
from trading_research.spot_maker import day as dd
from trading_research.spot_maker import registration as rg
from trading_research.spot_maker.checks import audit_day, reader_proof, rule_checks
from trading_research.spot_maker.context import MakerStudy, signal_days
from trading_research.spot_maker.ledger import code_freeze_commit
from trading_research.spot_maker.signal import copy_signal, day_features, decide, load_signal
from trading_research.spot_maker.touch import Proxies


def log_run(repo: Path, run: str, stage: str, days: Sequence[date], note: str = "") -> None:
    """Append a row to the run log, before the computation it records."""
    path = Path(repo) / rg.RUN_LOG_PATH
    rows = json.loads(path.read_text("utf-8")) if path.exists() else []
    rows.append({"date": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "run": run,
                 "stage": stage, "days": [d.isoformat() for d in days], "note": note})  # fmt: skip
    write_json(path, rows)


def require_code_freeze(study: MakerStudy) -> str:
    note = Path(study.repo) / rg.CODE_FREEZE_NOTE
    commit = code_freeze_commit(note.read_text("utf-8")) if note.exists() else None
    if commit is None:
        raise HeldOutLocked(f"no code-freeze note at {rg.CODE_FREEZE_NOTE}")

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["git", *args], cwd=study.repo, capture_output=True, text=True)

    if git("merge-base", "--is-ancestor", commit, "HEAD").returncode != 0:
        raise HeldOutLocked(f"HEAD does not descend from the code-freeze commit {commit}")
    changed = git("diff", "--name-only", commit, "--", *rg.CODE_PATHS).stdout.split()
    if changed:
        raise HeldOutLocked(f"the study code changed since the code freeze: {changed[:3]}")
    return git("rev-parse", commit).stdout.strip()


def require_config_pinned(study: MakerStudy, commit: str) -> None:
    """The configuration unchanged outside its placeholders since ``commit``."""
    from trading_research.market_making.prereg import unfrozen_text

    then = subprocess.run(["git", "show", f"{commit}:{rg.CONFIG_PATH.as_posix()}"], cwd=study.repo,
                          capture_output=True, text=True, check=True).stdout  # fmt: skip
    now = (Path(study.repo) / rg.CONFIG_PATH).read_text("utf-8")
    if unfrozen_text(then) != unfrozen_text(now):
        raise HeldOutLocked("the configuration changed outside its placeholders since the freeze")


def _check_day(study: MakerStudy, loaded: cp.DayInput, signal: Any,
               took: dict[str, float]) -> dict[str, Any]:  # fmt: skip
    """One usable reader-test day: the signal, every run and the fill rule's checks,
    timed; every number but the checks discarded."""
    began = time.perf_counter()
    frame = day_features(loaded.frame, loaded.history)
    took["features_s"] = time.perf_counter() - began
    began = time.perf_counter()
    decisions = decide(signal, frame, loaded.scored_from)
    took["signal_s"] = time.perf_counter() - began
    tape = loaded.tape()
    ctx = dd.DayContext(tape, Proxies.from_tape(tape, loaded.carried), decisions, study.table())
    began = time.perf_counter()
    record = dd.day_record(ctx, warm=loaded.warm)
    took["runs_s"] = time.perf_counter() - began
    began = time.perf_counter()
    orders = audit_day(ctx)
    probes = rule_checks(tape, ctx.proxies)
    took["checks_s"] = time.perf_counter() - began
    runs = [r for s in orders.values() for r in s.values()]
    voids = sorted({v for s in record["slots"].values() for v in s["void"]}
                   | {v for r in runs for v in r["voids"]})  # fmt: skip
    return {"fill_rule_orders": {"orders": sum(r["orders"] for r in runs),
                                 "violations": sum(r["violations"] for r in runs)},
            "fill_rule_probes": probes, "voids": voids}  # fmt: skip


def stage_0(study: MakerStudy) -> dict[str, Any]:
    """The reader-test days: copies, the loader, the signal, every run, the rule's checks."""
    disk = study.disk_check("0")
    source, days = study.require_source(), study.days(bk.READER_TEST)
    began = time.perf_counter()
    copy_signal(source, study.repo, study.binding)
    copies = cp.copy_days(source, study.repo, days, bk.reader_test_access("readers' proof"))
    timings: dict[str, Any] = {"copy_s": time.perf_counter() - began, "days": {}}
    access = bk.reader_test_access("stage 0 timing")
    signal = load_signal(study.repo, study.binding)
    results: dict[str, Any] = {}
    clock = time.perf_counter()
    for loaded in cp.walk(study.repo, days, access, study.table()):
        took: dict[str, float] = {"load_s": time.perf_counter() - clock}
        found = reader_proof(loaded)
        found["copy"] = copies[loaded.day.isoformat()]["status"]
        found["passes"] = False
        if loaded.usable:
            checks = _check_day(study, loaded, signal, took)
            found["checks"] = checks
            found["passes"] = bool(found["contract"] and found["sides_present"]
                                   and found["tick_registered"]
                                   and checks["fill_rule_probes"]["passes"]
                                   and checks["fill_rule_orders"]["violations"] == 0
                                   and not checks["voids"])  # fmt: skip
        results[loaded.day.isoformat()] = found
        timings["days"][loaded.day.isoformat()] = took
        clock = time.perf_counter()
    computed = [t for t in timings["days"].values() if "runs_s" in t]
    for key in ("load_s", "features_s", "signal_s", "runs_s", "checks_s"):
        timings[f"{key[:-2]}_s_a_day"] = sum(t[key] for t in computed) / max(len(computed), 1)
    return {"reader_test": results, "timings": timings, "disk": disk,
            "passes": all(r["passes"] for r in results.values()),
            "discarded": "every label, fill, trade and number of the days but these checks"}  # fmt: skip


def stage_1(study: MakerStudy) -> dict[str, Any]:
    """W and D copied inside the development access; each day's exclusions; the signal
    computed on D's usable days."""
    commit = require_code_freeze(study)
    require_config_pinned(study, commit)
    disk = study.disk_check("1")
    days = [*study.days("W"), *study.days("D")]
    access = bk.development_access()
    log_run(study.repo, "stage 1", "1", study.days("D"),
            "copies, the loader's exclusions and the signal on D's days")  # fmt: skip
    copies = cp.copy_days(study.require_source(), study.repo, days, access)
    excluded = [[d.day.isoformat(), r] for d in cp.walk(study.repo, days, access, study.table())
                for r in d.reasons]  # fmt: skip
    bad = {day for day, _ in excluded}
    usable = [d for d in study.days("D") if d.isoformat() not in bad]
    signal = signal_days(study, usable, access, study.work / "1" / "signal")
    return {"code_freeze": commit, "disk": disk, "excluded": excluded,
            "copies": {"days": len(copies),
                       "absent": sorted(k for k, v in copies.items() if v["status"] == "absent"),
                       "bytes": sum(int(v.get("bytes", 0)) for v in copies.values())},
            "signal": {"days": len(signal),
                       "computed": sum(1 for r in signal if r.get("usable"))}}  # fmt: skip
