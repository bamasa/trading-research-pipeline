"""H, F and P: each block opened through the study's ledger and read in one session.

``open`` checks the block's preconditions and writes its open entry
(:meth:`trading_research.pipeline.short_horizon_ledger.Ledger.open`); the entry
is committed and pushed by hand before anything of the block is fetched.
``read`` then runs the whole session:

1. a process of the session (``Ledger.session``), which refuses unless the open
   entry is committed and pushed, the tree is clean but for the ledger, the
   frozen configuration and files are unchanged and the environment is H's;
2. the block's fetch through ``data.fetch_stage.run_stage`` under the session's
   access, its per-file records written into the ledger; computing waits until
   they are committed (``require_fetch_committed``), so a first ``read``
   returns after the fetch and a second one, after the commit, computes;
3. every slot's per-day values (:mod:`.heldout`), every status and kill
   (:mod:`.statuses`), the measurements, the headline sentences and the
   numbers the pages show beside the statuses (:mod:`.page_numbers`), written
   to ``experiments/results/short_horizon_2025_<block>.json``, whose sha256
   closes the session;
4. the pages, written again from everything computed so far (:mod:`.pages`):
   the sentences, the finalists page, the S25 page, the regime map's file and
   the figures. A failure there is recorded in the block's record and never
   reopens the closed session; ``pages`` writes them again.

A failure is recorded in the ledger; a read that must close as failed (the same
stage failing twice with the same exception, or past five resumptions) writes
its results file with every slot ``not assessed (<block> read failed)``.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from trading_research.data import fetch_log as fl
from trading_research.data import study_access as sa
from trading_research.pipeline.short_horizon.context import Study

#: Each block's print archives in GB, for the disk check before its fetch.
PRINT_GB = {"H": 2.1, "F": 5.9, "P": 3.5}
HEAD_CHECK = "experiments/results/short_horizon_2025_head_check.csv"
#: What a read records beside the series when nothing was computed for the measurements.
NO_INFO: dict[str, Any] = {"flag_days": [], "flags_in_block": 0}


def ledger_for(study: Study) -> Any:
    """The study's ledger; the dry run's has no upstream to push to, no locked
    library to check and a fixed environment."""
    from trading_research.pipeline.short_horizon_ledger import Ledger

    if study.venue is None:
        return Ledger(repo=study.repo)
    return Ledger(
        repo=study.repo, upstream=None, require_locked_library=False, environment=_dry_environment
    )


def _dry_environment() -> str:
    return "dry run"


def results_path(study: Study, block: str) -> Path:
    from trading_research.pipeline import short_horizon_registration as shr

    return study.repo / shr.RESULTS_PATHS[block]


def read_results(study: Study, block: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(results_path(study, block).read_text(encoding="utf-8"))
    return loaded


def qualifying_for_p(study: Study) -> list[str]:
    """The slots that passed on F, the only ones P is opened for."""
    if not results_path(study, "F").exists():
        return []
    return list(read_results(study, "F").get("qualifying_for_P", []))


def open_block(study: Study, block: str) -> dict[str, Any]:
    """Write the block's open entry after every precondition; commit and push it next."""
    from trading_research.pipeline.short_horizon.state import StageOrderError

    for earlier in ("E", *("H", "F", "P")[: ("H", "F", "P").index(block)]):
        if not study.state.complete(earlier):
            raise StageOrderError(f"{block} is opened only after stage {earlier} is complete")
    qualifying = qualifying_for_p(study) if block == "P" else []
    entry = ledger_for(study).open(
        block, catalogue_sha256=study.catalogue().sha256, qualifying=qualifying
    )
    return {
        "entry": entry,
        "next": f"commit and push {entry['block']}'s open entry, then read {block}",
    }


def fetch_block(study: Study, block: str, access: sa.StudyAccess) -> fl.FetchLog:
    """The block's planes (for H, E's hourly bars too) under the session's access."""
    from trading_research.data.disk import stage_budget
    from trading_research.data.fetch_stage import StagePlan, head_check_present, run_stage

    opening = sa.opening_days(block)
    days = study.scale.days(block)
    plan = StagePlan(book=days, prints=days, bars=opening[sa.BARS], funding=days)
    opener = study.venue
    if opener is not None:
        present, book_workers, print_workers = opener.market.present(), 1, 1
    else:
        present = head_check_present(study.repo / HEAD_CHECK)
        book_workers, print_workers = (
            int(study.record("0")["replay_memory"]["workers"]),
            max(1, study.workers),
        )
    share = len(days) / max(1, sa.BLOCK_DAYS[block])
    budget = stage_budget(
        book_days=len(days), print_archives_gb=PRINT_GB[block] * share, workers=book_workers
    )
    log = fl.FetchLog(study.dir("reads", block) / "fetch_log.json")
    run_stage(study.symbol, plan, study.roots, log, budget=budget, book_workers=book_workers,
              print_workers=print_workers, expected_present=present, opener=opener, access=access)  # fmt: skip
    if not log.complete():
        raise RuntimeError(f"{block}'s fetch returned with days neither on disk nor excluded")
    return log


def fetch_records(log: fl.FetchLog) -> dict[str, dict[str, Any]]:
    """Each fetched file's sha256 and UTC download time, keyed ``plane/day``."""
    return {fl.FetchLog.key(r.plane, date.fromisoformat(r.day)): {"sha256": r.sha256, "downloaded_at": r.downloaded_at}
            for r in log.of() if r.status == fl.ON_DISK}  # fmt: skip


def measured(series: Any, days: list[date] | None = None) -> dict[str, Any]:
    """A series' numbers on some of its days, reported as a measurement (no status)."""
    import numpy as np

    from trading_research.evaluation.significance import newey_west_t
    from trading_research.market_making.verdicts import day_t
    from trading_research.pipeline.short_horizon.statuses import _series

    mask = series.mask(series.days if days is None else days)
    values = _series(series, series.value, mask)
    found = day_t(values) if len(values) >= 2 else None
    return {"days": len(values), "trades": int(np.nansum(series.trades[mask])),
            "mean": float(values.mean()) if len(values) else None,
            "t": None if found is None else found.t,
            "newey_west_t": newey_west_t(values).t if len(values) >= 2 else None,
            "median_day": float(np.median(values)) if len(values) else None}  # fmt: skip


def beside(block: str, series: dict[str, Any], info: dict[str, Any], gaps: dict[str, Any],
           flags_ns: Any, windows: dict[str, list[date]]) -> dict[str, dict[str, Any]]:  # fmt: skip
    """What the read reports beside each slot's status, every one a measurement: its
    numbers on all of the block; for a slot with a regime window above 0, its
    regime attribution; its regime, passive-exit and control-IC measurements; on
    H, FILL on its stream; and its verdict's numbers on its status window
    (``windows``) with the days excluded for sequence gaps put back (``gaps``)."""
    from trading_research.pipeline.short_horizon.beside_verdict import verdict_measures
    from trading_research.pipeline.short_horizon.heldout_measures import attribution, measured_rows
    from trading_research.pipeline.short_horizon.statuses import merge

    drawn = [date.fromisoformat(d) for d in info["flag_days"]]
    out: dict[str, dict[str, Any]] = {}
    for slot, s in series.items():
        found: dict[str, Any] = {f"all_of_{block}": measured(s)}
        policy = f"{'guard' if s.kind == 'maker' else 'pull'}_{s.window_h}h"
        found["regime_attribution"] = attribution(
            s, flags_ns, drawn, info["flags_in_block"], policy
        )
        found["measured"] = measured_rows(s)
        window = windows.get(slot, s.days)
        found["beside_verdict"] = verdict_measures(s, window)
        if slot in gaps:
            back = gaps[slot]
            found["gap_days_included"] = measured(merge(s, back), [*window, *back.days])
            found["gap_days_included"]["gap_days"] = len(back.days)
        else:
            found["gap_days_included"] = measured(s, window) | {"gap_days": 0}
        if slot in info.get("fill_rules", {}):
            found["fill_rules"] = info["fill_rules"][slot]
        out[slot] = found
    return out


def preset_on_d(study: Study) -> dict[str, str]:
    """Statuses set on D, each entering Holm with p = 1 on every block."""
    preset = {s: v for s, v in study.record("D")["statuses_on_D"].items() if v != "admitted"}
    for slot, found in study.record("C")["admission"].items():
        if found["status_on_D"] != "admitted":
            preset[slot] = found["status_on_D"]
    return preset


def statuses_h(study: Study, series: dict[str, Any], info: dict[str, Any] | None = None,
               gaps: dict[str, Any] | None = None, flags_ns: Any = None) -> dict[str, Any]:  # fmt: skip
    """H: the family on H (slots on S25's signal on H's days to 07-31), S25-Z alone,
    what is taken to F, and every slot's numbers on all of H as measurements."""
    from trading_research.pipeline.short_horizon import statuses as st

    preset = preset_on_d(study)
    masks, windows = {}, {}
    for slot, s in series.items():
        window = [d for d in s.days if d <= st.S25_WINDOW_END] if s.on_s25_signal else s.days
        masks[slot], windows[slot] = s.mask(window), window
    found = st.family(series, masks, "H", preset)
    out = {slot: a.record() for slot, a in found.items()}
    if "S25-Z" in series:
        out["S25-Z"] = st.s25_z(series["S25-Z"], masks["S25-Z"]).record()
    taken = [slot for slot, a in found.items()
             if slot not in preset and slot in series and st.taken_to_f(a, series[slot].on_s25_signal)]  # fmt: skip
    measurements = beside("H", series, info or NO_INFO, gaps or {}, flags_ns, windows)
    for slot, s in series.items():
        if not s.on_s25_signal:
            measurements[slot]["H_before_2025_08_01"] = measured(
                s, [d for d in s.days if d <= st.S25_WINDOW_END]
            )
    return {"assessments": out, "statuses": {slot: a["status"] for slot, a in out.items()},
            "taken_to_F": taken, "measurements": measurements}  # fmt: skip


def r_void(registration: Any, alarms: list[dict[str, Any]]) -> bool:
    """R is void when this run of the frozen detector does not reproduce D's alarms
    as the amendment records them (by time and channel)."""
    import pandas as pd

    recorded = {
        (pd.Timestamp(a["timestamp"]), a["channel"])
        for a in registration.value(("detector", "alarms_D"))
    }
    end = pd.Timestamp(sa.BLOCKS["D"][1], tz="UTC") + pd.Timedelta(days=1)
    again = {
        (pd.Timestamp(a["timestamp"]), a["statistic"])
        for a in alarms
        if pd.Timestamp(a["timestamp"]) < end
    }
    return recorded != again


def statuses_f(study: Study, series: dict[str, Any], alarms: list[dict[str, Any]], registration: Any,
               info: dict[str, Any] | None = None, gaps: dict[str, Any] | None = None,
               flags_ns: Any = None) -> dict[str, Any]:  # fmt: skip
    """F: the slots taken to F on F- (or all of F, labelled), R for H's candidates,
    the slots P is opened for, and every slot on all of F as a measurement."""
    import pandas as pd

    from trading_research.evaluation.family import FAMILY
    from trading_research.pipeline.short_horizon import statuses as st

    h = read_results(study, "H")
    taken, before = list(h["taken_to_F"]), dict(h["statuses"])
    days = next(iter(series.values())).days if series else []
    f_start = sa.BLOCKS["F"][0]
    in_f = sorted(
        pd.Timestamp(a["timestamp"]).date()
        for a in alarms
        if pd.Timestamp(a["timestamp"]).date() >= f_start
    )
    untestable = bool(registration.value(("detector", "r_untestable_declared")))
    void = r_void(registration, alarms)
    split = st.split_f(days, in_f[0] if in_f else None, untestable, void)
    preset = {slot: "not taken to F" for slot in FAMILY if slot not in taken}
    masks = {slot: series[slot].mask(split.minus) for slot in taken if slot in series}
    found = st.family(series, masks, "F", preset)
    after, r = {}, {}
    for slot in taken:
        base = st.after_f(before[slot], found[slot])
        after[slot] = base + (
            ", assessed on all of F" if split.on_all_of_f and base != "void" else ""
        )
        if before[slot] != "candidate":
            r[slot] = {"outcome": "not tested", "reason": "not a candidate on H"}
            continue

        def passes_before(d: date, slot: str = slot) -> bool:
            cut = {s: series[s].mask([x for x in days if x < d]) for s in masks}
            return st.family(series, cut, "F", preset)[slot].status == "passes"

        r[slot] = st.r_test(
            series[slot], days, split, untestable, found[slot].status == "passes", passes_before
        )
    qualifying = [s for s in taken if after[s].split(",")[0] in ("candidate", "exploratory: significant on F")
                  and found[s].status == "passes"]  # fmt: skip
    statuses = {slot: after.get(slot, status) for slot, status in before.items()}
    windows = {slot: list(split.minus) for slot in taken}
    return {"assessments": {s: a.record() for s, a in found.items()}, "statuses": statuses,
            "after_F": after, "split": {"b": None if split.b is None else split.b.isoformat(),
            "minus": len(split.minus), "plus": len(split.plus), "on_all_of_F": split.on_all_of_f,
            "reason": split.reason}, "R_void": void, "R": r, "qualifying_for_P": qualifying,
            "measurements": beside("F", series, info or NO_INFO, gaps or {}, flags_ns, windows)}  # fmt: skip


def statuses_p(study: Study, series: dict[str, Any], info: dict[str, Any] | None = None,
               gaps: dict[str, Any] | None = None, flags_ns: Any = None) -> dict[str, Any]:  # fmt: skip
    """P: the slots that passed on F (scored only by their mean when R is supported)."""
    from trading_research.evaluation.family import FAMILY
    from trading_research.pipeline.short_horizon import statuses as st

    f = read_results(study, "F")
    qualifying = list(f["qualifying_for_P"])
    supported = [s for s in qualifying if f["R"].get(s, {}).get("outcome") == "supported"]
    preset = {slot: "not read on P" for slot in FAMILY if slot not in qualifying}
    preset |= dict.fromkeys(supported, "scored on P only by its mean")
    masks = {slot: series[slot].mask(series[slot].days) for slot in qualifying if slot in series}
    found = st.family(series, masks, "P", preset)
    final = dict(f["statuses"])
    for slot in qualifying:
        assessment = found[slot]
        if slot in supported:
            values = st._series(series[slot], series[slot].value, masks[slot])
            assessment.mean = float(values.mean()) if len(values) else 0.0
            assessment.void = st.void_reasons(series[slot], ())
            assessment.status = "void" if assessment.void else ""
        final[slot] = st.after_p(f["after_F"][slot].split(",")[0], assessment, slot in supported)
    return {"assessments": {s: a.record() for s, a in found.items()}, "statuses": final,
            "measurements": beside("P", series, info or NO_INFO, gaps or {}, flags_ns, {})}  # fmt: skip


def sentences(block: str, statuses: dict[str, str], assessments: dict[str, Any]) -> list[str]:
    """The registered sentence of each slot with a registered status on the block."""
    from trading_research.market_making.verdicts import KillCheck
    from trading_research.reporting.headlines import HeadlineError, slot_on_block

    out = []
    for slot, status in statuses.items():
        kills = [KillCheck(**k) for k in assessments.get(slot, {}).get("kills", [])]
        try:
            out.append(slot_on_block(slot, block, status, kills))
        except HeadlineError as exc:  # a status the sentence does not carry (not taken there)
            out.append(f"{slot} on {block}: {status} ({exc})")
    return out


def regime_segments(
    series: dict[str, Any], alarms: list[dict[str, Any]], block: str
) -> list[dict[str, Any]]:
    """Each slot's daily net cut at the frozen detector's alarms: a measurement."""
    import pandas as pd

    from trading_research.reporting.regime_map import regime_map

    stamps = [pd.Timestamp(a["timestamp"]) for a in alarms]
    rows: list[dict[str, Any]] = []
    for slot, s in series.items():
        # S25-Z's net is its net at 0/0, beside its gross primary series
        net = s.second if s.kind == "zero" and s.second is not None else s.value
        daily = pd.Series(net, index=pd.Index(s.days))
        frame = regime_map(daily, stamps, blocks=dict.fromkeys(s.days, block), slot=slot)
        rows += json.loads(frame.to_json(orient="records", date_format="iso"))
    return rows


def unreadable_results(study: Study, block: str, why: str) -> dict[str, Any]:
    """Statuses when the block is unreadable or its read failed: nothing is assessed on
    it, and earlier statuses stand with the mark (H: every slot not assessed)."""
    from trading_research.evaluation.family import FAMILY

    slots = [*FAMILY, "S25-Z"]
    if block == "H":
        return {"statuses": dict.fromkeys(slots, f"not assessed (H {why})"), "taken_to_F": []}
    before = read_results(study, "H" if block == "F" else "F")
    statuses = dict(before["statuses"])
    if block == "F":
        for slot in before["taken_to_F"]:
            statuses[slot] = f"{statuses[slot]}, F {why}"
        return {"statuses": statuses, "qualifying_for_P": [], "R": {}}
    for slot in before["qualifying_for_P"]:
        base = before["after_F"][slot].split(",")[0]
        if base == "candidate":
            statuses[slot] = f"candidate, not confirmed (P {why})"
        elif base == "exploratory: significant on F":
            statuses[slot] = f"exploratory: significant on F, not confirmed (P {why})"
    return {"statuses": statuses}


def compute_results(study: Study, block: str, session: Any) -> dict[str, Any]:
    """Everything the block's read produces, statuses and kills included, and the
    numbers the pages show beside them (:mod:`.page_numbers`)."""
    from trading_research.pipeline.short_horizon.heldout import Block, compute_block
    from trading_research.pipeline.short_horizon.page_numbers import block_numbers

    blk = Block(study, block, session.access, study.scale.days(block))
    head = {"block": block, "session": list(session.key), "stamp": session.stamp,
            "scale": study.scale.name}  # fmt: skip
    if blk.validated["unreadable"]:
        return head | {"unreadable": True, "excluded": blk.validated["reasons"]} | unreadable_results(
            study, block, "unreadable")  # fmt: skip
    series, info = compute_block(blk)
    gaps = compute_block(blk.put_back(), days=blk.gap_days, kills=False)[0] if blk.gap_days else {}
    mdes = blk.value(("metrics", "mde", "value")) or {}
    for slot, s in series.items():
        s.mde = mdes.get(slot)
    flags = blk.flags_ns
    found = {"H": lambda: statuses_h(study, series, info, gaps, flags),
             "F": lambda: statuses_f(study, series, blk.detector["alarms"], blk.registration, info, gaps, flags),
             "P": lambda: statuses_p(study, series, info, gaps, flags)}[block]()  # fmt: skip
    found["sentences"] = sentences(block, found["statuses"], found["assessments"])
    if block == "H":
        found["sentences"] += regime_sentences(found["measurements"])
    found["regime_map"] = regime_segments(series, blk.detector["alarms"], block)
    found["pages"] = block_numbers(blk, series, found)
    return head | {"unreadable": False, "info": info} | found


def regime_sentences(measurements: dict[str, dict[str, Any]]) -> list[str]:
    """The regime-axis sentence of every slot with an attribution on H."""
    from trading_research.pipeline.short_horizon.heldout_measures import regime_sentence
    from trading_research.reporting.headlines import HeadlineError

    out = []
    for slot, found in measurements.items():
        row = found.get("regime_attribution")
        if row is None:
            continue
        try:
            out.append(regime_sentence(slot, row))
        except HeadlineError as exc:  # no flag inside the block: no random pause to compare with
            out.append(f"{slot} on H: its {row['window_h']} h guard has no regime sentence ({exc})")
    return out


def write_results(study: Study, block: str, results: dict[str, Any]) -> Path:
    from trading_research.pipeline.short_horizon.freeze import plain

    path = results_path(study, block)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(plain(results), indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return path


def read_block(study: Study, block: str, commit: Any = None) -> dict[str, Any]:
    """The block's read session; ``commit`` (the dry run's) commits the fetch records,
    which the author commits and pushes by hand in the registered study."""
    from trading_research.pipeline import short_horizon_registration as shr
    from trading_research.pipeline.short_horizon.pages import publish_or_note

    state = study.state
    if state.complete(block):
        return state.record(block)
    ledger = ledger_for(study)
    state.require_ready(block)
    session = ledger.session(block, workers=study.workers)
    if block not in state.entries():
        state.begin(block)
    stage = "fetch"
    try:
        records = fetch_records(fetch_block(study, block, session.access))
        if ledger.entries()[block][-1]["fetch"] != records:
            session.record_fetch(records)
        if ledger.committed_entries().get(block, [{}])[-1].get("fetch") != records:
            if commit is None:
                return {"block": block, "fetched": len(records),
                        "next": f"commit and push the ledger with {block}'s fetch records, then read again"}  # fmt: skip
            commit(study.repo, f"{block}'s fetch records")
        session.require_fetch_committed()
        stage = "compute"
        results = compute_results(study, block, session)
    except Exception as exc:
        failure = session.record_failure(stage, exc)
        if failure is None:
            raise
        why = "read failed"
        results = {"block": block, "read_failed": {"stage": failure.stage, "exception": failure.exception,
                   "reason": failure.reason}} | unreadable_results(study, block, why)  # fmt: skip
    path = write_results(study, block, results)
    digest = session.close(path.relative_to(study.repo))
    summary = {"block": block, "results": shr.RESULTS_PATHS[block].as_posix(), "results_sha256": digest,
               "statuses": results.get("statuses", {}), "pages": publish_or_note(study),
               "next": "commit the results file, the pages, their figures and the ledger"}  # fmt: skip
    return state.finish(block, summary)
