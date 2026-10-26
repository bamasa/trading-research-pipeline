"""The reads of H and P through the ledger: open, fetch, commit, compute, close.

Registered ("Access", "Status rules"). Each read is one session of the ledger
(:class:`trading_research.spot.ledger.Ledger`):

1. ``open``: the ledger checks the block's preconditions and writes its open
   entry; the author commits it before anything of the block is fetched. P opens
   only for H's candidates, by its deadline, once its last day is published on
   every plane of both venues (``HEAD`` requests) and the fee amendment covers
   its days.
2. ``read``, first process: the block is fetched under the session's sealed
   access (the disk check first), and each file's sha256 and download time go
   into the open entry; the author commits them.
3. ``read``, next process: refused until the fetch records are committed; then
   every row on every day (:func:`trading_research.spot.blockread.compute`), the
   statuses, the results file ``experiments/results/spot_2026_<block>.json``
   with every status, kill, measurement and headline sentence, the page, and
   the results file's sha256 into the ledger, which closes the session.

A process that fails records its stage and exception in the open entry; the
same stage failing again with the same exception, or a read unfinished after
five resumptions, closes the session as failed (on H every slot ``not assessed
(H read failed)``; on P every candidate ``candidate, not confirmed (P read
failed)``).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from datetime import date
from typing import Any

from trading_research.data import fetch_log as fl
from trading_research.market_making.prereg import HeldOutLocked
from trading_research.spot import archives as ar
from trading_research.spot import blockread as br
from trading_research.spot import heldout as ho
from trading_research.spot import registration as sr
from trading_research.spot import statuses as st
from trading_research.spot.context import SpotStudy
from trading_research.spot.stage0 import fetch
from trading_research.spot.state import plain, write_json

FETCH_STAGE: dict[str, int] = {"H": 2, "P": 3}
Committer = Callable[[Any, str], Any]


def published(study: SpotStudy) -> Callable[[str, str, date], bool]:
    """Whether a venue publishes a plane's day: a ``HEAD`` request answering 200."""

    def check(instrument: str, kind: str, day: date) -> bool:
        plane = next(p for p in ar.PLANES.values() if p.instrument == instrument and p.kind == kind)
        return ar.head(plane.url(day), opener=study.opener).status == 200

    return check


def h_results(study: SpotStudy) -> dict[str, Any]:
    path = study.repo / sr.RESULTS_PATHS["H"]
    if not path.exists():
        raise HeldOutLocked("H's results file is not written")
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def open_block(study: SpotStudy, block: str) -> dict[str, Any]:
    """The block's open entry, written into the ledger for the author to commit."""
    study.state.require_ready(block)
    candidates: list[str] = []
    if block == "P":
        candidates = st.candidates_in(h_results(study))
        if not candidates:
            raise HeldOutLocked("no slot is a candidate on H: P is not opened")
    with study.serving():
        entry = study.ledger().open(block, candidates=candidates,
                                    published=published(study) if block == "P" else None)  # fmt: skip
    return {"entry": entry, "next": f"commit {sr.LEDGER_PATH}, then run: read {block}"}


def fetch_records(log: fl.FetchLog, days: list[date]) -> dict[str, dict[str, Any]]:
    """Each fetched file's sha256 and UTC download time, keyed ``plane/day``."""
    wanted = {d.isoformat() for d in days}
    return {key: {"sha256": r.sha256, "downloaded_at": r.downloaded_at, "size": r.size}
            for key, r in log.records.items()
            if r.status == fl.ON_DISK and r.day in wanted}  # fmt: skip


def plan_for(
    study: SpotStudy, block: str, access: Any, registration: sr.Registration, label: str
) -> ho.ReadPlan:
    """The read's plan from the frozen configuration and the frozen files."""
    walk, gaps = registration.walk_levels, registration.max_sequence_gaps
    if walk is None or gaps is None:
        raise HeldOutLocked("the configuration is not frozen")
    return ho.ReadPlan(label=label, block=block, days=study.scale.days(block), access=access,
                       models=br.frozen_models(study, registration), max_gaps=gaps, depth=walk,
                       spec=registration.instrument_spec)  # fmt: skip


def _final(
    block: str, found: Mapping[str, Any], blind: bool, h: Mapping[str, Any] | None
) -> dict[str, str]:
    """P's final statuses by the registered table, from H's statuses and P's assessments."""
    if block != "P" or h is None:
        return {}
    statuses = {slot: v["status"] for slot, v in h["statuses"].items()}
    return {slot: st.final_status(statuses[slot], found.get(slot), p_state="read", blind=blind)
            for slot in sr.FAMILY}  # fmt: skip


def write_results(
    study: SpotStudy, block: str, session: Any, registration: sr.Registration,
    assessments: Mapping[str, st.Assessment], measurements: Mapping[str, Any],
) -> str:  # fmt: skip
    """``experiments/results/spot_2026_<block>.json``, closing the read session."""
    record = st.results_record(block, assessments, stamp=session.stamp, session=session.key,
                               blind=registration.h_blind, measurements=measurements)  # fmt: skip
    write_json(study.repo / sr.RESULTS_PATHS[block], plain(record))
    return session.close()


def failed(study: SpotStudy, block: str, session: Any, registration: sr.Registration,
           reason: Any) -> dict[str, Any]:  # fmt: skip
    """The session closed as failed: on H every slot not assessed; on P every candidate
    not confirmed (P read failed)."""
    if block == "H":
        assessments = st.on_h({}, blind=registration.h_blind, read_failed=True)
        after: dict[str, str] = {}
    else:
        h = h_results(study)
        statuses = {s: v["status"] for s, v in h["statuses"].items()}
        assessments = {s: st.Assessment(s, "P", "not assessed (P read failed)"
                                        if st.is_candidate(statuses[s]) else st.NOT_TAKEN)
                       for s in sr.FAMILY}  # fmt: skip
        after = {s: st.final_status(statuses[s], None, p_state="read failed",
                                    blind=registration.h_blind) for s in sr.FAMILY}  # fmt: skip
    measurements = {"read_failed": {"stage": reason.stage, "exception": reason.exception,
                                    "reason": reason.reason}, "after_P": after}  # fmt: skip
    digest = write_results(study, block, session, registration, assessments, measurements)
    return {"results_sha256": digest, "read_failed": measurements["read_failed"]}


def read_block(study: SpotStudy, block: str, *, commit: Committer | None = None) -> dict[str, Any]:
    """One process of the block's read session (module docstring)."""
    if study.state.complete(block):
        return study.state.record(block)
    study.state.require_ready(block)
    ledger = study.ledger()
    with study.serving():
        session = ledger.session(block, workers=study.workers)
    study.state.begin(block)
    registration = study.reload().require_frozen()
    days = list(study.scale.days(block))
    label = f"{block}_{len(ledger.entries()[block])}"
    if not ledger.entries()[block][-1]["fetch"]:
        try:
            with study.serving():
                log = fetch(study, FETCH_STAGE[block], tuple(days), session.access,
                            int(registration.walk_levels or 10))  # fmt: skip
        except Exception as exc:
            reason = session.record_failure("fetch", exc)
            if reason is None:
                raise
            return study.state.finish(block, failed(study, block, session, registration, reason))
        session.record_fetch(fetch_records(log, days))
        if commit is None:
            return {"next": f"commit {sr.LEDGER_PATH} (the fetch records), then run: read {block}"}
        commit(study.repo, f"{block}: the fetch records")
    session.require_fetch_committed()
    h = h_results(study) if block == "P" else None
    try:
        found = br.compute(
            study, plan_for(study, block, session.access, registration, label),
            blind=registration.h_blind, mde=registration.mde,
            h_statuses=None if h is None else {s: v["status"] for s, v in h["statuses"].items()},
        )  # fmt: skip
        after = _final(block, found["assessments"], registration.h_blind, h)
        if h is None:
            sentences = br.sentences(found)
        else:
            taken = [s for s in sr.FAMILY if st.is_candidate(h["statuses"][s]["status"])]
            sentences = {"P": [st.p_sentence(s, after[s]) for s in taken]}
    except Exception as exc:
        reason = session.record_failure("compute", exc)
        if reason is None:
            raise
        return study.state.finish(block, failed(study, block, session, registration, reason))
    measurements = {**found["measurements"], "sentences": sentences, "after_P": after}
    digest = write_results(study, block, session, registration, found["assessments"], measurements)
    from trading_research.spot import pages

    page = pages.publish(study)
    return study.state.finish(block, {"results_sha256": digest, "label": label,
                                      "statuses": {s: a.status for s, a in found["assessments"].items()},
                                      "after_P": after, "sentences": sentences, "page": page})  # fmt: skip


def final_statuses(study: SpotStudy, today: date | None = None) -> dict[str, str]:
    """Each slot's status by the registered table after P: from P's results once P is
    read; before it, "candidate (P not yet read)" for a candidate until P's deadline
    and "candidate, not confirmed (P not read by its deadline)" after it; every
    other status as H left it (with the fallback's mark when H was not blind)."""
    path = study.repo / sr.RESULTS_PATHS["H"]
    if not path.exists():
        return {}
    h = json.loads(path.read_text(encoding="utf-8"))
    p_path = study.repo / sr.RESULTS_PATHS["P"]
    if p_path.exists():
        after = json.loads(p_path.read_text(encoding="utf-8")).get("measurements", {})
        found = after.get("after_P") or {}
        if found:
            return {slot: str(found[slot]) for slot in sr.FAMILY if slot in found}
    try:
        deadline = study.ledger().p_deadline()
    except HeldOutLocked:  # H's results not yet committed: the deadline is not running
        deadline = None
    day = today or study.ledger().today()
    state = "not read by its deadline" if deadline and day > deadline else "not yet read"
    blind = bool(h.get("blind"))
    return {slot: st.final_status(str(h["statuses"][slot]["status"]), p_state=state, blind=blind)
            for slot in sr.FAMILY if slot in h.get("statuses", {})}  # fmt: skip
