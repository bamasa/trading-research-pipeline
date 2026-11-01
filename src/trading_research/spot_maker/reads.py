"""The reads of H (and P): opened by the ledger, copied by a first process, computed by a second.

:func:`open_block` appends the open entry (the author commits it before the
copy). :func:`read_block` starts a process of the read session: the first
process copies the block's days inside the sealed access (P's are fetched by
the spot study's fetch code, :mod:`~trading_research.spot_maker.prospective`)
and records them in the ledger (the author commits them); the next computes
every day, the block's summaries, the statuses under Holm with the frozen MDEs
(on P x sqrt(75 / P's usable days)), the
headlines, writes ``experiments/results/spot_maker_2026_<block>.json`` and the
page, and closes the session with the results file's sha256. A failure is
recorded; the same stage failing twice with the same exception, or a sixth
process, closes the read as failed (every slot ``not assessed (H read failed)``).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from trading_research.market_making.prereg import HeldOutLocked
from trading_research.spot.state import write_json
from trading_research.spot_maker import blocks as bk
from trading_research.spot_maker import copies as cp
from trading_research.spot_maker import prospective as pr
from trading_research.spot_maker import registration as rg
from trading_research.spot_maker import statuses as st
from trading_research.spot_maker.context import MakerStudy, compute_days
from trading_research.spot_maker.ledger import MakerLedger, ReadSession

UNREADABLE_SHARE = 0.25


def ledger_of(study: MakerStudy) -> MakerLedger:
    return MakerLedger(study.repo, study.binding, opener=study.opener)


def block_mde(mde: dict[str, float], block: str, usable_days: int) -> dict[str, float]:
    """The frozen MDEs; on P x sqrt(75 / P's usable days)."""
    if block != "P":
        return dict(mde)
    scale = math.sqrt(bk.BLOCK_DAYS["H"] / usable_days) if usable_days else math.inf
    return {s: v * scale for s, v in mde.items()}


def open_block(study: MakerStudy, block: str, *, force: bool = False) -> dict[str, Any]:
    candidates: tuple[str, ...] = ()
    if block == "P":
        h = h_results(study)
        candidates = tuple(
            sorted(s for s, v in h["statuses"].items() if v["status"] == "candidate")
        )
    return ledger_of(study).open(block, force=force, candidates=candidates)


def h_results(study: MakerStudy) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(
        (Path(study.repo) / rg.RESULTS_PATHS["H"]).read_text("utf-8")
    )
    return loaded


def preset(status: str) -> dict[str, dict[str, Any]]:
    return {s: {"status": status, "kills": [], "first_kill": None} for s in rg.FAMILY}


def compute(study: MakerStudy, session: ReadSession) -> dict[str, Any]:
    """Every day of the block, its summaries and statuses; the results' contents."""
    block = session.block
    days = list(study.days(block))
    records = compute_days(study, days, session.access, study.work / f"read_{block}" / "days")
    excluded = [[r["day"], r["reasons"]] for r in records if not r.get("usable")]
    registration = study.registration()
    frozen = registration.value(("metrics", "mde", "value"))
    mde = block_mde(frozen, block, len(days) - len(excluded))
    if len(days) and len(excluded) / len(days) > UNREADABLE_SHARE:
        statuses = preset(f"not assessed ({block} unreadable on Binance)")
        summaries: dict[str, Any] = {}
    else:
        summaries = {s: st.summarise(records, s) for s in rg.FAMILY}
        statuses = st.assess(summaries, block, mde)
    out = {
        "block": block, "read": session.access.stamp, "session": list(session.access.session),
        "days": len(days), "usable_days": len(days) - len(excluded), "excluded": excluded,
        "statuses": statuses, "summaries": summaries, "mde": mde,
        "calibration": registration.value(("calibration", "result")),
        "placebo_percentile": {s: st.placebo_percentile(v) for s, v in summaries.items()},
        "headlines": st.headlines(summaries, statuses) if summaries and block == "H" else [],
        "activity": activity(records),
    }  # fmt: skip
    if block == "P":
        h = h_results(study)["statuses"]
        out["after_P"] = {s: st.after_p(h[s]["status"], statuses[s]["status"]) for s in rg.FAMILY}
    return out


def activity(records: list[dict[str, Any]]) -> dict[str, Any]:
    """The venue's prints a day and the held unit's own P&L, over the usable days."""
    usable = [r for r in records if r.get("usable")]
    n = max(len(usable), 1)
    return {"usable_days": len(usable),
            "prints_a_day": sum(int(r["prints"]) for r in usable) / n,
            "inventory_pnl_bp_mean": sum(float(r["inventory_pnl_bp"]) for r in usable) / n,
            "inventory_pnl_bp_sum": sum(float(r["inventory_pnl_bp"]) for r in usable)}  # fmt: skip


def read_block(study: MakerStudy, block: str) -> dict[str, Any]:
    """One process of ``block``'s read session (module docstring)."""
    study.disk_check(f"{block} read")
    session = ledger_of(study).session(block, workers=study.workers)
    if not session.copies_committed():
        try:
            if block == "P":
                records = pr.fetch_days(study.repo, study.days(block), session.access,
                                        opener=study.opener)  # fmt: skip
            else:
                records = cp.copy_days(study.require_source(), study.repo, study.days(block),
                                       session.access)  # fmt: skip
        except HeldOutLocked:
            raise
        except Exception as exc:
            if session.record_failure("copy", exc):
                return _failed(study, session)
            raise
        session.record_copies(records)
        return {"block": block, "copied": len(records),
                "next": "commit the ledger's copy records, then read again"}  # fmt: skip
    try:
        results = compute(study, session)
    except HeldOutLocked:
        raise
    except Exception as exc:
        if session.record_failure("compute", exc):
            return _failed(study, session)
        raise
    path = Path(study.repo) / rg.RESULTS_PATHS[block]
    write_json(path, results)
    from trading_research.spot_maker.pages import write_page

    write_page(study)
    return {"block": block, "results_sha256": session.close(path),
            "statuses": {s: v["status"] for s, v in results["statuses"].items()}}  # fmt: skip


def _failed(study: MakerStudy, session: ReadSession) -> dict[str, Any]:
    block = session.block
    statuses = preset(f"not assessed ({block} read failed)")
    path = Path(study.repo) / rg.RESULTS_PATHS[block]
    write_json(path, {"block": block, "read": session.access.stamp, "statuses": statuses,
                      "summaries": {}, "headlines": []})  # fmt: skip
    return {"block": block, "results_sha256": session.close(path), "failed": True}
