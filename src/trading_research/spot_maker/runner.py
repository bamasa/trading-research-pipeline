"""The stages in their registered order, each run once (``spot/state.StageState``)."""

from __future__ import annotations

from typing import Any

from trading_research.spot.state import StageOrderError
from trading_research.spot_maker.context import STAGES, MakerStudy
from trading_research.spot_maker.development import amend, stage_d
from trading_research.spot_maker.stages import stage_0, stage_1


def run_stage(study: MakerStudy, stage: str) -> dict[str, Any]:
    """``stage``'s record: computed once, after every stage before it is complete."""
    if stage not in STAGES:
        raise StageOrderError(f"unknown stage {stage!r}; the stages are {STAGES}")
    state = study.state
    if stage == "0":
        return state.run("0", lambda: stage_0(study))
    if stage == "1":
        if state.complete("0") and not state.record("0")["passes"]:
            raise StageOrderError("stage 0 did not pass: a dated amendment fixes it before stage 1")
        return state.run("1", lambda: stage_1(study))
    if stage == "D":
        return state.run("D", lambda: stage_d(study, state.record("1")))
    return state.run(
        "amend", lambda: amend(study, state.record("0"), state.record("1"), state.record("D"))
    )
