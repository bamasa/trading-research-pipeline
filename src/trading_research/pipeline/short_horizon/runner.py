"""Which function computes each stage, and the stage machine around it.

Shared by the entry point and the dry run, so both run exactly the same code:
:func:`run_stage` refuses a stage out of order, returns a complete stage's
record without computing it again, and otherwise computes and records it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from trading_research.pipeline.short_horizon.context import Study
from trading_research.pipeline.short_horizon.state import ORDER, STAGES

StageFunction = Callable[[Study], Mapping[str, Any]]


def stage_function(stage: str) -> StageFunction:
    """The function that computes ``stage``'s record."""
    from trading_research.pipeline.short_horizon import fits, stage0, stage1

    functions: dict[str, StageFunction] = {"0": stage0.run, "1": stage1.run, "A": fits.run}
    if stage not in functions:
        raise NotImplementedError(f"stage {stage} is not wired into the entry point yet")
    return functions[stage]


def run_stage(study: Study, stage: str) -> dict[str, Any]:
    """Run one development stage through the stage machine."""
    if stage not in STAGES:
        raise ValueError(f"{stage!r} is not a development stage; they are {STAGES}")
    compute = stage_function(stage)
    return study.state.run(stage, lambda: compute(study))


def status(study: Study) -> dict[str, str]:
    """Each stage's state for this run."""
    entries = study.state.entries()
    out = {}
    for stage in ORDER:
        if study.state.complete(stage):
            out[stage] = f"complete {entries[stage].get('finished', '')}"
        elif stage in entries:
            out[stage] = f"started {entries[stage].get('started', '')}, not complete"
        else:
            out[stage] = "not run"
    return out
