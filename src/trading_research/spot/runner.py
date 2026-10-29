"""Which function computes each stage of the spot study, and the stage machine around it.

Shared by the entry point and the dry run, so both run exactly the same code:
:func:`run_stage` refuses a stage out of order, returns a complete stage's
record without computing it again, and otherwise computes and records it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from trading_research.spot.context import SpotStudy
from trading_research.spot.state import STAGES

StageFunction = Callable[[SpotStudy], Mapping[str, Any]]


def stage_function(stage: str) -> StageFunction:
    """The function that computes ``stage``'s record."""
    from trading_research.spot import development, freeze, stage0, stage1

    functions: dict[str, StageFunction] = {
        "0": stage0.run,
        "1": stage1.run,
        "D": development.run,
        "amend": freeze.run,
    }
    return functions[stage]


def run_stage(study: SpotStudy, stage: str) -> dict[str, Any]:
    """Run one development stage through the stage machine."""
    if stage not in STAGES:
        raise ValueError(f"{stage!r} is not a development stage; they are {STAGES}")
    compute = stage_function(stage)
    return study.state.run(stage, lambda: compute(study))
