"""The 2025 short-horizon study's single entry point: stages 0 to E, then the block reads.

Registered in ``docs/preregistration/short_horizon_2025.md`` with
``configs/short_horizon_2025.yaml``. Each stage is a subcommand, run in the
registered order; a stage refuses to start before the stages it reads from are
complete, returns its record without computing anything when it is already
complete, and resumes from its own checkpoints when it stopped part way::

    .venv/bin/python -m experiments.short_horizon_2025 0 --workers 8
    .venv/bin/python -m experiments.short_horizon_2025 1 --workers 8
    .venv/bin/python -m experiments.short_horizon_2025 A
    .venv/bin/python -m experiments.short_horizon_2025 B --workers 8
    .venv/bin/python -m experiments.short_horizon_2025 D --workers 8
    .venv/bin/python -m experiments.short_horizon_2025 C --workers 6
    .venv/bin/python -m experiments.short_horizon_2025 E
    .venv/bin/python -m experiments.short_horizon_2025 open H
    .venv/bin/python -m experiments.short_horizon_2025 read H --workers 8
    .venv/bin/python -m experiments.short_horizon_2025 status

``open`` writes a block's open entry into the study's ledger, which is then
committed and pushed by hand before ``read`` will start; ``read`` runs the
block's whole read session (fetch, every slot, statuses and kills) and closes
it. ``--run`` numbers a complete D run (R_D counts them); the worker count is
a runtime argument the ledger records, never part of the configuration.

``--dry-run`` runs every stage and every read end to end on a synthetic venue
in a temporary repository: no request reaches the network, no file of the
repository is written, and the outputs are discarded. It is a test of the
code, never a result.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from trading_research.pipeline.short_horizon.context import Study
from trading_research.pipeline.short_horizon.plan import REGISTERED
from trading_research.pipeline.short_horizon.state import ORDER, READS, STAGES

REPO = Path(__file__).resolve().parent.parent


def stage_function(stage: str) -> Callable[[Study], Mapping[str, Any]]:
    """The function that computes ``stage``'s record."""
    from trading_research.pipeline.short_horizon import stage0

    functions = {"0": stage0.run}
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


def parser() -> argparse.ArgumentParser:
    found = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    found.add_argument("stage", nargs="?", choices=[*STAGES, "open", "read", "status"])
    found.add_argument("block", nargs="?", choices=READS, help="the block to open or read")
    found.add_argument("--workers", type=int, default=1)
    found.add_argument("--run", type=int, default=1, help="the complete D run's number (R_D)")
    found.add_argument("--dry-run", action="store_true", help="every stage on a synthetic venue")
    found.add_argument("--keep", type=Path, default=None, help="dry run: keep its files here")
    return found


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.dry_run:
        from experiments._short_horizon_dry_run import dry_run

        report = dry_run(keep=args.keep, workers=args.workers)
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
        return 0
    if args.stage is None:
        parser().error("name a stage, open, read or status (or pass --dry-run)")
    study = Study(REPO, REGISTERED, run=args.run, workers=args.workers)
    if args.stage == "status":
        result: object = status(study)
    elif args.stage in ("open", "read"):
        if args.block is None:
            parser().error(f"{args.stage} needs a block: {', '.join(READS)}")
        raise NotImplementedError("the block reads are not wired into the entry point yet")
    else:
        result = run_stage(study, args.stage)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
