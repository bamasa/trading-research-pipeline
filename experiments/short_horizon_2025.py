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
    .venv/bin/python -m experiments.short_horizon_2025 pages

What each stage does is in ``trading_research.pipeline.short_horizon``. Between
the steps the author commits what the registration has committed: the code
freeze note before stage 1, the trial ledger after stage B, the results files,
the frozen configuration and the amendment after stage E, each block's open
entry (committed and pushed) before its read, its fetch records before its
computation, and its results file after it.

``open`` writes a block's open entry into the study's ledger; ``read`` runs the
block's read session (fetch, every slot, statuses and kills) and closes it,
returning after the fetch until its records are committed. ``pages`` writes the
pages again from what has been computed (stage E and every read write them
too): the headline sentences, the finalists page, the S25 page with the edge
ladder, the regime map's file and the figures in both themes. ``--run`` numbers a
complete D run (R_D counts them); the worker count is a runtime argument the
ledger records, never part of the configuration.

``--dry-run`` runs every stage and every block read end to end on a synthetic
venue in a temporary repository: no request reaches the network, no file of
the repository is written, and the outputs are discarded (``--keep DIR`` keeps
them). It is a test of the code, never a result.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from trading_research.pipeline.short_horizon import pages, reads, runner
from trading_research.pipeline.short_horizon.context import Study
from trading_research.pipeline.short_horizon.plan import REGISTERED
from trading_research.pipeline.short_horizon.state import READS, STAGES

REPO = Path(__file__).resolve().parent.parent


def parser() -> argparse.ArgumentParser:
    found = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    found.add_argument("command", nargs="?", choices=[*STAGES, "open", "read", "status", "pages"])
    found.add_argument("block", nargs="?", choices=READS, help="the block to open or read")
    found.add_argument("--workers", type=int, default=1)
    found.add_argument("--run", type=int, default=1, help="the complete D run's number (R_D)")
    found.add_argument("--dry-run", action="store_true", help="every stage on a synthetic venue")
    found.add_argument("--keep", type=Path, default=None, help="dry run: keep its files here")
    return found


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.dry_run:
        from trading_research.pipeline.short_horizon.dry_run import dry_run

        report: object = dry_run(keep=args.keep, workers=args.workers)
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
        return 0
    if args.command is None:
        parser().error("name a stage, open, read, status or pages (or pass --dry-run)")
    study = Study(REPO, REGISTERED, run=args.run, workers=args.workers)
    if args.command == "status":
        result: object = runner.status(study)
    elif args.command == "pages":
        result = pages.publish(study)
    elif args.command in ("open", "read"):
        if args.block is None:
            parser().error(f"{args.command} needs a block: {', '.join(READS)}")
        if args.command == "open":
            result = reads.open_block(study, args.block)
        else:
            result = reads.read_block(study, args.block)
    else:
        result = runner.run_stage(study, args.command)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


def entry() -> int:
    """``main``, with a refusal (a stage out of order, a locked block) as a message."""
    from trading_research.market_making.prereg import HeldOutLocked, PreRegistrationError
    from trading_research.pipeline.short_horizon.state import StageOrderError

    try:
        return main()
    except (StageOrderError, HeldOutLocked, PreRegistrationError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(entry())
