"""The spot study's single entry point: stages 0, 1, D and the D amendment, then the reads of H and P.

Registered in ``docs/preregistration/spot_2026.md`` with ``configs/spot_2026.yaml``.
Each stage is a subcommand, run in the registered order; a stage refuses to
start before the stages it reads from are complete, returns its record without
computing anything when it is already complete, and resumes from its own
checkpoints when it stopped part way::

    python -m experiments.spot_2026 0
    python -m experiments.spot_2026 1 --workers 4
    python -m experiments.spot_2026 D --frozen-2025 <checkout of the 2025 study>
    python -m experiments.spot_2026 amend
    python -m experiments.spot_2026 open H
    python -m experiments.spot_2026 read H      # fetches H, then asks for a commit
    python -m experiments.spot_2026 read H      # computes, writes the results, closes
    python -m experiments.spot_2026 open P      # only for H's candidates, by P's deadline
    python -m experiments.spot_2026 read P
    python -m experiments.spot_2026 status
    python -m experiments.spot_2026 pages

What each stage does is in ``trading_research.spot`` (``stage0``, ``stage1``,
``development``, ``freeze``, ``reads``). Between the steps the author commits
what the registration has committed: the code-freeze note before stage 1, the
frozen configuration, the amendment and the run log's manifest after
``amend``, each block's open entry before its read, its fetch records before
its computation, and its results file and page after it. Nothing is pushed.

``--run`` numbers a complete D run; the worker count is a runtime argument.
``--frozen-2025`` names the checkout holding the 2025 study's frozen files
(ignored by git) when the study runs elsewhere: they are copied in, each
sha256 checked at the copy.

``--dry-run`` runs every stage and the H read end to end on a synthetic venue in
a temporary repository holding the registration as registered: no request
reaches the network, no file of this repository is written, and the outputs are
discarded (``--keep DIR`` keeps them). It is a test of the code, never a result.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STAGES = ("0", "1", "D", "amend")
READS = ("H", "P")


def parser() -> argparse.ArgumentParser:
    found = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    found.add_argument("command", nargs="?", choices=[*STAGES, "open", "read", "status", "pages"])
    found.add_argument("block", nargs="?", choices=READS, help="the block to open or read")
    found.add_argument("--workers", type=int, default=1)
    found.add_argument("--run", type=int, default=1, help="the complete D run's number")
    found.add_argument(
        "--frozen-2025",
        type=Path,
        default=None,
        help="the checkout holding the 2025 study's frozen files",
    )
    found.add_argument("--dry-run", action="store_true", help="every stage on a synthetic venue")
    found.add_argument("--keep", type=Path, default=None, help="dry run: keep its files here")
    return found


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.dry_run:
        from trading_research.spot.dry_run import dry_run

        print(json.dumps(dry_run(keep=args.keep), indent=2, sort_keys=True, default=str))
        return 0
    if args.command is None:
        parser().error("name a stage, open, read, status or pages (or pass --dry-run)")
    from trading_research.spot import pages, reads, runner
    from trading_research.spot.context import SpotStudy

    study = SpotStudy(REPO, run=args.run, workers=args.workers)
    if args.frozen_2025 is not None:
        study.binding = replace(study.binding, root=args.frozen_2025.resolve())
    if args.command == "status":
        result: object = study.state.status()
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
    from trading_research.spot.state import StageOrderError

    try:
        return main()
    except (StageOrderError, HeldOutLocked, PreRegistrationError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(entry())
