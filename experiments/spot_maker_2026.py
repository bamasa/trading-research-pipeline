"""The maker study's single entry point: stages 0, 1, D and the D amendment, then the reads.

Registered in ``docs/preregistration/spot_maker_2026.md`` with
``configs/spot_maker_2026.yaml``. Each stage is a subcommand, run in the
registered order; a stage refuses to start before the stages it reads from are
complete and returns its record without computing when it is already complete::

    python -m experiments.spot_maker_2026 0 --source <the spot study's checkout>
    python -m experiments.spot_maker_2026 1 --source <the spot study's checkout>
    python -m experiments.spot_maker_2026 D --workers 4 --source <the spot study's checkout>
    python -m experiments.spot_maker_2026 amend
    python -m experiments.spot_maker_2026 open H
    python -m experiments.spot_maker_2026 read H --source <...>   # copies H, then asks for a commit
    python -m experiments.spot_maker_2026 read H --workers 4      # computes, writes the results, closes
    python -m experiments.spot_maker_2026 status
    python -m experiments.spot_maker_2026 pages

Between the steps the author commits what the registration commits: the
code-freeze note before stage 1, the frozen configuration, the amendment and the
run log's manifest after ``amend``, the open entry before the block's copy, the
copy records before its computation, and the results file and page after it.
Nothing is pushed. ``--source`` names the spot study's checkout, read only.

``--dry-run`` runs every stage and the H read end to end on the spot study's
synthetic venue in a temporary repository holding the registration as
registered; nothing of this checkout is written and the outputs are discarded
(``--keep DIR`` keeps them). It is a test of the code, never a result.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STAGES = ("0", "1", "D", "amend")


def parser() -> argparse.ArgumentParser:
    found = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawTextHelpFormatter)  # fmt: skip
    found.add_argument("command", nargs="?", choices=[*STAGES, "open", "read", "status", "pages"])
    found.add_argument("block", nargs="?", choices=("H", "P"))
    found.add_argument("--source", type=Path, default=None, help="the spot study's checkout")
    found.add_argument("--workers", type=int, default=1)
    found.add_argument("--run", type=int, default=1, help="the complete D run's number")
    found.add_argument("--dry-run", action="store_true")
    found.add_argument("--keep", type=Path, default=None)
    return found


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.dry_run:
        from trading_research.spot_maker.dry_run import dry_run

        print(json.dumps(dry_run(keep=args.keep), indent=1, sort_keys=True, default=str))
        return 0
    if args.command is None:
        parser().error("name a stage, open, read, status or pages (or pass --dry-run)")
    from trading_research.spot_maker import pages, reads, runner
    from trading_research.spot_maker.context import MakerStudy

    source = None if args.source is None else args.source.resolve()
    study = MakerStudy(REPO, source=source, workers=args.workers, run=args.run)
    if args.command == "status":
        result: object = study.state.status()
    elif args.command == "pages":
        result = str(pages.write_page(study).relative_to(REPO))
    elif args.command in ("open", "read"):
        if args.block is None:
            parser().error(f"{args.command} needs a block: H or P")
        opener = reads.open_block if args.command == "open" else reads.read_block
        result = opener(study, args.block)
    else:
        result = runner.run_stage(study, args.command)
    print(json.dumps(result, indent=1, sort_keys=True, default=str))
    return 0


def entry() -> int:
    from trading_research.market_making.prereg import HeldOutLocked, PreRegistrationError
    from trading_research.spot.state import StageOrderError

    try:
        return main()
    except (StageOrderError, HeldOutLocked, PreRegistrationError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(entry())
