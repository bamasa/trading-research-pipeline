"""The single entry point of the Bybit 2021 market-making study.

Registered in ``docs/preregistration/mm_bybit_2021.md`` with
``configs/mm_bybit_2021.yaml``. The stages, in order, each committed by the
author before the next::

    python -m experiments.mm_bybit_2021 0                 # reader's proof and timing
    python -m experiments.mm_bybit_2021 1                 # W and D fetched, W's inputs
    python -m experiments.mm_bybit_2021 D --workers 6     # every cell on D, choice, nulls
    python -m experiments.mm_bybit_2021 amend --freeze <commit>
    python -m experiments.mm_bybit_2021 read-H --workers 6  # opens H (local commit), reads it
    python -m experiments.mm_bybit_2021 --dry-run         # everything on a synthetic venue

The data root is ``data/mm_bybit_2021`` (ignored by git). Nothing is pushed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "mm_bybit_2021"


def main(argv: Sequence[str] | None = None) -> int:
    from trading_research.mm_bybit import dry_run, ledger, runner

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)  # fmt: skip
    parser.add_argument("command", nargs="?", choices=["0", "1", "D", "amend", "read-H"])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--freeze", help="the code-freeze commit (amend)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="read H a second time (stamped 'second read'; changes no status)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep", type=Path, help="keep the dry run's repository here")
    args = parser.parse_args(argv)
    if args.dry_run:
        record = dry_run.run(REPO, keep=args.keep, workers=args.workers)
        print(json.dumps({k: v.get("status") for k, v in record["primaries"].items()}))
        return 0
    if args.command == "0":
        out = runner.stage0(REPO, DATA)
    elif args.command == "1":
        out = runner.stage1(REPO, DATA)
    elif args.command == "D":
        out = runner.develop_d(REPO, DATA, workers=args.workers)
    elif args.command == "amend":
        if not args.freeze:
            parser.error("amend needs --freeze <code-freeze commit>")
        sha = runner.amend(REPO, code_freeze_commit=args.freeze,
                           day=datetime.now(UTC).date())  # fmt: skip
        out = {"frozen_configuration_sha256": sha}
    elif args.command == "read-H":
        access = ledger.open_h(REPO, DATA, force=args.force)
        out = runner.read_h(REPO, DATA, access, workers=args.workers)
        name = "H_second_read" if access.second_read else "H"
        ledger.close_h(REPO, runner.record_path(REPO, name))
    else:
        parser.print_help()
        return 2
    print(json.dumps(out, indent=1, default=str)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
