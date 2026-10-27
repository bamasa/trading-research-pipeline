"""One plugin's conformance check in a process of its own.

``check --all`` runs each plugin here rather than all in one process, for the
reason ``stages.ENSEMBLES`` gives for keeping the network out of the default
ensemble: in one process, the network after a booster can deadlock. A fresh
process per plugin, with one OpenMP thread unless ``OMP_NUM_THREADS`` says
otherwise, also bounds each check's time and keeps one plugin's crash from
ending the run.

    python -m trading_research.plugins.worker KIND NAME --output DIR [--execution ...]

Prints one JSON line: the plugin's line of the index.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

#: Seconds one plugin's check may take before it is recorded as unfinished.
TIMEOUT_S = 900


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind")
    parser.add_argument("name")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution", default="all")
    parser.add_argument("--fee-tier", default="base")
    parser.add_argument("--steps", type=int, default=72_000)
    parser.add_argument("--seed", type=int, default=20260101)
    args = parser.parse_args(argv)

    from trading_research.plugins import get
    from trading_research.plugins import page as pages
    from trading_research.plugins.conformance import check, make_setup

    executions: Any = (
        "all" if args.execution == "all" else [e for e in args.execution.split(",") if e]
    )
    setup = make_setup(
        executions=executions, fee_tier=args.fee_tier, steps=args.steps, seed=args.seed
    )
    result = check(get(args.name, args.kind), setup)
    path = pages.write(result, args.output)
    print(json.dumps(pages.summary_row(result, path)))
    return 0


def run(
    kind: str,
    name: str,
    origin: str,
    output: Path,
    *,
    execution: str = "all",
    fee_tier: str = "base",
    steps: int = 72_000,
    seed: int = 20260101,
    timeout_s: float = TIMEOUT_S,
) -> dict[str, Any]:
    """Check one plugin in a child process; its index line, or an unfinished one."""
    from trading_research.plugins import page as pages

    command = [
        sys.executable, "-m", "trading_research.plugins.worker", kind, name,
        "--output", str(output), "--execution", execution, "--fee-tier", fee_tier,
        "--steps", str(steps), "--seed", str(seed),
    ]  # fmt: skip
    # One OpenMP thread, unless the caller set another number: see cli_plugins._one_omp_thread.
    env = {**os.environ, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1")}
    try:
        done = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout_s, check=False, env=env
        )
    except subprocess.TimeoutExpired:
        row = pages.unfinished_row(kind, name, origin, f"timed out after {timeout_s:g} s")
        pages.write_unfinished(row, output)
        return row
    lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
    if done.returncode != 0 or not lines:
        reason = (done.stderr.strip().splitlines() or ["no output"])[-1]
        row = pages.unfinished_row(kind, name, origin, f"the check crashed: {reason}")
        pages.write_unfinished(row, output)
        return row
    finished: dict[str, Any] = json.loads(lines[-1])
    return finished


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
