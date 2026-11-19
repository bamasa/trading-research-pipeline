"""X1, X2 and X3 of xs_deep_forward_2026_12, read once on 30 November to 13 December 2026.

``docs/preregistration/xs_deep_forward_2026_12.md`` registers six rankers of
the 26 perpetuals, fitted on 16 to 29 November 2026 and scored on the
fortnight after. This script refuses to fetch or read before 15 December 2026,
when the last scored day's archive is out, and refuses to read if any source
of the study differs from the sha256 the code freeze recorded in
:data:`FROZEN`. It writes its result once: a second read is refused.

    PYTHONPATH=src python -m experiments.xs_deep_forward fetch --data data
    PYTHONPATH=src python -m experiments.xs_deep_forward read --data data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from trading_research.bench.tracks import FORWARD_SYMBOLS
from trading_research.xs_deep import evaluate, features, panel
from trading_research.xs_deep.rankers import NAMES

FIT_FIRST, FIT_LAST = date(2026, 11, 16), date(2026, 11, 29)
SCORED_FIRST, SCORED_LAST = date(2026, 11, 30), date(2026, 12, 13)
OPENS = date(2026, 12, 15)
FOLDER = "xs_deep_2026_12"
MINIMUM = 20
ALPHA = 0.05
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "results" / "xs_deep_forward"
SOURCES = tuple(
    f"src/trading_research/xs_deep/{name}.py"
    for name in ("__init__", "panel", "features", "rankers", "nets", "evaluate")
)
#: sha256 of each source at the code freeze; written by its amendment.
FROZEN: dict[str, str] = {}


def today() -> date:
    return datetime.now(UTC).date()


def unfrozen() -> list[str]:
    """Sources missing from :data:`FROZEN` or differing from it."""
    if not FROZEN:
        return list(SOURCES)
    return [
        s for s in SOURCES if FROZEN.get(s) != hashlib.sha256((ROOT / s).read_bytes()).hexdigest()
    ]


def refuse_before_open() -> None:
    if today() < OPENS:
        raise SystemExit(f"sealed until {OPENS}: the scored days' archives are not all out")


def fetch(data: Path) -> None:
    from trading_research.data.ensure import ensure_universe

    refuse_before_open()
    ensure_universe(FORWARD_SYMBOLS, FIT_FIRST, SCORED_LAST, root=data / FOLDER)


def load(data: Path) -> panel.Panel:
    return panel.load(
        data / FOLDER, FORWARD_SYMBOLS, panel.span(FIT_FIRST, SCORED_LAST), minimum=MINIMUM
    )


def run_one(data: Path, ranker: str) -> None:
    p = load(data)
    frame = features.build(p)
    at, s = evaluate.fold(
        ranker,
        frame,
        p.times,
        panel.span(FIT_FIRST, FIT_LAST),
        panel.span(SCORED_FIRST, SCORED_LAST),
    )
    d = evaluate.decisions(frame, at, s)
    d["day"] = (p.times[at] - np.timedelta64(1, "m")).date
    out = data / FOLDER / "scored" / f"{ranker}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(out, index=False)


def read(data: Path) -> dict[str, object]:
    refuse_before_open()
    if bad := unfrozen():
        raise SystemExit(f"not the frozen sources: {', '.join(bad)}")
    if (OUT / "result.json").exists():
        raise SystemExit("already read; the result is written once")
    p = load(data)
    for ranker in NAMES:
        cmd = [sys.executable, "-m", "experiments.xs_deep_forward", "run-one",
               "--ranker", ranker, "--data", str(data)]  # fmt: skip
        subprocess.run(cmd, check=True, cwd=ROOT)
    days = {
        n: pd.read_parquet(data / FOLDER / "scored" / f"{n}.parquet")
        .groupby("day")[["ic", "gross", "cost", "net"]]
        .mean()
        for n in NAMES
    }
    net = {n: evaluate.one_sided(d["net"]) for n, d in days.items()}
    net_holm = evaluate.holm({n: v["p"] for n, v in net.items()})
    x1 = {
        n: {**net[n], "p_holm": net_holm[n],
            "status": "passes" if net[n]["mean"] > 0 and net_holm[n] < ALPHA else "killed"}
        for n in NAMES
    }  # fmt: skip
    diff = {
        "X2": evaluate.one_sided(days["xs_transformer"]["ic"] - days["ridge"]["ic"]),
        "X3": evaluate.one_sided(days["xs_transformer"]["ic"] - days["tcn"]["ic"]),
    }
    diff_holm = evaluate.holm({k: v["p"] for k, v in diff.items()})
    x23 = {
        k: {**v, "p_holm": diff_holm[k],
            "status": "passes" if v["mean"] > 0 and diff_holm[k] < ALPHA else "killed"}
        for k, v in diff.items()
    }  # fmt: skip
    result = {
        "read_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "instruments": list(p.symbols),
        "X1": x1,
        **x23,
        "beside": {
            n: {
                "ic": float(d["ic"].mean()),
                "gross_bp": float(d["gross"].mean()),
                "cost_bp": float(d["cost"].mean()),
            }
            for n, d in days.items()
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("fetch", "read", "run-one", "check"))
    parser.add_argument("--ranker", choices=NAMES)
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    if args.command == "fetch":
        fetch(args.data)
    elif args.command == "run-one":
        refuse_before_open()
        run_one(args.data, args.ranker)
    elif args.command == "check":
        bad = unfrozen()
        print("frozen sources match" if not bad else f"differ: {', '.join(bad)}")
    else:
        print(json.dumps(read(args.data), indent=2, default=float))


if __name__ == "__main__":
    main()
