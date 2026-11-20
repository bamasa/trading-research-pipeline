"""The xs_deep rankers replayed walk-forward on the recorded 2024 spans: a measurement.

``docs/preregistration/xs_deep_forward_2026_12.md`` fixes six rankers and reads
them once on 30 November to 13 December 2026. This script replays the same
code, at the same settings, on the two recorded spans (1 February to 10 March
and 12 March to 20 April 2024): fitted on 14 days, scored on the next 7, moved
forward 7 days. Those spans were read by earlier studies, so this is **a
measurement on development data, not the study's result**.

Each ranker runs in its own process (LightGBM and torch each bring an OpenMP
runtime), writing its scored decision times under ``data/xs_deep/dev``;
``report`` writes ``experiments/results/xs_deep/``.

    PYTHONPATH=src python -m experiments.xs_deep_dev all --data data
    PYTHONPATH=src python -m experiments.xs_deep_dev run --ranker tcn --data data
    PYTHONPATH=src python -m experiments.xs_deep_dev report --data data
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from trading_research.bench.tracks import RECORDED_SYMBOLS
from trading_research.xs_deep import evaluate, features, panel
from trading_research.xs_deep.rankers import NAMES

#: (name, folder under the data root, first day, last day).
SPANS = (
    ("2024-02", "universe", date(2024, 2, 1), date(2024, 3, 10)),
    ("2024-04", "universe_fresh", date(2024, 3, 12), date(2024, 4, 20)),
)
OUT = Path(__file__).parent / "results" / "xs_deep"


def panel_for(data: Path, name: str, folder: str, first: date, last: date) -> panel.Panel:
    cache = data / "xs_deep" / f"panel_{name}.npz"
    if cache.exists():
        return panel.read(cache)
    built = panel.load(data / folder, RECORDED_SYMBOLS, panel.span(first, last))
    panel.save(built, cache)
    return built


def run(data: Path, ranker: str) -> Path:
    parts, scores = [], {}
    for name, folder, first, last in SPANS:
        p = panel_for(data, name, folder, first, last)
        frame = features.build(p)
        for k, (fit_days, score_days) in enumerate(evaluate.folds(panel.span(first, last))):
            t0 = time.time()
            at, s = evaluate.fold(ranker, frame, p.times, fit_days, score_days)
            d = evaluate.decisions(frame, at, s)
            d.insert(0, "span", name)
            d.insert(1, "fold", k)
            d["time"] = p.times[at]
            d["day"] = (p.times[at] - np.timedelta64(1, "m")).date
            parts.append(d)
            scores[f"{name}_{k}_bars"], scores[f"{name}_{k}_scores"] = at, s
            print(
                f"{ranker} {name} fold {k}: {score_days[0]} to {score_days[-1]}, "
                f"IC {d['ic'].mean():+.4f}, net {d['net'].mean():+.2f} bp ({time.time() - t0:.0f} s)",
                flush=True,
            )
    out = data / "xs_deep" / "dev" / f"{ranker}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(parts, ignore_index=True).to_parquet(out, index=False)
    np.savez_compressed(out.with_suffix(".npz"), **scores)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("all", "run", "report"))
    parser.add_argument("--ranker", choices=NAMES)
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    if args.command == "run":
        if args.ranker is None:
            parser.error("run needs --ranker")
        print(run(args.data, args.ranker))
    elif args.command == "all":
        for name, folder, first, last in SPANS:
            panel_for(args.data, name, folder, first, last)
        for ranker in NAMES:
            cmd = [
                sys.executable,
                "-m",
                "experiments.xs_deep_dev",
                "run",
                "--ranker",
                ranker,
                "--data",
                str(args.data),
            ]
            subprocess.run(cmd, check=True)
        from experiments.xs_deep_report import report

        report(args.data, OUT)
    else:
        from experiments.xs_deep_report import report

        report(args.data, OUT)


if __name__ == "__main__":
    main()
