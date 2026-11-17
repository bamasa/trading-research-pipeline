"""G1 and G2 of docs/preregistration/regime_gate_2026_11.md: the regime gate, read once each.

G1 runs the gate on the regime suite's ten episodes; G2 (``--forward``) on the
benchmark's forward fortnight, and only after that track has opened and its
own read (experiments/results/bench_forward_2026_11.json) is committed. Each
writes experiments/results/regime_gate_G<n>.json and .md and refuses a second
read; a rerun after a fix is reported as an amendment beside the first.

    PYTHONPATH=src python experiments/regime_gate.py --data data
    PYTHONPATH=src python experiments/regime_gate.py --data data --forward
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trading_research.bench import builder
from trading_research.bench.regime_gate import run_gate, verdict
from trading_research.bench.regimes import EPISODES, episode_panel
from trading_research.plugins import get

OUT = Path(__file__).parent / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--forward", action="store_true", help="G2, on the forward fortnight")
    args = parser.parse_args()
    if args.forward:
        forward(args.data)
        return
    data = args.data
    out = OUT / "regime_gate_G1.json"
    if out.exists():
        raise SystemExit(f"{out} exists: G1 is read once")
    build = builder(get("reversion", "strategy"), {})
    results = []
    for ep in EPISODES:
        r = run_gate(ep.name, episode_panel(ep, data), build)
        results.append(r)
        print(f"{ep.name:<16} alarms {len(r.alarms):>2}  paused {r.paused_share:5.1%}  "
              f"ungated {r.ungated_pct:+7.2f}%  gated {r.gated_pct:+7.2f}%  gain {r.gain:+6.2f}", flush=True)  # fmt: skip
    decided = verdict(results)
    out.write_text(json.dumps({"verdict": decided, "episodes": [r.as_dict() for r in results]},
                              indent=2) + "\n", encoding="utf-8")  # fmt: skip
    lines = ["# G1: the regime gate on the regime suite's ten episodes", "",
             f"**{decided['status']}**{': ' + decided['first_kill'] if decided['first_kill'] else ''}. "
             f"Gain {decided['gain_pct']:+.2f}% of capital over the ten episodes; it beats "
             f"{decided['beaten']} of 19 placebos; it helps in {decided['panels_helped']} of 10.", "",
             "| episode | alarms (+1 pauses, -1 resumes) | paused | ungated, % | gated, % | gain |",
             "|---|---|--:|--:|--:|--:|"]  # fmt: skip
    for r in results:
        marks = ", ".join(f"{at[5:16].replace('T', ' ')} {d:+d}" for at, d in r.alarms) or "none"
        lines.append(f"| {r.name} | {marks} | {r.paused_share:.0%} | {r.ungated_pct:+.2f} | "
                     f"{r.gated_pct:+.2f} | {r.gain:+.2f} |")  # fmt: skip
    (OUT / "regime_gate_G1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in decided.items() if k != "placebo_gains_pct"}))


def forward(data: Path) -> None:
    from trading_research.bench.manifest import FORWARD_TRACK
    from trading_research.bench.tracks import TRACKS, Panel

    out = OUT / "regime_gate_G2.json"
    if out.exists():
        raise SystemExit(f"{out} exists: G2 is read once")
    if not (OUT / "bench_forward_2026_11.json").exists():
        raise SystemExit("G2 is read after the forward track's own read is committed")
    part = TRACKS[FORWARD_TRACK].markets(data).parts[0]  # refuses before the track opens
    if not isinstance(part, Panel):
        raise SystemExit("the forward track is a panel of instruments")
    r = run_gate(FORWARD_TRACK, part, builder(get("reversion", "strategy"), {}))
    decided = verdict([r])
    out.write_text(json.dumps({"verdict": decided, "panel": r.as_dict()}, indent=2) + "\n",
                   encoding="utf-8")  # fmt: skip
    print(json.dumps({k: v for k, v in decided.items() if k != "placebo_gains_pct"}))


if __name__ == "__main__":
    main()
