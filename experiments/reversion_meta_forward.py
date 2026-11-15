"""M1 of docs/preregistration/reversion_meta_forward_2026_11.md: the model-chosen reversion on the forward fortnight.

Eight strategies, the reversion rule with a model choosing its trades (seven
models) and the rule with its kill switch, scored on the benchmark's forward
track by the benchmark's own runner, each in a process of its own (XGBoost and
PyTorch do not share one on macOS). Refuses before the track opens, before the
track's own read is committed, when a strategy's source is not the one frozen
below, and a second time.

    PYTHONPATH=src python experiments/reversion_meta_forward.py --data data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "results"
RESULT = OUT / "reversion_meta_forward_2026_11.json"
FAMILY = ("reversion_guarded", "reversion_meta", "reversion_meta_guarded", "reversion_lightgbm", "reversion_logistic",
          "reversion_transformer", "reversion_lstm", "reversion_gru", "reversion_tcn")  # fmt: skip
ALPHA = 0.05
#: The sources the family runs, by sha256, as committed with the registration.
FROZEN = {
    "src/trading_research/plugins/builtins/meta_reversion.py": "ef04d8107e9a1f20bccdf1c0fa2f3a2ab25bf2242ef33caedd0afe6e62030938",
    "src/trading_research/plugins/builtins/guarded.py": "eb0e480c3392cd0bdfab862a2b6461ca32abbfc50fae318cfe272ebaad4f68bf",
    "src/trading_research/plugins/builtins/strategies.py": "0eb9ae99255c2aea9929721fda74e6a7aca0a00acdd3b570b82eee439688ac98",
    "src/trading_research/models/sequence.py": "17da4c719b1416fcf138810cf859567b8221e562c76dcc222968ee15492c39a9",
    "src/trading_research/models/tcn.py": "c40964a014ed7329f3ea3ffe15101305dafd731c985ea8cd23fd123364579463",
}


def sha(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def changed() -> list[str]:
    return [p for p, h in FROZEN.items() if sha(p) != h]


def one(name: str, data: Path) -> None:
    """Score one strategy on the forward track and print its score as JSON."""
    from trading_research.bench import MarketCache, run
    from trading_research.bench.manifest import FORWARD_TRACK
    from trading_research.bench.tracks import TRACKS
    from trading_research.plugins import get

    card = run(get(name, "strategy"), {}, [TRACKS[FORWARD_TRACK]], MarketCache(data))
    print(json.dumps(card.scores[0].as_dict()))


def decide(scores: dict[str, dict]) -> dict[str, dict]:  # type: ignore[type-arg]
    from trading_research.pairs.stats import holm_adjusted

    p = [scores[n]["net_p"] if scores[n].get("net_p") is not None else math.nan for n in FAMILY]
    out = {}
    for n, holm in zip(FAMILY, holm_adjusted(p), strict=True):
        s = scores[n]
        ok = (
            s["status"] == "scored" and s["passed"] is True and math.isfinite(holm) and holm < ALPHA
        )
        out[n] = {"status": "passes" if ok else "killed", "holm_p": None if math.isnan(holm) else holm,
                  "net_bp": s.get("net_bp"), "trades": s.get("trades"), "verdict": s.get("verdict")}  # fmt: skip
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--one", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.one:
        one(args.one, args.data)
        return
    if RESULT.exists():
        raise SystemExit(f"{RESULT} exists: M1 is read once")
    if not (OUT / "bench_forward_2026_11.json").exists():
        raise SystemExit("M1 is read after the forward track's own read is committed")
    if bad := changed():
        raise SystemExit("not the frozen sources: " + ", ".join(bad))
    scores = {}
    for name in FAMILY:
        done = subprocess.run([sys.executable, __file__, "--data", str(args.data), "--one", name],
                              capture_output=True, text=True, check=True,
                              env={"PYTHONPATH": str(ROOT / "src"), "OMP_NUM_THREADS": "1",
                                   "PATH": "/usr/bin:/bin"})  # fmt: skip
        scores[name] = json.loads(done.stdout.strip().splitlines()[-1])
        print(name, scores[name]["verdict"], file=sys.stderr)
    forward = json.loads((OUT / "bench_forward_2026_11.json").read_text())
    payload = {"m1": decide(scores), "scores": scores,
               "reversion_net_bp": forward.get("p1", {}).get("net_bp"),
               "sources": {p: sha(p) for p in FROZEN}}  # fmt: skip
    RESULT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["m1"], indent=1))


if __name__ == "__main__":
    main()
