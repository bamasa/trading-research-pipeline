"""The animation's data: every curve the carry search drew, recomputed and checked against what it wrote.

Replays the sixteen rules on the development months, the chosen rule and its
nineteen placebos on the held-out months, with the engine and seeds of
``trp search carry``, and refuses to write if any summary differs from the one
the search wrote in ``experiments/results/search_carry``. Nothing is chosen
here: the held-out months were read once by the search, and this redraws them.

    PYTHONPATH=src python posts/carry_search/make_data.py --data data
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from trading_research.search_engine.carry import (
    DEVELOPMENT,
    GRID,
    HELD_OUT,
    PLACEBOS,
    load_universe,
    random_chooser,
    readable,
)
from trading_research.strategies.carry import Rule, replay, summary

HERE = Path(__file__).parent
RESULTS = HERE.parents[1] / "experiments" / "results" / "search_carry"


def curve(frame: pd.DataFrame) -> list[float]:
    """Equity in % from 0 at the start, at each month end; the last point includes the final close."""
    eq = np.cumprod(1 + frame["return_pct"].to_numpy() / 100)
    eq[-1] = frame.attrs.get("final_equity", eq[-1])
    return [0.0, *[round((v - 1) * 100, 4) for v in eq]]


def same(a: dict[str, float], b: dict[str, float]) -> bool:
    return all(math.isclose(a[k], b[k], rel_tol=1e-9, abs_tol=1e-9) or (math.isnan(a[k]) and math.isnan(b[k]))
               for k in ("total_pct", "worst_drawdown_pct"))  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data"))
    root = parser.parse_args().data / "carry_walkforward"
    dev = json.loads((RESULTS / "development.json").read_text())
    held = json.loads((RESULTS / "held_out.json").read_text())
    ledger = json.loads((RESULTS / "ledger.json").read_text())
    markets = load_universe(root)

    rules = {}
    for r in GRID:
        frame = replay(markets, DEVELOPMENT, r, readable)
        if not same(summary(frame), dev["board"][r.name]):
            raise SystemExit(f"{r.name} on D differs from the search's")
        rules[r.name] = curve(frame)
    chosen = Rule(**ledger["rule"])
    frame = replay(markets, HELD_OUT, chosen, readable)
    if not same(summary(frame), held["result"]):
        raise SystemExit("the chosen rule on H differs from the search's")
    placebos = []
    for j in range(PLACEBOS):
        p = replay(markets, HELD_OUT, chosen, readable, random_chooser(markets, chosen, 100 + j))
        if not same(summary(p), held["placebo"][j]):
            raise SystemExit(f"placebo {j} on H differs from the search's")
        placebos.append(curve(p))
    payload = {
        "development_months": [str(m) for m in DEVELOPMENT],
        "held_out_months": [str(m) for m in HELD_OUT],
        "rules": rules,
        "chosen": ledger["chosen"],
        "sha": ledger["development_sha256"][:8],
        "held_out": curve(frame),
        "coins": [c.removesuffix("USDT") for c in frame["coins"]],
        "placebos": placebos,
        "result": held["result"],
        "beats": held["beats_placebos"],
        "placebo_totals": sorted(p["total_pct"] for p in held["placebo"]),
    }
    (HERE / "data.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    print(f"{len(rules)} rules, chosen {ledger['chosen']}, {len(placebos)} placebos -> data.json")


if __name__ == "__main__":
    main()
