"""``trp search carry``: the automatic search for a funding carry, chosen on D and read once on H.

The registration is ``docs/preregistration/carry_search_2026_11.md``. The
search scores every rule of :data:`GRID` (how many coins to hold, on how many
months of funding, above what floor) on the development months, chooses the
one with the highest monthly Sharpe ratio there, writes that choice and the
development results' sha256 to the run's ledger, and only then replays the
chosen rule on the held-out months, once. Beside each it runs nineteen
placebos that hold the same number of coins picked at random each month, so
the result says what choosing by funding adds over holding any carry.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_research.strategies.carry import UNIVERSE, Market, Rule, load, replay, summary

DEVELOPMENT = pd.period_range("2021-01", "2023-06", freq="M")
HELD_OUT = pd.period_range("2023-07", "2025-06", freq="M")
GRID: tuple[Rule, ...] = tuple(
    Rule(slots=k, lookback=lb, threshold_bp_a_day=floor)
    for k, lb, floor in product((1, 2, 4, 8), (1, 3), (38.0 / 30.0, 0.0))
)
PLACEBOS = 19
#: funding_carry_2026 holds BTC and ETH out over these months; they are never read.
PROTECTED = {"BTCUSDT", "ETHUSDT"}
PROTECTED_MONTHS = set(pd.period_range("2020-12", "2021-06", freq="M")) | set(
    pd.period_range("2024-02", "2024-06", freq="M")
)
LAST = pd.Period("2025-06", "M")


def readable(coin: str, month: pd.Period) -> bool:
    return month <= LAST and not (coin in PROTECTED and month in PROTECTED_MONTHS)


def load_universe(root: Path) -> dict[str, Market]:
    first = DEVELOPMENT[0] - max(r.lookback for r in GRID)
    months = pd.period_range(first, LAST, freq="M")
    return {c: load(root, c, [m for m in months if readable(c, m)]) for c in UNIVERSE}


def random_chooser(
    markets: dict[str, Market], rule: Rule, seed: int
) -> Callable[[pd.Period], list[str]]:
    """Each month, ``rule.slots`` coins drawn at random from those the rule could have held."""
    rng = np.random.default_rng(seed)

    def pick(month: pd.Period) -> list[str]:
        pool = sorted(c for c in markets if readable(c, month) and readable(c, month - 1))
        return list(rng.choice(pool, size=min(rule.slots, len(pool)), replace=False))

    return pick


def development(markets: dict[str, Market]) -> dict[str, Any]:
    """Every rule on D, the choice, and the choice's placebos on D."""
    board = {r.name: summary(replay(markets, DEVELOPMENT, r, readable)) for r in GRID}
    best = max(GRID, key=lambda r: (board[r.name]["sharpe_monthly"], board[r.name]["total_pct"]))
    placebo = [summary(replay(markets, DEVELOPMENT, best, readable, random_chooser(markets, best, j)))
               for j in range(PLACEBOS)]  # fmt: skip
    return {"board": board, "chosen": best.name, "rule": best.__dict__, "placebo": placebo}


def held_out(markets: dict[str, Market], rule: Rule) -> dict[str, Any]:
    frame = replay(markets, HELD_OUT, rule, readable)
    placebo = [summary(replay(markets, HELD_OUT, rule, readable, random_chooser(markets, rule, 100 + j)))
               for j in range(PLACEBOS)]  # fmt: skip
    result = summary(frame)
    beaten = sum(p["total_pct"] < result["total_pct"] for p in placebo)
    status = "positive" if result["total_pct"] > 0 else "killed"
    return {"result": result, "placebo": placebo, "beats_placebos": beaten, "status": status,
            "months": frame.to_dict("records")}  # fmt: skip


def run(root: Path, run_dir: Path, on_line: Callable[[str], None] = print) -> Path:
    """D, the ledger, then H once. Refuses a second read of H."""
    run_dir.mkdir(parents=True, exist_ok=True)
    h_path, ledger = run_dir / "held_out.json", run_dir / "ledger.json"
    if h_path.exists():
        raise FileExistsError(f"{h_path} exists: the held-out months are read once")
    markets = load_universe(root)
    d = development(markets)
    d_bytes = json.dumps(d, sort_keys=True, default=float).encode()
    (run_dir / "development.json").write_bytes(d_bytes)
    ledger.write_text(json.dumps({"chosen": d["chosen"], "rule": d["rule"],
                                  "development_sha256": hashlib.sha256(d_bytes).hexdigest()},
                                 indent=2) + "\n")  # fmt: skip
    on_line(f"development: chose {d['chosen']} "
            f"(Sharpe {d['board'][d['chosen']]['sharpe_monthly']:.2f}); ledger written")  # fmt: skip
    h = held_out(markets, Rule(**d["rule"]))
    h_path.write_text(json.dumps(h, indent=2, default=float) + "\n")
    r = h["result"]
    on_line(f"held out, read once: {h['status']}, {r['total_pct']:+.1f}% "
            f"({r['a_year_pct']:+.1f}% a year), beats {h['beats_placebos']} of {PLACEBOS} placebos")  # fmt: skip
    return h_path


__all__ = ["DEVELOPMENT", "GRID", "HELD_OUT", "development", "held_out", "readable", "run"]
