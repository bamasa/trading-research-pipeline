"""``trp search basis``: the automatic search for a spot-perpetual basis trade, chosen on D and read once on H.

The registration is ``docs/preregistration/basis_search_2026_11.md``. Every
rule of :data:`GRID` (how rich the perpetual must be to open, how far back it
must come to close, over how long a median) is replayed on the development
months; the one with the highest monthly Sharpe ratio there is chosen, written
to the ledger with the development results' sha256, and only then replayed on
the held-out months, once. Beside it, nineteen placebos trade the same rule on
each coin's deviation shifted to random hours: as many signals, at the wrong
times, so the result says what reading the basis adds over holding the carry
at any hour.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

from trading_research.search_engine.carry import readable
from trading_research.strategies.basis import BasisRule, Panel, deviation, panel, replay, shifted
from trading_research.strategies.carry import UNIVERSE, load, summary

DEVELOPMENT = (pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2023-06-30 23:00", tz="UTC"))
HELD_OUT = (pd.Timestamp("2023-07-01", tz="UTC"), pd.Timestamp("2025-06-30 23:00", tz="UTC"))
GRID: tuple[BasisRule, ...] = tuple(
    BasisRule(entry_bp=e, exit_bp=x, window_days=w)
    for e, x, w in product((30.0, 45.0, 60.0), (0.0, 15.0), (7, 30))
)
PLACEBOS = 19


def load_panel(root: Path) -> Panel:
    months = pd.period_range("2020-12", "2025-06", freq="M")
    markets = {c: load(root, c, [m for m in months if readable(c, m)]) for c in UNIVERSE}
    hours = pd.date_range("2020-12-01", "2025-06-30 23:00", freq="1h", tz="UTC")
    return panel(markets, hours, readable)


def _score(s: dict[str, float]) -> tuple[float, float]:
    sharpe = s["sharpe_monthly"]
    return (sharpe if math.isfinite(sharpe) else -math.inf, s["total_pct"])


def _summary(frame: pd.DataFrame) -> dict[str, float]:
    out = summary(frame)
    out["trades"] = int(frame.attrs.get("trades", 0))
    return out


def _placebos(p: Panel, rule: BasisRule, span: tuple[pd.Timestamp, pd.Timestamp],
              seed: int) -> list[dict[str, float]]:  # fmt: skip
    dev = deviation(p, rule.window_days)
    lo, hi = int(p.hours.searchsorted(span[0])), int(p.hours.searchsorted(span[1], side="right"))
    return [_summary(replay(p, rule, *span, dev=shifted(dev, lo, hi, seed + j)))
            for j in range(PLACEBOS)]  # fmt: skip


def development(p: Panel) -> dict[str, Any]:
    board = {r.name: _summary(replay(p, r, *DEVELOPMENT)) for r in GRID}
    best = max(GRID, key=lambda r: _score(board[r.name]))
    return {"board": board, "chosen": best.name, "rule": best.__dict__,
            "placebo": _placebos(p, best, DEVELOPMENT, 0)}  # fmt: skip


def held_out(p: Panel, rule: BasisRule) -> dict[str, Any]:
    frame = replay(p, rule, *HELD_OUT)
    result = _summary(frame)
    placebo = _placebos(p, rule, HELD_OUT, 100)
    beaten = sum(x["total_pct"] < result["total_pct"] for x in placebo)
    if result["total_pct"] <= 0:
        status = "killed: not positive"
    elif beaten < PLACEBOS:
        status = "killed: a placebo earned as much"
    else:
        status = "positive"
    return {"result": result, "placebo": placebo, "beats_placebos": beaten, "status": status,
            "months": frame.to_dict("records")}  # fmt: skip


def run(root: Path, run_dir: Path, on_line: Callable[[str], None] = print) -> Path:
    """D, the ledger, then H once. Refuses a second read of H."""
    run_dir.mkdir(parents=True, exist_ok=True)
    h_path, ledger = run_dir / "held_out.json", run_dir / "ledger.json"
    if h_path.exists():
        raise FileExistsError(f"{h_path} exists: the held-out months are read once")
    p = load_panel(root)
    d = development(p)
    d_bytes = json.dumps(d, sort_keys=True, default=float).encode()
    (run_dir / "development.json").write_bytes(d_bytes)
    ledger.write_text(json.dumps({"chosen": d["chosen"], "rule": d["rule"],
                                  "development_sha256": hashlib.sha256(d_bytes).hexdigest()},
                                 indent=2) + "\n")  # fmt: skip
    c = d["board"][d["chosen"]]
    on_line(f"development: chose {d['chosen']} (Sharpe {c['sharpe_monthly']:.2f}, "
            f"{c['total_pct']:+.1f}%, {c['trades']} trades); ledger written")  # fmt: skip
    h = held_out(p, BasisRule(**d["rule"]))
    h_path.write_text(json.dumps(h, indent=2, default=float) + "\n")
    r = h["result"]
    on_line(f"held out, read once: {h['status']}, {r['total_pct']:+.1f}% "
            f"({r['a_year_pct']:+.1f}% a year, {r['trades']} trades), "
            f"beats {h['beats_placebos']} of {PLACEBOS} placebos")  # fmt: skip
    return h_path


__all__ = ["DEVELOPMENT", "GRID", "HELD_OUT", "development", "held_out", "run"]
