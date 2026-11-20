"""xs_weekly_2026_11: fetch, the development replay (a measurement), and the read of W1 and W2 on H.

    PYTHONPATH=src python -m experiments.xs_weekly fetch-dev --data data
    PYTHONPATH=src python -m experiments.xs_weekly dev --data data
    PYTHONPATH=src python -m experiments.xs_weekly fetch-heldout --data data
    PYTHONPATH=src python -m experiments.xs_weekly read --data data

Each ranker runs in its own process (LightGBM and torch each bring an OpenMP
runtime). ``fetch-heldout`` and ``read`` refuse if a source of the study
differs from :data:`FROZEN`; ``read`` writes its result once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_research.data import binance_monthly
from trading_research.xs_weekly import EXCLUDED, features, panel, portfolio
from trading_research.xs_weekly.rankers import MODELS, NAMES, walk

ROOT = Path(__file__).resolve().parents[1]
D_MONTHS = [str(m) for m in pd.period_range("2020-01", "2024-12", freq="M")]
H_MONTHS = [str(m) for m in pd.period_range("2025-01", "2026-10", freq="M")]
D_FIRST, D_LAST = "2021-07-05", "2024-12-30"
H_FIRST, H_LAST = "2025-01-06", "2026-10-26"
OUT_DEV = ROOT / "experiments" / "results" / "xs_weekly"
OUT_H = ROOT / "experiments" / "results" / "xs_weekly_heldout"
SOURCES = tuple(
    f"src/trading_research/{p}.py"
    for p in ("data/binance_monthly", "xs_weekly/__init__", "xs_weekly/panel", "xs_weekly/features",
              "xs_weekly/rankers", "xs_weekly/portfolio", "xs_deep/rankers", "xs_deep/nets",
              "xs_deep/features")
)  # fmt: skip
#: sha256 of each source at the code freeze; written by its amendment.
FROZEN: dict[str, str] = {}


def unfrozen() -> list[str]:
    if not FROZEN:
        return list(SOURCES)
    return [
        s for s in SOURCES if FROZEN.get(s) != hashlib.sha256((ROOT / s).read_bytes()).hexdigest()
    ]


def symbols(root: Path) -> list[str]:
    cached = root / "symbols.txt"
    if cached.exists():
        return cached.read_text().split()
    every = binance_monthly.list_symbols()
    syms = sorted(s for s in every if re.fullmatch(r"[A-Z0-9]+USDT", s) and s not in EXCLUDED)
    root.mkdir(parents=True, exist_ok=True)
    cached.write_text("\n".join(syms) + "\n")
    return syms


def fetch(root: Path, months: list[str]) -> dict[str, int]:
    syms = symbols(root)
    with ThreadPoolExecutor(4) as pool:
        listed = dict(zip(syms, pool.map(binance_monthly.listed_months, syms), strict=True))
    counts = {"have": 0, "got": 0, "missing": 0}
    for s in syms:
        wanted = sorted(set(months) & set(listed[s]))
        if wanted:
            for k, v in binance_monthly.download(root, [s], wanted).items():
                counts[k] += v
    return counts


def panel_path(root: Path, phase: str) -> Path:
    return root / f"panel_{phase}.npz"


def build_panel(root: Path, phase: str) -> None:
    months = D_MONTHS + (H_MONTHS if phase == "heldout" else [])
    panel.save(panel.build(root, symbols(root), months), panel_path(root, phase))


def mondays(p: panel.Panel, first: str, last: str) -> np.ndarray:
    d = p.days
    keep = (
        (d.dayofweek == 0)
        & (d >= pd.Timestamp(first, tz="UTC"))
        & (d <= pd.Timestamp(last, tz="UTC"))
    )
    return np.flatnonzero(keep)


def run_one(root: Path, phase: str, name: str) -> None:
    p = panel.read(panel_path(root, phase))
    frame = features.build(p)
    first, last = (D_FIRST, D_LAST) if phase == "dev" else (H_FIRST, H_LAST)
    picks = list(walk(frame, name, mondays(p, first, last)))
    out = root / phase / f"{name}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, days=np.array([t for t, _, _ in picks]),
                        coins=np.stack([c for _, c, _ in picks]), scores=np.stack([s for _, _, s in picks]))  # fmt: skip


def run_all(root: Path, phase: str) -> None:
    build_panel(root, phase)
    procs = []
    for name in NAMES:
        cmd = [sys.executable, "-m", "experiments.xs_weekly", "run-one", "--phase", phase,
               "--ranker", name, "--data", str(root.parent)]  # fmt: skip
        procs.append(subprocess.Popen(cmd, cwd=ROOT))
        if len(procs) == 3:  # three at a time
            for pr in procs:
                if pr.wait() != 0:
                    raise SystemExit(f"{pr.args} failed")
            procs = []
    for pr in procs:
        if pr.wait() != 0:
            raise SystemExit(f"{pr.args} failed")


def measure(root: Path, phase: str) -> dict[str, Any]:
    p = panel.read(panel_path(root, phase))
    frame = features.build(p)
    books = {}
    for name in NAMES:
        z = np.load(root / phase / f"{name}.npz")
        picks = list(zip(z["days"].tolist(), z["coins"], z["scores"], strict=True))
        books[name] = portfolio.book(frame, picks).set_index("day")
    common = sorted(set.intersection(*(set(b.index) for b in books.values())))
    books = {n: b.loc[common] for n, b in books.items()}
    base = books["low_vol"]["net"].to_numpy()
    return {
        "weeks": [str(p.days[d].date()) for d in common],
        "summary": {n: portfolio.summary(b) for n, b in books.items()},
        "weekly_net": {n: b["net"].round(6).tolist() for n, b in books.items()},
        "net": {n: portfolio.one_sided(b["net"].to_numpy()) for n, b in books.items()},
        "minus_low_vol": {
            n: portfolio.one_sided(books[n]["net"].to_numpy() - base) for n in MODELS
        },
        "universe_size": float(frame.universe[common].sum(axis=1).mean()),
    }


def holm(p: dict[str, float]) -> dict[str, float]:
    from trading_research.pairs.stats import holm_adjusted

    names = list(p)
    return dict(zip(names, map(float, holm_adjusted([p[k] for k in names])), strict=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "command", choices=("fetch-dev", "dev", "fetch-heldout", "read", "run-one", "check")
    )
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--phase", choices=("dev", "heldout"), default="dev")
    parser.add_argument("--ranker", choices=NAMES)
    a = parser.parse_args()
    root = a.data / "xs_weekly"
    if a.command == "fetch-dev":
        print(fetch(root, D_MONTHS), flush=True)
    elif a.command == "run-one":
        run_one(root, a.phase, a.ranker)
    elif a.command == "dev":
        run_all(root, "dev")
        OUT_DEV.mkdir(parents=True, exist_ok=True)
        (OUT_DEV / "development.json").write_text(
            json.dumps(measure(root, "dev"), indent=2, default=float) + "\n"
        )
    elif a.command == "check":
        bad = unfrozen()
        print("frozen sources match" if not bad else f"differ: {', '.join(bad)}")
    else:
        if bad := unfrozen():
            raise SystemExit(f"not the frozen sources: {', '.join(bad)}")
        if a.command == "fetch-heldout":
            print(fetch(root, H_MONTHS), flush=True)
            return
        if (OUT_H / "result.json").exists():
            raise SystemExit("already read; the result is written once")
        run_all(root, "heldout")
        r = measure(root, "heldout")
        for key, table in (("W1", "net"), ("W2", "minus_low_vol")):
            adj = holm({n: r[table][n]["p"] for n in MODELS})
            r[key] = {n: {**r[table][n], "p_holm": adj[n],
                          "status": "passes" if r[table][n]["mean"] > 0 and adj[n] < 0.05 else "killed"}
                      for n in MODELS}  # fmt: skip
        r["read_at"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        OUT_H.mkdir(parents=True, exist_ok=True)
        (OUT_H / "result.json").write_text(json.dumps(r, indent=2, default=float) + "\n")
        print(json.dumps({k: r[k] for k in ("W1", "W2")}, indent=1, default=float))


if __name__ == "__main__":
    main()
