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
from trading_research.xs_weekly import EXCLUDED, features, panel, portfolio, rankers
from trading_research.xs_weekly import net as net_study
from trading_research.xs_weekly.rankers import MODELS

#: Each study: its rankers, its walk, the rule its models are compared with, its keys and folders.
STUDIES: dict[str, dict[str, Any]] = {
    "gross": {"names": rankers.NAMES, "walk": rankers.walk, "baseline": "low_vol",
              "keys": ("W1", "W2"), "out": "xs_weekly"},
    "net": {"names": net_study.NAMES, "walk": net_study.walk, "baseline": "momentum",
            "keys": ("N1", "N2"), "out": "xs_weekly_net"},
}  # fmt: skip

ROOT = Path(__file__).resolve().parents[1]
D_MONTHS = [str(m) for m in pd.period_range("2020-01", "2024-12", freq="M")]
H_MONTHS = [str(m) for m in pd.period_range("2025-01", "2026-10", freq="M")]
D_FIRST, D_LAST = "2021-07-05", "2024-12-30"
H_FIRST, H_LAST = "2025-01-06", "2026-10-26"
RESULTS = ROOT / "experiments" / "results"
SOURCES = tuple(
    f"src/trading_research/{p}.py"
    for p in ("data/binance_monthly", "xs_weekly/__init__", "xs_weekly/panel", "xs_weekly/features",
              "xs_weekly/rankers", "xs_weekly/portfolio", "xs_weekly/net", "xs_deep/rankers", "xs_deep/nets",
              "xs_deep/features")
)  # fmt: skip
#: sha256 of each source at the code freeze; written by its amendment.
FROZEN: dict[str, str] = {
    "src/trading_research/data/binance_monthly.py": "015dfa3de0e5e70d86fa55734e79927514bb016e72986eabf211aa5c7a924b40",
    "src/trading_research/xs_weekly/__init__.py": "3759852c500cc7db7e01570539262623375401965aedc696f3fd2d0bcb1dcfa5",
    "src/trading_research/xs_weekly/panel.py": "488a10c0037bac49842f9f4477518355c6d5b3ba051126fe4db833b9174a3e8a",
    "src/trading_research/xs_weekly/features.py": "5101964ba0c0d3aa6e0b4db6fb0ebb064786171d2459bf722bcf5d8ef5c89aee",
    "src/trading_research/xs_weekly/rankers.py": "3ead61c5f592a8a9f6cdf6fe06f18ded911d0195dd5723be25763384b75c1132",
    "src/trading_research/xs_weekly/portfolio.py": "dfd028e3e2dd5e277d188940cd36662f5c0c192ddac467f8ae953f563ab441c0",
    "src/trading_research/xs_weekly/net.py": "c3ce42284bb719e1379aed72b1d23f0dd51f91c4ddd1c13ecb24a57899a13d8e",
    "src/trading_research/xs_deep/rankers.py": "d34594e1a7559617da9afb5d8f7eaddbb8d10ef4aa3b9779873c667831f1f0ac",
    "src/trading_research/xs_deep/nets.py": "6afd32f16939f853f9f3023a0119ac13ad93dd089405d8cc61051f09e5143510",
    "src/trading_research/xs_deep/features.py": "99f4922b980fd6015de819e2b54bddc70f36584613397d2fb226987b458ab433",
}


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


def run_one(root: Path, phase: str, name: str, study: str) -> None:
    p = panel.read(panel_path(root, phase))
    frame = features.build(p)
    first, last = (D_FIRST, D_LAST) if phase == "dev" else (H_FIRST, H_LAST)
    picks = list(STUDIES[study]["walk"](frame, name, mondays(p, first, last)))
    out = root / study / phase / f"{name}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, days=np.array([t for t, _, _ in picks]),
                        coins=np.stack([c for _, c, _ in picks]), scores=np.stack([s for _, _, s in picks]))  # fmt: skip


def run_all(root: Path, phase: str, study: str) -> None:
    procs = []
    for name in STUDIES[study]["names"]:
        cmd = [sys.executable, "-m", "experiments.xs_weekly", "run-one", "--phase", phase, "--study", study,
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


def measure(root: Path, phase: str, study: str) -> dict[str, Any]:
    p = panel.read(panel_path(root, phase))
    frame = features.build(p)
    if study == "net":
        frame = net_study.net(frame)
    names, baseline = STUDIES[study]["names"], STUDIES[study]["baseline"]
    books = {}
    for name in names:
        z = np.load(root / study / phase / f"{name}.npz")
        picks = list(zip(z["days"].tolist(), z["coins"], z["scores"], strict=True))
        books[name] = portfolio.book(frame, picks).set_index("day")
    common = sorted(set.intersection(*(set(b.index) for b in books.values())))
    books = {n: b.loc[common] for n, b in books.items()}
    base = books[baseline]["net"].to_numpy()
    return {
        "weeks": [str(p.days[d].date()) for d in common],
        "summary": {n: portfolio.summary(b) for n, b in books.items()},
        "weekly_net": {n: b["net"].round(6).tolist() for n, b in books.items()},
        "net": {n: portfolio.one_sided(b["net"].to_numpy()) for n, b in books.items()},
        "baseline": baseline,
        "minus_baseline": {
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
    parser.add_argument("--ranker")
    parser.add_argument("--study", choices=tuple(STUDIES), default="gross")
    a = parser.parse_args()
    root = a.data / "xs_weekly"
    if a.command == "fetch-dev":
        print(fetch(root, D_MONTHS), flush=True)
    elif a.command == "run-one":
        run_one(root, a.phase, a.ranker, a.study)
    elif a.command == "check":
        bad = unfrozen()
        print("frozen sources match" if not bad else f"differ: {', '.join(bad)}")
    elif a.command == "dev":
        out = RESULTS / STUDIES[a.study]["out"]
        if not panel_path(root, "dev").exists():
            build_panel(root, "dev")
        run_all(root, "dev", a.study)
        out.mkdir(parents=True, exist_ok=True)
        r = measure(root, "dev", a.study)
        (out / "development.json").write_text(json.dumps(r, indent=2, default=float) + "\n")
    else:
        if bad := unfrozen():
            raise SystemExit(f"not the frozen sources: {', '.join(bad)}")
        if a.command == "fetch-heldout":
            print(fetch(root, H_MONTHS), flush=True)
            return
        out = RESULTS / (STUDIES[a.study]["out"] + "_heldout")
        if (out / "result.json").exists():
            raise SystemExit("already read; the result is written once")
        if not panel_path(root, "heldout").exists():
            build_panel(root, "heldout")
        run_all(root, "heldout", a.study)
        r = measure(root, "heldout", a.study)
        k1, k2 = STUDIES[a.study]["keys"]
        for key, table in ((k1, "net"), (k2, "minus_baseline")):
            adj = holm({n: r[table][n]["p"] for n in MODELS})
            r[key] = {n: {**r[table][n], "p_holm": adj[n],
                          "status": "passes" if r[table][n]["mean"] > 0 and adj[n] < 0.05 else "killed"}
                      for n in MODELS}  # fmt: skip
        r["read_at"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        out.mkdir(parents=True, exist_ok=True)
        (out / "result.json").write_text(json.dumps(r, indent=2, default=float) + "\n")
        print(json.dumps({k: r[k] for k in (k1, k2)}, indent=1, default=float))


if __name__ == "__main__":
    main()
