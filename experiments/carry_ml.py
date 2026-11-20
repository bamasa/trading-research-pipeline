"""carry_ml_2026_11: the development replay (a measurement) and the read of C1 and C2 on H.

``docs/preregistration/carry_ml_2026_11.md`` registers five choosers of the
carry's one coin. ``dev`` replays them on the carry search's months, January
2021 to June 2025, compared from January 2022: a measurement on months read
before. ``fetch`` and ``read`` touch H, July 2025 to July 2026, and refuse if
any source of the study differs from :data:`FROZEN`; ``read`` writes its
result once.

    PYTHONPATH=src python -m experiments.carry_ml dev --data data
    PYTHONPATH=src python -m experiments.carry_ml fetch --data data
    PYTHONPATH=src python -m experiments.carry_ml read --data data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from trading_research.carry_ml import evaluate, features
from trading_research.carry_ml.models import MODELS, NAMES, scores
from trading_research.search_engine import carry as search
from trading_research.strategies.carry import UNIVERSE, Market, load

ROOT = Path(__file__).resolve().parents[1]
D_FIRST, D_LAST = pd.Period("2021-01", "M"), pd.Period("2025-06", "M")
COMPARED_FROM = pd.Period("2022-01", "M")
H_FIRST, H_LAST = pd.Period("2025-07", "M"), pd.Period("2026-07", "M")
HELD_BACK = {"BTCUSDT", "ETHUSDT"}  # funding_carry_2026, from July 2025 on
OUT_DEV = ROOT / "experiments" / "results" / "carry_ml"
OUT_H = ROOT / "experiments" / "results" / "carry_ml_heldout"
SOURCES = tuple(
    f"src/trading_research/carry_ml/{n}.py" for n in ("__init__", "features", "models", "evaluate")
)
#: sha256 of each source at the code freeze; written by its amendment.
FROZEN: dict[str, str] = {
    "src/trading_research/carry_ml/__init__.py": "66c8eac6cec71d631ec4ed5781a1c972c91cb506ed6a194b844b97edc1f254d4",
    "src/trading_research/carry_ml/features.py": "9b5cd92d52e926c1e37a1b112c088b14984fc8b9e33d344d5fe96dc4414d2588",
    "src/trading_research/carry_ml/models.py": "c7b37fac970bc57baf92bdb0d00ca5a50c3f7693abbd11c7f8b3841310a21475",
    "src/trading_research/carry_ml/evaluate.py": "fa40d88d7777729736e019030028703bbde23b1bd6e91464e5c248f6ebcb65b2",
}


def readable(coin: str, month: pd.Period) -> bool:
    if month <= D_LAST:
        return search.readable(coin, month)
    return coin not in HELD_BACK and month <= H_LAST


def unfrozen() -> list[str]:
    if not FROZEN:
        return list(SOURCES)
    return [
        s for s in SOURCES if FROZEN.get(s) != hashlib.sha256((ROOT / s).read_bytes()).hexdigest()
    ]


def markets(data: Path, last: pd.Period) -> dict[str, Market]:
    """Each coin's readable months up to ``last``: D's from the search's files, H's from this study's."""
    first = D_FIRST - features.HISTORY
    out = {}
    for coin in UNIVERSE:
        months = [m for m in pd.period_range(first, last, freq="M") if readable(coin, m)]
        parts = [load(data / "carry_walkforward", coin, [m for m in months if m <= D_LAST])]
        if any(m > D_LAST for m in months):
            parts.append(load(data / "carry_ml", coin, [m for m in months if m > D_LAST]))
        out[coin] = Market(*(pd.concat([getattr(p, k) for p in parts]).sort_index()
                             for k in ("spot", "perp", "funding")))  # fmt: skip
    return out


def score_all(rows_path: Path, months: pd.PeriodIndex, folder: Path) -> dict[str, pd.DataFrame]:
    """Every chooser's scores, each in its own process (LightGBM and torch do not share one)."""
    folder.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        cmd = [sys.executable, "-m", "experiments.carry_ml", "score", "--name", name,
               "--rows", str(rows_path), "--first", str(months[0]), "--last", str(months[-1]),
               "--out", str(folder / f"{name}.parquet")]  # fmt: skip
        subprocess.run(cmd, check=True, cwd=ROOT)
    out = {}
    for name in NAMES:
        s = pd.read_parquet(folder / f"{name}.parquet")
        s["month"] = pd.PeriodIndex(s["month"], freq="M")
        out[name] = s
    return out


def read_rows(path: Path) -> pd.DataFrame:
    rows = pd.read_parquet(path)
    rows["month"] = pd.PeriodIndex(rows["month"], freq="M")
    return rows


def write_rows(rows: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.assign(month=rows["month"].astype(str)).to_parquet(path, index=False)


def measure(mk: dict[str, Market], rows: pd.DataFrame, scored: dict[str, pd.DataFrame],
            months: pd.PeriodIndex) -> dict[str, Any]:  # fmt: skip
    """Replays, forecast quality, differences against ``rule_lb3`` and placebos on ``months``."""
    from trading_research.strategies.carry import summary

    chosen = {n: evaluate.picks(s[s["month"].isin(months)]) for n, s in scored.items()}
    frames = {n: evaluate.run(mk, months, chosen[n], readable) for n in NAMES}
    quality = {
        n: evaluate.forecast_quality(s[s["month"].isin(months)], rows) for n, s in scored.items()
    }
    base = frames["rule_lb3"]["return_pct"].to_numpy()
    placebo = [summary(evaluate.placebo(mk, months, rows, readable, seed=100 + j))
               for j in range(evaluate.PLACEBOS)]  # fmt: skip
    return {
        "months": [str(m) for m in months],
        "summary": {n: summary(f) for n, f in frames.items()},
        "monthly_return_pct": {n: f["return_pct"].round(4).tolist() for n, f in frames.items()},
        "coins": {n: [chosen[n][m] for m in months] for n in NAMES},
        "forecast_quality": {n: evaluate.one_sided(q) for n, q in quality.items()},
        "return_minus_rule": {
            n: evaluate.one_sided(frames[n]["return_pct"].to_numpy() - base) for n in MODELS
        },
        "quality_minus_rule": {
            n: evaluate.one_sided((quality[n] - quality["rule_lb3"]).dropna()) for n in MODELS
        },
        "same_coin_as_rule": {
            n: sum(chosen[n][m] == chosen["rule_lb3"][m] for m in months) / len(months)
            for n in NAMES
        },
        "placebo_total_pct": sorted(p["total_pct"] for p in placebo),
    }


def dev(data: Path) -> None:
    mk = markets(data, D_LAST)
    rows = features.table(mk, pd.period_range(D_FIRST, D_LAST, freq="M"), readable)
    path = data / "carry_ml" / "dev" / "rows.parquet"
    write_rows(rows, path)
    scored = score_all(path, pd.period_range(D_FIRST, D_LAST, freq="M"), path.parent)
    result = measure(mk, rows, scored, pd.period_range(COMPARED_FROM, D_LAST, freq="M"))
    OUT_DEV.mkdir(parents=True, exist_ok=True)
    (OUT_DEV / "development.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(json.dumps({n: result["summary"][n] for n in NAMES}, indent=1, default=float))


def fetch(data: Path) -> None:
    if bad := unfrozen():
        raise SystemExit(f"H is fetched only after the code freeze; not frozen: {', '.join(bad)}")
    markets(data, H_LAST)


def read(data: Path) -> dict[str, Any]:
    if bad := unfrozen():
        raise SystemExit(f"not the frozen sources: {', '.join(bad)}")
    if (OUT_H / "result.json").exists():
        raise SystemExit("already read; the result is written once")
    mk = markets(data, H_LAST)
    rows = features.table(mk, pd.period_range(D_FIRST, H_LAST, freq="M"), readable)
    path = data / "carry_ml" / "heldout" / "rows.parquet"
    write_rows(rows, path)
    h = pd.period_range(H_FIRST, H_LAST, freq="M")
    result = measure(mk, rows, score_all(path, h, path.parent), h)
    for key, table in (("C1", "return_minus_rule"), ("C2", "quality_minus_rule")):
        adj = evaluate.holm({n: result[table][n]["p"] for n in MODELS})
        result[key] = {n: {**result[table][n], "p_holm": adj[n],
                           "status": "passes" if result[table][n]["mean"] > 0 and adj[n] < 0.05
                           else "killed"} for n in MODELS}  # fmt: skip
    result["read_at"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    OUT_H.mkdir(parents=True, exist_ok=True)
    (OUT_H / "result.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("dev", "fetch", "read", "score", "check"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--name", choices=NAMES)
    parser.add_argument("--rows", type=Path)
    parser.add_argument("--first")
    parser.add_argument("--last")
    parser.add_argument("--out", type=Path)
    a = parser.parse_args()
    if a.command == "score":
        months = pd.period_range(a.first, a.last, freq="M")
        s = scores(read_rows(a.rows), a.name, months)
        s.assign(month=s["month"].astype(str)).to_parquet(a.out, index=False)
    elif a.command == "dev":
        dev(a.data)
    elif a.command == "fetch":
        fetch(a.data)
    elif a.command == "check":
        bad = unfrozen()
        print("frozen sources match" if not bad else f"differ: {', '.join(bad)}")
    else:
        result = read(a.data)
        print(json.dumps({k: result[k] for k in ("C1", "C2", "summary")}, indent=1, default=float))


if __name__ == "__main__":
    main()
