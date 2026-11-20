"""xvenue_funding_2026_11: the development replay (a measurement) and the read of V1 and V2 on H.

``dev`` fetches Bybit's side of January 2022 to June 2025 (and the three
months before) and replays the rule there. ``fetch`` and ``read`` touch H, July
2025 to July 2026, and refuse if a source differs from :data:`FROZEN`; ``read``
writes its result once.

    PYTHONPATH=src python -m experiments.xvenue_funding dev --data data
    PYTHONPATH=src python -m experiments.xvenue_funding fetch --data data
    PYTHONPATH=src python -m experiments.xvenue_funding read --data data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_research.data import bybit_funding, bybit_klines
from trading_research.market_making.verdicts import t_sf
from trading_research.strategies.carry import UNIVERSE, load, summary
from trading_research.xvenue import engine

ROOT = Path(__file__).resolve().parents[1]
COINS = tuple(c for c in UNIVERSE if c not in ("BTCUSDT", "ETHUSDT"))
D = pd.period_range("2022-01", "2025-06", freq="M")
H = pd.period_range("2025-07", "2026-07", freq="M")
D_LAST = D[-1]
PLACEBOS = 19
OUT_DEV = ROOT / "experiments" / "results" / "xvenue_funding"
OUT_H = ROOT / "experiments" / "results" / "xvenue_funding_heldout"
SOURCES = (
    "src/trading_research/xvenue/__init__.py",
    "src/trading_research/xvenue/engine.py",
    "src/trading_research/data/bybit_klines.py",
)
#: sha256 of each source at the code freeze; written by its amendment.
FROZEN: dict[str, str] = {}


def unfrozen() -> list[str]:
    if not FROZEN:
        return list(SOURCES)
    return [
        s for s in SOURCES if FROZEN.get(s) != hashlib.sha256((ROOT / s).read_bytes()).hexdigest()
    ]


def _day(p: pd.Period, end: bool = False) -> date:
    return (p.end_time if end else p.start_time).date()


def venues(data: Path, months: pd.PeriodIndex) -> dict[str, engine.Venues]:
    """Both venues' perpetual closes and funding for each coin over ``months`` and the lookback."""
    span = pd.period_range(months[0] - engine.LOOKBACK, months[-1], freq="M")
    bybit = data / "xvenue" / "bybit"
    out = {}
    for coin in COINS:
        parts = [load(data / "carry_walkforward", coin, [m for m in span if m <= D_LAST])]
        if span[-1] > D_LAST:
            parts.append(load(data / "carry_ml", coin, [m for m in span if m > D_LAST]))
        perp = pd.concat([p.perp for p in parts]).sort_index()
        funding = pd.concat([p.funding for p in parts]).sort_index()
        a, b = _day(span[0]), _day(span[-1], end=True)
        bybit_klines.download_range(coin, a, b, bybit / "klines")
        if not (bybit / "funding" / coin / f"{b}.parquet").exists():
            bybit_funding.download_range(coin, a, b, bybit / "funding")
        days = pd.date_range(a, b).date
        files = [bybit / "funding" / coin / f"{d}.parquet" for d in days]
        fr = pd.concat([pd.read_parquet(f) for f in files if f.exists()], ignore_index=True)
        rates = pd.Series(
            fr["rate"].to_numpy(), index=pd.DatetimeIndex(fr["timestamp"]).floor("1h")
        )
        out[coin] = engine.Venues(
            binance=perp[~perp.index.duplicated()].tz_localize("UTC") if perp.index.tz is None
            else perp[~perp.index.duplicated()],
            binance_funding=funding.tz_localize("UTC") if funding.index.tz is None else funding,
            bybit=bybit_klines.load(coin, a, b, bybit / "klines"),
            bybit_funding=rates.groupby(level=0).sum().sort_index(),
        )  # fmt: skip
    return out


def one_sided(x: np.ndarray) -> dict[str, float]:
    n, sd = len(x), float(np.std(x, ddof=1))
    t = float(np.mean(x) / (sd / math.sqrt(n))) if sd > 0 else math.nan
    return {"mean": float(np.mean(x)), "t": t, "p": t_sf(t, n - 1), "months": n}


def measure(vs: dict[str, engine.Venues], months: pd.PeriodIndex) -> dict[str, Any]:
    frame = engine.replay(vs, months)
    chosen = {m: engine.choose(vs, m) for m in months}
    rng = np.random.default_rng(100)
    placebo = []
    for _ in range(PLACEBOS):
        flips = {m: [(c, int(rng.choice([-1, 1]))) for c, _ in chosen[m]] for m in months}
        placebo.append(summary(engine.replay(vs, months, chooser=lambda m, f=flips: f[m])))
    spreads = {c: [engine.spread(v, m) for m in months] for c, v in vs.items()}
    return {
        "months": [str(m) for m in months],
        "summary": summary(frame),
        "monthly": frame.round(5).to_dict("records"),
        "split_pct": {k: float(frame[k].sum()) for k in ("funding_pct", "price_pct", "cost_pct")},
        "months_empty": int((frame["pairs"] == "").sum()),
        "monthly_mean": one_sided(frame["return_pct"].to_numpy()),
        "placebo_total_pct": sorted(p["total_pct"] for p in placebo),
        "spread_bp_a_day": {
            c: {
                "mean": float(np.nanmean([x for x in s if x is not None] or [np.nan])),
                "mean_abs": float(np.nanmean([abs(x) for x in s if x is not None] or [np.nan])),
                "months_above_cost": sum(
                    x is not None and abs(x) > engine.COST_A_DAY_BP for x in s
                ),
            }
            for c, s in spreads.items()
        },
    }


def dev(data: Path) -> None:
    result = measure(venues(data, D), D)
    OUT_DEV.mkdir(parents=True, exist_ok=True)
    (OUT_DEV / "development.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(json.dumps({k: result[k] for k in ("summary", "split_pct", "months_empty", "monthly_mean",
                                              "placebo_total_pct")}, indent=1, default=float))  # fmt: skip


def read(data: Path) -> dict[str, Any]:
    if bad := unfrozen():
        raise SystemExit(f"not the frozen sources: {', '.join(bad)}")
    if (OUT_H / "result.json").exists():
        raise SystemExit("already read; the result is written once")
    result = measure(venues(data, H), H)
    mean = result["monthly_mean"]
    total = result["summary"]["total_pct"]
    result["V1"] = {
        **mean,
        "status": "passes" if mean["mean"] > 0 and mean["p"] < 0.05 else "killed",
    }
    beaten = sum(p < total for p in result["placebo_total_pct"])
    result["V2"] = {"beats": beaten, "of": PLACEBOS,
                    "status": "passes" if beaten == PLACEBOS else "killed"}  # fmt: skip
    result["read_at"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    OUT_H.mkdir(parents=True, exist_ok=True)
    (OUT_H / "result.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("dev", "fetch", "read", "check"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    a = parser.parse_args()
    if a.command == "dev":
        dev(a.data)
    elif a.command == "check":
        bad = unfrozen()
        print("frozen sources match" if not bad else f"differ: {', '.join(bad)}")
    elif a.command == "fetch":
        if bad := unfrozen():
            raise SystemExit(
                f"H is fetched only after the code freeze; not frozen: {', '.join(bad)}"
            )
        venues(a.data, H)
    else:
        r = read(a.data)
        print(
            json.dumps(
                {k: r[k] for k in ("V1", "V2", "summary", "split_pct")}, indent=1, default=float
            )
        )


if __name__ == "__main__":
    main()
