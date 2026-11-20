"""The weekly long-short book: the ten coins ranked highest long, the ten lowest short.

Half the account is long and half short, equal weight, unlevered. A week's
return is the legs' simple returns from Monday's open to the next Monday's (or
a delisted coin's last close), plus the funding a short receives and minus what
a long pays, minus 7 bp of fee and slippage on the weight of every leg opened or
closed; a coin kept on the same side trades nothing.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from trading_research.market_making.verdicts import t_sf
from trading_research.xs_weekly.features import Frame

LEGS = 10
SIDE_BP = 5.0 + 2.0


def book(frame: Frame, picks: list[tuple[int, np.ndarray, np.ndarray]]) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    held: dict[int, int] = {}
    w = 0.5 / LEGS
    for t, idx, s in picks:
        ok = np.isfinite(s) & (idx >= 0) & np.isfinite(frame.gross[t, np.maximum(idx, 0)])
        c, sc = idx[ok], s[ok]
        order = np.argsort(sc, kind="stable")
        side = {int(x): -1 for x in c[order[:LEGS]]} | {int(x): 1 for x in c[order[-LEGS:]]}
        changed = len([k for k in held if side.get(k) != held[k]])
        changed += len([k for k in side if held.get(k) != side[k]])
        cost = changed * w * SIDE_BP / 1e4
        price = sum(v * w * math.expm1(frame.gross[t, k]) for k, v in side.items())
        funding = sum(-v * w * frame.funding_week[t, k] / 1e4 for k, v in side.items())
        ic = _spearman(sc, frame.target[t, c])
        rows.append({"day": t, "price": price, "funding": funding, "cost": -cost,
                     "net": price + funding - cost, "ic": ic, "turnover": changed / (2 * LEGS)})  # fmt: skip
        held = side
    return pd.DataFrame(rows)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return math.nan
    ra = np.argsort(np.argsort(a[ok], kind="stable"), kind="stable")
    rb = np.argsort(np.argsort(b[ok], kind="stable"), kind="stable")
    return float(np.corrcoef(ra, rb)[0, 1])


def summary(weeks: pd.DataFrame) -> dict[str, float]:
    r = weeks["net"].to_numpy()
    equity = np.cumprod(1 + r)
    peak = np.maximum.accumulate(np.r_[1.0, equity])
    years = len(r) / 52
    sd = float(r.std(ddof=1))
    return {"total_pct": float(equity[-1] - 1) * 100,
            "a_year_pct": float(equity[-1] ** (1 / years) - 1) * 100,
            "sharpe": float(r.mean() / sd * math.sqrt(52)) if sd > 0 else math.nan,
            "worst_drawdown_pct": float((np.r_[1.0, equity] / peak - 1).min() * 100),
            "price_pct": float(weeks["price"].sum() * 100), "funding_pct": float(weeks["funding"].sum() * 100),
            "cost_pct": float(weeks["cost"].sum() * 100), "ic": float(weeks["ic"].mean()),
            "turnover": float(weeks["turnover"].mean()), "weeks": len(r)}  # fmt: skip


def one_sided(x: np.ndarray) -> dict[str, float]:
    x = x[np.isfinite(x)]
    n, sd = len(x), float(np.std(x, ddof=1))
    t = float(x.mean() / (sd / math.sqrt(n))) if sd > 0 else math.nan
    return {"mean": float(x.mean()), "t": t, "p": t_sf(t, n - 1), "weeks": n}
