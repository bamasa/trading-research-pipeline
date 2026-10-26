"""The replication's pooled numbers per window, variant and fee, from its per-day rows.

* ``net_bp_trade``: the window's net in quote over its entry notional, x 1e4,
  every trade weighted by its notional (the spot study's daily value, pooled);
  ``net_bp_day_mean`` and ``day_t`` are the mean and the day-level t of the
  daily values over the days with a trade, descriptive only;
* ``net_quote_per_trade``: quote (FDUSD) per round trip of one 0.010 BTC clip;
* ``gross_bp_turnover``: the gross over both sides' notional, x 1e4, which is
  the break-even taker fee per side;
* ``share_days_positive``: days whose net is above zero, of the days with a trade.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import pandas as pd

KEYS: tuple[str, ...] = ("window", "variant", "fees")


def _ratio(top: float, bottom: float) -> float:
    return top / bottom * 1e4 if bottom > 0 else math.nan


def pooled_group(g: pd.DataFrame) -> dict[str, Any]:
    """One group's pooled numbers; a group's days are its rows (one per usable test day)."""
    traded = g[g["trades"] > 0]
    trades = int(g["trades"].sum())
    net, gross = float(g["net_quote"].sum()), float(g["gross_quote"].sum())
    notional, turnover = float(g["entry_notional"].sum()), float(g["turnover"].sum())
    values = traded["net_bp"].astype(float)
    sd = float(values.std(ddof=1)) if len(values) > 1 else math.nan
    return {
        "days": len(g), "days_with_trade": len(traded), "trades": trades,
        "trades_per_day": trades / len(g) if len(g) else math.nan,
        "net_bp_trade": _ratio(net, notional),
        "net_quote_per_trade": net / trades if trades else math.nan,
        "net_quote_per_day": net / len(g) if len(g) else math.nan,
        "share_days_positive": float((traded["net_quote"] > 0).mean()) if len(traded) else math.nan,
        "net_bp_day_mean": float(values.mean()) if len(values) else math.nan,
        "day_t": float(values.mean()) / sd * math.sqrt(len(values)) if sd > 0 else math.nan,
        "gross_bp_trade": _ratio(gross, notional),
        "gross_bp_turnover": _ratio(gross, turnover),
        "long_share": float(g["trades_long"].sum()) / trades if trades else math.nan,
        "decisions_per_day": float(g["decisions"].sum()) / len(g) if len(g) else math.nan,
        "ic_mean": float(g["ic"].mean()) if len(g) else math.nan,
        "market_bp_day_mean": float(g["market_bp"].mean()) if len(g) else math.nan,
    }  # fmt: skip


def pooled(rows: pd.DataFrame, windows: Sequence[str] = ("A", "B")) -> pd.DataFrame:
    """Every window's groups, then the windows together (``window`` = ``A+B``)."""
    found = []
    frames = [rows]
    if len(windows) > 1:
        frames.append(rows[rows["window"].isin(windows)].assign(window="+".join(windows)))
    for frame in frames:
        for key, g in frame.groupby(list(KEYS), sort=False):
            found.append(dict(zip(KEYS, key, strict=True)) | pooled_group(g))
    out = pd.DataFrame(found)
    order = {"two_sided": 0, "long_only": 1, "zero": 0, "regular_10bp": 1}
    places = {w: k for k, w in enumerate(windows)}
    ranks = out["window"].map(places).fillna(len(windows)) * 10 + out["variant"].map(order)
    out = out.assign(_rank=ranks + out["fees"].map(order) / 10)
    return out.sort_values("_rank", kind="stable").drop(columns="_rank").reset_index(drop=True)
