"""Every symbol's daily bars and funding as arrays of day by coin.

Days run from the first month's first day to the last month's last; a coin
with no bar on a day is NaN there. ``funding_bp`` is the day's summed funding
settlements in bp, paid by a long and received by a short.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from trading_research.data import binance_monthly


@dataclass
class Panel:
    days: pd.DatetimeIndex  # UTC midnights
    symbols: tuple[str, ...]
    open: np.ndarray  # (D, C)
    close: np.ndarray
    quote_volume: np.ndarray
    funding_bp: np.ndarray  # NaN where the coin has no bar that day


def build(root: Path, symbols: Sequence[str], months: Sequence[str]) -> Panel:
    first = pd.Period(months[0], "M").start_time
    last = pd.Period(months[-1], "M").end_time.normalize()
    days = pd.date_range(first, last, freq="1D", tz="UTC")
    cols: dict[str, list[np.ndarray]] = {
        "open": [],
        "close": [],
        "quote_volume": [],
        "funding_bp": [],
    }
    kept = []
    for s in symbols:
        k = binance_monthly.load(root, "klines", s, months)
        if k.empty:
            continue
        k = k.drop_duplicates("open_time").set_index("open_time").reindex(days)
        f = binance_monthly.load(root, "fundingRate", s, months)
        if f.empty:
            fund = pd.Series(0.0, index=days)
        else:
            t = pd.DatetimeIndex(f["time"]).floor("1D")
            fund = f.groupby(t)["rate"].sum().reindex(days, fill_value=0.0) * 1e4
        kept.append(s)
        for c in ("open", "close", "quote_volume"):
            cols[c].append(k[c].to_numpy(dtype="float64"))
        cols["funding_bp"].append(np.where(k["close"].notna(), fund.to_numpy(), np.nan))
    return Panel(days, tuple(kept), *(np.column_stack(cols[c]) for c in cols))


def save(p: Panel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, days=p.days.to_numpy(dtype="datetime64[ns]").astype("int64"),
                        symbols=np.array(p.symbols), open=p.open, close=p.close,
                        quote_volume=p.quote_volume, funding_bp=p.funding_bp)  # fmt: skip


def read(path: Path) -> Panel:
    z = np.load(path)
    return Panel(pd.DatetimeIndex(pd.to_datetime(z["days"], utc=True)),
                 tuple(str(s) for s in z["symbols"]), z["open"], z["close"],
                 z["quote_volume"], z["funding_bp"])  # fmt: skip
