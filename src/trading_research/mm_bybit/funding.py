"""Funding settlements of BTCUSDT from Bybit's public history.

Fetched once per block span with ``data/bybit_funding.fetch`` (reused; category
``linear``) and kept as one Parquet file per block. A day's settlements are the
rows stamped in it; every one must sit on the 8-hour grid from midnight
(00:00, 08:00, 16:00 UTC), or the study stops: a settlement off the grid needs
an amendment, not a guess. A day missing one of its three settlements is
excluded as ``no_funding``.
"""

from __future__ import annotations

import urllib.request
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from trading_research.data import bybit_funding
from trading_research.market_making.events import NS_PER_DAY, day_start_ns, to_ns
from trading_research.mm_bybit.access import Access, block_days

INTERVAL_NS = 8 * 3600 * 1_000_000_000
SETTLEMENTS_PER_DAY = 3


class FundingOffGrid(RuntimeError):
    """A settlement stamped off the registered 8-hour grid."""


def path(root: Path | str, block: str) -> Path:
    return Path(root) / "funding" / f"BTCUSDT_{block}.parquet"


def fetch_block(
    block: str,
    root: Path | str,
    access: Access,
    *,
    opener: bybit_funding.Opener = urllib.request.urlopen,
    days: Sequence[date] | None = None,
) -> pd.DataFrame:
    """Every settlement of the block's days (or of ``days``, kept under ``block``'s name)."""
    days = list(days) if days is not None else block_days(block)
    for day in days:
        access.require(day)
    frame = bybit_funding.fetch("BTCUSDT", days[0], days[-1], opener=opener)
    check_grid(frame)
    target = path(root, block)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False)
    return frame


def check_grid(frame: pd.DataFrame) -> None:
    stamps = to_ns(frame["timestamp"]) if len(frame) else np.zeros(0, dtype=np.int64)
    if np.any(stamps % INTERVAL_NS):
        raise FundingOffGrid("a funding settlement is off the 8-hour grid")


def day_settlements(frame: pd.DataFrame, day: date) -> tuple[np.ndarray, np.ndarray] | None:
    """The day's settlement stamps (ns) and rates, or None if one is missing."""
    stamps = to_ns(frame["timestamp"]) if len(frame) else np.zeros(0, dtype=np.int64)
    start = day_start_ns(day)
    inside = (stamps >= start) & (stamps < start + NS_PER_DAY)
    if int(inside.sum()) != SETTLEMENTS_PER_DAY:
        return None
    rates = frame["rate"].to_numpy(dtype=np.float64)[inside]
    order = np.argsort(stamps[inside], kind="stable")
    return stamps[inside][order], rates[order]


def load_block(root: Path | str, block: str) -> pd.DataFrame:
    return pd.read_parquet(path(root, block))
