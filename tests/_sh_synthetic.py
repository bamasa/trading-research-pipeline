"""Small synthetic event-time books and prints for the short-horizon study's tests.

Rows follow ``data/bybit.reconstruct``'s shape: at most one row per 100 ms
interval, stamped inside it, with intervals that have no row and one longer
quiet stretch. Prints fall between rows, often several on one stamp, and carry
the Bybit contract (``timestamp``, ``price``, ``size``, ``aggressor``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TICK = 0.1
GRID_NS = 100_000_000


def make_rows(
    n: int = 600,
    *,
    depth: int = 10,
    seed: int = 0,
    start: str = "2025-01-01",
    quiet_at: int | None = 300,
    quiet_s: float = 7.0,
) -> pd.DataFrame:
    """``n`` book rows with ``depth`` levels a side on a 0.1 tick."""
    rng = np.random.default_rng(seed)
    buckets = np.cumsum(1 + (rng.random(n) < 0.25).astype(np.int64))
    if quiet_at is not None and quiet_at < n:
        buckets[quiet_at:] += int(quiet_s * 1e9 / GRID_NS)
    offsets = rng.integers(0, GRID_NS, n)
    origin = pd.Timestamp(start, tz="UTC").value
    stamps = origin + buckets * GRID_NS + offsets

    moves = rng.choice([-1, 0, 0, 0, 1], size=n)
    bid0 = 500_000 + np.cumsum(moves)
    spread = 1 + (rng.random(n) < 0.15).astype(np.int64) + (rng.random(n) < 0.05)
    frame: dict[str, object] = {"timestamp": pd.to_datetime(stamps, unit="ns", utc=True)}
    bid_ticks = bid0.copy()
    ask_ticks = bid0 + spread
    for level in range(depth):
        if level > 0:
            bid_ticks = bid_ticks - 1 - (rng.random(n) < 0.3)
            ask_ticks = ask_ticks + 1 + (rng.random(n) < 0.3)
        frame[f"bid_price_{level}"] = np.round(bid_ticks * TICK, 1)
        frame[f"bid_size_{level}"] = np.round(rng.exponential(0.4, n) + 0.001, 3)
        frame[f"ask_price_{level}"] = np.round(ask_ticks * TICK, 1)
        frame[f"ask_size_{level}"] = np.round(rng.exponential(0.4, n) + 0.001, 3)
    return pd.DataFrame(frame)


def make_prints(rows: pd.DataFrame, *, seed: int = 1, rate: float = 1.5) -> pd.DataFrame:
    """Prints between consecutive rows, at the touch the earlier row shows."""
    rng = np.random.default_rng(seed)
    stamps = pd.DatetimeIndex(rows["timestamp"]).as_unit("ns").to_numpy("datetime64[ns]")
    stamps = stamps.view(np.int64)
    parts = []
    for j in range(1, len(rows)):
        count = rng.poisson(rate)
        if count == 0:
            continue
        lo, hi = stamps[j - 1], stamps[j]
        times = np.sort(rng.integers(lo + 1, hi + 1, count))
        if count > 1 and rng.random() < 0.4:
            times[1] = times[0]
        side = rng.choice([-1, 1], size=count)
        price = np.where(
            side > 0, rows["ask_price_0"].to_numpy()[j - 1], rows["bid_price_0"].to_numpy()[j - 1]
        )
        size = np.round(rng.exponential(0.01, count) + 0.001, 3)
        parts.append(pd.DataFrame({"ts": times, "price": price, "size": size, "aggressor": side}))
    out = pd.concat(parts, ignore_index=True).sort_values("ts", kind="stable")
    out.insert(0, "timestamp", pd.to_datetime(out.pop("ts"), unit="ns", utc=True))
    return out.reset_index(drop=True)
