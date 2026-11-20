"""One-minute bars of the recorded top of book, as arrays of time by instrument.

Each instrument-day's snapshots are put on a one-minute grid on their own,
labelled by the minute's end: the bar at 00:10 is the last snapshot at or
before 00:10:00. A minute with no snapshot repeats the previous one, at most
five times; a longer gap stays missing, and a day not on disk is missing
whole, never filled.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

#: Minutes a quiet book is carried forward before the bar counts as missing.
CARRY_MINUTES = 5

COLUMNS = ["timestamp", "bid_price_0", "bid_size_0", "ask_price_0", "ask_size_0"]


@dataclass
class Panel:
    """Bars of ``times`` by ``symbols``; NaN where an instrument has no bar."""

    times: pd.DatetimeIndex
    symbols: tuple[str, ...]
    mid: np.ndarray
    spread_bp: np.ndarray
    imbalance: np.ndarray

    def day_of(self) -> np.ndarray:
        """Each bar's calendar day, the day its minute starts in."""
        return np.asarray((self.times - np.timedelta64(1, "m")).date)


def span(first: date, last: date) -> list[date]:
    return [first + timedelta(days=k) for k in range((last - first).days + 1)]


def day_bars(book: pd.DataFrame, day: date) -> pd.DataFrame:
    """One instrument-day's snapshots as 1,440 bars labelled by the minute's end."""
    start = pd.Timestamp(day.isoformat(), tz="UTC")
    grid = pd.date_range(start + timedelta(minutes=1), periods=1440, freq="1min")
    book = book[(book["timestamp"] > start) & (book["timestamp"] <= grid[-1])]
    bars = (
        book.set_index("timestamp")
        .resample("1min", label="right", closed="right")
        .last()
        .reindex(grid)
        .ffill(limit=CARRY_MINUTES)
    )
    bid, ask = bars["bid_price_0"].to_numpy(), bars["ask_price_0"].to_numpy()
    bsz, asz = bars["bid_size_0"].to_numpy(), bars["ask_size_0"].to_numpy()
    ok = (bid > 0) & (ask >= bid)
    mid = np.where(ok, (bid + ask) / 2, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        return pd.DataFrame(
            {
                "mid": mid,
                "spread_bp": np.where(ok, (ask - bid) / mid * 1e4, np.nan),
                "imbalance": np.where(ok, (bsz - asz) / (bsz + asz), np.nan),
            },
            index=grid,
        )


def load(
    root: Path, symbols: Sequence[str], days: Sequence[date], *, minimum: int | None = None
) -> Panel:
    """The panel of ``symbols`` over ``days`` from ``root/<symbol>/<day>.parquet``.

    An instrument with no file for some day is dropped, and fewer than
    ``minimum`` instruments left raises: the panel is read whole or not at all.
    """
    kept = [s for s in symbols if all((root / s / f"{d}.parquet").exists() for d in days)]
    if len(kept) < (len(symbols) if minimum is None else minimum):
        missing = sorted(set(symbols) - set(kept))
        raise FileNotFoundError(f"{root}: no file for every day for {', '.join(missing)}")
    columns: dict[str, list[np.ndarray]] = {"mid": [], "spread_bp": [], "imbalance": []}
    times = None
    for symbol in kept:
        parts = []
        for day in days:
            book = pd.read_parquet(root / symbol / f"{day}.parquet", columns=COLUMNS)
            book["timestamp"] = pd.to_datetime(book["timestamp"], utc=True)
            parts.append(day_bars(book.sort_values("timestamp"), day))
        bars = pd.concat(parts)
        times = bars.index if times is None else times
        for key in columns:
            columns[key].append(bars[key].to_numpy(dtype="float64"))
    assert times is not None
    return Panel(
        times=pd.DatetimeIndex(times),
        symbols=tuple(kept),
        mid=np.column_stack(columns["mid"]),
        spread_bp=np.column_stack(columns["spread_bp"]),
        imbalance=np.column_stack(columns["imbalance"]),
    )


def save(panel: Panel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        times=panel.times.to_numpy(dtype="datetime64[ns]").astype("int64"),
        symbols=np.array(panel.symbols),
        mid=panel.mid,
        spread_bp=panel.spread_bp,
        imbalance=panel.imbalance,
    )


def read(path: Path) -> Panel:
    z = np.load(path)
    return Panel(
        times=pd.DatetimeIndex(pd.to_datetime(z["times"], utc=True)),
        symbols=tuple(str(s) for s in z["symbols"]),
        mid=z["mid"],
        spread_bp=z["spread_bp"],
        imbalance=z["imbalance"],
    )
