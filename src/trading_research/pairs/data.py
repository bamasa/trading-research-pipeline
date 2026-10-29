"""Bars for several instruments on one clock, or a clear refusal.

The input is the bars plane of the data contract (``BARS_SCHEMA``): one row per
closed bar, ``timestamp`` its close, ``open_time`` its open, as ``trp data
download --kind klines`` writes it under ``<root>/<SYMBOL>/klines-<interval>/``
and as ``trp pairs synthetic`` writes its generated pair.

Two legs are traded together only on bars both of them have. Rather than
inner-join and carry on, which would silently stitch a missing hour of one leg
to the next hour of the other, the loader refuses: every symbol must have the
same timestamps, every step between consecutive bars must be the interval, and
every price must be positive and finite.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from trading_research.pairs.stats import PairsError


class AlignmentError(PairsError):
    """The symbols' bars are not on one gap-free clock."""


def _one(symbol: str, frame: pd.DataFrame) -> pd.DataFrame:
    for column in ("timestamp", "open_time", "open", "close"):
        if column not in frame.columns:
            raise AlignmentError(f"{symbol}: bars have no {column!r} column")
    stamps = pd.to_datetime(frame["timestamp"], utc=True)
    if stamps.duplicated().any():
        first = stamps[stamps.duplicated()].iloc[0]
        raise AlignmentError(f"{symbol}: bar close {first} appears more than once")
    if not stamps.is_monotonic_increasing:
        raise AlignmentError(f"{symbol}: bars are not ordered by close")
    out = pd.DataFrame(
        {
            "open_time": pd.to_datetime(frame["open_time"], utc=True).to_numpy(),
            "open": frame["open"].to_numpy(dtype=np.float64),
            "close": frame["close"].to_numpy(dtype=np.float64),
        },
        index=pd.DatetimeIndex(stamps, name="timestamp"),
    )
    prices = out[["open", "close"]].to_numpy()
    if not (np.isfinite(prices).all() and (prices > 0).all()):
        bad = out.index[~(np.isfinite(prices).all(axis=1) & (prices > 0).all(axis=1))][0]
        raise AlignmentError(f"{symbol}: a missing or non-positive price at {bad}")
    return out


def align(frames: Mapping[str, pd.DataFrame], interval: str | pd.Timedelta) -> pd.DataFrame:
    """One frame of every symbol's ``open`` and ``close`` on a shared, gap-free clock.

    Columns are a two-level index ``(field, symbol)`` with fields ``open`` and
    ``close``, plus ``("open_time", "")``; the index is the bar close. Refuses
    a symbol whose closes differ from the first symbol's, and any step between
    consecutive bars other than ``interval``.
    """
    if len(frames) < 2:
        raise AlignmentError(f"a pair needs two symbols or more, got {len(frames)}")
    step = _interval(interval) if isinstance(interval, str) else pd.Timedelta(interval)
    parts = {symbol: _one(symbol, frame) for symbol, frame in frames.items()}
    names = list(parts)
    reference = parts[names[0]]
    if len(reference) < 2:
        raise AlignmentError(f"{names[0]}: fewer than two bars")
    for symbol in names[1:]:
        index = parts[symbol].index
        if not index.equals(reference.index):
            only_here = index.difference(reference.index)
            only_there = reference.index.difference(index)
            differing = sorted([*only_here[:1], *only_there[:1]])
            first = f", the first at {differing[0]}" if differing else ""
            raise AlignmentError(
                f"{symbol} and {names[0]} are not on the same bars: {len(only_here)} close(s) "
                f"only in {symbol}, {len(only_there)} only in {names[0]}{first}"
            )
        same_open = parts[symbol]["open_time"].to_numpy() == reference["open_time"].to_numpy()
        if not same_open.all():
            raise AlignmentError(f"{symbol} and {names[0]} disagree on a bar's open time")
    steps = np.diff(reference.index.to_numpy(dtype="datetime64[ns]").astype(np.int64))
    wrong = np.flatnonzero(steps != step.value)
    if len(wrong):
        k = int(wrong[0])
        raise AlignmentError(
            f"{len(wrong)} gap(s) or irregular step(s) in the bars; the first between "
            f"{reference.index[k]} and {reference.index[k + 1]}, where {step} was expected"
        )
    columns = {("open_time", ""): reference["open_time"]}
    for field in ("open", "close"):
        for symbol in names:
            columns[(field, symbol)] = parts[symbol][field]
    out = pd.DataFrame(columns, index=reference.index)
    out.columns = pd.MultiIndex.from_tuples(list(columns))
    return out


def load(
    root: Path | str,
    symbols: Sequence[str],
    interval: str,
    *,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """Read each symbol's bars from ``<root>/<SYMBOL>/klines-<interval>/`` and :func:`align` them."""
    from trading_research.data.ensure import DataUnavailable, load_bars

    step = _interval(interval)
    if len(set(symbols)) != len(symbols):
        raise AlignmentError(f"a symbol is named twice: {list(symbols)}")
    frames = {}
    for symbol in symbols:
        try:
            frames[symbol] = load_bars(symbol, interval, root=Path(root), start=start, end=end)
        except DataUnavailable as exc:
            raise AlignmentError(str(exc)) from exc
    return align(frames, step)


def _interval(interval: str) -> pd.Timedelta:
    """A Binance interval name (``1h``, ``15m``, ``1d``) as a duration."""
    units = {"m": "min", "h": "h", "d": "D", "w": "W"}
    if interval and interval[-1] in units and interval[:-1].isdigit():
        return pd.Timedelta(int(interval[:-1]), unit=units[interval[-1]])  # type: ignore[arg-type]
    raise AlignmentError(f"unknown bar interval {interval!r}; e.g. 1h, 15m, 1d")


def log_closes(bars: pd.DataFrame) -> pd.DataFrame:
    """Log close of every symbol, one column each."""
    return np.log(bars["close"])
