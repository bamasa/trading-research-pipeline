"""Regime flags for the guard (S3, H3): both detectors, effective when known.

H3 asks whether pulling or widening quotes after a regime-break flag earns more
than quoting through it. The flags come from the repository's two detectors,
with the settings the pre-registration fixed:

1. :func:`trading_research.validation.changepoint.detect` on the instrument's
   one-second grid (mid and spread in bp), with its defaults — window 2000,
   threshold 20, reference 40, minimum gap 20,000, all four statistics — run
   continuously from the first day given;
2. :func:`trading_research.validation.structural_breaks.detect_breaks` on
   one-minute log mid returns, ``MonitorSpec(history_len=250, online_len=60,
   statistics=("scale", "dependence"))`` at the recorded thresholds 7.77 and
   7.65.

When a flag is known
--------------------
Both grids are **right-labelled**: a row's label is the end of its bin, after
every observation it carries. A changepoint break at ``Break.index`` is the
exclusive end of the block of rows that fired it, so it is known at the label of
row ``index - 1``, when that block closed. A structural break at return
``index`` is known at that return's label, the close that produced it. Each is
effective that moment plus the decision latency (10 ms). A quoter reads the
flags through a :class:`~trading_research.market_making.signals.SignalTape` as
of strictly before each snapshot, like every other signal.

Both detectors are causal — each block, window and threshold reads only what
came before — so the flags found on a stretch of data are exactly those found
on any longer stretch that starts at the same place, up to the end of the
shorter one. ``tests/test_mm_flags.py`` pins this by truncation.

Reading days
------------
:func:`build_flags` reads the instrument's book one day at a time, only on the
days asked for, and only after the
:class:`~trading_research.market_making.prereg.Access` given permits every one
of them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from trading_research.market_making.events import NS_PER_S, day_start_ns, to_ns, within_day
from trading_research.market_making.quoters import FLAG_SIGNAL
from trading_research.market_making.signals import SignalTape

#: The registered changepoint settings: ``changepoint.detect``'s defaults.
CHANGEPOINT = {"window": 2000, "threshold": 20.0, "reference": 40, "minimum_gap": 20_000}
CHANGEPOINT_STATISTICS = ("volatility", "spread", "intensity", "autocorrelation")

#: The registered structural-break settings.
BREAKS_HISTORY, BREAKS_ONLINE = 250, 60
BREAKS_STATISTICS = ("scale", "dependence")
BREAKS_THRESHOLDS = {"scale": 7.77, "dependence": 7.65}

#: Decision latency added to every flag's effective time.
DECISION_LATENCY_NS = 10_000_000


@dataclass(frozen=True)
class FlagTimeline:
    """Every flag on one instrument over a span, in effective-time order.

    ``span`` is the half-open window ``[start, end)`` the detectors watched, in
    ns: the circle a shifted placebo wraps around.
    """

    symbol: str
    effective_ns: np.ndarray
    detector: np.ndarray
    statistic: np.ndarray
    span: tuple[int, int]

    def __post_init__(self) -> None:
        if not (len(self.effective_ns) == len(self.detector) == len(self.statistic)):
            raise ValueError("one detector and one statistic per flag")
        if len(self.effective_ns) > 1 and np.any(np.diff(self.effective_ns) < 0):
            raise ValueError("flags must be in effective-time order")

    def __len__(self) -> int:
        return len(self.effective_ns)

    def tape(self) -> SignalTape:
        """``flag_s``: from each flag's effective time, that time in epoch seconds.

        Flags effective at the same instant are one entry.
        """
        stamps = np.unique(self.effective_ns.astype(np.int64))
        return SignalTape(FLAG_SIGNAL, stamps, stamps.astype(np.float64) / NS_PER_S)

    def rate_per_day(self, days: int | None = None) -> float:
        """Flags per day over the span (or over ``days`` days)."""
        length = days if days is not None else (self.span[1] - self.span[0]) / (86_400 * NS_PER_S)
        return len(self) / length if length > 0 else float("nan")

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "symbol": self.symbol,
                "effective": pd.to_datetime(self.effective_ns, utc=True),
                "effective_ns": self.effective_ns,
                "detector": self.detector,
                "statistic": self.statistic,
            }
        )

    def shifted(self, offset_ns: int) -> FlagTimeline:
        """Every flag moved ``offset_ns`` later around the span's circle.

        The placebo of H3: the count and the spacing of the flags are kept (the
        gaps between consecutive flags, read around the circle, are the same
        multiset), only their position relative to the market moves.
        """
        start, end = self.span
        length = end - start
        moved = (self.effective_ns.astype(np.int64) - start + int(offset_ns)) % length + start
        order = np.argsort(moved, kind="stable")
        return FlagTimeline(
            self.symbol,
            moved[order],
            self.detector[order],
            self.statistic[order],
            self.span,
        )

    def placebo_offset(self, seed: int, *, margin_h: float = 6.0) -> int:
        """An offset uniform in ``[margin, span - margin]``, from ``seed``."""
        start, end = self.span
        margin = int(margin_h * 3600 * NS_PER_S)
        if end - start <= 2 * margin:
            raise ValueError("the span is too short for the placebo's margin")
        rng = np.random.default_rng(seed)
        return int(rng.integers(margin, end - start - margin + 1))


def flags_from_grids(
    symbol: str,
    one_second: pd.DataFrame,
    one_minute: pd.DataFrame,
    *,
    span: tuple[int, int],
    latency_ns: int = DECISION_LATENCY_NS,
) -> FlagTimeline:
    """Run both detectors on right-labelled grids and time their flags.

    ``one_second`` and ``one_minute`` carry ``timestamp`` (bin-end labels),
    ``bid_price_0`` and ``ask_price_0``, in time order.
    """
    from trading_research.validation import changepoint, structural_breaks

    stamps: list[int] = []
    detectors: list[str] = []
    statistics: list[str] = []

    labels = to_ns(one_second["timestamp"])
    bid = one_second["bid_price_0"].to_numpy(dtype=np.float64)
    ask = one_second["ask_price_0"].to_numpy(dtype=np.float64)
    mid = (bid + ask) / 2.0
    spread_bp = (ask - bid) / mid * 1e4
    window, reference = int(CHANGEPOINT["window"]), int(CHANGEPOINT["reference"])
    if len(mid) >= window * (reference + 2):
        for found in changepoint.detect(
            mid,
            spread_bp,
            window=window,
            threshold=float(CHANGEPOINT["threshold"]),
            reference=reference,
            minimum_gap=int(CHANGEPOINT["minimum_gap"]),
            statistics=CHANGEPOINT_STATISTICS,
        ):
            stamps.append(int(labels[found.index - 1]) + latency_ns)
            detectors.append("changepoint")
            statistics.append(found.statistic)

    minute_labels = pd.DatetimeIndex(one_minute["timestamp"])
    close = pd.Series(
        ((one_minute["bid_price_0"] + one_minute["ask_price_0"]) / 2.0).to_numpy(dtype=np.float64),
        index=minute_labels,
    )
    returns = structural_breaks.log_returns(close)
    if len(returns) > BREAKS_HISTORY:
        spec = structural_breaks.MonitorSpec(
            history_len=BREAKS_HISTORY,
            online_len=BREAKS_ONLINE,
            statistics=BREAKS_STATISTICS,
            threshold=dict(BREAKS_THRESHOLDS),
        )
        result = structural_breaks.detect_breaks(returns, spec, on_progress=None)
        return_labels = to_ns(pd.DatetimeIndex(returns.index))
        for found in result.breaks:
            stamps.append(int(return_labels[found.index]) + latency_ns)
            detectors.append("structural_breaks")
            statistics.append(found.statistic)

    order = np.argsort(np.array(stamps, dtype=np.int64), kind="stable")
    return FlagTimeline(
        symbol,
        np.array(stamps, dtype=np.int64)[order],
        np.array(detectors, dtype=object)[order],
        np.array(statistics, dtype=object)[order],
        span,
    )


def _book_file(symbol: str, day: date, roots: Sequence[Path]) -> Path | None:
    for root in roots:
        path = Path(root) / symbol / f"{day.isoformat()}.parquet"
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def read_grids(
    symbol: str, days: Sequence[date], *, book_roots: Sequence[Path], access: object
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The instrument's one-second and one-minute right-labelled grids on ``days``.

    One day of book is read at a time, and only after ``access`` permits every
    day asked for.
    """
    from trading_research.data.grid import to_grid
    from trading_research.market_making.prereg import Access

    if not isinstance(access, Access):
        raise TypeError("reading the book needs a prereg.Access")
    days = sorted(set(days))
    access.require(days, what="build flags from")
    columns = ["timestamp", "bid_price_0", "ask_price_0"]
    seconds: list[pd.DataFrame] = []
    minutes: list[pd.DataFrame] = []
    for day in days:
        path = _book_file(symbol, day, book_roots)
        if path is None:
            continue
        book = within_day(pd.read_parquet(path, columns=columns), day)
        seconds.append(to_grid(book, 1, label="right"))
        minutes.append(to_grid(book, 60, label="right"))
        del book
    if not seconds:
        raise FileNotFoundError(f"no book for {symbol} on the days asked")
    one_second = pd.concat(seconds, ignore_index=True)
    one_minute = pd.concat(minutes, ignore_index=True)
    for grid in (one_second, one_minute):
        if grid["timestamp"].duplicated().any():
            raise ValueError("a bin-end label appears twice across days")
    return one_second, one_minute


def build_flags(
    symbol: str,
    days: Sequence[date],
    *,
    book_roots: Sequence[Path],
    access: object,
    latency_ns: int = DECISION_LATENCY_NS,
) -> FlagTimeline:
    """Both detectors over ``days``, continuously from the first of them."""
    days = sorted(set(days))
    one_second, one_minute = read_grids(symbol, days, book_roots=book_roots, access=access)
    span = (day_start_ns(days[0]), day_start_ns(days[-1]) + 86_400 * NS_PER_S)
    return flags_from_grids(symbol, one_second, one_minute, span=span, latency_ns=latency_ns)


def guard_share(timeline: FlagTimeline, window_min: float) -> float:
    """Share of the span inside a guard window of ``window_min`` minutes."""
    start, end = timeline.span
    window = int(window_min * 60 * NS_PER_S)
    covered = 0
    reach = start
    for stamp in timeline.effective_ns.astype(np.int64):
        lo = max(int(stamp), reach)
        hi = min(int(stamp) + window, end)
        if hi > lo:
            covered += hi - lo
            reach = hi
    return covered / (end - start)


def flag_counts(timeline: FlagTimeline) -> Mapping[str, int]:
    """Flags per detector and statistic."""
    counts: dict[str, int] = {}
    for detector, statistic in zip(timeline.detector, timeline.statistic, strict=True):
        key = f"{detector}:{statistic}"
        counts[key] = counts.get(key, 0) + 1
    return counts
