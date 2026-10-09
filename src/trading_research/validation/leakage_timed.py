"""Leakage checks for features declared in seconds on row stamps.

The checks of :mod:`trading_research.validation.leakage`, carried over to
:class:`~trading_research.features.timed.TimedFeature`:

**Truncation.** Cut the data and recompute; nothing before the cut may move.
For a book feature the cut is at a row. For a trade-plane feature it is at a
time strictly inside a row interval, on a print, so that the last kept row's
interval is split in two: a feature that gives row *t* the prints after its
stamp (the forward rule) changes there, and one that follows the backward rule
does not.

**Streaming parity.** At each decision stamp, a live process holds only what
the declaration promises: the rows stamped at or after ``s_t - lookback_s``,
the ``lookback_rows`` rows before them, and the prints those rows own. The
value it computes from that buffer must equal the batch value. This is the
declared-lookback check of the row registry, run at every position asked for,
and it fails a feature that reads further back than it says as well as one that
reads forward.

Online rolling statistics carry rounding from rows that have left the window,
so the comparison allows a relative error of ``1e-7`` by default; a real
dependence on history moves values by far more.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from trading_research.features.prints import print_stamps_ns
from trading_research.features.timed import (
    TimedFeature,
    TimedRegistry,
    seconds_ns,
    stamps_ns,
)
from trading_research.validation.leakage import LeakageError, LeakageResult

RTOL: float = 1e-7
ATOL: float = 1e-10


def _compare(
    name: str, full: np.ndarray, part: np.ndarray, rtol: float, atol: float, what: str
) -> LeakageResult:
    both_nan = np.isnan(full) & np.isnan(part)
    same = np.isclose(full, part, rtol=rtol, atol=atol, equal_nan=False) | both_nan
    if same.all():
        return LeakageResult(name, leaks=False, first_divergence=None, n_diverged=0, detail="")
    diverged = np.flatnonzero(~same)
    first = int(diverged[0])
    detail = (
        f"{len(diverged)} of {len(full)} values differ {what}; first at position {first}: "
        f"batch={full[first]!r} other={part[first]!r}"
    )
    return LeakageResult(
        name, leaks=True, first_divergence=first, n_diverged=len(diverged), detail=detail
    )


def _print_cut(stamps: np.ndarray, print_stamps: np.ndarray, start: int) -> tuple[int, int]:
    """The first row interval from ``start`` that a cut on a print can split.

    The interval must hold prints at two distinct stamps strictly inside it;
    the cut is at the first of them, so the later ones are dropped. Returns
    ``(k, c)``: rows ``[0, k)`` are kept, and the prints stamped at or before
    ``c``, where ``s_{k-1} < c < s_k``.
    """
    for k in range(max(start, 1), len(stamps)):
        lo = int(np.searchsorted(print_stamps, stamps[k - 1], side="right"))
        hi = int(np.searchsorted(print_stamps, stamps[k], side="left"))
        if hi > lo and print_stamps[hi - 1] > print_stamps[lo]:
            return k, int(print_stamps[lo])
    raise ValueError("no row interval after the cut holds prints at two distinct stamps")


def check_timed_causal(
    feature: TimedFeature,
    rows: pd.DataFrame,
    prints: pd.DataFrame | None = None,
    *,
    cut: float = 0.6,
    rtol: float = RTOL,
    atol: float = ATOL,
) -> LeakageResult:
    """Truncate the data at ``cut`` of the rows and confirm nothing earlier moved."""
    if len(rows) < 100:
        raise ValueError(f"need at least 100 rows to check {feature.name!r}, got {len(rows)}")
    stamps = stamps_ns(rows)
    k = int(len(rows) * cut)
    kept_prints = prints
    if feature.plane == "trades":
        if prints is None:
            raise ValueError(f"feature {feature.name!r} reads prints; none were given")
        print_stamps = print_stamps_ns(prints)
        k, c = _print_cut(stamps, print_stamps, k)
        kept_prints = prints.iloc[: int(np.searchsorted(print_stamps, c, side="right"))].copy()
    elif prints is not None:
        last = stamps[k - 1]
        kept_prints = prints.iloc[
            : int(np.searchsorted(print_stamps_ns(prints), last, side="right"))
        ].copy()
    full = feature(rows, prints).to_numpy()[:k]
    part = feature(rows.iloc[:k].copy(), kept_prints).to_numpy()
    return _compare(feature.name, full, part, rtol, atol, f"when the data after row {k} is cut")


def streaming_window(
    feature: TimedFeature,
    rows: pd.DataFrame,
    prints: pd.DataFrame | None,
    position: int,
    *,
    stamps: np.ndarray | None = None,
    print_stamps: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """The rows and prints a live process holds at row ``position``."""
    stamps = stamps_ns(rows) if stamps is None else stamps
    now = stamps[position]
    first = int(np.searchsorted(stamps, now - seconds_ns(feature.lookback_s), side="left"))
    start = max(first - feature.lookback_rows, 0)
    window = rows.iloc[start : position + 1].copy()
    if prints is None:
        return window, None
    print_stamps = print_stamps_ns(prints) if print_stamps is None else print_stamps
    lo = 0 if start == 0 else int(np.searchsorted(print_stamps, stamps[start - 1], side="right"))
    hi = int(np.searchsorted(print_stamps, now, side="right"))
    return window, prints.iloc[lo:hi].copy()


def streaming_values(
    feature: TimedFeature,
    rows: pd.DataFrame,
    prints: pd.DataFrame | None,
    positions: Iterable[int],
) -> np.ndarray:
    """The feature as a live process computes it at each of ``positions``."""
    stamps = stamps_ns(rows)
    print_stamps = None if prints is None else print_stamps_ns(prints)
    values = []
    for position in positions:
        window, held = streaming_window(
            feature, rows, prints, position, stamps=stamps, print_stamps=print_stamps
        )
        values.append(float(feature(window, held).to_numpy()[-1]))
    return np.asarray(values, dtype="float64")


def check_streaming_parity(
    feature: TimedFeature,
    rows: pd.DataFrame,
    prints: pd.DataFrame | None = None,
    positions: Sequence[int] | None = None,
    *,
    rtol: float = RTOL,
    atol: float = ATOL,
) -> LeakageResult:
    """Streaming values from the declared buffer equal the batch values."""
    chosen = list(range(len(rows))) if positions is None else list(positions)
    batch = feature(rows, prints).to_numpy()[chosen]
    live = streaming_values(feature, rows, prints, chosen)
    return _compare(feature.name, batch, live, rtol, atol, "between batch and streaming")


def assert_timed_honest(
    registry: TimedRegistry,
    rows: pd.DataFrame,
    prints: pd.DataFrame | None = None,
    *,
    names: Sequence[str] | None = None,
    positions: Sequence[int] | None = None,
    cut: float = 0.6,
) -> None:
    """Raise unless every named feature passes truncation and streaming parity."""
    failed = []
    for name in registry.names() if names is None else names:
        feature = registry.get(name)
        for result in (
            check_timed_causal(feature, rows, prints, cut=cut),
            check_streaming_parity(feature, rows, prints, positions),
        ):
            if result.leaks:
                failed.append(result)
    if failed:
        detail = "\n".join(f"  {r}" for r in failed)
        raise LeakageError(f"{len(failed)} check(s) failed:\n{detail}")
