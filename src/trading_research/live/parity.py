"""The parity guarantee: streaming features equal batch features on a replayed day.

For a stored day, each feature set is computed twice:

* **batch**, the way the research pipeline computes it over the whole day:
  :func:`trading_research.features.build` for the registry,
  :func:`trading_research.features.timed.build_timed` for the timed set, and,
  for the prepared set, the ``prepare`` stage itself
  (:func:`trading_research.pipeline.stages.prepare`) run on the day written in
  the staged layout, its output file read back;
* **streaming**, by replaying the day's events through
  :class:`~trading_research.live.features.StreamingFeatures`, which at each
  boundary holds only the declared window.

The report has one line per feature: the boundaries compared, the largest
absolute and relative deviation, and the boundaries where one side is missing
and the other is not. Rolling dispersions and their z-scores are listed but not held to the tolerance when
the day is longer than the declared look-back (:data:`ROUNDING_SENSITIVE`).
A boundary is every row for the registry and timed sets,
and every row ``prepare`` keeps (each ``subsample``-th row of a UTC day) for
the prepared set, which is where a model trained on prepared rows decides.
"""

from __future__ import annotations

import re
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from trading_research.features import build
from trading_research.features.timed import build_timed
from trading_research.live.events import BookRow, Print, prints_to_frame, rows_to_frame
from trading_research.live.features import (
    NS_PER_DAY,
    FeatureSet,
    PreparedSet,
    RegistrySet,
    StreamingFeatures,
    TimedSet,
    prints_as_trades,
)
from trading_research.live.replay import StoredDay
from trading_research.validation.leakage_timed import ATOL, RTOL

#: Rolling dispersions and the z-scores divided by them. Over a window where the input
#: barely moves (a spread fixed at one tick, its basis points moving only with the mid)
#: the dispersion is of the order of the input's rounding, and pandas' online rolling
#: variance carries rounding from every row that has left its window: a batch frame
#: started at midnight and a streaming frame started at its declared look-back then
#: differ at that level, and the z-score divides by it. On a day shorter than the
#: declared look-back the two frames coincide and these columns are held to the same
#: tolerance as the rest (the test suite checks both cases).
ROUNDING_SENSITIVE = re.compile(r"_(std|z)\d+$|^trade_disp_bp_")


@dataclass(frozen=True)
class ParityReport:
    """One row per feature, and whether every one is within tolerance."""

    table: pd.DataFrame
    rtol: float = RTOL
    atol: float = ATOL

    @property
    def ok(self) -> bool:
        """Every feature within tolerance, except rounding-sensitive ones (reported, not held)."""
        table = self.table
        return bool((table["within_tolerance"] | table["rounding_sensitive"]).all())


def _day_frames(day: StoredDay) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = day.rows
    book = rows_to_frame(rows, symbol=day.symbol or "INSTRUMENT", source=day.kind, depth=day.depth)
    prints = prints_to_frame(day.prints)
    order = np.argsort(prints["timestamp"].to_numpy(), kind="stable")
    return book, prints.iloc[order].reset_index(drop=True)


def batch_prepared(
    book: pd.DataFrame, prints: pd.DataFrame, symbol: str, *, subsample: int, workdir: Path
) -> pd.DataFrame:
    """The ``prepare`` stage's output for the day, run on the staged layout."""
    from trading_research.pipeline.stages import feature_columns, load_prepared, prepare

    raw = workdir / "raw"
    for directory in ("bookTicker", "aggTrades"):
        (raw / symbol / directory).mkdir(parents=True, exist_ok=True)
    days = book["timestamp"].dt.date
    trades = prints_as_trades(prints, symbol)
    trade_days = trades["timestamp"].dt.date
    for when in sorted(days.unique()):
        book[days == when].to_parquet(raw / symbol / "bookTicker" / f"{when}.parquet", index=False)
        mine = trades[trade_days == when]
        if len(mine):
            mine.to_parquet(raw / symbol / "aggTrades" / f"{when}.parquet", index=False)
    out = prepare(raw, workdir / "prepared", symbol, horizon=1, subsample=subsample)
    frame = load_prepared(out)
    return frame[["timestamp", *feature_columns(frame)]]


def _streaming(
    day: StoredDay, feature_set: FeatureSet, boundaries: set[int] | None, subsample: int
) -> pd.DataFrame:
    stream = StreamingFeatures([feature_set], symbol=day.symbol or "INSTRUMENT", depth=day.depth)
    values: list[pd.Series] = []
    stamps: list[int] = []
    per_day: dict[int, int] = {}
    for event in day.events:
        if isinstance(event, Print):
            stream.add_print(event)
        elif isinstance(event, BookRow):
            index = stream.add_row(event)
            when = event.ts_ns // NS_PER_DAY
            in_day = per_day.get(when, 0)
            per_day[when] = in_day + 1
            due = (boundaries is None and in_day % subsample == 0) or (
                boundaries is not None and index in boundaries
            )
            if due:
                values.append(stream.compute(feature_set, index))
                stamps.append(event.ts_ns)
    frame = pd.DataFrame(values).reset_index(drop=True)
    frame.insert(0, "timestamp", pd.to_datetime(np.asarray(stamps, dtype=np.int64), utc=True))
    return frame


def compare(batch: pd.DataFrame, live: pd.DataFrame, set_name: str) -> pd.DataFrame:
    """Per-feature deviations between two frames aligned on ``timestamp``."""
    merged = batch.merge(live, on="timestamp", suffixes=("_batch", "_live"), how="inner")
    rows = []
    for column in [c for c in live.columns if c != "timestamp" and c in batch.columns]:
        a = merged[f"{column}_batch"].to_numpy(dtype="float64")
        b = merged[f"{column}_live"].to_numpy(dtype="float64")
        both = np.isfinite(a) & np.isfinite(b)
        diff = np.abs(a[both] - b[both])
        scale = np.maximum(np.abs(a[both]), 1e-300)
        nan_mismatch = int((np.isnan(a) != np.isnan(b)).sum())
        close = np.isclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True)
        rows.append(
            {
                "set": set_name,
                "feature": column,
                "boundaries": len(merged),
                "compared": int(both.sum()),
                "max_abs_dev": float(diff.max()) if len(diff) else 0.0,
                "max_rel_dev": float((diff / scale).max()) if len(diff) else 0.0,
                "nan_mismatch": nan_mismatch,
                "within_tolerance": bool(close.all()),
                "rounding_sensitive": bool(ROUNDING_SENSITIVE.search(column)),
            }
        )
    return pd.DataFrame(rows)


def parity(
    day: StoredDay,
    sets: Sequence[str] = ("registry", "timed", "prepared"),
    *,
    subsample: int = 50,
    every: int = 1,
    workdir: Path | None = None,
) -> ParityReport:
    """Compare streaming and batch features of ``day`` for each named set.

    ``every`` thins the registry and timed boundaries (1: every row);
    ``subsample`` is ``prepare``'s, so the prepared set is compared at every row
    ``prepare`` keeps.
    """
    book, prints = _day_frames(day)
    n = len(book)
    chosen = set(range(0, n, every))
    parts = []
    for name in sets:
        if name == "registry":
            registry_set = RegistrySet()
            batch = build(book, list(registry_set.columns))
            batch = batch.iloc[sorted(chosen)]
            live = _streaming(day, registry_set, chosen, 1)
        elif name == "timed":
            timed_set = TimedSet()
            batch = build_timed(
                book, list(timed_set.columns), prints=prints, registry=timed_set.registry
            )
            batch = batch.iloc[sorted(chosen)]
            live = _streaming(day, timed_set, chosen, 1)
        elif name == "prepared":
            symbol = day.symbol or "INSTRUMENT"
            if workdir is None:
                with tempfile.TemporaryDirectory() as scratch:
                    batch = batch_prepared(
                        book, prints, symbol, subsample=subsample, workdir=Path(scratch)
                    )
            else:
                batch = batch_prepared(book, prints, symbol, subsample=subsample, workdir=workdir)
            prepared_set = PreparedSet(
                names=[c for c in batch.columns if c != "timestamp"], symbol=symbol
            )
            live = _streaming(day, prepared_set, None, subsample)
        else:
            raise ValueError(f"unknown feature set {name!r}; known: registry, timed, prepared")
        parts.append(compare(batch.reset_index(drop=True), live, name))
    return ParityReport(pd.concat(parts, ignore_index=True))
