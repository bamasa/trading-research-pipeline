"""The registered features, computed on a stream from a window sized by their declarations.

Nothing here re-implements a feature. At each boundary the loop holds a
window of the latest book rows and prints, builds a frame in the book and
print contracts, and calls the **batch** function on it; the value at the
window's last row is the streaming value. The window is what each feature set
declares it reads, so the value equals the batch value over the whole day:

* **registry** (:data:`trading_research.features.REGISTRY`, book plane):
  each feature declares ``lookback`` rows, so the window is the row and the
  longest ``lookback`` before it, computed by
  :func:`trading_research.features.build`. Exact.
* **timed** (the 2025 additions,
  :data:`trading_research.features.micro2025.REGISTRY_2025`): the rows stamped
  at or after ``s_t - lookback_s``, the ``lookback_rows`` before them and the
  prints those rows own, the buffer
  :func:`trading_research.validation.leakage_timed.streaming_window` defines,
  computed by :func:`trading_research.features.timed.build_timed`. Exact.
* **prepared** (what the ``prepare`` stage writes and ``train`` fits on: the
  generated set of :mod:`trading_research.features.generated`, the trade
  features of :mod:`trading_research.features.align` and the registry
  features the generated set lacks), computed by the same calls ``prepare``
  makes. Its rolling windows reach 1,000 rows and its trade windows 300 s
  with a 60 s as-of tolerance, but its exponentially weighted means remember
  every row since the frame began. They are given a declared look-back here:
  the rows after which an earlier start moves the value by less than
  ``ewm_tolerance`` of the starting difference (13,810 rows at span 1,000 and
  1e-12). ``prepare`` also starts each day's frame 1,200 rows before midnight
  and reads only that day's prints, and the window keeps both rules, so the
  streaming frame starts where the batch frame starts whenever the batch frame
  is shorter than the declared look-back.

Online rolling statistics carry rounding from rows that have left the window,
so the parity tolerance is that of the timed-leakage checks
(:data:`~trading_research.validation.leakage_timed.RTOL`, ``ATOL``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from trading_research.data.schema import level_columns
from trading_research.features import REGISTRY, build
from trading_research.features.micro2025 import REGISTRY_2025
from trading_research.features.timed import TimedRegistry, build_timed, seconds_ns
from trading_research.live.events import DEPTH, BookRow, Print

NS_PER_DAY = 86_400 * 1_000_000_000

#: The generated set's longest exponentially weighted span and its longest window.
GENERATED_MAX_SPAN = 1000
GENERATED_MAX_WINDOW = 1000
#: ``align.build_trade_features``: the longest trade window and the as-of tolerance.
TRADE_WINDOW_S = 300.0
TRADE_TOLERANCE_S = 60.0
#: ``align.build_trade_features``'s impact term reads the mid this many rows back.
IMPACT_ROWS = 300
#: ``prepare``'s warm-up from the previous day, in rows.
PREPARE_WARMUP = 1200


def ewm_rows(span: int, tolerance: float) -> int:
    """Rows after which an EWM's starting value weighs less than ``tolerance``."""
    alpha = 2.0 / (span + 1.0)
    return math.ceil(math.log(tolerance) / math.log(1.0 - alpha))


# ---------------------------------------------------------------------------
# Buffers
# ---------------------------------------------------------------------------


@dataclass
class RowBuffer:
    """Book rows as arrays, the oldest dropped as the window allows."""

    depth: int = DEPTH
    symbol: str = ""
    source: str = "live"
    ts: list[int] = field(default_factory=list)
    seq: list[int] = field(default_factory=list)
    levels: list[tuple[float, ...]] = field(default_factory=list)
    #: Index of ``ts[0]`` in the session's row count.
    offset: int = 0

    def append(self, row: BookRow) -> int:
        d = self.depth
        values: list[float] = []
        for level in range(d):
            values += [row.bid_px[level], row.bid_sz[level], row.ask_px[level], row.ask_sz[level]]
        self.ts.append(row.ts_ns)
        self.seq.append(row.sequence_id)
        self.levels.append(tuple(values))
        return self.offset + len(self.ts) - 1

    def __len__(self) -> int:
        return len(self.ts)

    def drop_before(self, index: int) -> None:
        cut = index - self.offset
        if cut > 0:
            del self.ts[:cut], self.seq[:cut], self.levels[:cut]
            self.offset += cut

    def stamps(self) -> np.ndarray:
        return np.asarray(self.ts, dtype=np.int64)

    def frame(self, start: int, stop: int) -> pd.DataFrame:
        """Rows ``[start, stop)`` (session indices) as a frame in the book contract."""
        lo, hi = start - self.offset, stop - self.offset
        n = hi - lo
        values = np.asarray(self.levels[lo:hi], dtype=np.float64).reshape(n, 4 * self.depth)
        data: dict[str, object] = {
            "timestamp": pd.to_datetime(np.asarray(self.ts[lo:hi], dtype=np.int64), utc=True),
            "symbol": pd.array([self.symbol] * n, dtype="string"),
            "sequence_id": np.asarray(self.seq[lo:hi], dtype=np.int64),
            "source": pd.array([self.source] * n, dtype="string"),
        }
        for k, name in enumerate(level_columns(self.depth)):
            data[name] = values[:, k]
        return pd.DataFrame(data)


@dataclass
class PrintBuffer:
    """Prints as arrays, in arrival order."""

    ts: list[int] = field(default_factory=list)
    price: list[float] = field(default_factory=list)
    size: list[float] = field(default_factory=list)
    aggressor: list[int] = field(default_factory=list)

    def append(self, p: Print) -> None:
        self.ts.append(p.ts_ns)
        self.price.append(p.price)
        self.size.append(p.size)
        self.aggressor.append(p.aggressor)

    def drop_before_ns(self, stamp: int) -> None:
        keep = [i for i, t in enumerate(self.ts) if t >= stamp]
        if len(keep) < len(self.ts):
            self.ts = [self.ts[i] for i in keep]
            self.price = [self.price[i] for i in keep]
            self.size = [self.size[i] for i in keep]
            self.aggressor = [self.aggressor[i] for i in keep]

    def frame(self, after_ns: int | None, until_ns: int) -> pd.DataFrame:
        """Prints stamped in ``(after_ns, until_ns]``, stably sorted by stamp."""
        ts = np.asarray(self.ts, dtype=np.int64)
        keep = ts <= until_ns
        if after_ns is not None:
            keep &= ts > after_ns
        index = np.flatnonzero(keep)
        order = index[np.argsort(ts[index], kind="stable")]
        return pd.DataFrame(
            {
                "timestamp": pd.to_datetime(ts[order], utc=True),
                "price": np.asarray(self.price, dtype=np.float64)[order],
                "size": np.asarray(self.size, dtype=np.float64)[order],
                "aggressor": np.asarray(self.aggressor, dtype=np.int64)[order],
            }
        )


def prints_as_trades(prints: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Prints as the trade plane ``prepare`` hands to the generated set."""
    return pd.DataFrame(
        {
            "timestamp": prints["timestamp"],
            "symbol": pd.array([symbol] * len(prints), dtype="string"),
            "price": prints["price"].astype("float64"),
            "quantity": prints["size"].astype("float64"),
            "is_buyer_maker": prints["aggressor"].to_numpy() < 0,
            "trade_id": np.arange(len(prints), dtype=np.int64),
            "source": pd.array(["live"] * len(prints), dtype="string"),
        }
    )


# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Window:
    """What a set reads before boundary row ``t``: the frame's first row and the prints."""

    start: int
    prints_after_ns: int | None


class FeatureSet:
    """A named group of columns, its declared window and its batch computation."""

    name: str = ""
    columns: tuple[str, ...] = ()
    #: Rows the buffer must keep behind the newest row.
    keep_rows: int = 0
    #: Prints the buffer must keep behind the oldest kept row's stamp, in seconds.
    keep_prints_s: float = 0.0

    def window(self, stamps: np.ndarray, offset: int, t: int) -> Window:
        raise NotImplementedError

    def compute(self, rows: pd.DataFrame, prints: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


@dataclass
class RegistrySet(FeatureSet):
    """The row registry's book features: the row and its declared ``lookback`` rows."""

    names: Sequence[str] = ()

    def __post_init__(self) -> None:
        self.name = "registry"
        self.columns = tuple(self.names or REGISTRY.names("book"))
        self.lookback = REGISTRY.max_lookback(self.columns)
        self.keep_rows = self.lookback + 1

    def window(self, stamps: np.ndarray, offset: int, t: int) -> Window:  # noqa: ARG002
        return Window(max(t - self.lookback, offset), None)

    def compute(self, rows: pd.DataFrame, prints: pd.DataFrame) -> pd.DataFrame:  # noqa: ARG002
        return build(rows, list(self.columns))[list(self.columns)]


@dataclass
class TimedSet(FeatureSet):
    """Timed features: rows since ``s_t - lookback_s``, ``lookback_rows`` before, their prints."""

    names: Sequence[str] = ()
    registry: TimedRegistry = field(default_factory=lambda: REGISTRY_2025)
    grid_ms: int = 100

    def __post_init__(self) -> None:
        self.name = "timed"
        self.columns = tuple(self.names or self.registry.names())
        self.lookback_s = self.registry.longest_lookback_s(self.columns)
        self.lookback_rows = self.registry.longest_lookback_rows(self.columns)
        self.keep_rows = self.registry.max_rows(self.columns, grid_ms=self.grid_ms) + 2
        self.keep_prints_s = 1.0

    def window(self, stamps: np.ndarray, offset: int, t: int) -> Window:
        local = t - offset
        now = stamps[local]
        first = int(np.searchsorted(stamps[: local + 1], now - seconds_ns(self.lookback_s)))
        start = max(first - self.lookback_rows, 0)
        if start == 0:
            if offset > 0:
                raise RuntimeError("the row buffer was trimmed below the timed window")
            return Window(0, None)
        # The prints the window's rows own: after the stamp of the row before them.
        return Window(start + offset, int(stamps[start - 1]))

    def compute(self, rows: pd.DataFrame, prints: pd.DataFrame) -> pd.DataFrame:
        return build_timed(rows, list(self.columns), prints=prints, registry=self.registry)[
            list(self.columns)
        ]


@dataclass
class PreparedSet(FeatureSet):
    """The ``prepare`` stage's frame: the generated set, its trade features, the registry rest."""

    names: Sequence[str] = ()
    symbol: str = ""
    ewm_tolerance: float = 1e-12
    warmup: int = PREPARE_WARMUP

    def __post_init__(self) -> None:
        self.name = "prepared"
        self.columns = tuple(self.names)
        self.lookback = max(
            ewm_rows(GENERATED_MAX_SPAN, self.ewm_tolerance),
            GENERATED_MAX_WINDOW + 1,
            IMPACT_ROWS,
            REGISTRY.max_lookback(REGISTRY.names("book")),
        )
        self.prints_s = TRADE_WINDOW_S + TRADE_TOLERANCE_S
        self.keep_rows = self.lookback + 1
        self.keep_prints_s = self.prints_s + 1.0
        #: Session index of the first row of each UTC day seen, by day number.
        self.day_first: dict[int, int] = {}

    def note_row(self, index: int, ts_ns: int) -> None:
        self.day_first.setdefault(ts_ns // NS_PER_DAY, index)

    def window(self, stamps: np.ndarray, offset: int, t: int) -> Window:
        now = int(stamps[t - offset])
        day = now // NS_PER_DAY
        first_of_day = self.day_first.get(day, offset)
        start = max(t - self.lookback, first_of_day - self.warmup, offset)
        # prepare reads the day's own prints only, from midnight.
        after = max(now - seconds_ns(self.prints_s), day * NS_PER_DAY - 1)
        return Window(start, after)

    def compute(self, rows: pd.DataFrame, prints: pd.DataFrame) -> pd.DataFrame:
        frame = prepared_frame(rows, prints_as_trades(prints, self.symbol) if len(prints) else None)
        if not self.columns:
            self.columns = tuple(frame.columns)
        return frame.reindex(columns=list(self.columns))


def prepared_frame(book: pd.DataFrame, trades: pd.DataFrame | None) -> pd.DataFrame:
    """The feature columns ``prepare`` computes on one frame, by the calls it makes."""
    from trading_research.features.generated import generate

    generated = generate(book, trades)
    registry_names = [n for n in REGISTRY.names("book") if n not in generated.columns]
    extra = build(book, registry_names)[registry_names] if registry_names else None
    return generated if extra is None else pd.concat([generated, extra], axis=1)


# ---------------------------------------------------------------------------
# The streaming computation
# ---------------------------------------------------------------------------


@dataclass
class StreamingFeatures:
    """Rows and prints in, each set's values at the newest row out."""

    sets: Sequence[FeatureSet]
    symbol: str = ""
    depth: int = DEPTH
    rows: RowBuffer = field(init=False)
    prints: PrintBuffer = field(default_factory=PrintBuffer)
    count: int = 0

    def __post_init__(self) -> None:
        self.rows = RowBuffer(depth=self.depth, symbol=self.symbol)
        self.keep_rows = max((s.keep_rows for s in self.sets), default=1)
        self.keep_prints_s = max((s.keep_prints_s for s in self.sets), default=0.0)

    def add_print(self, p: Print) -> None:
        self.prints.append(p)

    def add_row(self, row: BookRow) -> int:
        index = self.rows.append(row)
        self.count = index + 1
        for s in self.sets:
            if isinstance(s, PreparedSet):
                s.note_row(index, row.ts_ns)
        # Keep twice the declared rows, trimming in blocks so the work is amortised.
        if len(self.rows) > 2 * self.keep_rows + 1000:
            self.rows.drop_before(index - self.keep_rows - 1)
            oldest = self.rows.ts[0]
            self.prints.drop_before_ns(oldest - seconds_ns(self.keep_prints_s))
        return index

    def compute(self, feature_set: FeatureSet, t: int | None = None) -> pd.Series:
        """The set's values at session row ``t`` (the newest row by default)."""
        t = self.count - 1 if t is None else t
        stamps = self.rows.stamps()
        window = feature_set.window(stamps, self.rows.offset, t)
        rows = self.rows.frame(window.start, t + 1)
        prints = self.prints.frame(window.prints_after_ns, int(stamps[t - self.rows.offset]))
        values = feature_set.compute(rows, prints)
        return values.iloc[-1]
