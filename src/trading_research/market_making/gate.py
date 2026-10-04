"""The second round's room gate, its hour mask, its placebo, and ``G(base)``.

The pre-registration (``docs/preregistration/market_making_round2.md``, "The
room gate") lets a quoter quote only while the recent market says a passive
fill is worth more than the fee it pays. Every print is scored from its resting
side, as :func:`.analysis.market_wide_markouts` scores it::

    markout_5s = side * (mid(tau + 5 s) - p) / p * 1e4

and at a decision at ``t`` the room is the volume-weighted mean of that markout
over the day's prints with ``t - W <= tau`` and ``tau + 5 s <= t``, less the
maker fee. It is defined only from ``W`` after midnight and with at least 20
such prints; otherwise the gate is closed.

How it reaches the simulator
----------------------------
A quoter sees no prints, so the gate is computed outside the simulator, from
the day's prints and book (:func:`print_markouts`), as a step function of time
(:func:`room_steps`), and handed in as a :class:`~.signals.SignalTape` of the
gate's state, open (1.0) or closed (0.0), named by the cell
(:func:`state_tape`). A tape value stamped ``s`` is read by decisions strictly
after ``s``; the steps are stamped one nanosecond before the time from which
they hold, so a decision at ``t`` reads exactly the state at ``t``:

* a print enters the window at ``tau + 5 s`` (stamp ``tau + 5 s - 1 ns``), when
  the snapshot that scores it has been seen;
* it leaves after ``tau + W`` (stamp ``tau + W``);
* the room is defined from ``00:00 + W`` (stamp one nanosecond earlier);
* an hour of the mask opens at its first nanosecond.

Each day's tape starts closed one nanosecond before midnight and ends before
the next midnight, so no state crosses a day.

``G(base)`` (:class:`GatedQuoter`) quotes as its base while the tape says open;
while closed it keeps only the side that reduces ``|position|``, at the base's
price and size, and quotes nothing when flat. The shifted-gate placebo moves
the whole block's state tape around the block (:func:`shift_tape`).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from trading_research.market_making import analysis
from trading_research.market_making.events import NS_PER_DAY, NS_PER_S, DayEvents
from trading_research.market_making.quoters import (
    NO_QUOTE,
    NO_QUOTES,
    MarketView,
    Quotes,
    SkewQuoter,
)
from trading_research.market_making.signals import SignalTape

#: The registered horizon, prints a window needs, and the searched values.
HORIZON_S = 5.0
MIN_PRINTS = 20
BASES = ("S1", "S1_touch")
KINDS = ("trailing", "both", "hours")
WINDOWS_MIN = (15, 60, 240)
MARGINS_BP = (0.0, 1.0, 2.0)

#: Where the state tape's names start; a cell's tape is ``gate_open|<label>``.
GATE_SIGNAL = "gate_open"

NS_PER_MIN = 60 * NS_PER_S
NS_PER_HOUR = 3600 * NS_PER_S


@dataclass(frozen=True)
class PrintMarkouts:
    """One day's prints scored from their resting side at the gate's horizon.

    Only prints with a markout (a snapshot at or before ``tau + horizon``) are
    kept; ``ts`` is in time order.
    """

    day_start_ns: int
    ts: np.ndarray
    markout_bp: np.ndarray
    size: np.ndarray
    horizon_s: float = HORIZON_S

    def __post_init__(self) -> None:
        if not len(self.ts) == len(self.markout_bp) == len(self.size):
            raise ValueError("one markout and one size per print")
        if len(self.ts) > 1 and np.any(np.diff(self.ts) < 0):
            raise ValueError("prints must be in time order")

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({"ts": self.ts, "markout_bp": self.markout_bp, "size": self.size})

    @classmethod
    def from_frame(
        cls, frame: pd.DataFrame, day_start_ns: int, horizon_s: float = HORIZON_S
    ) -> PrintMarkouts:
        return cls(
            int(day_start_ns),
            frame["ts"].to_numpy(dtype=np.int64),
            frame["markout_bp"].to_numpy(dtype=np.float64),
            frame["size"].to_numpy(dtype=np.float64),
            horizon_s,
        )


def print_markouts(events: DayEvents, horizon_s: float = HORIZON_S) -> PrintMarkouts:
    """Every print of the day, scored from its resting side, as the benchmark is."""
    mid = analysis.book_mid(events)
    ts = np.asarray(events.trade_ts, dtype=np.int64)
    price = events.trade_px * events.spec.tick
    resting = -events.trade_aggressor.astype(np.float64)
    later = analysis.mid_at(events.book_ts, mid, ts + round(horizon_s * NS_PER_S))
    value = resting * (later - price) / price * 1e4
    ok = np.isfinite(value)
    order = np.argsort(ts[ok], kind="stable")
    return PrintMarkouts(
        events.day_start_ns,
        ts[ok][order],
        value[ok][order],
        np.asarray(events.trade_sz, dtype=np.float64)[ok][order],
        horizon_s,
    )


# ---------------------------------------------------------------------------
# The trailing room and the hour mask
# ---------------------------------------------------------------------------


def room_steps(
    prints: PrintMarkouts, window_min: float, *, min_prints: int = MIN_PRINTS
) -> tuple[np.ndarray, np.ndarray]:
    """The trailing room before the fee, as a step function of time.

    Returns ``(stamps, values)``: ``values[k]`` is the volume-weighted markout
    over the window for every ``t`` in ``(stamps[k], stamps[k + 1]]``, NaN
    where the room is undefined (before ``W`` after midnight, or fewer than
    ``min_prints`` prints in the window). The first stamp is one nanosecond
    before midnight; none is at or after one nanosecond before the next.
    """
    start = prints.day_start_ns
    end = start + NS_PER_DAY
    window = round(window_min * NS_PER_MIN)
    tau = prints.ts
    adds = tau + round(prints.horizon_s * NS_PER_S) - 1
    removes = tau + window
    defined = start + window - 1
    stamps = np.unique(np.concatenate([[start - 1, defined], adds, removes]))
    stamps = stamps[(stamps >= start - 1) & (stamps < end - 1)]
    # Just after stamp s, the window holds the prints with close - 1 <= s and
    # tau + W > s: a contiguous run, because both bounds rise with tau.
    upper = np.searchsorted(adds, stamps, side="right")
    lower = np.searchsorted(removes, stamps, side="right")
    weights = np.r_[0.0, np.cumsum(prints.size)]
    weighted = np.r_[0.0, np.cumsum(prints.size * prints.markout_bp)]
    volume = weights[upper] - weights[lower]
    count = upper - lower
    ok = (count >= min_prints) & (volume > 0) & (stamps >= defined)
    values = np.full(len(stamps), np.nan)
    values[ok] = (weighted[upper[ok]] - weighted[lower[ok]]) / volume[ok]
    return stamps, values


def room_at(prints: PrintMarkouts, window_min: float, at_ns: np.ndarray) -> np.ndarray:
    """The room before the fee at each time, read from :func:`room_steps`."""
    stamps, values = room_steps(prints, window_min)
    index = np.searchsorted(stamps, np.asarray(at_ns, dtype=np.int64), side="left") - 1
    out = np.full(len(index), np.nan)
    known = index >= 0
    out[known] = values[index[known]]
    return out


def hourly_markouts(days: Iterable[PrintMarkouts]) -> pd.DataFrame:
    """Per UTC hour, the volume and volume-weighted markout of every print."""
    volume = np.zeros(24)
    weighted = np.zeros(24)
    for day in days:
        hour = ((day.ts - day.day_start_ns) // NS_PER_HOUR).astype(np.int64)
        inside = (hour >= 0) & (hour < 24)
        volume += np.bincount(hour[inside], weights=day.size[inside], minlength=24)
        weighted += np.bincount(
            hour[inside], weights=(day.size * day.markout_bp)[inside], minlength=24
        )
    with np.errstate(invalid="ignore", divide="ignore"):
        markout = np.where(volume > 0, weighted / volume, np.nan)
    return pd.DataFrame({"hour": np.arange(24), "volume": volume, "markout_bp": markout})


def hour_mask(hourly: pd.DataFrame, maker_bp: float, margin_bp: float) -> tuple[int, ...]:
    """The hours whose markout less the maker fee is at least the margin."""
    room = hourly["markout_bp"].to_numpy(dtype=np.float64) - maker_bp
    hours = hourly["hour"].to_numpy(dtype=np.int64)
    return tuple(int(h) for h in hours[np.isfinite(room) & (room >= margin_bp)])


# ---------------------------------------------------------------------------
# The cells
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateCell:
    """One registered gate cell: its base, kind, window (None for ``hours``)
    and margin."""

    base: str
    kind: str
    window_min: int | None
    margin_bp: float

    def __post_init__(self) -> None:
        if self.base not in BASES or self.kind not in KINDS:
            raise ValueError(f"no gate cell with base {self.base!r} and kind {self.kind!r}")
        if (self.kind == "hours") != (self.window_min is None):
            raise ValueError("a window belongs to the trailing and both kinds, and only to them")
        if self.window_min is not None and self.window_min not in WINDOWS_MIN:
            raise ValueError(f"window {self.window_min} is not registered")
        if float(self.margin_bp) not in MARGINS_BP:
            raise ValueError(f"margin {self.margin_bp} is not registered")

    @property
    def label(self) -> str:
        window = "-" if self.window_min is None else f"{self.window_min}"
        return f"{self.base}/{self.kind}/W={window}/mu={float(self.margin_bp):g}"

    @property
    def signal(self) -> str:
        return f"{GATE_SIGNAL}|{self.label}"

    def record(self) -> dict[str, Any]:
        """As the amendment freezes it."""
        margin = float(self.margin_bp)
        return {
            "base": self.base,
            "kind": self.kind,
            "window_min": self.window_min,
            "margin_bp": int(margin) if margin.is_integer() else margin,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> GateCell:
        window = record.get("window_min")
        return cls(
            str(record["base"]),
            str(record["kind"]),
            None if window is None else int(window),
            float(record["margin_bp"]),
        )


def gate_cells() -> list[GateCell]:
    """The 42 registered cells: each base, the trailing and both kinds over
    every window and margin, and the hours kind over every margin."""
    cells = []
    for base in BASES:
        for kind in ("trailing", "both"):
            for window in WINDOWS_MIN:
                for margin in MARGINS_BP:
                    cells.append(GateCell(base, kind, window, margin))
        for margin in MARGINS_BP:
            cells.append(GateCell(base, "hours", None, margin))
    return cells


def cell_neighbourhood(cell: GateCell, cells: Sequence[GateCell]) -> list[GateCell]:
    """The registered neighbourhood: the cells of the same base and kind within
    one step on the window and on the margin (on the margin alone for hours),
    the cell itself included, among ``cells``."""

    def step(levels: Sequence[float], a: float, b: float) -> bool:
        return abs(list(levels).index(a) - list(levels).index(b)) <= 1

    out = []
    for other in cells:
        if other.base != cell.base or other.kind != cell.kind:
            continue
        if not step(MARGINS_BP, float(other.margin_bp), float(cell.margin_bp)):
            continue
        if (
            cell.window_min is not None
            and other.window_min is not None
            and not step(WINDOWS_MIN, other.window_min, cell.window_min)
        ):
            continue
        out.append(other)
    return out


# ---------------------------------------------------------------------------
# The state tape, its placebo, and what it says about the tape
# ---------------------------------------------------------------------------


def _hour_steps(start: int, mask: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    stamps = start - 1 + np.arange(24, dtype=np.int64) * NS_PER_HOUR
    values = np.isin(np.arange(24), np.asarray(list(mask), dtype=np.int64)).astype(np.float64)
    return stamps, values


def _value_at(stamps: np.ndarray, values: np.ndarray, at: np.ndarray) -> np.ndarray:
    """The step function's value just after each of ``at`` (stamps included)."""
    index = np.searchsorted(stamps, at, side="right") - 1
    out = np.full(len(at), np.nan)
    known = index >= 0
    out[known] = values[index[known]]
    return out


def state_steps(
    prints: PrintMarkouts, cell: GateCell, maker_bp: float, mask: Sequence[int] = ()
) -> tuple[np.ndarray, np.ndarray]:
    """One day of the gate's state for ``cell`` at ``maker_bp``: 1.0 open,
    0.0 closed, as steps that hold just after their stamps."""
    start = prints.day_start_ns
    hours_s, hours_v = _hour_steps(start, mask)
    if cell.kind == "hours":
        stamps, open_ = hours_s, hours_v
    else:
        assert cell.window_min is not None
        room_s, room_v = room_steps(prints, cell.window_min)
        stamps = room_s if cell.kind == "trailing" else np.union1d(room_s, hours_s)
        room = _value_at(room_s, room_v, stamps) - maker_bp
        open_ = (np.isfinite(room) & (room >= float(cell.margin_bp))).astype(np.float64)
        if cell.kind == "both":
            open_ *= _value_at(hours_s, hours_v, stamps)
    return compress(stamps, open_)


def compress(stamps: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop every step that repeats the value before it."""
    if not len(stamps):
        return stamps, values
    keep = np.r_[True, values[1:] != values[:-1]]
    return stamps[keep], values[keep]


def state_tape(
    days: Sequence[PrintMarkouts], cell: GateCell, maker_bp: float, mask: Sequence[int] = ()
) -> SignalTape:
    """The gate's state over several days, each day starting closed, as the
    tape :class:`GatedQuoter` reads."""
    parts = [state_steps(day, cell, maker_bp, mask) for day in sorted(days, key=_start)]
    stamps = np.concatenate([p[0] for p in parts]) if parts else np.zeros(0, dtype=np.int64)
    values = np.concatenate([p[1] for p in parts]) if parts else np.zeros(0)
    return SignalTape(cell.signal, stamps.astype(np.int64), values)


def _start(day: PrintMarkouts) -> int:
    return day.day_start_ns


def placebo_offset(seed: int, span_ns: int, *, margin_h: float = 6.0) -> int:
    """The shifted-gate placebo's offset: uniform in ``[6 h, span - 6 h]``,
    in whole nanoseconds, from ``seed``. The same for every instrument."""
    margin = round(margin_h * NS_PER_HOUR)
    if span_ns <= 2 * margin:
        raise ValueError("the block is too short for the placebo's margin")
    rng = np.random.default_rng(seed)
    return int(rng.integers(margin, span_ns - margin + 1))


def shift_tape(tape: SignalTape, offset_ns: int, span: tuple[int, int]) -> SignalTape:
    """The state moved ``offset_ns`` later around the block ``[start, end)``:
    the state at ``t`` is the one that stood at ``start + (t - start - offset)
    mod (end - start)``. The share of the block spent open is unchanged. A tape
    with no state at the block's first nanosecond is closed there, as every
    day starts."""
    first, end = span
    length = end - first
    starts = tape.ts + 1  # a step holds from one nanosecond after its stamp
    inside = (starts >= first) & (starts < end)
    starts, values = starts[inside], tape.values[inside]  # noqa: PD011
    if not len(starts) or starts[0] != first:
        # Every day starts closed, so a block whose first day has no tape does too.
        starts, values = np.r_[first, starts], np.r_[0.0, values]
    shift = int(offset_ns) % length
    moved = first + (starts - first + shift) % length
    origin = first + (length - shift) % length
    standing = values[np.searchsorted(starts, origin, side="right") - 1]
    order = np.argsort(np.r_[first, moved], kind="stable")
    new_starts = np.r_[first, moved][order]
    new_values = np.r_[standing, values][order]
    last = np.r_[new_starts[1:] != new_starts[:-1], True]
    stamps, kept = compress(new_starts[last] - 1, new_values[last])
    return SignalTape(tape.name, stamps.astype(np.int64), kept)


def time_open(tape: SignalTape, start_ns: int, end_ns: int) -> float:
    """Nanoseconds of ``[start_ns, end_ns)`` the state tape spends open."""
    starts = tape.ts + 1
    ends = np.r_[starts[1:], np.iinfo(np.int64).max]
    low = np.maximum(starts, start_ns)
    high = np.minimum(ends, end_ns)
    span = np.clip(high - low, 0, None).astype(np.float64)
    return float(np.sum(span * (tape.values > 0.5)))  # noqa: PD011


def open_share(
    tape: SignalTape, day_starts: Sequence[int], *, from_s: float = 600.0, to_s: float = 86_280.0
) -> float:
    """The share of quoting time (00:10 to 23:58 by default) the gate is open."""
    total = 0.0
    opened = 0.0
    for start in day_starts:
        low = start + round(from_s * NS_PER_S)
        high = start + round(to_s * NS_PER_S)
        total += high - low
        opened += time_open(tape, low, high)
    return opened / total if total else math.nan


def room_persistence(
    prints: PrintMarkouts, tape: SignalTape, *, skip_first_min: float = 0.0
) -> dict[str, float]:
    """K4's ingredients for one day: the volume and volume-weighted markout of
    the prints that arrived while the gate was open, and while it was closed,
    leaving out the day's first ``skip_first_min`` minutes. A print arrives
    while open when the state at its time is open."""
    keep = prints.ts >= prints.day_start_ns + round(skip_first_min * NS_PER_MIN)
    ts, size, value = prints.ts[keep], prints.size[keep], prints.markout_bp[keep]
    state = tape.strictly_before(ts)
    opened = state > 0.5
    out: dict[str, float] = {}
    for name, mask in (("open", opened), ("closed", ~opened)):
        out[f"{name}_prints"] = float(mask.sum())
        out[f"{name}_volume"] = float(size[mask].sum())
        out[f"{name}_weighted"] = float((size * value)[mask].sum())
    return out


# ---------------------------------------------------------------------------
# G(base)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GatedQuoter:
    """``G(base)``: the base quoter while its gate is open.

    Reads the gate's state from the tape named ``signal``, as of strictly
    before the snapshot. While it is closed (or unknown) the base's quote is
    kept only on the side that reduces ``|position|``, at the base's price and
    size, so inventory can unwind passively; flat, nothing is quoted. The
    simulator then applies its rules as for every quoter.
    """

    base: SkewQuoter
    signal: str = GATE_SIGNAL
    name: str = "G"
    uses_future: bool = False

    def quotes(self, view: MarketView, position: float) -> Quotes:
        wanted = self.base.quotes(view, position)
        state = view.signals.get(self.signal, math.nan)
        if state == state and state > 0.5:
            return wanted
        if position > 0:
            return Quotes(NO_QUOTE, wanted.ask)
        if position < 0:
            return Quotes(wanted.bid, NO_QUOTE)
        return NO_QUOTES
