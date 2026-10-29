"""The side-correct print fill rule and its brackets: one function for label and executor.

Binance publishes no spot book, so a taker order there is priced from prints
(the registration, "Fills"). The rule, for an order of side ``s`` (+1 buys,
-1 sells) arriving at ``a`` = decision + latency:

* it fills at the **first reachable print of its side stamped at or after**
  ``a`` (a buy at a buyer-initiated print, a sell at a seller-initiated one),
  never at or after ``until`` and never at a print stamped before
  ``not_before`` (an exit's entry fill);
* the clip walks that print and the other reachable prints of the same side
  with the same stamp, in archive order, at their volume-weighted price; the
  part beyond that sweep is charged at the sweep's worst price (the highest
  for a buy, the lowest for a sell) and counted;
* prints flagged ``rpi`` are never filled against (they stay in the features);
* an order no such print follows is not filled (``found`` is False); an exit
  is then priced, with ``fallback``, at the last reachable print of its side
  stamped at or before its arrival, its age recorded.

The brackets price the same fills (same prints, same times) four more ways:
``optimistic`` (the whole clip at the first print's price), ``pessimistic``
(the part beyond the sweep at the literal worst price of a same-side reachable
print stamped in ``(s, s + 60 s]``, ``s`` the fill's stamp, and before 23:59 for
an entry or before the day's end for an exit, or at the sweep's worst price if
there is none), ``stale`` (the whole clip at the last same-side reachable print
stamped at or before the decision, or the rule's price when the day has none
before it) and ``worse_of_two`` (per fill the worst of the rule's, the
pessimistic and the stale price; for an exit priced without a later print also
of the last reachable print of the other side stamped at or before its
arrival). An order the rule does not fill has no price under any bracket.

Interface (stable; ``spot/print_taker.py`` and ``spot/walk_tables.py`` use it)
-------------------------------------------------------------------------------
``PrintTape.from_frame(prints, day)``
    One day's prints in the contract of ``spot/prints.py`` (``timestamp``,
    ``price``, ``size``, ``aggressor`` and, if present, ``rpi``), in archive
    order. Every stamp must lie in the day.
``fill_orders(tape, side, arrival_ns, decision_ns, *, until_ns, not_before_ns=None,
fallback=False, settings=None) -> Fills``
    Vectorised over orders; ``side`` may be a scalar.
``round_trips(tape, grid_ns, decision_ns, side, hold_ns, *, flatten=True,
settings=None) -> RoundTrips``
    Entry, exit decision, exit and busy-until for every decision, each
    independent of the others. ``flatten=True`` is the executor's day:
    entries need a print before 23:59, the exit is decided at the first grid
    row stamped at or after the entry fill's print + hold or at 23:59 if that
    comes first, and an exit with no later print falls back. ``flatten=False``
    is the label's: no 23:59, and a round trip without an exit row or an exit
    print before the day ends is incomplete. The next decision may be taken
    at a row stamped at or after ``busy_until_ns``.

The label (``spot/print_label.py``), the executor and the walk tables call
these two functions and nothing else for a print fill, so a label, a table
entry and an executed trade on the same decision cannot disagree.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time
from typing import NamedTuple

import numpy as np
import pandas as pd

from trading_research.market_making.events import day_start_ns, to_ns
from trading_research.search_engine.walk_tables import clock_ns

NS_PER_DAY = 86_400 * 1_000_000_000
BUY, SELL = 1, -1


@dataclass(frozen=True, kw_only=True)
class PrintFillSettings:
    """What a print fill depends on besides the prints. Defaults are the registered ones."""

    clip: float = 0.010
    latency_ns: int = 10_000_000
    #: A fill whose print comes more than this after the arrival is late.
    late_print_ns: int = 1_000_000_000
    #: The pessimistic bracket's window after the fill's print.
    depth_window_ns: int = 60_000_000_000
    #: Beyond-sweep size counted only above this share of the clip (rounding).
    beyond_tolerance: float = 1e-9
    no_entry_before: time = time(0, 1)
    no_entry_after: time = time(23, 58)
    flatten_at: time = time(23, 59)

    def __post_init__(self) -> None:
        if not self.clip > 0:
            raise ValueError(f"the clip must be positive, got {self.clip}")
        if self.latency_ns < 0 or self.late_print_ns < 0 or self.depth_window_ns < 0:
            raise ValueError("latency, late-print and window durations cannot be negative")
        if not self.no_entry_before <= self.no_entry_after <= self.flatten_at:
            raise ValueError("entry hours must end before the flatten")


@dataclass(frozen=True)
class _Groups:
    """One side's reachable prints grouped by stamp, priced for one clip and window."""

    first: np.ndarray  # first side-index of each group
    end: np.ndarray  # one past its last side-index
    ts: np.ndarray
    rule: np.ndarray  # sweep VWAP, beyond at the sweep's worst price
    optimistic: np.ndarray  # the first print's price
    pessimistic: np.ndarray  # beyond at the window's worst price (window to 23:59)
    pessimistic_exit: np.ndarray  # the same, the window to the day's end
    sweep: np.ndarray  # size of the sweep
    beyond: np.ndarray  # clip beyond the sweep (0 when the sweep covers it)
    of_print: np.ndarray  # group of each side-index


@dataclass(frozen=True)
class _Side:
    """One side's reachable prints, archive order kept."""

    index: np.ndarray  # positions in the tape
    ts: np.ndarray
    price: np.ndarray
    size: np.ndarray


def _range_extreme(values: np.ndarray, lo: np.ndarray, hi: np.ndarray, worst: str) -> np.ndarray:
    """``max`` (or ``min``) of ``values[lo:hi]`` per query, NaN where the range is empty.

    A sparse table: ``log2(n)`` levels of extremes over power-of-two spans, so
    each query is two lookups.
    """
    op = np.maximum if worst == "max" else np.minimum
    out = np.full(len(lo), np.nan)
    has = hi > lo
    if not has.any():
        return out
    table = [np.asarray(values, dtype=np.float64)]
    span = 1
    while 2 * span <= len(values):
        prev = table[-1]
        table.append(op(prev[: len(prev) - span], prev[span:]))
        span *= 2
    a, b = lo[has], hi[has]
    level = np.floor(np.log2(b - a)).astype(np.int64)
    result = np.empty(len(a))
    for k in np.unique(level):
        pick = level == k
        width = 1 << int(k)
        result[pick] = op(table[k][a[pick]], table[k][b[pick] - width])
    out[has] = result
    return out


class PrintTape:
    """One day's prints, with each side's reachable prints ready for the rule."""

    def __init__(
        self,
        day: date,
        ts: np.ndarray,
        price: np.ndarray,
        size: np.ndarray,
        side: np.ndarray,
        rpi: np.ndarray | None = None,
    ) -> None:
        self.day = day
        self.ts = np.asarray(ts, dtype=np.int64)
        self.price = np.asarray(price, dtype=np.float64)
        self.size = np.asarray(size, dtype=np.float64)
        self.side = np.asarray(side, dtype=np.int64)
        n = len(self.ts)
        self.rpi = np.zeros(n, dtype=bool) if rpi is None else np.asarray(rpi, dtype=bool)
        if not len(self.price) == len(self.size) == len(self.side) == len(self.rpi) == n:
            raise ValueError("one price, size, side and rpi flag per print")
        if n > 1 and bool(np.any(np.diff(self.ts) < 0)):
            raise ValueError("print stamps must be non-decreasing (archive order)")
        self.start_ns = day_start_ns(day)
        self.end_ns = self.start_ns + NS_PER_DAY
        if n and (self.ts[0] < self.start_ns or self.ts[-1] >= self.end_ns):
            raise ValueError(f"every print must be stamped on {day}")
        if not np.isin(self.side, (BUY, SELL)).all():
            raise ValueError("an aggressor is +1 (buyer-initiated) or -1 (seller-initiated)")
        for name, values in (("price", self.price), ("size", self.size)):
            if not (np.isfinite(values).all() and (values > 0).all()):
                raise ValueError(f"a {name} is not a positive finite number")
        self._sides: dict[int, _Side] = {}
        self._groups: dict[tuple[int, PrintFillSettings], _Groups] = {}
        self._reachable: np.ndarray | None = None

    @classmethod
    def from_frame(cls, prints: pd.DataFrame, day: date) -> PrintTape:
        """From a frame in the print contract; ``rpi`` is optional (absent: none)."""
        rpi = prints["rpi"].to_numpy(dtype=bool) if "rpi" in prints.columns else None
        return cls(
            day,
            to_ns(prints["timestamp"]),
            prints["price"].to_numpy(dtype=np.float64),
            prints["size"].to_numpy(dtype=np.float64),
            prints["aggressor"].to_numpy(dtype=np.int64),
            rpi,
        )

    def __len__(self) -> int:
        return len(self.ts)

    def flatten_ns(self, settings: PrintFillSettings) -> int:
        return self.start_ns + clock_ns(settings.flatten_at)

    def side_prints(self, side: int) -> _Side:
        """The reachable prints of ``side``, in archive order."""
        if side not in self._sides:
            index = np.flatnonzero((self.side == side) & ~self.rpi).astype(np.int64)
            self._sides[side] = _Side(index, self.ts[index], self.price[index], self.size[index])
        return self._sides[side]

    def groups(self, side: int, settings: PrintFillSettings) -> _Groups:
        """``side``'s prints grouped by stamp and priced for ``settings``' clip."""
        key = (side, settings)
        if key not in self._groups:
            self._groups[key] = _price_groups(self.side_prints(side), side, settings, self)
        return self._groups[key]

    @property
    def reachable(self) -> np.ndarray:
        """Positions of the reachable prints of either side, in archive order."""
        if self._reachable is None:
            self._reachable = np.flatnonzero(~self.rpi).astype(np.int64)
        return self._reachable


def _price_groups(
    prints: _Side, side: int, settings: PrintFillSettings, tape: PrintTape
) -> _Groups:
    n, clip = len(prints.ts), settings.clip
    new = np.ones(n, dtype=bool)
    new[1:] = prints.ts[1:] != prints.ts[:-1]
    first = np.flatnonzero(new).astype(np.int64)
    end = np.append(first[1:], n).astype(np.int64)
    of_print = (np.cumsum(new) - 1).astype(np.int64)
    count = len(first)
    # Exclusive prefix sums inside each group, summed in archive order.
    before = (
        pd.Series(prints.size).groupby(of_print).cumsum().groupby(of_print).shift(1, fill_value=0.0)
    ).to_numpy(dtype=np.float64)
    take = np.minimum(np.maximum(clip - before, 0.0), prints.size)
    cost = np.bincount(of_print, weights=take * prints.price, minlength=count)
    taken = np.bincount(of_print, weights=take, minlength=count)
    sweep = np.bincount(of_print, weights=prints.size, minlength=count)
    beyond = np.maximum(clip - taken, 0.0)
    beyond = np.where(beyond > settings.beyond_tolerance * clip, beyond, 0.0)
    op = np.maximum if side == BUY else np.minimum
    sweep_worst = op.reduceat(prints.price, first) if n else np.zeros(0)
    ts = prints.ts[first]
    worse = "max" if side == BUY else "min"

    def window(until_ns: int) -> np.ndarray:
        """The beyond part at the worst print of ``(s, s + 60 s]`` before ``until_ns``."""
        bound = np.minimum(ts + settings.depth_window_ns, until_ns - 1)
        hi = np.searchsorted(prints.ts, bound, side="right").astype(np.int64)
        worst = _range_extreme(prints.price, end, hi, worse)
        return (cost + beyond * np.where(np.isnan(worst), sweep_worst, worst)) / clip

    return _Groups(
        first=first,
        end=end,
        ts=ts,
        rule=(cost + beyond * sweep_worst) / clip,
        optimistic=prints.price[first] if n else np.zeros(0),
        pessimistic=window(tape.flatten_ns(settings)),
        pessimistic_exit=window(tape.end_ns),
        sweep=sweep,
        beyond=beyond,
        of_print=of_print,
    )


class Fills(NamedTuple):
    """One value per order. Prices are per unit of the clip, in the prints' quote."""

    side: np.ndarray
    arrival_ns: np.ndarray
    decision_ns: np.ndarray
    #: A reachable print of the side was found in ``[max(arrival, not_before), until)``.
    found: np.ndarray
    #: The fill's print (or the fallback's) as a position in the tape; -1 if none.
    print_index: np.ndarray
    #: Its stamp; -1 if none.
    print_ns: np.ndarray
    rule: np.ndarray
    optimistic: np.ndarray
    pessimistic: np.ndarray
    stale: np.ndarray
    worse_of_two: np.ndarray
    #: The size the sweep offered (NaN unless found), and the clip beyond it (0 if none).
    sweep: np.ndarray
    beyond: np.ndarray
    #: Print stamp less arrival, for found fills; -1 otherwise.
    wait_ns: np.ndarray
    late: np.ndarray
    #: Priced at the last print of its side before the arrival (no later print).
    fallback: np.ndarray
    #: The fallback print's age at the (effective) arrival; -1 otherwise.
    age_ns: np.ndarray

    def price(self, bracket: str = "rule") -> np.ndarray:
        """The price under ``bracket``: one of :data:`BRACKETS`."""
        if bracket not in BRACKETS:
            raise ValueError(f"unknown bracket {bracket!r}; expected one of {BRACKETS}")
        return np.asarray(getattr(self, bracket))

    @property
    def beyond_sweep(self) -> np.ndarray:
        """Counted as ``fill_beyond_sweep``: part of the clip lay beyond the sweep."""
        return np.asarray(self.beyond > 0.0)


#: The registered default rule and its four brackets, in the registration's order.
BRACKETS: tuple[str, ...] = ("rule", "optimistic", "pessimistic", "stale", "worse_of_two")


def _as_orders(side: int | np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    columns = [np.asarray(a, dtype=np.int64) for a in arrays]
    n = len(columns[0])
    if any(len(c) != n for c in columns):
        raise ValueError("one arrival, decision and bound per order")
    sides = np.broadcast_to(np.asarray(side, dtype=np.int64), (n,)).copy()
    if not np.isin(sides, (BUY, SELL)).all():
        raise ValueError("a side is +1 (buy) or -1 (sell)")
    return (sides, *columns)


def fill_orders(
    tape: PrintTape,
    side: int | np.ndarray,
    arrival_ns: np.ndarray,
    decision_ns: np.ndarray,
    *,
    until_ns: int | np.ndarray,
    not_before_ns: np.ndarray | None = None,
    fallback: bool = False,
    settings: PrintFillSettings | None = None,
) -> Fills:
    """Fill each order by the side-correct print rule, with every bracket.

    ``until_ns`` bounds the prints an order may fill at (exclusive): 23:59 for
    an executor's entry, the day's end otherwise; the pessimistic window ends
    there too (an order bounded after 23:59 reads its window to the day's end).
    ``not_before_ns`` (an exit's entry fill stamp) keeps an exit off earlier
    prints. With ``fallback`` an order no print follows is priced at the last
    reachable print of its side stamped at or before its effective arrival, or,
    if its side has none, of either side; otherwise every price is NaN.
    """
    settings = settings or PrintFillSettings()
    arrival = np.asarray(arrival_ns, dtype=np.int64)
    bound = np.broadcast_to(np.asarray(until_ns, dtype=np.int64), arrival.shape)
    floor = arrival if not_before_ns is None else np.asarray(not_before_ns, dtype=np.int64)
    sides, arrival, decision, bound, floor = _as_orders(side, arrival, decision_ns, bound, floor)
    n = len(arrival)
    effective = np.maximum(arrival, floor)
    out = {
        "found": np.zeros(n, dtype=bool),
        "fallback": np.zeros(n, dtype=bool),
        "print_index": np.full(n, -1, dtype=np.int64),
        "print_ns": np.full(n, -1, dtype=np.int64),
        "wait_ns": np.full(n, -1, dtype=np.int64),
        "age_ns": np.full(n, -1, dtype=np.int64),
        "sweep": np.full(n, np.nan),
        "beyond": np.zeros(n),
        "stale": np.full(n, np.nan),
        "other_side": np.full(n, np.nan),
        **{name: np.full(n, np.nan) for name in ("rule", "optimistic", "pessimistic")},
    }
    late_window = bound > tape.flatten_ns(settings)
    for s in (BUY, SELL):
        mine = np.flatnonzero(sides == s)
        if len(mine):
            _fill_side(tape, s, mine, effective, decision, bound, late_window, settings, out)
    if fallback:
        _fall_back(tape, sides, effective, out)
    found = out["found"]
    out["wait_ns"][found] = out["print_ns"][found] - arrival[found]
    priced = found | out["fallback"]
    stale = np.where(priced, np.where(np.isnan(out["stale"]), out["rule"], out["stale"]), np.nan)
    # Worse of two: the worst of the rule's, the pessimistic and the stale price, and
    # for an exit priced without a later print the other side's last print too.
    candidates = (out["rule"], out["pessimistic"], stale, out["other_side"])
    worst_buy, worst_sell = candidates[0], candidates[0]
    for values in candidates[1:]:
        worst_buy, worst_sell = np.fmax(worst_buy, values), np.fmin(worst_sell, values)
    worse = np.where(priced, np.where(sides > 0, worst_buy, worst_sell), np.nan)
    return Fills(
        side=sides,
        arrival_ns=arrival,
        decision_ns=decision,
        found=found,
        print_index=out["print_index"],
        print_ns=out["print_ns"],
        rule=out["rule"],
        optimistic=out["optimistic"],
        pessimistic=out["pessimistic"],
        stale=stale,
        worse_of_two=worse,
        sweep=out["sweep"],
        beyond=out["beyond"],
        wait_ns=out["wait_ns"],
        late=found & (out["wait_ns"] > settings.late_print_ns),
        fallback=out["fallback"],
        age_ns=out["age_ns"],
    )


def _fill_side(
    tape: PrintTape,
    side: int,
    mine: np.ndarray,
    effective: np.ndarray,
    decision: np.ndarray,
    bound: np.ndarray,
    late_window: np.ndarray,
    settings: PrintFillSettings,
    out: dict[str, np.ndarray],
) -> None:
    prints, groups = tape.side_prints(side), tape.groups(side, settings)
    last = np.searchsorted(prints.ts, decision[mine], side="right") - 1
    known = last >= 0
    out["stale"][mine[known]] = prints.price[last[known]]
    j = np.searchsorted(prints.ts, effective[mine], side="left")
    hit = j < len(prints.ts)
    hit[hit] = prints.ts[j[hit]] < bound[mine[hit]]
    rows, g = mine[hit], groups.of_print[j[hit]]
    out["found"][rows] = True
    out["print_index"][rows] = prints.index[j[hit]]
    out["print_ns"][rows] = groups.ts[g]
    for name in ("rule", "optimistic", "sweep", "beyond"):
        out[name][rows] = getattr(groups, name)[g]
    out["pessimistic"][rows] = np.where(
        late_window[rows], groups.pessimistic_exit[g], groups.pessimistic[g]
    )


def _fall_back(
    tape: PrintTape, sides: np.ndarray, effective: np.ndarray, out: dict[str, np.ndarray]
) -> None:
    for s in (BUY, SELL):
        missing = np.flatnonzero((sides == s) & ~out["found"])
        if not len(missing):
            continue
        prints = tape.side_prints(s)
        at = _last_at_or_before(prints.index, prints.ts, effective[missing])
        if (at < 0).any():
            reachable = tape.reachable
            either = _last_at_or_before(reachable, tape.ts[reachable], effective[missing])
            at = np.where(at >= 0, at, either)
        other = tape.side_prints(-s)
        opposite = _last_at_or_before(other.index, other.ts, effective[missing])
        ok = at >= 0
        rows, at, opposite = missing[ok], at[ok], opposite[ok]
        out["fallback"][rows] = True
        out["print_index"][rows] = at
        out["print_ns"][rows] = tape.ts[at]
        out["age_ns"][rows] = effective[rows] - tape.ts[at]
        for name in ("rule", "optimistic", "pessimistic"):
            out[name][rows] = tape.price[at]
        known = opposite >= 0
        out["other_side"][rows[known]] = tape.price[opposite[known]]


def _last_at_or_before(index: np.ndarray, ts: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """``index`` of the last print stamped at or before each target; -1 if none."""
    j = np.searchsorted(ts, targets, side="right") - 1
    if not len(ts):
        return np.full(len(targets), -1, dtype=np.int64)
    return np.where(j >= 0, index[np.maximum(j, 0)], -1).astype(np.int64)


class RoundTrips(NamedTuple):
    """One round trip per decision, each as if it were the only one."""

    decision_ns: np.ndarray
    side: np.ndarray
    entry: Fills
    #: The exit's decision stamp; -1 where there is no entry or no exit row.
    exit_decision_ns: np.ndarray
    #: ``hold`` (a grid row after the entry fill + hold) or ``day_end`` (23:59).
    exit_reason: np.ndarray
    exit: Fills
    #: The next decision may be taken at a row stamped at or after this; -1
    #: where no entry is made.
    busy_until_ns: np.ndarray

    @property
    def made(self) -> np.ndarray:
        """The entry found a print (otherwise counted as ``entries_no_print``)."""
        return np.asarray(self.entry.found)

    @property
    def complete(self) -> np.ndarray:
        """Entered and exited at a price: found, or by the fallback when flattening."""
        return np.asarray(self.entry.found & (self.exit.found | self.exit.fallback))


def round_trips(
    tape: PrintTape,
    grid_ns: np.ndarray,
    decision_ns: np.ndarray,
    side: int | np.ndarray,
    hold_ns: int,
    *,
    flatten: bool = True,
    settings: PrintFillSettings | None = None,
) -> RoundTrips:
    """Every decision's entry, exit decision, exit and busy-until (module docstring)."""
    settings = settings or PrintFillSettings()
    if hold_ns <= 0:
        raise ValueError("the hold must be positive")
    grid = np.asarray(grid_ns, dtype=np.int64)
    if len(grid) > 1 and bool(np.any(np.diff(grid) <= 0)):
        raise ValueError("grid stamps must be strictly increasing")
    decision = np.asarray(decision_ns, dtype=np.int64)
    latency, flatten_ns = settings.latency_ns, tape.flatten_ns(settings)
    entry = fill_orders(
        tape,
        side,
        decision + latency,
        decision,
        until_ns=flatten_ns if flatten else tape.end_ns,
        settings=settings,
    )
    n = len(decision)
    row = np.searchsorted(grid, entry.print_ns + hold_ns, side="left")
    has_row = entry.found & (row < len(grid))
    exit_decision = np.full(n, -1, dtype=np.int64)
    exit_decision[has_row] = grid[row[has_row]]
    reason = np.full(n, "hold", dtype=object)
    if flatten:
        late = entry.found & (~has_row | (exit_decision > flatten_ns))
        exit_decision[late] = flatten_ns
        reason[late] = "day_end"
    reason[exit_decision < 0] = ""
    priced = exit_decision >= 0
    exit_arrival = np.where(priced, exit_decision + latency, tape.end_ns)
    exit_ = fill_orders(
        tape,
        -entry.side,
        exit_arrival,
        np.where(priced, exit_decision, tape.end_ns),
        until_ns=tape.end_ns,
        not_before_ns=np.where(priced, entry.print_ns, tape.end_ns),
        fallback=flatten,
        settings=settings,
    )
    exit_ = _mask(exit_, priced)
    busy = np.where(entry.found, np.maximum(exit_.print_ns, exit_arrival), -1)
    return RoundTrips(decision, entry.side, entry, exit_decision, reason, exit_, busy)


def _mask(fills: Fills, keep: np.ndarray) -> Fills:
    """``fills`` with every order outside ``keep`` reset to not found."""
    drop = ~keep
    if not drop.any():
        return fills
    values = fills._asdict()
    for name in ("found", "late", "fallback"):
        values[name] = np.where(drop, False, values[name])
    for name in ("print_index", "print_ns", "wait_ns", "age_ns"):
        values[name] = np.where(drop, -1, values[name])
    for name in (*BRACKETS, "sweep"):
        values[name] = np.where(drop, np.nan, values[name])
    values["beyond"] = np.where(drop, 0.0, values["beyond"])
    return Fills(**values)
