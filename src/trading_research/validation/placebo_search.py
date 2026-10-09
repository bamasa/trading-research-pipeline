"""The placebo search: how good the best cell looks when no cell has information.

The choice on D picks one cell per stratum out of thousands, so the chosen
cell's statistic is biased upward by the act of choosing. This module re-runs
that whole choice (:func:`trading_research.validation.choice.rerun`) on
outcomes that carry no information and counts how often the best of those
looks at least as good as what was found:

* **Null A** flips the side of every trade in a UTC hour of the test days, the
  same sign for every cell in that hour. A flipped trade's P&L is the walk of
  the opposite side, funding included, and its G changes sign. The inputs are
  per-cell sums over each hour (:class:`HourCells`).
* **Null B** executes each source day's accepted entries on the day ``s``
  places later in the circular calendar order of the test days, for every
  shift from 7 to 143. The execution itself belongs to the executor's walk
  tables; here it is a callable returning the shifted cell-by-day matrix, and
  :func:`null_b_pairs` says which pairs of days contribute.
* **Admission (iv)** compares the observed A with the maximum over strata of
  each replication, by counting directly: p = (1 + count) / (1 + replications)
  against 0.05 / R_D, decided in exact rational arithmetic
  (:func:`passes_iv`). The interpolated placebo percentile of
  ``market_making.verdicts.exceeds_placebos`` admits more and is not used.
* **The refit replication** re-runs stratum C's selection on fits trained on
  labels shifted circularly by 28 days inside each training span
  (:func:`shift_labels_circularly`); C* must then also beat that A strictly.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import pandas as pd

from trading_research.validation.choice import (
    MIN_TRADES_PER_DAY,
    CellDays,
    ChoiceError,
    StratumOutcome,
    rerun,
)

#: Registered sizes: Null A's draws (seeds 0 to 998), Null B's shift range on
#: the 150 test days, the refit shift, and the family level of (iv).
NULL_A_DRAWS = 999
NULL_B_DAYS = 150
NULL_B_SHIFT_RANGE = (7, 143)
REFIT_SHIFT_DAYS = 28
ALPHA = 0.05


@dataclass(frozen=True)
class HourCells:
    """Per-cell sums over each UTC hour of the usable test days (cells x hours).

    ``hour_day`` gives each hour's day position (0 to ``days - 1``, hours in
    time order). ``net_kept`` is the summed net in USDT of the trades as
    taken, ``net_flipped`` the same trades' walks of the opposite side;
    ``notional`` the summed entry notional in USDT; ``g`` the summed G in bp.
    """

    hour_day: np.ndarray
    days: int
    trades: np.ndarray
    notional: np.ndarray
    net_kept: np.ndarray
    net_flipped: np.ndarray
    g: np.ndarray

    def __post_init__(self) -> None:
        arrays = (self.trades, self.notional, self.net_kept, self.net_flipped, self.g)
        shapes = {a.shape for a in arrays}
        if len(shapes) != 1 or self.trades.ndim != 2:
            raise ChoiceError(f"the hour arrays must share one 2-D shape, got {shapes}")
        if self.hour_day.shape != (self.trades.shape[1],):
            raise ChoiceError("hour_day needs one day position per hour")
        if np.any(np.diff(self.hour_day) < 0) or (
            len(self.hour_day) and (self.hour_day.min() < 0 or self.hour_day.max() >= self.days)
        ):
            raise ChoiceError("hour_day must be non-decreasing positions in [0, days)")

    @property
    def hours(self) -> int:
        return int(self.trades.shape[1])

    def _by_day(self, values: np.ndarray) -> np.ndarray:
        out = np.zeros((values.shape[0], self.days), dtype=np.float64)
        starts = np.searchsorted(self.hour_day, np.arange(self.days))
        ends = np.searchsorted(self.hour_day, np.arange(self.days), side="right")
        filled = ends > starts
        if filled.any():
            out[:, filled] = np.add.reduceat(values, starts[filled], axis=1)
        return out

    def under_signs(self, signs: np.ndarray) -> CellDays:
        """The cell-by-day matrix with each hour's trades kept (+1) or flipped (-1)."""
        sign = np.asarray(signs)
        if sign.shape != (self.hours,) or not np.isin(sign, (-1, 1)).all():
            raise ChoiceError("one sign of +1 or -1 per hour is needed")
        net = self._by_day(np.where(sign < 0, self.net_flipped, self.net_kept))
        g = self._by_day(self.g * sign)
        trades = self._by_day(self.trades).astype(np.int64)
        notional = self._by_day(self.notional)
        traded = trades > 0
        with np.errstate(invalid="ignore", divide="ignore"):
            net_bp = np.where(traded, net / notional * 1e4, np.nan)
            g_bp = np.where(traded, g / trades, np.nan)
        return CellDays(trades, net_bp, g_bp)

    def kept(self) -> CellDays:
        """The cell-by-day matrix as traded."""
        return self.under_signs(np.ones(self.hours, dtype=np.int64))

    @classmethod
    def from_trades(
        cls, trades: pd.DataFrame, keys: Sequence[str], days: Sequence[object]
    ) -> HourCells:
        """Sum trade rows into every UTC hour of the given usable days.

        Columns: ``key``, ``time`` (the entry, UTC), ``side`` (+1 long, -1
        short), ``notional`` (entry notional, USDT), ``pnl_long_bp`` and
        ``pnl_short_bp`` (the walked net of each side over the same entry and
        hold, fees and funding included, in bp of the entry notional). A
        trade's net is the walk of its side, its flipped net the other side's,
        and its G ``side * (pnl_long_bp - pnl_short_bp) / 2``. A trade outside
        the days or the keys is refused.
        """
        stamps = pd.DatetimeIndex(np.asarray(list(days), dtype="datetime64[ns]")).normalize()
        hours_per_day = 24
        row = pd.Index(list(keys)).get_indexer(pd.Index(trades["key"]))
        entry = pd.DatetimeIndex(trades["time"])
        if entry.tz is not None:
            entry = entry.tz_convert("UTC").tz_localize(None)
        day = stamps.get_indexer(entry.normalize())
        if (row < 0).any() or (day < 0).any():
            raise ChoiceError("a trade outside the keys or the usable days")
        side = trades["side"].to_numpy(dtype=np.float64)
        if not np.isin(side, (-1.0, 1.0)).all():
            raise ChoiceError("side must be +1 or -1")
        long_bp = trades["pnl_long_bp"].to_numpy(dtype=np.float64)
        short_bp = trades["pnl_short_bp"].to_numpy(dtype=np.float64)
        notional = trades["notional"].to_numpy(dtype=np.float64)
        hours = len(stamps) * hours_per_day
        flat = row * hours + day * hours_per_day + entry.hour.to_numpy()
        size = len(keys) * hours

        def total(weights: np.ndarray) -> np.ndarray:
            return np.bincount(flat, weights=weights, minlength=size).reshape(len(keys), hours)

        own = np.where(side > 0, long_bp, short_bp)
        other = np.where(side > 0, short_bp, long_bp)
        return cls(
            hour_day=np.repeat(np.arange(len(stamps)), hours_per_day),
            days=len(stamps),
            trades=np.bincount(flat, minlength=size).reshape(len(keys), hours).astype(np.int64),
            notional=total(notional),
            net_kept=total(own * notional / 1e4),
            net_flipped=total(other * notional / 1e4),
            g=total(side * (long_bp - short_bp) / 2.0),
        )


def null_a_signs(draw: int, hours: int) -> np.ndarray:
    """Draw ``draw``'s sign per hour: ``default_rng(draw).integers(0, 2)`` per
    hour in time order, 0 keeping the side (+1) and 1 flipping it (-1)."""
    return 1 - 2 * np.random.default_rng(draw).integers(0, 2, size=hours)


@dataclass(frozen=True)
class NullDraws:
    """A per stratum in every replication of one null (replications x strata)."""

    strata: tuple[str, ...]
    a: np.ndarray

    @property
    def replications(self) -> int:
        return int(self.a.shape[0])

    @property
    def maximum(self) -> np.ndarray:
        """The maximum over strata of each replication's A."""
        return self.a.max(axis=1)


def _replicate(
    outcomes: Callable[[int], CellDays],
    seeds: np.ndarray | range,
    strata: Mapping[str, np.ndarray],
    index: np.ndarray,
    fixed: Mapping[str, int] | None,
    min_trades_per_day: float,
) -> NullDraws:
    names = tuple(strata)
    a = np.empty((len(seeds), len(names)))
    for row, seed in enumerate(seeds):
        found = rerun(
            outcomes(int(seed)), dict(strata), index, dict(fixed or {}), min_trades_per_day
        )
        a[row] = [found[name].a for name in names]
    return NullDraws(names, a)


def null_a(
    hours: HourCells,
    strata: Mapping[str, np.ndarray],
    index: np.ndarray,
    draws: int = NULL_A_DRAWS,
    fixed: Mapping[str, int] | None = None,
    min_trades_per_day: float = MIN_TRADES_PER_DAY,
) -> NullDraws:
    """Null A: the whole choice and A re-run under ``draws`` hour-block sign
    flips, draw ``b`` seeded with ``b``. With ``fixed``, those strata keep
    their cell (a configuration fixed in advance, no selection)."""
    return _replicate(
        lambda b: hours.under_signs(null_a_signs(b, hours.hours)),
        range(draws),
        strata,
        index,
        fixed,
        min_trades_per_day,
    )


def global_flip(
    hours: HourCells,
    strata: Mapping[str, np.ndarray],
    index: np.ndarray,
    min_trades_per_day: float = MIN_TRADES_PER_DAY,
) -> dict[str, StratumOutcome]:
    """Every trade flipped at once: a diagnostic only, an anti-signal when
    information exists."""
    flipped = hours.under_signs(-np.ones(hours.hours, dtype=np.int64))
    return rerun(flipped, dict(strata), index, None, min_trades_per_day)


def null_b_shifts(
    days: int = NULL_B_DAYS, shift_range: tuple[int, int] = NULL_B_SHIFT_RANGE
) -> range:
    """Every registered shift: 7 to 143 inclusive on 150 days, 137 shifts."""
    low, high = shift_range
    if not 0 < low <= high < days:
        raise ChoiceError(f"shifts {shift_range} do not fit a circle of {days} days")
    return range(low, high + 1)


@dataclass(frozen=True)
class ShiftPairs:
    """Which source day executes on which target day under one shift."""

    shift: int
    source: np.ndarray
    target: np.ndarray
    #: Pairs with an excluded source or target day: they contribute nothing.
    excluded: int


def null_b_pairs(usable: np.ndarray, shift: int) -> ShiftPairs:
    """Source day ``i`` executes on target day ``(i + shift) mod n`` in the
    circular calendar order of the ``n`` test days; a pair with an unusable
    source or target contributes nothing and is counted. Pairs are listed in
    target order."""
    ok = np.asarray(usable, dtype=bool)
    n = len(ok)
    target = np.arange(n)
    source = (target - shift) % n
    keep = ok[source] & ok[target]
    return ShiftPairs(int(shift), source[keep], target[keep], int((~keep).sum()))


def null_b(
    shifted: Callable[[int], CellDays],
    strata: Mapping[str, np.ndarray],
    index: np.ndarray,
    shifts: range | None = None,
    fixed: Mapping[str, int] | None = None,
    min_trades_per_day: float = MIN_TRADES_PER_DAY,
) -> NullDraws:
    """Null B: the whole choice and A re-run on each shift's outcomes.

    ``shifted(s)`` returns the cell-by-day matrix of the source days' accepted
    entries executed on the target days' walk tables, one column per
    contributing pair of :func:`null_b_pairs`, entries that fall while the
    mapped previous position is open skipped. The five-trade filter divides by
    the number of columns returned.
    """
    return _replicate(shifted, shifts or null_b_shifts(), strata, index, fixed, min_trades_per_day)


def exceedances(null_maximum: np.ndarray, observed: float) -> int:
    """How many replications' maximum over strata is at or above ``observed``."""
    values = np.asarray(null_maximum, dtype=np.float64)
    if not np.isfinite(values).all() or not np.isfinite(observed):
        raise ChoiceError("the null maxima and the observed A must be finite")
    return int((values >= observed).sum())


def placebo_p(count: int, replications: int) -> float:
    """``(1 + count) / (1 + replications)``."""
    return (1 + count) / (1 + replications)


def _within(count: int, replications: int, runs: int, alpha: float) -> bool:
    # (1 + count) / (1 + replications) <= alpha / runs, in exact arithmetic.
    return (1 + count) * runs <= Fraction(str(alpha)) * (1 + replications)


def max_passing_count(replications: int, runs: int = 1, alpha: float = ALPHA) -> int:
    """The largest count of replications at or above A that still passes."""
    count = -1
    while _within(count + 1, replications, runs, alpha):
        count += 1
    return count


def passes_iv(
    count_a: int,
    replications_a: int,
    count_b: int,
    replications_b: int,
    runs: int = 1,
    alpha: float = ALPHA,
) -> bool:
    """(iv): max(p_A, p_B) <= alpha / R_D, decided on the counts."""
    if runs < 1:
        raise ChoiceError(f"R_D counts complete D runs and is at least 1, got {runs}")
    return _within(count_a, replications_a, runs, alpha) and _within(
        count_b, replications_b, runs, alpha
    )


@dataclass(frozen=True)
class SearchNullResult:
    """One stratum's chosen cell against both nulls."""

    stratum: str
    cell: int | None
    a: float
    count_a: int
    count_b: int
    p_a: float
    p_b: float
    #: Stratum C's refit A (None for a stratum not refitted).
    a_refit: float | None
    passes: bool

    @property
    def p(self) -> float:
        return max(self.p_a, self.p_b)


def admission_iv(
    observed: Mapping[str, StratumOutcome],
    a_null: NullDraws,
    b_null: NullDraws,
    runs: int = 1,
    alpha: float = ALPHA,
    refit: Mapping[str, float] | None = None,
) -> dict[str, SearchNullResult]:
    """Condition (iv) for every stratum's chosen cell.

    Each stratum's A is counted against the maximum over strata of each null
    replication. A stratum named in ``refit`` passes only if its A is also
    strictly above its own refit A. A stratum without a chosen cell does not
    pass, with p = 1.
    """
    refit = refit or {}
    out: dict[str, SearchNullResult] = {}
    for name, found in observed.items():
        a_refit = refit.get(name)
        if found.cell is None:
            out[name] = SearchNullResult(name, None, found.a, 0, 0, 1.0, 1.0, a_refit, False)
            continue
        count_a = exceedances(a_null.maximum, found.a)
        count_b = exceedances(b_null.maximum, found.a)
        passes = passes_iv(count_a, a_null.replications, count_b, b_null.replications, runs, alpha)
        if a_refit is not None:
            passes = passes and found.a > a_refit
        p_a = placebo_p(count_a, a_null.replications)
        p_b = placebo_p(count_b, b_null.replications)
        out[name] = SearchNullResult(
            name, found.cell, found.a, count_a, count_b, p_a, p_b, a_refit, passes
        )
    return out


def p_search(results: Mapping[str, SearchNullResult], runs: int = 1) -> float:
    """max(p_A, p_B) of the stratum with the larger A, times R_D, capped at 1."""
    chosen = [r for r in results.values() if r.cell is not None]
    if not chosen:
        return 1.0
    best = max(chosen, key=lambda r: r.a)
    return min(1.0, best.p * runs)


def selection_premium(observed_a: float, null_maximum: np.ndarray) -> tuple[float, float]:
    """The median of a null's maximum of A, and the observed A minus it."""
    median = float(np.median(np.asarray(null_maximum, dtype=np.float64)))
    return median, float(observed_a - median)


def shift_labels_circularly(
    labels: pd.Series,
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
    days: int = REFIT_SHIFT_DAYS,
) -> pd.Series:
    """Training labels moved ``days`` later around their training span.

    The span is ``[start, end)``. The label at stamp ``t`` becomes the one that
    stood at ``t - days``, wrapped to ``t - days + (end - start)`` when that
    falls before ``start``, matched on exact stamps. A stamp whose source has
    no label (a gap or an excluded day) gets NaN, and the fit drops it. Every
    pairing of features with outcomes is broken while the labels' own
    distribution and their dependence inside each day are kept.
    """
    if not isinstance(labels.index, pd.DatetimeIndex):
        raise ChoiceError("labels need a DatetimeIndex of row stamps")
    begin, finish = pd.Timestamp(start), pd.Timestamp(end)
    span = finish - begin
    lag = np.timedelta64(int(days), "D")
    if not np.timedelta64(0, "D") < lag < span:
        raise ChoiceError(f"a shift of {days} days does not fit the span {begin} to {finish}")
    stamps = labels.index
    if stamps.min() < begin or stamps.max() >= finish:
        raise ChoiceError("labels outside the training span")
    if stamps.has_duplicates:
        raise ChoiceError("two labels share one stamp")
    source = stamps - lag
    source = source.where(source >= begin, source + span)
    return pd.Series(labels.reindex(source).to_numpy(), index=stamps, name=labels.name)


def refit_replication(
    refitted: Callable[[int], CellDays],
    members: np.ndarray,
    index: np.ndarray,
    days: int = REFIT_SHIFT_DAYS,
    min_trades_per_day: float = MIN_TRADES_PER_DAY,
) -> StratumOutcome:
    """Stratum C's selection re-run on fits trained on shifted labels.

    ``refitted(days)`` redoes every boosted and logistic fit of every fold on
    labels from :func:`shift_labels_circularly` and returns stratum C's
    cell-by-day matrix on the test days, scored as in the search. The result's
    ``a`` is A_refit, which C* must beat strictly.
    """
    return rerun(refitted(days), {"C": members}, index, None, min_trades_per_day)["C"]
