"""The registered choice on the development period, as a re-runnable function.

The search scores every cell of its catalogue on every test day and then
chooses one cell per stratum. A placebo search is only honest if it repeats
that choice exactly on placebo outcomes, so the choice lives here as a pure
function of a cell-by-day matrix, with nothing read from disk and no state:

* :class:`CellDays` holds, for each cell and test day, the trades, the day's
  ``net_bp_trade`` and the day's mean G (NaN on a day without a trade);
* :func:`neighbourhood_index` lists each cell's neighbours once, as a padded
  integer matrix, from ordered axes given as numbers;
* :func:`choose` applies the rule: S is the neighbourhood median of each
  cell's mean daily net, the cell must average at least five trades a day,
  the highest S wins and ties go to the cell's own mean;
* :func:`statistic_a` is the admission statistic: the neighbourhood median of
  each cell's day-level t of G;
* :func:`carry_forward` re-runs the choice month by month and scores the
  chosen cell on the next month after an embargo;
* :func:`pbo_of_choice` puts the same rule through combinatorially symmetric
  cross-validation.

A cell without a trade scores 0 wherever a cell value enters a median, so a
sparse corner of the grid cannot drop out of its neighbours' medians.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading_research.evaluation.overfitting import CSCV_BLOCKS, Pbo, cscv_pbo

#: Trades a day a cell must average over the test days to be chosen.
MIN_TRADES_PER_DAY = 5.0


class ChoiceError(ValueError):
    """The inputs cannot describe a grid of cells over days."""


@dataclass(frozen=True)
class CellDays:
    """Per-cell daily values over the test days; every array is cells x days.

    ``net_bp`` is a day's ``net_bp_trade`` (the day's net over its entry
    notional, x 1e4) and ``g_bp`` the mean of G over the day's trades. Both are
    NaN on a day without a trade, where ``trades`` is 0.
    """

    trades: np.ndarray
    net_bp: np.ndarray
    g_bp: np.ndarray

    def __post_init__(self) -> None:
        shapes = {a.shape for a in (self.trades, self.net_bp, self.g_bp)}
        if len(shapes) != 1 or self.trades.ndim != 2:
            raise ChoiceError(f"trades, net_bp and g_bp must share one 2-D shape, got {shapes}")

    @property
    def cells(self) -> int:
        return int(self.trades.shape[0])

    @property
    def days(self) -> int:
        return int(self.trades.shape[1])

    def on_days(self, columns: np.ndarray | Sequence[int]) -> CellDays:
        """The same cells on a subset of the days (positions or a mask)."""
        index = np.asarray(columns)
        return CellDays(self.trades[:, index], self.net_bp[:, index], self.g_bp[:, index])

    @classmethod
    def from_ledger(
        cls, rows: pd.DataFrame, keys: Sequence[str], days: Sequence[object]
    ) -> CellDays:
        """Pivot trial-ledger rows of one run (``key``, ``day``, ``trades``,
        ``net_bp``, ``G_bp``) onto the given cell and day order.

        A (key, day) pair without a row, or with no trade, is a day without a
        trade. A row whose key or day is not in the given order is refused.
        """
        unknown_keys = set(rows["key"]) - set(keys)
        unknown_days = set(rows["day"]) - set(days)
        if unknown_keys or unknown_days:
            raise ChoiceError(
                f"rows outside the grid: {len(unknown_keys)} keys, {len(unknown_days)} days"
            )
        if rows.duplicated(["key", "day"]).any():
            raise ChoiceError("two ledger rows share one (key, day); pass one run")
        row = pd.Index(list(keys)).get_indexer(pd.Index(rows["key"]))
        col = pd.Index(list(days), dtype=object).get_indexer(pd.Index(rows["day"], dtype=object))
        trades = np.zeros((len(keys), len(days)), dtype=np.int64)
        net = np.full(trades.shape, np.nan)
        g = np.full(trades.shape, np.nan)
        count = rows["trades"].to_numpy(dtype=np.int64)
        traded = count > 0
        trades[row, col] = count
        net[row[traded], col[traded]] = rows["net_bp"].to_numpy(dtype=np.float64)[traded]
        g[row[traded], col[traded]] = rows["G_bp"].to_numpy(dtype=np.float64)[traded]
        return cls(trades, net, g)


def cell_means(values: np.ndarray) -> np.ndarray:
    """Each row's mean over its finite entries; 0 for a row with none."""
    finite = np.isfinite(values)
    count = finite.sum(axis=-1)
    total = np.where(finite, values, 0.0).sum(axis=-1)
    return np.where(count > 0, total / np.maximum(count, 1), 0.0)


def cell_day_t(values: np.ndarray) -> np.ndarray:
    """Each row's day-level t against zero over its finite entries.

    The same t as ``market_making.verdicts.day_t``; a row with fewer than two
    days, or with no dispersion, scores 0 (the null's statistic must be finite).
    """
    finite = np.isfinite(values)
    n = finite.sum(axis=-1)
    mean = cell_means(values)
    centred = np.where(finite, values - mean[..., None], 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        sd = np.sqrt((centred**2).sum(axis=-1) / (n - 1))
        t = mean / (sd / np.sqrt(n))
    return np.where((n >= 2) & np.isfinite(t), t, 0.0)


def neighbourhood_index(
    table: pd.DataFrame, axes: Sequence[str], by: Sequence[str] = ()
) -> np.ndarray:
    """Each cell's neighbourhood as a padded matrix of row positions (-1 pads).

    The block of ``validation/search.neighbourhood_scores``: every cell within
    one step on every ordered axis, diagonals and the cell itself included,
    among the cells sharing the cell's values in ``by`` (its categorical values
    and stratum). A step is to the adjacent value among those the axis takes in
    the cell's group. Axes must be numbers, never strings, so that a window of
    120 hours sorts after one of 4. An axis that is missing for every cell of a
    group is not an axis of that group; one missing for some is refused.
    """
    if not axes:
        raise ChoiceError("a neighbourhood needs at least one axis")
    missing = [c for c in [*axes, *by] if c not in table.columns]
    if missing:
        raise ChoiceError(f"columns {missing} are not in the table")
    for axis in axes:
        if not pd.api.types.is_numeric_dtype(table[axis]):
            raise ChoiceError(f"axis {axis!r} must be numeric, got {table[axis].dtype}")
    if table.duplicated([*by, *axes]).any():
        raise ChoiceError("two rows share one cell of the grid")
    groups = table.groupby(list(by), sort=False, dropna=False).indices if by else {(): None}
    lists: list[np.ndarray] = [np.empty(0, dtype=np.int64)] * len(table)
    for members in groups.values():
        rows = np.arange(len(table)) if members is None else np.asarray(members)
        steps = []
        for axis in axes:
            values = table[axis].to_numpy(dtype=np.float64)[rows]
            known = np.isfinite(values)
            if not known.any():
                continue
            if not known.all():
                raise ChoiceError(f"axis {axis!r} is missing for part of a group")
            steps.append(np.searchsorted(np.unique(values), values))
        position = np.column_stack(steps) if steps else np.zeros((len(rows), 1))
        for i, row in enumerate(rows):
            near = np.all(np.abs(position - position[i]) <= 1, axis=1)
            lists[row] = rows[near]
    width = max(len(a) for a in lists) if lists else 0
    out = np.full((len(table), width), -1, dtype=np.int64)
    for row, members_of in enumerate(lists):
        out[row, : len(members_of)] = members_of
    return out


def neighbourhood_median(values: np.ndarray, index: np.ndarray) -> np.ndarray:
    """The median of ``values`` over each cell's neighbourhood.

    ``values`` is (cells,) or (draws, cells); the result has its shape. A
    neighbour with a missing value does not contribute, and a cell whose own
    value is missing gets NaN, as in ``neighbourhood_scores``.
    """
    array = np.asarray(values, dtype=np.float64)
    padded = np.concatenate([array, np.full((*array.shape[:-1], 1), np.nan)], axis=-1)
    gathered = padded[..., np.where(index >= 0, index, array.shape[-1])]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = np.nanmedian(gathered, axis=-1)
    return np.where(np.isfinite(array), out, np.nan)


@dataclass(frozen=True)
class Choice:
    """The registered choice inside one stratum.

    ``cell`` is the chosen row (None when no cell averages enough trades) and
    ``peak`` the row with the highest S whatever its trade rate, reported
    beside it. ``s`` is S for every cell of the grid, not only the stratum's.
    """

    cell: int | None
    peak: int | None
    s: np.ndarray
    admissible: np.ndarray


def _best(candidates: np.ndarray, s: np.ndarray, own: np.ndarray) -> int | None:
    """Highest S, ties to the higher own mean, then to the earlier row."""
    usable = candidates[np.isfinite(s[candidates])]
    if not len(usable):
        return None
    order = np.lexsort((usable, -own[usable], -s[usable]))
    return int(usable[order[0]])


def choose(
    cells: CellDays,
    members: np.ndarray | Sequence[int],
    index: np.ndarray,
    min_trades_per_day: float = MIN_TRADES_PER_DAY,
) -> Choice:
    """The choice among ``members`` (one stratum's rows) on these days.

    S is the neighbourhood median of each cell's mean daily ``net_bp``, a cell
    without a trade scoring 0. A cell may be chosen only if its trades over
    the days, divided by the number of days, are at least
    ``min_trades_per_day``; the filter limits the choice, never the
    neighbours. The filter is not relaxed when no cell passes.
    """
    rows = np.asarray(members, dtype=np.int64)
    own = cell_means(cells.net_bp)
    s = neighbourhood_median(own, index)
    rate = cells.trades.sum(axis=1) / max(cells.days, 1)
    admissible = rate >= min_trades_per_day
    return Choice(_best(rows[admissible[rows]], s, own), _best(rows, s, own), s, admissible)


def statistic_a(cells: CellDays, index: np.ndarray, cell: int) -> float:
    """A: the median, over ``cell``'s neighbourhood, of each cell's day-level t
    of G, a cell without trades (or with fewer than two days) scoring 0."""
    near = index[cell][index[cell] >= 0]
    return float(np.median(cell_day_t(cells.g_bp[near])))


@dataclass(frozen=True)
class StratumOutcome:
    """One stratum's chosen cell and its A (0 when nothing was chosen)."""

    cell: int | None
    a: float


def rerun(
    cells: CellDays,
    strata: dict[str, np.ndarray],
    index: np.ndarray,
    fixed: dict[str, int] | None = None,
    min_trades_per_day: float = MIN_TRADES_PER_DAY,
) -> dict[str, StratumOutcome]:
    """The choice and A in every stratum, on one set of outcomes.

    A stratum named in ``fixed`` keeps that cell and makes no choice (a
    configuration fixed in advance, against the nulls with no selection). A
    stratum where no cell is admissible has no cell and A = 0, so that the
    maximum over strata of a null replication is still defined.
    """
    fixed = fixed or {}
    out: dict[str, StratumOutcome] = {}
    for name, members in strata.items():
        cell = fixed.get(name)
        if cell is None:
            cell = choose(cells, members, index, min_trades_per_day).cell
        out[name] = StratumOutcome(cell, 0.0 if cell is None else statistic_a(cells, index, cell))
    return out


#: The test months the carry-forward scores, and its embargo in days.
CARRY_FORWARD_MONTHS = ("2025-03", "2025-04", "2025-05", "2025-06")
CARRY_FORWARD_EMBARGO_DAYS = 5
#: The test months of condition (ii), and how many must be positive.
TEST_MONTHS = ("2025-02", *CARRY_FORWARD_MONTHS)
MONTHS_POSITIVE_REQUIRED = 4


def _stamps(dates: Sequence[object], days: int) -> pd.DatetimeIndex:
    stamps = pd.DatetimeIndex(np.asarray(list(dates), dtype="datetime64[ns]")).normalize()
    if len(stamps) != days:
        raise ChoiceError(f"{len(stamps)} dates for {days} days")
    return stamps


def _cell_mean(values: np.ndarray, cell: int, days: np.ndarray) -> float:
    return float(cell_means(values[cell : cell + 1, days])[0])


@dataclass(frozen=True)
class CarryForward:
    """Condition (iii): the choice re-run before each month, scored on it.

    A month whose re-run admits no cell, or whose chosen cell has no trade on
    the scored days, contributes 0 to the mean.
    """

    months: tuple[str, ...]
    cells: tuple[int | None, ...]
    g: tuple[float, ...]
    net: tuple[float, ...]

    @property
    def mean_g(self) -> float:
        return float(np.mean(self.g)) if self.g else float("nan")

    @property
    def mean_net(self) -> float:
        return float(np.mean(self.net)) if self.net else float("nan")

    @property
    def passes(self) -> bool:
        return bool(self.mean_g > 0)


def carry_forward(
    cells: CellDays,
    dates: Sequence[object],
    members: np.ndarray | Sequence[int],
    index: np.ndarray,
    months: Sequence[str] = CARRY_FORWARD_MONTHS,
    embargo_days: int = CARRY_FORWARD_EMBARGO_DAYS,
    min_trades_per_day: float = MIN_TRADES_PER_DAY,
) -> CarryForward:
    """For each month m, the choice on the test days before m, and the chosen
    cell's mean daily G and net on m's days from ``embargo_days`` after m's
    first day. ``dates`` gives each column's calendar day."""
    stamps = _stamps(dates, cells.days)
    chosen: list[int | None] = []
    g: list[float] = []
    net: list[float] = []
    for month in months:
        period = pd.Period(month, freq="M")
        start = period.start_time
        history = np.flatnonzero(stamps < start)
        scored = np.flatnonzero(
            (stamps >= start + np.timedelta64(int(embargo_days), "D")) & (stamps <= period.end_time)
        )
        cell = choose(cells.on_days(history), members, index, min_trades_per_day).cell
        chosen.append(cell)
        g.append(0.0 if cell is None else _cell_mean(cells.g_bp, cell, scored))
        net.append(0.0 if cell is None else _cell_mean(cells.net_bp, cell, scored))
    return CarryForward(tuple(months), tuple(chosen), tuple(g), tuple(net))


@dataclass(frozen=True)
class SlotConditions:
    """Conditions (i) and (ii) of slot admission for a chosen cell."""

    s: float
    monthly: tuple[float, ...]
    months_positive: int
    i: bool
    ii: bool


def slot_conditions(
    cells: CellDays,
    dates: Sequence[object],
    index: np.ndarray,
    cell: int,
    months: Sequence[str] = TEST_MONTHS,
    required: int = MONTHS_POSITIVE_REQUIRED,
) -> SlotConditions:
    """(i) the cell's S over all the days is > 0; (ii) in at least ``required``
    of ``months``, the neighbourhood median of each cell's mean daily net over
    that month's days is > 0. Pass the slot's own primary series as
    ``net_bp`` (with zero rows on days without a fill for a passive slot)."""
    stamps = _stamps(dates, cells.days)
    near = index[cell][index[cell] >= 0]
    s = float(np.median(cell_means(cells.net_bp[near])))
    monthly = []
    for month in months:
        days = np.flatnonzero(stamps.to_period("M") == pd.Period(month, freq="M"))
        monthly.append(float(np.median(cell_means(cells.net_bp[np.ix_(near, days)]))))
    positive = int(sum(value > 0 for value in monthly))
    return SlotConditions(s, tuple(monthly), positive, s > 0, positive >= required)


def not_admitted(failed: Sequence[str]) -> str:
    """The status of a slot that fails admission, naming every failed condition."""
    return f"not admitted on D ({', '.join(failed)})"


def pbo_of_choice(
    cells: CellDays,
    members: np.ndarray | Sequence[int],
    index: np.ndarray,
    blocks: int = CSCV_BLOCKS,
    min_trades_per_day: float = MIN_TRADES_PER_DAY,
) -> Pbo:
    """PBO of one stratum's choice by CSCV, with the identical rule.

    In each split the choice is made on the in-sample days among the cells
    admissible there, and ranked among those same cells by S computed on the
    out-of-sample days (the neighbourhood median of each cell's mean daily
    net, a cell without a trade scoring 0).
    """
    rows = np.asarray(members, dtype=np.int64)

    def pick(days: np.ndarray) -> tuple[int | None, np.ndarray]:
        found = choose(cells.on_days(days), rows, index, min_trades_per_day)
        return found.cell, rows[found.admissible[rows]]

    def score(days: np.ndarray) -> np.ndarray:
        return neighbourhood_median(cell_means(cells.on_days(days).net_bp), index)

    return cscv_pbo(cells.days, pick, score, blocks)
