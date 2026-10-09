"""Regime flags for the guard: the hourly break detector, causal fold by fold.

The guard after a regime break needs flag times, and a flag time is only
honest if it was known when the guard acted on it. This module turns hourly
bar closes into those times with :func:`~.structural_breaks.detect_breaks`,
in the two forms the study uses:

**Fold-wise, inside the walk-forward.** For a test month ``m`` the 24-value
hour-of-day profile is fitted on the hours from the stream's start to the last
hour before ``m``; one ``detect_breaks`` call runs over the stream from its
start to ``m``'s last hour; and the guard uses that run's alarms from 120 hours
before ``m`` (the longest guard window, so a flag just before the month still
guards its first days) to ``m``'s end. Nothing in a fold's flags can have read
a bar inside ``m`` that came after the flag: the profile stops before ``m`` and
the walk is causal.

**Frozen, for the held-out blocks.** The profile fitted once on the whole
development period, one run over the stream to the last opened hour.

The returns are divided by the root mean square of their UTC hour so that the
daily cycle of volatility is not read as a series of breaks. A missing hour
has its close carried forward, so its return is zero and the move over the
gap enters the next return; missing hours are counted, never repaired.

The guarantee is tested, not asserted: :func:`check_flag_causality` mutates
every bar after each hour of the test month and requires every flag at or
before that hour to stay as it was, on the pattern of
:mod:`trading_research.validation.leakage`.

Time conventions
----------------
A return is labelled by the close that ends it, on the hour boundary (a
Binance close time of ``00:59:59.999`` is the close at ``01:00``), and its UTC
hour is the hour its bar spans, so the return labelled ``01:00`` belongs to
hour 0. An alarm is effective at the label of the return that produced it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading_research.validation.structural_breaks import (
    MonitorResult,
    MonitorSpec,
    detect_breaks,
    log_returns,
)

#: The detector stream's first bar: the monitor is warm a month later.
STREAM_START = pd.Timestamp("2024-12-01", tz="UTC")
HOUR = pd.Timedelta(1, unit="h")
NS_PER_HOUR = 3_600 * 1_000_000_000

#: The monitor as registered, every length in hours.
DETECTOR_SPEC = MonitorSpec(
    history_len=365,
    online_len=90,
    min_history=121,
    statistics=("scale", "dependence"),
    minimum_gap=0,
)

#: The longest guard window. Alarms this long before a test month still guard it.
LEAD_IN_HOURS = 120

#: The span whose hours give the frozen profile: the development period.
FROZEN_PROFILE_SPAN = (pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-07-01", tz="UTC"))


class FlagError(ValueError):
    """The bars cannot give a flag stream as the registration defines it."""


def _utc(moment: pd.Timestamp | str) -> pd.Timestamp:
    stamp = pd.Timestamp(moment)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _ns(stamps: pd.Series | pd.Index) -> np.ndarray:
    """Timestamps as int64 nanoseconds since the epoch, UTC."""
    index = pd.DatetimeIndex(pd.to_datetime(stamps, utc=True))
    return index.tz_convert("UTC").tz_localize(None).as_unit("ns").to_numpy().astype(np.int64)


@dataclass(frozen=True)
class HourlyStream:
    """Hourly closes on a complete grid, and the hours that had to be filled."""

    close: pd.Series
    """One close per hour, labelled by the hour boundary that ends the bar."""
    missing: pd.DatetimeIndex
    """Labels whose bar was absent; their close is the previous one."""


def hourly_stream(
    bars: pd.DataFrame | pd.Series,
    *,
    end: pd.Timestamp | str,
    start: pd.Timestamp | str = STREAM_START,
) -> HourlyStream:
    """The closes of the bars from ``start`` to ``end`` on a complete hourly grid.

    ``bars`` is a frame with ``timestamp`` (the bar's close) and ``close``, as
    :func:`trading_research.data.binance.parse_klines` writes it, or a close
    series indexed by that timestamp. The grid's labels run from ``start`` plus
    one hour (the first bar's close) to ``end``. The first bar must be present:
    there is nothing to carry forward into it.
    """
    start, end = _utc(start), _utc(end)
    if isinstance(bars, pd.DataFrame):
        missing_columns = {"timestamp", "close"} - set(bars.columns)
        if missing_columns:
            raise FlagError(f"bars need 'timestamp' and 'close'; missing {sorted(missing_columns)}")
        series = pd.Series(
            bars["close"].to_numpy(dtype="float64"), index=pd.DatetimeIndex(bars["timestamp"])
        )
    else:
        series = pd.Series(bars.to_numpy(dtype="float64"), index=pd.DatetimeIndex(bars.index))
    labels = pd.DatetimeIndex(pd.to_datetime(series.index, utc=True))
    series.index = labels.ceil("h")
    if series.index.has_duplicates:
        raise FlagError("two bars close in the same hour")
    if not np.isfinite(series.to_numpy()).all() or (series.to_numpy() <= 0).any():
        raise FlagError("closes must be finite and positive")
    grid = pd.date_range(start + HOUR, end, freq="h")
    if len(grid) < 2:
        raise FlagError(f"the span {start} to {end} holds fewer than two hourly closes")
    on_grid = series.reindex(grid)
    absent = on_grid.isna()
    if absent.iloc[0]:
        raise FlagError(f"the first bar of the stream (closing {grid[0]}) is missing")
    close = on_grid.ffill().rename("close")
    return HourlyStream(close=close, missing=pd.DatetimeIndex(grid[absent.to_numpy()]))


def hours_of_day(labels: pd.DatetimeIndex) -> np.ndarray:
    """The UTC hour each return's bar spans: the hour before its closing label."""
    return ((_ns(pd.DatetimeIndex(labels)) - 1) // NS_PER_HOUR) % 24


def fit_hour_profile(
    returns: pd.Series, start: pd.Timestamp | str, end: pd.Timestamp | str
) -> np.ndarray:
    """Root mean square of the returns at each UTC hour, over the bars in ``[start, end)``.

    A bar is inside when it opens at or after ``start`` and closes at or before
    ``end``, so a profile fitted to a test month's first instant reads no bar of
    that month. Every hour of the day must have a positive value.
    """
    start, end = _utc(start), _utc(end)
    labels = pd.DatetimeIndex(returns.index)
    inside = np.asarray((labels - HOUR >= start) & (labels <= end))
    values = returns.to_numpy(dtype="float64")[inside]
    hours = hours_of_day(labels[inside])
    profile = np.full(24, np.nan)
    for hour in range(24):
        at = values[hours == hour]
        if len(at):
            profile[hour] = float(np.sqrt(np.mean(at * at)))
    if not (np.isfinite(profile) & (profile > 0)).all():
        empty = np.flatnonzero(~(np.isfinite(profile) & (profile > 0))).tolist()
        raise FlagError(f"no usable returns at UTC hour(s) {empty} between {start} and {end}")
    return profile


def normalise_by_hour(returns: pd.Series, profile: np.ndarray) -> pd.Series:
    """Each return divided by its UTC hour's value in ``profile``."""
    profile = np.asarray(profile, dtype="float64")
    if profile.shape != (24,) or not (profile > 0).all():
        raise FlagError("an hour profile is 24 positive numbers")
    hours = hours_of_day(pd.DatetimeIndex(returns.index))
    return (returns / profile[hours]).rename("normalised_return")


#: Columns of every alarm table this module returns.
ALARM_COLUMNS: tuple[str, ...] = ("timestamp", "statistic", "direction", "excursion", "odds_mean")


def alarm_table(result: MonitorResult) -> pd.DataFrame:
    """A run's alarms: when each became effective, what moved, and the mean channel then.

    ``odds_mean`` is read from the same run at the alarm's step; the mean
    channel is reported beside the alarm and never defines one.
    """
    frame = result.frame
    rows = []
    for brk in result.breaks:
        at = frame.loc[frame["index"] == brk.index].iloc[-1]
        rows.append(
            {
                "timestamp": pd.Timestamp(at["timestamp"]),
                "statistic": brk.statistic,
                "direction": int(brk.direction),
                "excursion": float(brk.excursion),
                "odds_mean": float(at["odds_mean"]),
            }
        )
    table = pd.DataFrame(rows, columns=list(ALARM_COLUMNS))
    table["timestamp"] = pd.to_datetime(table["timestamp"], utc=True)
    return table


def run_detector(
    close: pd.Series,
    profile: np.ndarray,
    *,
    end: pd.Timestamp | str,
    spec: MonitorSpec = DETECTOR_SPEC,
) -> MonitorResult:
    """One ``detect_breaks`` call over the stream's returns up to ``end``.

    The windows are anchored at the stream's first return, so every run over a
    longer stream walks the same windows as a shorter one up to its end.
    """
    end = _utc(end)
    kept = close[close.index <= end]
    returns = log_returns(kept)
    return detect_breaks(normalise_by_hour(returns, profile), spec, on_progress=None)


def flag_times(alarms: pd.DataFrame) -> np.ndarray:
    """Alarm times as int64 nanoseconds since the epoch, what the guard reads."""
    return _ns(alarms["timestamp"])


@dataclass(frozen=True)
class Fold:
    """One test month of the walk-forward: ``[start, end)``, both on hour boundaries."""

    start: pd.Timestamp
    end: pd.Timestamp
    lead_in_hours: int = LEAD_IN_HOURS

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", _utc(self.start))
        object.__setattr__(self, "end", _utc(self.end))
        if self.end <= self.start:
            raise FlagError(f"a fold must end after it starts: {self.start} to {self.end}")
        for moment in (self.start, self.end):
            if moment != moment.floor("h"):
                raise FlagError(f"fold bounds lie on hour boundaries, got {moment}")

    @property
    def name(self) -> str:
        return f"{self.start:%Y-%m}"

    @property
    def guard_from(self) -> pd.Timestamp:
        """The earliest alarm the guard in this month uses."""
        return self.start - self.lead_in_hours * HOUR

    @property
    def hours(self) -> pd.DatetimeIndex:
        """Every hour of the month, by its opening instant: the τ of the causality test."""
        return pd.date_range(self.start, self.end - HOUR, freq="h")


def walk_forward_months(first: str = "2025-02", last: str = "2025-06") -> list[Fold]:
    """The walk-forward's test months as folds, February to June by default."""
    starts = pd.date_range(_utc(f"{first}-01"), _utc(f"{last}-01"), freq="MS")
    return [Fold(start, start + pd.offsets.MonthBegin(1)) for start in starts]


@dataclass(frozen=True)
class FoldFlags:
    """A fold's run: its hour profile and every alarm, and the ones its guard uses."""

    fold: Fold
    profile: np.ndarray
    alarms: pd.DataFrame

    @property
    def guard_alarms(self) -> pd.DataFrame:
        """Alarms in ``[start - lead-in, end]``, which the guard in this month reads."""
        stamps = self.alarms["timestamp"]
        keep = (stamps >= self.fold.guard_from) & (stamps <= self.fold.end)
        return self.alarms[keep.to_numpy()].reset_index(drop=True)

    @property
    def month_alarms(self) -> pd.DataFrame:
        """Alarms inside the month itself, ``[start, end)``: where the regime map cuts."""
        stamps = self.alarms["timestamp"]
        keep = (stamps >= self.fold.start) & (stamps < self.fold.end)
        return self.alarms[keep.to_numpy()].reset_index(drop=True)

    def flags_ns(self) -> np.ndarray:
        return flag_times(self.guard_alarms)


def fold_flags(
    close: pd.Series,
    fold: Fold,
    *,
    stream_start: pd.Timestamp | str = STREAM_START,
    spec: MonitorSpec = DETECTOR_SPEC,
) -> FoldFlags:
    """Fold ``fold``'s flags from hourly closes, as :func:`hourly_stream` lays them out.

    The profile is fitted on the bars from ``stream_start`` to the fold's first
    instant, and one run covers the stream to the fold's last hour.
    """
    returns = log_returns(close[close.index <= fold.end])
    profile = fit_hour_profile(returns, stream_start, fold.start)
    result = run_detector(close, profile, end=fold.end, spec=spec)
    return FoldFlags(fold, profile, alarm_table(result))


def frozen_flags(
    close: pd.Series,
    *,
    end: pd.Timestamp | str,
    profile: np.ndarray | None = None,
    profile_span: tuple[pd.Timestamp, pd.Timestamp] = FROZEN_PROFILE_SPAN,
    spec: MonitorSpec = DETECTOR_SPEC,
) -> tuple[np.ndarray, pd.DataFrame]:
    """The frozen detector's profile and alarms over the stream to ``end``.

    ``profile`` is the recorded one; without it the profile is fitted on the
    bars of ``profile_span`` (the development period), which is how it is
    recorded in the first place.
    """
    if profile is None:
        returns = log_returns(close[close.index <= _utc(profile_span[1])])
        profile = fit_hour_profile(returns, *profile_span)
    return profile, alarm_table(run_detector(close, profile, end=end, spec=spec))


def hourly_closes(prices: pd.Series) -> pd.Series:
    """The last price of each hour of a finer series, labelled by the hour's close."""
    stamps = pd.DatetimeIndex(pd.to_datetime(prices.index, utc=True))
    values = pd.Series(prices.to_numpy(dtype="float64"), index=stamps).sort_index()
    labels = pd.DatetimeIndex(values.index).ceil("h")
    return values.groupby(labels).last().rename("close")


def flags_from_prices(
    prices: pd.Series, *, profile_end: pd.Timestamp | str, spec: MonitorSpec = DETECTOR_SPEC
) -> tuple[np.ndarray, pd.DataFrame]:
    """The detector on any price series: hourly closes, a profile fitted before ``profile_end``.

    For a search outside the registered study: the profile is fitted on the
    hours before ``profile_end`` (the end of the block the search may see),
    one run covers the whole series, and the alarms after ``profile_end`` were
    computed with nothing fitted on their hours.
    """
    closes = hourly_closes(prices)
    first = closes.index[0] - HOUR
    stream = hourly_stream(closes, start=first, end=closes.index[-1])
    profile = fit_hour_profile(log_returns(stream.close), first, profile_end)
    result = run_detector(stream.close, profile, end=stream.close.index[-1], spec=spec)
    return profile, alarm_table(result)


# ---------------------------------------------------------------------------
# The flag-causality test
# ---------------------------------------------------------------------------


class FlagLeakError(AssertionError):
    """A flag the guard uses changed when bars after it were changed."""


@dataclass(frozen=True)
class FoldRunner:
    """The fold-wise flags as a function of closes and fold: what the test mutates.

    A dataclass rather than a closure so that the causality test can hand it
    to worker processes.
    """

    stream_start: pd.Timestamp = STREAM_START
    spec: MonitorSpec = DETECTOR_SPEC

    def __call__(self, close: pd.Series, fold: Fold) -> pd.DataFrame:
        return fold_flags(close, fold, stream_start=self.stream_start, spec=self.spec).alarms


#: Anything mapping (closes, fold) to an alarm table with ``timestamp``.
FoldRun = Callable[[pd.Series, Fold], pd.DataFrame]

Flag = tuple[int, str, int]


def _flags_through(alarms: pd.DataFrame, since: pd.Timestamp, until: pd.Timestamp) -> list[Flag]:
    stamps = pd.to_datetime(alarms["timestamp"], utc=True)
    keep = ((stamps >= since) & (stamps <= until)).to_numpy()
    chosen = alarms[keep]
    times = _ns(chosen["timestamp"])
    statistics = chosen["statistic"].astype(str) if "statistic" in chosen else [""] * len(chosen)
    directions = chosen["direction"] if "direction" in chosen else [0] * len(chosen)
    return [(int(t), str(s), int(d)) for t, s, d in zip(times, statistics, directions, strict=True)]


def mutate_after(close: pd.Series, tau: pd.Timestamp, rng: np.random.Generator) -> pd.Series:
    """``close`` with every bar after ``tau`` replaced by a random walk ten times as rough.

    The walk starts from the last close at or before ``tau``, so the stream
    stays positive and continuous; a causal flag cannot tell the difference
    before ``tau``, and nearly any look-ahead can.
    """
    after = np.asarray(close.index > tau)
    if not after.any():
        return close.copy()
    before = close[~after]
    base = float(before.iloc[-1]) if len(before) else float(close.iloc[0])
    steps = np.diff(np.log(before.to_numpy(dtype="float64")))
    rough = 10.0 * float(steps.std()) if len(steps) > 1 and steps.std() > 0 else 0.05
    walk = base * np.exp(np.cumsum(rng.normal(0.0, rough, int(after.sum()))))
    out = close.copy()
    out[after] = walk
    return out


@dataclass(frozen=True)
class CausalityFailure:
    """One hour at which a used flag moved: the flags before and after the mutation."""

    tau: pd.Timestamp
    original: tuple[Flag, ...]
    mutated: tuple[Flag, ...]


@dataclass(frozen=True)
class CausalityResult:
    """The flag-causality test on one fold."""

    fold: Fold
    hours_checked: int
    failures: tuple[CausalityFailure, ...]

    @property
    def leaks(self) -> bool:
        return bool(self.failures)

    def __str__(self) -> str:
        if not self.leaks:
            return f"fold {self.fold.name}: flags causal over {self.hours_checked} hours"
        first = self.failures[0]
        return (
            f"fold {self.fold.name}: flags LEAK at {len(self.failures)} of "
            f"{self.hours_checked} hours; first at {first.tau}: "
            f"{len(first.original)} flag(s) before, {len(first.mutated)} after mutation"
        )


def _check_hours(
    payload: tuple[pd.Series, Fold, FoldRun, list[tuple[int, pd.Timestamp]], int, pd.DataFrame],
) -> list[CausalityFailure]:
    close, fold, run, hours, seed, alarms = payload
    failures = []
    for position, tau in hours:
        original = tuple(_flags_through(alarms, fold.guard_from, tau))
        rng = np.random.default_rng([seed, position])
        mutated_alarms = run(mutate_after(close, tau, rng), fold)
        mutated = tuple(_flags_through(mutated_alarms, fold.guard_from, tau))
        if mutated != original:
            failures.append(CausalityFailure(tau, original, mutated))
    return failures


def check_flag_causality(
    close: pd.Series,
    fold: Fold,
    *,
    run: FoldRun | None = None,
    hours: Sequence[pd.Timestamp] | None = None,
    seed: int = 0,
    workers: int = 1,
) -> CausalityResult:
    """For every hour τ of the fold, mutate every bar after τ and re-run the fold.

    Passes when every flag the guard uses at or before τ (from the lead-in's
    start to τ) is unchanged, by time, statistic and direction, for every τ.
    The flags of the lead-in may use bars up to the fold's start, so the test
    begins at the fold's first hour. ``hours`` restricts τ (for a quick check;
    the registration checks them all); ``workers`` > 1 spreads the hours over
    processes and gives the same answer, the mutation of each hour being
    seeded by ``seed`` and its position in the fold.
    """
    runner: FoldRun = run if run is not None else FoldRunner()
    taus = list(fold.hours) if hours is None else [_utc(t) for t in hours]
    positions = {t: i for i, t in enumerate(fold.hours)}
    numbered = [(positions.get(t, len(positions) + i), t) for i, t in enumerate(taus)]
    alarms = runner(close, fold)
    if workers <= 1 or len(numbered) < 2:
        failures = _check_hours((close, fold, runner, numbered, seed, alarms))
    else:
        from concurrent.futures import ProcessPoolExecutor

        chunks = [numbered[i::workers] for i in range(workers)]
        payloads = [(close, fold, runner, chunk, seed, alarms) for chunk in chunks if chunk]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            failures = [f for part in pool.map(_check_hours, payloads) for f in part]
        failures.sort(key=lambda f: f.tau)
    return CausalityResult(fold, len(numbered), tuple(failures))


def assert_flags_causal(
    close: pd.Series,
    folds: Sequence[Fold],
    *,
    run: FoldRun | None = None,
    seed: int = 0,
    workers: int = 1,
) -> list[CausalityResult]:
    """Run the test over every hour of every fold; raise if any leaks (K-flag-leak)."""
    results = [
        check_flag_causality(close, fold, run=run, seed=seed, workers=workers) for fold in folds
    ]
    leaking = [r for r in results if r.leaks]
    if leaking:
        detail = "\n".join(f"  {r}" for r in leaking)
        raise FlagLeakError(f"{len(leaking)} fold(s) use flags that read later bars:\n{detail}")
    return results


# ---------------------------------------------------------------------------
# What is recorded beside the flags
# ---------------------------------------------------------------------------


def alarm_rate(alarms: pd.DataFrame, start: pd.Timestamp | str, end: pd.Timestamp | str) -> float:
    """Alarms per day effective in ``[start, end)``."""
    start, end = _utc(start), _utc(end)
    days = (end - start) / pd.Timedelta(1, unit="D")
    if days <= 0:
        raise FlagError(f"an alarm rate needs a span, got {start} to {end}")
    stamps = pd.to_datetime(alarms["timestamp"], utc=True)
    return float(((stamps >= start) & (stamps < end)).sum()) / days


def unreproduced_alarms(
    earlier: pd.DataFrame, later: pd.DataFrame, *, through: pd.Timestamp | str
) -> pd.DataFrame:
    """Alarms up to ``through`` that one run has and the other does not.

    A causal walk over a longer stream repeats every alarm of a shorter one, by
    time and statistic; an empty frame is the reproduction check passing. Each
    row says which run had the alarm.
    """
    through = _utc(through)

    def keyed(alarms: pd.DataFrame) -> set[tuple[int, str]]:
        stamps = pd.to_datetime(alarms["timestamp"], utc=True)
        kept = alarms[(stamps <= through).to_numpy()]
        times = _ns(kept["timestamp"])
        return {(int(t), str(s)) for t, s in zip(times, kept["statistic"], strict=True)}

    first, second = keyed(earlier), keyed(later)
    rows = [{"timestamp": t, "statistic": s, "run": "earlier only"} for t, s in first - second]
    rows += [{"timestamp": t, "statistic": s, "run": "later only"} for t, s in second - first]
    out = pd.DataFrame(rows, columns=["timestamp", "statistic", "run"])
    out["timestamp"] = pd.to_datetime(out["timestamp"].astype("int64"), utc=True)
    return out.sort_values("timestamp").reset_index(drop=True)


def foldwise_against_frozen(folds: Sequence[FoldFlags], frozen: pd.DataFrame) -> pd.DataFrame:
    """Per test month, the alarms both lists hold and those only one does.

    The fold-wise flags are what the guard used on the development period, the
    frozen ones what the held-out blocks use; their difference is reported,
    never corrected.
    """
    frozen_times = pd.to_datetime(frozen["timestamp"], utc=True)
    rows = []
    for flags in folds:
        fold = flags.fold
        mine = set(_ns(flags.month_alarms["timestamp"]).tolist())
        inside = (frozen_times >= fold.start) & (frozen_times < fold.end)
        theirs = set(_ns(frozen_times[inside.to_numpy()]).tolist())
        rows.append(
            {
                "month": fold.name,
                "both": len(mine & theirs),
                "foldwise_only": len(mine - theirs),
                "frozen_only": len(theirs - mine),
            }
        )
    return pd.DataFrame(rows, columns=["month", "both", "foldwise_only", "frozen_only"])
