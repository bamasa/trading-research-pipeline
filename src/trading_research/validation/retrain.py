"""How often to retrain, treated as a parameter rather than an assumption.

Every result so far fits a model once per fold and applies it to a whole test
week. That is one choice out of many, and not obviously the right one: a model
fitted on a fortnight and used for seven days is, by the seventh day, acting on
a fortnight that ended a week ago.

The alternative is to refit as you go — train on the last *W* days, act for the
next *A* days, move forward, repeat. Three numbers describe the whole scheme:

``train_days``
    How much history the fit sees. More is not automatically better: the market
    a month ago may be a different market, and a longer window buys sample size
    with staleness.
``apply_days``
    How long a fit is used before being replaced. One means daily retraining.
``step_days``
    How far the window moves each time. Equal to ``apply_days`` gives
    continuous coverage with no gaps and no overlap, which is what a live system
    does; smaller values overlap and are useful only for measurement.

None of the three has an obvious value, so all three are searched — on
validation, and only there. The test span is scored once with whatever the
search chose.

Why this belongs in the project rather than in a notebook
---------------------------------------------------------
Retraining frequency is the parameter most often left implicit in this kind of
work, and it interacts with everything else: a short window makes the feature
selection noisier, a long one makes the model staler, and either can dominate
the difference between two models. Leaving it fixed at one arbitrary value and
comparing models underneath is comparing them at one point of a surface nobody
looked at.

Reuse
-----
Nothing here knows about instruments, features or models. It is given a
callable that fits on one date range and scores another, and it walks the
schedule. Swapping the instrument means pointing at different prepared data;
swapping the model means passing a different factory.
"""

from __future__ import annotations

from collections.abc import Callable, Container, Iterator, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import product
from typing import Any

import pandas as pd


class ScheduleError(ValueError):
    """A retraining schedule cannot be built from the days available."""


@dataclass(frozen=True)
class RetrainSchedule:
    """Train on ``train_days``, act for ``apply_days``, move ``step_days``."""

    train_days: int
    apply_days: int = 1
    step_days: int | None = None

    def __post_init__(self) -> None:
        if self.train_days < 1:
            raise ValueError(f"train_days must be at least 1, got {self.train_days}")
        if self.apply_days < 1:
            raise ValueError(f"apply_days must be at least 1, got {self.apply_days}")
        if self.step_days is not None and self.step_days < 1:
            raise ValueError(f"step_days must be at least 1, got {self.step_days}")

    @property
    def step(self) -> int:
        """Days between refits. Defaults to ``apply_days``: no gaps, no overlap."""
        return self.step_days if self.step_days is not None else self.apply_days

    @property
    def label(self) -> str:
        return f"train{self.train_days}/apply{self.apply_days}/step{self.step}"

    def minimum_days(self) -> int:
        """Fewest days that produce a single train-and-apply pair."""
        return self.train_days + self.apply_days

    def windows(
        self, days: Sequence[date]
    ) -> Iterator[tuple[tuple[date, date], tuple[date, date]]]:
        """Yield ``((train_start, train_end), (apply_start, apply_end))`` pairs.

        Both ranges are inclusive. Training always ends the day before the apply
        window opens: a fit that included any part of the period it is scored on
        would be reporting its own training data back as a result.
        """
        ordered = sorted(days)
        if len(ordered) < self.minimum_days():
            raise ScheduleError(
                f"{self.label} needs {self.minimum_days()} days, {len(ordered)} available"
            )

        start = 0
        while start + self.train_days + self.apply_days <= len(ordered):
            train = (ordered[start], ordered[start + self.train_days - 1])
            apply_from = start + self.train_days
            apply_to = apply_from + self.apply_days - 1
            yield train, (ordered[apply_from], ordered[apply_to])
            start += self.step

    def count(self, days: Sequence[date]) -> int:
        """How many refits the schedule implies over ``days``."""
        try:
            return sum(1 for _ in self.windows(days))
        except ScheduleError:
            return 0


#: A grid wide enough to show the shape of the surface without being a search
#: for the best number. Retraining daily against weekly is the comparison that
#: matters; the difference between 13 and 14 training days is not.
DEFAULT_GRID: tuple[RetrainSchedule, ...] = (
    RetrainSchedule(train_days=3, apply_days=1),
    RetrainSchedule(train_days=7, apply_days=1),
    RetrainSchedule(train_days=14, apply_days=1),
    RetrainSchedule(train_days=21, apply_days=1),
    RetrainSchedule(train_days=7, apply_days=3),
    RetrainSchedule(train_days=14, apply_days=3),
    RetrainSchedule(train_days=14, apply_days=7),
    RetrainSchedule(train_days=21, apply_days=7),
)


#: Fits on one date range and scores another. Returns a mapping of metrics; the
#: search reads one key from it and ignores the rest.
#:
#: Deliberately the only thing this module knows about models or data — swapping
#: the instrument or the model means supplying a different one of these.
Evaluator = Callable[[tuple[date, date], tuple[date, date]], dict[str, float]]


@dataclass
class ScheduleResult:
    """What one schedule achieved over one span."""

    schedule: RetrainSchedule
    refits: int
    metrics: dict[str, float]
    per_window: list[dict[str, Any]]

    def as_row(self) -> dict[str, Any]:
        return {"schedule": self.schedule.label, "refits": self.refits, **self.metrics}


#: Combines per-window metrics into one row. Supplied by the caller because the
#: right way to combine them depends on what they mean: averaging a per-trade
#: figure across windows weights a window with one trade the same as a window
#: with thirty, which is how a schedule that barely trades comes top.
Aggregator = Callable[[list[dict[str, Any]]], dict[str, float]]


def mean_of_windows(windows: list[dict[str, Any]]) -> dict[str, float]:
    """Average every numeric column across windows. The neutral default.

    Correct only for quantities that are already per-window comparable. Anything
    expressed per trade needs :func:`trading_research.pipeline.retraining.
    trade_weighted` instead.
    """
    numeric = pd.DataFrame(windows).select_dtypes("number")
    return {name: float(numeric[name].mean()) for name in numeric.columns}


def run_schedule(
    schedule: RetrainSchedule,
    days: Sequence[date],
    evaluate: Evaluator,
    *,
    apply_within: Container[date] | None = None,
    aggregate: Aggregator = mean_of_windows,
) -> ScheduleResult:
    """Walk a schedule over ``days``, refitting and scoring at each step.

    ``apply_within`` restricts which windows count: the schedule still walks all
    of ``days``, but only windows whose traded period falls inside the given set
    are evaluated. That is how a test span is scored without losing its first
    ``train_days`` days — the training window reaches back into history, which
    is exactly what a live system does on its first day.

    Windows that the evaluator cannot score — too few usable rows, a class
    missing from the training block — are skipped and counted rather than
    aborting the run. A schedule that fails half its windows is a finding about
    that schedule, and losing the other half to an exception hides it.
    """
    rows: list[dict[str, Any]] = []
    for train, apply in schedule.windows(days):
        if apply_within is not None and not (apply[0] in apply_within and apply[1] in apply_within):
            continue
        try:
            metrics = evaluate(train, apply)
        except Exception as exc:
            rows.append(
                {
                    "train_start": train[0],
                    "train_end": train[1],
                    "apply_start": apply[0],
                    "apply_end": apply[1],
                    "skipped": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        rows.append(
            {
                "train_start": train[0],
                "train_end": train[1],
                "apply_start": apply[0],
                "apply_end": apply[1],
                **metrics,
            }
        )

    scored = [r for r in rows if "skipped" not in r]
    if not scored:
        raise ScheduleError(f"{schedule.label}: every window failed to score")

    aggregated = aggregate(scored)
    # How many windows the average rests on: a schedule with a good figure over
    # three windows and one with the same figure over thirty are not equivalent,
    # and the figure alone hides which is which.
    aggregated["windows"] = float(len(scored))
    aggregated["windows_skipped"] = float(len(rows) - len(scored))

    return ScheduleResult(
        schedule=schedule,
        refits=len(rows),
        metrics=aggregated,
        per_window=rows,
    )


@dataclass
class SearchOutcome:
    """The chosen schedule, the evidence behind it, and the test score."""

    chosen: RetrainSchedule
    objective: str
    validation: pd.DataFrame
    test: ScheduleResult | None = None

    def summary(self) -> str:
        line = f"chose {self.chosen.label} on validation by {self.objective}"
        if self.test is not None:
            value = self.test.metrics.get(self.objective)
            line += f"; test {self.objective} = {value:.3f}"
        return line


def search(
    validation_days: Sequence[date],
    evaluate: Evaluator,
    *,
    grid: Sequence[RetrainSchedule] = DEFAULT_GRID,
    objective: str = "net_per_trade_bp",
    minimum_windows: int = 3,
    test_days: Sequence[date] | None = None,
    history: Sequence[date] | None = None,
    aggregate: Aggregator = mean_of_windows,
    on_result: Callable[[ScheduleResult], None] | None = None,
) -> SearchOutcome:
    """Choose a retraining schedule on validation, then score it once on test.

    ``minimum_windows`` guards against a schedule that looks best because it was
    measured over two windows. A long training window over a short validation
    span produces very few refits, and the fewer there are the more the average
    is one lucky window.

    ``history`` is days before validation that may be trained on but are never
    scored — normally the days that chose the features. Without them a 21-day
    window costs three weeks of validation before it trades at all, so long and
    short schedules would be compared over different periods and the comparison
    would be about span length rather than about staleness.

    ``test_days`` is scored with the chosen schedule and nothing else. It is
    never consulted while choosing.
    """
    rows: list[dict[str, Any]] = []
    results: dict[str, ScheduleResult] = {}
    scored_days = set(validation_days)
    lead = list(history) if history is not None else []

    for schedule in grid:
        span = _with_warmup(lead, validation_days, schedule.train_days)
        windows = sum(
            1
            for _, apply in _safe_windows(schedule, span)
            if apply[0] in scored_days and apply[1] in scored_days
        )
        if windows < minimum_windows:
            rows.append(
                {
                    "schedule": schedule.label,
                    "refits": windows,
                    "skipped": f"fewer than {minimum_windows} windows",
                }
            )
            continue
        try:
            result = run_schedule(
                schedule, span, evaluate, apply_within=scored_days, aggregate=aggregate
            )
        except ScheduleError as exc:
            rows.append({"schedule": schedule.label, "refits": 0, "skipped": str(exc)})
            continue

        results[schedule.label] = result
        rows.append(result.as_row())
        if on_result is not None:
            on_result(result)

    table = pd.DataFrame(rows)
    scorable = [r for label, r in results.items() if objective in r.metrics]
    if not scorable:
        raise ScheduleError(f"no schedule produced {objective!r} on validation")

    best = max(scorable, key=lambda r: r.metrics[objective])

    test_result = None
    if test_days is not None:
        test_result = run_schedule(
            best.schedule,
            _with_warmup([*lead, *validation_days], test_days, best.schedule.train_days),
            evaluate,
            apply_within=set(test_days),
            aggregate=aggregate,
        )

    return SearchOutcome(
        chosen=best.schedule,
        objective=objective,
        validation=table,
        test=test_result,
    )


def _safe_windows(
    schedule: RetrainSchedule, days: Sequence[date]
) -> list[tuple[tuple[date, date], tuple[date, date]]]:
    """``schedule.windows``, returning nothing rather than raising on a short span."""
    try:
        return list(schedule.windows(days))
    except ScheduleError:
        return []


def _with_warmup(
    history: Sequence[date],
    span: Sequence[date],
    train_days: int,
) -> list[date]:
    """Prepend enough earlier days for the first window of ``span`` to be traded.

    Without this the first ``train_days`` of a test span are consumed training a
    model that is then scored on what is left, which both shortens the test and
    makes its length depend on the schedule chosen — so two schedules would be
    compared over different amounts of test data.

    Using validation days to *train* on before trading the test period is not a
    leak. They are in the past relative to every day being traded, and it is
    what a live system does on the morning it starts.
    """
    ordered = sorted(span)
    lead = [d for d in sorted(history) if d < ordered[0]]
    return lead[-train_days:] + ordered


def split_span(
    days: Sequence[date],
    *,
    validation_fraction: float = 0.5,
) -> tuple[list[date], list[date]]:
    """Split a run of days into a validation span and a test span, in order.

    Chronological, and with a gap of exactly nothing: the schedules themselves
    keep training strictly before applying, so the only boundary that needs
    protecting is this one — and a schedule beginning on the first test day
    trains on validation days, which is allowed. Training on data from the past
    is what a live system does.
    """
    ordered = sorted(days)
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(f"validation_fraction must be in (0, 1), got {validation_fraction}")
    cut = int(len(ordered) * validation_fraction)
    if cut < 2 or len(ordered) - cut < 2:
        raise ScheduleError(f"{len(ordered)} days cannot be split into two usable spans")
    return ordered[:cut], ordered[cut:]


def days_between(start: date, end: date) -> list[date]:
    """Every day in ``[start, end]`` inclusive."""
    if end < start:
        raise ValueError(f"end ({end}) is before start ({start})")
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def describe_grid(grid: Sequence[RetrainSchedule] = DEFAULT_GRID) -> pd.DataFrame:
    """Tabulate a grid, for the run manifest."""
    return pd.DataFrame(
        [
            {
                "label": s.label,
                "train_days": s.train_days,
                "apply_days": s.apply_days,
                "step_days": s.step,
                "minimum_days": s.minimum_days(),
            }
            for s in grid
        ]
    )


def expand_grid(
    train_days: Sequence[int],
    apply_days: Sequence[int],
    step_days: Sequence[int | None] = (None,),
) -> tuple[RetrainSchedule, ...]:
    """Build a grid from the values to try, skipping impossible combinations."""
    out = []
    for train, apply, step in product(train_days, apply_days, step_days):
        out.append(RetrainSchedule(train_days=train, apply_days=apply, step_days=step))
    return tuple(out)
