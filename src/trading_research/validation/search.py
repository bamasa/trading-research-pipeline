"""Searching a configuration space without evaluating all of it.

The full configuration here — when to trade, how much history, how often to
refit, how long to hold, how long to wait, how selective to be — is a few
hundred combinations, and each one costs a fit per window. Evaluating the
product takes hours and is the wrong shape of effort besides: most candidates
are obviously poor after two windows, and spending twenty on them buys nothing.

Two ideas do almost all the work.

**Successive halving.** Score every candidate on a small number of windows,
keep the best fraction, score those on more, repeat. A candidate that survives
to the last round has been measured on the full span; one that dies in the
first cost a fifth of that. Total spend is roughly the cost of evaluating a
handful of candidates properly instead of all of them badly.

**Sampling rather than enumerating.** A grid spends its budget in proportion to
how finely each axis was divided, which is a statement about the person who
wrote the grid rather than about the problem. Random configurations spread the
same budget over the space and find the good regions of the axes that matter.

Neither is exotic; both are standard hyperparameter practice. They are here
because the alternative in this repository was a nested product over six
dimensions, which is how a search ends up measuring the size of its own grid.

A caution that applies to any search, and to this one twice over: every extra
candidate is another chance to fit the validation block. Halving reduces the
*cost* of a wide search, not its capacity to overfit — if anything it makes
widening easier, so the number of candidates should be chosen for what the
validation span can support and not for what the budget allows.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


class SearchError(ValueError):
    """No candidate could be scored."""


#: Scores one candidate over a given number of windows and returns its metrics.
#: The second argument is how much evidence to gather — larger is slower and
#: more reliable — and what it means is the caller's business.
Scorer = Callable[[Any, int], dict[str, float]]

#: Scores every candidate of a rung at once, in the order given, so the caller
#: can spread the work over processes. An entry may be an exception instead of
#: metrics: that candidate is skipped, as a :data:`Scorer` raising would be.
BatchScorer = Callable[[Sequence[Any], int], Sequence["dict[str, float] | BaseException"]]


@dataclass
class Rung:
    """One round of successive halving."""

    budget: int
    candidates: int
    survivors: int
    results: list[tuple[Any, dict[str, float]]] = field(default_factory=list)


@dataclass
class SearchOutcome:
    """The winner, every candidate that was tried, and the shape of the search."""

    best: Any
    best_metrics: dict[str, float]
    objective: str
    table: pd.DataFrame
    rungs: list[Rung]

    @property
    def evaluations(self) -> int:
        return int(sum(len(r.results) for r in self.rungs))

    def summary(self) -> str:
        spent = sum(len(r.results) * r.budget for r in self.rungs)
        exhaustive = len(self.rungs[0].results) * self.rungs[-1].budget if self.rungs else 0
        saved = 1 - spent / exhaustive if exhaustive else 0.0
        return (
            f"{self.evaluations} evaluations over {len(self.rungs)} rungs, "
            f"{saved:.0%} cheaper than scoring every candidate at full budget; "
            f"best {self.objective} = {self.best_metrics.get(self.objective, float('nan')):.3f}"
        )


def successive_halving(
    candidates: Sequence[Any],
    score: Scorer | None = None,
    *,
    score_batch: BatchScorer | None = None,
    objective: str = "net_per_trade_bp",
    budgets: Sequence[int] = (4, 8, 24),
    keep_fraction: float = 0.34,
    minimum_trades: float = 50.0,
    label: Callable[[Any], str] | None = None,
    on_rung: Callable[[Rung], None] | None = None,
) -> SearchOutcome:
    """Run the candidates through rounds of increasing evidence.

    ``budgets`` is how much evidence each round gathers — for a retraining
    search, the number of windows. The last budget should be the full span, so
    that whatever wins was measured on everything.

    ``minimum_trades`` drops candidates that barely traded before they are
    ranked. Without it the early rounds promote whichever candidate happened to
    take three lucky trades in four windows, and the search spends its budget on
    noise.

    Give either ``score``, called once per candidate, or ``score_batch``, called
    once per rung with every candidate still alive; the search is the same
    either way, and the batch form lets the scoring run in parallel.
    """
    if not candidates:
        raise SearchError("no candidates to search")
    if not budgets:
        raise SearchError("no budgets given")
    if (score is None) == (score_batch is None):
        raise SearchError("give exactly one of score and score_batch")

    name = label or (lambda c: str(c))
    alive = list(candidates)
    rungs: list[Rung] = []
    rows: list[dict[str, Any]] = []

    for depth, budget in enumerate(budgets):
        survivors = max(1, int(len(alive) * keep_fraction)) if depth + 1 < len(budgets) else 1
        rung = Rung(budget=budget, candidates=len(alive), survivors=survivors)

        outcomes: list[dict[str, float] | BaseException]
        if score_batch is not None:
            outcomes = list(score_batch(alive, budget))
            if len(outcomes) != len(alive):
                raise SearchError(
                    f"the batch scorer returned {len(outcomes)} results for {len(alive)} candidates"
                )
        else:
            assert score is not None
            outcomes = []
            for candidate in alive:
                try:
                    outcomes.append(score(candidate, budget))
                except Exception as exc:  # a bad candidate must not end the search
                    outcomes.append(exc)

        for candidate, outcome in zip(alive, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                rows.append(
                    {
                        "rung": depth,
                        "budget": budget,
                        "candidate": name(candidate),
                        "skipped": f"{type(outcome).__name__}: {outcome}",
                    }
                )
                continue
            rung.results.append((candidate, outcome))
            rows.append({"rung": depth, "budget": budget, "candidate": name(candidate), **outcome})

        if on_rung is not None:
            on_rung(rung)
        rungs.append(rung)

        eligible = [
            (c, m)
            for c, m in rung.results
            if objective in m
            and not pd.isna(m[objective])
            # A scorer that reports no trade count does not get a pass on the
            # trade filter: np.inf here waved through anything that forgot the
            # column.
            and m.get("trades", 0.0) >= minimum_trades
        ]
        if not eligible:
            raise SearchError(
                f"no candidate produced {objective!r} with at least "
                f"{minimum_trades:g} trades at budget {budget}"
            )

        eligible.sort(key=lambda pair: pair[1][objective], reverse=True)
        alive = [c for c, _ in eligible[:survivors]]

    best = alive[0]
    best_metrics = next(m for c, m in rungs[-1].results if c is best)
    return SearchOutcome(
        best=best,
        best_metrics=best_metrics,
        objective=objective,
        table=pd.DataFrame(rows),
        rungs=rungs,
    )


def sample_configurations(
    space: dict[str, Sequence[Any]],
    n: int,
    *,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Draw ``n`` distinct configurations at random from a space of choices.

    Random rather than gridded, because a grid divides its budget by how finely
    each axis was cut and most axes do not deserve equal attention. Distinct,
    because duplicates would quietly shrink the search while appearing not to.

    Deterministic given a seed: a search nobody can repeat is not evidence.
    """
    if n < 1:
        raise SearchError(f"n must be at least 1, got {n}")
    rng = np.random.default_rng(seed)
    keys = list(space)

    total = 1
    for key in keys:
        total *= len(space[key])
    if n >= total:
        # Small spaces are enumerated. Sampling ninety of a hundred by rejection
        # is slower than listing all hundred, and the caller asked for coverage.
        from itertools import product

        return [
            dict(zip(keys, values, strict=True)) for values in product(*(space[k] for k in keys))
        ]

    seen: set[tuple] = set()
    out: list[dict[str, Any]] = []
    while len(out) < n:
        choice = tuple(space[key][int(rng.integers(len(space[key])))] for key in keys)
        if choice in seen:
            continue
        seen.add(choice)
        out.append(dict(zip(keys, choice, strict=True)))
    return out


def neighbourhood_scores(table: pd.DataFrame, axes: Sequence[str], value: str) -> pd.Series:
    """Each cell's ``value`` as the median over the cell and its grid neighbours.

    The generalisation of the neighbourhood choice in
    :mod:`trading_research.pipeline.discovery` to any set of ordered axes.
    Taking the outright best of many cells is how §25 and §26 were fooled: the
    maximum of a noisy surface is whichever cell noise favoured. A cell whose
    neighbours also score well describes a region, and a region is what an
    effect looks like.

    The neighbourhood of a cell is every cell of ``table`` that lies within one
    step of it along every axis in ``axes`` — the block of up to ``3 ** len(axes)``
    cells around it, the cell itself included — where a step is to the adjacent
    value among those each axis takes in ``table``. Cells absent from
    ``table`` (not measured) simply do not contribute. Scored by the median, so
    one adjacent outlier cannot carry the group. Rows with a missing ``value``
    neither score nor contribute.

    Returns a series aligned with ``table``'s index.
    """
    if not axes:
        raise SearchError("a neighbourhood needs at least one axis")
    missing = [a for a in [*axes, value] if a not in table.columns]
    if missing:
        raise SearchError(f"columns {missing} are not in the table")
    if table[list(axes)].duplicated().any():
        raise SearchError("two rows share one cell of the grid; one value per cell is needed")
    levels = {a: sorted(table[a].dropna().unique().tolist()) for a in axes}
    position = pd.DataFrame(
        {a: table[a].map({v: i for i, v in enumerate(levels[a])}) for a in axes},
        index=table.index,
    )
    values = table[value].to_numpy(dtype=np.float64)
    coordinates = position.to_numpy(dtype=np.float64)
    known = np.isfinite(values) & np.all(np.isfinite(coordinates), axis=1)
    out = np.full(len(table), np.nan)
    for row in np.flatnonzero(known):
        near = known & np.all(np.abs(coordinates - coordinates[row]) <= 1, axis=1)
        out[row] = float(np.median(values[near]))
    return pd.Series(out, index=table.index, name=f"{value}_neighbourhood")
