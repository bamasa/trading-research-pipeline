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
    score: Scorer,
    *,
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
    """
    if not candidates:
        raise SearchError("no candidates to search")
    if not budgets:
        raise SearchError("no budgets given")

    name = label or (lambda c: str(c))
    alive = list(candidates)
    rungs: list[Rung] = []
    rows: list[dict[str, Any]] = []

    for depth, budget in enumerate(budgets):
        survivors = max(1, int(len(alive) * keep_fraction)) if depth + 1 < len(budgets) else 1
        rung = Rung(budget=budget, candidates=len(alive), survivors=survivors)

        for candidate in alive:
            try:
                metrics = score(candidate, budget)
            except Exception as exc:  # a bad candidate must not end the search
                rows.append(
                    {
                        "rung": depth,
                        "budget": budget,
                        "candidate": name(candidate),
                        "skipped": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            rung.results.append((candidate, metrics))
            rows.append({"rung": depth, "budget": budget, "candidate": name(candidate), **metrics})

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
