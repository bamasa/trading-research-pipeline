"""Choosing how to leave a trade, rather than assuming it.

The entry side of this project is tuned: features are selected, a confidence
threshold is swept on validation, the retraining schedule is searched. The exit
was a constant — hold for the label horizon and leave — set once and never
examined. That is an odd asymmetry, because the exit decides what a correct
prediction is actually worth.

Six rules are available, and they answer different questions:

============================  ==================================================
``hold_periods``              leave after a fixed time
``take_profit_bp``            leave once far enough in front
``stop_loss_bp``              leave once far enough behind
``trailing_stop_bp``          leave once enough has been given back from the peak
``exit_below_confidence``     leave when the model stops believing the trade
``exit_on_flip``              leave when the model believes the opposite
============================  ==================================================

The first four read the price; the last two read the model's ongoing opinion,
which is the only place in this pipeline where a prediction made *after* the
entry is used for anything.

None of them is free, and none is obviously right:

- a take-profit caps winners and lets losers run their course;
- a stop-loss converts noise into realised losses;
- a trailing stop does not cap winners but shaves every one of them, and at
  these horizons the path is mostly noise to shave against;
- a confidence exit leaves early on exactly the trades whose signal was
  weakest, which is either prudence or selling the bottom depending on whether
  the signal was informative or the model was merely unsure.

Since the sign of each effect is arguable and the magnitude is not knowable in
advance, all of them are searched — on validation, and only there. The test span
is scored once with whatever the search chose.

Reuse
-----
Nothing here knows about instruments, features or models. It is given a callable
that scores one policy and it walks the grid, the same arrangement as
:mod:`trading_research.validation.retrain`. Both searches can therefore run over
the same predictions without either knowing the other exists.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from itertools import product
from typing import Any

import pandas as pd

from trading_research.backtest.execution import ThinningRules


class ExitSearchError(ValueError):
    """No exit policy could be scored."""


@dataclass(frozen=True)
class ExitPolicy:
    """One complete way of leaving a trade."""

    hold_periods: int
    cooldown_periods: int = 0
    take_profit_bp: float | None = None
    stop_loss_bp: float | None = None
    trailing_stop_bp: float | None = None
    exit_below_confidence: float | None = None
    exit_on_flip: bool = False

    @property
    def label(self) -> str:
        """Short name listing only what is switched on.

        A policy with everything off is "hold{n}" rather than a string of
        Nones, because the baseline is the thing every other row is compared
        against and it should read like one.
        """
        parts = [f"hold{self.hold_periods}"]
        if self.cooldown_periods:
            parts.append(f"cd{self.cooldown_periods}")
        if self.take_profit_bp is not None:
            parts.append(f"tp{self.take_profit_bp:g}")
        if self.stop_loss_bp is not None:
            parts.append(f"sl{self.stop_loss_bp:g}")
        if self.trailing_stop_bp is not None:
            parts.append(f"trail{self.trailing_stop_bp:g}")
        if self.exit_below_confidence is not None:
            parts.append(f"conf<{self.exit_below_confidence:g}")
        if self.exit_on_flip:
            parts.append("flip")
        return "/".join(parts)

    @property
    def family(self) -> str:
        """Which rule this policy is testing, for grouping the results."""
        if self.take_profit_bp is not None and self.stop_loss_bp is not None:
            return "take-profit + stop-loss"
        if self.take_profit_bp is not None:
            return "take-profit"
        if self.stop_loss_bp is not None:
            return "stop-loss"
        if self.trailing_stop_bp is not None:
            return "trailing stop"
        if self.exit_below_confidence is not None:
            return "confidence decay"
        if self.exit_on_flip:
            return "prediction flip"
        return "clock only"

    def rules(self) -> ThinningRules:
        return ThinningRules(
            hold_periods=self.hold_periods,
            cooldown_periods=self.cooldown_periods,
            take_profit_bp=self.take_profit_bp,
            stop_loss_bp=self.stop_loss_bp,
            trailing_stop_bp=self.trailing_stop_bp,
            exit_below_confidence=self.exit_below_confidence,
            exit_on_flip=self.exit_on_flip,
        )

    def with_hold(self, hold_periods: int) -> ExitPolicy:
        return replace(self, hold_periods=hold_periods)


def default_grid(hold_periods: int, *, cooldown_periods: int = 0) -> tuple[ExitPolicy, ...]:
    """A grid wide enough to show each rule's shape, small enough to trust.

    One rule at a time, plus the two-sided bracket. Deliberately not a full
    cross product: with six switches and a handful of levels each, the best of
    several hundred policies on a validation block is mostly a measurement of
    how many were tried.

    The levels are chosen around the round-trip cost — roughly a third of it, a
    whole one, and twice — because a threshold far below the cost fires on
    noise and one far above never fires at all.
    """
    base = ExitPolicy(hold_periods=hold_periods, cooldown_periods=cooldown_periods)
    out = [base]
    out += [replace(base, take_profit_bp=v) for v in (4.0, 11.0, 22.0)]
    out += [replace(base, stop_loss_bp=v) for v in (4.0, 11.0, 22.0)]
    out += [replace(base, trailing_stop_bp=v) for v in (4.0, 11.0, 22.0)]
    out += [
        replace(base, take_profit_bp=tp, stop_loss_bp=sl) for tp, sl in ((11.0, 11.0), (22.0, 11.0))
    ]
    out += [replace(base, exit_below_confidence=v) for v in (0.34, 0.40, 0.45)]
    out += [replace(base, exit_on_flip=True)]
    # Shorter and longer clocks, since the horizon is itself a choice the label
    # made rather than one the exit had to inherit.
    out += [replace(base, hold_periods=max(1, hold_periods // n)) for n in (2, 4)]
    out += [replace(base, hold_periods=hold_periods * 2)]
    return tuple(out)


def expand_grid(
    hold_periods: Sequence[int],
    take_profit_bp: Sequence[float | None] = (None,),
    stop_loss_bp: Sequence[float | None] = (None,),
    trailing_stop_bp: Sequence[float | None] = (None,),
    exit_below_confidence: Sequence[float | None] = (None,),
    exit_on_flip: Sequence[bool] = (False,),
    cooldown_periods: Sequence[int] = (0,),
) -> tuple[ExitPolicy, ...]:
    """Full cross product, for when a targeted sweep is genuinely wanted."""
    return tuple(
        ExitPolicy(
            hold_periods=hold,
            cooldown_periods=cooldown,
            take_profit_bp=tp,
            stop_loss_bp=sl,
            trailing_stop_bp=trail,
            exit_below_confidence=conf,
            exit_on_flip=flip,
        )
        for hold, cooldown, tp, sl, trail, conf, flip in product(
            hold_periods,
            cooldown_periods,
            take_profit_bp,
            stop_loss_bp,
            trailing_stop_bp,
            exit_below_confidence,
            exit_on_flip,
        )
    )


#: Scores one policy and returns a mapping of metrics. The search reads one key
#: from it and ignores the rest. Deliberately the only thing this module knows
#: about data or models.
PolicyEvaluator = Callable[[ExitPolicy], dict[str, float]]


@dataclass
class ExitSearchOutcome:
    """The chosen policy, the evidence behind it, and the test score."""

    chosen: ExitPolicy
    objective: str
    validation: pd.DataFrame
    test: dict[str, float] | None = None
    #: The untouched baseline, scored on test alongside the winner. Not a
    #: second choice: it is fixed before the search runs, so scoring it costs
    #: no selection. Without it the winner's test figure has nothing to be read
    #: against, and "improved" is a claim rather than a measurement.
    test_baseline: dict[str, float] | None = None

    def summary(self) -> str:
        line = f"chose {self.chosen.label} on validation by {self.objective}"
        if self.test is not None:
            line += f"; test {self.objective} = {self.test.get(self.objective, float('nan')):.3f}"
        if self.test_baseline is not None:
            line += f" against {self.test_baseline.get(self.objective, float('nan')):.3f} for the baseline"
        return line


def search(
    evaluate: PolicyEvaluator,
    *,
    grid: Sequence[ExitPolicy],
    objective: str = "net_per_trade_bp",
    minimum_trades: int = 50,
    baseline: ExitPolicy | None = None,
    on_test: PolicyEvaluator | None = None,
    on_result: Callable[[ExitPolicy, dict[str, float]], None] | None = None,
) -> ExitSearchOutcome:
    """Score every policy on validation, take the best, score test once.

    ``minimum_trades`` rules out policies that look good because they barely
    traded. Every rule here can be tightened until it selects a handful of
    trades, and a handful of trades is where per-trade figures stop meaning
    anything — a point §6 makes at length.

    ``on_test`` is called with the winner, and with ``baseline`` if one is
    given. Neither is consulted while choosing. Scoring the baseline on test is
    not a second bite: it is fixed before the search starts, so it carries no
    selection — and it is the only thing that makes the winner's test figure
    readable, since "better" needs something to be better than.
    """
    rows: list[dict[str, Any]] = []
    scored: list[tuple[ExitPolicy, dict[str, float]]] = []

    for policy in grid:
        try:
            metrics = evaluate(policy)
        except Exception as exc:  # one bad policy must not end the sweep
            rows.append({"policy": policy.label, "family": policy.family, "skipped": str(exc)})
            continue

        row = {"policy": policy.label, "family": policy.family, **metrics}
        if metrics.get("trades", 0.0) < minimum_trades:
            row["skipped"] = f"fewer than {minimum_trades} trades"
        else:
            scored.append((policy, metrics))
        rows.append(row)
        if on_result is not None:
            on_result(policy, metrics)

    table = pd.DataFrame(rows)
    eligible = [(p, m) for p, m in scored if objective in m and pd.notna(m[objective])]
    if not eligible:
        raise ExitSearchError(f"no policy produced {objective!r} on validation")

    best_policy, _ = max(eligible, key=lambda pair: pair[1][objective])

    test_metrics = test_baseline = None
    if on_test is not None:
        test_metrics = on_test(best_policy)
        if baseline is not None and baseline != best_policy:
            test_baseline = on_test(baseline)

    return ExitSearchOutcome(
        chosen=best_policy,
        objective=objective,
        validation=table,
        test=test_metrics,
        test_baseline=test_baseline,
    )
