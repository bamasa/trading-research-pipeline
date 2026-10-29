"""The regime-policy axis: what a strategy does for a while after a break flag.

A flag is an alarm of the hourly break detector
(:mod:`trading_research.validation.regime_flags`), effective at the close of
the hour whose return produced it. A policy is an action and a window ``w`` in
hours:

``none``
    No guard; window 0.
``pull_<w>h``
    For taker and passive entry, no new entry decision in ``(flag, flag + w]``
    (:func:`~trading_research.pipeline.execution.guarded_rows`); an open
    position runs to its exit, and a resting passive entry is cancelled at the
    first event after the flag, since its fill would be a new entry
    (:func:`cancel_times`). For the market maker, no quotes in the window.
``widen_<w>h``
    The market maker only: its half-spread doubled in the window.

Which action goes with which execution is not decided here: every policy is
checked against :data:`~trading_research.pipeline.execution.VALID_PAIRS`, so a
taker can never be given a spread to widen. The windows searched are 4, 24 and
120 hours; 120 is the monitor's re-arm time, so the longest guard covers the
span after an alarm in which the monitor cannot alarm again.

Regime attribution asks whether a guard did anything a random pause would not:
the slot's mean minus its window-0 twin's, against the same difference under
random flag sets with as many flags, at hours drawn uniformly over the block's
usable days (:func:`regime_attribution`). It is a measurement, never a kill.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from trading_research.pipeline import market_making as mm
from trading_research.pipeline.execution import (
    VALID_PAIRS,
    ExecutionMode,
    ExecutionResult,
    SignalStream,
    check_pair,
    execute,
    guarded_rows,
)

NS_PER_HOUR = 3_600 * 1_000_000_000

#: The guard windows searched, in hours.
REGIME_WINDOWS_H: tuple[int, ...] = (4, 24, 120)

#: Actions, and the execution-layer regime policy each one is.
ACTIONS: dict[str, str] = {"none": "none", "pull": "guard_pull", "widen": "guard_widen"}

#: Random flag sets for regime attribution: a grid slot, and a simulator slot.
TAKER_SETS = 200
SIMULATOR_SETS = 20


@dataclass(frozen=True)
class RegimePolicy:
    """An action after each flag and how many hours it lasts."""

    action: str = "none"
    window_h: int = 0

    def __post_init__(self) -> None:
        if self.action not in ACTIONS:
            raise ValueError(f"action must be one of {sorted(ACTIONS)}, got {self.action!r}")
        if (self.action == "none") != (self.window_h == 0):
            raise ValueError(f"{self.action!r} with a window of {self.window_h} h means nothing")
        if self.window_h < 0:
            raise ValueError(f"a guard window cannot be negative, got {self.window_h}")

    @classmethod
    def parse(cls, name: str) -> RegimePolicy:
        """``none``, ``pull_4h``, ``widen_24h`` and so on."""
        if name == "none":
            return cls()
        action, _, window = name.partition("_")
        if not window.endswith("h") or not window[:-1].isdigit():
            raise ValueError(f"a regime policy is 'none' or '<action>_<hours>h', got {name!r}")
        return cls(action, int(window[:-1]))

    @property
    def name(self) -> str:
        return "none" if self.action == "none" else f"{self.action}_{self.window_h}h"

    @property
    def execution_policy(self) -> str:
        """The regime policy :func:`~trading_research.pipeline.execution.execute` reads."""
        return ACTIONS[self.action]

    @property
    def window_ns(self) -> int:
        return self.window_h * NS_PER_HOUR

    @property
    def guard_window_min(self) -> float:
        """The market maker's guard window: 60 minutes an hour."""
        return 60.0 * self.window_h

    def check(self, mode: ExecutionMode | str) -> ExecutionMode:
        """The mode, after refusing a pairing that is not in ``VALID_PAIRS``."""
        return check_pair(mode, self.execution_policy)


def window_hours(name: str) -> int:
    """A policy as a number on an ordered axis: its window, ``none`` being 0."""
    return RegimePolicy.parse(name).window_h


def policies_for(
    mode: ExecutionMode | str, windows: Sequence[int] = REGIME_WINDOWS_H
) -> list[RegimePolicy]:
    """Every policy ``mode`` can carry at ``windows``, ``none`` first."""
    name = ExecutionMode(mode).family
    out = [RegimePolicy()]
    for action in ("pull", "widen"):
        if (name, ACTIONS[action]) in VALID_PAIRS:
            out += [RegimePolicy(action, int(w)) for w in windows]
    return out


#: The searched axis of the taker strata: none and pull at each window.
SEARCH_POLICIES: tuple[str, ...] = tuple(p.name for p in policies_for("taker"))


# ---------------------------------------------------------------------------
# The policy applied to a decision stream
# ---------------------------------------------------------------------------


def _entry_guard(policy: RegimePolicy) -> bool:
    if policy.action == "widen":
        raise ValueError("widen acts on a quoter's spread; a decision stream can only be pulled")
    return policy.action == "pull"


def guard_decisions(
    labels_ns: np.ndarray, decision: np.ndarray, flags_ns: np.ndarray, policy: RegimePolicy
) -> np.ndarray:
    """``decision`` on a grid with every entry inside a guard window set to 0.

    A copy; rows at a flag's own instant are not guarded, rows up to and
    including ``flag + w`` are.
    """
    out = np.asarray(decision).copy()
    if _entry_guard(policy):
        out[guarded_rows(np.asarray(labels_ns), np.asarray(flags_ns), policy.window_ns)] = 0
    return out


def guard_stream(stream: SignalStream, flags_ns: np.ndarray, policy: RegimePolicy) -> SignalStream:
    """``stream`` without the decisions that fall inside a guard window."""
    if not _entry_guard(policy):
        return stream
    keep = ~guarded_rows(np.asarray(stream.ts), np.asarray(flags_ns), policy.window_ns)
    return replace(
        stream,
        ts=np.asarray(stream.ts)[keep],
        direction=np.asarray(stream.direction)[keep],
        strength=np.asarray(stream.strength)[keep],
    )


def cancel_times(
    posted_ns: np.ndarray, ends_ns: np.ndarray, flags_ns: np.ndarray, policy: RegimePolicy
) -> np.ndarray:
    """Per resting passive entry, the flag that cancels it, or -1.

    An order resting from ``posted_ns`` until ``ends_ns`` (its fill or its
    expiry, whichever the simulator reaches first) is cancelled by the first
    flag at or after it was posted and before it ends: a fill after that flag
    would be a new entry inside the guard. The simulator cancels it at its
    first event after the returned time. Under ``none`` nothing is cancelled.
    """
    posted = np.asarray(posted_ns, dtype=np.int64)
    ends = np.asarray(ends_ns, dtype=np.int64)
    if posted.shape != ends.shape:
        raise ValueError("one end per posted order")
    out = np.full(len(posted), -1, dtype=np.int64)
    flags = np.unique(np.asarray(flags_ns, dtype=np.int64))
    if not _entry_guard(policy) or len(flags) == 0:
        return out
    first = np.searchsorted(flags, posted, side="left")
    inside = first < len(flags)
    candidate = np.where(inside, flags[np.minimum(first, len(flags) - 1)], -1)
    hit = inside & (candidate < ends)
    out[hit] = candidate[hit]
    return out


def execute_with_policy(
    signals: SignalStream | None,
    mode: ExecutionMode | str,
    policy: RegimePolicy,
    flags_ns: np.ndarray | None,
    **kwargs: Any,
) -> ExecutionResult:
    """:func:`~trading_research.pipeline.execution.execute` under ``policy``.

    The pairing is checked against ``VALID_PAIRS`` first; the market maker's
    guard window is 60 times the policy's hours.
    """
    policy.check(mode)
    if policy.action != "none" and flags_ns is None:
        raise ValueError(f"{policy.name} needs the flag times")
    return execute(
        signals,
        mode,
        regime_policy=policy.execution_policy,
        flags_ns=flags_ns,
        guard_window_min=policy.guard_window_min if policy.window_h else 15.0,
        **kwargs,
    )


def maker_plan(plan: mm.MakerPlan, policy: RegimePolicy) -> mm.MakerPlan:
    """``plan`` with the market maker's guard set from ``policy``."""
    policy.check("market_maker")
    if policy.action == "none":
        return replace(plan, regime_policy="none")
    return replace(
        plan, regime_policy=policy.execution_policy, guard_window_min=policy.guard_window_min
    )


# ---------------------------------------------------------------------------
# Regime attribution
# ---------------------------------------------------------------------------


def usable_hours(days: Sequence[date]) -> np.ndarray:
    """Every hour boundary inside the usable days, int64 ns: where a random flag may fall."""
    starts = pd.DatetimeIndex(pd.to_datetime(sorted(set(days)))).tz_localize("UTC")
    base = starts.as_unit("ns").tz_localize(None).to_numpy().astype(np.int64)
    return (base[:, None] + NS_PER_HOUR * np.arange(24)[None, :]).ravel()


def flags_inside(flags_ns: np.ndarray, days: Sequence[date]) -> int:
    """How many flags fall inside the usable days: the count a random set matches."""
    hours = usable_hours(days)
    flags = np.asarray(flags_ns, dtype=np.int64)
    day_of = hours[::24]
    which = np.searchsorted(day_of, flags, side="right") - 1
    known = which >= 0
    return int((flags[known] - day_of[which[known]] < 24 * NS_PER_HOUR).sum())


def random_flag_sets(
    days: Sequence[date], n_flags: int, *, sets: int = TAKER_SETS, first_seed: int = 0
) -> list[np.ndarray]:
    """``sets`` flag sets of ``n_flags`` distinct hours each, uniform over ``days``.

    Set ``k`` is drawn with seed ``first_seed + k``, so seeds 0 to 199 give the
    registered 200 and the first 20 of them a simulator slot's 20.
    """
    hours = usable_hours(days)
    if n_flags > len(hours):
        raise ValueError(f"{n_flags} flags do not fit in {len(hours)} usable hours")
    out = []
    for k in range(sets):
        rng = np.random.default_rng(first_seed + k)
        out.append(np.sort(rng.choice(hours, size=n_flags, replace=False)))
    return out


@dataclass(frozen=True)
class Attribution:
    """A guarded slot against its window-0 twin, and against random pauses."""

    policy: str
    difference: float
    """The slot's mean minus its twin's, under the real flags."""
    reference: float
    """The 95th percentile of the same difference under the random flag sets."""
    draws: np.ndarray
    n_flags: int

    @property
    def exceeds(self) -> bool:
        return bool(self.difference > self.reference)

    def row(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "difference": self.difference,
            "random_p95": self.reference,
            "exceeds_random": self.exceeds,
            "flags": self.n_flags,
            "sets": len(self.draws),
        }


def regime_attribution(
    score: Callable[[np.ndarray], float],
    flags_ns: np.ndarray,
    twin_mean: float,
    days: Sequence[date],
    *,
    policy: str = "",
    sets: int = TAKER_SETS,
    n_flags: int | None = None,
    quantile: float = 0.95,
) -> Attribution:
    """The guard's effect beside the effect of pausing at random hours.

    ``score`` runs the slot under a set of flag times and returns the mean of
    its daily series; ``twin_mean`` is the window-0 twin's. ``n_flags``
    defaults to the flags inside ``days``. A measurement only: a guard that
    adds nothing does not make a paying strategy a false positive.
    """
    count = flags_inside(flags_ns, days) if n_flags is None else int(n_flags)
    observed = float(score(np.asarray(flags_ns, dtype=np.int64))) - float(twin_mean)
    draws = np.array(
        [float(score(f)) - float(twin_mean) for f in random_flag_sets(days, count, sets=sets)]
    )
    reference = float(np.quantile(draws, quantile)) if len(draws) else float("nan")
    return Attribution(policy, observed, reference, draws, count)
