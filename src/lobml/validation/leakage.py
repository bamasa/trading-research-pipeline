"""Mechanical checks that a feature cannot see the future.

The central property is **truncation invariance**. If a feature's value at row
*t* depends only on rows up to *t*, then deleting everything after some cut *T*
must leave every value at or before *T* unchanged. Compute the feature twice —
once on the full frame, once on the frame truncated at *T* — and compare.

This is worth stating as a property rather than a habit because the ways
look-ahead gets in are all silent:

``rolling(window, center=True)``
    Reads forward. Truncating changes earlier values, so this fails.
``fillna(method="bfill")``
    Copies a later row backwards. Truncating changes what gets copied, so this
    fails.
``(x - x.mean()) / x.std()``
    Uses statistics of the whole sample. Truncating changes the mean, so every
    value shifts and this fails.

None of the three raises, none looks wrong in a plot, and each one inflates a
backtest. The check is cheap, needs no labels, and turns "we were careful" into
something a build can enforce.

What it does not catch
----------------------
Truncation invariance is necessary, not sufficient. It says a feature does not
read later *rows*; it cannot say the data in those rows was correctly timed in
the first place. A feature stamped with the moment a message was *received*
rather than when the event *happened* is causal with respect to row order and
still unusable in practice. That is a data-contract question, which is why
:mod:`lobml.data.schema` is explicit that ``timestamp`` means the moment
information became observable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from lobml.features.registry import REGISTRY, Feature, Registry


@dataclass(frozen=True)
class LeakageResult:
    """Outcome of checking one feature."""

    name: str
    leaks: bool
    first_divergence: int | None
    n_diverged: int
    detail: str

    def __str__(self) -> str:
        if not self.leaks:
            return f"{self.name}: causal"
        return f"{self.name}: LEAKS — {self.detail}"


class LeakageError(AssertionError):
    """A feature's value depends on data that would not have been available."""


def check_feature(
    feature: Feature,
    df: pd.DataFrame,
    *,
    cut: float = 0.6,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> LeakageResult:
    """Check one feature for truncation invariance.

    ``cut`` is where to split, as a fraction of the frame. It defaults to 0.6
    rather than 0.5 for no deep reason beyond avoiding a boundary that happens
    to align with a round window length.

    Comparison ignores positions that are ``NaN`` in both runs — warm-up rows
    legitimately have no value — but a position that is ``NaN`` in one and not
    the other counts as divergence, since that is precisely what a feature
    reaching past the end of its data looks like.
    """
    if len(df) < 100:
        raise ValueError(f"need at least 100 rows to check {feature.name!r}, got {len(df)}")

    boundary = int(len(df) * cut)
    full = np.asarray(feature(df), dtype="float64")[:boundary]
    truncated = np.asarray(feature(df.iloc[:boundary].copy()), dtype="float64")

    both_nan = np.isnan(full) & np.isnan(truncated)
    close = np.isclose(full, truncated, rtol=rtol, atol=atol, equal_nan=False)
    same = close | both_nan

    if same.all():
        return LeakageResult(
            feature.name, leaks=False, first_divergence=None, n_diverged=0, detail=""
        )

    diverged = np.flatnonzero(~same)
    first = int(diverged[0])
    lag = boundary - first
    detail = (
        f"{len(diverged)} of {boundary} values changed when data after row {boundary} was removed; "
        f"first at row {first}, i.e. {lag} row(s) before the cut. "
        f"full={full[first]!r} truncated={truncated[first]!r}"
    )
    return LeakageResult(
        feature.name,
        leaks=True,
        first_divergence=first,
        n_diverged=len(diverged),
        detail=detail,
    )


def check_registry(
    df: pd.DataFrame,
    *,
    plane: str | None = None,
    registry: Registry | None = None,
    cut: float = 0.6,
) -> list[LeakageResult]:
    """Check every registered feature for the given plane."""
    reg = registry if registry is not None else REGISTRY
    features = [f for f in reg.features.values() if plane is None or f.plane == plane]
    if not features:
        raise ValueError(f"no features registered for plane {plane!r}")
    return [check_feature(f, df, cut=cut) for f in sorted(features, key=lambda f: f.name)]


def assert_causal(
    df: pd.DataFrame,
    *,
    plane: str | None = None,
    registry: Registry | None = None,
    cut: float = 0.6,
) -> None:
    """Raise if any registered feature reads data it should not have."""
    results = check_registry(df, plane=plane, registry=registry, cut=cut)
    leaking = [r for r in results if r.leaks]
    if leaking:
        detail = "\n".join(f"  {r}" for r in leaking)
        raise LeakageError(f"{len(leaking)} feature(s) depend on future data:\n{detail}")


def check_declared_lookback(
    feature: Feature,
    df: pd.DataFrame,
    *,
    position: int | None = None,
) -> LeakageResult:
    """Check that a feature needs no more history than it claims.

    Complementary to truncation invariance, and aimed at the opposite mistake.
    Truncation catches a feature reading *forwards*; this catches one reading
    further *backwards* than it admits — which is not a leak, but understates
    the embargo a split must apply, and so leaks across the boundary instead.

    The test: compute the feature at one position from the full history, then
    from only ``lookback + 1`` rows ending there. A feature that keeps its
    promise gives the same answer.
    """
    lookback = feature.lookback
    at = position if position is not None else len(df) - 1
    start = at - lookback
    if start < 0:
        raise ValueError(
            f"need at least {lookback + 1} rows before position {at} for {feature.name!r}"
        )

    full = np.asarray(feature(df), dtype="float64")[at]
    window = np.asarray(feature(df.iloc[start : at + 1].copy()), dtype="float64")[-1]

    if np.isnan(full) and np.isnan(window):
        return LeakageResult(
            feature.name, leaks=False, first_divergence=None, n_diverged=0, detail=""
        )
    if np.isclose(full, window, rtol=1e-9, atol=1e-12, equal_nan=False):
        return LeakageResult(
            feature.name, leaks=False, first_divergence=None, n_diverged=0, detail=""
        )

    return LeakageResult(
        feature.name,
        leaks=True,
        first_divergence=at,
        n_diverged=1,
        detail=(
            f"declares lookback={lookback} but needs more history: "
            f"full={full!r} from {lookback + 1} rows={window!r}"
        ),
    )


def assert_lookback_honest(
    df: pd.DataFrame,
    *,
    plane: str | None = None,
    registry: Registry | None = None,
) -> None:
    """Raise if any feature needs more history than it declares."""
    reg = registry if registry is not None else REGISTRY
    features = [f for f in reg.features.values() if plane is None or f.plane == plane]
    results = [check_declared_lookback(f, df) for f in sorted(features, key=lambda f: f.name)]
    wrong = [r for r in results if r.leaks]
    if wrong:
        detail = "\n".join(f"  {r}" for r in wrong)
        raise LeakageError(f"{len(wrong)} feature(s) understate their lookback:\n{detail}")
