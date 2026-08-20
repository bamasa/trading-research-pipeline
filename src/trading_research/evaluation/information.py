"""Which source carries the signal, measured before any model is chosen.

The mistake this module corrects was one of order. For most of this project the
loop was: pick some features, fit a model, look at the money, try a different
model. That answers "did this arrangement work" and never answers "was there
anything here to find", so a negative result is ambiguous — the source may have
been empty, or the model may have been wrong for it.

An audit inverts the order. For each candidate source of information, measure
how much it says about the forward move *before* committing to a model, and
measure what it adds on top of the sources already in hand. Then the shortfall
has an address: either a source is empty and no model will help, or it carries
signal the current model is failing to use.

Three measurements, cheapest first
----------------------------------
**Linear ceiling.** The multiple correlation of the block with the forward
move — the information coefficient a linear model would achieve fitted on the
block, computed in closed form. Cheap, and a lower bound on what is there.

**Mutual information.** How much the block reduces uncertainty about the sign
of the move, estimated by nearest neighbours. Sees interactions and non-linear
structure that the correlation cannot, at the cost of being upward-biased on
small samples — so it is reported beside the correlation rather than instead of
it, and the two disagreeing is itself informative.

**Incremental contribution.** The linear ceiling of a block *after* projecting
out everything the sources already accepted can explain. This is the number
that decides whether to add a source: a block correlating 0.05 on its own adds
nothing if the existing features already span it.

Everything is fitted on a training block and measured on a held-out one. An
information coefficient computed in-sample is a statement about the sample.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


class AuditError(ValueError):
    """A block cannot be measured as given."""


@dataclass(frozen=True)
class SourceAudit:
    """What one block of features is worth, alone and on top of the others."""

    source: str
    columns: int
    rows: int
    #: Best information coefficient a linear combination of this block achieves
    #: on held-out data.
    linear_ic: float
    #: Nearest-neighbour mutual information with the sign of the move, in nats.
    mutual_information: float
    #: Linear IC after removing what the already-accepted sources explain.
    incremental_ic: float
    #: The single best column in the block, and its own IC — so a block that
    #: owes everything to one column is visible as such.
    best_column: str
    best_column_ic: float


def _standardise(values: np.ndarray) -> np.ndarray:
    centred = values - np.nanmean(values, axis=0)
    scale = np.nanstd(centred, axis=0)
    scale[scale == 0] = np.nan
    return centred / scale


def linear_ceiling(
    train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame, test_y: pd.Series
) -> tuple[float, np.ndarray]:
    """IC of the best linear combination, fitted on train and scored on test.

    Ridge rather than ordinary least squares, with a small penalty: on collinear
    book features the unpenalised solution is numerically unstable and produces
    a ceiling that is really a report on the condition number.
    """
    x = train_x.to_numpy(dtype="float64")
    y = train_y.to_numpy(dtype="float64")
    ok = np.isfinite(x).all(axis=1) & np.isfinite(y)
    if ok.sum() < 500:
        raise AuditError(f"only {int(ok.sum())} usable training rows")
    x, y = x[ok], y[ok]

    centre, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1.0
    x = (x - centre) / scale
    penalty = 1e-3 * len(x)
    coefficients = np.linalg.solve(x.T @ x + penalty * np.eye(x.shape[1]), x.T @ (y - y.mean()))

    tx = test_x.to_numpy(dtype="float64")
    ty = test_y.to_numpy(dtype="float64")
    ok = np.isfinite(tx).all(axis=1) & np.isfinite(ty)
    if ok.sum() < 200:
        raise AuditError(f"only {int(ok.sum())} usable test rows")
    predicted = ((tx[ok] - centre) / scale) @ coefficients
    if np.std(predicted) == 0 or np.std(ty[ok]) == 0:
        return 0.0, np.zeros(len(test_x))

    full = np.full(len(test_x), np.nan)
    full[ok] = predicted
    return float(np.corrcoef(predicted, ty[ok])[0, 1]), full


def mutual_information(
    train_x: pd.DataFrame, train_y: pd.Series, *, sample: int = 20_000, seed: int = 0
) -> float:
    """Nearest-neighbour mutual information with the sign of the move.

    Subsampled, because the estimator is O(n log n) with a large constant and
    the answer stops moving long before the sample is exhausted. Upward-biased
    on small samples in a way that grows with dimension, so it compares blocks
    of similar width more safely than blocks of very different widths — which is
    stated here because the temptation is to read it as an absolute quantity.
    """
    from sklearn.feature_selection import mutual_info_classif

    x = train_x.to_numpy(dtype="float64")
    y = np.sign(train_y.to_numpy(dtype="float64"))
    ok = np.isfinite(x).all(axis=1) & np.isfinite(y)
    if ok.sum() < 500:
        raise AuditError(f"only {int(ok.sum())} usable rows for mutual information")
    x, y = x[ok], y[ok]
    if len(x) > sample:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(x), size=sample, replace=False)
        x, y = x[keep], y[keep]
    if len(np.unique(y)) < 2:
        return 0.0
    return float(np.sum(mutual_info_classif(x, y.astype(int), random_state=seed)))


def _residualise(target: np.ndarray, against: np.ndarray) -> np.ndarray:
    """Remove from ``target`` whatever ``against`` linearly explains."""
    ok = np.isfinite(target) & np.isfinite(against).all(axis=1)
    out = np.full_like(target, np.nan)
    if ok.sum() < 100:
        return out
    design = np.column_stack([np.ones(ok.sum()), against[ok]])
    coefficients, *_ = np.linalg.lstsq(design, target[ok], rcond=None)
    out[ok] = target[ok] - design @ coefficients
    return out


def audit(
    blocks: dict[str, Sequence[str]],
    train_x: pd.DataFrame,
    train_y: pd.Series,
    test_x: pd.DataFrame,
    test_y: pd.Series,
    *,
    with_mutual_information: bool = True,
) -> pd.DataFrame:
    """Measure every block alone, then in the order that maximises what is added.

    Blocks are accepted greedily: the strongest joins first, then whichever adds
    most on top of it, and so on. The ``incremental_ic`` column is therefore the
    contribution *in the order accepted*, which is the number that answers
    "should I go and build this source" — a block that duplicates one already in
    hand shows near zero however well it scores alone.
    """
    remaining = dict(blocks)
    accepted: list[str] = []
    accepted_columns: list[str] = []
    results: list[SourceAudit] = []

    while remaining:
        scored = []
        for name, columns in remaining.items():
            usable = [c for c in columns if c in train_x.columns]
            if not usable:
                continue
            try:
                alone, _ = linear_ceiling(train_x[usable], train_y, test_x[usable], test_y)
            except AuditError:
                continue

            if accepted_columns:
                # What this block adds once the accepted ones have had their say.
                base_train = train_x[accepted_columns].to_numpy(dtype="float64")
                base_test = test_x[accepted_columns].to_numpy(dtype="float64")
                residual_train = pd.Series(
                    _residualise(train_y.to_numpy(dtype="float64"), base_train),
                    index=train_y.index,
                )
                residual_test = pd.Series(
                    _residualise(test_y.to_numpy(dtype="float64"), base_test),
                    index=test_y.index,
                )
                try:
                    incremental, _ = linear_ceiling(
                        train_x[usable], residual_train, test_x[usable], residual_test
                    )
                except AuditError:
                    incremental = float("nan")
            else:
                incremental = alone
            scored.append((name, usable, alone, incremental))

        if not scored:
            break
        scored.sort(key=lambda row: abs(row[3]) if np.isfinite(row[3]) else -1, reverse=True)
        name, usable, alone, incremental = scored[0]

        ics = {}
        for column in usable:
            values = test_x[column].to_numpy(dtype="float64")
            ok = np.isfinite(values) & np.isfinite(test_y.to_numpy(dtype="float64"))
            if ok.sum() > 200 and np.std(values[ok]) > 0:
                ics[column] = abs(float(np.corrcoef(values[ok], test_y.to_numpy()[ok])[0, 1]))
        best_column = max(ics, key=lambda k: ics[k]) if ics else "none"

        results.append(
            SourceAudit(
                source=name,
                columns=len(usable),
                rows=len(test_x),
                linear_ic=alone,
                mutual_information=(
                    mutual_information(train_x[usable], train_y)
                    if with_mutual_information
                    else float("nan")
                ),
                incremental_ic=incremental,
                best_column=best_column,
                best_column_ic=ics.get(best_column, float("nan")),
            )
        )
        accepted.append(name)
        accepted_columns.extend(usable)
        del remaining[name]

    return pd.DataFrame(
        [
            {
                "order_accepted": i + 1,
                "source": r.source,
                "columns": r.columns,
                "linear_ic": r.linear_ic,
                "incremental_ic": r.incremental_ic,
                "mutual_information": r.mutual_information,
                "best_column": r.best_column,
                "best_column_ic": r.best_column_ic,
            }
            for i, r in enumerate(results)
        ]
    )
