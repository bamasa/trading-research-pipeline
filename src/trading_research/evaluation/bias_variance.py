"""Which half of the error an ensemble could actually fix.

An ensemble is not a general-purpose improvement. Bagging reduces variance and
leaves bias roughly where it was; boosting reduces bias and can raise variance;
stacking helps when different models are wrong in different places. Choosing
between them without knowing which term dominates is guessing, and this project
has a specific reason to look: the fitted logistic model scores an information
coefficient of 0.011 on held-out data while raw queue imbalance — one column, no
fitting — scores several times that. A model that loses to its own input is a
model whose error is not bias.

The decomposition
-----------------
For squared error against a target *y*, over models fitted on bootstrap resamples
of the training set, the expected error at a point decomposes as

    E[(y - f(x))^2] = (y - E[f(x)])^2 + E[(f(x) - E[f(x)])^2] + noise
                       \\_____ bias^2 _____/  \\______ variance ______/

The middle term is what averaging removes. Estimating it needs the *spread of
predictions across refits*, which is why this module resamples: a single fitted
model cannot tell you how much of its error was luck in the training draw.

Classification
--------------
For a 0/1 target the same idea is applied to predicted probabilities rather than
to labels, following Kohavi and Wolpert's formulation closely enough for the
comparison to be meaningful and loosely enough to say so. What matters here is
not the exact partition — the constants differ between the several published
definitions — but the *ratio*: if variance is several times bias, averaging
helps; if bias dominates, only a different model class does.

Time series, not bootstrap samples
----------------------------------
Textbook bias-variance resamples rows at random. On a series where neighbouring
rows are near-duplicates, random resampling leaves nearly the same training set
every time, variance is measured as near zero, and the conclusion is wrong.

So the resamples here are **contiguous blocks** drawn from the training span:
each fit sees a different stretch of history, which is the variation a real
refit is exposed to. That measures something slightly different from the
textbook quantity and something much more relevant — how much this model's
predictions depend on which fortnight it happened to be fitted on.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


class DecompositionError(ValueError):
    """The data or the configuration cannot support a decomposition."""


@dataclass(frozen=True)
class Decomposition:
    """The error of one model, split into the parts an ensemble treats differently."""

    model: str
    fits: int
    #: Mean squared error of the average prediction. Named for the textbook
    #: term, but it is bias *plus the target's own noise*: separating them
    #: needs repeated observations of the same conditions, which a price series
    #: does not offer. Read it against ``target_variance`` below.
    bias_squared: float
    #: Variance of the target itself — the error of predicting its mean every
    #: time. When ``bias_squared`` is within a hair of this, the model explains
    #: essentially nothing, and "bias-dominated" means "mostly unpredictable"
    #: rather than "needs a better model class".
    target_variance: float
    #: Mean spread of predictions across fits — what averaging removes.
    variance: float
    #: Total expected squared error across fits.
    error: float
    #: Correlation of the *averaged* prediction with the target. The number the
    #: strategy actually cares about, reported beside the decomposition because
    #: a model can have lovely variance and no skill.
    ensemble_ic: float
    #: Mean correlation of a *single* fit. The gap between this and the above is
    #: exactly what averaging buys.
    single_ic: float

    @property
    def variance_share(self) -> float:
        total = self.bias_squared + self.variance
        return float(self.variance / total) if total > 0 else float("nan")

    @property
    def explained_share(self) -> float:
        """Fraction of the target's variance the averaged prediction removes.

        Zero means the model is no better than always predicting the mean. This
        is the number that decides how to read ``variance_share``: a model
        explaining nothing has all its error in the "bias" term by arithmetic,
        whatever its model class could in principle do.
        """
        if self.target_variance <= 0:
            return float("nan")
        return float(1.0 - self.bias_squared / self.target_variance)

    @property
    def verdict(self) -> str:
        """What the numbers recommend, in one line."""
        share = self.variance_share
        if not np.isfinite(share):
            return "no usable decomposition"
        explained = self.explained_share
        if np.isfinite(explained) and explained < 0.005:
            # The honest reading when almost nothing is explained: the error is
            # the target's own noise, and no ensemble arrangement touches it.
            return (
                "explains under 0.5% of the target: the error is mostly the "
                "target's noise, which no ensemble reduces"
            )
        if share > 0.6:
            return "variance-dominated: averaging (bagging, deeper ensembles) should help"
        if share < 0.3:
            return "bias-dominated: needs a different model class or better features"
        return "mixed: stacking diverse models is the arrangement that addresses both"


def decompose(
    fit_predict: Callable[[pd.DataFrame, pd.Series, pd.DataFrame], np.ndarray],
    features: pd.DataFrame,
    target: pd.Series,
    test_features: pd.DataFrame,
    test_target: pd.Series,
    *,
    name: str = "model",
    fits: int = 12,
    block_fraction: float = 0.5,
    seed: int = 0,
) -> Decomposition:
    """Fit ``fits`` times on different contiguous stretches and split the error.

    ``fit_predict`` takes a training frame, its target, and a test frame, and
    returns predictions for the test frame. Anything that can be trained and
    asked for a continuous score fits the signature — a single model, a
    pipeline, or an ensemble, which is the point: the same instrument measures
    the thing and its proposed remedy.
    """
    if fits < 2:
        raise DecompositionError(f"need at least two fits to see variance, got {fits}")
    if not 0.0 < block_fraction <= 1.0:
        raise DecompositionError(f"block_fraction must be in (0, 1], got {block_fraction}")

    n = len(features)
    block = int(n * block_fraction)
    if block < 500 or n - block < 1:
        raise DecompositionError(f"{n} training rows cannot support {fits} blocks of {block}")

    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, size=fits)
    y = test_target.to_numpy(dtype="float64")

    predictions = []
    for start in starts:
        window = slice(int(start), int(start) + block)
        try:
            predicted = fit_predict(features.iloc[window], target.iloc[window], test_features)
        except Exception:  # a failed refit is information, not a crash
            continue
        predicted = np.asarray(predicted, dtype="float64")
        if len(predicted) != len(y):
            raise DecompositionError(
                f"{name} returned {len(predicted)} predictions for {len(y)} rows"
            )
        predictions.append(predicted)

    if len(predictions) < 2:
        raise DecompositionError(f"{name}: only {len(predictions)} fits succeeded")

    stack = np.vstack(predictions)
    ok = np.isfinite(y) & np.isfinite(stack).all(axis=0)
    if ok.sum() < 100:
        raise DecompositionError(f"{name}: only {int(ok.sum())} comparable rows")

    stack, y = stack[:, ok], y[ok]
    average = stack.mean(axis=0)
    bias_squared = float(np.mean((y - average) ** 2))
    variance = float(np.mean(np.var(stack, axis=0)))
    error = float(np.mean((y[None, :] - stack) ** 2))

    def ic(values: np.ndarray) -> float:
        if np.std(values) == 0 or np.std(y) == 0:
            return float("nan")
        return float(np.corrcoef(values, y)[0, 1])

    return Decomposition(
        model=name,
        fits=len(predictions),
        bias_squared=bias_squared,
        target_variance=float(np.var(y)),
        variance=variance,
        error=error,
        ensemble_ic=ic(average),
        single_ic=float(np.nanmean([ic(row) for row in stack])),
    )


def compare(decompositions: Sequence[Decomposition]) -> pd.DataFrame:
    """Tabulate several models side by side, ordered by what averaging would buy."""
    rows = [
        {
            "model": d.model,
            "fits": d.fits,
            "bias_squared": d.bias_squared,
            "target_variance": d.target_variance,
            "explained_share": d.explained_share,
            "variance": d.variance,
            "variance_share": d.variance_share,
            "error": d.error,
            "single_ic": d.single_ic,
            "ensemble_ic": d.ensemble_ic,
            # What averaging over refits actually bought, in IC. This is the
            # empirical version of the recommendation, and it is the column to
            # believe when it disagrees with the variance share.
            "ic_gain_from_averaging": d.ensemble_ic - d.single_ic,
            "verdict": d.verdict,
        }
        for d in decompositions
    ]
    table = pd.DataFrame(rows)
    return table.sort_values("ic_gain_from_averaging", ascending=False).reset_index(drop=True)
