"""Combining models, and the one detail that decides whether it works.

Four arrangements, each addressing a different failure:

* **Bagging** — fit the same learner on resampled data and average. Reduces
  variance, leaves bias. The right tool when a model's predictions swing with
  which stretch of history it saw, which
  :mod:`trading_research.evaluation.bias_variance` measures rather than assumes.
* **Random forests and extremely randomised trees** — bagging plus feature
  subsampling, and in the second case random split points too. More
  decorrelation between members, which is where the variance reduction comes
  from; extra randomisation trades a little more bias for a lot less variance,
  and on noisy financial features that trade is usually worth taking.
* **Voting** — average the probabilities of *different* model classes. Helps
  only insofar as the members are wrong in different places, so a diversity
  figure is reported alongside the result. Two members correlating 0.98 are one
  member with extra steps.
* **Stacking** — fit a second-stage model on the first stage's predictions, so
  the weights are learned rather than assumed equal. The most powerful of the
  four and by far the easiest to get wrong.

Why stacking is easy to get wrong
---------------------------------
The second stage must be trained on predictions the first stage made for rows it
did **not** see. Train it on in-sample predictions and it learns that the
strongest base model is always right — because on its own training data it is —
and the stack then fails on anything new. The standard fix is out-of-fold
prediction, and the standard fix has a standard failure on time series: K-fold
puts future rows in the training folds of past ones.

So the folds here are **forward-chaining**: fold *k* trains on everything before
it and predicts fold *k*, exactly as a walk-forward does. Fewer usable rows than
K-fold, no leakage. Purging between the training rows and the fold they predict
is the caller's business, since only the caller knows the label horizon; the
purge parameter is exposed for that.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from trading_research.models.base import CLASSES, Model, clean


class StackingError(ValueError):
    """The ensemble cannot be built or fitted as configured."""


class Bagged(Model):
    """Average a learner fitted on several contiguous stretches of history.

    Contiguous rather than randomly resampled, for the reason in
    :mod:`trading_research.evaluation.bias_variance`: neighbouring rows in a
    book series are near-duplicates, so a random bootstrap draws essentially the
    same training set every time and averages away nothing. Different stretches
    of history is the variation a refit actually faces.
    """

    def __init__(
        self,
        build: Any,
        *,
        members: int = 8,
        block_fraction: float = 0.6,
        seed: int = 0,
    ) -> None:
        super().__init__(name="bagged")
        if members < 2:
            raise StackingError(f"need at least two members, got {members}")
        if not 0.0 < block_fraction <= 1.0:
            raise StackingError(f"block_fraction must be in (0, 1], got {block_fraction}")
        self.build: Any = build
        self.members = members
        self.block_fraction = block_fraction
        self.seed = seed
        self.members_: list[Any] = []

    def fit(self, x: pd.DataFrame, y: pd.Series) -> Bagged:
        x, y = clean(x, y)
        block = max(500, int(len(x) * self.block_fraction))
        if len(x) < block:
            raise StackingError(f"{len(x)} rows cannot support blocks of {block}")
        rng = np.random.default_rng(self.seed)
        starts = rng.integers(0, len(x) - block + 1, size=self.members)

        self.members_ = []
        for start in starts:
            window = slice(int(start), int(start) + block)
            part_y = y.iloc[window]
            if part_y.nunique() < 2:
                continue
            member = self.build()
            member.fit(x.iloc[window], part_y)
            self.members_.append(member)
        if not self.members_:
            raise StackingError("no member could be fitted")
        self.fitted_ = True
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        stack = np.stack([m.predict_proba(x) for m in self.members_])
        return stack.mean(axis=0)

    def member_disagreement(self, x: pd.DataFrame) -> float:
        """Mean standard deviation of member probabilities — the diversity that pays."""
        self._check_fitted()
        stack = np.stack([m.predict_proba(x) for m in self.members_])
        return float(np.mean(np.std(stack, axis=0)))


class Voting(Model):
    """Average the probabilities of several different model classes.

    ``weights`` default to equal. Unequal weights chosen by looking at
    performance are stacking done badly, so if the weights are to be learned,
    use :class:`Stacked` and learn them properly.
    """

    def __init__(
        self, members: Sequence[object], *, weights: Sequence[float] | None = None
    ) -> None:
        super().__init__(name="voting")
        if len(members) < 2:
            raise StackingError(f"need at least two members, got {len(members)}")
        if weights is not None and len(weights) != len(members):
            raise StackingError(f"{len(weights)} weights for {len(members)} members")
        self.members: list[Any] = list(members)
        self.weights = np.asarray(weights, dtype="float64") if weights else None

    def fit(self, x: pd.DataFrame, y: pd.Series) -> Voting:
        x, y = clean(x, y)
        for member in self.members:
            member.fit(x, y)
        self.fitted_ = True
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        stack = np.stack([m.predict_proba(x) for m in self.members])
        if self.weights is None:
            return stack.mean(axis=0)
        weights = self.weights / self.weights.sum()
        return np.tensordot(weights, stack, axes=(0, 0))

    def diversity(self, x: pd.DataFrame) -> float:
        """Mean pairwise correlation of member scores. Near 1 means one model."""
        self._check_fitted()
        scores = np.stack(
            [
                m.predict_proba(x)[:, CLASSES.index(1)] - m.predict_proba(x)[:, CLASSES.index(-1)]
                for m in self.members
            ]
        )
        if scores.shape[0] < 2 or np.any(np.std(scores, axis=1) == 0):
            return float("nan")
        matrix = np.corrcoef(scores)
        upper = matrix[np.triu_indices_from(matrix, k=1)]
        return float(np.mean(upper))


def forward_chaining_folds(n: int, folds: int, *, purge: int = 0) -> list[tuple[slice, slice]]:
    """Training and prediction ranges where training is always in the past.

    Fold *k* trains on rows before it and predicts the block that follows,
    leaving ``purge`` rows between the two so a label that reads forward cannot
    reach the rows it is about to be scored on.

    The first block is never predicted — nothing precedes it to train on — which
    is why a stack has fewer second-stage rows than the series has rows, and why
    the fold count trades sample size against how recent the training is.
    """
    if folds < 2:
        raise StackingError(f"need at least two folds, got {folds}")
    size = n // (folds + 1)
    if size <= purge:
        raise StackingError(f"{n} rows in {folds + 1} blocks cannot absorb a purge of {purge}")
    out = []
    for k in range(1, folds + 1):
        train_end = k * size - purge
        if train_end <= 0:
            continue
        out.append((slice(0, train_end), slice(k * size, (k + 1) * size)))
    return out


class Stacked(Model):
    """Learn how to combine base models, on predictions they did not fit.

    The second stage sees, for each row, what each base model said about that
    row while trained only on rows before it. That is what makes the learned
    weights mean anything out of sample.
    """

    def __init__(
        self,
        build_base: Sequence[Any],
        build_meta: Any,
        *,
        folds: int = 4,
        purge: int = 0,
        pass_through: bool = False,
    ) -> None:
        super().__init__(name="stacked")
        if len(build_base) < 2:
            raise StackingError(f"need at least two base models, got {len(build_base)}")
        self.build_base: list[Any] = list(build_base)
        self.build_meta: Any = build_meta
        self.folds = folds
        self.purge = purge
        #: Give the second stage the original features as well as the base
        #: predictions. More expressive and much easier to overfit; off by
        #: default for that reason.
        self.pass_through = pass_through
        self.base_: list[Any] = []
        self.meta_: Any = None

    def _scores(self, models: Sequence[Any], x: pd.DataFrame) -> np.ndarray:
        return np.column_stack(
            [
                m.predict_proba(x)[:, CLASSES.index(1)] - m.predict_proba(x)[:, CLASSES.index(-1)]
                for m in models
            ]
        )

    def fit(self, x: pd.DataFrame, y: pd.Series) -> Stacked:
        x, y = clean(x, y)
        folds = forward_chaining_folds(len(x), self.folds, purge=self.purge)
        if not folds:
            raise StackingError(f"{len(x)} rows produced no usable folds")

        rows, targets, frames = [], [], []
        for train, predict in folds:
            if y.iloc[train].nunique() < 2:
                continue
            members = [build() for build in self.build_base]
            for member in members:
                member.fit(x.iloc[train], y.iloc[train])
            rows.append(self._scores(members, x.iloc[predict]))
            targets.append(y.iloc[predict])
            frames.append(x.iloc[predict])
        if not rows:
            raise StackingError("no fold could be fitted")

        meta_x = pd.DataFrame(
            np.vstack(rows), columns=[f"base_{i}" for i in range(len(self.build_base))]
        )
        if self.pass_through:
            extra = pd.concat(frames, ignore_index=True)
            meta_x = pd.concat([meta_x, extra.reset_index(drop=True)], axis=1)
        meta_y = pd.concat(targets, ignore_index=True)
        if meta_y.nunique() < 2:
            raise StackingError("out-of-fold predictions carry one class only")

        self.meta_ = self.build_meta()
        self.meta_.fit(meta_x, meta_y)

        # The deployed base models are refitted on everything, since at
        # prediction time there is no reason to withhold data from them.
        self.base_ = [build() for build in self.build_base]
        for member in self.base_:
            member.fit(x, y)
        self.fitted_ = True
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        assert self.meta_ is not None
        meta_x = pd.DataFrame(
            self._scores(self.base_, x), columns=[f"base_{i}" for i in range(len(self.base_))]
        )
        if self.pass_through:
            meta_x = pd.concat([meta_x, x.reset_index(drop=True)], axis=1)
        return self.meta_.predict_proba(meta_x)
