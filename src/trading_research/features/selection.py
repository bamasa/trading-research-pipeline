"""Cutting a wide generated set back down to something a model can use.

Three stages, cheapest first, each removing what the next would only be slowed
by:

1. **Variance.** A column that barely moves carries no information and can
   still be given weight by a model fitting noise.
2. **Correlation.** Among near-duplicates keep one. The generated set produces
   these by construction — a 200-observation mean and a 200-span EWM of the
   same series are nearly the same column — and leaving both in splits their
   importance and makes the result harder to read without making it better.
3. **Mutual information.** What survives is ranked by how much it says about
   the target, keeping the strongest.

Fitted on training data only
----------------------------
The whole thing is a transformer with ``fit`` and ``transform``, and ``fit``
must only ever see the training block. Selecting features on the full sample is
one of the most effective ways to manufacture an edge that is not there: with a
few hundred candidates, some will look predictive on the test period by chance,
and choosing them *because* of that is choosing them for their test
performance.

This is also where an inherited mistake gets corrected. The author's 2025
research code, which this project draws on, ran its selection over a **random**
train/test split — `train_test_split(..., stratify=y)` — on time-series data.
For the selection step that does not leak the future into the past directly,
but it does let neighbouring, near-identical observations sit on both sides of
the split, which makes every candidate look more informative than it is. Here
the selection sees a contiguous block of training data and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class SelectionReport:
    """What each stage removed, and why."""

    n_input: int
    dropped_low_variance: list[str] = field(default_factory=list)
    dropped_correlated: list[str] = field(default_factory=list)
    dropped_low_information: list[str] = field(default_factory=list)
    selected: list[str] = field(default_factory=list)
    scores: pd.Series | None = None

    def summary(self) -> str:
        return (
            f"{self.n_input} -> {len(self.selected)} features "
            f"(variance -{len(self.dropped_low_variance)}, "
            f"correlation -{len(self.dropped_correlated)}, "
            f"information -{len(self.dropped_low_information)})"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "n_input": self.n_input,
            "n_selected": len(self.selected),
            "dropped_low_variance": len(self.dropped_low_variance),
            "dropped_correlated": len(self.dropped_correlated),
            "dropped_low_information": len(self.dropped_low_information),
            "selected": list(self.selected),
        }


@dataclass
class FeatureSelector:
    """Three-stage selection, fitted on training data only.

    ``max_features`` caps the result. On a sample of this size a model given
    three hundred columns spends most of its capacity on the noisiest of them,
    and the cap is a blunter but more reliable control than tuning
    regularisation.
    """

    variance_threshold: float = 1e-6
    correlation_threshold: float = 0.95
    max_features: int = 40
    mutual_info_sample: int = 100_000
    seed: int = 42
    #: What the ranking stage measures. One of ``"ic"``, ``"direction"``,
    #: ``"label"``.
    #:
    #: The choice matters more than the thresholds above, and the default was
    #: settled by comparing all three on four days of real BTCUSDT data:
    #:
    #: ``"label"`` — mutual information with the three-class target. Ranked the
    #: rolling standard deviation of the spread first and time-of-day fourth.
    #: It rewards whatever separates "a big move happened" from "nothing
    #: happened", which is volatility, not direction.
    #:
    #: ``"direction"`` — mutual information with the sign of the move, on rows
    #: where a tradeable move occurred. Intended to fix the above and made it
    #: worse: time-of-day rose to second and third. Restricting to moving rows
    #: shrinks the sample, and nearest-neighbour mutual information is biased
    #: upward on small samples, so with a couple of hundred candidates
    #: something always scores well by accident.
    #:
    #: ``"ic"`` — absolute correlation with the signed forward return. Puts
    #: queue imbalance and microprice deviation at the top and drops the
    #: calendar terms entirely. Linear, so it cannot see interactions; it also
    #: cannot invent them, which on this sample size is the more valuable
    #: property.
    score_target: str = "ic"

    _forward: pd.Series | None = field(default=None, init=False, repr=False)
    selected_: list[str] = field(default_factory=list, init=False)
    report_: SelectionReport | None = field(default=None, init=False)

    def fit(
        self, x: pd.DataFrame, y: pd.Series, forward: pd.Series | None = None
    ) -> FeatureSelector:
        """Choose features. Must be given the training block and nothing else.

        ``forward`` is the continuous forward return, required by the ``"ic"``
        ranking and ignored by the others.
        """
        report = SelectionReport(n_input=x.shape[1])

        usable = x.notna().all(axis=1) & y.notna()
        if usable.sum() < 100:
            raise ValueError(f"only {int(usable.sum())} complete rows; cannot select features")
        frame = x.loc[usable]
        target = y.loc[usable]
        self._forward = forward.loc[usable] if forward is not None else None

        # --- 1. variance -------------------------------------------------
        variances = frame.var()
        keep = [
            c
            for c in frame.columns
            if np.isfinite(variances[c]) and variances[c] > self.variance_threshold
        ]
        report.dropped_low_variance = [c for c in frame.columns if c not in keep]
        frame = frame[keep]

        # --- 2. correlation ----------------------------------------------
        keep = self._drop_correlated(frame)
        report.dropped_correlated = [c for c in frame.columns if c not in keep]
        frame = frame[keep]

        # --- 3. mutual information ---------------------------------------
        scores = self._mutual_information(frame, target)
        report.scores = scores
        chosen = list(scores.head(self.max_features).index)
        report.dropped_low_information = [c for c in frame.columns if c not in chosen]

        report.selected = chosen
        self.selected_ = chosen
        self.report_ = report
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        if not self.selected_:
            raise RuntimeError("selector has not been fitted")
        missing = [c for c in self.selected_ if c not in x.columns]
        if missing:
            raise ValueError(f"missing selected feature(s): {missing[:5]}")
        return x[self.selected_]

    def fit_transform(
        self, x: pd.DataFrame, y: pd.Series, forward: pd.Series | None = None
    ) -> pd.DataFrame:
        return self.fit(x, y, forward).transform(x)

    # ------------------------------------------------------------------
    def _drop_correlated(self, frame: pd.DataFrame) -> list[str]:
        """Keep one column from each group of near-duplicates.

        Walks the columns in order and drops any that correlate above the
        threshold with something already kept. Order-dependent, and
        deliberately so: an ordering rule that looked at the target would be
        making a selection decision disguised as a deduplication one.
        """
        if frame.shape[1] < 2:
            return list(frame.columns)

        corr = frame.corr().abs().to_numpy()
        columns = list(frame.columns)
        keep: list[int] = []
        for i in range(len(columns)):
            if all(
                not np.isfinite(corr[i, j]) or corr[i, j] < self.correlation_threshold for j in keep
            ):
                keep.append(i)
        return [columns[i] for i in keep]

    def _mutual_information(self, frame: pd.DataFrame, target: pd.Series) -> pd.Series:
        """Rank columns by mutual information with the chosen target.

        Under ``score_target="direction"`` the ranking is computed only on rows
        where a tradeable move occurred, against its sign. That is the question
        a directional model has to answer: given that something is about to
        happen, which way. Scoring the full three-class target instead answers
        "is something about to happen", which volatility predicts well and
        which no amount of skill converts into profit on its own.

        Subsampled for speed, from the head of the block rather than at random:
        a random subsample of a time series draws neighbouring, near-identical
        rows, which makes every candidate look more informative than it is.
        """
        from sklearn.feature_selection import mutual_info_classif

        if self.score_target == "ic":
            return self._information_coefficient(frame)

        if self.score_target == "direction":
            moving = target != 0
            if int(moving.sum()) < 200:
                raise ValueError(
                    f"only {int(moving.sum())} directional rows in the training block; "
                    f"lower the label threshold or lengthen the horizon"
                )
            scoring_x = frame.loc[moving]
            scoring_y = (target.loc[moving] > 0).astype(int)
        else:
            scoring_x = frame
            scoring_y = target.astype(int)

        n = min(self.mutual_info_sample, len(scoring_x))
        sample_x = scoring_x.iloc[:n]
        sample_y = scoring_y.iloc[:n]

        if sample_y.nunique() < 2:
            # A block with one class tells us nothing; fall back to variance
            # ordering rather than returning an arbitrary selection.
            return scoring_x.var().sort_values(ascending=False)

        scores = mutual_info_classif(
            sample_x.to_numpy(),
            sample_y.to_numpy(),
            random_state=self.seed,
        )
        return pd.Series(scores, index=sample_x.columns).sort_values(ascending=False)

    def _information_coefficient(self, frame: pd.DataFrame) -> pd.Series:
        """Rank by absolute correlation with the forward return.

        Linear, and deliberately so. Mutual information is more general and, on
        this data, less trustworthy: it is estimated by nearest neighbours,
        which is biased upward on small samples, and with a couple of hundred
        candidate columns something always scores well by chance. On a
        four-day BTCUSDT sample it put the rolling standard deviation of the
        spread first and time-of-day second and third — the first two of which
        predict *whether* the market moves, and the third of which cannot
        predict direction at all and is simply tracking where the rally
        happened to fall in the day.

        Correlation with the signed forward return measures exactly the thing
        that turns into money: how much of the move a feature explains, with
        its sign. It cannot see non-linear structure, and it will not invent
        any either.
        """
        if self._forward is None:
            raise ValueError("score_target='ic' needs the forward return; pass forward= to fit()")

        forward = self._forward.to_numpy(dtype="float64")
        scores: dict[str, float] = {}
        for column in frame.columns:
            values = frame[column].to_numpy(dtype="float64")
            usable = np.isfinite(values) & np.isfinite(forward)
            if usable.sum() < 100 or np.std(values[usable]) == 0:
                scores[column] = 0.0
                continue
            scores[column] = abs(float(np.corrcoef(values[usable], forward[usable])[0, 1]))
        return pd.Series(scores).sort_values(ascending=False)
