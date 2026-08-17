"""Probability calibration.

A classifier's output is a score between zero and one. Whether it is a
*probability* — whether the moments it scores at 0.7 really come out positive
70% of the time — is a separate question, and boosted trees in particular are
usually poor at it. Class weighting, which this project uses because the target
is 99% HOLD, makes it worse: weighting deliberately distorts the output
distribution to stop the fit collapsing.

What calibration does and does not change
-----------------------------------------
Both methods here are **monotonic**: isotonic regression fits a non-decreasing
step function, Platt scaling a logistic curve. Neither can reorder the
predictions. So for a strategy that trades whenever the score clears a
threshold, and that finds the threshold by sweeping it on validation,
calibration changes nothing at all — the same trades are selected, at a
different numeric threshold.

It matters when the number is used as a number:

- sizing a position by expected value, where 0.55 and 0.95 must mean what they
  say rather than merely rank correctly;
- comparing models whose scores live on different scales;
- any decision rule combining a probability with a payoff, which is where a
  cost-aware gate belongs and where an uncalibrated score quietly ruins the
  arithmetic.

Fitted on data the model did not train on
-----------------------------------------
The calibrator learns from the model's mistakes, so it has to see predictions
the model did not fit. Calibrating on the training block teaches it to correct
an overconfidence that only appears there, and the correction is then wrong
everywhere else. Here the training block is split chronologically and the tail
is reserved — a random split would put near-identical neighbours on both sides,
which is the same mistake in a less visible form.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trading_research.models.base import CLASSES, Model


class CalibratedModel(Model):
    """Wraps any model and calibrates its probabilities.

    One calibrator per class, each mapping that class's raw score to a
    frequency, followed by renormalisation so the row sums to one. That is the
    standard one-vs-rest construction; it is not guaranteed to preserve the
    ordering *between* classes, only within each, which is why the trade
    decision still compares like with like.
    """

    def __init__(
        self,
        base: Model,
        *,
        method: str = "isotonic",
        holdout_fraction: float = 0.2,
    ) -> None:
        super().__init__(name=f"{base.name}+{method}")
        if method not in ("isotonic", "sigmoid"):
            raise ValueError(f"method must be 'isotonic' or 'sigmoid', got {method!r}")
        if not 0.0 < holdout_fraction < 0.5:
            raise ValueError(f"holdout_fraction must be in (0, 0.5), got {holdout_fraction}")

        self.base = base
        self.method = method
        self.holdout_fraction = holdout_fraction
        self.calibrators_: dict[int, Any] = {}

    def fit(self, x: pd.DataFrame, y: pd.Series) -> CalibratedModel:
        split = int(len(x) * (1 - self.holdout_fraction))
        if split < 100 or len(x) - split < 100:
            raise ValueError(
                f"need at least 100 rows each side of the calibration split, "
                f"got {split} and {len(x) - split}"
            )

        # Chronological, not random: the calibrator must see predictions the
        # model did not fit, and neighbouring rows here are near-duplicates.
        fit_x, fit_y = x.iloc[:split], y.iloc[:split]
        cal_x, cal_y = x.iloc[split:], y.iloc[split:]

        self.base.fit(fit_x, fit_y)
        raw = self.base.predict_proba(cal_x)
        self.calibrators_ = self._fit_calibrators(raw, cal_y.to_numpy())
        self.fitted_ = True
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self._apply(self.base.predict_proba(x))

    # ------------------------------------------------------------------
    def _fit_calibrators(self, raw: np.ndarray, y: np.ndarray) -> dict[int, Any]:
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression

        out: dict[int, Any] = {}
        for column, cls in enumerate(CLASSES):
            target = (y == cls).astype(int)
            scores = raw[:, column]
            # A class absent from the calibration slice gives the calibrator
            # nothing to learn from; leaving it out means that column passes
            # through unchanged, which is the honest default.
            if target.sum() == 0 or target.sum() == len(target):
                continue

            if self.method == "isotonic":
                model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                model.fit(scores, target)
            else:
                model = LogisticRegression(C=1e10, solver="lbfgs")
                model.fit(scores.reshape(-1, 1), target)
            out[int(cls)] = model
        return out

    def _apply(self, raw: np.ndarray) -> np.ndarray:
        out = raw.copy()
        for column, cls in enumerate(CLASSES):
            calibrator = self.calibrators_.get(int(cls))
            if calibrator is None:
                continue
            scores = raw[:, column]
            if self.method == "isotonic":
                out[:, column] = calibrator.predict(scores)
            else:
                out[:, column] = calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]

        # One-vs-rest calibration does not preserve the sum, so renormalise.
        # A row that calibrates to all zeros — possible with isotonic on a
        # sparse class — falls back to HOLD rather than to a division by zero.
        total = out.sum(axis=1, keepdims=True)
        degenerate = (total <= 0).ravel()
        out = np.divide(out, total, out=np.zeros_like(out), where=total > 0)
        out[degenerate] = 0.0
        out[degenerate, CLASSES.index(0)] = 1.0
        return out

    def describe(self) -> dict[str, Any]:
        out = super().describe()
        out.update(
            {
                "method": self.method,
                "holdout_fraction": self.holdout_fraction,
                "calibrated_classes": sorted(self.calibrators_),
                "base": self.base.describe(),
            }
        )
        return out


def reliability(proba: np.ndarray, y: np.ndarray, cls: int, bins: int = 10) -> pd.DataFrame:
    """Predicted probability against observed frequency, in buckets.

    The measurement calibration is judged by. A well-calibrated model puts the
    observed frequency on the diagonal; a boosted tree with class weights
    usually sits well above it, claiming more confidence than it earns.
    """
    column = CLASSES.index(cls)
    frame = pd.DataFrame({"predicted": proba[:, column], "actual": (y == cls).astype(int)})
    frame["bucket"] = pd.qcut(frame["predicted"], q=bins, duplicates="drop")
    grouped = frame.groupby("bucket", observed=True).agg(
        predicted=("predicted", "mean"), observed=("actual", "mean"), n=("actual", "size")
    )
    return grouped.reset_index(drop=True)


def expected_calibration_error(proba: np.ndarray, y: np.ndarray, cls: int, bins: int = 10) -> float:
    """Average gap between predicted and observed frequency, weighted by bucket size."""
    table = reliability(proba, y, cls, bins)
    if table.empty:
        return float("nan")
    weight = table["n"] / table["n"].sum()
    return float((weight * (table["predicted"] - table["observed"]).abs()).sum())
