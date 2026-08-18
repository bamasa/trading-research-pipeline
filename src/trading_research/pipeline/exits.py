"""Wiring the exit search to prepared data.

:mod:`trading_research.validation.exits` walks a grid of exit policies and knows
nothing else. This module produces the material it walks over.

Fit once, score many times
--------------------------
An exit rule is applied to trades a model has already produced, so the model
does not need refitting for each policy. The walk-forward runs once and its
predictions are kept; every policy in the grid is then scored against the same
probabilities. Refitting per policy would multiply the sweep by the size of the
grid and change nothing about the answer.

That also fixes what is being compared. If each policy saw a separately fitted
model, differences between policies and differences between fits would be mixed
together, and the network — which moves by more between runs than the policies
differ by — would make the table meaningless.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from trading_research.backtest.costs import TakerCosts
from trading_research.backtest.evaluate import choose_confidence, decide
from trading_research.backtest.execution import score, thin
from trading_research.models.base import CLASSES, clean
from trading_research.pipeline.stages import add_label, build_model
from trading_research.validation.splits import fold_masks

if TYPE_CHECKING:
    from trading_research.validation.exits import ExitPolicy, PolicyEvaluator

#: Column names for the three class probabilities, in CLASSES order.
PROBA_COLUMNS = tuple(f"p_{cls}" for cls in CLASSES)


def walk_forward_predictions(
    frame: pd.DataFrame,
    features: Sequence[str],
    folds: Sequence[Any],
    *,
    threshold_bp: float,
    costs: TakerCosts,
    model: str = "logistic",
    params: dict[str, Any] | None = None,
    calibrate: str | None = None,
    purge: int = 0,
    on_fold: Any = None,
) -> tuple[dict[str, pd.DataFrame], float]:
    """Fit once per fold and return the scored validation and test blocks.

    Each block carries everything an exit rule can read: the price path, the
    realised forward move, the spread and the model's probability for each
    direction at every observation.

    The confidence threshold is chosen on validation, averaged across folds, and
    returned alongside. It is held fixed for the whole exit sweep, because a
    search that moved the entry threshold and the exit rule together would not
    be able to say which one produced the difference.
    """
    columns = list(features)
    labelled = add_label(frame, threshold_bp)
    blocks: dict[str, list[pd.DataFrame]] = {"validation": [], "test": []}
    thresholds: list[float] = []

    for fold in folds:
        masks = fold_masks(fold, labelled["timestamp"], purge=purge)
        train_x, train_y = clean(
            labelled.loc[masks.train, columns], labelled.loc[masks.train, "label"]
        )
        estimator = build_model(model, **(params or {}))
        if calibrate:
            from trading_research.models.calibration import CalibratedModel

            estimator = CalibratedModel(estimator, method=calibrate)
        estimator.fit(train_x, train_y)

        for name, mask in (("validation", masks.validation), ("test", masks.test)):
            block = labelled.loc[mask]
            ok = block[columns].notna().all(axis=1) & block["forward_bp"].notna()
            usable = block.loc[ok]
            proba = estimator.predict_proba(usable[columns])

            scored = pd.DataFrame(
                {
                    "fold": fold.index,
                    "forward_bp": usable["forward_bp"].to_numpy(),
                    "spread_bp_now": usable["spread_bp_now"].to_numpy(),
                    "mid": (
                        usable["mid"].to_numpy()
                        if "mid" in usable.columns
                        else np.full(len(usable), np.nan)
                    ),
                }
            )
            for i, column in enumerate(PROBA_COLUMNS):
                scored[column] = proba[:, i]
            blocks[name].append(scored)

            if name == "validation":
                confidence, _ = choose_confidence(
                    proba, usable["forward_bp"], usable["spread_bp_now"], costs
                )
                thresholds.append(float(confidence))

        if on_fold is not None:
            on_fold(fold)

    return (
        {name: pd.concat(parts, ignore_index=True) for name, parts in blocks.items()},
        float(np.mean(thresholds)),
    )


def policy_evaluator(
    block: pd.DataFrame,
    costs: TakerCosts,
    *,
    min_confidence: float,
) -> PolicyEvaluator:
    """Score one exit policy over pre-computed predictions, fold by fold.

    Folds are thinned separately and their trades pooled. Running the block as
    one series would let a position opened at the end of one fold close inside
    the next, which is not a trade anyone could have taken — and with folds
    stepped one day apart it would also mean holding through a boundary the
    walk-forward exists to enforce.
    """
    buy = PROBA_COLUMNS[CLASSES.index(1)]
    sell = PROBA_COLUMNS[CLASSES.index(-1)]

    def evaluate(policy: ExitPolicy) -> dict[str, float]:
        rules = policy.rules()
        trades = []
        for _, fold in block.groupby("fold", sort=True):
            proba = fold[list(PROBA_COLUMNS)].to_numpy()
            decision = decide(proba, min_confidence=min_confidence)
            trades += thin(
                decision,
                fold["forward_bp"].to_numpy(),
                fold["spread_bp_now"].to_numpy(),
                rules,
                mid=fold["mid"].to_numpy() if rules.needs_price_path else None,
                p_buy=fold[buy].to_numpy() if rules.needs_probabilities else None,
                p_sell=fold[sell].to_numpy() if rules.needs_probabilities else None,
            )

        result = score(trades, costs)
        # Which rule actually fired. A policy whose take-profit never triggers
        # is the baseline wearing a different label, and the summary line alone
        # cannot tell the two apart.
        if trades:
            shares = pd.Series([t.exit_reason for t in trades]).value_counts(normalize=True)
            for reason, share in shares.items():
                result[f"share_{reason}"] = float(share)
        return result

    return evaluate
