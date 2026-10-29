"""The signal a search step scores: features, labels at a hold, a walk-forward fit, decisions.

Everything here is a composition of the repository's own stages and
evaluators, so a candidate is scored exactly as ``exit-search`` scores one:

* features come from :func:`~trading_research.pipeline.stages.prepare` (one
  file per day, rolling windows warmed across days) and are chosen by
  :func:`~trading_research.pipeline.stages.select` on a training window only;
* a candidate (a hold, a label threshold, a rule or a model with its
  parameters) is fitted fold by fold by
  :func:`~trading_research.pipeline.exits.walk_forward_predictions`, its entry
  threshold chosen on each fold's validation day, and scored on validation by
  :func:`~trading_research.pipeline.exits.policy_evaluator` under the clock
  exit at the hold: one position at a time, a taker round trip per trade;
* the chosen signal is then fitted once with
  :func:`~trading_research.pipeline.stages.train`, scored with
  :func:`~trading_research.pipeline.stages.predict`, and turned into decisions by
  :func:`~trading_research.backtest.evaluate.decide` at a threshold chosen on a
  day the execution is not scored on.

The forward move at each hold is :func:`relabel`'s: the same log move of the
mid ``prepare`` writes, within each day, recomputed from the ``mid`` it carries.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_research.backtest.costs import TakerCosts
from trading_research.backtest.evaluate import choose_confidence, decide
from trading_research.models.base import CLASSES
from trading_research.pipeline.exits import policy_evaluator, walk_forward_predictions
from trading_research.validation.exits import ExitPolicy
from trading_research.validation.splits import WalkForwardSpec, walk_forward

#: The rules :mod:`trading_research.models.classical` defines: fitted to nothing.
RULES: tuple[str, ...] = ("order_flow", "momentum", "mean_reversion", "breakout", "spread_capture")
#: The learned models the method step compares against them, each at its defaults.
MODELS: tuple[str, ...] = ("logistic", "lightgbm", "xgboost")
#: Parameters the model step tries for each learned model, its defaults first.
PARAMETERS: dict[str, tuple[dict[str, Any], ...]] = {
    "logistic": ({}, {"C": 0.01}, {"C": 0.1}, {"C": 10.0}),
    "lightgbm": (
        {},
        {"num_leaves": 7, "n_estimators": 100},
        {"num_leaves": 31, "n_estimators": 300},
    ),
    "xgboost": (
        {},
        {"max_depth": 2, "n_estimators": 150, "learning_rate": 0.1},
        {"max_depth": 6, "n_estimators": 300, "learning_rate": 0.03},
    ),
}


@dataclass(frozen=True)
class Candidate:
    """One point the search scores: a hold, a label threshold, a rule or a model and its parameters."""

    hold_rows: int
    threshold_bp: float
    model: str
    params: tuple[tuple[str, Any], ...] = ()

    @property
    def family(self) -> str:
        return "rule" if self.model in RULES else "model"

    @property
    def label(self) -> str:
        params = ",".join(f"{k}={v}" for k, v in self.params) or "defaults"
        return f"{self.model}[{params}]/hold {self.hold_rows * 5}s/label > {self.threshold_bp:g} bp"


def relabel(frame: pd.DataFrame, hold_rows: int) -> pd.DataFrame:
    """``frame`` with ``forward_bp`` the log move of the mid over ``hold_rows`` rows,
    within each day (the last rows of a day have none), as ``prepare`` computes it."""
    out = frame.copy()
    mid = out["mid"].to_numpy(dtype=np.float64)
    day = out["timestamp"].dt.date.to_numpy()
    forward = np.full(len(out), np.nan)
    later = np.arange(len(out)) + hold_rows
    inside = later < len(out)
    same = np.zeros(len(out), dtype=bool)
    same[inside] = day[later[inside]] == day[inside]
    forward[same] = np.log(mid[later[same]] / mid[same]) * 1e4
    out["forward_bp"] = forward
    return out


def spec_for(days: int, folds: int = 3) -> WalkForwardSpec:
    """A walk-forward over ``days`` search days: one validation and one test day per
    fold, stepped a day, every remaining day in training."""
    folds = max(1, min(folds, days - 3))
    train = days - 2 - (folds - 1)
    if train < 1:
        raise ValueError(
            f"{days} search day(s) cannot hold a fold of training, validation and test"
        )
    return WalkForwardSpec(
        train_days=train, validation_days=1, test_days=1, step_days=1, n_folds=folds
    )


def evaluate(
    frame: pd.DataFrame,
    columns: Sequence[str],
    days: Sequence[date],
    candidate: Candidate,
    costs: TakerCosts,
) -> dict[str, float]:
    """One candidate on the search block's validation days, as ``exit-search`` scores a policy."""
    labelled = relabel(frame, candidate.hold_rows)
    schedule = walk_forward(list(days), spec_for(len(days)))
    columns = list(columns)
    blocks, confidence = walk_forward_predictions(
        labelled,
        columns,
        schedule,
        threshold_bp=candidate.threshold_bp,
        costs=costs,
        model=candidate.model,
        params=dict(candidate.params) or None,
        purge=candidate.hold_rows,
    )
    metrics = policy_evaluator(blocks["validation"], costs, min_confidence=confidence)(
        ExitPolicy(hold_periods=candidate.hold_rows)
    )
    return {**metrics, "min_confidence": float(confidence)}


def scored_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Candidates' rows as a table, the failures kept with their reason."""
    return pd.DataFrame(rows)


def neighbourhood(table: pd.DataFrame, first: str, second: str) -> pd.DataFrame:
    """``reproduce``'s neighbourhood rule on two axes of any name.

    :func:`trading_research.pipeline.discovery._with_neighbourhood` scores each
    cell by the median of itself and its neighbours along its two axes, at the
    same rate; here the axes are renamed onto its own and back, at one rate.
    """
    from trading_research.pipeline.discovery import _with_neighbourhood

    renamed = table.rename(columns={first: "lookback_s", second: "hold_s"}).assign(rate=0.0)
    scored = _with_neighbourhood(renamed)
    return scored.rename(columns={"lookback_s": first, "hold_s": second}).drop(columns="rate")


def choose(
    table: pd.DataFrame, objective: str = "neighbourhood_bp"
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The chosen row (best ``objective``, the cell's own net as the tie-break, as
    ``reproduce`` ranks) and the outright peak of the cell's own net."""
    ranked = table.sort_values([objective, "net_per_trade_bp"], na_position="first")
    winner = {str(k): v for k, v in ranked.iloc[-1].to_dict().items()}
    peak = {str(k): v for k, v in table.loc[table["net_per_trade_bp"].idxmax()].to_dict().items()}
    return winner, peak


def fit_signal(
    prepared: Path,
    features: Path,
    out: Path,
    *,
    train: tuple[date, date],
    candidate: Candidate,
) -> Path:
    """The chosen signal fitted once on ``train`` by the ``train`` stage."""
    from trading_research.pipeline.stages import train as train_stage

    return train_stage(
        prepared, features, out, train_start=train[0], train_end=train[1],
        threshold_bp=candidate.threshold_bp, model=candidate.model,
        params=dict(candidate.params) or None,
    )  # fmt: skip


def predictions(prepared: Path, model: Path, out: Path, days: Sequence[date]) -> pd.DataFrame:
    """The ``predict`` stage over ``days``, read back."""
    from trading_research.pipeline.stages import load_predictions, predict

    return load_predictions(predict(prepared, model, out, start=min(days), end=max(days)))


def confidence_on(frame: pd.DataFrame, costs: TakerCosts) -> float:
    """The entry threshold chosen on ``frame``'s days, which nothing is then scored on."""
    proba = frame[[f"p_{c}" for c in CLASSES]].to_numpy()
    threshold, _ = choose_confidence(proba, frame["forward_bp"], frame["spread_bp_now"], costs)
    return float(threshold)


def decisions(frame: pd.DataFrame, min_confidence: float) -> np.ndarray:
    """-1 / 0 / +1 per row, by :func:`decide`."""
    proba = frame[[f"p_{c}" for c in CLASSES]].to_numpy()
    return decide(proba, min_confidence=min_confidence)
