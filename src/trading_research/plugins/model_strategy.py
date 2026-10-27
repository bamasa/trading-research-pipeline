"""A model plugin turned into a strategy, the way the repository's searches do it.

The model is fitted on the front of the history against a label, the
confidence threshold is chosen on the rest of the history by
:func:`trading_research.backtest.evaluate.choose_confidence` (net of taker
costs at the fee tier, every row scored), and the decision is
:func:`trading_research.backtest.evaluate.decide` at that threshold: the
machinery of ``experiments/strategies.py``, so a model is executed exactly as
the comparisons executed the built-in ones.

The model reads every registered book feature, plus any input it declares
that the book set lacks, taken from the generated set
(:func:`trading_research.features.generated.generate`).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trading_research.plugins.common import book_features, forward_bp, predict_rows

GENERATED_LOOKBACK = 51


def model_features(frame: pd.DataFrame, inputs: tuple[str, ...] = ()) -> tuple[pd.DataFrame, int]:
    """The book features, and any declared input they lack from the generated set."""
    x, lookback = book_features(frame)
    missing = [c for c in inputs if c not in x.columns]
    if missing:
        from trading_research.features.generated import generate

        wide = generate(frame)
        absent = [c for c in missing if c not in wide.columns]
        if absent:
            raise KeyError(f"no registered feature set provides {absent}")
        x = pd.concat([x, wide[missing]], axis=1)
        lookback = max(lookback, GENERATED_LOOKBACK)
    return x, lookback


def model_label(history: pd.DataFrame, target: str, horizon: int) -> pd.Series:
    """The label a model is fitted on: a direction leaving 30% of rows long or short,
    or the forward move in bp for a regression model."""
    if target == "regression":
        return pd.Series(forward_bp(history, horizon), index=history.index)
    from trading_research.labels.directional import DirectionalLabel, threshold_for_share

    threshold = threshold_for_share(history, horizon, 0.3)
    return DirectionalLabel(horizon, threshold)(history)


class ModelStrategy:
    """Decisions from a model's probabilities at a threshold chosen on validation."""

    inputs = ("bid_price_*", "bid_size_*", "ask_price_*", "ask_size_*")

    def __init__(
        self, model: Any, *, horizon: int = 30, fees: Any = None, fit_share: float = 0.7
    ) -> None:
        from trading_research.backtest.costs import BYBIT_BASE

        self.model = model
        self.horizon = horizon
        self.fees = fees or BYBIT_BASE
        self.fit_share = fit_share
        self.target = getattr(model, "target", "classification")
        declared = tuple(getattr(model, "inputs", ()) or ())
        self.model_inputs = declared
        from trading_research.features import REGISTRY

        # The book set's longest look-back covers the generated columns a rule reads.
        book = REGISTRY.max_lookback(REGISTRY.names("book"))
        self.lookback = max(book, GENERATED_LOOKBACK) + int(getattr(model, "lookback", 0))
        self.confidence = float("nan")

    def features(self, frame: pd.DataFrame) -> pd.DataFrame:
        x, _ = model_features(frame, self.model_inputs)
        return x

    def fit(self, history: pd.DataFrame) -> ModelStrategy:
        from trading_research.backtest.costs import TakerCosts
        from trading_research.backtest.evaluate import choose_confidence

        split = int(len(history) * self.fit_share)
        train = history.iloc[: split - self.horizon]
        x = self.features(train)
        y = model_label(train, self.target, self.horizon)
        ok = np.isfinite(x.to_numpy(dtype="float64")).all(axis=1) & np.isfinite(y.to_numpy())
        self.model.fit(x.loc[ok], y.loc[ok])

        proba = predict_rows(self.model, self.features(history))[split:]
        validation = history.iloc[split:]
        move = forward_bp(validation, self.horizon)
        usable = np.isfinite(proba).all(axis=1) & np.isfinite(move)
        costs = TakerCosts(fee_bp_per_side=self.fees.taker_bp)
        self.confidence, _ = choose_confidence(
            proba[usable],
            pd.Series(move[usable]),
            pd.Series(validation["spread_bp"].to_numpy()[usable]),
            costs,
        )
        return self

    def decide(self, frame: pd.DataFrame) -> np.ndarray:
        from trading_research.backtest.evaluate import decide

        proba = predict_rows(self.model, self.features(frame))
        out = np.zeros(len(frame), dtype=np.int8)
        usable = np.isfinite(proba).all(axis=1)
        if usable.any():
            out[usable] = decide(proba[usable], min_confidence=self.confidence)
        return out
