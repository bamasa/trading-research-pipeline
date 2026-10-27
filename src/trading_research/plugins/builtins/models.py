"""Every name :func:`trading_research.pipeline.stages.build_model` knows, as a model plugin.

The sixteen single models of ``stages.MODELS`` and the named combinations
(``ENSEMBLES``, ``UNCERTAIN``, ``META``, ``COMPOSITES``) are each registered
under their own name. The adapter builds the model with ``build_model``, so the
object fitted is exactly what ``trading-research train --model <name>`` fits, and
adds only what the protocol declares:

``inputs``
    The columns a rule reads (its ``column``, and the scale or spread column it
    consults); empty for a learned model, which reads every column it is given.
``lookback``
    Earlier rows a prediction reads: the network's window less one, else 0.
``target``
    ``regression`` for the models fitted on a basis-point move
    (``stages.REGRESSION_MODELS`` and the confidence bounds), else
    ``classification``.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from trading_research.pipeline import stages
from trading_research.plugins.registry import add

#: What each rule model reads, beyond nothing.
RULE_INPUTS: dict[str, tuple[str, ...]] = {
    "momentum": ("log_mid_ret20",),
    "mean_reversion": ("log_mid_ret20",),
    "order_flow": ("queue_imbalance",),
    "breakout": ("log_mid_ret50", "log_mid_vol50"),
    "spread_capture": ("queue_imbalance", "spread_bp"),
}

#: Settings the conformance suite fits the networks with, so a check takes a
#: minute rather than an hour. Recorded on each page.
CHECK_PARAMS: dict[str, dict[str, Any]] = {
    "tcn": {"max_epochs": 3},
    "tcn_gaussian": {"epochs": 2},
}

#: One line for each named combination, which has no class of its own.
COMBINATIONS: dict[str, str] = {
    **{n: f"Average of {', '.join(m)}" for n, m in stages.ENSEMBLES.items()},
    **{
        n: f"Confidence bound: mean from {m[0]}, error variance from {m[1]}"
        for n, m in stages.UNCERTAIN.items()
    },
    **{n: f"Meta-labelled: {m[0]} picks the side, {m[1]} decides" for n, m in stages.META.items()},
    **{n: f"{kind.capitalize()}: {', '.join(m)}" for n, (kind, m) in stages.COMPOSITES.items()},
}


class BuiltinModel:
    """A model from ``build_model`` with the protocol's declarations."""

    def __init__(self, name: str, *, inputs: tuple[str, ...], target: str, **params: Any) -> None:
        self.name = name
        self.model = stages.build_model(name, **params)
        self.inputs = inputs
        self.target = target
        self.lookback = max(_windows(self.model), default=1) - 1

    def fit(self, x: pd.DataFrame, y: pd.Series) -> BuiltinModel:
        self.model.fit(x, y)
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict_proba(x), dtype="float64")

    def describe(self) -> dict[str, Any]:
        out: dict[str, Any] = self.model.describe()
        return out


def _windows(model: Any, seen: set[int] | None = None) -> list[int]:
    """The input windows of a model and of every model inside it (an ensemble's
    members, a meta model's two stages, a bound's regressors)."""
    from trading_research.models.base import Model

    seen = set() if seen is None else seen
    if id(model) in seen:
        return []
    seen.add(id(model))
    found = [int(model.window)] if getattr(model, "window", None) else []
    for value in vars(model).values():
        parts = value if isinstance(value, list | tuple) else [value]
        for part in parts:
            if isinstance(part, Model):
                found += _windows(part, seen)
    return found


def _class_of(name: str) -> Any:
    module, cls = stages.MODELS[name].split(":")
    return getattr(importlib.import_module(module), cls)


def _factory(name: str, inputs: tuple[str, ...], target: str) -> Callable[..., BuiltinModel]:
    def build(**params: Any) -> BuiltinModel:
        return BuiltinModel(name, inputs=inputs, target=target, **params)

    build.__doc__ = f"The ``{name}`` model of ``build_model``."
    return build


def _register() -> None:
    names = [*stages.MODELS, *stages.ENSEMBLES, *stages.UNCERTAIN, *stages.META, *stages.COMPOSITES]
    for name in names:
        regression = name in stages.REGRESSION_MODELS or name in stages.UNCERTAIN
        target = "regression" if regression else "classification"
        single = name in stages.MODELS
        add(
            "model",
            name,
            _factory(name, RULE_INPUTS.get(name, ()), target),
            summary=None if single else COMBINATIONS[name],
            check_params=CHECK_PARAMS.get(name),
            defined_by=_class_of(name) if single else stages.build_model,
        )


_register()
