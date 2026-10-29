"""The labels of ``labels/``, registered as label plugins.

``directional``
    :class:`~trading_research.labels.directional.DirectionalLabel`.
the seven targets of :data:`~trading_research.labels.targets.TARGETS`
    Each under its own name, built by
    :func:`~trading_research.labels.targets.build_target` on the frame with a
    ``forward_bp`` column (the mid's log move over the horizon) added.
``walked_taker``, ``taker_forward_move``, ``control_move``
    The labels of :mod:`trading_research.labels.taker_vwap`, whose hold is in
    seconds; the horizon they declare is
    :func:`~trading_research.labels.taker_vwap.purge_rows` on the frame's grid.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from trading_research.labels.targets import TARGETS
from trading_research.plugins.common import forward_bp
from trading_research.plugins.registry import add, register_label


@register_label("directional")
class Directional:
    """Forward mid return over the horizon, cut into sell, hold and buy at a threshold."""

    kind = "classification"

    def __init__(self, *, horizon: int = 24, threshold_bp: float = 11.0) -> None:
        from trading_research.labels.directional import DirectionalLabel

        self.label = DirectionalLabel(horizon=horizon, threshold_bp=threshold_bp)
        self.horizon = horizon

    def compute(self, frame: pd.DataFrame) -> pd.Series:
        return self.label(frame)


class Target:
    """One entry of ``TARGETS``, at a horizon and a cost."""

    def __init__(self, name: str, *, horizon: int = 24, cost_bp: float = 11.0, **params: Any):
        self.name = name
        self.horizon = horizon
        self.cost_bp = cost_bp
        self.params = params
        self.kind = TARGETS[name][0].kind

    def compute(self, frame: pd.DataFrame) -> pd.Series:
        from trading_research.labels.targets import build_target

        with_move = frame.assign(forward_bp=forward_bp(frame, self.horizon))
        values, _ = build_target(
            self.name, with_move, cost_bp=self.cost_bp, horizon=self.horizon, **self.params
        )
        return values


def _target(name: str) -> Callable[..., Target]:
    def build(**params: Any) -> Target:
        return Target(name, **params)

    return build


for _name, (_spec, _function) in TARGETS.items():
    add(
        "label", _name, _target(_name), summary=_spec.description.capitalize(), defined_by=_function
    )


class _Timed:
    def __init__(self, *, hold_s: float = 30.0, grid_seconds: float = 1.0) -> None:
        from trading_research.labels.taker_vwap import purge_rows

        self.hold_s = hold_s
        self.horizon = purge_rows(hold_s, round(grid_seconds * 1000))


@register_label("walked_taker")
class WalkedTaker(_Timed):
    """The walked-book taker label: long, short or hold after both walks and fees."""

    kind = "classification"

    def __init__(self, *, clip: float = 0.01, label_fee_bp: float = 0.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        from trading_research.labels.taker_vwap import WalkedTakerLabel
        from trading_research.plugins.market import depth_of

        self.clip, self.label_fee_bp, self.depth_of = clip, label_fee_bp, depth_of
        self.make = WalkedTakerLabel

    def compute(self, frame: pd.DataFrame) -> pd.Series:
        label = self.make(self.hold_s, self.clip, self.label_fee_bp, depth=self.depth_of(frame))
        return label(frame)


@register_label("taker_forward_move")
class TakerForwardMove(_Timed):
    """The mid's move between the entry and exit snapshots of a taker round trip, bp."""

    kind = "regression"

    def compute(self, frame: pd.DataFrame) -> pd.Series:
        from trading_research.labels.taker_vwap import forward_move_bp

        return forward_move_bp(frame, self.hold_s)


@register_label("control_move")
class ControlMove(_Timed):
    """The smoothed control move: mean mid after the hold over mean mid before, bp."""

    kind = "regression"

    def compute(self, frame: pd.DataFrame) -> pd.Series:
        from trading_research.labels.taker_vwap import control_move_bp

        return control_move_bp(frame, self.hold_s)
