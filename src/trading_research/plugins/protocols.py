"""What a plugin of each kind promises, as small structural protocols.

A plugin does not subclass anything: any object with these attributes and
methods is one. The declarations are claims the conformance suite
(:mod:`trading_research.plugins.conformance`) tests rather than trusts.

Rows and time
-------------
A **frame** is one instrument on a regular grid, one row per label, each row
labelled by the end of its bin so that a row holds only what had been observed
at its label (:func:`trading_research.data.grid.to_grid` with ``label="right"``).
Its columns are the book plane's level columns, ``mid``, ``log_mid``,
``spread_bp``, the prints each row owns under the backward rule
(:func:`trading_research.features.prints.bucket_prints`) and, for strategies
that read a cross-section, ``peer_<k>_mid`` columns of other instruments on the
same labels (:mod:`trading_research.plugins.market`). Look-backs and horizons
are counted in rows of that frame.

The five kinds
--------------
``Strategy``
    ``fit(history)`` sees only the rows before trading starts; ``decide(frame)``
    returns -1, 0 or +1 per row, and the decision at row *t* may read rows
    ``t - lookback`` to *t* and nothing after *t*. ``horizon`` is the clock exit:
    the rows a position opened at *t* is held. A strategy whose decisions carry
    state from row to row (a band it stays inside, a position it remembers) says
    so with ``stateful = True``; its look-back then bounds the signal, not the
    memory.
``Model``
    ``fit(x, y)`` on a training block, ``predict_proba(x)`` as an ``(n, 3)``
    array in the order of :data:`trading_research.models.base.CLASSES` (sell,
    hold, buy). ``inputs`` names the columns it reads (empty: every column it is
    given); ``lookback`` the earlier rows a prediction reads (0: the row alone).
``FeatureSet``
    ``compute(book, trades)`` from the data contracts
    (:mod:`trading_research.data.schema`) to a frame on the book's index, the
    value at row *t* reading rows ``t - lookback`` to *t*.
``Label``
    ``compute(frame)`` reading forward ``horizon`` rows and no further; NaN where
    the outcome is not in the data.
``Execution``
    ``execute(decisions, market, hold=)`` turns per-row decisions into attempts
    under a fill rule (``fill_rule``) and a fee table row (``fees``), and
    returns :class:`trading_research.pipeline.execution.ExecutionResult`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from trading_research.backtest.costs import FeeTier
    from trading_research.pipeline.execution import ExecutionResult
    from trading_research.plugins.market import MarketData


@runtime_checkable
class Strategy(Protocol):
    """Decisions from a frame, causal in its rows."""

    lookback: int
    horizon: int
    inputs: tuple[str, ...]

    def fit(self, history: pd.DataFrame) -> Any: ...

    def decide(self, frame: pd.DataFrame) -> np.ndarray: ...


@runtime_checkable
class Model(Protocol):
    """A three-class predictor fitted on a training block."""

    inputs: tuple[str, ...]
    lookback: int

    def fit(self, x: pd.DataFrame, y: pd.Series) -> Any: ...

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray: ...


@runtime_checkable
class FeatureSet(Protocol):
    """Columns computed from the book and trades planes."""

    lookback: int
    inputs: tuple[str, ...]

    def compute(self, book: pd.DataFrame, trades: pd.DataFrame | None = None) -> pd.DataFrame: ...


@runtime_checkable
class Label(Protocol):
    """A target that reads forward exactly its horizon."""

    horizon: int
    kind: str

    def compute(self, frame: pd.DataFrame) -> pd.Series: ...


@runtime_checkable
class Execution(Protocol):
    """Decisions into fills, under a fill rule and a fee table."""

    fill_rule: str
    fees: FeeTier

    def execute(
        self, decisions: np.ndarray, market: MarketData, *, hold: int
    ) -> ExecutionResult: ...


#: Each kind's protocol, the attributes it declares and the methods it calls.
PROTOCOLS: dict[str, type] = {
    "strategy": Strategy,
    "model": Model,
    "features": FeatureSet,
    "label": Label,
    "execution": Execution,
}

DECLARED: dict[str, tuple[str, ...]] = {
    "strategy": ("lookback", "horizon", "inputs"),
    "model": ("inputs", "lookback"),
    "features": ("lookback", "inputs"),
    "label": ("horizon", "kind"),
    "execution": ("fill_rule", "fees"),
}

METHODS: dict[str, tuple[str, ...]] = {
    "strategy": ("fit", "decide"),
    "model": ("fit", "predict_proba"),
    "features": ("compute",),
    "label": ("compute",),
    "execution": ("execute",),
}
