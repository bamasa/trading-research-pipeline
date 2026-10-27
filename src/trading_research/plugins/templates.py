"""What ``trading-research strategies new`` writes: a plugin that already passes ``check``.

Each template is a working plugin of its kind, small enough to read in a
minute, with its declarations stated and explained, and a test file that
builds it and runs the protocol and look-ahead checks on a small synthetic
market. Replace the body; keep the declarations true; run ``check``.
"""

from __future__ import annotations

import keyword
import re

STRATEGY = '''"""{title}: a strategy plugin. Replace this line with what it does and why.

Written by ``trading-research strategies new {name} --kind strategy``. The
contract (``trading_research.plugins.protocols.Strategy``):

- ``fit(history)`` sees only the rows before trading starts; anything chosen
  from data (a threshold here) is chosen there;
- ``decide(frame)`` returns -1, 0 or +1 per row, and the decision at row t reads
  rows t - lookback .. t and nothing after t;
- ``horizon`` is the clock exit: the rows a position opened at t is held.

Check it with ``trading-research strategies check {name}``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_research.plugins import register_strategy


@register_strategy("{name}")
class {cls}:
    """Follow the mid's move over the look-back when it is in the top share of moves."""

    #: Columns of the frame the decisions read.
    inputs = ("mid",)

    def __init__(self, *, lookback: int = 20, hold: int = 30, share: float = 0.05) -> None:
        #: Rows of history a decision reads, the row itself excluded.
        self.lookback = lookback
        #: Rows a position is held: the clock exit.
        self.horizon = hold
        self.share = share
        self.threshold = float("nan")

    def _move_bp(self, frame: pd.DataFrame) -> np.ndarray:
        mid = frame["mid"].to_numpy(dtype="float64")
        out = np.full(len(mid), np.nan)
        out[self.lookback :] = np.log(mid[self.lookback :] / mid[: len(mid) - self.lookback]) * 1e4
        return out

    def fit(self, history: pd.DataFrame) -> {cls}:
        moves = np.abs(self._move_bp(history))
        self.threshold = float(np.nanquantile(moves, 1.0 - self.share))
        return self

    def decide(self, frame: pd.DataFrame) -> np.ndarray:
        move = self._move_bp(frame)
        out = np.zeros(len(frame), dtype=np.int8)
        strong = np.isfinite(move) & (np.abs(move) >= self.threshold)
        out[strong] = np.sign(move[strong]).astype(np.int8)
        return out
'''

MODEL = '''"""{title}: a model plugin. Replace this line with what it predicts and from what.

Written by ``trading-research strategies new {name} --kind model``. The
contract (``trading_research.plugins.protocols.Model``): ``fit(x, y)`` on a
training block, ``predict_proba(x)`` as an (n, 3) array in the order sell,
hold, buy (``trading_research.models.base.CLASSES``), each row reading only
its own inputs (``lookback = 0``) or the declared number of earlier rows.

Check it with ``trading-research strategies check {name}``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_research.models.base import CLASSES
from trading_research.plugins import register_model


@register_model("{name}")
class {cls}:
    """Class frequencies where the touch imbalance is high, low and in between."""

    #: The columns it reads; ``()`` would mean every column it is given.
    inputs = ("queue_imbalance",)
    #: Earlier rows a prediction reads: none.
    lookback = 0
    #: What it is fitted on: a three-class direction.
    target = "classification"

    def __init__(self, *, bins: int = 3) -> None:
        self.bins = bins
        self.edges: np.ndarray | None = None
        self.table: np.ndarray | None = None

    def fit(self, x: pd.DataFrame, y: pd.Series) -> {cls}:
        values = x[self.inputs[0]].to_numpy(dtype="float64")
        self.edges = np.quantile(values, np.linspace(0, 1, self.bins + 1)[1:-1])
        cells = np.searchsorted(self.edges, values)
        labels = y.to_numpy(dtype="float64")
        table = np.ones((self.bins, len(CLASSES)))  # one pseudo-count each
        for k, cls in enumerate(CLASSES):
            np.add.at(table[:, k], cells[labels == cls], 1.0)
        self.table = table / table.sum(axis=1, keepdims=True)
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if self.edges is None or self.table is None:
            raise RuntimeError("{name} has not been fitted")
        cells = np.searchsorted(self.edges, x[self.inputs[0]].to_numpy(dtype="float64"))
        return self.table[cells]
'''

FEATURES = '''"""{title}: a feature-set plugin. Replace this line with what it measures.

Written by ``trading-research strategies new {name} --kind features``. The
contract (``trading_research.plugins.protocols.FeatureSet``): ``compute(book,
trades)`` from the data contracts (``trading_research.data.schema``) to a frame
on the book's index, the value at row t reading rows t - lookback .. t only.
Warm-up rows are NaN, never a guess.

Check it with ``trading-research strategies check {name}``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_research.plugins import register_features


@register_features("{name}")
class {cls}:
    """The touch's mid move and the touch size's change over a window, in bp and logs."""

    inputs = ("bid_price_0", "ask_price_0", "bid_size_0", "ask_size_0")

    def __init__(self, *, window: int = 20) -> None:
        self.window = window
        #: A change over ``window`` rows reads ``window`` rows before the row.
        self.lookback = window

    def compute(self, book: pd.DataFrame, trades: pd.DataFrame | None = None) -> pd.DataFrame:
        mid = (book["bid_price_0"] + book["ask_price_0"]) / 2.0
        size = book["bid_size_0"] + book["ask_size_0"]
        return pd.DataFrame(
            {{
                f"mid_change_{{self.window}}_bp": np.log(mid / mid.shift(self.window)) * 1e4,
                f"touch_size_change_{{self.window}}": np.log(size / size.shift(self.window)),
            }},
            index=book.index,
        )
'''

LABEL = '''"""{title}: a label plugin. Replace this line with the question it asks.

Written by ``trading-research strategies new {name} --kind label``. The
contract (``trading_research.plugins.protocols.Label``): ``compute(frame)``
reads forward ``horizon`` rows and no further, and is NaN wherever its outcome
is not in the data (the last ``horizon`` rows at least).

Check it with ``trading-research strategies check {name}``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_research.plugins import register_label


@register_label("{name}")
class {cls}:
    """Sign of the mid's move over the horizon where it clears a threshold in bp."""

    kind = "classification"

    def __init__(self, *, horizon: int = 30, threshold_bp: float = 1.0) -> None:
        self.horizon = horizon
        self.threshold_bp = threshold_bp

    def compute(self, frame: pd.DataFrame) -> pd.Series:
        mid = frame["mid"].to_numpy(dtype="float64")
        move = np.full(len(mid), np.nan)
        move[: -self.horizon] = np.log(mid[self.horizon :] / mid[: -self.horizon]) * 1e4
        label = np.where(np.abs(move) > self.threshold_bp, np.sign(move), 0.0)
        return pd.Series(np.where(np.isnan(move), np.nan, label), index=frame.index)
'''

EXECUTION = '''"""{title}: an execution plugin. Replace this line with how it fills.

Written by ``trading-research strategies new {name} --kind execution``. The
contract (``trading_research.plugins.protocols.Execution``):
``execute(decisions, market, hold=)`` turns one decision per row of
``market.frame`` into attempts, under the fill rule it states in ``fill_rule``
and the fee table's row in ``fees``, and returns
``trading_research.pipeline.execution.ExecutionResult`` with the attempts'
columns (``ATTEMPT_COLUMNS``). An attempt's outcome reads the market up to its
exit and nothing after.

Check it with ``trading-research strategies check {name}``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trading_research.backtest.costs import FeeTier, fee_tier
from trading_research.pipeline.execution import (
    ATTEMPT_COLUMNS,
    ExecutionMode,
    ExecutionResult,
    summarise,
)
from trading_research.plugins import register_execution


@register_execution("{name}")
class {cls}:
    """Cross at the touch on the decision row and at the touch ``hold`` rows later."""

    fill_rule = "crosses at the touch on the decision row and back at the touch after the hold"

    def __init__(self, *, fees: FeeTier | str = "base") -> None:
        self.fees = fees if isinstance(fees, FeeTier) else fee_tier(fees)

    def execute(self, decisions: np.ndarray, market: Any, *, hold: int) -> ExecutionResult:
        frame = market.frame
        bid = frame["bid_price_0"].to_numpy(dtype="float64")
        ask = frame["ask_price_0"].to_numpy(dtype="float64")
        stamps = market.labels_ns
        rows, free = [], 0
        for row in np.flatnonzero(np.asarray(decisions) != 0):
            exit_row = row + hold
            if row < free or exit_row >= len(frame):
                continue
            side = int(np.sign(decisions[row]))
            entry = ask[row] if side > 0 else bid[row]
            leave = bid[exit_row] if side > 0 else ask[exit_row]
            gross = side * (leave / entry - 1.0) * 1e4
            cost = 2.0 * self.fees.taker_bp
            day = pd.Timestamp(stamps[row]).strftime("%Y-%m-%d")
            rows.append((market.symbol, day, int(stamps[row]), True, gross, cost, gross - cost, "clock"))
            free = exit_row + 1
        attempts = pd.DataFrame(rows, columns=list(ATTEMPT_COLUMNS))
        return ExecutionResult(attempts, summarise(attempts), ExecutionMode.TAKER)
'''

TEST = '''"""Tests of the {kind} plugin ``{name}``, as ``strategies new`` wrote them.

They build it from the registry and run the conformance checks that do not
need a long market: the protocol and the look-ahead tests. ``trading-research
strategies check {name}`` runs every check on the full demo market.
"""

from __future__ import annotations

import pytest

from trading_research.plugins import conformance, registry
from trading_research.plugins.market import demo_market


@pytest.fixture(scope="module")
def setup() -> conformance.Setup:
    return conformance.make_setup(executions=["taker"], market=demo_market(steps=24_000, peers=1))


def test_{ident}_is_registered() -> None:
    spec = registry.get("{name}", "{kind}")
    assert spec.origin == "contrib"


def test_{ident}_passes_its_checks(setup: conformance.Setup) -> None:
    result = conformance.check(registry.get("{name}", "{kind}"), setup)
    failed = [(r.check, r.detail) for r in result.results if r.status == conformance.FAIL]
    assert not failed, failed
'''

TEMPLATES: dict[str, str] = {
    "strategy": STRATEGY,
    "model": MODEL,
    "features": FEATURES,
    "label": LABEL,
    "execution": EXECUTION,
}


def check_name(name: str) -> str:
    """A plugin name usable as a module name: lower case, digits, underscores."""
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or keyword.iskeyword(name):
        raise ValueError(
            f"{name!r} is not a usable plugin name: start with a letter, then lower-case "
            "letters, digits and underscores"
        )
    return name


def class_name(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_"))


def render(kind: str, name: str) -> tuple[str, str]:
    """The plugin module and its test module."""
    name = check_name(name)
    fields = {"name": name, "cls": class_name(name), "title": name.replace("_", " ").capitalize()}
    module = TEMPLATES[kind].format(**fields)
    test = TEST.format(name=name, kind=kind, ident=name)
    return module, test
