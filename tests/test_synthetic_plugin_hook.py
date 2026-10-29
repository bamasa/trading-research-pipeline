"""The hook the plugin conformance suite calls: planted edge, null and placebo for a plugin."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
import pytest

from trading_research import synthetic
from trading_research.plugins import conformance, registry


def _rows(kind: str, name: str, factory: Any) -> dict[str, dict[str, Any]]:
    return {r["check"]: r for r in synthetic.plugin_checks(kind=kind, name=name, build=factory)}


class _Peeking:
    """Trades the sign of the mid's move over the next 30 rows: it reads the future."""

    inputs = ("mid",)
    lookback = 0
    horizon = 30

    def fit(self, history: pd.DataFrame) -> _Peeking:
        return self

    def decide(self, frame: pd.DataFrame) -> np.ndarray:
        mid = frame["mid"].to_numpy(dtype=np.float64)
        ahead = np.full(len(mid), np.nan)
        ahead[:-30] = np.log(mid[30:] / mid[:-30]) * 1e4
        out = np.where(ahead > 3.0, 1, np.where(ahead < -3.0, -1, 0))
        out[np.arange(len(out)) % 20 != 0] = 0
        return out.astype(np.int8)


@pytest.fixture
def temporary() -> Iterator[list[tuple[str, str]]]:
    added: list[tuple[str, str]] = []
    yield added
    for kind, name in added:
        registry.unregister(kind, name)


def test_the_suite_finds_the_hook_and_it_judges_the_worked_example() -> None:
    spec = registry.get("book_imbalance", "strategy")
    found = conformance.synthetic_checks(spec, spec.build)
    assert {r.check for r in found} == {
        "d. synthetic: planted edge",
        "d. synthetic: no leak (planted days)",
        "d. synthetic: no leak (null days)",
        "d. synthetic: null",
        "d. synthetic: placebo",
    }
    assert all(r.status == conformance.PASS for r in found), [(r.check, r.detail) for r in found]
    assert "taker fee 5.5 bp a side" in found[0].detail


def test_a_strategy_reading_the_future_fails_the_leak_row() -> None:
    rows = _rows("strategy", "peeking", _Peeking)
    leak = rows["no leak (planted days)"]
    assert leak["status"] == "fail", leak
    assert leak["numbers"]["residual_t"] >= 3


def test_a_cross_sectional_strategy_is_not_applicable_and_said_so() -> None:
    rows = _rows("strategy", "relative", registry.get("relative", "strategy").build)
    assert list(rows) == ["planted edge"]
    assert rows["planted edge"]["status"] == "n/a"
    assert "peer_*_mid" in rows["planted edge"]["detail"]


def test_a_model_is_put_through_on_its_declared_inputs() -> None:
    rows = _rows("model", "order_flow", registry.get("order_flow", "model").build)
    assert rows["planted edge"]["status"] == "pass", rows["planted edge"]
    assert rows["null"]["status"] == "pass"


def test_kinds_that_make_no_decisions_and_broken_protocols_are_not_applicable() -> None:
    assert _rows("features", "book", lambda: None)["planted edge"]["status"] == "n/a"

    class Bare:
        def decide(self, frame: pd.DataFrame) -> np.ndarray:
            return np.zeros(len(frame))

    detail = _rows("strategy", "bare", Bare)["planted edge"]["detail"]
    assert detail.startswith("n/a: the protocol is not met") and "lookback" in detail
