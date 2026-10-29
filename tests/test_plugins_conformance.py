"""The conformance suite passes honest plugins and fails each kind of look-ahead it names."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from trading_research.plugins import conformance, registry
from trading_research.plugins import page as pages
from trading_research.plugins.market import demo_market, mutate_after


@pytest.fixture(scope="module")
def setup() -> conformance.Setup:
    return conformance.make_setup(
        executions=["taker", "passive"], market=demo_market(steps=24_000, peers=1)
    )


@pytest.fixture
def temporary() -> Iterator[list[tuple[str, str]]]:
    added: list[tuple[str, str]] = []
    yield added
    for kind, name in added:
        registry.unregister(kind, name)


def _register(
    added: list[tuple[str, str]], kind: str, name: str, factory: Any
) -> registry.PluginSpec:
    registry.discover()
    added.append((kind, name))
    return registry.add(kind, name, factory)


def _status(result: conformance.Conformance, check: str) -> str:
    return result.status(check)


class _Imbalance:
    inputs = ("bid_size_0", "ask_size_0")
    lookback = 0
    horizon = 10

    def __init__(self, peek: int = 0) -> None:
        self.peek = peek

    def fit(self, history: pd.DataFrame) -> _Imbalance:
        return self

    def decide(self, frame: pd.DataFrame) -> np.ndarray:
        bid, ask = frame["bid_size_0"].to_numpy(), frame["ask_size_0"].to_numpy()
        value = (bid - ask) / (bid + ask)
        if self.peek:
            value = np.r_[value[self.peek :], np.zeros(self.peek)]
        out = np.zeros(len(frame), dtype=np.int8)
        out[value > 0.5], out[value < -0.5] = 1, -1
        return out


class _Centred(_Imbalance):
    """A z-score on the whole frame's mean: no row read later, the sample's statistics are."""

    inputs = ("mid",)

    def decide(self, frame: pd.DataFrame) -> np.ndarray:
        mid = frame["mid"].to_numpy()
        with np.errstate(invalid="ignore"):
            z = (mid - mid.mean()) / mid.std()
        return np.where(z > 1, 1, np.where(z < -1, -1, 0)).astype(np.int8)


class _Understated(_Imbalance):
    """Reads twenty rows back while declaring none."""

    def decide(self, frame: pd.DataFrame) -> np.ndarray:
        bid = frame["bid_size_0"].rolling(20, min_periods=20).mean().to_numpy()
        ask = frame["ask_size_0"].rolling(20, min_periods=20).mean().to_numpy()
        value = (bid - ask) / (bid + ask)
        return np.where(value > 0.2, 1, np.where(value < -0.2, -1, 0)).astype(np.int8)


def test_an_honest_strategy_passes_and_runs_through_each_execution(
    setup: conformance.Setup, temporary: list[tuple[str, str]]
) -> None:
    result = conformance.check(_register(temporary, "strategy", "honest", _Imbalance), setup)
    assert result.passed, [(r.check, r.detail) for r in result.results]
    assert result.table is not None
    assert list(result.table["execution"]) == ["taker", "passive"]
    assert (result.table["attempts"] > 0).all()
    assert _status(result, "d.") == conformance.PASS


@pytest.mark.parametrize(
    ("factory", "check"),
    [
        (lambda: _Imbalance(peek=1), "b. no look-ahead"),
        (_Centred, "b. no look-ahead"),
        (_Understated, "b. declared look-back"),
    ],
    ids=["one-row-peek", "whole-sample-statistics", "understated-lookback"],
)
def test_a_strategy_that_reads_the_future_fails(
    setup: conformance.Setup, temporary: list[tuple[str, str]], factory: Any, check: str
) -> None:
    result = conformance.check(_register(temporary, "strategy", "leaky", factory), setup)
    assert _status(result, check) == conformance.FAIL
    assert not result.passed


def test_a_strategy_without_its_declarations_fails_the_protocol(
    setup: conformance.Setup, temporary: list[tuple[str, str]]
) -> None:
    class Bare:
        def decide(self, frame: pd.DataFrame) -> np.ndarray:
            return np.zeros(len(frame))

    result = conformance.check(_register(temporary, "strategy", "bare", Bare), setup)
    assert _status(result, "a.") == conformance.FAIL
    assert "lookback" in result.results[0].detail
    assert _status(result, "c.") == conformance.SKIPPED


def test_mutation_leaves_the_rows_up_to_the_cut_alone(setup: conformance.Setup) -> None:
    frame = setup.frame
    row = len(frame) // 2
    mutated = mutate_after(frame, row, seed=3)
    pd.testing.assert_frame_equal(mutated.iloc[: row + 1], frame.iloc[: row + 1], check_exact=True)
    after = mutated.iloc[row + 1 :]
    assert not np.allclose(after["mid"], frame["mid"].iloc[row + 1 :])
    assert not np.allclose(after["bid_size_0"], frame["bid_size_0"].iloc[row + 1 :])
    assert (after["ask_price_0"] > after["bid_price_0"]).all()


class _BatchScaled:
    """Standardises each predict call by that call's own rows: the future enters the past."""

    inputs = ("queue_imbalance",)
    lookback = 0

    def fit(self, x: pd.DataFrame, y: pd.Series) -> _BatchScaled:
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        value = x["queue_imbalance"].to_numpy()
        with np.errstate(invalid="ignore"):
            z = np.tanh((value - value.mean()) / value.std())
        out = np.zeros((len(x), 3))
        out[:, 2] = (1 + z) / 3
        out[:, 0] = (1 - z) / 3
        out[:, 1] = 1 - out[:, 0] - out[:, 2]
        return out


def test_a_model_scaled_on_its_prediction_batch_fails(
    setup: conformance.Setup, temporary: list[tuple[str, str]]
) -> None:
    result = conformance.check(_register(temporary, "model", "batch_scaled", _BatchScaled), setup)
    assert _status(result, "a.") == conformance.PASS
    assert result.results[2].check == "b. no look-ahead: predict"
    assert result.results[2].status == conformance.FAIL


def test_a_built_in_model_passes(setup: conformance.Setup) -> None:
    result = conformance.check(registry.get("order_flow", "model"), setup)
    assert result.passed, [(r.check, r.detail) for r in result.results]


class _CentredFeatures:
    inputs = ("bid_price_0", "ask_price_0")
    lookback = 5

    def compute(self, book: pd.DataFrame, trades: pd.DataFrame | None = None) -> pd.DataFrame:
        mid = (book["bid_price_0"] + book["ask_price_0"]) / 2
        return pd.DataFrame({"centred": mid.rolling(11, center=True).mean()}, index=book.index)


class _LongFeatures(_CentredFeatures):
    def compute(self, book: pd.DataFrame, trades: pd.DataFrame | None = None) -> pd.DataFrame:
        mid = (book["bid_price_0"] + book["ask_price_0"]) / 2
        return pd.DataFrame({"long": mid.rolling(50).mean()}, index=book.index)


def test_feature_sets_fail_a_centred_window_and_an_understated_lookback(
    setup: conformance.Setup, temporary: list[tuple[str, str]]
) -> None:
    centred = conformance.check(
        _register(temporary, "features", "centred", _CentredFeatures), setup
    )
    assert _status(centred, "b. no look-ahead") == conformance.FAIL
    long = conformance.check(_register(temporary, "features", "long", _LongFeatures), setup)
    assert _status(long, "b. no look-ahead") == conformance.PASS
    assert _status(long, "b. declared look-back") == conformance.FAIL
    assert _status(long, "c.") == conformance.NA


class _FarLabel:
    """Declares a ten-row horizon and reads forty."""

    kind = "regression"
    horizon = 10

    def compute(self, frame: pd.DataFrame) -> pd.Series:
        mid = frame["mid"]
        return np.log(mid.shift(-40) / mid) * 1e4


def test_a_label_reading_past_its_horizon_fails(
    setup: conformance.Setup, temporary: list[tuple[str, str]]
) -> None:
    result = conformance.check(_register(temporary, "label", "far", _FarLabel), setup)
    assert _status(result, "b.") == conformance.FAIL
    honest = conformance.check(registry.get("directional", "label"), setup)
    assert honest.passed


class _Clairvoyant:
    """Fills each attempt at the best mid of the next 600 rows: an execution that peeks."""

    fill_rule = "at the best later price"

    def __init__(self, fees: Any = "base") -> None:
        from trading_research.backtest.costs import fee_tier

        self.fees = fee_tier(fees) if isinstance(fees, str) else fees

    def execute(self, decisions: np.ndarray, market: Any, *, hold: int) -> Any:
        from trading_research.pipeline.execution import (
            ATTEMPT_COLUMNS,
            ExecutionMode,
            ExecutionResult,
            summarise,
        )

        mid = market.frame["mid"].to_numpy()
        stamps = market.labels_ns
        rows = []
        for row in np.flatnonzero(decisions):
            best = mid[row : row + 600].max() if decisions[row] > 0 else mid[row : row + 600].min()
            gross = float(decisions[row] * (best / mid[row] - 1) * 1e4)
            day = pd.Timestamp(stamps[row]).strftime("%Y-%m-%d")
            rows.append((market.symbol, day, int(stamps[row]), True, gross, 0.0, gross, "peek"))
        table = pd.DataFrame(rows, columns=list(ATTEMPT_COLUMNS))
        return ExecutionResult(table, summarise(table), ExecutionMode.TAKER)


def test_an_execution_that_reads_later_prices_fails(
    setup: conformance.Setup, temporary: list[tuple[str, str]]
) -> None:
    result = conformance.check(
        _register(temporary, "execution", "clairvoyant", _Clairvoyant), setup
    )
    assert _status(result, "a.") == conformance.PASS
    assert _status(result, "b.") == conformance.FAIL


@pytest.mark.parametrize("name", ["taker", "passive", "maker", "event_taker"])
def test_the_built_in_executions_pass(setup: conformance.Setup, name: str) -> None:
    result = conformance.check(registry.get(name, "execution"), setup)
    assert result.passed, [(r.check, r.detail) for r in result.results]


def test_synthetic_checks_are_skipped_and_said_so_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "trading_research.synthetic", None)
    spec = registry.get("book_imbalance", "strategy")
    [result] = conformance.synthetic_checks(spec, spec.build)
    assert result.status == conformance.SKIPPED
    assert result.detail == "skipped: synthetic checks not on this branch"


def test_synthetic_checks_come_from_the_hook_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    import types

    module = types.ModuleType("trading_research.synthetic")
    seen: dict[str, Any] = {}

    def plugin_checks(*, kind: str, name: str, build: Any) -> list[dict[str, Any]]:
        seen.update(kind=kind, name=name, built=build())
        return [{"check": "planted edge", "status": "pass", "detail": "recovered"}]

    module.plugin_checks = plugin_checks  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "trading_research.synthetic", module)
    spec = registry.get("book_imbalance", "strategy")
    [result] = conformance.synthetic_checks(spec, spec.build)
    assert (result.check, result.status) == ("d. synthetic: planted edge", "pass")
    assert seen["kind"] == "strategy" and seen["name"] == "book_imbalance"


def test_the_page_records_every_check_and_the_execution_table(
    setup: conformance.Setup, tmp_path: Path
) -> None:
    result = conformance.check(registry.get("book_imbalance", "strategy"), setup)
    path = pages.write(result, tmp_path)
    text = path.read_text()
    assert path == tmp_path / "book_imbalance" / "page.md"
    for check in (
        "a. protocol",
        "b. no look-ahead",
        "b. declared look-back",
        "c. execution",
        "d. synthetic",
    ):
        assert f"| {check}" in text
    assert "| taker | base |" in text and "| passive | base |" in text
    assert "sha256 of that file" in text and str(Path.home()) not in text
    index = pages.write_index([pages.summary_row(result)], tmp_path).read_text()
    assert "[book_imbalance](book_imbalance/page.md)" in index
