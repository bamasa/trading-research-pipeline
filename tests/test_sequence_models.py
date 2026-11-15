"""The transformer, LSTM, GRU and window MLP: honest, aligned, and able to see a window.

Each is fitted by the TCN's loop, so the TCN's tests of that loop (tail
validation, history, refusals) cover them; these check what each network adds:
that it cannot read forward, that its output lines up with its input, and that
it finds a signal only a window carries.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch", reason="the sequence models need the optional 'deep' extra")

from trading_research.models.base import CLASSES  # noqa: E402
from trading_research.models.sequence import (  # noqa: E402
    GRUClassifier,
    LSTMClassifier,
    TransformerClassifier,
    WindowMLP,
)
from trading_research.pipeline.stages import build_model  # noqa: E402

KINDS = {"transformer": TransformerClassifier, "lstm": LSTMClassifier, "gru": GRUClassifier,
         "mlp": WindowMLP}  # fmt: skip


def quick(kind: str, **overrides: object):  # type: ignore[no-untyped-def]
    params: dict[str, object] = {"window": 16, "channels": 16, "max_epochs": 2, "patience": 1,
                                 "batch_size": 256, "device": "cpu"}  # fmt: skip
    params.update(overrides)
    return KINDS[kind](**params)


@pytest.fixture(scope="module")
def window_signal() -> tuple[pd.DataFrame, pd.Series]:
    """The direction is the sign of the last twenty steps' sum: nothing in one row."""
    rng = np.random.default_rng(0)
    n = 6000
    driver = rng.normal(size=n)
    hidden = pd.Series(driver).rolling(20).sum().to_numpy()
    label = np.where(np.isnan(hidden), np.nan, np.sign(hidden) * (np.abs(hidden) > 3))
    x = pd.DataFrame({"driver": driver, "noise": rng.normal(size=n)})
    ok = ~np.isnan(label)
    return x[ok].reset_index(drop=True), pd.Series(label[ok]).reset_index(drop=True)


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_built_by_name_and_described(kind: str) -> None:
    model = build_model(kind, window=8, channels=8)
    assert type(model) is KINDS[kind]
    assert model.name == kind


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_probabilities_aligned_and_warm_up_neutral(kind: str, window_signal) -> None:
    x, y = window_signal
    model = quick(kind)
    proba = model.fit(x, y).predict_proba(x)
    assert proba.shape == (len(x), len(CLASSES))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(proba[: model.window - 1, CLASSES.index(0)], 1.0)


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_a_later_row_never_moves_an_earlier_prediction(kind: str, window_signal) -> None:
    """Change everything after row 600: predictions up to row 600 stay the same."""
    x, y = window_signal
    model = quick(kind, dropout=0.0).fit(x, y)
    changed = x.copy()
    changed.iloc[601:] = changed.iloc[601:].to_numpy()[::-1] * 3.0
    before, after = model.predict_proba(x), model.predict_proba(changed)
    assert np.allclose(before[:601], after[:601], atol=1e-5)
    assert not np.allclose(before[700:], after[700:], atol=1e-3)


@pytest.mark.parametrize("kind", ["transformer", "lstm", "gru"])
def test_finds_a_signal_only_a_window_can_see(kind: str, window_signal) -> None:
    x, y = window_signal
    model = quick(kind, window=32, channels=32, max_epochs=8, patience=3,
                  learning_rate=3e-3).fit(x, y)  # fmt: skip
    proba = model.predict_proba(x)
    predicted = np.where(proba[:, CLASSES.index(1)] > proba[:, CLASSES.index(-1)], 1, -1)
    scored = (np.arange(len(y)) >= model.window) & (y.to_numpy() != 0)
    assert float((predicted[scored] == y.to_numpy()[scored]).mean()) > 0.7


def test_heads_must_divide_the_width() -> None:
    with pytest.raises(ValueError, match="must divide"):
        TransformerClassifier(channels=10, heads=4)


def test_the_deep_ensemble_averages_four_networks() -> None:
    model = build_model("ensemble_deep", window=8, channels=8)
    assert [m.name for m in model.members] == ["tcn", "transformer", "lstm", "gru"]
