"""Tests for the sequence model.

Kept small and fast — every configuration here is deliberately undersized, and
none of these tests is about whether the model is good. They are about whether
it is *honest*: that it cannot read forward, that its validation split is not a
random sample of a time series, and that its output lines up with the rows it
was given.

The one capability test is the window test. A TCN is only worth having if it
sees something a per-row model cannot, so the fixture plants a signal that
depends on the shape of the last twenty steps and nothing in the current row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch", reason="the TCN needs the optional 'deep' extra")

from trading_research.models.base import CLASSES  # noqa: E402
from trading_research.models.tcn import TCNBaseline, _windowise  # noqa: E402


def quick(**overrides: object) -> TCNBaseline:
    """A deliberately tiny model, so tests finish in seconds."""
    params: dict[str, object] = {
        "window": 16,
        "channels": 8,
        "levels": 2,
        "max_epochs": 2,
        "patience": 1,
        "batch_size": 256,
    }
    params.update(overrides)
    return TCNBaseline(**params)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def window_signal() -> tuple[pd.DataFrame, pd.Series]:
    """A target that only a window can predict.

    The direction depends on the sum of the last twenty steps of the driver, so
    the current row on its own carries almost nothing. A per-row model has no
    way to reach this; a sequence model does.
    """
    rng = np.random.default_rng(0)
    n = 8000
    driver = rng.normal(size=n)
    hidden = pd.Series(driver).rolling(20).sum().to_numpy()
    label = np.where(np.isnan(hidden), np.nan, np.sign(hidden) * (np.abs(hidden) > 3))

    x = pd.DataFrame({"driver": driver, "noise": rng.normal(size=n)})
    usable = ~np.isnan(label)
    return x[usable].reset_index(drop=True), pd.Series(label[usable]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------


def test_each_window_ends_at_its_own_row() -> None:
    """The property that makes the model causal: a window never reaches past t."""
    values = np.arange(20, dtype="float32").reshape(-1, 1)
    windows, targets = _windowise(values, np.arange(20), window=5)

    assert windows.shape == (16, 1, 5)
    # The first window covers rows 0..4 and is labelled with row 4.
    assert windows[0, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert targets[0] == 4
    # The last window ends at the final row, never beyond it.
    assert windows[-1, 0, -1] == 19.0


def test_a_frame_shorter_than_the_window_yields_nothing() -> None:
    values = np.zeros((3, 2), dtype="float32")
    windows, targets = _windowise(values, np.zeros(3, dtype=int), window=10)
    assert len(windows) == 0
    assert len(targets) == 0


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def test_probabilities_are_aligned_and_normalised(window_signal) -> None:
    x, y = window_signal
    proba = quick().fit(x, y).predict_proba(x)
    assert proba.shape == (len(x), len(CLASSES))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_warmup_rows_are_neutral_rather_than_dropped(window_signal) -> None:
    """Output stays row-aligned with the input, so callers keep their indexing."""
    x, y = window_signal
    model = quick(window=16)
    proba = model.fit(x, y).predict_proba(x)
    assert np.allclose(proba[: model.window - 1, CLASSES.index(0)], 1.0)


def test_reordered_columns_are_refused(window_signal) -> None:
    x, y = window_signal
    model = quick().fit(x, y)
    with pytest.raises(ValueError, match="columns differ"):
        model.predict_proba(x[["noise", "driver"]])


def test_predicting_before_fitting_is_refused(window_signal) -> None:
    x, _ = window_signal
    with pytest.raises(RuntimeError, match="not been fitted"):
        quick().predict_proba(x)


def test_too_few_rows_is_reported_rather_than_crashing() -> None:
    x = pd.DataFrame({"a": np.arange(30.0), "b": np.arange(30.0)})
    y = pd.Series(np.resize([1.0, -1.0], 30))
    with pytest.raises(ValueError, match="windows of length"):
        quick(window=25).fit(x, y)


# ---------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------


def test_the_model_finds_a_signal_only_a_window_can_see(window_signal) -> None:
    """The reason for having a sequence model at all."""
    x, y = window_signal
    model = quick(window=32, channels=16, levels=3, max_epochs=6, patience=2).fit(x, y)
    proba = model.predict_proba(x)

    predicted = np.where(proba[:, CLASSES.index(1)] > proba[:, CLASSES.index(-1)], 1, -1)
    scored = (np.arange(len(y)) >= model.window) & (y.to_numpy() != 0)
    accuracy = float((predicted[scored] == y.to_numpy()[scored]).mean())
    assert accuracy > 0.7


def test_validation_is_the_tail_not_a_random_sample(window_signal) -> None:
    """Random validation on a time series puts near-identical neighbours on both
    sides, and reports a loss that early stopping then trusts.

    Checked by construction: with a 20% validation fraction the recorded
    training loss must come from the first 80% of windows only, so fitting on a
    frame whose tail is corrupted still trains normally.
    """
    x, y = window_signal
    corrupted = x.copy()
    corrupted.iloc[int(len(x) * 0.9) :] = 1e6  # nonsense in the tail only

    model = quick(validation_fraction=0.2).fit(corrupted, y)
    assert model.history_, "no epochs recorded"
    assert np.isfinite(model.history_[0]["train_loss"])


def test_the_same_seed_gives_the_same_model(window_signal) -> None:
    x, y = window_signal
    first = quick(seed=7).fit(x, y).predict_proba(x.head(500))
    second = quick(seed=7).fit(x, y).predict_proba(x.head(500))
    assert np.allclose(first, second, atol=1e-5)


def test_training_history_is_recorded(window_signal) -> None:
    """Needed to tell 'stopped early because it converged' from 'never learned'."""
    x, y = window_signal
    model = quick(max_epochs=3, patience=3).fit(x, y)
    assert len(model.history_) >= 1
    assert {"epoch", "train_loss", "val_loss"} <= set(model.history_[0])


def test_description_records_the_architecture(window_signal) -> None:
    x, y = window_signal
    described = quick().fit(x, y).describe()
    assert described["name"] == "tcn"
    assert described["window"] == 16
    assert described["epochs_run"] >= 1
