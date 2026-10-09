"""Tests for the network with a Gaussian head.

Undersized throughout, on the CPU. They check that the loss is the registered
one, that the head cannot leave its band, that a prediction never reads a later
row, that the checkpoint is the validation's best epoch, and that a fit is
reproduced bit for bit from its seed.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch", reason="the network needs the optional 'deep' extra")

from trading_research.models.base import CLASSES  # noqa: E402
from trading_research.models.tcn import (  # noqa: E402
    GaussianTCN,
    _build_gaussian_net,
    _input_moments,
    gaussian_nll_loss,
    rank_ic,
    seed_everything,
)
from trading_research.models.windows import Day, WindowDataset  # noqa: E402

SMALL = {"window": 16, "channels": 8, "levels": 2, "batch_size": 128, "device": "cpu"}


def small(**overrides: object) -> GaussianTCN:
    params: dict[str, object] = {**SMALL, "max_epochs": 3, "patience": 1}
    params.update(overrides)
    return GaussianTCN(**params)  # type: ignore[arg-type]


def make_frame(n: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    """A move set by the last ten steps of a driver, with noise scaled by a known input."""
    rng = np.random.default_rng(seed)
    driver = rng.normal(size=n)
    hidden = pd.Series(driver).rolling(10).sum().to_numpy()
    vol = 1.0 + 2.0 * (rng.random(n) > 0.7)
    move = 2.0 * hidden + rng.normal(size=n) * vol * 2.0
    x = pd.DataFrame({"driver": driver * 50 + 1000, "vol": vol, "noise": rng.normal(size=n)})
    return x, pd.Series(move)


@pytest.fixture(scope="module")
def data() -> dict[str, object]:
    x, y = make_frame(3000, 0)
    vx, vy = make_frame(1000, 1)
    return {"x": x, "y": y, "vx": vx, "vy": vy}


@pytest.fixture(scope="module")
def fitted(data) -> GaussianTCN:
    return small().fit(data["x"], data["y"], validation=(data["vx"], data["vy"]))


# ---------------------------------------------------------------------------
# Loss and head
# ---------------------------------------------------------------------------


def test_the_loss_is_the_registered_one() -> None:
    mu = torch.tensor([0.5, -1.0])
    log_var = torch.tensor([0.0, 1.0])
    target = torch.tensor([1.0, 1.0])
    expected = np.mean([0.5 * (0.0 + 0.25), 0.5 * (1.0 + 4.0 * math.exp(-1.0))]) + 0.01 * np.mean(
        [(0.0 - 0.3) ** 2, (1.0 - 0.3) ** 2]
    )
    loss = gaussian_nll_loss(mu, log_var, target, log_v=0.3, prior_weight=0.01)
    assert float(loss) == pytest.approx(expected, rel=1e-6)


def test_the_log_variance_cannot_leave_its_band() -> None:
    seed_everything(0)
    net = _build_gaussian_net(
        n_features=2,
        channels=4,
        levels=1,
        kernel_size=3,
        dropout=0.0,
        mean=np.zeros(2),
        scale=np.ones(2),
        log_v=1.5,
        band=0.5,
    )
    with torch.no_grad():
        net.core.head.bias.copy_(torch.tensor([0.0, 50.0]))
        _, high = net(torch.randn(3, 2, 8))
        net.core.head.bias.copy_(torch.tensor([0.0, -50.0]))
        _, low = net(torch.randn(3, 2, 8))
    assert torch.allclose(high, torch.full((3,), 2.0))
    assert torch.allclose(low, torch.full((3,), 1.0))


def test_an_untrained_head_starts_at_the_prior() -> None:
    seed_everything(0)
    net = _build_gaussian_net(
        n_features=2,
        channels=4,
        levels=1,
        kernel_size=3,
        dropout=0.0,
        mean=np.zeros(2),
        scale=np.ones(2),
        log_v=0.7,
        band=6.0,
    )
    mu, log_var = net(torch.randn(5, 2, 8))
    assert torch.all(mu == 0)
    assert torch.allclose(log_var, torch.full((5,), 0.7))


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def test_it_learns_the_move_and_its_spread(fitted, data) -> None:
    mu, sigma = fitted.predict_gaussian(data["vx"])
    ok = np.isfinite(mu)
    assert rank_ic(mu, data["vy"].to_numpy()) > 0.3
    assert np.corrcoef(sigma[ok], data["vx"]["vol"].to_numpy()[ok])[0, 1] > 0.2


def test_the_checkpoint_is_the_best_validation_epoch(fitted) -> None:
    ics = [h["validation_ic"] for h in fitted.history_]
    assert fitted.checkpoint_epoch_ == int(np.argmax(ics)) + 1
    run = len(fitted.history_)
    assert run == fitted.max_epochs or run - fitted.checkpoint_epoch_ == fitted.patience


def test_a_fit_is_reproduced_from_its_seed_and_its_checkpoint_epoch(fitted, data) -> None:
    again = small().fit(data["x"], data["y"], validation=(data["vx"], data["vy"]))
    assert again.weights_sha256() == fitted.weights_sha256()
    replay = small(epochs=fitted.checkpoint_epoch_).fit(data["x"], data["y"])
    assert replay.weights_sha256() == fitted.weights_sha256()
    other = small(seed=7, epochs=1).fit(data["x"], data["y"])
    assert other.weights_sha256() != small(epochs=1).fit(data["x"], data["y"]).weights_sha256()


def test_a_fit_without_validation_or_epochs_is_refused(data) -> None:
    with pytest.raises(ValueError, match="validation days"):
        small().fit(data["x"], data["y"])


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def test_a_prediction_never_reads_a_later_row(fitted, data) -> None:
    x = data["vx"].copy()
    before, _ = fitted.predict_gaussian(x)
    x.iloc[500:] = x.iloc[500:].to_numpy() * -3.0 + 7.0
    after, _ = fitted.predict_gaussian(x)
    np.testing.assert_array_equal(before[:500], after[:500])
    assert not np.allclose(before[500:], after[500:], equal_nan=True)


def test_rows_without_a_window_are_unscored_and_neutral(fitted, data) -> None:
    mu, sigma = fitted.predict_gaussian(data["vx"])
    assert np.isnan(mu[: fitted.window - 1]).all() and np.isnan(sigma[: fitted.window - 1]).all()
    assert np.isfinite(mu[fitted.window - 1 :]).all()
    proba = fitted.predict_proba(data["vx"])
    assert proba.shape == (len(data["vx"]), 3)
    assert np.allclose(proba[: fitted.window - 1, CLASSES.index(0)], 1.0)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_days_give_the_frame_s_predictions_across_a_contiguous_midnight(fitted, data) -> None:
    values = data["vx"].to_numpy(dtype="float32")
    first, second = values[:400], values[400:]
    days = [
        Day("a", lambda: first, np.arange(len(first))),
        Day("b", lambda: second, np.arange(len(second)), follows_previous=True),
    ]
    by_day = fitted.predict_days(days, batch_size=97)
    whole, _ = fitted.predict_gaussian(data["vx"])
    joined = np.concatenate([by_day[0][0], by_day[1][0]])
    np.testing.assert_allclose(joined, whole, rtol=1e-5, atol=1e-5, equal_nan=True)


def test_reordered_columns_and_unfitted_models_are_refused(fitted, data) -> None:
    with pytest.raises(ValueError, match="columns differ"):
        fitted.predict_gaussian(data["vx"][["vol", "driver", "noise"]])
    with pytest.raises(RuntimeError, match="not been fitted"):
        small().predict_gaussian(data["vx"])


def test_saved_weights_load_to_the_same_predictions(fitted, data, tmp_path) -> None:
    path = fitted.save(tmp_path / "net.pt")
    loaded = GaussianTCN.load(path)
    assert loaded.weights_sha256() == fitted.weights_sha256()
    assert loaded.checkpoint_epoch_ == fitted.checkpoint_epoch_
    np.testing.assert_array_equal(
        loaded.predict_mean_bp(data["vx"]), fitted.predict_mean_bp(data["vx"])
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_input_moments_keep_their_digits_far_from_zero() -> None:
    rng = np.random.default_rng(3)
    parts = [1e5 + rng.normal(size=(500, 2)) * [1e-3, 1.0] for _ in range(3)]
    days = [Day(f"d{i}", lambda p=p: p, np.arange(500)) for i, p in enumerate(parts)]
    mean, scale = _input_moments(WindowDataset(days, window=1))
    stacked = np.concatenate(parts)
    np.testing.assert_allclose(mean, stacked.mean(axis=0), rtol=1e-12)
    np.testing.assert_allclose(scale, stacked.std(axis=0), rtol=1e-6)


def test_rank_ic_ignores_missing_rows_and_is_nan_when_undefined() -> None:
    a = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    b = np.array([10.0, 20.0, 30.0, np.nan, 50.0])
    assert rank_ic(a, b) == pytest.approx(1.0)
    assert math.isnan(rank_ic(np.ones(5), np.arange(5.0)))
    assert math.isnan(rank_ic(np.arange(2.0), np.arange(2.0)))


def test_the_network_is_registered_as_a_regression_model() -> None:
    from trading_research.pipeline.stages import REGRESSION_MODELS, build_model

    model = build_model("tcn_gaussian", scale_bp=5.0)
    assert isinstance(model, GaussianTCN)
    assert model.scale_bp == 5.0
    assert "tcn_gaussian" in REGRESSION_MODELS
