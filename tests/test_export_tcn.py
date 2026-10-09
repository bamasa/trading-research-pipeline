"""Tests for exporting the Gaussian-head network to ONNX.

The exported graph must read the same raw windows the network reads and give
back the same mean and log-variance, standardisation and clamp included, and an
export that changes a decision must not be left on disk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch", reason="the network needs the optional 'deep' extra")
pytest.importorskip("onnxruntime", reason="ONNX export is an optional extra")
pytest.importorskip("onnx", reason="the torch exporter needs onnx")

from trading_research.models.export import ExportError, export_gaussian_onnx  # noqa: E402
from trading_research.models.tcn import GaussianTCN  # noqa: E402


@pytest.fixture(scope="module")
def fitted() -> tuple[GaussianTCN, pd.DataFrame]:
    rng = np.random.default_rng(0)
    n = 1500
    driver = rng.normal(size=n)
    move = 3.0 * pd.Series(driver).rolling(8).sum().fillna(0).to_numpy() + rng.normal(size=n)
    # Inputs far from zero, so the standardisation inside the graph is exercised.
    x = pd.DataFrame({"driver": driver * 20 + 3000.0, "other": rng.normal(size=n)})
    model = GaussianTCN(window=16, channels=8, levels=2, batch_size=128, epochs=2, device="cpu")
    return model.fit(x.iloc[:1000], pd.Series(move[:1000])), x.iloc[1000:].reset_index(drop=True)


def test_the_exported_network_reproduces_the_original(fitted, tmp_path) -> None:
    model, sample = fitted
    mu, _ = model.predict_gaussian(sample)
    # A cost at which about half the rows trade, so the decision count is
    # exercised rather than trivially zero; halfway between two rows, since a
    # row exactly on the cost is the case the check exists to flag.
    ordered = np.sort(np.abs(mu[np.isfinite(mu)]))
    cost = float(ordered[len(ordered) // 2 - 1 : len(ordered) // 2 + 1].mean())
    trades = np.abs(mu[np.isfinite(mu)]) > cost
    assert 0 < trades.sum() < len(trades)
    report = export_gaussian_onnx(model, sample, tmp_path / "net.onnx", cost_bp=cost, k=0.0)
    assert report.ok
    assert report.rows_checked == len(sample) - model.window + 1
    assert report.max_absolute_difference < 1e-4
    assert report.max_decision_difference == 0
    assert (tmp_path / "net.onnx").exists()


def test_a_disagreeing_export_is_deleted_rather_than_left(fitted, tmp_path) -> None:
    model, sample = fitted
    with pytest.raises(ExportError, match="does not reproduce"):
        export_gaussian_onnx(model, sample, tmp_path / "net.onnx", tolerance=-1.0)
    assert not (tmp_path / "net.onnx").exists()


def test_a_sample_too_short_for_one_window_is_refused(fitted, tmp_path) -> None:
    model, sample = fitted
    with pytest.raises(ExportError, match="complete window"):
        export_gaussian_onnx(model, sample.head(model.window - 1), tmp_path / "net.onnx")


def test_an_unfitted_model_is_refused(tmp_path) -> None:
    with pytest.raises(ExportError, match="not a fitted network"):
        export_gaussian_onnx(GaussianTCN(), pd.DataFrame({"a": [1.0]}), tmp_path / "n.onnx")
