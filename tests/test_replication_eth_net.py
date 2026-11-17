"""The replication's network and training loop, on small inputs."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from trading_research.replication_eth.net import WINDOW, DeepTCN  # noqa: E402
from trading_research.replication_eth.train import (  # noqa: E402
    IGNORE,
    class_weights,
    fit,
    predict,
    targets,
)


def test_network_shape_and_causal_stack() -> None:
    torch.manual_seed(0)
    net = DeepTCN(56).eval()
    x = torch.randn(4, 56, WINDOW)
    assert net(x).shape == (4, 3)
    later = x.clone()
    later[:, :, 30:] += 1.0
    with torch.no_grad():
        np.testing.assert_allclose(
            net.tcn(x)[:, :, :30].numpy(), net.tcn(later)[:, :, :30].numpy(), atol=1e-6
        )
    blocks = list(net.tcn)
    assert [b.body[0].dilation[0] for b in blocks] == [1, 2, 4, 8, 16]
    assert blocks[0].shortcut is not None and blocks[1].shortcut is None


def test_targets_and_weights() -> None:
    y = targets(np.array([1, 2, 3, 2, 2]), np.array([1e-4, 1e-4, 1e-4, 1e-4, 1e-6]), 2e-5)
    np.testing.assert_array_equal(y, [0, 1, 2, 1, IGNORE])
    w = class_weights(np.array([0, 1, 1, 1, 2, IGNORE]))
    assert w.mean() == pytest.approx(1.0) and w[0] == pytest.approx(w[2]) and w[0] > w[1]


def test_fit_and_predict_run_end_to_end() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 56)).astype(np.float32)
    y = rng.integers(0, 3, 400)
    y[::3] = IGNORE
    model, history = fit(
        x[:300], y[:300], x[300:], y[300:], seed=0, device="cpu", log=lambda _: None
    )
    assert len(history) == 5
    probs = predict(model, x[300:], "cpu")
    assert np.isnan(probs[: WINDOW - 1]).all()
    np.testing.assert_allclose(probs[WINDOW - 1 :].sum(axis=1), 1.0, rtol=1e-5)
