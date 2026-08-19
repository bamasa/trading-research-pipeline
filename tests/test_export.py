"""Tests for the ONNX export.

The export itself is uninteresting; verifying it is the point. A converted model
is a different implementation of the same arithmetic, and on a probability
sitting near a threshold a discrepancy of 1e-6 turns a trade into no trade.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_research.models.export import ExportError, ExportReport, export_onnx
from trading_research.pipeline.stages import build_model

onnx = pytest.importorskip("skl2onnx", reason="ONNX export is an optional extra")
pytest.importorskip("onnxruntime", reason="ONNX export is an optional extra")


@pytest.fixture
def fitted() -> tuple[object, pd.DataFrame]:
    rng = np.random.default_rng(0)
    n = 3000
    signal = rng.normal(size=n)
    x = pd.DataFrame({"signal": signal, "noise": rng.normal(size=n)})
    y = pd.Series(np.where(signal > 1.0, 1.0, np.where(signal < -1.0, -1.0, 0.0)))
    return build_model("logistic").fit(x, y), x


def test_an_exported_model_reproduces_the_original(fitted, tmp_path) -> None:
    model, x = fitted
    report = export_onnx(model, x.head(1000), tmp_path / "m.onnx")
    assert report.ok
    assert report.max_absolute_difference < 1e-5
    assert report.max_decision_difference == 0
    assert (tmp_path / "m.onnx").exists()


def test_the_check_counts_decisions_not_only_probabilities(fitted, tmp_path) -> None:
    """A difference of 1e-7 matters if it sits on the threshold and not
    otherwise, so the report carries both."""
    model, x = fitted
    report = export_onnx(model, x.head(500), tmp_path / "m.onnx", min_confidence=0.4)
    assert "decision(s) changed" in report.summary()
    assert report.rows_checked == 500


def test_a_disagreeing_export_is_deleted_rather_than_left(fitted, tmp_path) -> None:
    """An export that silently disagrees is worse than no export at all."""
    model, x = fitted
    path = tmp_path / "m.onnx"
    with pytest.raises(ExportError, match="does not reproduce"):
        export_onnx(model, x.head(500), path, tolerance=-1.0)
    assert not path.exists()


def test_a_model_without_a_scikit_learn_estimator_is_refused(tmp_path) -> None:
    model = build_model("order_flow")
    model.fit(pd.DataFrame({"queue_imbalance": [0.1] * 50}), pd.Series([0.0] * 50))
    with pytest.raises(ExportError, match="does not expose"):
        export_onnx(model, pd.DataFrame({"queue_imbalance": [0.1] * 5}), tmp_path / "m.onnx")


def test_the_report_knows_whether_it_passed() -> None:
    from pathlib import Path

    good = ExportReport(Path("x"), 100, 1e-9, 0, 1e-5)
    drifted = ExportReport(Path("x"), 100, 1e-3, 0, 1e-5)
    flipped = ExportReport(Path("x"), 100, 1e-9, 2, 1e-5)
    assert good.ok
    assert not drifted.ok
    assert not flipped.ok
