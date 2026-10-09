"""Getting a fitted model out of Python.

A model that only runs inside the research environment is not deployable, and
the gap is usually discovered late — after the strategy looks good and someone
asks how it will be served. ONNX is the standard answer: one file, no Python at
inference, and runtimes in every language a trading system is likely to be
written in.

The export is not the interesting part. **Verifying it** is.

A converted model is a different implementation of the same arithmetic, and
differences appear for dull reasons: float32 against float64, a different order
of summation, an operator the converter approximated. On a probability near a
threshold, a discrepancy of 1e-6 changes a trade into no trade. So nothing is
exported without checking the converted model against the original on real
rows, and the check is a hard failure rather than a warning — an export that
silently disagrees with the model it came from is worse than no export.
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class ExportError(RuntimeError):
    """A model could not be exported, or the export did not match."""


@dataclass
class ExportReport:
    """What was exported, and how closely it reproduces the original."""

    path: Path
    rows_checked: int
    max_absolute_difference: float
    max_decision_difference: int
    tolerance: float
    #: What ``max_absolute_difference`` is measured in.
    quantity: str = "probability"

    @property
    def ok(self) -> bool:
        return self.max_absolute_difference <= self.tolerance and self.max_decision_difference == 0

    def summary(self) -> str:
        return (
            f"{self.path.name}: {self.rows_checked:,} rows checked, "
            f"largest {self.quantity} difference {self.max_absolute_difference:.2e}, "
            f"{self.max_decision_difference} decision(s) changed"
        )


def export_onnx(
    model: Any,
    sample: pd.DataFrame,
    path: Path | str,
    *,
    tolerance: float = 1e-5,
    min_confidence: float = 0.5,
) -> ExportReport:
    """Convert a fitted model to ONNX and verify it against the original.

    ``sample`` is real feature rows, not synthetic ones. A converter checked on
    zeros or on random noise passes while disagreeing on the distribution the
    model will actually meet, and the operators that differ are usually the ones
    exercised by realistic inputs.

    ``min_confidence`` is the threshold the strategy would trade on. The
    comparison counts how many *decisions* change, not only how far the
    probabilities moved: a difference of 1e-7 matters if it happens to sit on
    the threshold, and does not otherwise.
    """
    try:
        import onnxruntime
        from skl2onnx import to_onnx
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ExportError(
            "ONNX export needs skl2onnx and onnxruntime. Install with: uv sync --extra export"
        ) from exc

    # Each wrapper keeps its fitted scikit-learn object under a different name;
    # asking for all of them beats making every model implement an accessor for
    # a concern most of them do not have.
    estimator = next(
        (
            getattr(model, attribute)
            for attribute in ("pipeline_", "estimator_", "model_", "booster_")
            if getattr(model, attribute, None) is not None
        ),
        None,
    )
    if estimator is None:
        raise ExportError(
            f"{getattr(model, 'name', type(model).__name__)} does not expose a fitted "
            "scikit-learn estimator; only the tabular models can be exported this way"
        )

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    features = sample.to_numpy(dtype="float32")

    converted = to_onnx(estimator, features[:1], target_opset=17)
    out.write_bytes(converted.SerializeToString())

    session = onnxruntime.InferenceSession(out.read_bytes(), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    raw = session.run(None, {name: features})

    # scikit-learn classifiers export label and probability outputs; the
    # probabilities arrive as a list of dicts from some converters and as an
    # array from others, so both shapes are handled rather than assumed.
    probabilities = raw[-1]
    if isinstance(probabilities, list):
        probabilities = np.array([[row[k] for k in sorted(row)] for row in probabilities])
    exported = np.asarray(probabilities, dtype="float64")

    original = np.asarray(model.predict_proba(sample), dtype="float64")
    if exported.shape != original.shape:
        raise ExportError(f"exported model returns {exported.shape}, original {original.shape}")

    difference = float(np.max(np.abs(exported - original)))
    changed = int(
        np.sum((exported.max(axis=1) >= min_confidence) != (original.max(axis=1) >= min_confidence))
    )

    report = ExportReport(
        path=out,
        rows_checked=len(sample),
        max_absolute_difference=difference,
        max_decision_difference=changed,
        tolerance=tolerance,
    )
    if not report.ok:
        out.unlink(missing_ok=True)
        raise ExportError(
            f"the exported model does not reproduce the original: {report.summary()}. "
            "The file has been removed rather than left to be deployed."
        )
    return report


def _bound_side(mu: np.ndarray, log_variance: np.ndarray, cost_bp: float, k: float) -> np.ndarray:
    """+1, -1 or 0: the side the confidence bound takes at ``cost_bp`` and ``k``."""
    sigma = np.exp(0.5 * log_variance)
    return np.where(mu - k * sigma > cost_bp, 1, np.where(mu + k * sigma < -cost_bp, -1, 0))


def export_gaussian_onnx(
    model: Any,
    sample: pd.DataFrame,
    path: Path | str,
    *,
    tolerance: float = 1e-4,
    cost_bp: float = 11.0,
    k: float = 1.0,
) -> ExportReport:
    """Export the Gaussian-head network to ONNX and verify it against torch.

    The graph reads raw windows, ``(batch, features, window)`` as the every-row
    store holds them, and returns ``mu_bp`` and ``log_variance``; the input
    standardisation, the head's scale and the clamp travel inside it.
    ``sample`` is a contiguous frame of real every-row inputs, and every row of
    it with a complete window is checked. ``tolerance`` is in bp (on ``mu``)
    and in log units (on the variance); a decision is the side the confidence
    bound takes at ``cost_bp`` and ``k``, and no decision may change.
    """
    try:
        import onnxruntime
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ExportError(
            "exporting the network needs torch and onnxruntime: uv sync --extra deep --extra export"
        ) from exc
    from trading_research.models.windows import Day, WindowDataset

    if getattr(model, "net_", None) is None or not hasattr(model, "predict_days"):
        raise ExportError(f"{getattr(model, 'name', type(model).__name__)} is not a fitted network")

    values = sample.to_numpy(dtype="float32")
    day = Day("sample", lambda: values, np.arange(len(values)))
    batches = WindowDataset([day], window=model.window).inference_batches(4096)
    windows = [w for _, _, w in batches]
    if not windows:
        raise ExportError("no row of the sample has a complete window")
    inputs = np.concatenate(windows)
    [(mu, log_variance)] = model.predict_days([day])
    finite = np.isfinite(mu)
    mu, log_variance = mu[finite], log_variance[finite]

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    net = copy.deepcopy(model.net_).cpu().eval()
    batch = {0: "batch"}
    with warnings.catch_warnings():
        # The TorchScript exporter is the one that needs no extra package; its
        # deprecation notice says nothing about this graph, and the parity
        # check below is what decides whether the file is kept.
        warnings.simplefilter("ignore")
        torch.onnx.export(
            net,
            (torch.from_numpy(inputs[:1]),),
            str(out),
            dynamo=False,
            input_names=["windows"],
            output_names=["mu_bp", "log_variance"],
            dynamic_axes={"windows": batch, "mu_bp": batch, "log_variance": batch},
            opset_version=17,
        )
    session = onnxruntime.InferenceSession(out.read_bytes(), providers=["CPUExecutionProvider"])
    got_mu, got_log_variance = (
        np.asarray(a, dtype="float64") for a in session.run(None, {"windows": inputs})
    )
    difference = float(
        max(np.max(np.abs(got_mu - mu)), np.max(np.abs(got_log_variance - log_variance)))
    )
    changed = int(
        np.sum(
            _bound_side(got_mu, got_log_variance, cost_bp, k)
            != _bound_side(mu, log_variance, cost_bp, k)
        )
    )
    report = ExportReport(
        path=out,
        rows_checked=len(mu),
        max_absolute_difference=difference,
        max_decision_difference=changed,
        tolerance=tolerance,
        quantity="mu (bp) or log-variance",
    )
    if not report.ok:
        out.unlink(missing_ok=True)
        raise ExportError(
            f"the exported network does not reproduce the original: {report.summary()}. "
            "The file has been removed rather than left to be deployed."
        )
    return report
