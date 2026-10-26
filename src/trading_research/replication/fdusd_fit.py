"""The replication's stores, its fit and its thresholds, by the spot study's own functions.

As the spot study's stage 1 and D code do it (``spot.stage1.write_store``,
``spot.fits.fit_fold``), for S25-PF's one cell: the inputs at every tenth row,
cast as the spot stores are (float16), with the print label at hold 50 s and
label fee 0 (``spot.print_label.print_targets``), its purge stamp and the scored
flag; the training rows of the fold (``spot.fits.training_set``: purged, scored,
at most 4 million taken evenly in time); ``gbm`` at the repository's defaults,
seed 42 (``spot.fits.fit_model``); and the hold-first thresholds at 240 trades
a day on the inner validation's every kept row (``spot.fits.rules_at``).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_research.data.feature_store import to_stored
from trading_research.features.timed import stamps_ns
from trading_research.replication.fdusd_fetch import ROOT
from trading_research.replication.fdusd_windows import (
    HOLD_S,
    INPUTS,
    LABEL_FEE_BP,
    LAYOUT,
    RATE,
    Window,
    open_day,
)
from trading_research.spot import days as dy
from trading_research.spot.fits import finite_theta, fit_model, rules_at, save_model, training_set
from trading_research.spot.plan import MAX_TRAINING_ROWS, TARGET_STRIDE, Fold
from trading_research.spot.print_label import print_targets


def store_path(day: date, root: Path | str = ROOT) -> Path:
    return Path(root) / "work" / "stores" / f"{day.isoformat()}.parquet"


def write_store(day: date, root: Path | str = ROOT) -> int:
    """Inputs at every tenth row with the label, its purge stamp and the scored flag."""
    path = store_path(day, root)
    if path.exists():
        return len(pd.read_parquet(path, columns=["scored"]))
    found = open_day(day, root)
    if not found.usable:
        return 0
    frame = found.features()
    stamps = stamps_ns(frame)
    at = np.arange(0, len(frame), TARGET_STRIDE, dtype=np.int64)
    targets = print_targets(found.tape(), stamps, (HOLD_S,), (LABEL_FEE_BP,), at=at)
    stored = to_stored(frame.iloc[at].reset_index(drop=True), LAYOUT)
    stored["scored"] = stamps[at] >= found.scored_from
    out = pd.concat([stored, targets.drop(columns=["timestamp"]).reset_index(drop=True)], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.part")
    out.to_parquet(partial, compression="zstd", index=False)
    partial.replace(path)
    return len(out)


def inner_rows(model: Any, fold: Fold, day: date, root: Path | str = ROOT) -> np.ndarray | None:
    """The model's probabilities on the day's kept inner-validation rows (None: not usable)."""
    found = open_day(day, root)
    if not found.usable:
        return None
    frame = found.features()
    proba = dy.predict(model, frame, LAYOUT, found.scored_from)
    keep = fold.inner_keep(stamps_ns(frame), HOLD_S) & np.isfinite(proba).all(axis=1)
    return proba[keep]


def fit_window(window: Window, root: Path | str = ROOT) -> dict[str, Any]:
    """The window's fit and thresholds; the model is written under ``work/<window>/``."""
    fold = window.fold()
    paths = [store_path(d, root) for d in fold.train if store_path(d, root).exists()]
    frame = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    x, y = training_set(frame, fold, HOLD_S, LABEL_FEE_BP, INPUTS, MAX_TRAINING_ROWS)
    del frame
    model = fit_model(x, y, {})
    if model is None:
        raise RuntimeError(f"window {window.name}: fewer than two classes in training")
    digest = save_model(model, Path(root) / "work" / window.name / "model.pkl")
    found = [inner_rows(model, fold, d, root) for d in fold.inner]
    probas = [p for p in found if p is not None]
    proba = np.concatenate(probas) if probas else np.zeros((0, 3))
    rule = rules_at(proba, len(probas), [RATE])[RATE]
    classes = {str(int(c)): int((y == c).sum()) for c in sorted(y.unique())}
    return {"window": window.name, "train_days": len(paths), "rows": len(y),
            "classes": classes,
            "inner_days": len(probas), "inner_rows": len(proba), "theta": finite_theta(rule),
            "sha256": digest, "hold_s": HOLD_S, "rate": RATE, "label_fee_bp": LABEL_FEE_BP}  # fmt: skip
