"""Stage 1's leakage tests: features, folds and flags. Any failure stops the study.

* **Features** (K-leak): every timed feature of the set, and the tensor, through
  truncation and streaming parity (``validation/leakage_timed``) on a slice of a
  W day with its prints.
* **The fold-level test** (K-leak): in every fold, one fit of each model at a
  hold of 20 s (stratum C on the micro-selected inputs at a 0 bp label fee),
  the hold-first thresholds at the four rates, the gate levels, and the
  network trained for one epoch on the CPU with its inner validation scored,
  compared by sha256 after every price, label, feature and gate input at a row
  after the cutoff is mutated (``validation/fold_leakage.check_fold``). The
  cutoff is the end of the fold's inner validation plus the hold.
* **The flag-causality test** (K-flag-leak): in every fold, mutating every
  hourly bar after an hour leaves every flag the guard uses at or before it
  unchanged.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_research.pipeline.short_horizon import features as ft
from trading_research.pipeline.short_horizon.fits import RATES, thresholds
from trading_research.pipeline.short_horizon.inputs import Inputs
from trading_research.pipeline.short_horizon.plan import NS_PER_S, Fold
from trading_research.search_engine.market_gate import SPREAD_COLUMN, VOLATILITY_COLUMN

HOLD_S = 20.0
LABEL = ft.label_column(HOLD_S, 0.0)
FORWARD = ft.forward_column(HOLD_S)
PRICES = ("bid_price_0", "ask_price_0")


class LeakError(AssertionError):
    """A leakage test failed: the study stops until a dated amendment fixes it."""


def feature_leakage(inputs: Inputs, day: date, rows_used: int, positions: int) -> dict[str, Any]:
    """Truncation and streaming parity of every feature and the tensor on a slice;
    streaming parity at ``positions`` rows spread evenly over its second half."""
    from trading_research.features.lob_tensor import register_tensor
    from trading_research.features.timed import TimedRegistry
    from trading_research.validation.leakage_timed import assert_timed_honest

    rows = inputs.book(day)
    start = max(0, len(rows) // 2 - rows_used // 2)
    rows = rows.iloc[start : start + rows_used].reset_index(drop=True)
    stamps = ft.stamps_ns(rows)
    prints = inputs.prints(day)
    when = ft.stamps_ns(prints)
    prints = prints[(when > stamps[0]) & (when <= stamps[-1])].reset_index(drop=True)
    at = np.linspace(len(rows) // 2, len(rows) - 1, positions).round().astype(int).tolist()
    assert_timed_honest(ft.registry(), rows, prints, positions=at)
    tensor = TimedRegistry()
    register_tensor(inputs.calibration, registry=tensor)
    assert_timed_honest(tensor, rows, positions=at)
    return {
        "day": day.isoformat(),
        "rows": len(rows),
        "positions": len(at),
        "features": len(ft.micro_columns()) + len(ft.TENSOR),
    }


def fold_frame(inputs: Inputs, fold: Fold) -> pd.DataFrame:
    """The fold's training and inner-validation targets rows, with every input."""
    parts = []
    for day in (*fold.train, *fold.inner):
        targets = inputs.targets(day)
        x = inputs.features(day, inputs.columns("micro"), every_row=False).astype(np.float64)
        book = inputs.book(day).iloc[targets["row"].to_numpy()]
        frame = pd.concat([targets.reset_index(drop=True), x.reset_index(drop=True)], axis=1)
        for column in PRICES:
            frame[column] = book[column].to_numpy(dtype=np.float64)
        parts.append(frame)
    out = pd.concat(parts, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out[ft.TIME], unit="ns", utc=True)
    return out


def network_fold(path: Path, columns: list[str], spans: tuple[int, int, int],
                 params: dict[str, Any]) -> dict[str, Any]:  # fmt: skip
    """The network's one-epoch fit on the CPU from a frame on disk; run in a fresh
    process. ``spans`` are the training end, the inner validation's start and end."""
    from trading_research.models.tcn import GaussianTCN

    train_end, inner_start, inner_end = spans
    frame = pd.read_parquet(path)
    stamps = ft.stamps_ns(frame)
    known = np.isfinite(frame[FORWARD].to_numpy())
    train = (stamps < train_end) & known
    inner = (stamps >= inner_start) & (stamps < inner_end) & known
    model = GaussianTCN(**{**params, "epochs": 1, "device": "cpu"})
    model.fit(frame.loc[train, columns], frame.loc[train, FORWARD],
              validation=(frame.loc[inner, columns], frame.loc[inner, FORWARD]))  # fmt: skip
    history = [list(map(float, h)) for h in getattr(model, "history_", [])]
    return {"weights": model.weights_sha256(), "history": history,
            "checkpoint": int(model.checkpoint_epoch_ or 0)}  # fmt: skip


def fold_fit(inputs: Inputs, fold: Fold, scratch: Path, network: bool) -> Any:
    """What the fold-level test compares: every fitted artefact of the fold at 20 s."""
    from dataclasses import asdict

    from trading_research.pipeline.short_horizon.models import (
        C_MODELS,
        bound_ridge,
        classifier,
        in_fresh_process,
    )
    from trading_research.pipeline.short_horizon.plan import midnight_ns
    from trading_research.search_engine.market_gate import fit_gate_levels

    scale = inputs.study.scale
    micro = inputs.columns("micro")
    wide = inputs.columns("tensor+micro")
    spans = (fold.train_end_ns(HOLD_S), midnight_ns(fold.inner_start), fold.inner_end_ns(HOLD_S))

    def fit(frame: pd.DataFrame) -> dict[str, Any]:
        stamps = ft.stamps_ns(frame)
        train = stamps < spans[0]
        inner = (stamps >= spans[1]) & (stamps < spans[2])
        complete = frame[micro].notna().all(axis=1).to_numpy()
        labelled = train & complete & np.isfinite(frame[LABEL].to_numpy())
        out: dict[str, Any] = {}
        for name in C_MODELS:
            model = classifier(name, scale).fit(
                frame.loc[labelled, micro], frame.loc[labelled, LABEL]
            )
            out[name] = model
            proba = np.asarray(model.predict_proba(frame.loc[inner & complete, micro]))
            rules = thresholds(proba, max(1, len(fold.inner))) if len(proba) else {}
            out[f"theta_{name}"] = {f"{r:g}": asdict(rules[r]) for r in RATES if r in rules}
        moved = train & complete & np.isfinite(frame[FORWARD].to_numpy())
        out["bound_ridge"] = bound_ridge(scale).fit(
            frame.loc[moved, micro], frame.loc[moved, FORWARD]
        )
        out["gate"] = fit_gate_levels(
            frame.loc[train, [VOLATILITY_COLUMN, SPREAD_COLUMN]]
        ).to_dict()
        if network:
            path = scratch / f"{fold.name}.parquet"
            frame.loc[:, [ft.TIME, FORWARD, *wide]].to_parquet(path, index=False)
            params = scale.params("tcn")
            out["tcn"] = in_fresh_process(network_fold, path, wide, spans, params)
            out["tcn_inner_ic"] = [h[-1] for h in out["tcn"]["history"]]
        return out

    return fit


def fold_level(inputs: Inputs, folds: list[Fold], network: bool) -> list[dict[str, Any]]:
    """The fold-level leakage test in every fold; raises on any change."""
    from trading_research.validation.fold_leakage import assert_folds_causal, check_fold

    scratch = inputs.study.dir("1", "fold_leakage")
    results = []
    for fold in folds:
        frame = fold_frame(inputs, fold)
        cutoff = pd.Timestamp(
            fold.inner_end_ns(HOLD_S) + round(HOLD_S * NS_PER_S), unit="ns", tz="UTC"
        )
        changed = [
            LABEL,
            FORWARD,
            VOLATILITY_COLUMN,
            SPREAD_COLUMN,
            *inputs.columns("tensor+micro"),
        ]
        results.append(check_fold(fold_fit(inputs, fold, scratch, network), frame, cutoff,
                                  price_columns=PRICES, label_columns=changed, fold=fold.name))  # fmt: skip
    try:
        assert_folds_causal(results)
    except AssertionError as exc:
        raise LeakError(f"K-leak: the fold-level leakage test failed: {exc}") from exc
    return [
        {"fold": r.fold, "artefacts": sorted(r.digests), "cutoff": str(r.cutoff)} for r in results
    ]


def run(inputs: Inputs, network: bool) -> dict[str, Any]:
    """Every leakage test of stage 1; raises :class:`LeakError` on a leak."""
    from trading_research.data import study_access as sa
    from trading_research.pipeline.short_horizon import flags
    from trading_research.pipeline.short_horizon.plan import walk_forward
    from trading_research.validation.leakage import LeakageError

    study = inputs.study
    w_days = inputs.usable("W")
    if not w_days:
        raise LeakError("W has no usable day to run the feature leakage checks on")
    try:
        features = feature_leakage(
            inputs, w_days[0], study.scale.leakage_rows, study.scale.leakage_positions
        )
    except LeakageError as exc:
        raise LeakError(f"K-leak: a feature reads past its cutoff: {exc}") from exc
    folds = walk_forward(inputs.usable("D"))
    fold_results = fold_level(inputs, [f for f in folds if f.train and f.inner], network)
    close = flags.hourly_close(
        study.symbol, study.roots.bars, sa.BLOCKS["D"][1], sa.development_access()
    )
    causality = flags.causality(close, study.scale.causality_hours, study.workers)
    leaking = [c for c in causality if c["leaks"]]
    if leaking:
        raise LeakError(f"K-flag-leak: {[c['detail'] for c in leaking]}")
    return {"features": features, "fold_level": fold_results, "flag_causality": causality}
