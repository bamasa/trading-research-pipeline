"""Stage 1's leakage tests: features, folds and flags. Any failure stops the study.

* **Features** (K-leak): every timed feature of the set, and the tensor, through
  truncation and streaming parity (``validation/leakage_timed``) on a slice of a
  W day with its prints.
* **The fold-level test** (K-leak): in every fold, the production fit path run
  twice, on stage 1's artefacts and on a scratch work root in which every
  price, label, feature and gate input at a row after the cutoff is mutated
  and the labels recomputed: one fit of each model at a hold of 20 s (stratum
  C on the micro-selected inputs at a 0 bp label fee) with the hold-first
  thresholds at the four rates, the gate levels, and the network trained for
  one epoch on the CPU with its inner validation scored, compared by sha256
  (``validation/fold_leakage``). The cutoff is the end of the fold's inner
  validation plus the hold, or the last exit snapshot a target row before
  that end reads, if later (:func:`cutoff_ns`).
* **The flag-causality test** (K-flag-leak): in every fold, mutating every
  hourly bar after an hour leaves every flag the guard uses at or before it
  unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_research.pipeline.short_horizon import features as ft
from trading_research.pipeline.short_horizon.fits import (
    Configuration,
    FoldData,
    NetworkJob,
    fit_main_process,
    fit_networks,
)
from trading_research.pipeline.short_horizon.inputs import Inputs
from trading_research.pipeline.short_horizon.models import C_MODELS
from trading_research.pipeline.short_horizon.plan import NS_PER_S, Fold, midnight_ns

NS_PER_DAY = 86_400 * NS_PER_S

HOLD_S = 20.0


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


def cutoff_ns(inputs: Inputs, fold: Fold, hold_s: float = HOLD_S) -> int:
    """The fold's cutoff: the end of its inner validation plus the hold, or, where
    later, the latest exit snapshot that a target row stamped before that end reads
    (the 10 ms latency and the grid can put it a snapshot past). A day's book ends
    at its midnight, so only the day the inner validation ends on can reach past."""
    from trading_research.labels.taker_vwap import entry_exit_snapshots

    end = fold.inner_end_ns(hold_s)
    cutoff = end + round(hold_s * NS_PER_S)
    for day in fold.inner:
        if midnight_ns(day) + NS_PER_DAY <= cutoff:
            continue
        stamps = ft.stamps_ns(inputs.book(day))
        rows = ft.target_rows(len(stamps))
        rows = rows[stamps[rows] < end]
        exits = entry_exit_snapshots(stamps, hold_s)[1][rows]
        exits = exits[exits < len(stamps)]
        if len(exits):
            cutoff = max(cutoff, int(stamps[exits].max()))
    return cutoff


def _redraw(frame: pd.DataFrame, after: np.ndarray, columns: Sequence[str], seed: int) -> None:
    """Every value of ``columns`` at the rows ``after`` drawn anew, in place: from a
    column's own few values, or from a normal three times as wide as its spread
    (clipped to the stored dtype's range); a column with no value gets ones."""
    rng = np.random.default_rng(seed)
    count = int(after.sum())
    if not count:
        return
    for column in columns:
        values = frame[column].to_numpy(dtype=np.float64).copy()
        finite = values[np.isfinite(values)]
        distinct = pd.unique(finite)
        if len(distinct) == 0:
            drawn = np.ones(count)
        elif len(distinct) <= 10:
            drawn = rng.choice(distinct, size=count)
        else:
            drawn = rng.normal(float(finite.mean()), 3.0 * float(finite.std()) + 1.0, count)
        values[after] = np.clip(drawn, -60_000.0, 60_000.0)
        frame[column] = values.astype(frame[column].to_numpy().dtype)


def _write(frame: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, compression="zstd", index=False)


def _link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source.resolve())


def mutated_inputs(inputs: Inputs, fold: Fold, cutoff: int, root: Path) -> Inputs:
    """Stage 1's artefacts as a scratch work root in which every row after
    ``cutoff`` is mutated, of the fold's training, inner-validation and test days.

    A day wholly before the cutoff is linked unchanged. On the day the cutoff
    falls on, every book price after it is moved (``validation/fold_leakage``'s
    mutation) and the day's targets are recomputed from that book, as stage 1
    computes them, so a target that read past the cutoff changes; its stored
    features after the cutoff are drawn anew. A day wholly after it has every
    target and feature drawn anew."""
    from trading_research.data import study_loader as sl
    from trading_research.validation.fold_leakage import mutate_after_cutoff

    study = replace(inputs.study, work_root=root)
    real = inputs.study.work / "1"
    scratch = study.work / "1"
    stores = [real / "features" / name for name in ("every_row", "every_tenth_row")]
    holds = [float(h) for h in inputs.record["stores"]["holds_with_targets"]]
    clip = sl.clip_for(study.symbol)
    stamp = pd.Timestamp(cutoff, unit="ns", tz="UTC")
    for k, day in enumerate(sorted({*fold.train, *fold.inner, *fold.test})):
        name = f"{day.isoformat()}.parquet"
        start = midnight_ns(day)
        if start + NS_PER_DAY <= cutoff:
            for folder in (real / "targets", *stores):
                _link(folder / name, scratch / folder.relative_to(real) / name)
            continue
        if start > cutoff:
            targets = pd.read_parquet(real / "targets" / name)
            moved = [c for c in targets.columns if c not in ("row", ft.TIME)]
            _redraw(targets, targets[ft.TIME].to_numpy() > cutoff, moved, seed=k)
        else:
            book = inputs.book(day)
            prices = [c for c in book.columns if "_price_" in c]
            book = mutate_after_cutoff(book, stamp, price_columns=prices, seed=k)
            targets = ft.day_targets(book, holds, (0.0, 5.5), clip)
        _write(targets, scratch / "targets" / name)
        for folder in stores:
            frame = pd.read_parquet(folder / name)
            after = ft.stamps_ns(frame) > cutoff
            _redraw(frame, after, [c for c in frame.columns if c != ft.TIME], seed=k + 1)
            _write(frame, scratch / folder.relative_to(real) / name)
    return Inputs(study, access=inputs.access, given=inputs.record)


def leak_configurations(network: bool) -> tuple[Configuration, ...]:
    """Each model once at 20 s: C's on the micro-selected inputs at a 0 bp label fee."""
    out = [Configuration(f"fold-level|C|{m}|micro|hold_s=20|label_fee_bp=0", "C", m, "micro",
                         HOLD_S, 0.0) for m in C_MODELS]  # fmt: skip
    out.append(Configuration("fold-level|R|bound_ridge|micro|hold_s=20", "R", "bound_ridge",
                             "micro", HOLD_S, 0.0))  # fmt: skip
    if network:
        out.append(Configuration("fold-level|R|tcn|tensor+micro|hold_s=20", "R", "tcn",
                                 "tensor+micro", HOLD_S, 0.0))  # fmt: skip
    return tuple(out)


def production_fit(inputs: Inputs, fold: Fold, out: Path, network: bool) -> dict[str, Any]:
    """What the fold-level test compares, from the code the study fits with: the
    fold's training rows (``FoldData.load``), each C model and the bound ridge
    (``fit_main_process``, C's thresholds on the inner validation at the four
    rates), the gate levels (``FoldData.gate_levels``), and the network through
    ``fit_networks`` (``window_days`` over the every-row store) for one epoch on
    the CPU, in a process of its own, with no test day to predict."""
    from trading_research.pipeline.short_horizon.models import in_fresh_process

    configs = leak_configurations(network)
    data = FoldData.load(inputs, fold.train, inputs.columns("micro"))
    found: dict[str, Any] = {}
    for cfg in configs:
        if cfg.model == "tcn":
            continue
        _, entry = fit_main_process(inputs, fold, cfg, data)
        found[f"{cfg.model}/weights"] = entry["weights_digest"]
        found[f"{cfg.model}/rows"] = entry["rows"]
        if cfg.stratum == "C":
            found[f"{cfg.model}/thresholds"] = entry["thresholds"]
            found[f"{cfg.model}/classes"] = entry.get("classes_present")
    found["gate_levels"] = data.gate_levels(fold.train_end_ns(HOLD_S)).to_dict()
    del data
    networks = tuple(cfg for cfg in configs if cfg.model == "tcn")
    if networks:
        job = NetworkJob(inputs.study, replace(fold, test=()), networks, out / "network",
                         epochs=1, device="cpu", record=inputs.record)  # fmt: skip
        fitted = in_fresh_process(fit_networks, job)[networks[0].name]
        found["tcn/weights_sha256"] = fitted["weights_sha256"]
        found["tcn/state_sha256"] = fitted.get("state_sha256")
        found["tcn/inner_ic"] = [h.get("validation_ic") for h in fitted.get("history", [])]
        found["tcn/checkpoint_epoch"] = fitted["checkpoint_epoch"]
    return found


def fold_level(inputs: Inputs, folds: list[Fold], network: bool) -> list[dict[str, Any]]:
    """The fold-level leakage test in every fold, on the production fit path; raises
    :class:`LeakError` on any change. Each fold is fitted twice, on stage 1's
    artefacts and on a scratch work root with every row after its cutoff mutated
    (:func:`mutated_inputs`), and every artefact is compared by sha256."""
    import shutil

    from trading_research.validation.fold_leakage import (
        FoldLeakResult,
        assert_folds_causal,
        digest,
    )

    scratch = inputs.study.dir("1", "fold_leakage")
    results, out = [], []
    for fold in folds:
        folder = scratch / fold.name
        shutil.rmtree(folder, ignore_errors=True)
        cutoff = cutoff_ns(inputs, fold)
        real = production_fit(inputs, fold, folder / "real", network)
        moved = production_fit(mutated_inputs(inputs, fold, cutoff, folder / "mutated"), fold,
                               folder / "moved", network)  # fmt: skip
        stamp = pd.Timestamp(cutoff, unit="ns", tz="UTC")
        digests = {k: digest(v) for k, v in real.items()}
        result = FoldLeakResult(fold.name, stamp, digests, {k: digest(v) for k, v in moved.items()})
        results.append(result)
        out.append({"fold": fold.name, "cutoff": str(stamp), "artefacts": digests,
                    "past_inner_end_plus_hold_ns": cutoff - fold.inner_end_ns(HOLD_S)
                    - round(HOLD_S * NS_PER_S)})  # fmt: skip
        shutil.rmtree(folder, ignore_errors=True)
    try:
        assert_folds_causal(results)
    except AssertionError as exc:
        raise LeakError(f"K-leak: the fold-level leakage test failed: {exc}") from exc
    return out


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
    folds = [f for f in walk_forward(inputs.usable("D")) if f.train and f.inner]
    fold_results = fold_level(inputs, folds[: study.scale.leakage_folds], network)
    close = flags.hourly_close(
        study.symbol, study.roots.bars, sa.BLOCKS["D"][1], sa.development_access()
    )
    causality = flags.causality(
        close, study.scale.causality_hours, study.workers, study.scale.leakage_folds
    )
    leaking = [c for c in causality if c["leaks"]]
    if leaking:
        raise LeakError(f"K-flag-leak: {[c['detail'] for c in leaking]}")
    return {"features": features, "fold_level": fold_results, "flag_causality": causality}
