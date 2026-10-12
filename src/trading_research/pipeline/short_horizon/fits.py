"""Stage A: the walk-forward fits, their predictions on the test days, and the candidate-row tapes.

A **signal configuration** is what one fit serves: stratum C's model, inputs,
hold and label fee, or stratum R's model and hold (its inputs fixed by the
model). For each fold (February to June) and configuration:

* **C** (``gbm``, ``logistic``): fitted on the fold's training rows (the
  every-tenth rows of its training days, stamped before the purge, at most the
  registered number of rows taken evenly in time) on the walked-book label at
  the configuration's hold and fee; the hold-first thresholds at 25, 50, 100
  and 240 trades a day set on every scored row of the inner validation;
* **R**: ``bound_ridge`` on the forward move at the hold; ``tcn`` on the
  every-row store's windows, its checkpoint read on the inner validation's
  realised forward moves, fitted in a process of its own;
* **gate levels** per hold from the fold's training rows (the market gate's
  inputs, kept beside the targets).

On each test day, every configuration's predictions on every row (none on a
cold day's first N minutes) give its **candidate-row tape**, the loosest
cell's rows (240 trades a day, or k = beta = 0), with gate and guard states,
and its **daily IC** on all scored rows. A fold whose record is on disk is
not fitted again.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_research.pipeline.short_horizon import features as ft
from trading_research.pipeline.short_horizon.inputs import Inputs
from trading_research.pipeline.short_horizon.plan import NS_PER_S, Fold, midnight_ns
from trading_research.search_engine.catalogue import parse_key
from trading_research.search_engine.market_gate import (
    SPREAD_COLUMN,
    VOLATILITY_COLUMN,
    GateLevels,
    fit_gate_levels,
)
from trading_research.strategies.hold_first import HoldFirstRule, fit_hold_first

RATES: tuple[float, ...] = (25.0, 50.0, 100.0, 240.0)
LOOSEST_RATE = 240.0


@dataclass(frozen=True)
class Configuration:
    """One fit's worth of cells."""

    name: str
    stratum: str
    model: str
    inputs: str
    hold_s: float
    fee_bp: float

    @property
    def target(self) -> str:
        if self.stratum == "C":
            return ft.label_column(self.hold_s, self.fee_bp)
        return ft.forward_column(self.hold_s)

    @property
    def folder(self) -> str:
        return self.name.replace("|", "__").replace("+", "_")


def configurations(keys: Sequence[str]) -> list[Configuration]:
    """The configurations behind ``keys``, in the order their first cell appears."""
    out: dict[str, Configuration] = {}
    for key in keys:
        cell = parse_key(key)
        if cell.configuration not in out:
            fee = cell.label_fee_bp if cell.stratum == "C" else 0.0
            out[cell.configuration] = Configuration(
                cell.configuration, cell.stratum, cell.model, cell.inputs, cell.hold_s, fee
            )
    return list(out.values())


@dataclass
class FoldData:
    """A fold's training rows (every tenth) with every input column and target."""

    x: pd.DataFrame
    targets: pd.DataFrame

    @classmethod
    def load(cls, inputs: Inputs, days: Sequence[date], columns: Sequence[str]) -> FoldData:
        xs, ts = [], []
        for day in days:
            targets = inputs.targets(day)
            x = inputs.features(day, columns, every_row=False)
            if len(x) != len(targets):
                raise RuntimeError(f"{day}: {len(x)} stored rows against {len(targets)} targets")
            xs.append(x.astype(np.float32))
            ts.append(targets)
        if not xs:
            raise RuntimeError("a fold has no training day")
        return cls(pd.concat(xs, ignore_index=True), pd.concat(ts, ignore_index=True))

    def rows(self, before_ns: int, max_rows: int) -> np.ndarray:
        """The training rows' positions: every row stamped before ``before_ns``,
        thinned evenly in time to at most ``max_rows`` (:func:`thin_positions`).
        A kept row without a label is dropped by the fit, not replaced."""
        at = np.flatnonzero(self.targets[ft.TIME].to_numpy() < before_ns)
        return at[thin_positions(len(at), max_rows)]

    def shifted(self, target: str, start_ns: int, end_ns: int, days: int) -> np.ndarray:
        """The training labels moved ``days`` later, circularly within ``[start_ns,
        end_ns)``: the label at stamp ``t`` becomes the one at the last training row
        at or before ``start + ((t - start - days) mod span)``, within twice the
        rows' median spacing, else missing. On a span shorter than ``days`` the
        shift wraps as often as it takes (``days`` mod the span). Event-time rows
        never repeat a stamp a whole number of days apart, so the source is matched
        as of its time rather than on an exact stamp."""
        stamps = self.targets[ft.TIME].to_numpy()
        values = self.targets[target].to_numpy(dtype=np.float64)
        inside = np.flatnonzero((stamps >= start_ns) & (stamps < end_ns))
        out = np.full(len(stamps), np.nan)
        if len(inside) < 2:
            return out
        times = stamps[inside]
        lag = days * 86_400 * NS_PER_S
        source = start_ns + np.mod(times - start_ns - lag, end_ns - start_ns)
        at = np.searchsorted(times, source, side="right") - 1
        tolerance = 2 * float(np.median(np.diff(times)))
        ok = (at >= 0) & (source - times[np.maximum(at, 0)] <= tolerance)
        out[inside[ok]] = values[inside[at[ok]]]
        return out

    def gate_levels(self, before_ns: int) -> GateLevels:
        keep = self.targets[ft.TIME].to_numpy() < before_ns
        return fit_gate_levels(self.targets.loc[keep, [VOLATILITY_COLUMN, SPREAD_COLUMN]])


def thin_positions(total: int, max_rows: int) -> np.ndarray:
    """Positions ``floor(i * total / max_rows)`` for ``i < max_rows``, in integer
    arithmetic; every position when ``total`` is at most ``max_rows``."""
    if max_rows < 1:
        raise ValueError(f"max_rows must be at least 1, got {max_rows}")
    if total <= max_rows:
        return np.arange(total, dtype=np.int64)
    return (np.arange(max_rows, dtype=np.int64) * np.int64(total)) // np.int64(max_rows)


def thresholds(proba: np.ndarray, days: int) -> dict[float, HoldFirstRule]:
    """Hold-first's per-side thresholds at every registered rate on the inner validation."""
    return {rate: fit_hold_first(proba, days=float(days), trades_per_day=rate) for rate in RATES}


def inner_proba(inputs: Inputs, model: Any, columns: Sequence[str], days: Sequence[date],
                before_ns: int) -> np.ndarray:  # fmt: skip
    """Class probabilities on every scored inner-validation row stamped before ``before_ns``."""
    parts = []
    for day in days:
        x = inputs.features(day, ["timestamp", *columns], every_row=True)
        stamps = ft.stamps_ns(x)
        x = x.drop(columns="timestamp")
        keep = (stamps >= inputs.scored_from(day)) & (stamps < before_ns)
        if keep.any():
            proba = on_complete_rows(model.predict_proba, x[keep].astype(np.float64), 3)
            parts.append(proba[np.isfinite(proba).all(axis=1)])
    return np.concatenate(parts) if parts else np.zeros((0, 3))


def save_json(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    partial.write_text(
        json.dumps(record, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    partial.replace(path)


def rules_record(rules: dict[float, HoldFirstRule]) -> dict[str, dict[str, float]]:
    return {f"{rate:g}": asdict(rule) for rate, rule in rules.items()}


def rules_from(record: dict[str, dict[str, float]]) -> dict[float, HoldFirstRule]:
    return {float(rate): HoldFirstRule(**values) for rate, values in record.items()}


def on_complete_rows(predict: Any, x: pd.DataFrame, width: int) -> np.ndarray:
    """``predict`` on the rows with every input present; NaN on the others, which
    are not scored (as a warm-up row is dropped from a fit, never imputed)."""
    complete = x.notna().all(axis=1).to_numpy()
    out = np.full((len(x), width), np.nan) if width > 1 else np.full(len(x), np.nan)
    if complete.any():
        out[complete] = np.asarray(predict(x.loc[complete]), dtype=np.float64)
    return out


def window_days(
    inputs: Inputs,
    days: Sequence[date],
    columns: Sequence[str],
    target: str | None,
    before_ns: int | None,
    max_rows: int,
    target_dir: Path | None = None,
) -> list[Any]:
    """Days of the every-row store for the network: windows end at the every-tenth
    rows with a target (training, validation) or at every row (inference).
    ``target_dir`` holds targets stage 1 did not keep (the control label)."""
    from trading_research.models.windows import Day, thin_evenly

    store = inputs.stores["every_row"]
    picked: list[np.ndarray] = []
    targets: list[np.ndarray | None] = []
    for day in days:
        if target is None:
            picked.append(np.arange(store_rows(inputs, day), dtype=np.int64))
            targets.append(None)
            continue
        frame = (
            inputs.targets(day)
            if target_dir is None
            else pd.read_parquet(target_dir / f"{day}.parquet")
        )
        ok = np.ones(len(frame), dtype=bool)
        if before_ns is not None:
            ok &= frame[ft.TIME].to_numpy() < before_ns
        picked.append(frame["row"].to_numpy()[ok])
        targets.append(frame[target].to_numpy(dtype=np.float64)[ok])
    if target is not None:
        # thinned over every row of the span, then the rows without a target dropped
        kept = thin_evenly(picked, max_rows)
        positions = [np.searchsorted(p, k) for p, k in zip(picked, kept, strict=True)]
        values = [t[i] for t, i in zip(targets, positions, strict=True) if t is not None]
        known = [np.isfinite(v) for v in values]
        picked = [k[f] for k, f in zip(kept, known, strict=True)]
        targets = [v[f] for v, f in zip(values, known, strict=True)]
    out = []
    for k, day in enumerate(days):
        follows = k > 0 and (day - days[k - 1]).days == 1 and not inputs.cold(day)

        def load(d: date = day) -> np.ndarray:
            return store.read_day(d, columns=list(columns)).to_numpy(dtype=np.float32)

        out.append(
            Day(str(day), load, picked[k], targets[k] if target is not None else None, follows)
        )
    return out


def store_rows(inputs: Inputs, day: date) -> int:
    """Rows of a day in the every-row store (one per book row)."""
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    return int(pq.ParquetFile(inputs.stores["every_row"].path(day)).metadata.num_rows)


@dataclass(frozen=True)
class NetworkJob:
    """Every network fit of one fold, run in a process of its own."""

    study: Any
    fold: Fold
    configurations: tuple[Configuration, ...]
    out: Path
    epochs: int | None = None
    #: Train on the control label kept here, rather than on the configuration's own.
    control_dir: Path | None = None


def fit_networks(job: NetworkJob) -> dict[str, dict[str, Any]]:
    """Fit each network on the fold, its checkpoint on the inner validation, and
    write its mean and log-variance on every row of every test day."""
    from trading_research.pipeline.short_horizon.models import network, save

    inputs = Inputs(job.study)
    fold, scale = job.fold, job.study.scale
    found: dict[str, dict[str, Any]] = {}
    for cfg in job.configurations:
        folder = job.out / cfg.folder
        record_path = folder / "fit.json"
        if record_path.exists():
            found[cfg.name] = json.loads(record_path.read_text(encoding="utf-8"))
            continue
        columns = inputs.columns(cfg.inputs)
        target = cfg.target if job.control_dir is None else f"control_{cfg.hold_s:g}"
        train = window_days(inputs, fold.train, columns, target, fold.train_end_ns(cfg.hold_s),
                            scale.max_training_rows, job.control_dir)  # fmt: skip
        inner = window_days(inputs, fold.inner, columns, ft.forward_column(cfg.hold_s),
                            fold.inner_end_ns(cfg.hold_s), scale.max_training_rows)  # fmt: skip
        changes = {} if job.epochs is None else {"epochs": job.epochs}
        known = sum(int(np.isfinite(d.targets).sum()) for d in train if d.targets is not None)
        if known < 100:  # too few targets to fit on: no score, no decision this fold
            folder.mkdir(parents=True, exist_ok=True)
            for day in fold.test:
                empty = np.full(store_rows(inputs, day), np.nan)
                np.savez(folder / f"{day}.npz", mu=empty, log_variance=empty)
            record: dict[str, Any] = {"weights_sha256": None, "checkpoint_epoch": None,
                                      "note": f"{known} targets"}  # fmt: skip
            save_json(record_path, record)
            found[cfg.name] = record
            continue
        model = network(scale, **changes).fit_days(train, inner or None)
        sha = save(model, folder / "weights.pt")
        for day, (mu, log_variance) in zip(
            fold.test,
            model.predict_days(window_days(inputs, fold.test, columns, None, None, 1)),
            strict=True,
        ):
            np.savez(folder / f"{day}.npz", mu=mu, log_variance=log_variance)
        record = {"weights_sha256": sha, "checkpoint_epoch": int(model.checkpoint_epoch_ or 0),
                  "history": [{k: float(v) for k, v in h.items()} for h in getattr(model, "history_", [])]}  # fmt: skip
        save_json(record_path, record)
        found[cfg.name] = record
    return found


def network_prediction(out: Path, cfg: Configuration, day: date) -> tuple[np.ndarray, np.ndarray]:
    """A network's mean and sigma in bp on every row of a test day."""
    with np.load(out / cfg.folder / f"{day}.npz") as saved:
        return saved["mu"], np.exp(0.5 * saved["log_variance"])


def rank_ic(score: np.ndarray, outcome: np.ndarray) -> float:
    """Spearman correlation on rows where both are finite; NaN when undefined."""
    ok = np.isfinite(score) & np.isfinite(outcome)
    if ok.sum() < 3:
        return float("nan")
    a = pd.Series(score[ok]).rank().to_numpy()
    b = pd.Series(outcome[ok]).rank().to_numpy()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def fit_main_process(
    inputs: Inputs, fold: Fold, cfg: Configuration, data: FoldData, shift_days: int | None = None
) -> tuple[Any, dict[str, Any]]:
    """A C or bound-ridge fit on the fold, with C's thresholds; the network is not here.
    With ``shift_days`` the training labels are circularly shifted first (the refit)."""
    from trading_research.pipeline.short_horizon.models import bound_ridge, classifier
    from trading_research.validation.fold_leakage import digest

    scale = inputs.study.scale
    columns = inputs.columns(cfg.inputs)
    end = fold.train_end_ns(cfg.hold_s)
    labels = data.targets[cfg.target].to_numpy(dtype=np.float64)
    if shift_days is not None:
        labels = data.shifted(cfg.target, midnight_ns(fold.train[0]), end, shift_days)
    rows = data.rows(end, scale.max_training_rows)
    rows = rows[np.isfinite(labels[rows])]
    from trading_research.models.base import clean

    x, y = clean(
        data.x.iloc[rows][columns].astype(np.float64).reset_index(drop=True),
        pd.Series(labels[rows]),
    )
    never = {rate: HoldFirstRule(float("inf"), float("inf")) for rate in RATES}
    enough = y.nunique() >= 2 if cfg.stratum == "C" else len(y) >= 2
    if not enough:  # nothing to fit on: the configuration makes no decision this fold
        empty: dict[str, Any] = {
            "rows": len(y),
            "weights_digest": None,
            "note": "too few rows or classes",
        }
        return None, empty | ({"thresholds": rules_record(never)} if cfg.stratum == "C" else {})
    model = classifier(cfg.model, scale) if cfg.stratum == "C" else bound_ridge(scale)
    model.fit(x, y)
    entry: dict[str, Any] = {"rows": len(rows), "weights_digest": digest(model)}
    if cfg.stratum == "C":
        proba = inner_proba(inputs, model, columns, fold.inner, fold.inner_end_ns(cfg.hold_s))
        # without an inner-validation row the configuration cannot trade this fold
        rules = thresholds(proba, len(fold.inner)) if len(proba) else never
        entry["thresholds"] = rules_record(rules)
        entry["classes_present"] = sorted(int(c) for c in pd.unique(y.dropna()))
    return model, entry


def predictions(
    inputs: Inputs,
    cfg: Configuration,
    model: Any,
    frame: pd.DataFrame,
    network_out: Path,
    day: date,
) -> dict[str, np.ndarray]:
    """One configuration's scores on every row of a test day (NaN where unscored)."""
    stamps = ft.stamps_ns(frame)
    unscored = stamps < inputs.scored_from(day)
    if model is None and cfg.model != "tcn":  # the fold had nothing to fit on
        empty = np.full(len(frame), np.nan)
        return (
            {"proba": np.full((len(frame), 3), np.nan)}
            if cfg.stratum == "C"
            else {"mu": empty, "sigma": empty}
        )
    if cfg.model == "tcn":
        mu, sigma = network_prediction(network_out, cfg, day)
        mu, sigma = mu.astype(np.float64), sigma.astype(np.float64)
    elif cfg.stratum == "R":
        x = frame[inputs.columns(cfg.inputs)].astype(np.float64)
        mu = on_complete_rows(model.predict_mean_bp, x, 1)
        sigma = on_complete_rows(model.predict_sigma_bp, x, 1)
    else:
        x = frame[inputs.columns(cfg.inputs)].astype(np.float64)
        proba = on_complete_rows(model.predict_proba, x, 3)
        proba[unscored] = np.nan
        return {"proba": proba}
    mu, sigma = np.where(unscored, np.nan, mu), np.where(unscored, np.nan, sigma)
    return {"mu": mu, "sigma": sigma}


def run_fold(
    inputs: Inputs,
    fold: Fold,
    configs: Sequence[Configuration],
    close: pd.Series,
    root: Path,
    shift_days: int | None = None,
) -> dict[str, Any]:
    """Every configuration's fit on one fold, and its tape and IC on each test day;
    with ``shift_days``, on training labels shifted that many days (the refit)."""
    from trading_research.backtest.walk import walk_cost_bp
    from trading_research.data.study_loader import clip_for
    from trading_research.labels.taker_vwap import forward_move_bp
    from trading_research.pipeline.short_horizon import flags
    from trading_research.pipeline.short_horizon.models import in_fresh_process
    from trading_research.search_engine.market_gate import gate_inputs, gate_states
    from trading_research.search_engine.tapes import (
        ev_gate_tape,
        hold_first_tape,
        tape_path,
        write_tape,
    )

    path = root / "folds" / f"{fold.name}.json"
    if path.exists():
        loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return loaded
    study = inputs.study
    columns = sorted({c for cfg in configs for c in inputs.columns(cfg.inputs)})
    data = FoldData.load(inputs, fold.train, columns)
    holds = sorted({cfg.hold_s for cfg in configs})
    levels = {hold: data.gate_levels(fold.train_end_ns(hold)) for hold in holds}
    flag_record = flags.fold_flags(close, fold.name, root / "flags")
    flags_ns = flags.guard_flags(flag_record)
    models: dict[str, Any] = {}
    entries: dict[str, dict[str, Any]] = {}
    for cfg in configs:
        if cfg.model != "tcn":
            models[cfg.name], entries[cfg.name] = fit_main_process(
                inputs, fold, cfg, data, shift_days
            )
    del data
    networks = tuple(cfg for cfg in configs if cfg.model == "tcn")
    network_out = root / "network" / fold.name
    if networks:
        entries |= in_fresh_process(fit_networks, NetworkJob(study, fold, networks, network_out))
    ic: dict[str, dict[str, float]] = {cfg.name: {} for cfg in configs}
    clip = clip_for(study.symbol)
    for day in fold.test:
        book = inputs.book(day)
        frame = inputs.features(day, ["timestamp", *columns], every_row=True)
        if len(frame) != len(book):
            raise RuntimeError(f"{day}: {len(frame)} stored rows against {len(book)} book rows")
        stamps = ft.stamps_ns(book)
        given = gate_inputs(book)
        cost = walk_cost_bp(book, clip)
        forward = {hold: forward_move_bp(book, hold).to_numpy() for hold in holds}
        for cfg in configs:
            gates = gate_states(given, levels[cfg.hold_s])
            found = predictions(inputs, cfg, models.get(cfg.name), frame, network_out, day)
            if cfg.stratum == "C":
                rule = rules_from(entries[cfg.name]["thresholds"])[LOOSEST_RATE]
                tape = hold_first_tape(day, cfg.name, stamps, found["proba"], rule, gates, flags_ns)
                score = found["proba"][:, 2] - found["proba"][:, 0]
            else:
                tape = ev_gate_tape(
                    day, cfg.name, stamps, found["mu"], found["sigma"], cost, gates, flags_ns
                )
                score = found["mu"]
            write_tape(tape, tape_path(root / "tapes", cfg.name, day))
            ic[cfg.name][day.isoformat()] = rank_ic(score, forward[cfg.hold_s])
    record = {
        "fold": fold.name,
        "days": {k: [d.isoformat() for d in getattr(fold, k)] for k in ("train", "inner", "test")},
        "gate_levels": {f"{h:g}": levels[h].to_dict() for h in holds},
        "flags": {"alarms": len(flag_record["alarms"]), "guard_flags": len(flags_ns),
                  "profile": flag_record["profile"]},
        "fits": entries,
        "ic": ic,
    }  # fmt: skip
    save_json(path, record)
    return record


def run(study: Any) -> dict[str, Any]:
    """Stage A's record: every fold's fits, thresholds, gate levels, flags and daily IC."""
    from trading_research.data import study_access as sa
    from trading_research.pipeline.short_horizon import flags
    from trading_research.pipeline.short_horizon.plan import walk_forward

    inputs = Inputs(study)
    configs = configurations(study.keys)
    folds = walk_forward(inputs.usable("D"))
    d_end = sa.BLOCKS["D"][1]
    close = flags.hourly_close(study.symbol, study.roots.bars, d_end, sa.development_access())
    root = study.dir("A")
    records = [run_fold(inputs, fold, configs, close, root) for fold in folds]
    return {
        "commit": study.commit(),
        "configurations": [cfg.name for cfg in configs],
        "folds": {r["fold"]: {k: len(v) for k, v in r["days"].items()} for r in records},
        "fits": sum(len(r["fits"]) for r in records),
        "mean_daily_ic": {
            cfg.name: float(
                np.nanmean([v for r in records for v in r["ic"][cfg.name].values()] or [np.nan])
            )
            for cfg in configs
        },
    }
