"""D's fits: every fold's models on the stores, their thresholds on the inner validation.

For each fold (:func:`trading_research.spot.plan.folds`) and each fitted row,
one ``gbm`` (``models/gbm.py``, three classes, the repository's defaults, seed
42) per hold and label fee of the row's cells, on the every-tenth-row store of
the training days: the rows the fold keeps (:meth:`~trading_research.spot.plan.Fold.train_keep`:
stamped before the inner validation less the hold plus 61 minutes, with a label
that reads nothing after the inner validation's first stamp, and scored on its
day), at most the registered number taken evenly in time, a row without a label
or an input dropped. The thresholds per side by the 2025 hold-first rule at each
cell's rate on the inner validation's every row
(:meth:`~trading_research.spot.plan.Fold.inner_keep`), computed per day and
never stored. A fit with fewer than two classes has no model and its cells never
trade (an infinite threshold, frozen as 2: no probability reaches it).

S25-T is never fitted: :func:`frozen_2025` loads the 2025 study's frozen fits
and thresholds, every file checked against the sha256 its configuration records
and that configuration against its frozen sha256.
"""

from __future__ import annotations

import math
import pickle
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from trading_research.features.timed import stamps_ns
from trading_research.market_making.prereg import HeldOutLocked
from trading_research.models.base import clean
from trading_research.spot import days as dy
from trading_research.spot.context import SpotStudy
from trading_research.spot.ledger import file_sha256
from trading_research.spot.plan import ROWS, Cell, Fold, Row, fit_key
from trading_research.spot.print_label import label_column
from trading_research.spot.runs import RowModels
from trading_research.strategies.hold_first import HoldFirstRule, fit_hold_first

NEVER = HoldFirstRule(math.inf, math.inf)
#: S25-T's 2025 fits, by (hold, label fee), and its cells' keys in the 2025 configuration.
FIT_2025 = "BTCUSDT__C__gbm__micro__hold_s={h:g}__label_fee_bp={f:g}.pkl"
KEY_2025 = (
    "BTCUSDT|C|gbm|micro|open|hold_s={h:g}|trades_per_day={r:g}|label_fee_bp={f:g}|regime=none"
)


def thin_positions(total: int, max_rows: int) -> np.ndarray:
    """Positions ``floor(i * total / max_rows)``: evenly in time, as in 2025."""
    if total <= max_rows:
        return np.arange(total, dtype=np.int64)
    return (np.arange(max_rows, dtype=np.int64) * np.int64(total)) // np.int64(max_rows)


def training_set(
    frame: pd.DataFrame, fold: Fold, hold_s: float, fee_bp: float, columns: Sequence[str],
    max_rows: int,
) -> tuple[pd.DataFrame, pd.Series]:  # fmt: skip
    """The fold's training rows for one fit, as the purge and the stride leave them."""
    stamps = stamps_ns(frame)
    keep = fold.train_keep(stamps, frame["reads_until_ns"].to_numpy(), hold_s)
    keep &= frame["scored"].to_numpy(dtype=bool)
    at = np.flatnonzero(keep)
    at = at[thin_positions(len(at), max_rows)]
    x = frame.iloc[at][list(columns)].astype(np.float64).replace([np.inf, -np.inf], np.nan)
    y = frame.iloc[at][label_column(hold_s, fee_bp)].astype(np.float64)
    return clean(x.reset_index(drop=True), y.reset_index(drop=True))


def fit_model(x: pd.DataFrame, y: pd.Series, params: Mapping[str, Any]) -> Any | None:
    """A ``gbm`` on the rows, or None when they hold fewer than two classes."""
    from trading_research.models.gbm import GradientBoostedBaseline

    if len(y) < 2 or y.nunique() < 2:
        return None
    return GradientBoostedBaseline(**dict(params)).fit(x, y)


def rules_at(proba: np.ndarray, days: int, rates: Sequence[float]) -> dict[float, HoldFirstRule]:
    """The hold-first thresholds per rate on the kept inner-validation rows."""
    if not len(proba) or days <= 0:
        return dict.fromkeys(rates, NEVER)
    return {rate: fit_hold_first(proba, days=float(days), trades_per_day=rate) for rate in rates}


def finite_theta(rule: HoldFirstRule) -> dict[str, float]:
    """A threshold as frozen: a side that never trades is written as 2."""
    return {k: (v if math.isfinite(v) else 2.0) for k, v in asdict(rule).items()}


def save_model(model: Any, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return file_sha256(path)


def load_model(path: Path, digest: str | None = None) -> Any:
    """A fit written by :func:`save_model` (this study's own files, under the ignored
    ``models/`` or ``artifacts/``), checked against ``digest`` when one is given."""
    if digest is not None and file_sha256(path) != digest:
        raise HeldOutLocked(f"{path} is not at its frozen sha256 {digest}")
    with path.open("rb") as handle:
        return pickle.load(handle)


def store_path(study: SpotStudy, store: str, day: Any) -> Path:
    return study.work / "1" / "stores" / store / f"{day}.parquet"


def store_rows(
    study: SpotStudy, store: str, days: Sequence[Any], columns: Sequence[str]
) -> pd.DataFrame:
    """The every-tenth-row store of ``days``, with the targets and the scored flag."""
    keep = ["timestamp", "scored", "reads_until_ns", *columns]
    frames = []
    for day in days:
        path = store_path(study, store, day)
        if not path.exists():  # the day is not usable on the store's venue
            continue
        frame = pd.read_parquet(path)
        targets = [c for c in frame.columns if c.startswith("label_")]
        frames.append(frame[[*keep, *targets]])
    if not frames:
        return pd.DataFrame(columns=keep)
    return pd.concat(frames, ignore_index=True)


def fitted_rows(store: str) -> list[Row]:
    return [r for r in ROWS.values() if r.fitted and r.store == store]


def inner_probas(
    fold: Fold, store: str, models: Mapping[tuple[str, float, float], Any], layout: Any,
    load: Callable[[Any], dy.DayInputs],
) -> tuple[dict[tuple[str, float, float], np.ndarray], int]:  # fmt: skip
    """Every model's probabilities on the kept rows of the inner validation's usable
    days of the store's venue (features computed per day, never stored)."""
    name = dy.STORES[store][0]
    found: dict[tuple[str, float, float], list[np.ndarray]] = {k: [] for k in models}
    used = 0
    for inputs in dy.walk_days(fold.inner, load):
        if not inputs.usable(name):
            continue
        used += 1
        frame = dy.store_frame(store, inputs)
        stamps = stamps_ns(frame)
        for key, model in models.items():
            proba = dy.predict(model, frame, layout, inputs.scored_from(name))
            keep = fold.inner_keep(stamps, key[1]) & np.isfinite(proba).all(axis=1)
            found[key].append(proba[keep])
    return {k: np.concatenate(v) if v else np.zeros((0, 3)) for k, v in found.items()}, used


StoreReader = Callable[[SpotStudy, str, Sequence[Any], Sequence[str]], pd.DataFrame]


def fit_fold(
    study: SpotStudy, fold: Fold, load: Callable[[Any], dy.DayInputs], touch_size: float, *,
    read: StoreReader | None = None, folder: Path | None = None,
) -> dict[str, Any]:  # fmt: skip
    """Every fit of the fold and every cell's thresholds; the models are written under
    the work directory (the frozen fold's are copied to ``models/`` by the amendment).
    ``read`` reads the stores (:func:`store_rows`) and ``folder`` takes the models:
    the fold-level leakage test runs this same path on mutated data."""
    out: dict[str, Any] = {"fold": fold.name, "train": list(fold.train), "inner": list(fold.inner),
                           "fits": {}, "thresholds": {}, "inner_days": {}}  # fmt: skip
    folder = folder or study.dir("D", "fits", fold.name)
    read = read or store_rows
    for store in dy.STORES:
        rows = fitted_rows(store)
        if not rows:
            continue
        columns = sorted({c for r in rows for c in r.inputs})
        frame = read(study, store, fold.train, columns)
        models: dict[tuple[str, float, float], Any] = {}
        for row in rows:
            for hold, fee in row.fits:
                x, y = training_set(
                    frame, fold, hold, fee, row.inputs, study.scale.max_training_rows
                )
                model = fit_model(x, y, study.scale.model_params)
                key = fit_key(row.name, hold, fee)
                digest = None if model is None else save_model(model, folder / key)
                out["fits"][key] = {"rows": len(y), "classes": sorted(int(c) for c in y.unique()),
                                    "sha256": digest}  # fmt: skip
                if model is not None:
                    models[(row.name, hold, fee)] = model
        del frame
        probas, used = inner_probas(fold, store, models, dy.layout(store, touch_size), load)
        out["inner_days"][store] = used
        for row in rows:
            cells: dict[str, dict[str, float]] = {}
            for cell in row.cells:
                proba = probas.get((row.name, cell.hold_s, cell.fee_bp), np.zeros((0, 3)))
                cells[cell.key] = finite_theta(rules_at(proba, used, [cell.rate])[cell.rate])
            out["thresholds"][row.name] = cells
    return out


def row_models(
    study: SpotStudy, record: Mapping[str, Any], touch_size: float, folder: Path | None = None
) -> dict[str, RowModels]:
    """The fitted rows' models and rules as a fold's record (or the frozen files) give them."""
    folder = folder or study.work / "D" / "fits" / str(record["fold"])
    out = {}
    for name, row in ROWS.items():
        if not row.fitted:
            continue
        models = {}
        for hold, fee in row.fits:
            entry = record["fits"].get(fit_key(name, hold, fee), {})
            if entry.get("sha256"):
                models[(hold, fee)] = load_model(folder / fit_key(name, hold, fee), entry["sha256"])
        rules = {c: HoldFirstRule(**record["thresholds"][name][c.key]) for c in row.cells}
        out[name] = RowModels(name, models, rules, dy.layout(row.store, touch_size))
    return out


def check_2025(study: SpotStudy) -> Path:
    """The 2025 configuration at its frozen sha256 and every 2025 file S25-T loads at
    the sha256 the binding records; the checkout that holds them."""
    b = study.binding
    config = study.repo / b.config
    if not config.is_file() or file_sha256(config) != b.config_sha256:
        raise HeldOutLocked(f"{b.config} is not at its frozen sha256 {b.config_sha256}")
    root = b.root or study.repo
    for relative, digest in b.files.items():
        target = root / relative
        if not target.is_file() or file_sha256(target) != digest:
            raise HeldOutLocked(f"the 2025 file {relative} is missing or not at {digest}")
    return root


def frozen_2025(study: SpotStudy) -> RowModels:
    """S25-T: the 2025 study's frozen fits and thresholds, unchanged, checked by sha256."""
    import json

    from trading_research.spot import registration as sr
    from trading_research.spot.book_inputs import touch_size_2025

    root = check_2025(study)
    config = yaml.safe_load((study.repo / study.binding.config).read_text(encoding="utf-8"))
    thresholds = config["slots"]["S25"]["thresholds"]
    fitted = json.loads((root / sr.FROZEN_DIR_2025 / "fitted_values.json").read_text())
    row = ROWS["S25-T"]
    models = {
        (h, f): load_model(root / sr.FROZEN_DIR_2025 / "weights" / FIT_2025.format(h=h, f=f))
        for h, f in row.fits
    }
    rules = {c: HoldFirstRule(**thresholds[KEY_2025.format(h=c.hold_s, r=c.rate, f=c.fee_bp)])
             for c in row.cells}  # fmt: skip
    return RowModels(row.name, models, rules, dy.layout("book", touch_size_2025(fitted)))


def cells_rules(record: Mapping[str, Any]) -> dict[str, dict[Cell, HoldFirstRule]]:
    return {
        row: {c: HoldFirstRule(**record[row][c.key]) for c in ROWS[row].cells} for row in record
    }
