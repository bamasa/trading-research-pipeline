"""The pipeline as separate stages, each writing its output to disk.

Five stages, each runnable on its own:

1. ``prepare``  — raw book and trades to a feature matrix, one file per day
2. ``select``   — choose features on a training window, write the list
3. ``train``    — fit a model on a training window, write the model
4. ``predict``  — run a model over a period, write probabilities
5. ``backtest`` — turn probabilities into trades and profit

Why the files, rather than one function that does everything: a stage can be
re-run without repeating what came before, which matters when feature building
takes five minutes and model fitting takes thirty seconds. More importantly,
every intermediate is inspectable. A pipeline that only produces a final number
is a pipeline whose middle nobody checks, and the middle is where the mistakes
in this kind of work actually live.

Each stage writes a manifest beside its output recording what produced it — the
configuration, the input files, the code version. That is what makes a result
traceable back to its inputs rather than to a memory of how it was run.

Ordering guarantee
------------------
Stages that fit anything — ``select`` and ``train`` — take an explicit date
range and read only that range. They cannot see later data because they are
never handed it. ``predict`` and ``backtest`` do not fit anything at all.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_research import __version__

MANIFEST = "manifest.json"


class StageError(RuntimeError):
    """A stage cannot run with the inputs it was given."""


@dataclass
class StageManifest:
    """Provenance for one stage's output."""

    stage: str
    trading_research_version: str = __version__
    inputs: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)

    def write(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / MANIFEST
        path.write_text(json.dumps(asdict(self), indent=2, default=str) + "\n", encoding="utf-8")
        return path

    @staticmethod
    def read(directory: Path) -> dict[str, Any]:
        path = Path(directory) / MANIFEST
        if not path.exists():
            raise StageError(f"no {MANIFEST} in {directory}; run the earlier stage first")
        loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return loaded


# ---------------------------------------------------------------------------
# Stage 1: prepare
# ---------------------------------------------------------------------------


def prepare(
    raw_dir: Path | str,
    out_dir: Path | str,
    symbol: str,
    *,
    horizon: int,
    subsample: int = 1,
    warmup: int = 1200,
    overwrite: bool = False,
) -> Path:
    """Build the feature matrix, one file per day.

    Processed a day at a time because the full series does not fit: 30M rows at
    ~200 columns is tens of gigabytes, and an earlier version that built it in
    one go drove the machine into swap.

    Rolling windows are warmed from the tail of the previous day rather than
    restarted at each boundary, so a day break does not produce a block of
    missing values or, worse, a window that quietly means something different
    from every other window.
    """
    from trading_research.features.generated import generate
    from trading_research.features.registry import REGISTRY

    raw = Path(raw_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    book_files = sorted((raw / symbol / "bookTicker").glob("*.parquet"))
    if not book_files:
        raise StageError(f"no book data under {raw / symbol / 'bookTicker'}")
    trade_files = {p.stem: p for p in (raw / symbol / "aggTrades").glob("*.parquet")}

    written: list[str] = []
    tail: pd.DataFrame | None = None

    for book_file in book_files:
        day = book_file.stem
        target = out / f"{day}.parquet"
        if target.exists() and not overwrite:
            written.append(day)
            tail = None  # cannot warm from a day that was not read
            continue

        book = pd.read_parquet(book_file)
        for column in ("symbol", "source"):
            book[column] = book[column].astype("string")

        warm_rows = 0
        if tail is not None:
            warm_rows = len(tail)
            book = pd.concat([tail, book], ignore_index=True)
        tail = book.tail(warmup).copy()

        trades = None
        if day in trade_files:
            trades = pd.read_parquet(trade_files[day])
            for column in ("symbol", "source"):
                trades[column] = trades[column].astype("string")

        features = generate(book, trades).iloc[warm_rows:]
        mid = ((book["bid_price_0"] + book["ask_price_0"]) / 2).to_numpy()
        forward = np.full(len(mid), np.nan)
        if horizon < len(mid):
            forward[:-horizon] = np.log(mid[horizon:] / mid[:-horizon]) * 1e4

        features.insert(0, "timestamp", book["timestamp"].iloc[warm_rows:])
        features["forward_bp"] = forward[warm_rows:]
        features["spread_bp_now"] = REGISTRY.get("spread_bp")(book).iloc[warm_rows:]

        features.iloc[::subsample].reset_index(drop=True).to_parquet(
            target, compression="zstd", index=False
        )
        written.append(day)
        del book, trades, features

    StageManifest(
        stage="prepare",
        inputs={"raw_dir": str(raw), "symbol": symbol},
        params={"horizon": horizon, "subsample": subsample, "warmup": warmup},
        outputs={"days": written, "n_days": len(written)},
    ).write(out)
    return out


def load_prepared(
    directory: Path | str,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """Read prepared days, optionally restricted to a date range.

    The range is applied by filename before anything is read, so a stage asked
    for a training window never has later data in memory at all. That is a
    stronger guarantee than filtering after loading: code cannot accidentally
    use what was never there.
    """
    path = Path(directory)
    files = sorted(path.glob("*.parquet"))
    if not files:
        raise StageError(f"no prepared data in {path}; run prepare first")

    if start is not None or end is not None:
        chosen = []
        for file in files:
            day = date.fromisoformat(file.stem)
            if start is not None and day < start:
                continue
            if end is not None and day > end:
                continue
            chosen.append(file)
        files = chosen

    if not files:
        raise StageError(f"no prepared days in {path} between {start} and {end}")

    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """The feature columns of a prepared frame, excluding bookkeeping."""
    reserved = {"timestamp", "forward_bp", "spread_bp_now", "label"}
    return [c for c in frame.columns if c not in reserved]


# ---------------------------------------------------------------------------
# Labelling helper, shared by the stages that need a target
# ---------------------------------------------------------------------------


def add_label(frame: pd.DataFrame, threshold_bp: float) -> pd.DataFrame:
    """Attach the three-class label to a prepared frame.

    Not done in ``prepare`` on purpose: the threshold belongs to the experiment,
    not to the data. Baking it in would mean re-running feature generation —
    minutes per instrument — to answer "what if we only traded larger moves".
    """
    move = frame["forward_bp"]
    frame = frame.copy()
    frame["label"] = np.where(move.isna(), np.nan, np.sign(move) * (move.abs() > threshold_bp))
    return frame


# ---------------------------------------------------------------------------
# Stage 2: select
# ---------------------------------------------------------------------------


def select(
    prepared_dir: Path | str,
    out_dir: Path | str,
    *,
    train_start: date,
    train_end: date,
    threshold_bp: float,
    max_features: int = 40,
    score_target: str = "ic",
) -> Path:
    """Choose features on a training window and write the list.

    Reads only the training window. That is enforced by loading only those
    files rather than by filtering afterwards, so later data is not merely
    unused — it is never in memory. Selecting over the whole sample is one of
    the most effective ways to manufacture an edge: with a few hundred
    candidates, some will look predictive on the test period by chance, and
    choosing them for that is choosing them for their test performance.
    """
    from trading_research.features.selection import FeatureSelector

    out = Path(out_dir)
    frame = add_label(load_prepared(prepared_dir, train_start, train_end), threshold_bp)
    columns = feature_columns(frame)

    usable = frame[columns].notna().all(axis=1) & frame["label"].notna()
    if int(usable.sum()) < 1000:
        raise StageError(
            f"only {int(usable.sum())} complete rows in {train_start}..{train_end}; "
            f"widen the window or shorten the longest feature window"
        )

    selector = FeatureSelector(max_features=max_features, score_target=score_target)
    selector.fit(
        frame.loc[usable, columns], frame.loc[usable, "label"], frame.loc[usable, "forward_bp"]
    )
    assert selector.report_ is not None

    out.mkdir(parents=True, exist_ok=True)
    (out / "features.json").write_text(
        json.dumps({"features": selector.selected_}, indent=2) + "\n", encoding="utf-8"
    )
    if selector.report_.scores is not None:
        selector.report_.scores.rename("score").to_frame().to_csv(out / "scores.csv")

    StageManifest(
        stage="select",
        inputs={"prepared_dir": str(prepared_dir), "train": [str(train_start), str(train_end)]},
        params={
            "threshold_bp": threshold_bp,
            "max_features": max_features,
            "score_target": score_target,
        },
        outputs=selector.report_.to_dict(),
    ).write(out)
    return out


def load_selected(directory: Path | str) -> list[str]:
    path = Path(directory) / "features.json"
    if not path.exists():
        raise StageError(f"no features.json in {directory}; run select first")
    payload: dict[str, list[str]] = json.loads(path.read_text(encoding="utf-8"))
    return payload["features"]


# ---------------------------------------------------------------------------
# Stage 3: train
# ---------------------------------------------------------------------------

MODELS = {
    "always_hold": "trading_research.models.base:AlwaysHold",
    "class_prior": "trading_research.models.base:ClassPrior",
    "logistic": "trading_research.models.linear:LogisticBaseline",
    "xgboost": "trading_research.models.gbm:GradientBoostedBaseline",
    "tcn": "trading_research.models.tcn:TCNBaseline",
}


def build_model(name: str, **params: Any) -> Any:
    """Instantiate a model by name, so a config can select one as a string."""
    if name not in MODELS:
        raise StageError(f"unknown model {name!r}; known: {', '.join(sorted(MODELS))}")
    module_path, class_name = MODELS[name].split(":")
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)(**params)


def train(
    prepared_dir: Path | str,
    features_dir: Path | str,
    out_dir: Path | str,
    *,
    train_start: date,
    train_end: date,
    threshold_bp: float,
    model: str = "logistic",
    params: dict[str, Any] | None = None,
) -> Path:
    """Fit one model on the training window and write it to disk.

    Like ``select``, this reads only the training window. Nothing later exists
    as far as this stage is concerned.
    """
    import pickle

    out = Path(out_dir)
    columns = load_selected(features_dir)
    frame = add_label(load_prepared(prepared_dir, train_start, train_end), threshold_bp)

    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise StageError(f"selected features missing from prepared data: {missing[:5]}")

    usable = frame[columns].notna().all(axis=1) & frame["label"].notna()
    x, y = frame.loc[usable, columns], frame.loc[usable, "label"]
    if len(x) < 500:
        raise StageError(f"only {len(x)} usable training rows")

    estimator = build_model(model, **(params or {}))
    estimator.fit(x, y)

    out.mkdir(parents=True, exist_ok=True)
    with (out / "model.pkl").open("wb") as handle:
        pickle.dump({"model": estimator, "features": columns}, handle)

    StageManifest(
        stage="train",
        inputs={
            "prepared_dir": str(prepared_dir),
            "features_dir": str(features_dir),
            "train": [str(train_start), str(train_end)],
        },
        params={"model": model, "threshold_bp": threshold_bp, **(params or {})},
        outputs={"n_rows": len(x), "n_features": len(columns), "model": estimator.describe()},
    ).write(out)
    return out


def load_model(directory: Path | str) -> tuple[Any, list[str]]:
    import pickle

    path = Path(directory) / "model.pkl"
    if not path.exists():
        raise StageError(f"no model.pkl in {directory}; run train first")
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return payload["model"], payload["features"]


# ---------------------------------------------------------------------------
# Stage 4: predict
# ---------------------------------------------------------------------------


def predict(
    prepared_dir: Path | str,
    model_dir: Path | str,
    out_dir: Path | str,
    *,
    start: date,
    end: date,
) -> Path:
    """Run a fitted model over a period and write the probabilities.

    Fits nothing. Kept separate from ``train`` so that the same model can be
    scored over validation and over test without any chance of the second run
    differing from the first.
    """
    from trading_research.models.base import CLASSES

    out = Path(out_dir)
    estimator, columns = load_model(model_dir)
    frame = load_prepared(prepared_dir, start, end)

    usable = frame[columns].notna().all(axis=1)
    proba = estimator.predict_proba(frame.loc[usable, columns])

    result = pd.DataFrame(
        {
            "timestamp": frame.loc[usable, "timestamp"].to_numpy(),
            "forward_bp": frame.loc[usable, "forward_bp"].to_numpy(),
            "spread_bp_now": frame.loc[usable, "spread_bp_now"].to_numpy(),
        }
    )
    for i, cls in enumerate(CLASSES):
        result[f"p_{cls}"] = proba[:, i]

    out.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out / "predictions.parquet", compression="zstd", index=False)

    StageManifest(
        stage="predict",
        inputs={"prepared_dir": str(prepared_dir), "model_dir": str(model_dir)},
        params={"start": str(start), "end": str(end)},
        outputs={"n_rows": len(result), "dropped_incomplete": int((~usable).sum())},
    ).write(out)
    return out


def load_predictions(directory: Path | str) -> pd.DataFrame:
    path = Path(directory) / "predictions.parquet"
    if not path.exists():
        raise StageError(f"no predictions.parquet in {directory}; run predict first")
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Stage 5: backtest
# ---------------------------------------------------------------------------


def backtest(
    predictions_dir: Path | str,
    out_dir: Path | str,
    *,
    min_confidence: float,
    hold_periods: int,
    cooldown_periods: int = 0,
    fee_bp_per_side: float = 5.0,
    slippage_bp: float = 0.5,
    allow_reversal: bool = False,
) -> Path:
    """Turn probabilities into trades and profit.

    Fits nothing and chooses nothing: ``min_confidence`` arrives from whoever
    selected it on validation. Keeping the choice outside this stage is what
    stops the threshold from quietly being tuned on whatever period the
    backtest happens to be run over.
    """
    from trading_research.backtest.costs import TakerCosts
    from trading_research.backtest.evaluate import decide
    from trading_research.backtest.execution import ThinningRules, score, thin, trades_to_frame
    from trading_research.models.base import CLASSES

    out = Path(out_dir)
    frame = load_predictions(predictions_dir)
    proba = frame[[f"p_{c}" for c in CLASSES]].to_numpy()

    costs = TakerCosts(fee_bp_per_side=fee_bp_per_side, slippage_bp=slippage_bp)
    rules = ThinningRules(
        hold_periods=hold_periods,
        cooldown_periods=cooldown_periods,
        allow_reversal=allow_reversal,
    )

    decision = decide(proba, min_confidence=min_confidence)
    trades = thin(
        decision,
        frame["forward_bp"].to_numpy(),
        frame["spread_bp_now"].to_numpy(),
        rules,
    )
    summary = score(trades, costs)

    # The same signals without thinning, for comparison. Reported together
    # because the interesting number is not either total but what thinning did
    # to profit *per trade*.
    untinned = thin(
        decision,
        frame["forward_bp"].to_numpy(),
        frame["spread_bp_now"].to_numpy(),
        ThinningRules(hold_periods=1, cooldown_periods=0),
    )
    baseline = score(untinned, costs)

    out.mkdir(parents=True, exist_ok=True)
    trades_to_frame(trades).to_parquet(out / "trades.parquet", compression="zstd", index=False)

    StageManifest(
        stage="backtest",
        inputs={"predictions_dir": str(predictions_dir)},
        params={
            "min_confidence": min_confidence,
            "hold_periods": hold_periods,
            "cooldown_periods": cooldown_periods,
            "allow_reversal": allow_reversal,
            "costs": costs.describe(),
        },
        outputs={"thinned": summary, "every_signal": baseline},
    ).write(out)
    return out
