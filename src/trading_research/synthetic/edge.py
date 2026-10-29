"""Recovering a planted edge: features, labels, a model or rule, the taker executor.

:func:`recover` runs the repository's own pipeline on a :class:`BookDataset`
and measures three numbers per trade:

* the **recovered** gross: the side times the move of the book's mid from the
  snapshot the entry walked to the one the exit walked, in bp — what the trade
  earned before spread and fees;
* the **planted** gross: the same pair of snapshots read on the generator's
  latent drift (:meth:`BookDataset.planted_move_bp`) — what the trade was owed by
  the planted effect alone, with the unpredictable part removed;
* the **net** at the stated fee, from the executor.

Their difference (recovered minus planted) is pure noise for a causal strategy,
whatever it trades: a strategy that reads the future would pick trades whose
noise is favourable, and the difference would be positive. So the checks read
the residual as a leak detector as well as a measure of bias.

The days are split in order: the first share trains (a rule only fits a scale),
the next chooses the confidence threshold through
:func:`trading_research.backtest.evaluate.choose_confidence`, and the rest are
traded by :func:`trading_research.backtest.taker.execute_day`, one position at a
time. Every statistic is clustered by day with
:func:`trading_research.evaluation.significance.assess`. The placebo executes
each test day's decisions, at the same times of day, on every other test day.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, time
from typing import Any, Protocol

import numpy as np
import pandas as pd

from trading_research.backtest.costs import FeeTier, TakerCosts
from trading_research.synthetic.book_days import BookDataset
from trading_research.synthetic.common import NS_PER_S

#: Book features every strategy is offered. Registry names.
BOOK_FEATURES: tuple[str, ...] = (
    "queue_imbalance",
    "microprice_dev_bp",
    "log_depth_ratio",
    "spread_bp",
    "ofi_20_norm",
    "mid_return_1_bp",
    "mid_return_20_bp",
    "realized_vol_50_bp",
)


#: The default stated fee: Bybit's base taker rate, 5.5 bp a side.
STATED_FEE = FeeTier("stated", 2.0, 5.5)


class ModelLike(Protocol):
    def fit(self, x: pd.DataFrame, y: pd.Series) -> Any: ...

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray: ...


#: A strategy: a registered model or rule name (``"order_flow"``, or
#: ``"order_flow:flow_imbalance_6"`` to point a rule at another column), an object
#: with ``fit`` and ``predict_proba``, or a callable from the feature frame to
#: decisions in {-1, 0, +1} that fits nothing.
StrategyLike = str | ModelLike | Callable[[pd.DataFrame], np.ndarray]


@dataclass(frozen=True)
class EdgeSettings:
    """How the pipeline is run on a dataset. The fee is stated, never implied."""

    fee: FeeTier = STATED_FEE
    clip: float = 0.01
    latency_ns: int = 10_000_000
    train_share: float = 0.375
    validation_share: float = 0.125
    #: Learned models fit on every ``train_stride``-th training row.
    train_stride: int = 5
    #: Label threshold in bp; None uses the round trip at the stated fee and
    #: the median spread.
    label_threshold_bp: float | None = None
    objective: str = "net_per_trade_bp"
    placebo: bool = True


@dataclass
class EdgeReport:
    """What one strategy did on one dataset, with the planted value beside it."""

    strategy: str
    feature: str
    null: bool
    nominal_bp: float
    fee_bp: float
    confidence: float
    trades: pd.DataFrame
    summary: dict[str, float]
    placebo: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "feature": self.feature,
            "null": self.null,
            "nominal_bp": self.nominal_bp,
            "fee_taker_bp": self.fee_bp,
            "confidence": self.confidence,
            **self.summary,
            **{f"placebo_{k}": v for k, v in self.placebo.items()},
        }


def strategy_name(strategy: StrategyLike) -> str:
    if isinstance(strategy, str):
        return strategy
    return str(getattr(strategy, "name", getattr(strategy, "__name__", type(strategy).__name__)))


def day_features(dataset: BookDataset, day: date) -> pd.DataFrame:
    """Book and flow features of one day, plus the forward move and the spread."""
    from trading_research.features import REGISTRY, build
    from trading_research.features.flow import build_flow_features
    from trading_research.features.prints import bucket_prints

    book = dataset.book[day]
    frame = build(book, list(BOOK_FEATURES), registry=REGISTRY)
    flow = build_flow_features(bucket_prints(dataset.prints(day), book))
    frame = pd.concat([frame, flow], axis=1)
    mid = ((book["bid_price_0"] + book["ask_price_0"]) / 2.0).to_numpy(dtype=np.float64)
    log_mid = np.log(mid)
    frame["log_mid_ret20"] = frame["mid_return_20_bp"]
    ret50 = np.full(len(mid), np.nan)
    ret50[50:] = (log_mid[50:] - log_mid[:-50]) * 1e4
    frame["log_mid_ret50"] = ret50
    frame["log_mid_vol50"] = frame["realized_vol_50_bp"]
    horizon = round(dataset.spec.horizon_s / dataset.spec.step_s)
    forward = np.full(len(mid), np.nan)
    forward[:-horizon] = (log_mid[horizon:] - log_mid[:-horizon]) * 1e4
    frame["forward_bp"] = forward
    frame["spread_bp_now"] = frame["spread_bp"]
    frame["day"] = day.isoformat()
    return frame


RESERVED = ("timestamp", "forward_bp", "spread_bp_now", "day", "label")


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in RESERVED]


def _resolve(strategy: StrategyLike) -> tuple[str, Any]:
    """('model', fitted-later object) or ('rule', callable)."""
    if isinstance(strategy, str):
        from trading_research.pipeline.stages import build_model

        name, _, column = strategy.partition(":")
        return "model", build_model(name, column=column) if column else build_model(name)
    if hasattr(strategy, "fit") and hasattr(strategy, "predict_proba"):
        return "model", strategy
    if callable(strategy):
        return "rule", strategy
    raise TypeError(f"not a strategy: {strategy!r}")


def split_days(days: list[date], settings: EdgeSettings) -> tuple[list[date], ...]:
    n = len(days)
    train = max(1, round(n * settings.train_share))
    validation = max(1, round(n * settings.validation_share))
    if train + validation >= n - 1:
        raise ValueError(f"{n} days leave fewer than two test days for this split")
    return days[:train], days[train : train + validation], days[train + validation :]


def _taker_settings(dataset: BookDataset, settings: EdgeSettings) -> Any:
    from trading_research.backtest.taker import TakerSettings

    spec = dataset.spec

    def clock(seconds: float) -> time:
        seconds = max(0.0, seconds)
        return time(int(seconds // 3600), int(seconds % 3600 // 60), int(seconds % 60))

    end = float(spec.session_s)
    return TakerSettings(
        clip=settings.clip,
        fees=settings.fee,
        latency_ns=settings.latency_ns,
        no_entry_before=clock(min(60.0, spec.warmup_s)),
        no_entry_after=clock(end - spec.horizon_s - 10.0),
        flatten_at=clock(end - 2.0),
    )


def execute(
    dataset: BookDataset,
    day: date,
    decision_ns: np.ndarray,
    side: np.ndarray,
    settings: EdgeSettings,
    *,
    source_day: date | None = None,
) -> pd.DataFrame:
    """Trade one day's decisions with the taker executor and score each trade."""
    from trading_research.backtest.taker import TakerBook, execute_day

    book = TakerBook.from_frame(dataset.book[day], day, depth=dataset.spec.depth)
    hold = int(dataset.spec.horizon_s * NS_PER_S)
    result = execute_day(book, decision_ns, side, hold, _taker_settings(dataset, settings))
    trades = result.trades.copy()
    if trades.empty:
        return trades
    side_f = trades["side"].to_numpy(dtype=np.float64)
    entry_mid = trades["entry_mid"].to_numpy(dtype=np.float64)
    exit_mid = trades["exit_mid"].to_numpy(dtype=np.float64)
    trades["gross_mid_bp"] = side_f * np.log(exit_mid / entry_mid) * 1e4
    planted = dataset.planted_move_bp(
        day,
        trades["entry_snapshot_ns"].to_numpy(dtype=np.int64),
        trades["exit_snapshot_ns"].to_numpy(dtype=np.int64),
    )
    trades["planted_bp"] = side_f * planted
    trades["residual_bp"] = trades["gross_mid_bp"] - trades["planted_bp"]
    trades["day"] = day.isoformat()
    trades["source_day"] = (source_day or day).isoformat()
    return trades


def _significance(trades: pd.DataFrame, value: str, cluster: str = "day") -> tuple[float, float]:
    """Cluster mean and t of ``value``; NaN when there is too little to say."""
    from trading_research.evaluation.significance import SignificanceError, assess

    if trades.empty:
        return float("nan"), float("nan")
    try:
        result = assess(trades, value=value, cluster=cluster, series=None)
    except SignificanceError:
        return float(trades[value].mean()), float("nan")
    return result.cluster_mean_bp, result.cluster_t


def _se(trades: pd.DataFrame, value: str, cluster: str = "day") -> float:
    means = trades.groupby(cluster)[value].mean().to_numpy(dtype=np.float64)
    if len(means) < 2:
        return float("nan")
    return float(np.std(means, ddof=1) / math.sqrt(len(means)))


def summarise(trades: pd.DataFrame) -> dict[str, float]:
    """Trade-level and day-clustered figures of recovered, planted, residual and net."""
    if trades.empty:
        return {"trades": 0.0, "days_traded": 0.0}
    out: dict[str, float] = {
        "trades": float(len(trades)),
        "days_traded": float(trades["day"].nunique()),
    }
    for name, column in (
        ("recovered", "gross_mid_bp"),
        ("planted", "planted_bp"),
        ("residual", "residual_bp"),
        ("net", "net_bp"),
    ):
        mean, t = _significance(trades, column)
        out[f"{name}_bp"] = float(trades[column].mean())
        out[f"{name}_day_mean_bp"] = mean
        out[f"{name}_t"] = t
    out["residual_se_bp"] = _se(trades, "residual_bp")
    return out


def recover(
    strategy: StrategyLike,
    dataset: BookDataset,
    settings: EdgeSettings | None = None,
    *,
    frames: dict[date, pd.DataFrame] | None = None,
) -> EdgeReport:
    """Run the pipeline for one strategy on one dataset and score it against the truth."""
    from trading_research.backtest.evaluate import choose_confidence, decide
    from trading_research.models.base import clean

    settings = settings or EdgeSettings()
    frames = frames or {day: day_features(dataset, day) for day in dataset.days}
    train_days, validation_days, test_days = split_days(dataset.days, settings)
    kind, model = _resolve(strategy)
    spec = dataset.spec
    costs = TakerCosts(fee_bp_per_side=settings.fee.taker_bp, slippage_bp=0.0)
    threshold = settings.label_threshold_bp
    train = pd.concat([frames[d] for d in train_days], ignore_index=True)
    if threshold is None:
        spread = float(np.nanmedian(train["spread_bp_now"].to_numpy(dtype=np.float64)))
        threshold = float(np.asarray(costs.round_trip_bp(spread)))
    columns = _feature_columns(train)

    confidence = float("nan")
    decisions: dict[date, np.ndarray] = {}
    if kind == "model":
        move = train["forward_bp"]
        label = np.where(move.isna(), np.nan, np.sign(move) * (move.abs() > threshold))
        x, y = clean(train[columns], pd.Series(label, index=train.index))
        stride = 1 if getattr(model, "column", "") else settings.train_stride
        model.fit(x.iloc[::stride], y.iloc[::stride])
        validation = pd.concat([frames[d] for d in validation_days], ignore_index=True)
        ok = validation[columns].notna().all(axis=1) & validation["forward_bp"].notna()
        confidence, _ = choose_confidence(
            model.predict_proba(validation.loc[ok, columns]),
            validation.loc[ok, "forward_bp"],
            validation.loc[ok, "spread_bp_now"],
            costs,
            objective=settings.objective,
        )
        for day in test_days:
            frame = frames[day]
            usable = frame[columns].notna().all(axis=1).to_numpy()
            out = np.zeros(len(frame), dtype=np.int64)
            if usable.any():
                proba = model.predict_proba(frame.loc[usable, columns])
                out[usable] = decide(proba, min_confidence=confidence)
            decisions[day] = out
    else:
        for day in test_days:
            # The forward move is the label's, never a feature: a rule never sees it.
            visible = frames[day].drop(columns=["forward_bp"])
            values = np.asarray(model(visible), dtype=np.float64)
            decisions[day] = np.sign(np.nan_to_num(values)).astype(np.int64)

    parts = []
    for day in test_days:
        when, side = _decision_arrays(frames[day], decisions[day])
        parts.append(execute(dataset, day, when, side, settings))
    trades = pd.concat([p for p in parts if not p.empty] or [pd.DataFrame()], ignore_index=True)
    report = EdgeReport(
        strategy=strategy_name(strategy),
        feature=spec.feature,
        null=spec.null,
        nominal_bp=0.0 if spec.null else spec.size_bp,
        fee_bp=settings.fee.taker_bp,
        confidence=confidence,
        trades=trades,
        summary=summarise(trades),
    )
    if settings.placebo:
        report.placebo = placebo(dataset, frames, decisions, test_days, settings)
    report.summary["label_threshold_bp"] = threshold
    report.summary["test_days"] = float(len(test_days))
    return report


def _decision_arrays(frame: pd.DataFrame, decision: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from trading_research.features.timed import stamps_ns

    stamps = stamps_ns(frame)
    taken = np.flatnonzero(decision != 0)
    return stamps[taken].astype(np.int64), decision[taken].astype(np.int64)


def placebo(
    dataset: BookDataset,
    frames: dict[date, pd.DataFrame],
    decisions: dict[date, np.ndarray],
    test_days: list[date],
    settings: EdgeSettings,
) -> dict[str, float]:
    """Each test day's decisions, at the same times of day, on every other test day.

    Every shift ``s`` from 1 to n - 1 moves day ``i``'s decisions to day
    ``(i + s) mod n``: the count, the sides and the timing of the decisions are
    kept, only their alignment with the market is broken. Pooled over shifts,
    clustered by (shift, day).
    """
    from trading_research.synthetic.common import day_start_ns

    n = len(test_days)
    pooled: list[pd.DataFrame] = []
    worst = -math.inf
    for shift in range(1, n):
        shifted: list[pd.DataFrame] = []
        for i, source in enumerate(test_days):
            target = test_days[(i + shift) % n]
            when, side = _decision_arrays(frames[source], decisions[source])
            moved = when - day_start_ns(source) + day_start_ns(target)
            part = execute(dataset, target, moved, side, settings, source_day=source)
            if not part.empty:
                part["cell"] = target.isoformat()
                shifted.append(part)
        if shifted:
            frame = pd.concat(shifted, ignore_index=True)
            _, t = _significance(frame, "gross_mid_bp")
            if np.isfinite(t):
                worst = max(worst, t)
            pooled.append(frame)
    if not pooled:
        return {"shifts": float(n - 1), "trades": 0.0}
    every = pd.concat(pooled, ignore_index=True)
    mean, t = _significance(every, "gross_mid_bp", cluster="cell")
    planted, _ = _significance(every, "planted_bp", cluster="cell")
    return {
        "shifts": float(n - 1),
        "trades": float(len(every)),
        "recovered_bp": mean,
        "recovered_t": t,
        "planted_bp": planted,
        "max_shift_t": worst if np.isfinite(worst) else float("nan"),
    }


def load_strategy(text: str) -> StrategyLike:
    """A strategy named on the command line.

    ``package.module:attribute`` imports the attribute (a class is instantiated
    with no arguments); anything else is a registered model or rule name, with
    an optional ``:column`` for a rule.
    """
    head, _, attribute = text.partition(":")
    if attribute and "." in head:
        import importlib

        found = getattr(importlib.import_module(head), attribute)
        strategy: StrategyLike = found() if isinstance(found, type) else found
        return strategy
    return text
