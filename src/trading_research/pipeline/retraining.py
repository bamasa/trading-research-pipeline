"""Wiring a retraining schedule to prepared data.

:mod:`trading_research.validation.retrain` knows how to walk a schedule and
nothing else — it is handed a callable that fits on one date range and scores
another. This module builds that callable out of the pipeline stages, which is
the only place the two need to meet.

The split is deliberate. Everything instrument-specific, model-specific and
cost-specific lives here; the search itself stays reusable. Pointing the sweep
at a different instrument is a different ``prepared_dir``; a different model is
a different ``model`` string.

Where the confidence threshold comes from
-----------------------------------------
Every strategy here needs a threshold on the model's score, and choosing it on
the period being scored is the most direct way to manufacture a result. In the
walk-forward that threshold comes from a separate validation block. A retraining
schedule has no such block — it has a training window and the days it acts on —
so the **tail of the training window** is reserved for it: the model fits on the
front, the threshold is swept on the tail, and the apply window sees neither.

That costs sample size, and it is the honest arrangement: everything used to
configure the trade decision comes from strictly before the first day it trades.

Speed
-----
A schedule with a 7-day window over 30 days reads the same day up to seven
times. Prepared days are read once and kept, which turns the sweep from
I/O-bound into fit-bound. The cache is per-evaluator, so a sweep over several
models each get their own — memory in exchange for not re-reading, and the
prepared frames are small once subsampled.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_research.backtest.costs import TakerCosts
from trading_research.backtest.evaluate import choose_confidence, decide
from trading_research.backtest.execution import ThinningRules, score, thin
from trading_research.pipeline.stages import StageError, add_label, build_model
from trading_research.validation.retrain import Evaluator, ScheduleResult


class DayCache:
    """Prepared days, read once and kept in memory."""

    def __init__(self, prepared_dir: Path | str) -> None:
        self.dir = Path(prepared_dir)
        self._days: dict[date, pd.DataFrame] = {}

    def available(self) -> list[date]:
        files = sorted(self.dir.glob("*.parquet"))
        if not files:
            raise StageError(f"no prepared data in {self.dir}; run prepare first")
        return [date.fromisoformat(f.stem) for f in files]

    def span(self, start: date, end: date) -> pd.DataFrame:
        wanted = [start + timedelta(days=i) for i in range((end - start).days + 1)]
        frames = []
        for day in wanted:
            if day not in self._days:
                path = self.dir / f"{day}.parquet"
                if not path.exists():
                    continue
                self._days[day] = pd.read_parquet(path)
            frames.append(self._days[day])
        if not frames:
            raise StageError(f"no prepared days between {start} and {end}")
        return pd.concat(frames, ignore_index=True)


def prepared_evaluator(
    prepared_dir: Path | str,
    features: Sequence[str],
    *,
    threshold_bp: float,
    hold_periods: int,
    costs: TakerCosts,
    model: str = "logistic",
    params: dict[str, Any] | None = None,
    calibrate: str | None = None,
    cooldown_periods: int = 0,
    inner_validation_fraction: float = 0.25,
    max_train_rows: int | None = None,
    minimum_train_rows: int = 2000,
) -> Evaluator:
    """Build an evaluator that fits on one span and trades the next.

    ``max_train_rows`` keeps the *tail* of the training block when set. Only
    needed for the network, whose fit time is linear in rows and whose training
    block here is far longer than it can use in the budget; the tabular models
    see everything.
    """
    columns = list(features)
    cache = DayCache(prepared_dir)

    def evaluate(train: tuple[date, date], apply: tuple[date, date]) -> dict[str, float]:
        frame = add_label(cache.span(*train), threshold_bp)
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise StageError(f"features missing from prepared data: {missing[:5]}")

        usable = frame[columns].notna().all(axis=1) & frame["label"].notna()
        block = frame.loc[usable]
        if len(block) < minimum_train_rows:
            raise StageError(f"only {len(block)} usable rows in {train[0]}..{train[1]}")

        # Front fits, tail chooses the threshold. Chronological, because
        # neighbouring rows 100 ms apart are near-duplicates and a random split
        # would put the same moment on both sides.
        cut = int(len(block) * (1 - inner_validation_fraction))
        fit = block.iloc[:cut]
        inner = block.iloc[cut:]
        if len(fit) < 500 or len(inner) < 500:
            raise StageError(f"training block too small to split: {len(fit)}/{len(inner)}")
        if max_train_rows is not None:
            fit = fit.tail(max_train_rows)
        if fit["label"].nunique() < 2:
            raise StageError("training block has one class only")

        estimator = build_model(model, **(params or {}))
        if calibrate:
            from trading_research.models.calibration import CalibratedModel

            estimator = CalibratedModel(estimator, method=calibrate)
        estimator.fit(fit[columns], fit["label"])

        confidence, _ = choose_confidence(
            estimator.predict_proba(inner[columns]),
            inner["forward_bp"],
            inner["spread_bp_now"],
            costs,
        )

        out_frame = cache.span(*apply)
        ok = out_frame[columns].notna().all(axis=1) & out_frame["forward_bp"].notna()
        acted = out_frame.loc[ok]
        if acted.empty:
            raise StageError(f"no usable rows in {apply[0]}..{apply[1]}")

        decision = decide(estimator.predict_proba(acted[columns]), min_confidence=confidence)
        trades = thin(
            decision,
            acted["forward_bp"].to_numpy(),
            acted["spread_bp_now"].to_numpy(),
            ThinningRules(hold_periods=hold_periods, cooldown_periods=cooldown_periods),
        )
        result = score(trades, costs)
        result["confidence"] = float(confidence)
        result["train_rows"] = float(len(fit))
        return result

    return evaluate


#: Quantities that add across windows. Everything else is averaged.
TOTALS = ("trades", "gross_bp", "cost_bp", "net_bp")

#: Per-trade figures, recomputed from the totals rather than averaged.
RATIOS = {
    "gross_per_trade_bp": ("gross_bp", "trades"),
    "net_per_trade_bp": ("net_bp", "trades"),
}


def trade_weighted(windows: list[dict[str, Any]]) -> dict[str, float]:
    """Combine window results by trade rather than by window.

    Found by reading a result that could not be true: a sweep reported +15 bp
    per trade on test while the total over the same windows was -37 bp. Both
    numbers were right. Averaging a per-trade figure across windows gives a
    window with one lucky trade the same weight as a window with thirty losing
    ones, and since the schedules being compared differ precisely in how much
    they trade, the objective was rewarding the ones that barely traded.

    Totals are summed, per-trade figures are recomputed from those totals, and
    the window-average is kept under a ``_window_mean`` suffix — the gap between
    the two is worth seeing rather than quietly resolving.
    """
    frame = pd.DataFrame(windows)
    numeric = frame.select_dtypes("number")
    out = {name: float(numeric[name].mean()) for name in numeric.columns}

    for name in TOTALS:
        if name in numeric.columns:
            out[name] = float(numeric[name].sum())

    trades = out.get("trades", 0.0)
    for name, (numerator, denominator) in RATIOS.items():
        if numerator in out and denominator in out and name in numeric.columns:
            out[f"{name}_window_mean"] = float(numeric[name].mean())
            out[name] = (
                float(out[numerator] / out[denominator]) if out[denominator] else float("nan")
            )

    if "hit_rate" in numeric.columns and "trades" in numeric.columns and trades:
        # Weighted by the trades each window actually took, for the same reason.
        out["hit_rate"] = float((numeric["hit_rate"] * numeric["trades"]).sum() / trades)
    return out


def summarise(table: pd.DataFrame, objective: str = "net_per_trade_bp") -> pd.DataFrame:
    """Order a validation table by the objective, keeping the columns worth reading."""
    keep = [
        c
        for c in (
            "schedule",
            "refits",
            "windows",
            "windows_skipped",
            "trades",
            "hit_rate",
            "gross_per_trade_bp",
            objective,
            "net_bp",
            "skipped",
        )
        if c in table.columns
    ]
    ordered = table[keep]
    if objective in ordered.columns:
        ordered = ordered.sort_values(objective, ascending=False, na_position="last")
    return ordered.reset_index(drop=True)


def positive_window_share(result: ScheduleResult) -> float:
    """Share of *trading* windows whose net result was positive.

    An aggregate figure hides whether a schedule was consistently slightly ahead
    or wrong most of the time and rescued by one window, which is the
    distinction that decides whether a schedule is worth anything.

    Windows that took no trades are excluded. Counting them drags the share
    towards zero and makes a selective schedule look worse than an indiscriminate
    one purely for having stayed out of the market — on the run that prompted
    this, ten of twenty-five windows never traded, and including them turned 47%
    into 28%.
    """
    trading = [r for r in result.per_window if "net_bp" in r and float(r.get("trades", 0.0)) > 0.0]
    if not trading:
        return float("nan")
    return float(np.mean([r["net_bp"] > 0 for r in trading]))
