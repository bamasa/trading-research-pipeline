"""Why the fitted model loses to one of its own inputs, and what fixes it.

The finding that prompted this: on held-out data the logistic model scores an
information coefficient of 0.011 against the two-minute move, while raw queue
imbalance — a single column, no fitting — scores several times that. A model
that loses to its own input is not short of capacity. Something else is wrong,
and there are only two candidates.

**Bias.** The model class cannot express the relationship. Adding trees fixes
this; averaging does not.

**Variance.** The model can express it but the estimate swings with which
stretch of history it was fitted on, so what it learned is largely the
particular fortnight. Averaging fixes this; a fancier model class makes it
worse.

:mod:`trading_research.evaluation.bias_variance` measures which, by refitting
each candidate on many different contiguous stretches and splitting the error
into the part that survives averaging and the part that does not. Contiguous
rather than randomly resampled, because neighbouring rows five seconds apart are
near-duplicates and a random bootstrap draws the same training set every time.

The ladder
----------
Eight candidates, arranged so each step answers one question:

1. ``imbalance`` — the raw feature, thresholded. The benchmark everything must
   beat, and the reason this study exists.
2. ``logistic`` — the linear model on eight features.
3. ``xgboost`` / ``lightgbm`` — two boosters that split differently.
4. ``random_forest`` / ``extra_trees`` — bagged trees, the second with random
   split thresholds as well as random features.
5. ``bagged_logistic`` — the linear model averaged over stretches of history.
   If variance is the problem, this is the cheapest possible fix and it should
   show.
6. ``voting`` — a linear model and a tree ensemble averaged, with their
   correlation reported: two members agreeing at 0.98 are one member.
7. ``stacked`` — weights learned on forward-chained out-of-fold predictions,
   so the second stage only ever sees predictions for rows the first stage did
   not fit.

Each is measured three ways: the bias-variance split, the information
coefficient on held-out data, and what it earns after costs. The three do not
have to agree, and where they disagree the last one is the answer.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from experiments._common import RESULTS, emit
from experiments.grand_search import (
    COSTS,
    MICRO,
    ROWS_PER_DAY,
    build_features,
    normalise,
    to_grid,
)
from trading_research.backtest.execution import ThinningRules, thin
from trading_research.evaluation.bias_variance import compare, decompose
from trading_research.models.base import CLASSES
from trading_research.models.stacking import Bagged, Stacked, Voting
from trading_research.pipeline.stages import build_model

HORIZON = 24
HOLD = 24


def score_of(proba: np.ndarray) -> np.ndarray:
    return proba[:, CLASSES.index(1)] - proba[:, CLASSES.index(-1)]


def candidates() -> dict[str, object]:
    """Each entry builds a fresh, unfitted model."""
    return {
        "logistic": lambda: build_model("logistic"),
        "xgboost": lambda: build_model("xgboost"),
        "lightgbm": lambda: build_model("lightgbm"),
        "random_forest": lambda: build_model("random_forest"),
        "extra_trees": lambda: build_model("extra_trees"),
        "bagged_logistic": lambda: Bagged(
            lambda: build_model("logistic"), members=8, block_fraction=0.5
        ),
        "voting": lambda: Voting(
            [build_model("logistic"), build_model("extra_trees"), build_model("lightgbm")]
        ),
        "stacked": lambda: Stacked(
            [
                lambda: build_model("logistic"),
                lambda: build_model("extra_trees"),
                lambda: build_model("lightgbm"),
            ],
            lambda: build_model("logistic"),
            folds=4,
            purge=HORIZON,
        ),
    }


def trade(score: np.ndarray, forward: np.ndarray, spread: np.ndarray, rate: float) -> dict:
    ok = np.isfinite(score) & np.isfinite(forward)
    if ok.sum() < 500:
        return {"trades": 0.0, "net_per_trade_bp": float("nan")}
    days = len(score) / ROWS_PER_DAY
    share = min(0.999, max(1, int(rate * days)) / int(ok.sum()))
    threshold = float(np.quantile(np.abs(score[ok]), 1 - share))
    decision = np.zeros(len(score), dtype=int)
    strong = ok & (np.abs(score) >= threshold)
    decision[strong] = np.sign(score[strong]).astype(int)
    trades = thin(
        decision,
        np.nan_to_num(forward),
        spread,
        ThinningRules(hold_periods=HOLD, cooldown_periods=HOLD),
    )
    if not trades:
        return {"trades": 0.0, "net_per_trade_bp": float("nan")}
    net = np.array(
        [t.direction * t.move_bp - float(COSTS.round_trip_bp(t.entry_spread_bp)) for t in trades]
    )
    gross = np.array([t.direction * t.move_bp for t in trades])
    return {
        "trades": float(len(net)),
        "gross_per_trade_bp": float(gross.mean()),
        "net_per_trade_bp": float(net.mean()),
        "net_bp": float(net.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BICOUSDT")
    parser.add_argument("--book", type=Path, default=Path("data/book"))
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--fits", type=int, default=8)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    files = sorted((args.book / args.symbol).glob("*.parquet"))
    book = to_grid(pd.concat([pd.read_parquet(p) for p in files], ignore_index=True))
    mid = ((book["bid_price_0"] + book["ask_price_0"]) / 2).to_numpy()
    spread = ((book["ask_price_0"] - book["bid_price_0"]) / mid * 1e4).to_numpy()
    frame = normalise(build_features(book), window=4000).astype("float32")

    forward = np.full(len(mid), np.nan)
    forward[:-HOLD] = (mid[HOLD:] / mid[:-HOLD] - 1.0) * 1e4
    label = pd.Series(np.sign(np.nan_to_num(forward)).astype(int))

    columns = list(MICRO)
    usable = frame[columns].notna().all(axis=1).to_numpy() & np.isfinite(forward)
    cut = int(len(book) * 0.65)
    train = np.flatnonzero(usable[: cut - HORIZON])
    test = np.flatnonzero(usable[cut:]) + cut
    print(f"{args.symbol}: {len(train):,} training rows, {len(test):,} held-out rows")

    train_x = frame.iloc[train][columns]
    train_y = label.iloc[train]
    test_x = frame.iloc[test][columns]
    test_forward = forward[test]
    test_spread = spread[test]

    # The benchmark: the single strongest raw feature, unfitted.
    raw = frame["queue_imbalance"].to_numpy()[test]
    rows = [
        {
            "model": "imbalance (raw feature, no fitting)",
            "held_out_ic": float(np.corrcoef(raw, test_forward)[0, 1]),
            **trade(raw, test_forward, test_spread, args.rate),
        }
    ]

    decompositions = []
    for name, build in candidates().items():
        print(f"  {name} ...", flush=True)

        def fit_predict(x, y, target, build=build):
            return score_of(build().fit(x, y).predict_proba(target))

        try:
            decompositions.append(
                decompose(
                    fit_predict,
                    train_x,
                    train_y,
                    test_x,
                    pd.Series(test_forward),
                    name=name,
                    fits=args.fits,
                    block_fraction=0.5,
                )
            )
        except Exception as exc:
            print(f"    decomposition failed: {type(exc).__name__}: {exc}")

        try:
            score = fit_predict(train_x, train_y, test_x)
        except Exception as exc:
            print(f"    fit failed: {type(exc).__name__}: {exc}")
            continue
        entry = {
            "model": name,
            "held_out_ic": float(np.corrcoef(score, test_forward)[0, 1]),
            **trade(score, test_forward, test_spread, args.rate),
        }
        rows.append(entry)

    RESULTS.mkdir(parents=True, exist_ok=True)
    performance = pd.DataFrame(rows).sort_values("held_out_ic", ascending=False)
    performance.insert(0, "symbol", args.symbol)
    emit(performance, f"ensemble_performance_{args.symbol}")

    if decompositions:
        table = compare(decompositions)
        table.insert(0, "symbol", args.symbol)
        emit(table, f"ensemble_bias_variance_{args.symbol}")


if __name__ == "__main__":
    main()
