"""§6 — why "trade only the best signals" looks profitable.

The most instructive experiment here, and the one worth reading before any other
number in the project.

Trading only the most confident signals appears to work: push the cutoff high
enough and the surviving trades are profitable. The catch is *when* the cutoff
was decided. Choosing it after seeing how each cutoff scored on test is
selection, not edge, and it produces a rising curve on any data at all — noise
included.

This runs both. The same model, the same folds, the same trades; the only
difference is whether the cutoff came from the test period or from validation.
The gap between the two curves is the size of the mistake.

Selectivity is expressed as a number of trades rather than a quantile, and the
trades are pooled across folds before ranking. That matters: a quantile applied
inside each fold separately answers a different and much duller question, since
a fold contributes its own best trades whether or not they were good in absolute
terms. Ranking the pooled set is what someone reporting "my best 30 trades made
money" would actually be doing.

    uv run python -m experiments.selection_leak --symbols BTCUSDT
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from experiments._common import COSTS, emit, load_book_features, parser, round_trip
from experiments.walk_forward import FEATURES, HOLD, HORIZON
from trading_research.backtest.execution import ThinningRules, thin
from trading_research.labels.directional import BUY, SELL
from trading_research.labels.directional import HOLD as HOLD_CLASS
from trading_research.models.base import CLASSES, clean
from trading_research.pipeline.stages import add_label, build_model
from trading_research.validation.splits import WalkForwardSpec, fold_masks, walk_forward

#: How many trades to keep, out of every trade the strategy took across all
#: folds. The smallest of these is about three trades a month.
TOP_N = (1000, 300, 100, 30)

#: A permissive base threshold: everything above it is a candidate, and the
#: selection happens afterwards. Set low so the ranking has something to rank.
BASE_CONFIDENCE = 0.34


def trades_with_confidence(
    proba: np.ndarray,
    forward: np.ndarray,
    spread: np.ndarray,
) -> pd.DataFrame:
    """Take every candidate trade and record how confident the model was."""
    directional = np.maximum(proba[:, CLASSES.index(BUY)], proba[:, CLASSES.index(SELL)])
    decision = np.where(
        directional < BASE_CONFIDENCE,
        HOLD_CLASS,
        np.where(proba[:, CLASSES.index(BUY)] >= proba[:, CLASSES.index(SELL)], BUY, SELL),
    )
    taken = thin(decision, forward, spread, ThinningRules(hold_periods=HOLD, cooldown_periods=HOLD))
    if not taken:
        return pd.DataFrame(columns=["confidence", "net_bp"])

    costs = COSTS.round_trip_bp(np.array([t.entry_spread_bp for t in taken]))
    return pd.DataFrame(
        {
            "confidence": [directional[t.entry_index] for t in taken],
            "net_bp": [t.direction * t.move_bp for t in taken] - np.asarray(costs),
        }
    )


def main() -> None:
    p = parser(__doc__ or "")
    p.add_argument("--model", default="logistic")
    args = p.parse_args()
    warnings.filterwarnings("ignore")

    spec = WalkForwardSpec(train_days=14, validation_days=7, test_days=7, step_days=1, n_folds=7)
    rows = []

    for symbol in args.symbols:
        frame = load_book_features(symbol, FEATURES, days=args.days)
        mid = frame["mid"].to_numpy()
        forward_all = np.full(len(mid), np.nan)
        forward_all[:-HORIZON] = np.log(mid[HORIZON:] / mid[:-HORIZON]) * 1e4
        frame["forward_bp"] = forward_all
        frame["spread_bp_now"] = frame["spread_bp"]

        cost = round_trip(symbol, frame)
        frame = add_label(frame, cost)
        days = sorted(set(frame["timestamp"].dt.date))[: spec.required_days]
        frame = frame[frame["timestamp"].dt.date.isin(set(days))].reset_index(drop=True)

        test_trades, validation_trades = [], []
        for fold in walk_forward(days, spec):
            masks = fold_masks(fold, frame["timestamp"], purge=HOLD)
            train_x, train_y = clean(
                frame.loc[masks.train, list(FEATURES)], frame.loc[masks.train, "label"]
            )
            model = build_model(args.model).fit(train_x, train_y)

            for mask, sink in (
                (masks.validation, validation_trades),
                (masks.test, test_trades),
            ):
                block = frame.loc[mask]
                ok = block[list(FEATURES)].notna().all(axis=1)
                sink.append(
                    trades_with_confidence(
                        model.predict_proba(block.loc[ok, list(FEATURES)]),
                        block.loc[ok, "forward_bp"].to_numpy(),
                        block.loc[ok, "spread_bp_now"].to_numpy(),
                    )
                )
            print(f"  {symbol} fold {fold.index} done", flush=True)

        on_test = pd.concat(test_trades, ignore_index=True)
        on_validation = pd.concat(validation_trades, ignore_index=True)

        rows.append(
            {
                "symbol": symbol,
                "selectivity": "all signals",
                "chosen_on_test_bp": float(on_test["net_bp"].mean()),
                "chosen_on_validation_bp": float(on_test["net_bp"].mean()),
                "trades_kept": len(on_test),
            }
        )
        for n in TOP_N:
            if n > len(on_test):
                continue
            # Chosen on test: rank the test trades and keep the best n. The
            # cutoff could only have been known after the period was over.
            best_on_test = on_test.nlargest(n, "confidence")["net_bp"].mean()

            # Chosen on validation: find the confidence that would have kept n
            # trades on validation, and apply that number to test unchanged.
            if n > len(on_validation):
                continue
            threshold = float(on_validation.nlargest(n, "confidence")["confidence"].min())
            kept = on_test[on_test["confidence"] >= threshold]

            rows.append(
                {
                    "symbol": symbol,
                    "selectivity": f"top {n:,}",
                    "chosen_on_test_bp": float(best_on_test),
                    "chosen_on_validation_bp": float(kept["net_bp"].mean())
                    if len(kept)
                    else np.nan,
                    "trades_kept": len(kept),
                }
            )

    table = pd.DataFrame(rows)
    table["difference_bp"] = table["chosen_on_test_bp"] - table["chosen_on_validation_bp"]
    emit(table, "selection_leak")

    print()
    emit(raw_signal_strength(args), "signal_strength_outcomes")


def raw_signal_strength(args) -> pd.DataFrame:
    """Does acting only on the strongest raw signals pay better?

    The second check in the section, and a blunter one: no model, no threshold
    fitted anywhere, just the moments where queue imbalance is most extreme and
    what happened next. If selectivity created edge rather than merely selecting
    it, this would rise as the cut tightens.
    """
    rows = []
    for symbol in args.symbols:
        frame = load_book_features(symbol, ("queue_imbalance",), days=args.days)
        mid = frame["mid"].to_numpy()
        forward = np.full(len(mid), np.nan)
        forward[:-HORIZON] = np.log(mid[HORIZON:] / mid[:-HORIZON]) * 1e4

        signal = frame["queue_imbalance"].to_numpy()
        usable = ~np.isnan(forward) & ~np.isnan(signal)
        signal, forward = signal[usable], forward[usable]
        # Signed: a strong imbalance predicts a direction, so the outcome worth
        # measuring is the move in the direction it pointed.
        outcome = np.sign(signal) * forward

        for share in (0.10, 0.01, 0.001, 0.0001):
            cut = np.quantile(np.abs(signal), 1.0 - share)
            strongest = np.abs(signal) >= cut
            rows.append(
                {
                    "symbol": symbol,
                    "top_share": share,
                    "moments": int(strongest.sum()),
                    "mean_outcome_bp": float(outcome[strongest].mean()),
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
