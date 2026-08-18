"""§4 — the walk-forward comparison between models.

Fits one model per fold and scores it on test, with the confidence threshold
chosen on validation. The table this produces is the core negative result: zero
positive folds, and the simplest model ahead of the other two.

**One model per invocation.** XGBoost and PyTorch each bundle an OpenMP runtime
and deadlock when both are used in one process on macOS — the network fits 40k
rows in twenty seconds alone and never returns after a boosted tree has run.
The script therefore takes a single ``--model`` and is called repeatedly.

    for m in logistic xgboost tcn; do
        uv run python -m experiments.walk_forward --model $m --symbols BTCUSDT
    done
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from experiments._common import (
    COSTS,
    emit,
    load,
    load_book_features,
    parser,
    round_trip,
)
from trading_research.backtest.evaluate import choose_confidence, decide
from trading_research.backtest.execution import ThinningRules, score, thin
from trading_research.features.selection import FeatureSelector
from trading_research.models.base import clean
from trading_research.pipeline.stages import add_label, build_model, feature_columns
from trading_research.validation.splits import WalkForwardSpec, fold_masks, walk_forward

#: The hand-picked set. Ten features chosen for being readable rather than for
#: scoring well — the 194 generated columns are compared against them in §4.
#:
#: Read from the raw book rather than from prepared files: two of these are
#: registry features computed from bid and ask sizes, which prepare does not
#: carry forward. Using whichever subset happened to survive into the prepared
#: set would quietly answer a different question.
FEATURES = (
    "spread_bp",
    "queue_imbalance",
    "microprice_dev_bp",
    "log_depth_ratio",
    "log_total_depth",
    "ofi_20_norm",
    "mid_return_1_bp",
    "mid_return_20_bp",
    "realized_vol_50_bp",
    "spread_bp_ratio_50",
)

#: Label horizon in rows: 2 min on a 5 s grid.
HORIZON = 24

#: Rows to hold a position: the label horizon, 2 min on a 5 s grid.
HOLD = 24

#: None, one holding period, five.
COOLDOWNS = (0, 24, 120)

#: The network trains on the tail of each block rather than all of it. Apple's
#: MPS backend hangs on tensors this size, so it runs on CPU where a full block
#: costs about two minutes per epoch — an hour for the schedule, to answer a
#: question the cost arithmetic has already bounded. Stated because it is a
#: budget decision, not a methodological one.
TCN_TRAIN_ROWS = 60_000


def main() -> None:
    p = parser(__doc__ or "")
    p.add_argument("--model", default="logistic")
    p.add_argument(
        "--features",
        choices=("hand", "wide"),
        default="hand",
        help="'hand' reads ten registry features from the raw book; "
        "'wide' reads the 194 generated columns and selects inside each fold.",
    )
    p.add_argument("--max-features", type=int, default=40)
    args = p.parse_args()
    warnings.filterwarnings("ignore")

    spec = WalkForwardSpec(train_days=14, validation_days=7, test_days=7, step_days=1, n_folds=7)
    rows = []

    for symbol in args.symbols:
        if args.features == "wide":
            # The generated set already carries forward_bp and spread_bp_now,
            # and selection happens per fold below.
            frame = load(symbol, root=args.prepared_root, days=args.days)
            columns = feature_columns(frame)
        else:
            frame = load_book_features(symbol, FEATURES, days=args.days)
            columns = list(FEATURES)

            mid = frame["mid"].to_numpy()
            forward = np.full(len(mid), np.nan)
            forward[:-HORIZON] = np.log(mid[HORIZON:] / mid[:-HORIZON]) * 1e4
            frame["forward_bp"] = forward
            frame["spread_bp_now"] = frame["spread_bp"]

        cost = round_trip(symbol, frame)
        frame = add_label(frame, cost)

        days = sorted(set(frame["timestamp"].dt.date))[: spec.required_days]
        frame = frame[frame["timestamp"].dt.date.isin(set(days))].reset_index(drop=True)
        print(f"{symbol}: {len(frame):,} rows, {len(days)} days, cost {cost:.2f} bp", flush=True)

        for fold in walk_forward(days, spec):
            masks = fold_masks(fold, frame["timestamp"], purge=HOLD)

            fold_columns = columns
            if args.features == "wide":
                # Fitted inside the training block only. Selecting over the
                # whole sample is among the most effective ways to manufacture
                # an edge: with two hundred candidates, some will look
                # predictive on test by chance, and choosing them for that is
                # choosing them for their test performance.
                block = frame.loc[masks.train]
                usable = block[columns].notna().all(axis=1) & block["label"].notna()
                selector = FeatureSelector(max_features=args.max_features, score_target="ic")
                selector.fit(
                    block.loc[usable, columns],
                    block.loc[usable, "label"],
                    block.loc[usable, "forward_bp"],
                )
                fold_columns = selector.selected_

            train_x, train_y = clean(
                frame.loc[masks.train, fold_columns], frame.loc[masks.train, "label"]
            )
            if args.model == "tcn":
                train_x, train_y = train_x.tail(TCN_TRAIN_ROWS), train_y.tail(TCN_TRAIN_ROWS)

            model = build_model(args.model).fit(train_x, train_y)

            # The threshold comes from validation and is applied unchanged to
            # test. Choosing it on test is the most common way a short-horizon
            # result is overstated; §6 shows what it does to this one.
            validation = frame.loc[masks.validation]
            ok = validation[fold_columns].notna().all(axis=1)
            confidence, _ = choose_confidence(
                model.predict_proba(validation.loc[ok, fold_columns]),
                validation.loc[ok, "forward_bp"],
                validation.loc[ok, "spread_bp_now"],
                COSTS,
            )

            test = frame.loc[masks.test]
            ok = test[fold_columns].notna().all(axis=1)
            decision = decide(
                model.predict_proba(test.loc[ok, fold_columns]), min_confidence=confidence
            )
            forward = test.loc[ok, "forward_bp"].to_numpy()
            spread = test.loc[ok, "spread_bp_now"].to_numpy()

            for cooldown in COOLDOWNS:
                trades = thin(
                    decision,
                    forward,
                    spread,
                    ThinningRules(hold_periods=HOLD, cooldown_periods=cooldown),
                )
                rows.append(
                    {
                        "symbol": symbol,
                        "model": args.model,
                        "features": args.features,
                        "cooldown": cooldown,
                        "fold": fold.index,
                        "confidence": confidence,
                        **score(trades, COSTS),
                    }
                )
            print(f"  fold {fold.index} done", flush=True)

    per_fold = pd.DataFrame(rows)
    tag = args.model if args.features == "hand" else f"{args.model}_wide"
    emit(per_fold, f"walk_forward_{tag}_folds")

    group = ["symbol", "model", "features", "cooldown"]
    summary = per_fold.groupby(group).agg(
        trades=("trades", "sum"),
        trades_per_fold=("trades", "mean"),
        hit_rate=("hit_rate", "mean"),
        gross_per_trade_bp=("gross_per_trade_bp", "mean"),
        net_per_trade_bp=("net_per_trade_bp", "mean"),
        net_bp=("net_bp", "sum"),
    )
    summary["folds_positive"] = per_fold.groupby(group)["net_bp"].apply(
        lambda s: int((s > 0).sum())
    )
    print()
    emit(summary.reset_index(), f"walk_forward_{tag}")


if __name__ == "__main__":
    main()
