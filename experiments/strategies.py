"""Rule-based strategies against learned models, on identical machinery.

The question this answers is the one a repository full of machine learning
should answer first: does the learning do anything a decades-old rule does not?

Every strategy here goes through the same walk-forward, the same confidence
threshold swept on validation, the same thinning and the same cost model. The
rules fit nothing beyond a normalisation scale and never see the label.

    uv run python -m experiments.strategies --symbols BTCUSDT --model order_flow
"""

from __future__ import annotations

import warnings

import pandas as pd

from experiments._common import COSTS, RESULTS, emit, load, parser
from trading_research.backtest.evaluate import choose_confidence, decide
from trading_research.backtest.execution import ThinningRules, score, thin
from trading_research.models.base import clean
from trading_research.pipeline.stages import add_label, build_model, load_selected
from trading_research.validation.splits import WalkForwardSpec, fold_masks, walk_forward

#: Columns the rules read, on top of whatever the model features are. Passed to
#: every strategy so the comparison uses one frame.
RULE_COLUMNS = (
    "log_mid_ret20",
    "log_mid_ret50",
    "log_mid_vol50",
    "queue_imbalance",
    "spread_bp",
)

COST_BP = {"BTCUSDT": 11.02, "XRPUSDT": 12.68, "XMRUSDT": 11.81}
HOLD = 24


def main() -> None:
    p = parser(__doc__ or "")
    p.add_argument("--model", default="order_flow")
    p.add_argument("--features-prefix", default="feat10")
    p.add_argument("--cooldown", type=int, default=24)
    args = p.parse_args()
    warnings.filterwarnings("ignore")

    spec = WalkForwardSpec(train_days=14, validation_days=7, test_days=7, step_days=1, n_folds=7)
    rows = []

    for symbol in args.symbols:
        frame = load(symbol, root=args.prepared_root)
        selected = load_selected(args.prepared_root / f"{args.features_prefix}_{symbol}")
        # A rule reads its own column and a learned model reads the selected
        # set; handing both the union keeps one frame and one code path.
        columns = list(dict.fromkeys([*selected, *RULE_COLUMNS]))
        columns = [c for c in columns if c in frame.columns]

        cost = COST_BP[symbol]
        frame = add_label(frame, cost)
        days = sorted(set(frame["timestamp"].dt.date))[: spec.required_days]
        frame = frame[frame["timestamp"].dt.date.isin(set(days))].reset_index(drop=True)
        print(f"{symbol} {args.model}: {len(frame):,} rows, {len(days)} days", flush=True)

        for fold in walk_forward(days, spec):
            masks = fold_masks(fold, frame["timestamp"], purge=HOLD)
            train_x, train_y = clean(
                frame.loc[masks.train, columns], frame.loc[masks.train, "label"]
            )
            model = build_model(args.model).fit(train_x, train_y)

            validation = frame.loc[masks.validation]
            ok = validation[columns].notna().all(axis=1)
            confidence, _ = choose_confidence(
                model.predict_proba(validation.loc[ok, columns]),
                validation.loc[ok, "forward_bp"],
                validation.loc[ok, "spread_bp_now"],
                COSTS,
            )

            test = frame.loc[masks.test]
            ok = test[columns].notna().all(axis=1)
            decision = decide(model.predict_proba(test.loc[ok, columns]), min_confidence=confidence)
            trades = thin(
                decision,
                test.loc[ok, "forward_bp"].to_numpy(),
                test.loc[ok, "spread_bp_now"].to_numpy(),
                ThinningRules(hold_periods=HOLD, cooldown_periods=args.cooldown),
            )
            rows.append(
                {
                    "symbol": symbol,
                    "strategy": args.model,
                    "fold": fold.index,
                    "confidence": confidence,
                    **score(trades, COSTS),
                }
            )
            print(f"  fold {fold.index} done", flush=True)

    per_fold = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    per_fold.to_csv(
        RESULTS / f"strategies_{args.model}_{'_'.join(args.symbols)}_folds.csv", index=False
    )

    summary = per_fold.groupby(["symbol", "strategy"]).agg(
        trades=("trades", "sum"),
        hit_rate=("hit_rate", "mean"),
        gross_per_trade_bp=("gross_per_trade_bp", "mean"),
        net_per_trade_bp=("net_per_trade_bp", "mean"),
        net_bp=("net_bp", "sum"),
    )
    summary["folds_positive"] = per_fold.groupby(["symbol", "strategy"])["net_bp"].apply(
        lambda s: int((s > 0).sum())
    )
    emit(summary.reset_index(), f"strategies_{args.model}_{'_'.join(args.symbols)}")


if __name__ == "__main__":
    main()
