"""Does asking a different question get a better answer?

Four targets, each paired with the models that can be fitted on it, all run
through the same walk-forward, threshold sweep, thinning and cost model. The
only thing that varies is what the model was asked to predict.

    uv run python -m experiments.targets --symbols BTCUSDT --target net_pnl
"""

from __future__ import annotations

import warnings

import pandas as pd

from experiments._common import COSTS, RESULTS, emit, load, parser
from trading_research.backtest.evaluate import choose_confidence, decide
from trading_research.backtest.execution import ThinningRules, score, thin
from trading_research.labels.targets import TARGETS, build_target
from trading_research.models.base import clean
from trading_research.pipeline.stages import REGRESSION_MODELS, build_model, load_selected
from trading_research.validation.splits import WalkForwardSpec, fold_masks, walk_forward

COST_BP = {"BTCUSDT": 11.02, "XRPUSDT": 12.84}
HOLD = 24
RULE_COLUMNS = ("log_mid_ret20", "log_mid_ret50", "log_mid_vol50", "queue_imbalance", "spread_bp")


def main() -> None:
    p = parser(__doc__ or "")
    p.add_argument("--target", default="direction", choices=sorted(TARGETS))
    p.add_argument("--model", default=None, help="Defaults to the natural pair for the target.")
    p.add_argument("--features-prefix", default="feat10")
    args = p.parse_args()
    warnings.filterwarnings("ignore")

    spec = WalkForwardSpec(train_days=14, validation_days=7, test_days=7, step_days=1, n_folds=7)
    rows = []

    for symbol in args.symbols:
        cost = COST_BP[symbol]
        frame = load(symbol, root=args.prepared_root)
        selected = load_selected(args.prepared_root / f"{args.features_prefix}_{symbol}")
        columns = [c for c in dict.fromkeys([*selected, *RULE_COLUMNS]) if c in frame.columns]

        days = sorted(set(frame["timestamp"].dt.date))[: spec.required_days]
        frame = frame[frame["timestamp"].dt.date.isin(set(days))].reset_index(drop=True)
        target, target_spec = build_target(args.target, frame, cost_bp=cost, horizon=HOLD)
        frame = frame.assign(label=target)

        models = (
            [args.model]
            if args.model
            else (
                ["ridge", "xgboost_regressor"]
                if target_spec.kind == "regression"
                else ["logistic", "xgboost"]
            )
        )
        for model_name in models:
            if (model_name in REGRESSION_MODELS) != (target_spec.kind == "regression"):
                print(f"  skipping {model_name}: wrong kind for {args.target}", flush=True)
                continue

            for fold in walk_forward(days, spec):
                masks = fold_masks(fold, frame["timestamp"], purge=target_spec.horizon)
                train_x, train_y = clean(
                    frame.loc[masks.train, columns], frame.loc[masks.train, "label"]
                )
                params = {"scale_bp": cost} if model_name in REGRESSION_MODELS else {}
                model = build_model(model_name, **params).fit(train_x, train_y)

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
                block = test.loc[ok]
                decision = decide(model.predict_proba(block[columns]), min_confidence=confidence)
                trades = thin(
                    decision,
                    block["forward_bp"].to_numpy(),
                    block["spread_bp_now"].to_numpy(),
                    ThinningRules(hold_periods=HOLD, cooldown_periods=HOLD),
                )
                rows.append(
                    {
                        "symbol": symbol,
                        "target": args.target,
                        "kind": target_spec.kind,
                        "model": model_name,
                        "fold": fold.index,
                        **score(trades, COSTS, days=block["timestamp"].dt.date.nunique()),
                    }
                )
            print(f"  {symbol} {args.target} + {model_name} done", flush=True)

    per_fold = pd.DataFrame(rows)
    if per_fold.empty:
        print("nothing ran")
        return
    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = f"{args.target}_{'_'.join(args.symbols)}"
    per_fold.to_csv(RESULTS / f"targets_{tag}_folds.csv", index=False)

    summary = per_fold.groupby(["symbol", "target", "model"]).agg(
        trades=("trades", "sum"),
        trades_per_day=("trades_per_day", "mean"),
        gross_per_trade_bp=("gross_per_trade_bp", "mean"),
        net_per_trade_bp=("net_per_trade_bp", "mean"),
        net_bp=("net_bp", "sum"),
        max_drawdown_bp=("max_drawdown_bp", "max"),
    )
    summary["folds_positive"] = per_fold.groupby(["symbol", "target", "model"])["net_bp"].apply(
        lambda s: int((s > 0).sum())
    )
    emit(summary.reset_index(), f"targets_{tag}")


if __name__ == "__main__":
    main()
