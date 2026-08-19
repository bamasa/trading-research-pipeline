"""Trading less, on purpose: what happens at a chosen number of trades a day.

Selectivity has been set indirectly everywhere else — a threshold swept to
maximise something, with the trade count falling out as a consequence. This
sets it directly. Pick a target rate, find the confidence threshold on
validation that produces it, apply that threshold to test, and see what the
surviving trades are worth.

Two reasons the direct version is worth running even though §6 covers the
indirect one. It answers the question a desk actually asks — "what if we only
took the best fifty a day" — in the units the question is asked in. And it
separates two things the sweep conflates: whether a strategy has an edge, and
whether its confidence ranks trades usefully. A strategy whose per-trade result
improves as the rate falls is ranking well even if it never reaches profit;
one whose result is flat is not ranking at all, and its threshold is arbitrary.

    uv run python -m experiments.trade_rate --symbols BTCUSDT
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from experiments._common import COSTS, RESULTS, emit, load, parser
from trading_research.backtest.evaluate import decide
from trading_research.backtest.execution import ThinningRules, score, thin
from trading_research.models.base import CLASSES, clean
from trading_research.pipeline.stages import add_label, build_model, load_selected
from trading_research.validation.splits import WalkForwardSpec, fold_masks, walk_forward

#: Trades a day to aim for. The low end is where a desk would start; the high
#: end is roughly what the unrestricted strategies do.
TARGETS = (5, 10, 30, 50, 100, 250)

STRATEGIES = ("order_flow", "model_gated_by_rule", "logistic", "breakout")

RULE_COLUMNS = (
    "log_mid_ret20",
    "log_mid_ret50",
    "log_mid_vol50",
    "queue_imbalance",
    "spread_bp",
)
COST_BP = {"BTCUSDT": 11.02, "XRPUSDT": 12.68, "XMRUSDT": 11.81}
HOLD = 24


def threshold_for_rate(
    proba: np.ndarray,
    days: float,
    target_per_day: float,
    *,
    hold_periods: int,
    cooldown: int,
) -> float:
    """Confidence that yields roughly ``target_per_day`` trades on this block.

    Found by bisection on the realised trade count rather than by a quantile of
    the scores, because thinning breaks the relationship between the two: one
    position at a time means raising the threshold removes far fewer trades than
    it removes signals.
    """
    directional = np.maximum(proba[:, CLASSES.index(1)], proba[:, CLASSES.index(-1)])
    wanted = target_per_day * days

    low, high = float(np.min(directional)), float(np.max(directional))
    best = high
    for _ in range(24):
        middle = (low + high) / 2
        decision = np.where(
            directional >= middle,
            np.where(proba[:, CLASSES.index(1)] >= proba[:, CLASSES.index(-1)], 1, -1),
            0,
        )
        # Count entries the same way the backtest will, so the target is in the
        # units the result is reported in.
        taken = thin(
            decision,
            np.zeros(len(decision)),
            np.zeros(len(decision)),
            ThinningRules(hold_periods=hold_periods, cooldown_periods=cooldown),
        )
        if len(taken) > wanted:
            low = middle
        else:
            high = middle
            best = middle
    return best


def main() -> None:
    p = parser(__doc__ or "")
    p.add_argument("--features-prefix", default="feat10")
    p.add_argument("--cooldown", type=int, default=HOLD)
    args = p.parse_args()
    warnings.filterwarnings("ignore")

    spec = WalkForwardSpec(train_days=14, validation_days=7, test_days=7, step_days=1, n_folds=7)
    rows = []

    for symbol in args.symbols:
        frame = load(symbol, root=args.prepared_root)
        selected = load_selected(args.prepared_root / f"{args.features_prefix}_{symbol}")
        columns = [c for c in dict.fromkeys([*selected, *RULE_COLUMNS]) if c in frame.columns]
        frame = add_label(frame, COST_BP[symbol])
        days = sorted(set(frame["timestamp"].dt.date))[: spec.required_days]
        frame = frame[frame["timestamp"].dt.date.isin(set(days))].reset_index(drop=True)

        for strategy in STRATEGIES:
            for fold in walk_forward(days, spec):
                masks = fold_masks(fold, frame["timestamp"], purge=HOLD)
                train_x, train_y = clean(
                    frame.loc[masks.train, columns], frame.loc[masks.train, "label"]
                )
                model = build_model(strategy).fit(train_x, train_y)

                validation = frame.loc[masks.validation]
                ok = validation[columns].notna().all(axis=1)
                validation_proba = model.predict_proba(validation.loc[ok, columns])
                validation_days = validation.loc[ok, "timestamp"].dt.date.nunique()

                test = frame.loc[masks.test]
                ok = test[columns].notna().all(axis=1)
                block = test.loc[ok]
                test_proba = model.predict_proba(block[columns])
                test_days = block["timestamp"].dt.date.nunique()

                for target in TARGETS:
                    # The threshold is found on validation and applied to test
                    # unchanged, so the realised test rate will not match the
                    # target exactly — that mismatch is itself informative.
                    threshold = threshold_for_rate(
                        validation_proba,
                        validation_days,
                        target,
                        hold_periods=HOLD,
                        cooldown=args.cooldown,
                    )
                    decision = decide(test_proba, min_confidence=threshold)
                    trades = thin(
                        decision,
                        block["forward_bp"].to_numpy(),
                        block["spread_bp_now"].to_numpy(),
                        ThinningRules(hold_periods=HOLD, cooldown_periods=args.cooldown),
                    )
                    rows.append(
                        {
                            "symbol": symbol,
                            "strategy": strategy,
                            "target_per_day": target,
                            "fold": fold.index,
                            "threshold": threshold,
                            **score(trades, COSTS, days=test_days),
                        }
                    )
            print(f"  {symbol} {strategy} done", flush=True)

    per_fold = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    per_fold.to_csv(RESULTS / f"trade_rate_{'_'.join(args.symbols)}_folds.csv", index=False)

    summary = per_fold.groupby(["symbol", "strategy", "target_per_day"]).agg(
        trades=("trades", "sum"),
        actual_per_day=("trades_per_day", "mean"),
        gross_per_trade_bp=("gross_per_trade_bp", "mean"),
        net_per_trade_bp=("net_per_trade_bp", "mean"),
        net_bp=("net_bp", "sum"),
        max_drawdown_bp=("max_drawdown_bp", "max"),
    )
    summary["folds_positive"] = per_fold.groupby(["symbol", "strategy", "target_per_day"])[
        "net_bp"
    ].apply(lambda s: int((s > 0).sum()))
    emit(summary.reset_index(), f"trade_rate_{'_'.join(args.symbols)}")


if __name__ == "__main__":
    main()
