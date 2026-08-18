"""Running several strategies and choosing between them every day.

The rest of this project picks one configuration and keeps it. This runs a
stable of strategies side by side, records what each did on each day, and then
walks forward choosing tomorrow's from the days already finished — including the
option of trading nothing.

The recorded results are what every strategy *would* have made each day, which
a backtest has and a live system would not. The choice is what a live system
could have made: it reads only days strictly before the one being traded.

    uv run python -m experiments.daily_allocation --symbols BTCUSDT
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from experiments._common import COSTS, RESULTS, emit, load, parser
from trading_research.backtest.evaluate import choose_confidence, decide
from trading_research.backtest.execution import ThinningRules, thin, trades_to_frame
from trading_research.models.base import clean
from trading_research.pipeline.stages import add_label, build_model, load_selected
from trading_research.validation.allocator import allocate, default_configs
from trading_research.validation.splits import WalkForwardSpec, fold_masks, walk_forward

#: The stable. A rule, a model, the composite that beat both, and the ensemble —
#: enough variety that they can plausibly take turns, which is the only
#: condition under which switching between them is worth anything.
STRATEGIES = ("order_flow", "logistic", "model_gated_by_rule", "breakout", "spread_capture")

RULE_COLUMNS = (
    "log_mid_ret20",
    "log_mid_ret50",
    "log_mid_vol50",
    "queue_imbalance",
    "spread_bp",
)
COST_BP = {"BTCUSDT": 11.02, "XRPUSDT": 12.68}
HOLD = 24


def daily_results(frame: pd.DataFrame, columns: list[str], strategy: str, spec) -> pd.DataFrame:
    """What one strategy made on each day of the test blocks."""
    rows = []
    for fold in walk_forward(sorted(set(frame["timestamp"].dt.date)), spec):
        masks = fold_masks(fold, frame["timestamp"], purge=HOLD)
        train_x, train_y = clean(frame.loc[masks.train, columns], frame.loc[masks.train, "label"])
        model = build_model(strategy).fit(train_x, train_y)

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
        block = test.loc[ok].reset_index(drop=True)
        decision = decide(model.predict_proba(block[columns]), min_confidence=confidence)
        trades = thin(
            decision,
            block["forward_bp"].to_numpy(),
            block["spread_bp_now"].to_numpy(),
            ThinningRules(hold_periods=HOLD, cooldown_periods=HOLD),
        )
        if not trades:
            continue

        taken = trades_to_frame(trades)
        taken["day"] = block["timestamp"].dt.date.to_numpy()[taken["entry_index"].to_numpy()]
        taken["net_bp"] = (taken["direction"] * taken["move_bp"]) - np.asarray(
            COSTS.round_trip_bp(taken["entry_spread_bp"])
        )
        # One fold per day would double-count: folds step a day at a time and
        # their test blocks overlap, so each day is kept once, from the first
        # fold that traded it.
        for day, group in taken.groupby("day"):
            rows.append(
                {
                    "day": day,
                    "fold": fold.index,
                    "strategy": strategy,
                    "trades": float(len(group)),
                    "net_bp": float(group["net_bp"].sum()),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return (
        out.sort_values(["day", "fold"]).drop_duplicates("day", keep="first").drop(columns="fold")
    )


def main() -> None:
    p = parser(__doc__ or "")
    p.add_argument("--features-prefix", default="feat10")
    args = p.parse_args()
    warnings.filterwarnings("ignore")

    spec = WalkForwardSpec(train_days=14, validation_days=7, test_days=7, step_days=1, n_folds=7)

    for symbol in args.symbols:
        frame = load(symbol, root=args.prepared_root)
        selected = load_selected(args.prepared_root / f"{args.features_prefix}_{symbol}")
        columns = [c for c in dict.fromkeys([*selected, *RULE_COLUMNS]) if c in frame.columns]
        frame = add_label(frame, COST_BP[symbol])
        days = sorted(set(frame["timestamp"].dt.date))[: spec.required_days]
        frame = frame[frame["timestamp"].dt.date.isin(set(days))].reset_index(drop=True)

        parts = []
        for strategy in STRATEGIES:
            result = daily_results(frame, columns, strategy, spec)
            parts.append(result)
            print(f"  {symbol} {strategy}: {len(result)} days", flush=True)
        daily = pd.concat(parts, ignore_index=True)
        daily["net_per_trade_bp"] = daily["net_bp"] / daily["trades"].replace(0, np.nan)
        daily.to_csv(RESULTS / f"daily_{symbol}.csv", index=False)

        # Each strategy on its own, over the same days, as the thing the
        # selector has to beat.
        alone = daily.groupby("strategy").agg(
            trades=("trades", "sum"), net_bp=("net_bp", "sum"), days=("day", "nunique")
        )
        alone["net_per_trade_bp"] = alone["net_bp"] / alone["trades"]
        alone["trades_per_day"] = alone["trades"] / alone["days"]
        print()
        emit(alone.reset_index(), f"daily_alone_{symbol}")

        rows = []
        for config in default_configs():
            try:
                result = allocate(daily, config)
            except Exception as exc:  # a config the span cannot support
                print(f"  {config.label}: {exc}", flush=True)
                continue
            rows.append(
                {
                    "config": config.label,
                    "switches": result.switches,
                    "days_aside": result.days_aside,
                    **{
                        k: result.metrics.get(k)
                        for k in (
                            "trades",
                            "trades_per_day",
                            "net_bp",
                            "net_per_trade_bp",
                            "net_bp_per_day",
                            "max_drawdown_bp",
                            "profit_factor",
                        )
                    },
                }
            )
        print()
        emit(pd.DataFrame(rows), f"daily_allocation_{symbol}")


if __name__ == "__main__":
    main()
