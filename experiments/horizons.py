"""Short horizons, where high-frequency trading actually lives.

Everything so far used a two-minute horizon, which for this kind of strategy is
a long time. This sweeps from ten seconds to five minutes on a finer grid —
3.3 seconds a row rather than 7.4 — so the short end is resolvable at all.

Both ends of a trade move together. Predicting ten seconds ahead and holding
for two minutes measures neither: the position spends most of its life on a
forecast that expired. So the holding period is tied to the horizon, and the
question the sweep asks is where the ratio of edge to cost is least bad.

§3 says this should not help — the information coefficient falls at the rate
volatility rises, so their product barely moves. That was measured from one
second to twenty minutes on the coarse grid, with features built for the long
end. This re-asks it with features built for the short one.

    uv run python -m experiments.horizons --symbols BTCUSDT
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from experiments._common import COSTS, RESULTS, emit, parser
from trading_research.backtest.evaluate import choose_confidence, decide
from trading_research.backtest.execution import ThinningRules, score, thin
from trading_research.models.base import clean
from trading_research.pipeline.stages import add_label, build_model
from trading_research.validation.splits import WalkForwardSpec, fold_masks, walk_forward

#: Horizons in seconds. The upper end is where the user's brief stops; the
#: lower end is about as short as a 3.3 s grid can resolve.
HORIZONS_S = (10, 20, 30, 60, 120, 300)

#: Features built for this end of the problem, plus the two that carried the
#: signal at the long end.
SHORT_FEATURES = (
    "queue_imbalance",
    "microprice_dev_bp",
    "quote_intensity_20",
    "imbalance_slope_10",
    "touch_persistence_20",
    "mid_reversal_10",
    "spread_pressure_20",
    "size_shock_10",
    "micro_drift_5",
    "realised_vol_20",
)

COST_BP = {"BTCUSDT": 11.02, "XRPUSDT": 12.84}


def main() -> None:
    p = parser(__doc__ or "")
    p.add_argument("--model", default="logistic")
    p.add_argument("--prepared-name", default="fine")
    args = p.parse_args()
    warnings.filterwarnings("ignore")

    rows = []
    for symbol in args.symbols:
        frame = pd.concat(
            [
                pd.read_parquet(path)
                for path in sorted(
                    (args.prepared_root / f"{args.prepared_name}_{symbol}").glob("*.parquet")
                )
            ],
            ignore_index=True,
        )
        cost = COST_BP[symbol]
        available = [c for c in SHORT_FEATURES if c in frame.columns]
        missing = [c for c in SHORT_FEATURES if c not in frame.columns]
        if missing:
            print(f"  missing from prepared data: {missing}", flush=True)

        seconds_per_row = float(frame["timestamp"].diff().dt.total_seconds().median())
        mid = frame["mid"].to_numpy()
        print(
            f"{symbol}: {len(frame):,} rows at {seconds_per_row:.1f} s, {len(available)} features",
            flush=True,
        )

        for seconds in HORIZONS_S:
            step = max(1, round(seconds / seconds_per_row))
            forward = np.full(len(mid), np.nan)
            forward[:-step] = np.log(mid[step:] / mid[:-step]) * 1e4
            block = frame.assign(forward_bp=forward)
            block = add_label(block, cost)

            # Hold for what was predicted, and wait as long again before
            # re-entering. A forecast held past its horizon is a position
            # running on an expired opinion.
            spec = WalkForwardSpec(
                train_days=14, validation_days=7, test_days=7, step_days=1, n_folds=5
            )
            days = sorted(set(block["timestamp"].dt.date))[: spec.required_days]
            usable = block[block["timestamp"].dt.date.isin(set(days))].reset_index(drop=True)

            for fold in walk_forward(days, spec):
                masks = fold_masks(fold, usable["timestamp"], purge=step)
                train_x, train_y = clean(
                    usable.loc[masks.train, available], usable.loc[masks.train, "label"]
                )
                if train_y.nunique() < 2:
                    continue
                model = build_model(args.model).fit(train_x, train_y)

                validation = usable.loc[masks.validation]
                ok = validation[available].notna().all(axis=1) & validation["forward_bp"].notna()
                confidence, _ = choose_confidence(
                    model.predict_proba(validation.loc[ok, available]),
                    validation.loc[ok, "forward_bp"],
                    validation.loc[ok, "spread_bp_now"],
                    COSTS,
                )

                test = usable.loc[masks.test]
                ok = test[available].notna().all(axis=1) & test["forward_bp"].notna()
                scored = test.loc[ok]
                decision = decide(model.predict_proba(scored[available]), min_confidence=confidence)
                trades = thin(
                    decision,
                    scored["forward_bp"].to_numpy(),
                    scored["spread_bp_now"].to_numpy(),
                    ThinningRules(hold_periods=step, cooldown_periods=step),
                )
                rows.append(
                    {
                        "symbol": symbol,
                        "model": args.model,
                        "horizon_s": seconds,
                        "hold_rows": step,
                        "fold": fold.index,
                        **score(trades, COSTS, days=scored["timestamp"].dt.date.nunique()),
                    }
                )
            print(f"  {seconds:>4} s (hold {step} rows) done", flush=True)

    per_fold = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model}_{'_'.join(args.symbols)}"
    per_fold.to_csv(RESULTS / f"horizons_{tag}_folds.csv", index=False)

    summary = per_fold.groupby(["symbol", "model", "horizon_s"]).agg(
        trades=("trades", "sum"),
        trades_per_day=("trades_per_day", "mean"),
        gross_per_trade_bp=("gross_per_trade_bp", "mean"),
        net_per_trade_bp=("net_per_trade_bp", "mean"),
        max_drawdown_bp=("max_drawdown_bp", "max"),
    )
    summary["folds_positive"] = per_fold.groupby(["symbol", "model", "horizon_s"])["net_bp"].apply(
        lambda s: int((s > 0).sum())
    )
    emit(summary.reset_index(), f"horizons_{tag}")


if __name__ == "__main__":
    main()
