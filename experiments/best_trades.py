"""Re-run the best configuration and keep every trade, for the scorecard."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from experiments._common import COSTS, RESULTS
from experiments.horizons import SHORT_FEATURES
from trading_research.backtest.evaluate import choose_confidence, decide
from trading_research.backtest.execution import ThinningRules, thin, trades_to_frame
from trading_research.models.base import clean
from trading_research.pipeline.stages import add_label, build_model
from trading_research.validation.splits import WalkForwardSpec, fold_masks, walk_forward

warnings.filterwarnings("ignore")
COST = 11.02
frame = pd.concat(
    [pd.read_parquet(p) for p in sorted(Path("artifacts/fine_BTCUSDT").glob("*.parquet"))],
    ignore_index=True,
)
seconds = float(frame["timestamp"].diff().dt.total_seconds().median())
step = max(1, round(120 / seconds))
mid = frame["mid"].to_numpy()
forward = np.full(len(mid), np.nan)
forward[:-step] = np.log(mid[step:] / mid[:-step]) * 1e4
frame = add_label(frame.assign(forward_bp=forward), COST)
cols = [c for c in SHORT_FEATURES if c in frame.columns]

spec = WalkForwardSpec(train_days=14, validation_days=7, test_days=7, step_days=1, n_folds=5)
days = sorted(set(frame["timestamp"].dt.date))[: spec.required_days]
frame = frame[frame["timestamp"].dt.date.isin(set(days))].reset_index(drop=True)

out = []
for fold in walk_forward(days, spec):
    m = fold_masks(fold, frame["timestamp"], purge=step)
    tx, ty = clean(frame.loc[m.train, cols], frame.loc[m.train, "label"])
    model = build_model("logistic").fit(tx, ty)
    va = frame.loc[m.validation]
    ok = va[cols].notna().all(axis=1) & va["forward_bp"].notna()
    conf, _ = choose_confidence(
        model.predict_proba(va.loc[ok, cols]),
        va.loc[ok, "forward_bp"],
        va.loc[ok, "spread_bp_now"],
        COSTS,
    )
    te = frame.loc[m.test]
    ok = te[cols].notna().all(axis=1) & te["forward_bp"].notna()
    s = te.loc[ok]
    dec = decide(model.predict_proba(s[cols]), min_confidence=conf)
    tr = thin(
        dec,
        s["forward_bp"].to_numpy(),
        s["spread_bp_now"].to_numpy(),
        ThinningRules(hold_periods=step, cooldown_periods=step),
    )
    f = trades_to_frame(tr)
    if f.empty:
        continue
    f["fold"] = fold.index
    f["gross_bp"] = f["direction"] * f["move_bp"]
    f["net_bp"] = f["gross_bp"] - np.asarray(COSTS.round_trip_bp(f["entry_spread_bp"]))
    out.append(f)
    print(f"fold {fold.index}: {len(f)} trades", flush=True)

all_trades = pd.concat(out, ignore_index=True)
RESULTS.mkdir(parents=True, exist_ok=True)
all_trades.to_csv(RESULTS / "best_trades_BTCUSDT.csv", index=False)
print(
    f"\n{len(all_trades)} trades, net/trade {all_trades.net_bp.mean():.2f}, dispersion {all_trades.net_bp.std():.2f}"
)
print(f"days covered: {len(days)}")
