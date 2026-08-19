"""One model on many instruments, against one model per instrument.

§17 said the sample is what limits every conclusion here: thirty-four days of
one instrument is one realisation of one market. Normalised features can be
stacked, and stacking is the only thing available that multiplies the data by
more than a little.

The comparison is deliberately narrow. Same features, same model, same folds,
same costs; the only difference is whether the training block is one instrument
or several. If pooling helps, the per-instrument result on each member should
improve; if the instruments have nothing in common, pooling dilutes each with
the others' noise and every member gets worse.

    uv run python -m experiments.pooled --symbols BTCUSDT XRPUSDT XMRUSDT
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from experiments._common import COSTS, RESULTS, emit, parser
from trading_research.backtest.evaluate import choose_confidence, decide
from trading_research.backtest.execution import ThinningRules, score, thin
from trading_research.features.normalise import RollingNormaliser, coverage, normalise_pooled
from trading_research.models.base import clean
from trading_research.pipeline.stages import add_label, build_model
from trading_research.validation.splits import WalkForwardSpec, fold_masks, walk_forward

FEATURES = (
    "queue_imbalance",
    "microprice_dev_bp",
    "spread_bp",
    "log_mid_ret20",
    "log_mid_ret50",
    "log_mid_vol50",
    "quote_intensity_20",
    "imbalance_slope_10",
    "micro_drift_5",
    "realised_vol_20",
)

COST_BP = {"BTCUSDT": 11.02, "XRPUSDT": 12.84, "XMRUSDT": 11.81}
HOLD = 24


def load_prepared(symbol: str, root: Path, days: int) -> pd.DataFrame:
    files = sorted((root / f"prepared_{symbol}").glob("*.parquet"))[:days]
    frame = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    return add_label(frame, COST_BP[symbol])


def evaluate(
    train: pd.DataFrame, test: pd.DataFrame, columns: list[str], model_name: str
) -> dict[str, float]:
    """Fit on one block, choose the threshold on its tail, trade the other."""
    fit_x, fit_y = clean(train[columns], train["label"])
    if fit_y.nunique() < 2 or len(fit_x) < 1000:
        print(f"      skipped: {len(fit_x)} usable rows, {fit_y.nunique()} classes", flush=True)
        return {}
    model = build_model(model_name).fit(fit_x, fit_y)

    cut = int(len(train) * 0.75)
    inner = train.iloc[cut:]
    ok = inner[columns].notna().all(axis=1) & inner["forward_bp"].notna()
    if ok.sum() < 500:
        print(f"      skipped: {int(ok.sum())} inner rows for the threshold", flush=True)
        return {}
    confidence, _ = choose_confidence(
        model.predict_proba(inner.loc[ok, columns]),
        inner.loc[ok, "forward_bp"],
        inner.loc[ok, "spread_bp_now"],
        COSTS,
    )

    ok = test[columns].notna().all(axis=1) & test["forward_bp"].notna()
    block = test.loc[ok]
    if block.empty:
        return {}
    decision = decide(model.predict_proba(block[columns]), min_confidence=confidence)
    trades = thin(
        decision,
        block["forward_bp"].to_numpy(),
        block["spread_bp_now"].to_numpy(),
        ThinningRules(hold_periods=HOLD, cooldown_periods=HOLD),
    )
    return score(trades, COSTS, days=block["timestamp"].dt.date.nunique())


def main() -> None:
    p = parser(__doc__ or "")
    p.add_argument("--model", default="logistic")
    p.add_argument("--window", type=int, default=4000)

    args = p.parse_args()
    warnings.filterwarnings("ignore")

    frames = {s: load_prepared(s, args.prepared_root, (args.days or 34)) for s in args.symbols}
    columns = [c for c in FEATURES if all(c in f.columns for f in frames.values())]
    print(f"{len(columns)} shared features across {len(frames)} instruments", flush=True)

    normaliser = RollingNormaliser(window=args.window)
    scaled = {s: normaliser.transform(f, columns) for s, f in frames.items()}
    for s, f in scaled.items():
        print(f"  {s}: {len(f):,} rows, coverage {coverage(f, columns):.1%}", flush=True)
    pooled = normalise_pooled(frames, columns, normaliser)

    spec = WalkForwardSpec(train_days=14, validation_days=7, test_days=7, step_days=1, n_folds=5)
    rows = []

    for symbol in args.symbols:
        own = scaled[symbol]
        days = sorted(set(own["timestamp"].dt.date))[: spec.required_days]
        own = own[own["timestamp"].dt.date.isin(set(days))].reset_index(drop=True)
        pooled_here = pooled[pooled["timestamp"].dt.date.isin(set(days))].reset_index(drop=True)

        for fold in walk_forward(days, spec):
            own_masks = fold_masks(fold, own["timestamp"], purge=HOLD)
            test_block = own.loc[own_masks.test]

            alone = evaluate(own.loc[own_masks.train], test_block, columns, args.model)
            if alone:
                rows.append({"symbol": symbol, "fit": "alone", "fold": fold.index, **alone})

            # Same test block, same folds; the training block is every
            # instrument over the same days rather than one.
            # Purged per instrument: the pooled frame interleaves several
            # series, so a row-count purge would remove a fraction of the
            # horizon from each.
            pooled_masks = fold_masks(
                fold, pooled_here["timestamp"], purge=HOLD, groups=pooled_here["symbol"]
            )
            together = evaluate(
                pooled_here.loc[pooled_masks.train], test_block, columns, args.model
            )
            if together:
                rows.append({"symbol": symbol, "fit": "pooled", "fold": fold.index, **together})
        print(f"  {symbol} done", flush=True)

    per_fold = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    per_fold.to_csv(RESULTS / f"pooled_{args.model}_folds.csv", index=False)

    summary = per_fold.groupby(["symbol", "fit"]).agg(
        trades=("trades", "sum"),
        gross_per_trade_bp=("gross_per_trade_bp", "mean"),
        net_per_trade_bp=("net_per_trade_bp", "mean"),
        folds=("fold", "size"),
    )
    emit(summary.reset_index(), f"pooled_{args.model}")


if __name__ == "__main__":
    main()
