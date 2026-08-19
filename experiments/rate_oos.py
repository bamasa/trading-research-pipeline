"""Does the promising trade rate survive a period it was not chosen on?

A sweep over four instruments and six target rates found two or three positive
cells. At twenty-two trades and forty basis points of per-trade dispersion those
sit around two standard errors, and twenty-four cells produce that many by
chance — so the finding cannot be distinguished from luck on the data that
produced it.

It can be distinguished on data that did not. Bybit publishes back to 2023, so
the same *procedure* runs on months the sweep never saw.

The procedure, not the model
----------------------------
A single fit applied for two months would go stale — §10 measured daily
refitting as the best schedule available. So what is held fixed is the method:
the feature set, the rolling normalisation, the refit cadence, the way the
threshold is chosen on validation, and the target rate. Nothing is re-tuned.
Freezing the weights instead would test whether one model decays, which is a
different and much less interesting question.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from experiments._common import RESULTS, emit
from trading_research.backtest.costs import TakerCosts
from trading_research.backtest.execution import ThinningRules, thin
from trading_research.features.normalise import RollingNormaliser
from trading_research.models.regression import RidgeBaseline

#: Bybit USD-M perpetual taker, 0.055% a side.
COSTS = TakerCosts(fee_bp_per_side=5.5, slippage_bp=0.5)

HOLD = 24
SUBSAMPLE = 50
ROWS_PER_DAY = 86400 / 5

#: Rates the sweep favoured, plus their neighbours as controls.
RATES = (("2/week", 2 / 7), ("1/day", 1.0), ("2/day", 2.0), ("5/day", 5.0))

#: Days of history each refit sees, and how often it refits.
TRAIN_DAYS = 14
VALIDATION_DAYS = 5
APPLY_DAYS = 2

FEATURES = ("imbalance", "spread_bp", "ret20")


def load(symbol: str, root: Path = Path("data/book")) -> pd.DataFrame:
    files = sorted((root / symbol).glob("*.parquet"))
    if not files:
        raise SystemExit(f"no book data for {symbol}")
    frame = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    return frame.iloc[::SUBSAMPLE].reset_index(drop=True)


def prepare(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, np.ndarray, pd.Series]:
    mid = ((frame["bid_price_0"] + frame["ask_price_0"]) / 2).astype("float64")
    values = mid.to_numpy()
    bid, ask = frame["bid_size_0"].to_numpy(), frame["ask_size_0"].to_numpy()
    spread = ((frame["ask_price_0"] - frame["bid_price_0"]) / mid * 1e4).to_numpy()

    ret20 = np.full(len(values), np.nan)
    ret20[20:] = np.log(values[20:] / values[:-20]) * 1e4
    forward = np.full(len(values), np.nan)
    forward[:-HOLD] = np.log(values[HOLD:] / values[:-HOLD]) * 1e4

    raw = pd.DataFrame(
        {"imbalance": (bid - ask) / (bid + ask), "spread_bp": spread, "ret20": ret20}
    )
    normaliser = RollingNormaliser(window=4000, exclude=frozenset({"imbalance"}))
    return (
        normaliser.transform(raw, list(FEATURES)),
        pd.Series(forward),
        spread,
        frame["timestamp"],
    )


def main() -> None:
    warnings.filterwarnings("ignore")
    frame = load("BTCUSDT")
    features, forward, spread, timestamps = prepare(frame)
    usable = features.notna().all(axis=1) & forward.notna()

    train_rows = int(TRAIN_DAYS * ROWS_PER_DAY)
    validation_rows = int(VALIDATION_DAYS * ROWS_PER_DAY)
    apply_rows = int(APPLY_DAYS * ROWS_PER_DAY)
    step = train_rows + validation_rows

    days = timestamps.dt.date
    # The sweep that produced the candidate used everything up to 9 March. Days
    # after it are the ones that can settle the question.
    fresh_from = pd.Timestamp("2024-03-10").date()

    trades: list[dict[str, object]] = []
    start = 0
    while start + step + apply_rows <= len(frame):
        train = slice(start, start + train_rows)
        validation = slice(start + train_rows, start + step)
        apply = slice(start + step, start + step + apply_rows)

        train_mask = usable.iloc[train]
        if train_mask.sum() < 10_000 or usable.iloc[validation].sum() < 3_000:
            start += apply_rows
            continue

        model = RidgeBaseline(scale_bp=12.0).fit(
            features.iloc[train][train_mask], forward.iloc[train][train_mask]
        )
        validation_mask = usable.iloc[validation]
        on_validation = model.predict_edge_bp(features.iloc[validation][validation_mask])
        apply_mask = usable.iloc[apply]
        if apply_mask.sum() < 1000:
            start += apply_rows
            continue
        on_apply = model.predict_edge_bp(features.iloc[apply][apply_mask])

        block_days = float(apply_mask.sum() / ROWS_PER_DAY)
        validation_days = float(validation_mask.sum() / ROWS_PER_DAY)
        block_forward = forward.iloc[apply][apply_mask].to_numpy()
        block_spread = spread[apply][apply_mask.to_numpy()]
        block_day = days.iloc[apply][apply_mask].iloc[0]

        for name, per_day in RATES:
            wanted = max(1, int(per_day * validation_days))
            share = min(0.999, wanted / max(len(on_validation), 1))
            threshold = float(np.quantile(np.abs(on_validation), 1 - share))
            decision = np.where(np.abs(on_apply) < threshold, 0, np.sign(on_apply)).astype(int)
            taken = thin(
                decision,
                block_forward,
                block_spread,
                ThinningRules(hold_periods=HOLD, cooldown_periods=HOLD),
            )
            for trade in taken:
                cost = float(COSTS.round_trip_bp(trade.entry_spread_bp))
                trades.append(
                    {
                        "rate": name,
                        "day": block_day,
                        "period": "chosen on" if block_day < fresh_from else "fresh",
                        "net_bp": trade.direction * trade.move_bp - cost,
                        "days": block_days,
                    }
                )
        start += apply_rows

    table = pd.DataFrame(trades)
    if table.empty:
        raise SystemExit("no trades produced")

    summary = (
        table.groupby(["rate", "period"])
        .agg(
            trades=("net_bp", "size"),
            net_per_trade_bp=("net_bp", "mean"),
            total_bp=("net_bp", "sum"),
            positive=("net_bp", lambda s: float((s > 0).mean())),
            dispersion_bp=("net_bp", "std"),
        )
        .reset_index()
    )
    # Two standard errors, so a reader can see at a glance whether a cell is
    # distinguishable from zero rather than having to compute it.
    summary["two_se_bp"] = 2 * summary["dispersion_bp"] / np.sqrt(summary["trades"])
    RESULTS.mkdir(parents=True, exist_ok=True)
    emit(summary, "rate_out_of_sample")


if __name__ == "__main__":
    main()
