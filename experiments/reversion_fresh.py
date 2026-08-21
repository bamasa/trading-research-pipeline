"""The test that has killed four findings, run on the fifth.

Everything about the reversion candidate has been measured on one span: 1
February to 10 March 2024, cut into a block that chose its parameters and a
block that did not. That second block is what makes it a candidate rather than a
number. It is not what makes it a result.

This is the test that would. The configuration is frozen in
:mod:`trading_research.strategies.reversion` — lookback, hold, cooldown, trade
rate, the instruments excluded — and nothing here re-tunes any of it. The entry
threshold comes from the *original* span, because a threshold refitted on the
period being judged is the selection step this whole exercise exists to avoid.
Then it runs on 12 March to 20 April, which no part of the search, the
parameter choice or the model fitting has seen.

The conditions that would kill it were written down in ``docs/findings.md``
before this ran:

1. A median at or below zero across instruments.
2. The holding-period profile failing to repeat.

Both are reported below whichever way they come out. Four earlier candidates
reached this point and did not survive it; the value of the register is that
they are all still listed, with what killed each one.
"""

from __future__ import annotations

import argparse
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from experiments._common import RESULTS, emit
from experiments.information_audit import to_grid
from trading_research.backtest.costs import TakerCosts
from trading_research.backtest.execution import ThinningRules, thin
from trading_research.strategies.reversion import (
    ReversionConfig,
    decide,
    index_level,
    signal,
    threshold_for_rate,
)

COSTS = TakerCosts(fee_bp_per_side=5.5, slippage_bp=0.5)
ROWS_PER_DAY = 86_400 // 5

ORIGINAL = Path("data/universe")
FRESH = Path("data/universe_fresh")
ORIGINAL_SPAN = (date(2024, 2, 1), date(2024, 3, 10))
FRESH_SPAN = (date(2024, 3, 12), date(2024, 4, 20))

#: Holding periods, in rows, for the profile check. The frozen configuration
#: uses 120; the others are here only to see whether the shape repeats.
PROFILE = (24, 60, 120, 240, 480)


def load_panel(root: Path, *, min_days: int) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    prices, books = {}, {}
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(directory.glob("*.parquet"))
        if len(files) < min_days:
            continue
        frame = to_grid(
            pd.concat(
                [
                    pd.read_parquet(f, columns=["timestamp", "bid_price_0", "ask_price_0"])
                    for f in files
                ],
                ignore_index=True,
            )
        )
        index = pd.DatetimeIndex(frame["timestamp"])
        keep = ~index.duplicated()
        frame, index = frame[keep], index[keep]
        frame.index = index
        mid = ((frame["bid_price_0"] + frame["ask_price_0"]) / 2).to_numpy()
        prices[directory.name] = pd.Series(np.log(mid), index=index)
        books[directory.name] = frame
    return pd.DataFrame(prices).ffill().dropna(), books


def run(
    panel: pd.DataFrame,
    books: dict[str, pd.DataFrame],
    config: ReversionConfig,
    thresholds: dict[str, float],
    hold: int,
) -> pd.DataFrame:
    """Trade every instrument with a threshold supplied from elsewhere."""
    names = list(panel.columns)
    values = panel.to_numpy()
    rows = []
    for i, symbol in enumerate(names):
        if symbol not in thresholds or symbol not in books:
            continue
        book = books[symbol].reindex(panel.index).ffill()
        mid = ((book["bid_price_0"] + book["ask_price_0"]) / 2).to_numpy()
        spread_bp = (book["ask_price_0"] - book["bid_price_0"]).to_numpy() / mid * 1e4

        raw = signal(index_level(values, exclude=i), config)
        decision = decide(raw, thresholds[symbol])

        forward = np.full(len(mid), np.nan)
        forward[:-hold] = (mid[hold:] / mid[:-hold] - 1.0) * 1e4
        decision[~np.isfinite(forward)] = 0

        trades = thin(
            decision,
            np.nan_to_num(forward),
            spread_bp,
            ThinningRules(hold_periods=hold, cooldown_periods=config.cooldown),
        )
        if not trades:
            continue
        gross = np.array([t.direction * t.move_bp for t in trades])
        cost = np.array([float(COSTS.round_trip_bp(t.entry_spread_bp)) for t in trades])
        net = gross - cost
        equity = np.cumsum(net)
        peak = np.maximum.accumulate(np.maximum(equity, 0.0))
        rows.append(
            {
                "symbol": symbol,
                "hold_s": hold * 5,
                "trades": len(net),
                "gross_per_trade_bp": float(gross.mean()),
                "cost_per_trade_bp": float(cost.mean()),
                "net_per_trade_bp": float(net.mean()),
                "net_bp": float(net.sum()),
                "hit_rate": float((net > 0).mean()),
                "max_drawdown_bp": float(np.max(peak - equity)),
                "two_se_bp": float(2 * net.std() / np.sqrt(len(net))),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=ORIGINAL)
    parser.add_argument("--fresh", type=Path, default=FRESH)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    config = ReversionConfig()
    print(f"frozen configuration: {config}")

    original, _ = load_panel(args.original, min_days=35)
    fresh, fresh_books = load_panel(args.fresh, min_days=30)
    shared = [c for c in original.columns if c in fresh.columns]
    original, fresh = original[shared], fresh[shared]
    print(
        f"original {original.index[0].date()}..{original.index[-1].date()} "
        f"({len(original):,} rows); fresh {fresh.index[0].date()}..{fresh.index[-1].date()} "
        f"({len(fresh):,} rows); {len(shared)} instruments in both"
    )

    # Thresholds from the original span only. This is the line that makes the
    # test a test: refitting them on the fresh data would re-tune the strategy
    # on the period being judged.
    values = original.to_numpy()
    thresholds = {}
    for i, symbol in enumerate(shared):
        raw = signal(index_level(values, exclude=i), config)
        try:
            thresholds[symbol] = threshold_for_rate(raw, ROWS_PER_DAY, config)
        except ValueError:
            continue
    print(f"{len(thresholds)} thresholds carried over unchanged\n")

    frozen = run(fresh, fresh_books, config, thresholds, config.hold)
    if frozen.empty:
        raise SystemExit("no trades on the fresh span")

    RESULTS.mkdir(parents=True, exist_ok=True)
    emit(
        frozen.sort_values("net_per_trade_bp", ascending=False),
        "reversion_fresh_by_instrument",
    )

    positive = int((frozen["net_per_trade_bp"] > 0).sum())
    median = float(frozen["net_per_trade_bp"].median())
    print(
        f"\nKILL CONDITION 1 — median at or below zero: "
        f"median {median:+.2f} bp, {positive}/{len(frozen)} instruments positive "
        f"-> {'SURVIVES' if median > 0 else 'KILLED'}"
    )

    # The second condition: does the shape repeat?
    profile = []
    for hold in PROFILE:
        part = run(fresh, fresh_books, config, thresholds, hold)
        if part.empty:
            continue
        profile.append(
            {
                "hold_s": hold * 5,
                "instruments": len(part),
                "median_trades": float(part["trades"].median()),
                "median_gross_bp": float(part["gross_per_trade_bp"].median()),
                "median_net_bp": float(part["net_per_trade_bp"].median()),
                "positive": float((part["net_per_trade_bp"] > 0).mean()),
            }
        )
    shape = pd.DataFrame(profile)
    emit(shape, "reversion_fresh_profile")

    if not shape.empty:
        best = int(shape.loc[shape["median_net_bp"].idxmax(), "hold_s"])
        print(
            f"\nKILL CONDITION 2 — profile fails to repeat: best hold {best}s "
            f"(the frozen configuration uses {config.hold_seconds}s) "
            f"-> {'SURVIVES' if best in (300, 600, 1200) else 'KILLED'}"
        )


if __name__ == "__main__":
    main()
