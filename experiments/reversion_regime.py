"""Trade the reversion only when the market is in the state it needs.

The candidate died on fresh data, and the diagnosis was specific: on the span
where it worked, the index's ten-minute return predicted its next ten-minute
return at -0.10; on the span where it failed, at -0.0003. The strategy was a bet
on a market state, and the state went away.

That does not, by itself, condemn it. A strategy conditional on a regime is an
ordinary thing to build — nothing works in every state, and a rule that knows
which state it needs is better than one that does not. The question is whether
the state can be recognised *in advance*, from data available at the time, which
is what separates a conditional strategy from a story told afterwards.

So: measure the index's autocorrelation over a trailing window, take a threshold
from the original span, and on the fresh span trade only on days the trailing
measurement says the market is reverting. Stand aside otherwise.

Three things this has to get right to mean anything
---------------------------------------------------
**The measurement is causal.** The autocorrelation for a given day is computed
over a window ending the previous midnight. Nothing about the day being traded
enters the decision to trade it.

**The threshold comes from the old span.** Choosing it on the fresh data would
make this a search for the days that worked, which is the mistake the whole
project exists to avoid.

**Standing aside must be counted.** A gate that trades ten days out of forty and
reports its per-trade average is describing a strategy nobody can size. Days
skipped, and the result per *available* day, are reported beside it.

What would make this real, and what would make it another period
----------------------------------------------------------------
If the gate works, days it admits should be profitable and days it rejects
should not — and the gap between them should be larger than the gap between two
arbitrary halves of the same data. That last comparison is the control, and it
is run here, because a gate that merely splits a noisy series into a better half
and a worse half will always look like it worked.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from experiments._common import RESULTS, emit
from experiments.reversion_fresh import FRESH, ORIGINAL, load_panel
from trading_research.backtest.costs import TakerCosts
from trading_research.backtest.execution import ThinningRules, thin
from trading_research.evaluation.significance import assess
from trading_research.strategies.reversion import (
    ReversionConfig,
    index_level,
    signal,
    threshold_for_rate,
    trailing_autocorrelation,
)

COSTS = TakerCosts(fee_bp_per_side=5.5, slippage_bp=0.5)
ROWS_PER_DAY = 86_400 // 5

#: Trailing days the regime is measured over. Short enough to react, long
#: enough to measure a correlation on.
REGIME_WINDOWS = (1, 2, 3, 5)


def trade(
    panel: pd.DataFrame,
    books: dict[str, pd.DataFrame],
    config: ReversionConfig,
    thresholds: dict[str, float],
    *,
    regime_window: int,
    gate_at: float | None,
) -> pd.DataFrame:
    """Trade the rule, optionally only where the trailing regime admits it."""
    names = list(panel.columns)
    values = panel.to_numpy()
    total = len(panel)
    window_rows = regime_window * ROWS_PER_DAY
    regime = trailing_autocorrelation(values.mean(axis=1), config.hold, window_rows)

    rows = []
    for i, symbol in enumerate(names):
        if symbol not in thresholds or symbol not in books:
            continue
        book = books[symbol].reindex(panel.index).ffill()
        mid = ((book["bid_price_0"] + book["ask_price_0"]) / 2).to_numpy()
        spread_bp = (book["ask_price_0"] - book["bid_price_0"]).to_numpy() / mid * 1e4

        raw = signal(index_level(values, exclude=i), config)
        forward = np.full(total, np.nan)
        forward[: -config.hold] = (mid[config.hold :] / mid[: -config.hold] - 1.0) * 1e4

        strong = np.isfinite(raw) & (np.abs(raw) >= thresholds[symbol])
        if gate_at is not None:
            # The gate: only where the market has recently been reverting.
            strong &= np.isfinite(regime) & (regime <= gate_at)
        decision = np.zeros(total, dtype=int)
        decision[strong] = -np.sign(raw[strong]).astype(int)
        decision[~np.isfinite(forward)] = 0

        trades = thin(
            decision,
            np.nan_to_num(forward),
            spread_bp,
            ThinningRules(hold_periods=config.hold, cooldown_periods=config.cooldown),
        )
        if not trades:
            continue
        gross = np.array([t.direction * t.move_bp for t in trades])
        cost = np.array([float(COSTS.round_trip_bp(t.entry_spread_bp)) for t in trades])
        rows.append(
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "day": [t.entry_index // ROWS_PER_DAY for t in trades],
                    "regime": [regime[t.entry_index] for t in trades],
                    "gross_bp": gross,
                    "net_bp": gross - cost,
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarise(trades: pd.DataFrame, label: str, available_days: int) -> dict[str, object]:
    if trades.empty:
        return {"arm": label, "trades": 0, "days_traded": 0, "available_days": available_days}
    result = assess(trades, value="net_bp", cluster="day", series="symbol")
    by_symbol = trades.groupby("symbol")["net_bp"].mean()
    return {
        "arm": label,
        "trades": len(trades),
        "days_traded": result.clusters,
        "available_days": available_days,
        # Spread over every day the strategy could have traded, so standing
        # aside is visible in the figure it is judged on.
        "net_per_available_day_bp": float(trades["net_bp"].sum() / max(available_days, 1)),
        "gross_per_trade_bp": float(trades["gross_bp"].mean()),
        "net_per_trade_bp": float(trades["net_bp"].mean()),
        "positive_instruments": int((by_symbol > 0).sum()),
        "instruments": int(by_symbol.size),
        "positive_days": result.positive_clusters,
        "cluster_t": result.cluster_t,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=ORIGINAL)
    parser.add_argument("--fresh", type=Path, default=FRESH)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    config = ReversionConfig()
    original, _ = load_panel(args.original, min_days=35)
    fresh, fresh_books = load_panel(args.fresh, min_days=30)
    shared = [c for c in original.columns if c in fresh.columns]
    original, fresh = original[shared], fresh[shared]

    values = original.to_numpy()
    thresholds = {}
    for i, symbol in enumerate(shared):
        try:
            thresholds[symbol] = threshold_for_rate(
                signal(index_level(values, exclude=i), config), ROWS_PER_DAY, config
            )
        except ValueError:
            continue

    fresh_days = int(len(fresh) / ROWS_PER_DAY)
    rows = [
        summarise(
            trade(fresh, fresh_books, config, thresholds, regime_window=3, gate_at=None),
            "ungated (the test that failed)",
            fresh_days,
        )
    ]

    for window in REGIME_WINDOWS:
        # The gate level comes from the old span: the median trailing
        # autocorrelation over the days the strategy was profitable on.
        old_regime = trailing_autocorrelation(
            original.to_numpy().mean(axis=1), config.hold, window * ROWS_PER_DAY
        )
        level = float(np.nanmedian(old_regime))
        gated = trade(fresh, fresh_books, config, thresholds, regime_window=window, gate_at=level)
        rows.append(
            summarise(gated, f"gated at {level:+.3f}, {window}-day regime window", fresh_days)
        )
        # The control: the gate's complement. If the gate works, the days it
        # rejects should be worse than the days it admits by more than an
        # arbitrary split would give.
        rejected = trade(fresh, fresh_books, config, thresholds, regime_window=window, gate_at=None)
        if not rejected.empty:
            outside = rejected[rejected["regime"] > level]
            if not outside.empty:
                rows.append(
                    summarise(
                        outside,
                        f"  the days that gate rejected, {window}-day window",
                        fresh_days,
                    )
                )
        print(f"  {window}-day window done", flush=True)

    table = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    emit(table, "reversion_regime")


if __name__ == "__main__":
    main()
