"""Would a system that adapts every day have made money, where a frozen one did not?

The frozen test in :mod:`experiments.reversion_fresh` carried February's entry
threshold into March and April unchanged and lost on 26 instruments of 26. That
answers "is this effect stable" and it does not answer the question a desk would
ask, which is different: nobody deploys a threshold from six weeks ago. A live
system refits on a rolling window, and if conditions change it follows them.

So this runs the same rule with everything recomputed daily from the trailing
window, on the same fresh span, and reports what an adaptive operator would have
had. Three arms, so the source of any difference is visible:

* **frozen** — February's threshold, applied unchanged. The previous test.
* **adaptive threshold** — the threshold recomputed each day from the trailing
  window, everything else fixed.
* **adaptive sign and threshold** — the *direction* of the relationship also
  re-estimated daily. If the index-to-instrument relationship inverted rather
  than vanished, this arm finds it and the frozen one cannot.

Nothing here uses information from after the day it trades. The trailing window
ends the previous midnight, which is what makes this an honest simulation of
adapting rather than a second look at the answer.

A caution that applies to the third arm in particular: re-estimating the sign
daily is a parameter with a great deal of freedom, and a strategy that flips
direction whenever the last fortnight suggests it will fit noise happily. Its
result should be read against the second arm, not on its own.
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
from trading_research.strategies.reversion import ReversionConfig, index_level, signal

COSTS = TakerCosts(fee_bp_per_side=5.5, slippage_bp=0.5)
ROWS_PER_DAY = 86_400 // 5

#: Trailing days the adaptive arms fit on. Short enough to follow a change,
#: long enough to hold a threshold steady -- and swept, because picking one is
#: exactly the choice this project keeps getting wrong.
WINDOWS = (3, 5, 10, 20)


def arm_results(
    panel: pd.DataFrame,
    books: dict[str, pd.DataFrame],
    config: ReversionConfig,
    *,
    mode: str,
    window_days: int,
    frozen_thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Trade the fresh span under one adaptation policy."""
    names = list(panel.columns)
    values = panel.to_numpy()
    total = len(panel)
    trailing = window_days * ROWS_PER_DAY
    rows = []

    for i, symbol in enumerate(names):
        if symbol not in books:
            continue
        book = books[symbol].reindex(panel.index).ffill()
        mid = ((book["bid_price_0"] + book["ask_price_0"]) / 2).to_numpy()
        spread_bp = (book["ask_price_0"] - book["bid_price_0"]).to_numpy() / mid * 1e4
        raw = signal(index_level(values, exclude=i), config)

        forward = np.full(total, np.nan)
        forward[: -config.hold] = (mid[config.hold :] / mid[: -config.hold] - 1.0) * 1e4

        decision = np.zeros(total, dtype=int)
        for start in range(trailing, total, ROWS_PER_DAY):
            past = slice(max(0, start - trailing), start)
            block = slice(start, min(total, start + ROWS_PER_DAY))

            if mode == "frozen":
                assert frozen_thresholds is not None
                if symbol not in frozen_thresholds:
                    continue
                threshold, sign = frozen_thresholds[symbol], -1.0
            else:
                history = raw[past]
                usable = history[np.isfinite(history)]
                if len(usable) < 2_000:
                    continue
                share = min(
                    0.999,
                    max(1, int(config.trades_per_day * window_days)) / len(usable),
                )
                threshold = float(np.quantile(np.abs(usable), 1.0 - share))
                sign = -1.0
                if mode == "adaptive_sign":
                    # Re-estimate the direction from the trailing window only.
                    past_forward = forward[past]
                    ok = np.isfinite(history) & np.isfinite(past_forward)
                    if ok.sum() < 1_000 or np.std(history[ok]) == 0:
                        continue
                    correlation = float(np.corrcoef(history[ok], past_forward[ok])[0, 1])
                    if not np.isfinite(correlation) or correlation == 0:
                        continue
                    sign = float(np.sign(correlation))

            window = raw[block]
            strong = np.isfinite(window) & (np.abs(window) >= threshold)
            decision[block][strong] = (sign * np.sign(window[strong])).astype(int)

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
        net = gross - cost
        rows.append(
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "day": [t.entry_index // ROWS_PER_DAY for t in trades],
                    "gross_bp": gross,
                    "cost_bp": cost,
                    "net_bp": net,
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarise(trades: pd.DataFrame, label: str) -> dict[str, object]:
    if trades.empty:
        return {"arm": label, "trades": 0}
    by_symbol = trades.groupby("symbol")["net_bp"].mean()
    result = assess(trades, value="net_bp", cluster="day", series="symbol")
    return {
        "arm": label,
        "trades": len(trades),
        "instruments": int(by_symbol.size),
        "positive_instruments": int((by_symbol > 0).sum()),
        "gross_per_trade_bp": float(trades["gross_bp"].mean()),
        "net_per_trade_bp": float(trades["net_bp"].mean()),
        "median_instrument_bp": float(by_symbol.median()),
        "days": result.clusters,
        "positive_days": result.positive_clusters,
        "cluster_t": result.cluster_t,
        "verdict": result.verdict,
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
    print(
        f"fresh span {fresh.index[0].date()}..{fresh.index[-1].date()}, "
        f"{len(shared)} instruments, {len(fresh) / ROWS_PER_DAY:.0f} days"
    )

    from trading_research.strategies.reversion import threshold_for_rate

    values = original.to_numpy()
    frozen_thresholds = {}
    for i, symbol in enumerate(shared):
        try:
            frozen_thresholds[symbol] = threshold_for_rate(
                signal(index_level(values, exclude=i), config), ROWS_PER_DAY, config
            )
        except ValueError:
            continue

    rows = [
        summarise(
            arm_results(
                fresh,
                fresh_books,
                config,
                mode="frozen",
                window_days=10,
                frozen_thresholds=frozen_thresholds,
            ),
            "frozen: February's threshold",
        )
    ]
    for window in WINDOWS:
        rows.append(
            summarise(
                arm_results(fresh, fresh_books, config, mode="adaptive", window_days=window),
                f"adaptive threshold, {window}-day window",
            )
        )
        rows.append(
            summarise(
                arm_results(fresh, fresh_books, config, mode="adaptive_sign", window_days=window),
                f"adaptive sign and threshold, {window}-day window",
            )
        )
        print(f"  {window}-day window done", flush=True)

    table = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    emit(
        table[
            [
                "arm",
                "trades",
                "instruments",
                "positive_instruments",
                "gross_per_trade_bp",
                "net_per_trade_bp",
                "days",
                "positive_days",
                "cluster_t",
            ]
        ],
        "reversion_adaptive",
    )
    best = table.loc[table["net_per_trade_bp"].idxmax()]
    print(f"\nbest arm: {best['arm']}")
    print(
        f"  {best['net_per_trade_bp']:+.2f} bp per trade, "
        f"{best['positive_instruments']}/{best['instruments']} instruments, "
        f"{best['positive_days']}/{best['days']} days positive"
    )
    print(f"  {best['verdict']}")


if __name__ == "__main__":
    main()
