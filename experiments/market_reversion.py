"""The market goes too far over ten minutes and comes back. Can that be traded?

The information audit turned up the first signal in this project that is both
strong and reproducible: the recent return of an equal-weighted index of
twenty-six perpetuals predicts the *next* move of every one of them, negatively,
at an information coefficient around -0.06. Twenty-six instruments out of
twenty-six, on both halves of the data.

Three checks were run before taking it seriously.

**Is it an artefact of shared price noise?** The classic trap: a price that
appears at the end of the lookback window and the start of the forward window
contributes its measurement noise with opposite signs, manufacturing negative
correlation from nothing. Inserting a gap between the two windows kills such an
artefact immediately. Here the effect *strengthens* slightly out to a
thirty-second gap and then decays over ten minutes, which is the shape of an
economic effect rather than an accounting one.

**Is it cross-sectional or market-wide?** Market-wide, decisively. Against a
market-neutral target — an instrument's move minus the index's over the same
window — the coefficient collapses from -0.056 to +0.004, and the twenty-six
instruments stop agreeing. So this is one bet expressed twenty-six ways, not
twenty-six bets. Trading the whole panel is the same position levered, and the
diversification a portfolio would normally buy is not available here.

**Does it clear the cost?** That is what this script measures, and the identity
says it is not obvious. Edge per trade is roughly IC times the dispersion
of the move, and that dispersion scales
with the square root of the horizon while the cost of a round trip does not
scale at all. At two minutes, a high-volatility instrument moves about 40 bp and
costs 15 to trade: 0.06 x 40 = 2.4 bp of edge against 15. At ten minutes the
move is about 90 bp and the cost is unchanged.

So the horizon is the axis that matters, and it is swept here alongside the
lookback, the trade rate and the instrument. Everything is chosen on the search
block and read once on the held-out one, as everywhere else in this document.
"""

from __future__ import annotations

import argparse
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from experiments._common import RESULTS, emit
from experiments.information_audit import to_grid
from trading_research.backtest.costs import TakerCosts
from trading_research.backtest.execution import ThinningRules, thin

COSTS = TakerCosts(fee_bp_per_side=5.5, slippage_bp=0.5)
ROWS_PER_DAY = 86_400 // 5

#: Lookback windows for the index return, in rows of five seconds.
LOOKBACKS = (24, 60, 120, 240, 480)

#: Holding periods, likewise. The identity says this is the axis that matters:
#: the move grows with its square root, the cost does not grow at all.
HORIZONS = (24, 60, 120, 240, 480, 960)

#: Trades a day, applied by taking the strongest signals.
RATES = (2.0, 5.0, 20.0, 60.0)

SEARCH_SHARE = 0.65


def load_panel(root: Path, *, min_days: int = 35) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Log mid prices on one grid, and each instrument's book beside them."""
    prices: dict[str, pd.Series] = {}
    books: dict[str, pd.DataFrame] = {}
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
        mid = ((frame["bid_price_0"] + frame["ask_price_0"]) / 2).to_numpy()
        index = pd.DatetimeIndex(frame["timestamp"])
        series = pd.Series(np.log(mid), index=index)
        prices[directory.name] = series[~series.index.duplicated()]
        frame = frame[~index.duplicated()]
        frame.index = index[~index.duplicated()]
        books[directory.name] = frame
    if len(prices) < 5:
        raise SystemExit(f"need at least five instruments, found {len(prices)}")
    panel = pd.DataFrame(prices).ffill().dropna()
    return panel, books


def backtest(
    signal: np.ndarray,
    mid: np.ndarray,
    spread_bp: np.ndarray,
    *,
    horizon: int,
    rate_per_day: float,
    threshold: float | None = None,
) -> tuple[dict[str, float], float]:
    """Take the strongest signals at the requested rate and charge every trade.

    Returns the metrics and the threshold used, so a threshold chosen on the
    search block can be applied unchanged to the held-out one.
    """
    forward = np.full(len(mid), np.nan)
    forward[:-horizon] = (mid[horizon:] / mid[:-horizon] - 1.0) * 1e4
    ok = np.isfinite(signal) & np.isfinite(forward)
    if ok.sum() < 2_000:
        return {"trades": 0.0, "net_per_trade_bp": float("nan")}, float("nan")

    if threshold is None:
        days = len(mid) / ROWS_PER_DAY
        share = min(0.999, max(1, int(rate_per_day * days)) / int(ok.sum()))
        threshold = float(np.quantile(np.abs(signal[ok]), 1 - share))

    decision = np.zeros(len(mid), dtype=int)
    strong = ok & (np.abs(signal) >= threshold)
    # Reversion: the index went up, so this goes down. The sign is the whole
    # claim, and getting it backwards would produce a mirror-image result that
    # looks equally plausible.
    decision[strong] = -np.sign(signal[strong]).astype(int)

    trades = thin(
        decision,
        np.nan_to_num(forward),
        spread_bp,
        ThinningRules(hold_periods=horizon, cooldown_periods=horizon),
    )
    if not trades:
        return {"trades": 0.0, "net_per_trade_bp": float("nan")}, threshold

    gross = np.array([t.direction * t.move_bp for t in trades])
    cost = np.array([float(COSTS.round_trip_bp(t.entry_spread_bp)) for t in trades])
    net = gross - cost
    equity = np.cumsum(net)
    peak = np.maximum.accumulate(np.maximum(equity, 0.0))
    return (
        {
            "trades": float(len(net)),
            "gross_per_trade_bp": float(gross.mean()),
            "cost_per_trade_bp": float(cost.mean()),
            "net_per_trade_bp": float(net.mean()),
            "net_bp": float(net.sum()),
            "hit_rate": float((net > 0).mean()),
            "max_drawdown_bp": float(np.max(peak - equity)),
            "two_se_bp": float(2 * net.std() / np.sqrt(len(net))),
        },
        threshold,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=Path, default=Path("data/universe"))
    parser.add_argument("--symbols", nargs="+", default=None)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    panel, books = load_panel(args.universe)
    cut = int(len(panel) * SEARCH_SHARE)
    print(
        f"{panel.shape[1]} instruments, {len(panel):,} rows "
        f"({len(panel) / ROWS_PER_DAY:.0f} days); search to "
        f"{panel.index[cut].date()}, held out after"
    )

    symbols = args.symbols or list(panel.columns)
    values = panel.to_numpy()
    names = list(panel.columns)

    rows = []
    for symbol in symbols:
        if symbol not in books:
            continue
        i = names.index(symbol)
        others = [j for j in range(len(names)) if j != i]
        book = books[symbol].reindex(panel.index).ffill()
        mid = ((book["bid_price_0"] + book["ask_price_0"]) / 2).to_numpy()
        spread_bp = (book["ask_price_0"] - book["bid_price_0"]).to_numpy() / mid * 1e4
        level = values[:, others].mean(axis=1)

        for lookback, horizon, rate in product(LOOKBACKS, HORIZONS, RATES):
            signal = np.full(len(level), np.nan)
            signal[lookback:] = (level[lookback:] - level[:-lookback]) * 1e4

            search, threshold = backtest(
                signal[:cut], mid[:cut], spread_bp[:cut], horizon=horizon, rate_per_day=rate
            )
            if not search.get("trades"):
                continue
            final, _ = backtest(
                signal[cut:],
                mid[cut:],
                spread_bp[cut:],
                horizon=horizon,
                rate_per_day=rate,
                threshold=threshold,
            )
            rows.append(
                {
                    "symbol": symbol,
                    "lookback_s": lookback * 5,
                    "hold_s": horizon * 5,
                    "rate": rate,
                    "search_trades": search["trades"],
                    "search_net_bp": search["net_per_trade_bp"],
                    "final_trades": final.get("trades", 0.0),
                    "final_gross_bp": final.get("gross_per_trade_bp", float("nan")),
                    "final_cost_bp": final.get("cost_per_trade_bp", float("nan")),
                    "final_net_bp": final.get("net_per_trade_bp", float("nan")),
                    "final_total_bp": final.get("net_bp", float("nan")),
                    "final_two_se_bp": final.get("two_se_bp", float("nan")),
                }
            )
        print(f"  {symbol} done", flush=True)

    table = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS / "market_reversion.csv", index=False)

    scored = table.dropna(subset=["search_net_bp"])
    print(
        f"\n{len(scored)} cells; {int((scored['final_net_bp'] > 0).sum())} positive on the held-out block"
    )

    # The honest reading: pick on the search block, report the held-out result.
    picked = scored.loc[scored.groupby("symbol")["search_net_bp"].idxmax()]
    emit(
        picked[
            [
                "symbol",
                "lookback_s",
                "hold_s",
                "rate",
                "search_net_bp",
                "final_trades",
                "final_gross_bp",
                "final_cost_bp",
                "final_net_bp",
                "final_two_se_bp",
            ]
        ].sort_values("final_net_bp", ascending=False),
        "market_reversion_picked",
    )

    # And the shape of the surface, so the horizon story is visible.
    emit(
        scored.groupby("hold_s")
        .agg(
            cells=("final_net_bp", "size"),
            median_gross_bp=("final_gross_bp", "median"),
            median_cost_bp=("final_cost_bp", "median"),
            median_net_bp=("final_net_bp", "median"),
            positive=("final_net_bp", lambda s: float((s > 0).mean())),
        )
        .reset_index(),
        "market_reversion_by_horizon",
    )


if __name__ == "__main__":
    main()
