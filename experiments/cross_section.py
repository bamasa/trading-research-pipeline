"""Relative value across a universe, and pairs within it.

Every earlier section asked one instrument where it was going. This asks a
panel where its members stand relative to each other, which is a different
statistical bet and the one most systematic trading actually runs on.

Two strategies, one protocol:

* **Cross-sectional reversion.** Rank the universe by its recent return net of
  the cross-sectional mean, buy the laggards, sell the leaders, hold, close.
  Market-neutral by construction, so what is measured is the relative part
  alone.
* **Pairs.** Fit a hedge ratio and an Ornstein-Uhlenbeck residual on the search
  block, keep the pairs whose reversion is fast enough to be traded and whose
  amplitude clears two round trips, and trade those pairs forward on the block
  the fitting never saw.

The cost, which is the whole difficulty
---------------------------------------
Both legs cross. A cross-sectional basket of *k* names a side pays 2k round
trips to express one view; a pair pays two. Breadth spreads risk, it does not
spread cost — every name pays in full — so the per-name edge still has to clear
the per-name cost. That is why this is a harder bet than it looks, and why the
tables below report gross and cost separately rather than only their difference.

Protocol
--------
The panel is cut once. Parameters — lookback, basket size, holding period,
entry threshold, which pairs — are chosen on the search block. The final block
is read once with those choices frozen. Same discipline as §26, for the same
reason: this project has buried four findings that did not survive it.
"""

from __future__ import annotations

import argparse
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from experiments._common import RESULTS, emit
from trading_research.backtest.costs import TakerCosts
from trading_research.strategies.relative import CrossSection, backtest
from trading_research.strategies.sizing import VolatilityTarget, apply_scale
from trading_research.strategies.spread import (
    SpreadError,
    fit_hedge,
    fit_ou,
    rank_pairs,
    signal,
    z_score,
)

COSTS = TakerCosts(fee_bp_per_side=5.5, slippage_bp=0.5)

#: One row per second after the archives are reconstructed on a 1 s grid.
GRID_SECONDS = 1
ROWS_PER_DAY = 86_400 // GRID_SECONDS

#: Share of the panel the parameter search may see.
SEARCH_SHARE = 0.65

LOOKBACKS = (30, 60, 120, 300, 600)
BASKETS = (1, 2, 3, 5)
HOLDS = (30, 60, 120, 300, 600)
ENTRIES = (1.5, 2.0, 3.0)


def load_panel(root: Path, *, min_days: int = 20) -> tuple[pd.DataFrame, pd.Series]:
    """Align every instrument onto one index of log mid prices.

    Instruments missing more than a little of the span are dropped rather than
    filled: a name that was not trading is not a name that stood still, and
    carrying a stale price into a cross-sectional rank invents a laggard.
    """
    series: dict[str, pd.Series] = {}
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(directory.glob("*.parquet"))
        if len(files) < min_days:
            continue
        frame = pd.concat(
            [
                pd.read_parquet(f, columns=["timestamp", "bid_price_0", "ask_price_0"])
                for f in files
            ],
            ignore_index=True,
        )
        mid = (frame["bid_price_0"] + frame["ask_price_0"]) / 2.0
        series[directory.name] = (
            pd.Series(np.log(mid.to_numpy()), index=pd.DatetimeIndex(frame["timestamp"]))
            .groupby(level=0)
            .last()
        )

    if len(series) < 4:
        raise SystemExit(f"need at least four instruments, found {len(series)}")
    panel = pd.DataFrame(series).sort_index()
    # Keep only rows where most of the universe is present; a rank computed over
    # three of twenty-six names is not a cross-section.
    panel = panel[panel.notna().sum(axis=1) >= max(4, int(0.8 * panel.shape[1]))]
    panel = panel.ffill().dropna()
    return panel, pd.Series(panel.index)


def spreads_bp(root: Path, panel: pd.DataFrame) -> pd.Series:
    """Median half-spread per instrument, for the per-name cost."""
    out = {}
    for name in panel.columns:
        files = sorted((root / name).glob("*.parquet"))[:5]
        frame = pd.concat(
            [pd.read_parquet(f, columns=["bid_price_0", "ask_price_0"]) for f in files],
            ignore_index=True,
        )
        mid = (frame["bid_price_0"] + frame["ask_price_0"]) / 2.0
        out[name] = float(np.median((frame["ask_price_0"] - frame["bid_price_0"]) / mid * 1e4))
    return pd.Series(out)


def sweep_cross_section(panel: pd.DataFrame, cost_bp: float, bounds: slice) -> pd.DataFrame:
    """Every (lookback, basket, hold) on one block."""
    block = panel.iloc[bounds]
    rows = []
    for lookback, basket, hold in product(LOOKBACKS, BASKETS, HOLDS):
        if 2 * basket > panel.shape[1]:
            continue
        config = CrossSection(lookback=lookback, basket=basket, hold=hold)
        try:
            trades = backtest(block, config, cost_bp_per_name=cost_bp)
        except Exception:
            continue
        if len(trades) < 30:
            continue
        net = trades["net_bp"]
        rows.append(
            {
                "lookback_s": lookback,
                "basket": basket,
                "hold_s": hold,
                "rebalances": len(trades),
                "gross_bp": float(trades["gross_bp"].mean()),
                "cost_bp": float(trades["cost_bp"].mean()),
                "net_bp": float(net.mean()),
                "total_bp": float(net.sum()),
                "hit_rate": float((net > 0).mean()),
                "two_se_bp": float(2 * net.std() / np.sqrt(len(net))),
            }
        )
    return pd.DataFrame(rows)


def run_pairs(
    panel: pd.DataFrame, cost_bp: float, search: slice, final: slice
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choose pairs on the search block; trade them on the final one."""
    training = panel.iloc[search]
    candidates = rank_pairs(training, step=60, min_r_squared=0.02)
    if candidates.empty:
        return candidates, pd.DataFrame()

    # A pair is only tradeable if one standard deviation of its residual is
    # worth more than the two round trips it takes to capture, and if it reverts
    # inside a holding period a person would accept.
    round_trip = 2 * cost_bp
    keep = candidates[
        (candidates["amplitude_bp"] > round_trip) & (candidates["half_life_rows"] < 3_600)
    ].head(20)

    rows = []
    for _, pair in keep.iterrows():
        a, b = str(pair["a"]), str(pair["b"])
        try:
            hedge = fit_hedge(training[a].to_numpy(), training[b].to_numpy())
            ou = fit_ou(hedge.residual(training[a].to_numpy(), training[b].to_numpy()), step=60)
        except SpreadError:
            continue
        for entry in ENTRIES:
            for name, block in (("search", search), ("final", final)):
                part = panel.iloc[block]
                residual = hedge.residual(part[a].to_numpy(), part[b].to_numpy())
                position = signal(z_score(residual, ou), entry=entry, exit_at=0.5)
                changes = np.flatnonzero(np.diff(position, prepend=0) != 0)
                if len(changes) < 4:
                    continue
                # Profit of the residual position, in basis points of the first
                # leg. Both legs pay a round trip at every change.
                move = np.diff(residual, prepend=residual[0]) * 1e4
                gross = float(np.sum(position[:-1] * move[1:]))
                cost = float(len(changes) * 2 * cost_bp)
                rows.append(
                    {
                        "pair": f"{a}/{b}",
                        "entry_z": entry,
                        "block": name,
                        "half_life_rows": float(pair["half_life_rows"]),
                        "amplitude_bp": float(pair["amplitude_bp"]),
                        "turns": len(changes),
                        "gross_bp": gross,
                        "cost_bp": cost,
                        "net_bp": gross - cost,
                        "net_per_turn_bp": (gross - cost) / len(changes),
                    }
                )
    return keep, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=Path, default=Path("data/universe"))
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    panel, stamps = load_panel(args.universe)
    cut = int(len(panel) * SEARCH_SHARE)
    search, final = slice(0, cut), slice(cut, len(panel))
    half_spread = spreads_bp(args.universe, panel)
    # Per name, per round trip: two taker fees, two slippages, one spread.
    cost_bp = float(COSTS.round_trip_bp(float(half_spread.median())))
    print(
        f"{panel.shape[1]} instruments, {len(panel):,} rows "
        f"({len(panel) / ROWS_PER_DAY:.0f} days), median round trip {cost_bp:.2f} bp"
    )
    print(
        f"search {stamps.iloc[0].date()}..{stamps.iloc[cut - 1].date()}, "
        f"final {stamps.iloc[cut].date()}..{stamps.iloc[-1].date()}"
    )
    RESULTS.mkdir(parents=True, exist_ok=True)

    # ---- cross-sectional reversion ----------------------------------------
    on_search = sweep_cross_section(panel, cost_bp, search)
    if on_search.empty:
        raise SystemExit("no cross-sectional configuration produced enough rebalances")
    emit(on_search.nlargest(10, "net_bp"), "cross_section_search")

    best = on_search.loc[on_search["net_bp"].idxmax()]
    frozen = CrossSection(
        lookback=int(best["lookback_s"]), basket=int(best["basket"]), hold=int(best["hold_s"])
    )
    print(f"\nchosen on the search block: {frozen}")
    forward = backtest(panel.iloc[final], frozen, cost_bp_per_name=cost_bp)
    verdict = []
    for name, trades in (("search (chose it)", None), ("final (never searched)", forward)):
        if trades is None:
            verdict.append(
                {
                    "block": name,
                    "rebalances": int(best["rebalances"]),
                    "gross_bp": float(best["gross_bp"]),
                    "cost_bp": float(best["cost_bp"]),
                    "net_bp": float(best["net_bp"]),
                    "total_bp": float(best["total_bp"]),
                }
            )
            continue
        net = trades["net_bp"]
        verdict.append(
            {
                "block": name,
                "rebalances": len(trades),
                "gross_bp": float(trades["gross_bp"].mean()),
                "cost_bp": float(trades["cost_bp"].mean()),
                "net_bp": float(net.mean()),
                "total_bp": float(net.sum()),
                "two_se_bp": float(2 * net.std() / np.sqrt(len(net))) if len(net) else float("nan"),
            }
        )
    emit(pd.DataFrame(verdict), "cross_section_verdict")

    # ---- volatility-scaled overlay, on the same trades ---------------------
    if len(forward) > 10:
        realised = forward["gross_bp"].rolling(20, min_periods=5).std().shift(1)
        scale = VolatilityTarget(target_bp=float(np.nanmedian(realised)) or 10.0).scale(
            realised.fillna(realised.median()).to_numpy()
        )
        scaled = apply_scale(forward["net_bp"].to_numpy(), scale)
        emit(
            pd.DataFrame(
                [
                    {
                        "overlay": "none",
                        "mean_bp": float(forward["net_bp"].mean()),
                        "sd_bp": float(forward["net_bp"].std()),
                    },
                    {
                        "overlay": "volatility-scaled",
                        "mean_bp": float(np.mean(scaled)),
                        "sd_bp": float(np.std(scaled)),
                    },
                ]
            ),
            "cross_section_sizing",
        )

    # ---- pairs -------------------------------------------------------------
    chosen, results = run_pairs(panel, cost_bp, search, final)
    if not chosen.empty:
        emit(chosen.head(10), "pairs_candidates")
    if not results.empty:
        summary = (
            results.groupby(["block", "entry_z"])
            .agg(
                pairs=("pair", "nunique"),
                turns=("turns", "sum"),
                gross_bp=("gross_bp", "sum"),
                net_bp=("net_bp", "sum"),
                net_per_turn_bp=("net_per_turn_bp", "mean"),
                positive_pairs=("net_bp", lambda s: float((s > 0).mean())),
            )
            .reset_index()
        )
        emit(summary, "pairs_summary")
        results.to_csv(RESULTS / "pairs_detail.csv", index=False)


if __name__ == "__main__":
    main()
