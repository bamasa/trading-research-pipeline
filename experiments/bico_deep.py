"""Everything, aimed at one instrument: where — if anywhere — is BICOUSDT positive?

BICOUSDT earned this treatment by elimination. Of the three instruments the
corrected searches ran on, it is the only one whose gross edge did not flip sign
on the held-out block (+4.0 bp search, +1.5 bp final, ±9.6). That is not a
result; it is the absence of a refutation. This script is the follow-up: instead
of one more configuration sweep, it asks *where* the strategy makes and loses
its money — which windows, which regimes, which trades — and whether any of that
structure persists.

The multiplicity problem, stated before the mining starts
---------------------------------------------------------
Everything below is a search over windows and conditions, and a search over
windows will find positive windows in pure noise. Nothing found here is a
finding. The discipline is:

1. Every subgroup or window result is compared against what chance would
   produce — the count of positive cells against its binomial expectation, and
   persistence (is a positive window more likely after a positive window?)
   against the unconditional rate.
2. Anything that looks alive earns a test on **days this repository has never
   downloaded** — Bybit publishes past 10 March — and only that test would be
   reported as a result.

What runs
---------
* **Window map** — ten configurations spanning every family (five bare rules,
  the learned models, boosting, regression on the leak-free smoothed target, an
  uncertainty gate, the corrected search's winner), walked daily over all 39
  days, aggregated into 2-day windows. Rules with no ML are the user-visible
  control: if only the fitted models lose, the fitting is the problem; if
  everything loses alike, the cost is.
* **Fortnight arrangements** — fit once on 14 days, choose the threshold on the
  tail, trade the next 7 untouched, sliding the whole arrangement two days at a
  time. The direct test of "train on one fortnight, apply on the next week".
* **Break-segmented training** — CUSUM breaks cut the span; fit inside segment
  *k*, trade segment *k+1*. The regime story taken literally.
* **Trade autopsy** — every trade of the winner configuration, tagged with its
  entry context (hour, spread, volatility, imbalance, direction, exit reason).
  Conditions that look profitable on the search block are then applied,
  unchanged, to the final block.
* **A TCN arm** — one dilated-convolution configuration, refit every five days
  on capped rows. Kept small because §8 already measured the architecture
  underperforming logistic on this data; it is here so "did you try the
  network" has a current answer.

Book volumes enter through the depth plane (visible size, imbalance through ten
levels). Traded volume does not: the Bybit archives this project uses carry the
order book only.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from experiments._common import RESULTS, emit
from experiments.grand_search import (
    COSTS,
    ROWS_PER_DAY,
    Config,
    Data,
    build_features,
    normalise,
    run_block,
    to_grid,
)
from trading_research.backtest.evaluate import choose_confidence, decide
from trading_research.backtest.execution import ThinningRules, thin
from trading_research.pipeline.stages import build_model
from trading_research.validation.changepoint import segment

DAY = int(ROWS_PER_DAY)

#: One representative per family. Horizons and holds follow what each family
#: preferred across the earlier sweeps rather than one size forced on all.
CANDIDATES: dict[str, Config] = {
    "rule:order_flow": Config(
        "micro", "order_flow", "direction", 24, 24, 24, "clock", "net_bp", "1d", 10, "open"
    ),
    "rule:momentum": Config(
        "micro", "momentum", "direction", 60, 60, 60, "clock", "net_bp", "1d", 10, "open"
    ),
    "rule:mean_reversion": Config(
        "micro", "mean_reversion", "direction", 24, 24, 24, "clock", "net_bp", "1d", 10, "open"
    ),
    "rule:breakout": Config(
        "micro", "breakout", "direction", 60, 60, 60, "clock", "net_bp", "1d", 10, "open"
    ),
    "rule:spread_capture": Config(
        "micro", "spread_capture", "direction", 24, 24, 24, "clock", "net_bp", "1d", 10, "open"
    ),
    "ml:logistic": Config(
        "micro", "logistic", "direction", 24, 24, 24, "clock", "net_per_trade_bp", "1d", 10, "open"
    ),
    "ml:xgboost_depth": Config(
        "depth",
        "xgboost",
        "triple_barrier",
        60,
        60,
        60,
        "take_profit_stop",
        "net_per_trade_bp",
        "1d",
        10,
        "open",
        exit_level=25.0,
    ),
    "ml:ridge_smoothed": Config(
        "micro",
        "ridge",
        "forward_smoothed",
        60,
        60,
        0,
        "clock",
        "net_per_trade_bp",
        "1d",
        10,
        "open",
    ),
    "ml:bound_ridge": Config(
        "micro",
        "bound_ridge",
        "magnitude",
        120,
        120,
        0,
        "clock",
        "net_per_trade_bp",
        "2d",
        10,
        "open",
    ),
    "mix:winner240": Config(
        "micro",
        "agree",
        "direction",
        24,
        60,
        60,
        "take_profit_stop",
        "net_bp",
        "1d",
        10,
        "open",
        exit_level=25.0,
        norm_window=16000,
    ),
}


def load(symbol: str) -> tuple[Data, pd.Series, list[int]]:
    files = sorted(Path(f"data/book/{symbol}").glob("*.parquet"))
    book = to_grid(pd.concat([pd.read_parquet(p) for p in files], ignore_index=True))
    raw = build_features(book)
    mid = ((book["bid_price_0"] + book["ask_price_0"]) / 2).to_numpy()
    spread = ((book["ask_price_0"] - book["bid_price_0"]) / mid * 1e4).to_numpy()
    frames = {}
    for window in (1000, 4000, 16000):
        frame = normalise(raw, window=window).astype("float32")
        frame["mid"] = mid
        frames[window] = frame
    cut = int(len(book) * 0.65)
    data = Data(frames, spread, mid, search_end=cut)
    breaks = [b.index for b in segment(mid, spread, minimum_rows=40_000).breaks]
    return data, book["timestamp"], breaks


# --------------------------------------------------------------------------- #
# 1. The window map
# --------------------------------------------------------------------------- #


def window_map(data: Data, timestamps: pd.Series, breaks: list[int]) -> pd.DataFrame:
    """Every candidate, walked over the whole span, cut into 2-day windows."""
    rows = []
    total = len(timestamps)
    for name, config in CANDIDATES.items():
        trades = run_block(config, data, [(0, total)], breaks)
        if trades.empty:
            continue
        trades["window"] = (trades["at"] // (2 * DAY)).astype(int)
        for window, group in trades.groupby("window"):
            rows.append(
                {
                    "candidate": name,
                    "window": int(window),
                    "starts": str(timestamps.iloc[int(window) * 2 * DAY].date()),
                    "trades": len(group),
                    "net_bp": float(group["net_bp"].sum()),
                    "gross_per_trade_bp": float(group["gross_bp"].mean()),
                }
            )
    return pd.DataFrame(rows)


def window_verdict(table: pd.DataFrame) -> pd.DataFrame:
    """Positive-window counts against chance, and whether positivity persists."""
    rows = []
    for name, group in table.groupby("candidate"):
        ordered = group.sort_values("window")
        positive = (ordered["net_bp"] > 0).to_numpy()
        n = len(positive)
        base = float(positive.mean())
        after_positive = positive[1:][positive[:-1]] if n > 1 else np.array([])
        rows.append(
            {
                "candidate": name,
                "windows": n,
                "positive": int(positive.sum()),
                "share_positive": base,
                # Persistence: a real regime makes positive windows cluster. In
                # noise, P(positive | previous positive) equals the base rate.
                "persistence": float(after_positive.mean())
                if len(after_positive)
                else float("nan"),
                "best_window_bp": float(ordered["net_bp"].max()),
                "total_bp": float(ordered["net_bp"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("total_bp", ascending=False)


# --------------------------------------------------------------------------- #
# 2. Fit a fortnight, trade a week
# --------------------------------------------------------------------------- #


def fortnight_scan(data: Data, timestamps: pd.Series) -> pd.DataFrame:
    """One fit on 14 days, threshold on the tail, 7 days traded untouched.

    The arrangement asked for in so many words: maybe a model fitted on one
    stretch works on the next before the market moves on. Slid two days at a
    time so every alignment gets a chance.
    """
    rows = []
    total = len(timestamps)
    for name in ("rule:order_flow", "ml:logistic", "ml:xgboost_depth"):
        config = CANDIDATES[name]
        features = data.frames[config.norm_window]
        columns = data.plane[config.plane]
        target = data.target(config.target, config.horizon)
        forward = data.forward(config.hold)
        usable = features[columns].notna().all(axis=1).to_numpy() & target.notna().to_numpy()

        for start in range(14 * DAY, total - 7 * DAY, 2 * DAY):
            train = np.zeros(total, dtype=bool)
            train[start - 14 * DAY : start - max(config.horizon, config.hold)] = True
            train &= usable
            index = np.flatnonzero(train)
            if len(index) < 20_000:
                continue
            cut_at = int(len(index) * 0.75)
            fit_index, inner = index[:cut_at], index[cut_at:]

            estimator = build_model(config.model)
            y = target.to_numpy()
            if config.model in ("logistic", "xgboost") and len(np.unique(y[fit_index])) < 2:
                continue
            estimator.fit(
                features.iloc[fit_index][columns], pd.Series(y[fit_index], index=fit_index)
            )
            confidence, _ = choose_confidence(
                estimator.predict_proba(features.iloc[inner][columns]),
                pd.Series(forward[inner]),
                pd.Series(data.spread_bp[inner]),
                COSTS,
                objective=config.objective,
            )

            block = slice(start, start + 7 * DAY)
            ok = (
                np.isfinite(forward[block])
                & features.iloc[block][columns].notna().all(axis=1).to_numpy()
            )
            proba = estimator.predict_proba(features.iloc[block][columns].ffill().fillna(0.0))
            decision = np.where(ok, decide(proba, min_confidence=confidence), 0)
            trades = thin(
                decision,
                forward[block],
                data.spread_bp[block],
                ThinningRules(hold_periods=config.hold, cooldown_periods=config.cooldown),
            )
            net = [
                t.direction * t.move_bp - float(COSTS.round_trip_bp(t.entry_spread_bp))
                for t in trades
            ]
            rows.append(
                {
                    "candidate": name,
                    "apply_starts": str(timestamps.iloc[start].date()),
                    "trades": len(net),
                    "net_bp": float(np.sum(net)) if net else 0.0,
                    "net_per_trade_bp": float(np.mean(net)) if net else float("nan"),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 3. Between the breaks
# --------------------------------------------------------------------------- #


def break_segments(data: Data, timestamps: pd.Series) -> pd.DataFrame:
    """Fit inside one detected regime, trade the following one."""
    cuts = segment(data.mid, data.spread_bp, minimum_rows=30_000)
    rows = []
    config = CANDIDATES["ml:logistic"]
    for (a_start, a_end), (b_start, b_end) in zip(cuts.bounds, cuts.bounds[1:], strict=False):
        bounded = Config(**{**config.__dict__, "refit": "5d", "train_days": 30})
        trades = run_block(bounded, data, [(b_start, b_end)], breaks=[])
        if trades.empty:
            continue
        rows.append(
            {
                "trained_on": f"{timestamps.iloc[a_start].date()}..{timestamps.iloc[a_end - 1].date()}",
                "applied_to": f"{timestamps.iloc[b_start].date()}..{timestamps.iloc[b_end - 1].date()}",
                "trades": len(trades),
                "net_bp": float(trades["net_bp"].sum()),
                "net_per_trade_bp": float(trades["net_bp"].mean()),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 4. The trade autopsy
# --------------------------------------------------------------------------- #


def autopsy(data: Data, timestamps: pd.Series, breaks: list[int]) -> pd.DataFrame:
    """Tag every trade of the winner with its context; test survivors forward.

    Conditions are formed on the search block and applied verbatim to the final
    block. A condition that only exists on the block that suggested it is the
    definition of what this project keeps burying.
    """
    config = CANDIDATES["mix:winner240"]
    features = data.frames[config.norm_window]

    def tagged(bounds: list[tuple[int, int]]) -> pd.DataFrame:
        trades = run_block(config, data, bounds, breaks)
        if trades.empty:
            return trades
        at = trades["at"].to_numpy()
        trades["hour"] = timestamps.iloc[at].dt.hour.to_numpy() // 6
        trades["spread_q"] = pd.qcut(data.spread_bp[at], 3, labels=False, duplicates="drop")
        vol = features["log_mid_vol50"].to_numpy()[at]
        trades["vol_q"] = pd.qcut(vol, 3, labels=False, duplicates="drop")
        return trades

    cut = data.search_end
    search = tagged([(0, cut)])
    final = tagged([(cut, len(timestamps))])
    rows = []
    for dimension in ("exit_reason", "hour", "spread_q", "vol_q"):
        if search.empty or dimension not in search:
            continue
        for value, group in search.groupby(dimension):
            if len(group) < 30:
                continue
            match = final[final[dimension] == value] if not final.empty else final
            rows.append(
                {
                    "condition": f"{dimension}={value}",
                    "search_trades": len(group),
                    "search_net_bp": float(group["net_bp"].mean()),
                    "final_trades": len(match),
                    "final_net_bp": float(match["net_bp"].mean()) if len(match) else float("nan"),
                }
            )
    return pd.DataFrame(rows).sort_values("search_net_bp", ascending=False)


# --------------------------------------------------------------------------- #
# 5. The network, so the answer is current
# --------------------------------------------------------------------------- #


def tcn_arm(data: Data, breaks: list[int]) -> pd.DataFrame:
    config = Config("micro", "tcn", "direction", 24, 24, 24, "clock", "net_bp", "5d", 10, "open")
    rows = []
    for name, bounds in (
        ("search", [(0, data.search_end)]),
        ("final", [(data.search_end, len(data.mid))]),
    ):
        trades = run_block(config, data, bounds, breaks, max_train_rows=30_000)
        rows.append(
            {
                "block": name,
                "trades": len(trades),
                "net_per_trade_bp": float(trades["net_bp"].mean()) if len(trades) else float("nan"),
                "net_bp": float(trades["net_bp"].sum()) if len(trades) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BICOUSDT")
    parser.add_argument("--skip-tcn", action="store_true")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    data, timestamps, breaks = load(args.symbol)
    print(f"{args.symbol}: {len(timestamps):,} rows, {len(breaks)} breaks")
    RESULTS.mkdir(parents=True, exist_ok=True)

    windows = window_map(data, timestamps, breaks)
    emit(windows, "bico_windows")
    emit(window_verdict(windows), "bico_window_verdict")
    emit(fortnight_scan(data, timestamps), "bico_fortnight")
    emit(break_segments(data, timestamps), "bico_breaks")
    emit(autopsy(data, timestamps, breaks), "bico_autopsy")
    if not args.skip_tcn:
        emit(tcn_arm(data, breaks), "bico_tcn")


if __name__ == "__main__":
    main()
