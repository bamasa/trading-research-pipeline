"""How long to learn from, and how long the learning lasts.

Two numbers govern every walk-forward in this project: how many days a fit sees,
and how many days it is trusted for afterwards. Both have been assumptions
throughout — ten days and one, mostly, inherited from the first script that
needed a number — and §26's refit axis only ever compared three fixed cadences
against each other at a single training length.

They deserve to be searched rather than assumed, because they are the two
parameters a regime story actually predicts. If the market reorganises every
few days, a short fit applied briefly should beat a long fit applied for a
fortnight, and the surface over the two axes should have a ridge. If there is no
structure to track, the surface is flat and noisy and the whole regime framing
is decoration.

The design
----------
A full grid, not a sample: nine training lengths by seven apply lengths, on four
instruments, for three model families that fail differently — a rule that fits
nothing, a linear classifier, and gradient boosting. 756 cells, each a complete
walk-forward over every day available for that instrument.

Reading it
----------
The headline number is not the best cell. With 756 cells, the best one is
whichever cell noise favoured, and this project has already buried three
findings that were exactly that. What the grid is asked instead:

* **Is the positive rate above chance?** Compared against the binomial
  expectation for the number of cells.
* **Is there a ridge?** A real regime effect makes neighbouring cells agree —
  positive cells cluster along a diagonal. Noise scatters them. Measured as the
  correlation between a cell and its neighbours in both axes.
* **Does the best geometry agree across instruments?** A property of markets
  transfers; a property of one instrument's February does not.

Only if all three point the same way does anything here get carried to fresh
data.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from experiments._common import RESULTS, emit
from experiments.grand_search import (
    ROWS_PER_DAY,
    Config,
    Data,
    build_features,
    normalise,
    run_block,
    to_grid,
)
from trading_research.validation.changepoint import segment

#: Days of history each fit sees.
TRAIN_DAYS = (1, 2, 3, 5, 7, 10, 14, 20, 30)

#: Days that fit is then trusted for, before the next refit replaces it.
APPLY_DAYS = (0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 14.0)

#: Three ways of being wrong. The rule fits nothing, so its row isolates the
#: effect of the apply window from the effect of staleness in a fitted model:
#: a rule cannot go stale, so any structure in its row is the market's, not the
#: model's.
MODELS = {
    "order_flow": {"plane": "micro", "target": "direction", "horizon": 24, "hold": 24},
    "logistic": {"plane": "micro", "target": "direction", "horizon": 24, "hold": 24},
    "xgboost": {"plane": "depth", "target": "triple_barrier", "horizon": 60, "hold": 60},
}


def load(symbol: str, root: Path) -> tuple[Data, list[int], int]:
    files = sorted((root / symbol).glob("*.parquet"))
    if not files:
        raise SystemExit(f"no book data for {symbol} under {root}")
    book = to_grid(pd.concat([pd.read_parquet(p) for p in files], ignore_index=True))
    raw = build_features(book)
    mid = ((book["bid_price_0"] + book["ask_price_0"]) / 2).to_numpy()
    spread = ((book["ask_price_0"] - book["bid_price_0"]) / mid * 1e4).to_numpy()
    frame = normalise(raw, window=4000).astype("float32")
    frame["mid"] = mid
    data = Data({4000: frame}, spread, mid, search_end=len(book))
    breaks = [b.index for b in segment(mid, spread, minimum_rows=40_000).breaks]
    return data, breaks, len(book)


def ridge_score(surface: pd.DataFrame) -> float:
    """Do neighbouring cells agree?

    A regime effect is smooth in both axes: if 5 days trained and 2 days applied
    works, 5-by-3 works nearly as well. Noise has no such obligation. Measured
    as the correlation between each cell and the mean of its four neighbours;
    near zero means the surface is scatter.
    """
    values = surface.to_numpy(dtype="float64")
    padded = np.pad(values, 1, mode="edge")
    neighbours = (padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]) / 4.0
    a, b = values.ravel(), neighbours.ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 8 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--symbols", nargs="+", default=["BTCUSDT", "XRPUSDT", "CRVUSDT", "BICOUSDT"]
    )
    parser.add_argument("--book", type=Path, default=Path("data/book"))
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    rows: list[dict[str, object]] = []
    for symbol in args.symbols:
        data, breaks, total = load(symbol, args.book)
        span_days = total / ROWS_PER_DAY
        print(f"\n{symbol}: {total:,} rows, {span_days:.0f} days, {len(breaks)} breaks")

        for model, shape in MODELS.items():
            for train_days in TRAIN_DAYS:
                if train_days > span_days - 3:
                    continue
                for apply_days in APPLY_DAYS:
                    config = Config(
                        plane=str(shape["plane"]),
                        model=model,
                        target=str(shape["target"]),
                        horizon=int(shape["horizon"]),
                        hold=int(shape["hold"]),
                        cooldown=int(shape["hold"]),
                        exit="clock",
                        objective="net_bp",
                        refit=f"{apply_days:g}d",
                        train_days=train_days,
                        gate="open",
                    )
                    try:
                        trades = run_block(config, data, [(0, total)], breaks)
                    except Exception as exc:  # a bad cell must not end the grid
                        rows.append(
                            {
                                "symbol": symbol,
                                "model": model,
                                "train_days": train_days,
                                "apply_days": apply_days,
                                "skipped": f"{type(exc).__name__}: {exc}"[:60],
                            }
                        )
                        continue
                    if trades.empty:
                        continue
                    net = trades["net_bp"]
                    rows.append(
                        {
                            "symbol": symbol,
                            "model": model,
                            "train_days": train_days,
                            "apply_days": apply_days,
                            "trades": len(net),
                            "net_per_trade_bp": float(net.mean()),
                            "gross_per_trade_bp": float(trades["gross_bp"].mean()),
                            "net_bp": float(net.sum()),
                        }
                    )
                print(f"  {model:12s} train={train_days:2d}d done")

    table = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS / "window_geometry.csv", index=False)
    scored = table.dropna(subset=["net_per_trade_bp"])
    print(f"\n{len(scored)} cells scored, {len(table) - len(scored)} skipped")

    # 1. Positive rate against chance.
    summary = []
    for (symbol, model), group in scored.groupby(["symbol", "model"]):
        positive = int((group["net_per_trade_bp"] > 0).sum())
        surface = group.pivot_table(
            index="train_days", columns="apply_days", values="net_per_trade_bp"
        )
        summary.append(
            {
                "symbol": symbol,
                "model": model,
                "cells": len(group),
                "positive": positive,
                "share_positive": positive / len(group),
                "best_cell_bp": float(group["net_per_trade_bp"].max()),
                "median_cell_bp": float(group["net_per_trade_bp"].median()),
                "gross_median_bp": float(group["gross_per_trade_bp"].median()),
                "neighbour_agreement": ridge_score(surface),
            }
        )
    emit(pd.DataFrame(summary), "window_geometry_summary")

    # 2. Does the best geometry agree across instruments?
    best = (
        scored.loc[scored.groupby(["symbol", "model"])["net_per_trade_bp"].idxmax()][
            ["symbol", "model", "train_days", "apply_days", "trades", "net_per_trade_bp"]
        ]
        .sort_values(["model", "symbol"])
        .reset_index(drop=True)
    )
    emit(best, "window_geometry_best")

    # 3. The marginal effect of each axis, pooled — the question "does training
    #    longer help at all" separated from which cell won.
    marginals = []
    for axis in ("train_days", "apply_days"):
        for value, group in scored.groupby(axis):
            marginals.append(
                {
                    "axis": axis,
                    "value": value,
                    "cells": len(group),
                    "median_net_bp": float(group["net_per_trade_bp"].median()),
                    "median_gross_bp": float(group["gross_per_trade_bp"].median()),
                }
            )
    emit(pd.DataFrame(marginals), "window_geometry_marginals")


if __name__ == "__main__":
    main()
