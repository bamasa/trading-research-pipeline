"""Ranking instruments on the venue they would actually be traded on.

The screen in §19-20 ran on Binance best-bid-ask, because that is what Binance
publishes for free. Every backtest since §24 runs on Bybit, whose archives carry
the full book. Ranking on one venue and trading on another is an assumption
nobody stated: fees differ, spreads differ, and the instruments are not even the
same set.

There is a second reason to redo it, which matters more. Binance stopped
publishing daily ``bookTicker`` after 30 March 2024, so the Binance screen can
only ever be measured on one window — and a ranking measured once cannot be
distinguished from a ranking of noise. Bybit publishes continuously, so the same
screen runs on two windows seven weeks apart and the two can be compared.

That comparison is the point of this script. A screen ranks forty-odd
instruments on an estimate; the top of any such list is partly the instruments
whose estimate was luckiest. If the ordering does not survive being measured
again on a different fortnight, then "CRVUSDT is the best candidate" is the same
kind of statement as the findings §25 and §26 disposed of, and the instrument
selection step has to be treated as unresolved rather than done.

Cheap on purpose
----------------
Only the touch is reconstructed — one level a side, on a one-second grid. A day
costs about eleven seconds that way against the two minutes ten levels take, so
a wide universe over two windows is an hour rather than a day, and the screen
stays what a screen should be: much cheaper than the study it precedes.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from experiments._common import RESULTS, emit
from trading_research.backtest.costs import TakerCosts
from trading_research.data.bybit import BybitArchiveError, download_range
from trading_research.data.screen import screen_directory

#: Bybit USD-M perpetual taker, and the slippage assumed throughout.
COSTS = TakerCosts(fee_bp_per_side=5.5, slippage_bp=0.5)

#: Two windows, seven weeks apart, both well inside the period the rest of the
#: project uses. Far enough apart that a ranking surviving both is saying
#: something about the instruments rather than about a fortnight.
WINDOWS = {
    "february": (date(2024, 2, 5), date(2024, 2, 7)),
    "march": (date(2024, 3, 25), date(2024, 3, 27)),
}

ROOT = Path("data/screen_bybit")


def universe(limit: int, *, before: datetime | None = None) -> list[str]:
    """USDT perpetuals that existed before the windows, by current turnover.

    Turnover today is a poor proxy for turnover in early 2024 and is used only
    to order the candidates, not to score them — an instrument that has since
    become popular still gets screened on its 2024 book.
    """
    before = before or datetime(2024, 1, 1, tzinfo=UTC)
    url = "https://api.bybit.com/v5/market/instruments-info?category=linear&limit=1000"
    with urllib.request.urlopen(url, timeout=60) as response:
        listed = json.load(response)["result"]["list"]
    eligible = [
        row["symbol"]
        for row in listed
        if row["symbol"].endswith("USDT")
        and datetime.fromtimestamp(int(row["launchTime"]) / 1000, tz=UTC) < before
    ]

    url = "https://api.bybit.com/v5/market/tickers?category=linear"
    with urllib.request.urlopen(url, timeout=60) as response:
        tickers = {
            row["symbol"]: float(row["turnover24h"])
            for row in json.load(response)["result"]["list"]
        }
    eligible.sort(key=lambda s: -tickers.get(s, 0.0))
    return eligible[:limit]


def _fetch(job: tuple[str, str, date, date]) -> tuple[str, str, str]:
    """Reconstruct the touch for one instrument over one window."""
    window, symbol, start, end = job
    out = ROOT / window / symbol / "bookTicker"
    if list(out.glob("*.parquet")):
        return window, symbol, "cached"
    try:
        download_range(symbol, start, end, out, depth=1, grid_ms=1000, workers=1)
    except (BybitArchiveError, ValueError, OSError) as exc:
        return window, symbol, f"{type(exc).__name__}: {exc}"[:80]
    return window, symbol, "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruments", type=int, default=80)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--horizon-s", type=float, default=120.0)
    args = parser.parse_args()

    symbols = universe(args.instruments)
    print(f"{len(symbols)} instruments, {len(WINDOWS)} windows")

    jobs = [
        (window, symbol, start, end)
        for window, (start, end) in WINDOWS.items()
        for symbol in symbols
    ]
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_fetch, job) for job in jobs]
        for future in as_completed(futures):
            window, symbol, status = future.result()
            done += 1
            if status not in ("ok", "cached"):
                print(f"  [{done}/{len(jobs)}] {window} {symbol}: {status}")
            elif done % 20 == 0:
                print(f"  [{done}/{len(jobs)}] ...")

    tables = {}
    for window in WINDOWS:
        table = screen_directory(ROOT / window, horizon_s=args.horizon_s, costs=COSTS)
        scored = table.dropna(subset=["edge_over_cost"]) if "edge_over_cost" in table else table
        tables[window] = scored.set_index("symbol")
        print(f"\n{window}: {len(scored)} instruments screened")

    first, second = list(WINDOWS)
    both = tables[first].join(
        tables[second], lsuffix=f"_{first}", rsuffix=f"_{second}", how="inner"
    )
    emit(both.reset_index(), "screen_bybit_raw")

    from scipy.stats import spearmanr

    rows = []
    for column in (
        "information_coefficient",
        "volatility_bp",
        "median_spread_bp",
        "headroom",
        "edge_over_cost",
    ):
        a, b = both[f"{column}_{first}"], both[f"{column}_{second}"]
        result = spearmanr(a, b)
        rows.append(
            {
                "quantity": column,
                "instruments": len(both),
                "rank_correlation": float(result.statistic),
                "p_value": float(result.pvalue),
            }
        )

    # How much of the top of one window's list survives into the other's. The
    # number a person actually acts on, since a screen is used by taking its
    # head rather than by reading its whole ordering.
    for k in (5, 10, 20):
        top_a = set(both[f"edge_over_cost_{first}"].nlargest(k).index)
        top_b = set(both[f"edge_over_cost_{second}"].nlargest(k).index)
        rows.append(
            {
                "quantity": f"top-{k} overlap",
                "instruments": len(both),
                "rank_correlation": len(top_a & top_b) / k,
                "p_value": float("nan"),
            }
        )
    emit(pd.DataFrame(rows), "screen_bybit_stability")

    RESULTS.mkdir(parents=True, exist_ok=True)
    ranked = both.assign(
        mean_edge_over_cost=lambda d: (
            (d[f"edge_over_cost_{first}"] + d[f"edge_over_cost_{second}"]) / 2
        )
    ).nlargest(20, "mean_edge_over_cost")
    emit(
        ranked.reset_index()[
            [
                "symbol",
                f"information_coefficient_{first}",
                f"information_coefficient_{second}",
                f"round_trip_bp_{first}",
                f"edge_over_cost_{first}",
                f"edge_over_cost_{second}",
                "mean_edge_over_cost",
            ]
        ],
        "screen_bybit_ranked",
    )


if __name__ == "__main__":
    main()
