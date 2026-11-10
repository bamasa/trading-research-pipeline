"""The numbers and series behind the post "Same strategy, different market".

The repository's ``reversion`` strategy, unchanged, on the 26 Bybit perpetuals
of the benchmark's two recorded tracks. Its one fitted value, the entry
threshold, is set on 1 to 25 February 2024 (the bybit-2024-02 track's fitted
rows) and then held fixed: the strategy trades 26 February to 10 March (that
track's scored span) and 12 March to 19 April 2024 (all of the bybit-2024-04
track) without refitting. The benchmark itself refits on 12 March to 5 April
before scoring April; this file does not, which is the point of the post.

Trades are filled by the benchmark's taker at Bybit's base fee (5.5 bp a side,
0.5 bp slippage, the spread on both legs). Writes ``data.json`` (the series
the animation draws) and ``numbers.json`` (every number the post quotes).

    PYTHONPATH=src python posts/btc_ath_2024/make_data.py --data data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trading_research.bench.score import _taker
from trading_research.bench.tracks import TRACKS
from trading_research.plugins import get
from trading_research.strategies.reversion import index_level

HERE = Path(__file__).parent
REPO = HERE.parent.parent
SPLIT = pd.Timestamp("2024-03-11", tz="UTC")  # the two tracks leave 11 March out; they meet here


def trades(feb, apr) -> pd.DataFrame:  # type: ignore[no-untyped-def]
    """Every filled trade of the strategy fitted once on 1-25 February."""
    spec, rows = get("reversion", "strategy"), []
    for symbol in feb.grids:
        strategy = spec.build()
        first = feb.market(symbol)
        start = int(pd.DatetimeIndex(first.frame["timestamp"]).searchsorted(feb.start))
        strategy.fit(first.frame.iloc[:start])
        for market, skip in ((first, start), (apr.market(symbol), 0)):
            decisions = np.asarray(strategy.decide(market.frame), dtype=np.int64).copy()
            decisions[:skip] = 0
            filled = _taker(decisions, market, int(strategy.horizon), int(strategy.cooldown or 0))
            rows.append(filled.assign(symbol=symbol)[["ts", "symbol", "gross_bp", "net_bp"]])
    out = pd.concat(rows, ignore_index=True)
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    return out.sort_values(["ts", "symbol"], ignore_index=True)


def daily_autocorrelation(panel, lag: int = 120) -> pd.Series:  # type: ignore[no-untyped-def]
    """Each day's correlation of the index's ten-minute return with its next one."""
    log_mid = pd.DataFrame({s: np.log(m) for s, m in panel.mids.items()}).sort_index().ffill()
    level = pd.Series(index_level(log_mid.to_numpy(dtype=np.float64)), index=log_mid.index)
    pairs = pd.DataFrame({"past": level - level.shift(lag), "next": level.shift(-lag) - level})
    pairs = pairs.dropna()
    by_day = pairs.groupby(pairs.index.floor("1D"))
    return by_day.apply(lambda g: g["past"].corr(g["next"]) if len(g) > 1000 else np.nan).dropna()


def index_ten_minute_returns(panel) -> pd.Series:  # type: ignore[no-untyped-def]
    """The equal-weighted index's non-overlapping ten-minute log returns over a panel."""
    log_mid = pd.DataFrame({s: np.log(m) for s, m in panel.mids.items()}).sort_index().ffill()
    level = pd.Series(index_level(log_mid.to_numpy(dtype=np.float64)), index=log_mid.index)
    return level.iloc[::120].diff().dropna()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=REPO / "data")
    data_root = parser.parse_args().data
    feb = TRACKS["bybit-2024-02"].markets(data_root).parts[0]
    apr = TRACKS["bybit-2024-04"].markets(data_root).parts[0]
    if list(feb.grids) != list(apr.grids):
        raise SystemExit("the two tracks must hold the same instruments in the same order")
    n = len(feb.grids)
    t = trades(feb, apr)
    before, after = t[t["ts"] < SPLIT], t[t["ts"] >= SPLIT]

    # The February half is the benchmark's own scored span: it must match the board.
    board = json.loads((REPO / "experiments/results/bench_board.json").read_text())
    row = next(r for r in board["rows"] if r["name"] == "reversion")["bybit-2024-02"]
    if len(before) != row["trades"] or abs(before["net_bp"].mean() - row["net_bp"]) > 1e-6:
        raise SystemExit("the February trades do not reproduce the benchmark's reversion card")

    # Capital split equally over the instruments, one position at a time in each,
    # no leverage, no compounding: the portfolio's return is the sum of every
    # trade's net bp divided by the number of instruments.
    t["equity_pct"] = t["net_bp"].cumsum() / n / 100.0
    hourly = t.set_index("ts")["equity_pct"].resample("1h").last().dropna()
    peak_at = hourly.idxmax()
    zero_at = hourly[(hourly.index > peak_at) & (hourly < 0)].index[0]
    btc = pd.concat([p.mids["BTCUSDT"].resample("1h").last().dropna() for p in (feb, apr)])
    autocorr = pd.concat([daily_autocorrelation(feb), daily_autocorrelation(apr)])
    autocorr = autocorr[autocorr.index >= pd.Timestamp("2024-02-01", tz="UTC")]
    index10 = pd.concat([index_ten_minute_returns(feb), index_ten_minute_returns(apr)])
    index10 = index10[index10.index >= pd.Timestamp("2024-02-01", tz="UTC")]

    def span(a: str, b: str) -> float:
        return round(float(autocorr[a:b].mean()), 3)  # type: ignore[misc]

    iso = "%Y-%m-%dT%H:%M"

    def per_trade(x: pd.DataFrame) -> dict[str, float]:
        return {"trades": len(x), "gross_bp_per_trade": round(float(x["gross_bp"].mean()), 2),
                "net_bp_per_trade": round(float(x["net_bp"].mean()), 2)}  # fmt: skip

    numbers = {
        "instruments": n,
        "fitted": ["2024-02-01", "2024-02-25"],
        "before": per_trade(before),
        "after": per_trade(after),
        "peak": {"at": peak_at.strftime(iso), "equity_pct": round(float(hourly.max()), 1)},
        "below_zero": {"at": zero_at.strftime(iso)},
        "end": {"at": hourly.index[-1].strftime(iso), "equity_pct": round(float(hourly.iloc[-1]), 1)},
        "btc_high": {"at": btc.idxmax().strftime(iso), "hourly_mid": round(float(btc.max()))},
        "autocorrelation_mean": {"2024-02-26..2024-03-10": span("2024-02-26", "2024-03-10"),
                                 "2024-03-12..2024-04-05": span("2024-03-12", "2024-04-05"),
                                 "2024-04-06..2024-04-19": span("2024-04-06", "2024-04-19")},
    }  # fmt: skip
    series = {
        "equity_pct": [[i.strftime(iso), round(float(v), 3)] for i, v in hourly.items()],
        "btc": [[i.strftime(iso), round(float(v), 1)] for i, v in btc.items()],
        "autocorrelation": [
            [i.strftime("%Y-%m-%d"), round(float(v), 4)] for i, v in autocorr.items()
        ],
        "index_10min": [[i.strftime(iso), float(f"{v:.6g}")] for i, v in index10.items()],
    }
    (HERE / "numbers.json").write_text(json.dumps(numbers, indent=2) + "\n", encoding="utf-8")
    (HERE / "data.json").write_text(
        json.dumps(series, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    print(json.dumps(numbers, indent=2))


if __name__ == "__main__":
    main()
