"""``trp search pairs``: pairs trading on real perpetuals, chosen on D and read once on H.

The registration is ``docs/preregistration/pairs_search_2026_11.md``. Fifteen
USD-M perpetuals, every pair of them (105), walked forward by
:func:`trading_research.pairs.backtest.backtest`: each week the hedge and the
bands are fitted on the formation window before it. Each week the pairs whose
formation Engle-Granger p-value survives Holm's correction across all 105 are
the candidates, and the ``slots`` with the smallest p-values are traded. Each
trade pays four taker fills and the funding of both legs while it is held.

Four rules (the entry band, the formation's length) are replayed on the
development months; the best monthly Sharpe is chosen, written to the ledger,
then replayed once on the held-out months. Nineteen placebos trade as many
pairs each week, drawn at random from all 105: what choosing by cointegration
adds over trading any pair with the same bands.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import asdict
from datetime import timedelta
from itertools import combinations, product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_research.pairs.backtest import PairRule, backtest
from trading_research.pairs.costs import LegCosts
from trading_research.pairs.stats import holm_adjusted
from trading_research.strategies.carry import Market, load, summary

#: The carry universe less BTC and ETH, which funding_carry_2026 holds out in
#: months that would cut every pair's clock.
UNIVERSE: tuple[str, ...] = (
    "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "DOTUSDT", "LINKUSDT", "LTCUSDT",
    "AVAXUSDT", "ATOMUSDT", "TRXUSDT", "BCHUSDT", "ETCUSDT", "FILUSDT", "XLMUSDT",
)  # fmt: skip
DEVELOPMENT = (pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2023-07-01", tz="UTC"))
HELD_OUT = (pd.Timestamp("2023-07-01", tz="UTC"), pd.Timestamp("2025-07-01", tz="UTC"))
GRID: tuple[PairRule, ...] = tuple(
    PairRule(formation=f, trading=168, entry_z=e, exit_z=0.5, stop_z=4.0, max_hold=72)
    for e, f in product((2.0, 2.5), (720, 1440))
)
SLOTS, ALPHA, PLACEBOS = 5, 0.05, 19


def _costs() -> LegCosts:
    """Binance USD-M's base taker fee and the carry's 2 bp of slippage, a fill."""
    from trading_research.backtest.costs import TakerCosts

    return LegCosts(TakerCosts(fee_bp_per_side=5.0, slippage_bp=2.0), "Binance USD-M base taker")


def name(rule: PairRule) -> str:
    return f"entry{rule.entry_z:g}-formation{rule.formation // 24}d"


def bars_of(markets: dict[str, Market], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Hourly bars of every perpetual on one clock, labelled by their close, as ``pairs.data.align`` lays them out.

    An hour the archive misses carries the last close; a bar's open is the
    close before it.
    """
    opens = pd.date_range(start, end - timedelta(hours=1), freq="1h")
    columns: dict[tuple[str, str], Any] = {("open_time", ""): opens}
    closes = {c: markets[c].perp.reindex(opens.union(pd.DatetimeIndex(markets[c].perp.index))).ffill().reindex(opens)
              for c in UNIVERSE}  # fmt: skip
    for c in UNIVERSE:
        columns[("open", c)] = closes[c].shift(1).fillna(closes[c]).to_numpy()
    for c in UNIVERSE:
        columns[("close", c)] = closes[c].to_numpy()
    out = pd.DataFrame(columns, index=opens + timedelta(hours=1))
    out.columns = pd.MultiIndex.from_tuples(list(columns))
    if out.isna().any().any():
        raise ValueError("a coin has no price at the start of the span")
    return out


def funding_bp(markets: dict[str, Market], trades: pd.DataFrame) -> np.ndarray:
    """Each trade's funding, in bp of leg a's notional: long legs pay the rate, short legs receive it.

    The settlements counted are those after the entry fill and at or before the exit fill.
    """

    def paid(coin_of: pd.Series) -> np.ndarray:
        out = np.zeros(len(trades))
        for coin in coin_of.unique():
            f = markets[coin].funding.sort_index()
            cum = np.r_[0.0, np.cumsum(f.to_numpy(dtype=np.float64))]
            rows = np.flatnonzero((coin_of == coin).to_numpy())
            a = f.index.searchsorted(
                pd.DatetimeIndex(trades["entry_fill"].iloc[rows]), side="right"
            )
            b = f.index.searchsorted(pd.DatetimeIndex(trades["exit_fill"].iloc[rows]), side="right")
            out[rows] = cum[b] - cum[a]
        return out

    if trades.empty:
        return np.zeros(0)
    side, beta = (
        trades["side"].to_numpy(dtype=np.float64),
        trades["beta"].to_numpy(dtype=np.float64),
    )
    # side +1 is long a and short beta of b
    return -side * (paid(trades["a"]) - beta * paid(trades["b"])) * 1e4


def walk(markets: dict[str, Market], rule: PairRule, span: tuple[pd.Timestamp, pd.Timestamp]
         ) -> tuple[pd.DataFrame, pd.DataFrame]:  # fmt: skip
    """Every pair walked over the span (the first formation ending at its start): trades and windows."""
    start = span[0] - timedelta(hours=rule.formation)
    bars = bars_of(markets, start, span[1])
    costs = _costs()
    trades, windows = [], []
    for a, b in combinations(UNIVERSE, 2):
        r = backtest(bars, a, b, rule, costs)
        w = r.windows.assign(a=a, b=b)
        trades.append(r.trades)
        windows.append(w[["window", "a", "b", "trading_start", "p_value", "eg_t", "beta"]])
    t = pd.concat(trades, ignore_index=True)
    t["funding_bp"] = funding_bp(markets, t)
    t["net_with_funding_bp"] = t["net_bp"] + t["funding_bp"]
    return t, pd.concat(windows, ignore_index=True)


def chosen(windows: pd.DataFrame) -> set[tuple[int, str, str]]:
    """Each week's pairs passing Holm at ALPHA across all pairs, the SLOTS smallest p-values."""
    out: set[tuple[int, str, str]] = set()
    for key, group in windows.groupby("window"):
        w = int(str(key))
        g = group.assign(p_holm=holm_adjusted(group["p_value"].to_numpy()))
        g = g[g["p_holm"] <= ALPHA].sort_values(["p_value", "eg_t"]).head(SLOTS)
        out |= {(w, a, b) for a, b in zip(g["a"], g["b"], strict=True)}
    return out


def random_pairs(windows: pd.DataFrame, picked: set[tuple[int, str, str]], seed: int
                 ) -> set[tuple[int, str, str]]:  # fmt: skip
    """As many pairs each week as ``picked`` holds, drawn at random from all of them."""
    rng = np.random.default_rng(seed)
    per_week = pd.Series([w for w, _, _ in picked]).value_counts().to_dict()
    out: set[tuple[int, str, str]] = set()
    for key, group in windows.groupby("window"):
        w = int(str(key))
        k = int(per_week.get(w, 0))
        if k:
            rows = group.iloc[rng.choice(len(group), size=k, replace=False)]
            out |= {(w, a, b) for a, b in zip(rows["a"], rows["b"], strict=True)}
    return out


def portfolio(trades: pd.DataFrame, picked: set[tuple[int, str, str]],
              span: tuple[pd.Timestamp, pd.Timestamp]) -> pd.DataFrame:  # fmt: skip
    """Each month's return on the capital: a slot is a fifth of it, split over both legs' notional."""
    keys = list(zip(trades["window"], trades["a"], trades["b"], strict=True))
    mine = trades[[k in picked for k in keys]]
    weight = 1.0 / SLOTS / (1.0 + mine["beta"].abs())
    ret = mine["net_with_funding_bp"] * weight / 1e4
    month = pd.DatetimeIndex(mine["exit_fill"]).tz_convert("UTC").strftime("%Y-%m")
    by_month = ret.groupby(month.to_numpy()).sum()
    months = pd.period_range(
        span[0].tz_localize(None), span[1].tz_localize(None) - timedelta(days=1), freq="M"
    )
    frame = pd.DataFrame({"month": [str(m) for m in months]})
    frame["return_pct"] = [float(by_month.get(m, 0.0)) * 100 for m in frame["month"]]
    frame.attrs["final_equity"] = float(np.prod(1 + frame["return_pct"].to_numpy() / 100))
    frame.attrs["trades"] = len(mine)
    frame.attrs["net_bp_a_trade"] = (
        float(mine["net_with_funding_bp"].mean()) if len(mine) else math.nan
    )
    frame.attrs["gross_bp_a_trade"] = float(mine["gross_bp"].mean()) if len(mine) else math.nan
    return frame


def _summary(frame: pd.DataFrame) -> dict[str, float]:
    out = summary(frame)
    out.update({k: frame.attrs[k] for k in ("trades", "net_bp_a_trade", "gross_bp_a_trade")})
    return out


def _evaluate(markets: dict[str, Market], rule: PairRule, span: tuple[pd.Timestamp, pd.Timestamp],
              seed: int) -> dict[str, Any]:  # fmt: skip
    trades, windows = walk(markets, rule, span)
    picked = chosen(windows)
    result = _summary(portfolio(trades, picked, span))
    placebo = [_summary(portfolio(trades, random_pairs(windows, picked, seed + j), span))
               for j in range(PLACEBOS)]  # fmt: skip
    every = _summary(portfolio(trades, set(zip(windows["window"], windows["a"], windows["b"],
                                                strict=True)), span))  # fmt: skip
    return {"result": result, "placebo": placebo, "all_pairs": every,
            "months": portfolio(trades, picked, span).to_dict("records")}  # fmt: skip


def load_markets(root: Path) -> dict[str, Market]:
    months = pd.period_range("2020-10", "2025-06", freq="M")
    return {c: load(root, c, list(months)) for c in UNIVERSE}


def _score(s: dict[str, float]) -> tuple[float, float]:
    sharpe = s["sharpe_monthly"]
    return (sharpe if math.isfinite(sharpe) else -math.inf, s["total_pct"])


def run(root: Path, run_dir: Path, on_line: Callable[[str], None] = print) -> Path:
    """D, the ledger, then H once. Refuses a second read of H."""
    run_dir.mkdir(parents=True, exist_ok=True)
    h_path, ledger = run_dir / "held_out.json", run_dir / "ledger.json"
    if h_path.exists():
        raise FileExistsError(f"{h_path} exists: the held-out months are read once")
    markets = load_markets(root)
    board = {}
    for rule in GRID:
        board[name(rule)] = _evaluate(markets, rule, DEVELOPMENT, 0)
        r = board[name(rule)]["result"]
        on_line(f"  D {name(rule)}: {r['total_pct']:+.2f}%, Sharpe {r['sharpe_monthly']:.2f}, "
                f"{r['trades']} trades, net {r['net_bp_a_trade']:+.1f} bp a trade")  # fmt: skip
    best = max(GRID, key=lambda r: _score(board[name(r)]["result"]))
    d = {"board": board, "chosen": name(best), "rule": asdict(best)}
    d_bytes = json.dumps(d, sort_keys=True, default=str).encode()
    (run_dir / "development.json").write_bytes(d_bytes)
    ledger.write_text(json.dumps({"chosen": d["chosen"], "rule": d["rule"],
                                  "development_sha256": hashlib.sha256(d_bytes).hexdigest()},
                                 indent=2) + "\n")  # fmt: skip
    on_line(f"development: chose {name(best)}; ledger written")
    h = _evaluate(markets, best, HELD_OUT, 100)
    beaten = sum(p["total_pct"] < h["result"]["total_pct"] for p in h["placebo"])
    if h["result"]["total_pct"] <= 0:
        status = "killed: not positive"
    elif beaten < PLACEBOS:
        status = "killed: a placebo earned as much"
    else:
        status = "positive"
    h.update({"beats_placebos": beaten, "status": status})
    h_path.write_text(json.dumps(h, indent=2, default=str) + "\n")
    r = h["result"]
    on_line(f"held out, read once: {status}, {r['total_pct']:+.1f}% ({r['a_year_pct']:+.1f}% a year, "
            f"{r['trades']} trades, net {r['net_bp_a_trade']:+.1f} bp a trade), "
            f"beats {beaten} of {PLACEBOS} placebos")  # fmt: skip
    return h_path


__all__ = ["DEVELOPMENT", "GRID", "HELD_OUT", "UNIVERSE", "chosen", "portfolio", "run", "walk"]
