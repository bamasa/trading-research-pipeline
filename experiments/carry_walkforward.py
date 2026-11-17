"""The walk-forward funding-carry replay of docs/preregistration/carry_walkforward_2026_11.md.

Every month from January 2021 to June 2025 the rule looks at the previous
month's funding of seventeen coins, takes the four highest above the round
trip's cost, and holds a delta-neutral carry on each (long spot, short the
USD-M perpetual) for the month. Writes experiments/results/carry_walkforward.md
and .json; refuses a second read.

    PYTHONPATH=src python experiments/carry_walkforward.py --data data
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from trading_research.data.binance import ArchiveSpec, download_range
from trading_research.validation.kill_switch import kill_switch

UNIVERSE = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT",
            "DOTUSDT", "LINKUSDT", "LTCUSDT", "AVAXUSDT", "ATOMUSDT", "TRXUSDT", "BCHUSDT",
            "ETCUSDT", "FILUSDT", "XLMUSDT")  # fmt: skip
FIRST, LAST = pd.Period("2021-01", "M"), pd.Period("2025-06", "M")
SLOTS, THRESHOLD_BP_A_DAY = 4, 38.0 / 30.0
LEG_COST_BP = {"spot": 10.0 + 2.0, "perp": 5.0 + 2.0}  # fee and slippage, a side
HELD_OUT = {"BTCUSDT", "ETHUSDT"}
HELD_OUT_MONTHS = set(pd.period_range("2020-12", "2021-06", freq="M")) | set(
    pd.period_range("2024-02", "2024-06", freq="M")
)
OUT = Path(__file__).parent / "results" / "carry_walkforward"


def readable(coin: str, month: pd.Period) -> bool:
    return not (coin in HELD_OUT and month in HELD_OUT_MONTHS) and month <= LAST


@dataclass
class Market:
    spot: pd.Series  # hourly close, labelled by the hour's open
    perp: pd.Series
    funding: pd.Series  # rate per settlement, at the settlement hour


def load(root: Path, coin: str) -> Market:
    months = [m for m in pd.period_range(FIRST - 1, LAST, freq="M") if readable(coin, m)]
    frames: dict[str, list[pd.DataFrame]] = {"spot": [], "perp": [], "funding": []}
    for m in months:
        a, b = m.start_time.date(), m.end_time.date()
        for key, market, spec in (
            ("spot", "spot", ArchiveSpec("spot", "klines", coin, "1h")),
            ("perp", "um", ArchiveSpec("futures-um", "klines", coin, "1h")),
            ("funding", "um", ArchiveSpec("futures-um", "fundingRate", coin)),
        ):
            base = root / market
            download_range(spec, a, b, base)
            folder = base / coin / spec.directory
            files = [folder / f"{d:%Y-%m-%d}.parquet" for d in pd.date_range(a, b)]
            frames[key] += [pd.read_parquet(f) for f in files if f.exists()]

    def hourly(key: str) -> pd.Series:
        f = pd.concat(frames[key], ignore_index=True)
        s = pd.Series(f["close"].astype(float).to_numpy(), index=pd.DatetimeIndex(f["open_time"]))
        return s[~s.index.duplicated()].sort_index()

    fr = pd.concat(frames["funding"], ignore_index=True)
    funding = pd.Series(fr["funding_rate"].astype(float).to_numpy(),
                        index=pd.DatetimeIndex(fr["timestamp"]).floor("1h"))  # fmt: skip
    return Market(hourly("spot"), hourly("perp"), funding.groupby(level=0).sum().sort_index())


def choose(markets: dict[str, Market], month: pd.Period) -> list[str]:
    """The four coins with the highest funding a day last month, above the threshold."""
    prev = month - 1
    scores = {}
    for coin, mk in markets.items():
        if not (readable(coin, prev) and readable(coin, month)):
            continue
        rates = mk.funding[(mk.funding.index >= prev.start_time.tz_localize("UTC"))
                           & (mk.funding.index <= prev.end_time.tz_localize("UTC"))]  # fmt: skip
        if len(rates):
            scores[coin] = float(rates.sum()) * 1e4 / prev.days_in_month
    ranked = sorted(
        (c for c, s in scores.items() if s > THRESHOLD_BP_A_DAY), key=lambda c: -scores[c]
    )
    return ranked[:SLOTS]


def hourly_pnl(mk: Market, q: float, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """The carry's result each hour from start to end: funding at settlements, the basis's change."""
    hours = pd.date_range(start, end, freq="1h")
    # the last close known at each hour, from the whole history, so an hour the
    # archive misses carries the price before it rather than a gap
    spot = mk.spot.reindex(hours, method="ffill")
    perp = mk.perp.reindex(hours, method="ffill")
    if spot.isna().any() or perp.isna().any():
        raise ValueError(f"no price before {start} for a chosen coin")
    basis = q * (spot.diff().fillna(0.0) - perp.diff().fillna(0.0))
    rate = mk.funding.reindex(hours, fill_value=0.0)
    return basis + rate * q * perp


def cost(notional_leg: float) -> float:
    return notional_leg * (LEG_COST_BP["spot"] + LEG_COST_BP["perp"]) / 1e4


def run(
    markets: dict[str, Market], guarded: bool, chooser=None, slots: int = SLOTS
) -> tuple[pd.DataFrame, dict[str, list[str]]]:  # type: ignore[no-untyped-def]
    equity, rows, chosen_by_month = 1.0, [], {}
    open_q: dict[str, float] = {}  # coin -> quantity held into the month
    for month in pd.period_range(FIRST, LAST, freq="M"):
        start = month.start_time.tz_localize("UTC")
        end = (month.end_time.tz_localize("UTC")).floor("1h")
        chosen = (chooser or choose)(markets, month)
        chosen_by_month[str(month)] = chosen
        pnl = 0.0
        for coin in list(open_q):
            if coin not in chosen:  # dropped: closed at the month's first hour
                pnl -= cost(open_q.pop(coin) * float(markets[coin].spot.asof(start)))
        for coin in chosen:
            mk = markets[coin]
            if coin not in open_q:
                q = equity / slots / 2 / float(mk.spot.asof(start))
                pnl -= cost(q * float(mk.spot.asof(start)))
                open_q[coin] = q
            q = open_q[coin]
            series = hourly_pnl(mk, q, start, end)
            if guarded and len(series) > 72:
                stop = kill_switch(series, warm_up=72).first_stop
                if stop is not None:
                    series = series.iloc[: stop.position + 2]  # closed the hour after
                    pnl -= cost(q * float(mk.spot.asof(series.index[-1])))
                    open_q.pop(coin)
            pnl += float(series.sum())
        rows.append({"month": str(month), "return_pct": pnl / equity * 100, "equity": equity + pnl,
                     "coins": len(chosen)})  # fmt: skip
        equity += pnl
    for coin, q in open_q.items():  # closed at the end
        equity -= cost(q * float(markets[coin].spot.iloc[-1]))
    rows[-1]["equity"] = equity
    return pd.DataFrame(rows), chosen_by_month


def summary(frame: pd.DataFrame) -> dict[str, float]:
    eq = frame["equity"].to_numpy()
    peak = np.maximum.accumulate(np.r_[1.0, eq])
    years = len(frame) / 12
    return {"total_pct": (eq[-1] - 1) * 100, "cagr_pct": (eq[-1] ** (1 / years) - 1) * 100,
            "worst_drawdown_pct": float((np.r_[1.0, eq] / peak - 1).min() * 100),
            "months_in_cash": int((frame["coins"] == 0).sum()),
            "losing_months": int((frame["return_pct"] < 0).sum())}  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data"))
    root = parser.parse_args().data / "carry_walkforward"
    if (OUT / "results.json").exists():
        raise SystemExit("the replay is read once")
    markets = {coin: load(root, coin) for coin in UNIVERSE}
    plain, chosen = run(markets, guarded=False)
    guard, _ = run(markets, guarded=True)
    plain["year"] = plain["month"].str[:4]
    yearly = plain.groupby("year")["return_pct"].apply(lambda r: (np.prod(1 + r / 100) - 1) * 100)
    btc = markets["BTCUSDT"].spot

    def btc_only(_: dict[str, Market], month: pd.Period) -> list[str]:
        ok = readable("BTCUSDT", month - 1) and readable("BTCUSDT", month)
        return ["BTCUSDT"] if ok else []

    btc_carry, _ = run({"BTCUSDT": markets["BTCUSDT"]}, guarded=False, chooser=btc_only, slots=1)
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"plain": summary(plain), "kill_switch": summary(guard),
               "yearly_pct": yearly.round(2).to_dict(), "months": plain.to_dict("records"),
               "months_kill_switch": guard.to_dict("records"), "chosen": chosen,
               "btc_carry_always_on": summary(btc_carry),
               "btc_spot_first_last": [float(btc.iloc[0]), float(btc.iloc[-1])]}  # fmt: skip
    (OUT / "results.json").write_text(json.dumps(payload, indent=1, default=float) + "\n")
    (OUT / "results.md").write_text(page(payload), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("plain", "kill_switch", "btc_carry_always_on",
                                              "yearly_pct")}, indent=1))  # fmt: skip


def page(p: dict) -> str:  # type: ignore[type-arg]
    def row(name: str, s: dict) -> str:  # type: ignore[type-arg]
        return (f"| {name} | {s['total_pct']:+.1f}% | {s['cagr_pct']:+.1f}% | "
                f"{s['worst_drawdown_pct']:.1f}% | {s['losing_months']} | {s['months_in_cash']} |")  # fmt: skip

    lines = ["# Funding carry, chosen month by month from the past alone", "",
             "January 2021 to June 2025, the rule of "
             "[the registration](../../../docs/preregistration/carry_walkforward_2026_11.md), "
             "read once. Returns on the whole capital, compounded monthly.", "",
             "| | total | a year | worst drawdown | losing months | months in cash |",
             "|---|--:|--:|--:|--:|--:|",
             row("the rule", p["plain"]), row("the rule with the kill switch", p["kill_switch"]),
             row("BTC carry, always on, where readable", p["btc_carry_always_on"]), "",
             "| year | the rule |", "|---|--:|"]  # fmt: skip
    lines += [f"| {y} | {v:+.1f}% |" for y, v in p["yearly_pct"].items()]
    lines += ["", "| month | return | coins chosen |", "|---|--:|---|"]
    lines += [f"| {m['month']} | {m['return_pct']:+.2f}% | {', '.join(c.removesuffix('USDT') for c in p['chosen'][m['month']]) or 'cash'} |"
              for m in p["months"]]  # fmt: skip
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
