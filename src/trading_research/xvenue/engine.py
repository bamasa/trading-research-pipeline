"""One coin's perpetual on two venues: short where the funding is higher, long where it is lower.

At each month's first hour every coin's spread is Binance's funding a day
minus Bybit's over the previous :data:`LOOKBACK` months, in bp. The rule holds
the :data:`SLOTS` coins whose spread's size is largest above the round trip
spread over thirty days, each short on the higher-funding venue and long on the
other, equal quantity. Each hour a pair earns the long leg's price change minus
the short leg's on its quantity, plus the funding the short leg receives and
minus what the long leg pays, at each venue's own settlements.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pandas as pd

SLOTS, LOOKBACK = 4, 3
#: Taker fee plus 2 bp of slippage, a side: Binance USD-M 5, Bybit linear 5.5.
LEG_BP = {"binance": 5.0 + 2.0, "bybit": 5.5 + 2.0}
ROUND_TRIP_BP = 2 * (LEG_BP["binance"] + LEG_BP["bybit"])
COST_A_DAY_BP = ROUND_TRIP_BP / 30
#: A window counts when each venue has at least this share of three settlements a day.
COMPLETE = 0.97

#: A pair: the coin and +1 (short Binance, long Bybit) or -1 (long Binance, short Bybit).
Pair = tuple[str, int]
Chooser = Callable[[pd.Period], list[Pair]]


@dataclass
class Venues:
    binance: pd.Series  # hourly perpetual close, by the bar's open (UTC)
    binance_funding: pd.Series  # rate per settlement, at its hour
    bybit: pd.Series
    bybit_funding: pd.Series


def _span(first: pd.Period, last: pd.Period) -> tuple[pd.Timestamp, pd.Timestamp, int]:
    a = first.start_time.tz_localize("UTC")
    b = (last + 1).start_time.tz_localize("UTC")
    return a, b, (b - a).days


def _window(rates: pd.Series, a: pd.Timestamp, b: pd.Timestamp) -> pd.Series:
    return rates[(rates.index >= a) & (rates.index < b)]


def spread(v: Venues, month: pd.Period, lookback: int = LOOKBACK) -> float | None:
    """Binance's funding a day minus Bybit's, bp, over the months before ``month``; None if incomplete."""
    a, b, days = _span(month - lookback, month - 1)
    out = []
    for rates in (v.binance_funding, v.bybit_funding):
        w = _window(rates, a, b)
        if len(w) < COMPLETE * 3 * days:
            return None
        out.append(float(w.sum()) * 1e4 / days)
    return out[0] - out[1]


def choose(venues: dict[str, Venues], month: pd.Period) -> list[Pair]:
    s = {c: spread(v, month) for c, v in venues.items()}
    ok = {c: x for c, x in s.items() if x is not None and abs(x) > COST_A_DAY_BP}
    ranked = sorted(ok, key=lambda c: -abs(ok[c]))[:SLOTS]
    return [(c, 1 if ok[c] > 0 else -1) for c in ranked]


def hourly(v: Venues, q: float, sign: int, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """A pair's result each hour, split into the price gap and the funding."""
    hours = pd.date_range(start, end, freq="1h")
    pb = v.binance.reindex(hours, method="ffill")
    py = v.bybit.reindex(hours, method="ffill")
    if pb.isna().any() or py.isna().any():
        raise ValueError(f"no price before {start} on a venue")
    rb = v.binance_funding.reindex(hours, fill_value=0.0)
    ry = v.bybit_funding.reindex(hours, fill_value=0.0)
    gap = q * (py.diff().fillna(0.0) - pb.diff().fillna(0.0))
    funding = q * (rb * pb - ry * py)
    return pd.DataFrame({"price": sign * gap, "funding": sign * funding})


def leg_cost(v: Venues, q: float, at: pd.Timestamp) -> float:
    """One side of both legs: the fee and slippage on each venue's notional."""
    pb, py = float(v.binance.asof(at)), float(v.bybit.asof(at))  # type: ignore[arg-type]
    return q * (pb * LEG_BP["binance"] + py * LEG_BP["bybit"]) / 1e4


def replay(
    venues: dict[str, Venues], months: Sequence[pd.Period], chooser: Chooser | None = None
) -> pd.DataFrame:
    """The rule month by month: the account's return each month, and its split, in %."""
    equity, rows = 1.0, []
    held: dict[str, tuple[int, float]] = {}
    for month in months:
        start = month.start_time.tz_localize("UTC")
        end = (month + 1).start_time.tz_localize("UTC") - pd.offsets.Hour(1)
        chosen = dict(chooser(month) if chooser else choose(venues, month))
        cost = price = funding = 0.0
        for coin in [c for c in held if chosen.get(c) != held[c][0]]:
            cost += leg_cost(venues[coin], held.pop(coin)[1], start)
        for coin, sign in chosen.items():
            v = venues[coin]
            if coin not in held:
                q = equity / SLOTS / 2 / float(v.binance.asof(start))  # type: ignore[arg-type]
                held[coin] = (sign, q)
                cost += leg_cost(v, q, start)
            h = hourly(v, held[coin][1], sign, start, end)
            price, funding = price + float(h["price"].sum()), funding + float(h["funding"].sum())
        pnl = price + funding - cost
        rows.append({"month": str(month), "return_pct": pnl / equity * 100,
                     "funding_pct": funding / equity * 100, "price_pct": price / equity * 100,
                     "cost_pct": -cost / equity * 100,
                     "pairs": ",".join(f"{c}{'+' if s > 0 else '-'}" for c, s in chosen.items())})  # fmt: skip
        equity += pnl
    final = equity - sum(leg_cost(venues[c], q, end) for c, (_, q) in held.items())
    out = pd.DataFrame(rows)
    out.attrs["final_equity"] = final
    return out
