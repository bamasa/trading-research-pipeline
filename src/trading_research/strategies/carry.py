"""Funding carry across a universe of coins, chosen month by month from the past alone.

A delta-neutral carry holds a coin's spot long and its perpetual short in
equal quantity: the price cancels and the position collects the funding the
perpetual's longs pay its shorts. :func:`replay` runs a rule month by month: at
each month's first hour it scores every readable coin on the funding of the
previous ``lookback`` months, holds the ``slots`` highest above
``threshold_bp_a_day``, and leaves an unfilled slot in cash. A coin kept from
one month to the next pays no round trip; one dropped is closed, one added is
opened. Funding is received at each settlement on the perpetual's quantity at
its price; the basis is carried hour by hour; costs are Binance Regular's.

This is the engine of ``trp search carry``
(:mod:`trading_research.search_engine.carry`) and of
``experiments/carry_walkforward.py``, whose registered replay it generalises.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

UNIVERSE: tuple[str, ...] = (
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "DOTUSDT",
    "LINKUSDT", "LTCUSDT", "AVAXUSDT", "ATOMUSDT", "TRXUSDT", "BCHUSDT", "ETCUSDT", "FILUSDT",
    "XLMUSDT",
)  # fmt: skip
#: Fee and slippage a side, bp: Binance Regular without the BNB discount, 2 bp slippage.
LEG_COST_BP = {"spot": 10.0 + 2.0, "perp": 5.0 + 2.0}
#: A round trip's 38 bp spread over thirty days.
COST_BP_A_DAY = 38.0 / 30.0


@dataclass
class Market:
    spot: pd.Series  # hourly close, labelled by the hour's open (UTC)
    perp: pd.Series
    funding: pd.Series  # rate per settlement, at the settlement hour


@dataclass(frozen=True)
class Rule:
    """One way of choosing: how many coins, on how many months of funding, above what."""

    slots: int = 4
    lookback: int = 1
    threshold_bp_a_day: float = COST_BP_A_DAY

    @property
    def name(self) -> str:
        floor = "cost" if self.threshold_bp_a_day > 0 else "none"
        return f"top{self.slots}-lb{self.lookback}-floor_{floor}"


Readable = Callable[[str, pd.Period], bool]


def load(root: Path, coin: str, months: Sequence[pd.Period]) -> Market:
    """Hourly spot and perpetual closes and the funding settlements of the given months."""
    from trading_research.data.binance import ArchiveSpec, download_range

    frames: dict[str, list[pd.DataFrame]] = {"spot": [], "perp": [], "funding": []}
    for m in months:
        a, b = m.start_time.date(), m.end_time.date()
        for key, market, spec in (
            ("spot", "spot", ArchiveSpec("spot", "klines", coin, "1h")),
            ("perp", "um", ArchiveSpec("futures-um", "klines", coin, "1h")),
            ("funding", "um", ArchiveSpec("futures-um", "fundingRate", coin)),
        ):
            base = root / market  # spot and perpetual klines share a directory name
            download_range(spec, a, b, base)
            folder = base / coin / spec.directory
            files = [folder / f"{d:%Y-%m-%d}.parquet" for d in pd.date_range(a, b)]
            frames[key] += [pd.read_parquet(f) for f in files if f.exists()]

    def hourly(key: str) -> pd.Series:
        f = pd.concat(frames[key], ignore_index=True)
        s = pd.Series(f["close"].astype(float).to_numpy(), index=pd.DatetimeIndex(f["open_time"]))
        return s[~s.index.duplicated()].sort_index()

    fr = pd.concat(frames["funding"], ignore_index=True)
    rates = pd.Series(fr["funding_rate"].astype(float).to_numpy(),
                      index=pd.DatetimeIndex(fr["timestamp"]).floor("1h"))  # fmt: skip
    return Market(hourly("spot"), hourly("perp"), rates.groupby(level=0).sum().sort_index())


def _utc(t: pd.Timestamp) -> pd.Timestamp:
    return t.tz_localize("UTC") if t.tzinfo is None else t


def scores(
    markets: dict[str, Market], month: pd.Period, rule: Rule, readable: Readable
) -> dict[str, float]:
    """Each candidate's funding a day, in bp, over the ``lookback`` months before ``month``."""
    window = [month - k for k in range(1, rule.lookback + 1)]
    out = {}
    for coin, mk in markets.items():
        if not readable(coin, month) or not all(readable(coin, m) for m in window):
            continue
        a, b = _utc(window[-1].start_time), _utc(window[0].end_time)
        rates = mk.funding[(mk.funding.index >= a) & (mk.funding.index <= b)]
        if len(rates):
            out[coin] = float(rates.sum()) * 1e4 / sum(m.days_in_month for m in window)
    return out


def choose(
    markets: dict[str, Market], month: pd.Period, rule: Rule, readable: Readable
) -> list[str]:
    s = scores(markets, month, rule, readable)
    ranked = sorted((c for c, v in s.items() if v > rule.threshold_bp_a_day), key=lambda c: -s[c])
    return ranked[: rule.slots]


def hourly_pnl(mk: Market, q: float, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """The carry's result each hour: funding at settlements and the basis's change."""
    hours = pd.date_range(start, end, freq="1h")
    spot = mk.spot.reindex(hours, method="ffill")  # the last close known, never a gap
    perp = mk.perp.reindex(hours, method="ffill")
    if spot.isna().any() or perp.isna().any():
        raise ValueError(f"no price before {start} for a chosen coin")
    basis = q * (spot.diff().fillna(0.0) - perp.diff().fillna(0.0))
    return basis + mk.funding.reindex(hours, fill_value=0.0) * q * perp


def _price(mk: Market, at: pd.Timestamp) -> float:
    return float(mk.spot.asof(at))  # type: ignore[arg-type]


def cost(notional_leg: float) -> float:
    return notional_leg * (LEG_COST_BP["spot"] + LEG_COST_BP["perp"]) / 1e4


def replay(
    markets: dict[str, Market],
    months: Sequence[pd.Period] | pd.PeriodIndex,
    rule: Rule,
    readable: Readable,
    chooser: Callable[[pd.Period], list[str]] | None = None,
) -> pd.DataFrame:
    """The rule month by month on the given months: each month's return on the whole capital."""
    equity, rows = 1.0, []
    held: dict[str, float] = {}
    for month in months:
        start, end = _utc(month.start_time), _utc(month.end_time).floor("1h")
        chosen = chooser(month) if chooser else choose(markets, month, rule, readable)
        pnl = 0.0
        for coin in [c for c in held if c not in chosen]:
            pnl -= cost(held.pop(coin) * _price(markets[coin], start))
        for coin in chosen:
            mk = markets[coin]
            if coin not in held:
                held[coin] = equity / rule.slots / 2 / _price(mk, start)
                pnl -= cost(held[coin] * _price(mk, start))
            pnl += float(hourly_pnl(mk, held[coin], start, end).sum())
        rows.append(
            {"month": str(month), "return_pct": pnl / equity * 100, "coins": ",".join(chosen)}
        )
        equity += pnl
    for coin, q in held.items():
        equity -= cost(q * float(markets[coin].spot.iloc[-1]))
    out = pd.DataFrame(rows)
    out.attrs["final_equity"] = equity
    return out


def summary(frame: pd.DataFrame) -> dict[str, float]:
    r = frame["return_pct"].to_numpy() / 100
    equity = np.cumprod(1 + r)
    peak = np.maximum.accumulate(np.r_[1.0, equity])
    years = len(r) / 12
    final = float(frame.attrs.get("final_equity", equity[-1]))
    sd = float(r.std(ddof=1)) if len(r) > 1 else float("nan")
    return {"total_pct": (final - 1) * 100, "a_year_pct": (final ** (1 / years) - 1) * 100,
            "worst_drawdown_pct": float((np.r_[1.0, equity] / peak - 1).min() * 100),
            "sharpe_monthly": float(r.mean() / sd * np.sqrt(12)) if sd > 0 else float("nan"),
            "losing_months": int((r < 0).sum()), "months": len(r)}  # fmt: skip


__all__ = ["COST_BP_A_DAY", "UNIVERSE", "Market", "Rule", "choose", "cost", "load", "replay",
           "scores", "summary"]  # fmt: skip
