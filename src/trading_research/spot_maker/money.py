"""A run's fills and what the statuses read from them: money, markouts, capacity, voids.

Every executor writes its fills to a :class:`FillLog` (time, side, price, size,
maker or taker, kind); :class:`Pricing` charges each the fee table's maker or
taker fee at its instant, at one tier. From the day's fills:

* :func:`money`: net, gross, fees and turnover in quote; ``net_bp_fill`` (net
  over turnover x 1e4, NaN on a day without a fill); ``net_bp_clip_day`` (net
  over one clip at the day's first print x 1e4, 0 on a day without a fill);
  and, a measurement for MK-S, ``net_bp_round_trip`` (net over the entries'
  notional x 1e4: a round trip is an entry and its exit of the same size);
* :func:`markouts`: per maker fill, ``side x (mid proxy at t + h - p) / p x
  1e4`` at each horizon, and the spread part against the mid proxy just before
  the fill's print;
* :func:`capacity`: the maker fills' volume and the venue's printed volume at
  the prices the slot filled at that day (K-cap's sums);
* :func:`voids`: K-units (each fee against a fresh look-up of the table for its
  kind and instant), K-acct (the day ends flat and ``spot/account`` reconciles
  the net in both bookings), K-inv and K-pos (``spot/account``'s balance and
  position checks, the unit being the slot's limit).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from trading_research.spot import account as acc
from trading_research.spot.costs import FeeNotSourced, FeeTable
from trading_research.spot.fills import PrintTape
from trading_research.spot_maker.blocks import INSTRUMENT
from trading_research.spot_maker.registration import REMAINDER_BTC
from trading_research.spot_maker.touch import Proxies

NS_PER_S = 1_000_000_000


@dataclass(frozen=True)
class Pricing:
    """The fee table at one tier."""

    table: FeeTable
    tier: str

    def fee_bp(self, ts: np.ndarray, maker: np.ndarray) -> np.ndarray:
        stamps = np.asarray(ts, dtype=np.int64)
        if not len(stamps):
            return np.zeros(0)
        maker_bp, taker_bp = self.table.fees_bp(INSTRUMENT, self.tier, stamps)
        return np.where(np.asarray(maker, dtype=bool), maker_bp, taker_bp)


@dataclass(frozen=True)
class DayFills:
    """One run's fills of one day, in time order (ties in the order they were made)."""

    ts: np.ndarray
    side: np.ndarray
    price: np.ndarray
    size: np.ndarray
    maker: np.ndarray
    kind: np.ndarray
    fee_bp: np.ndarray

    def __len__(self) -> int:
        return len(self.ts)

    @property
    def notional(self) -> np.ndarray:
        return self.price * self.size

    @property
    def fees(self) -> np.ndarray:
        return self.notional * self.fee_bp * 1e-4

    @property
    def net(self) -> float:
        return float((-self.side * self.notional).sum() - self.fees.sum())

    def account_fills(self) -> acc.Fills:
        return acc.Fills(self.ts, self.side, self.price, self.size, self.fee_bp)


@dataclass
class FillLog:
    """What an executor fills, row by row."""

    rows: list[tuple[int, int, float, float, bool, str]] = field(default_factory=list)

    def add(self, ts: int, side: int, price: float, size: float, maker: bool, kind: str) -> None:
        self.rows.append((int(ts), int(side), float(price), float(size), bool(maker), kind))

    def day(self, pricing: Pricing) -> DayFills:
        if not self.rows:
            z = np.zeros(0)
            return DayFills(z.astype(np.int64), z.astype(np.int64), z, z, z.astype(bool),
                            np.zeros(0, dtype=object), z)  # fmt: skip
        order = sorted(range(len(self.rows)), key=lambda k: (self.rows[k][0], k))
        cols = list(zip(*(self.rows[k] for k in order), strict=True))
        ts, maker = np.asarray(cols[0], dtype=np.int64), np.asarray(cols[4], dtype=bool)
        return DayFills(ts, np.asarray(cols[1], dtype=np.int64), np.asarray(cols[2], dtype=float),
                        np.asarray(cols[3], dtype=float), maker, np.asarray(cols[5], dtype=object),
                        pricing.fee_bp(ts, maker))  # fmt: skip


def money(fills: DayFills, first_price: float, clip: float) -> dict[str, float]:
    turnover = float(fills.notional.sum())
    net, fees = fills.net, float(fills.fees.sum())
    taker = ~fills.maker
    entries = float(fills.notional[fills.kind == "entry"].sum())
    return {
        "entry_notional": entries,
        "net_bp_round_trip": net / entries * 1e4 if entries > 0 else math.nan,
        "net_quote": net, "gross_quote": net + fees, "fees_quote": fees, "turnover": turnover,
        "net_bp_fill": net / turnover * 1e4 if turnover > 0 else math.nan,
        "gross_bp_fill": (net + fees) / turnover * 1e4 if turnover > 0 else math.nan,
        "net_bp_clip_day": net / (clip * first_price) * 1e4,
        "fills": float(len(fills)), "maker_fills": float(fills.maker.sum()),
        "taker_fills": float(taker.sum()), "taker_notional": float(fills.notional[taker].sum()),
        "maker_buy_btc": float(fills.size[fills.maker & (fills.side > 0)].sum()),
        "maker_sell_btc": float(fills.size[fills.maker & (fills.side < 0)].sum()),
    }  # fmt: skip


def tier_net_bp(fills: DayFills, table: FeeTable, tiers: Sequence[str]) -> dict[str, float]:
    """``net_bp_fill`` at other tiers and at zero fees; NaN where a tier is not sourced."""
    turnover = float(fills.notional.sum())
    gross = fills.net + float(fills.fees.sum())
    out = {"zero_fees": gross / turnover * 1e4 if turnover > 0 else math.nan}
    for tier in tiers:
        try:
            fee = fills.notional * Pricing(table, tier).fee_bp(fills.ts, fills.maker) * 1e-4
        except FeeNotSourced:
            out[tier] = math.nan
            continue
        out[tier] = (gross - float(fee.sum())) / turnover * 1e4 if turnover > 0 else math.nan
    return out


def markouts(fills: DayFills, proxies: Proxies, horizons_s: Sequence[float]) -> dict[str, Any]:
    """Per maker fill: each horizon's markout and the spread part, in bp (NaN where the
    mid proxy is undefined)."""
    keep = fills.maker
    ts, side, price = fills.ts[keep], fills.side[keep], fills.price[keep]
    out: dict[str, Any] = {}
    for horizon in horizons_s:
        later = proxies.mid(ts + round(horizon * NS_PER_S))
        out[f"markout_{horizon:g}s"] = side * (later - price) / price * 1e4
    before = proxies.mid(ts, strict=True)
    out["spread"] = side * (before - price) / price * 1e4
    return out


def markout_sums(fills: DayFills, proxies: Proxies, horizons_s: Sequence[float]) -> dict[str, float]:
    """Each per-fill quantity as a sum and a count over the day's maker fills."""
    out: dict[str, float] = {}
    for name, values in markouts(fills, proxies, horizons_s).items():
        ok = np.isfinite(values)
        out[f"{name}_sum"], out[f"{name}_n"] = float(values[ok].sum()), float(ok.sum())
    return out


def capacity(fills: DayFills, tape: PrintTape) -> tuple[float, float]:
    """``(the maker fills' BTC, the venue's printed BTC at the prices they filled at)``."""
    maker = fills.maker
    if not maker.any():
        return 0.0, 0.0
    prices = np.unique(fills.price[maker])
    hit = np.isin(np.round(tape.price, 8), np.round(prices, 8))
    return float(fills.size[maker].sum()), float(tape.size[hit].sum())


def voids(fills: DayFills, *, first_price: float, last_price: float, unit: float,
          table: FeeTable, tier: str) -> list[str]:  # fmt: skip
    """The void conditions the run's fills fire, in the registration's order."""
    fired: list[str] = []
    if len(fills):
        wanted = Pricing(FeeTable(table.intervals, table.verdict, table.undated), tier).fee_bp(
            fills.ts, fills.kind != "flatten"
        )
        mismatch = ~np.isclose(fills.fee_bp, wanted, rtol=1e-12, atol=0.0)
        if mismatch.any() or bool((fills.maker != (fills.kind != "flatten")).any()):
            fired.append("K-units")
    position = float(np.sum(fills.side * fills.size)) if len(fills) else 0.0
    day = acc.account_day(fills.account_fills(), fills.net, first_price, last_price, unit)
    if day.k_acct or abs(position) > REMAINDER_BTC * 10:
        fired.append("K-acct")
    if day.k_inv:
        fired.append("K-inv")
    if day.k_cap:
        fired.append("K-pos")
    return fired
