"""What is reported beside every verdict, from a slot's per-day sums in ``SlotDays.extra``.

Every run records, per day, what these numbers need (``day/<name>``): a taker
slot its gross, funding, turnover, entry notional, the result by side and the
executor's counters (:func:`taker_day_sums`); a passive slot each side's
turnover, the weights that move ``net_per_100_clip_day`` to another fee pair,
the making part, the result by side and the simulator's counters
(``passive.BESIDE_COLUMNS``). The quoter and the market maker never read a fee,
so one run at one fee pair gives every other pair exactly:

    net(m, x) = net(m0, x0) - (m - m0) 1e-4 N_maker - (x - x0) 1e-4 N_taker

with (m0, x0) the run's own pair: the base tier for S-P and S-M, 0/0 for
S25-Z. From them, on the days a status is taken on (:func:`verdict_measures`):

* the fee break-even per side: ``gross_bp_turnover`` for a taker slot; for S-P,
  S-M and S25-Z the maker fee at which net is zero with the taker exit at the
  base fee (5.5 bp) and, separately, at 1.5 bp;
* the result at every published tier for BTCUSDT (``g1``), each in the slot's
  own unit and in USDT, and how many tiers pay (net > 0 at their own maker and
  taker fees); for S-P also at maker 0 and -1 bp (the cheapest published maker
  fee and the programme's rebate) with the taker exit at the base fee;
* the result split by side, the maximum drawdown of the daily series, net in
  USDT a day, funding, the counts of skipped entries and late snapshots and of
  flattens beyond the visible book;
* the fill decomposition (the making part over the passive fills and the
  rest) for the passive slots.

Every fee other than the base tier is a measurement, with the fee caveat
(``backtest.costs.fee_caveat``) naming the period of the days it is applied to.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date
from typing import Any

import numpy as np

from trading_research.backtest.costs import (
    BYBIT_BASE,
    BYBIT_MM_PROGRAMME_MAKER_BP,
    FeeTier,
    fee_caveat,
    tiers_for,
)

NAN = math.nan
#: The fee pair each kind of slot is run at (maker, taker), in bp.
REFERENCE: dict[str, tuple[float, float]] = {
    "taker": (BYBIT_BASE.maker_bp, BYBIT_BASE.taker_bp),
    "entry": (BYBIT_BASE.maker_bp, BYBIT_BASE.taker_bp),
    "maker": (BYBIT_BASE.maker_bp, BYBIT_BASE.taker_bp),
    "zero": (0.0, 0.0),
}
#: The taker exits the passive break-even is reported at: the base fee, the cheapest tier's.
EXIT_FEES_BP: tuple[float, ...] = (BYBIT_BASE.taker_bp, 1.5)
TAKER_COUNTERS: tuple[str, ...] = (
    "entries_skipped", "entries_busy", "entries_outside_hours", "entries_no_snapshot",
    "entry_late_snapshot", "exit_late_snapshot", "flatten_beyond_visible_book",
)  # fmt: skip
PASSIVE_COUNTERS: tuple[str, ...] = (
    "entries_skipped", "exit_late_snapshot", "flatten_beyond_visible_book", "time_above_one_clip_s",
)  # fmt: skip


def taker_day_sums(day: Any) -> dict[str, float]:
    """A taker slot's day (``backtest.taker.TakerDay``) as ``day/<name>`` sums."""
    trades = day.trades
    long, short = trades[trades["side"] > 0], trades[trades["side"] < 0]
    out = {"gross_usdt": float(trades["gross"].sum()), "funding_usdt": float(trades["funding"].sum()),
           "net_usdt": float(trades["net"].sum()), "turnover": float(day.turnover),
           "entry_notional": float(trades["notional"].sum()),
           "net_long_usdt": float(long["net"].sum()), "net_short_usdt": float(short["net"].sum()),
           "notional_long": float(long["notional"].sum()), "notional_short": float(short["notional"].sum()),
           "trades_long": float(len(long)), "trades_short": float(len(short))}  # fmt: skip
    out |= {name: float(day.counters.get(name, 0.0)) for name in TAKER_COUNTERS}
    return {f"day/{name}": value for name, value in out.items()}


def day_sum(s: Any, name: str, mask: np.ndarray) -> float:
    """The sum of ``day/<name>`` over the masked days (NaN when the slot lacks it)."""
    values = s.extra.get(f"day/{name}")
    return NAN if values is None else float(np.nansum(np.asarray(values)[mask]))


def _day(s: Any, name: str) -> np.ndarray:
    return np.asarray(s.extra.get(f"day/{name}", np.full(len(s.days), np.nan)), dtype=np.float64)


def passive_break_even(s: Any, mask: np.ndarray, taker_bp: float) -> float:
    """The maker fee at which the passive slot's net on the masked days is zero with
    the taker exit at ``taker_bp``; NaN without a passive fill."""
    m0, x0 = REFERENCE[s.kind]
    maker, taker = day_sum(s, "passive_notional", mask), day_sum(s, "taker_notional", mask)
    net = day_sum(s, "net_usdt", mask)
    if not maker > 0 or not math.isfinite(taker):
        return NAN
    return m0 + (net - (taker_bp - x0) * 1e-4 * taker) / maker * 1e4


def making_bp_per_fill(s: Any, mask: np.ndarray) -> float:
    """``spread + adverse(5 s)`` over the passive fills, the mean in bp per fill (the
    registration's K-dir quantity); notional-weighted when only the old sums exist."""
    fills = day_sum(s, "making_fills", mask)
    if math.isfinite(fills):
        return day_sum(s, "making_bp_sum", mask) / fills if fills > 0 else NAN
    if s.making is None or s.passive_notional is None:
        return NAN
    notional = float(np.nansum(s.passive_notional[mask]))
    return float(np.nansum(s.making[mask])) / notional * 1e4 if notional > 0 else NAN


def _mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if len(finite) else NAN


def series_at(s: Any, mask: np.ndarray, maker_bp: float, taker_bp: float) -> np.ndarray:
    """The slot's daily series on the masked days at another fee pair, as its status
    reads it: a taker's net per traded notional on its days with a trade; S-P's and
    S-M's ``net_per_100_clip_day`` on every day; S25-Z's net per entry notional."""
    m0, x0 = REFERENCE[s.kind]
    dm, dx = (maker_bp - m0) * 1e-4, (taker_bp - x0) * 1e-4
    with np.errstate(divide="ignore", invalid="ignore"):
        if s.kind == "taker":
            out = s.value - dx * _day(s, "turnover") / _day(s, "entry_notional") * 1e4
        elif s.kind == "zero":
            net = _day(s, "net_usdt") - dm * _day(s, "passive_notional")
            entered = _day(s, "entry_notional")
            out = np.where(entered > 0, (net - dx * _day(s, "taker_notional")) / entered * 1e4, NAN)
        else:
            weights = dm * _day(s, "per100_maker_weight") + dx * _day(s, "per100_taker_weight")
            out = np.nan_to_num(s.value, nan=0.0) - np.nan_to_num(weights, nan=0.0)
    return np.asarray(out, dtype=np.float64)[mask]


def net_usdt_at(s: Any, mask: np.ndarray, maker_bp: float, taker_bp: float) -> float:
    """The net in USDT over the masked days at another fee pair."""
    m0, x0 = REFERENCE[s.kind]
    net = day_sum(s, "net_usdt", mask)
    if s.kind == "taker":
        return net - (taker_bp - x0) * 1e-4 * day_sum(s, "turnover", mask)
    maker, taker = day_sum(s, "passive_notional", mask), day_sum(s, "taker_notional", mask)
    return net - (maker_bp - m0) * 1e-4 * maker - (taker_bp - x0) * 1e-4 * taker


def tier_results(
    s: Any, mask: np.ndarray, tiers: Sequence[FeeTier] | None = None
) -> list[dict[str, Any]]:
    """The result at every published tier for BTCUSDT, in the slot's unit and in USDT."""
    rows = tiers if tiers is not None else tiers_for("g1")
    return [{"tier": t.name, "maker_bp": t.maker_bp, "taker_bp": t.taker_bp,
             "mean": _mean(series_at(s, mask, t.maker_bp, t.taker_bp)),
             "net_usdt": net_usdt_at(s, mask, t.maker_bp, t.taker_bp)} for t in rows]  # fmt: skip


def tiers_paying(s: Any, mask: np.ndarray, tiers: Sequence[FeeTier] | None = None) -> int:
    """How many published tiers pay: net in USDT > 0 at the tier's maker and taker fees."""
    return int(sum(row["net_usdt"] > 0 for row in tier_results(s, mask, tiers)))


def max_drawdown(values: np.ndarray) -> float:
    """The largest fall of the cumulative daily series from its running peak (>= 0)."""
    finite = values[np.isfinite(values)]
    if not len(finite):
        return NAN
    path = np.r_[0.0, np.cumsum(finite)]
    return float((np.maximum.accumulate(path) - path).max())


def by_side(s: Any, mask: np.ndarray) -> dict[str, Any]:
    """The result split by side: net in USDT, the entries' notional, trades, and bp."""
    out: dict[str, Any] = {}
    for side in ("long", "short"):
        net, notional = day_sum(s, f"net_{side}_usdt", mask), day_sum(s, f"notional_{side}", mask)
        out[side] = {"net_usdt": net, "notional": notional, "trades": day_sum(s, f"trades_{side}", mask),
                     "net_bp": net / notional * 1e4 if notional > 0 else NAN}  # fmt: skip
    return out


def verdict_measures(s: Any, days: Sequence[date]) -> dict[str, Any]:
    """Everything reported beside a slot's verdict on ``days`` (its status window)."""
    from trading_research.pipeline.short_horizon.statuses import _series

    mask = s.mask(days)
    values = _series(s, s.value, mask)
    passive = s.kind != "taker"
    usable = int(mask.sum())
    net = day_sum(s, "net_usdt", mask)
    out: dict[str, Any] = {
        "series_days": len(values), "trades_a_day": float(np.nansum(s.trades[mask])) / max(usable, 1),
        "share_of_days_positive": float((values > 0).mean()) if len(values) else NAN,
        "median_day": float(np.median(values)) if len(values) else NAN,
        "max_drawdown": max_drawdown(values), "net_usdt_a_day": net / max(usable, 1),
        "funding_usdt": day_sum(s, "funding_usdt", mask), "by_side": by_side(s, mask),
        "tiers": tier_results(s, mask), "tiers_paying": tiers_paying(s, mask),
        "counters": {n: day_sum(s, n, mask) for n in (PASSIVE_COUNTERS if passive else TAKER_COUNTERS)},
        "fee_caveat": fee_caveat(period_of(days)),
    }  # fmt: skip
    if s.kind == "taker":
        turnover = day_sum(s, "turnover", mask)
        out["gross_bp_turnover"] = (
            day_sum(s, "gross_usdt", mask) / turnover * 1e4 if turnover > 0 else NAN
        )
        return out
    out["break_even_maker_bp"] = {
        f"exit_{x:g}": passive_break_even(s, mask, x) for x in EXIT_FEES_BP
    }
    making = day_sum(s, "making_usdt", mask)
    out["fill_decomposition"] = {"making_usdt": making, "rest_usdt": net - making,
                                 "making_bp_per_fill": making_bp_per_fill(s, mask)}  # fmt: skip
    if s.slot == "S-P":
        out["maker_fees"] = {f"maker_{m:g}": {"mean": _mean(series_at(s, mask, m, BYBIT_BASE.taker_bp)),
                                              "net_usdt": net_usdt_at(s, mask, m, BYBIT_BASE.taker_bp)}
                             for m in (0.0, BYBIT_MM_PROGRAMME_MAKER_BP)}  # fmt: skip
    return out


def period_of(days: Sequence[Any]) -> str:
    """The period a fee caveat names: the first and last of ``days`` (ISO dates)."""
    found = sorted(str(d) for d in days)
    return f"{found[0]} to {found[-1]}" if found else "no day"


def crossing_tiers(crossing: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """A0's crossing run (at 0/0) at the base tier and every published tier: each
    day's net less the tier's taker fee on its turnover; the mean daily net in bp of
    traded notional and the net in USDT over O's days. Every tier other than the
    base carries the fee caveat with O's dates (the days in ``crossing``)."""
    caveat = fee_caveat(period_of([c["day"] for c in crossing if "day" in c]))
    out = []
    for tier in tiers_for("g1"):
        cut = tier.taker_bp * 1e-4
        daily = [(c["net"] - cut * c["turnover"]) / c["notional"] * 1e4
                 for c in crossing if c["notional"] > 0]  # fmt: skip
        row = {"tier": tier.name, "taker_bp": tier.taker_bp,
               "mean_net_bp": float(np.mean(daily)) if daily else NAN,
               "net_usdt": float(sum(c["net"] - cut * c["turnover"] for c in crossing))}  # fmt: skip
        out.append(row if tier.name == BYBIT_BASE.name else row | {"fee_caveat": caveat})
    return out
