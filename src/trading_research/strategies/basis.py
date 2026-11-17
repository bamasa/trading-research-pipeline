"""The spot-perpetual basis trade: open the carry when the perpetual is unusually rich, close when it is not.

The basis is the perpetual's premium over spot, ``perp / spot - 1`` in bp, at
each hourly close. It wanders around a level of its own (a few bp below zero on
Binance's USD-M contracts) and now and then jumps when longs crowd the
perpetual. When it is ``entry_bp`` above its median of the previous
``window_days``, the rule buys spot and sells the perpetual in equal quantity,
the carry's position: it earns the premium's fall back towards its level, and
the funding the longs pay while it waits. It closes when the premium is back
within ``exit_bp`` of the median, or after ``max_hold_hours``.

Positions are hourly, decided on an hour's close and opened or closed at that
close; at most ``slots`` coins at once, each sized at ``1 / slots`` of the
equity split half a leg; the strongest deviations fill free slots first. The
costs, the funding and the basis's hour-by-hour carry are
:mod:`trading_research.strategies.carry`'s, so the two are measured alike. Only
the perpetual being rich is traded: the other side would need spot sold short.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading_research.strategies.carry import LEG_COST_BP, Market

Readable = Callable[[str, pd.Period], bool]


@dataclass(frozen=True)
class BasisRule:
    entry_bp: float = 45.0
    exit_bp: float = 0.0
    window_days: int = 7
    slots: int = 4
    max_hold_hours: int = 24 * 14

    @property
    def name(self) -> str:
        return f"entry{self.entry_bp:g}-exit{self.exit_bp:g}-w{self.window_days}d"


@dataclass
class Panel:
    """Every coin on one hourly index: spot, perpetual, funding, readability, the deviation."""

    hours: pd.DatetimeIndex
    coins: list[str]
    spot: np.ndarray  # (hours, coins)
    perp: np.ndarray
    funding: np.ndarray
    readable: np.ndarray  # bool


def panel(markets: dict[str, Market], hours: pd.DatetimeIndex, readable: Readable) -> Panel:
    coins = sorted(markets)
    months = (
        hours.tz_localize(None).to_period("M") if hours.tz is not None else hours.to_period("M")
    )
    cols: dict[str, list[np.ndarray]] = {"spot": [], "perp": [], "funding": [], "ok": []}
    for c in coins:
        mk = markets[c]
        ok = np.array([readable(c, m) for m in months])
        spot = mk.spot.reindex(hours).to_numpy(dtype=np.float64)
        perp = mk.perp.reindex(hours).to_numpy(dtype=np.float64)
        ok &= np.isfinite(spot) & np.isfinite(perp)
        cols["spot"].append(pd.Series(spot).ffill().to_numpy())
        cols["perp"].append(pd.Series(perp).ffill().to_numpy())
        cols["funding"].append(mk.funding.reindex(hours, fill_value=0.0).to_numpy(dtype=np.float64))
        cols["ok"].append(ok)
    stack = {k: np.column_stack(v) for k, v in cols.items()}
    return Panel(hours, coins, stack["spot"], stack["perp"], np.nan_to_num(stack["funding"]),
                 stack["ok"])  # fmt: skip


def deviation(p: Panel, window_days: int) -> np.ndarray:
    """The basis above its median over the previous ``window_days`` (the current hour excluded), bp."""
    basis = np.where(p.readable, (p.perp / p.spot - 1.0) * 1e4, np.nan)
    frame = pd.DataFrame(basis)
    hours = 24 * window_days
    med = frame.rolling(hours, min_periods=hours // 2).median().shift(1).to_numpy()
    return basis - med


def replay(
    p: Panel,
    rule: BasisRule,
    start: pd.Timestamp,
    end: pd.Timestamp,
    dev: np.ndarray | None = None,
) -> pd.DataFrame:
    """The rule hour by hour from ``start`` to ``end``: each month's return on the whole capital.

    ``dev`` replaces the deviation (the placebos pass a shifted one); the
    positions are always marked on the real prices.
    """
    dev = deviation(p, rule.window_days) if dev is None else dev
    cost_side = (LEG_COST_BP["spot"] + LEG_COST_BP["perp"]) / 1e4
    lo, hi = int(p.hours.searchsorted(start)), int(p.hours.searchsorted(end, side="right"))
    equity = 1.0
    held: dict[int, tuple[float, int]] = {}  # coin index -> (quantity, hour opened)
    month_pnl: dict[str, float] = {}
    month_start: dict[str, float] = {}
    trades = 0
    for t in range(lo, hi):
        key = str(p.hours[t].strftime("%Y-%m"))
        month_start.setdefault(key, equity)
        pnl = 0.0
        if t > lo:  # carry of positions held over the hour that just closed
            for j, (q, _) in held.items():
                pnl += q * ((p.spot[t, j] - p.spot[t - 1, j]) - (p.perp[t, j] - p.perp[t - 1, j]))
                pnl += q * p.funding[t, j] * p.perp[t, j]
        for j in list(held):
            q, opened = held[j]
            gone = not p.readable[t, j] or (t + 1 < hi and not p.readable[t + 1, j])
            if gone or t - opened >= rule.max_hold_hours or not dev[t, j] > rule.exit_bp:
                pnl -= q * p.spot[t, j] * cost_side
                del held[j]
        free = rule.slots - len(held)
        if free > 0:
            ok = p.readable[t] & np.isfinite(dev[t]) & (dev[t] > rule.entry_bp)
            if t + 1 < hi:
                ok &= p.readable[t + 1]
            for j in [
                j for j in np.argsort(-np.nan_to_num(dev[t], nan=-1e9)) if ok[j] and j not in held
            ][:free]:
                q = (equity + pnl) / rule.slots / 2 / p.spot[t, j]
                pnl -= q * p.spot[t, j] * cost_side
                held[j] = (q, t)
                trades += 1
        equity += pnl
        month_pnl[key] = month_pnl.get(key, 0.0) + pnl
    for j, (q, _) in held.items():  # closed at the end
        equity -= q * p.spot[hi - 1, j] * cost_side
        month_pnl[key] -= q * p.spot[hi - 1, j] * cost_side
    rows = [{"month": m, "return_pct": month_pnl[m] / month_start[m] * 100} for m in month_pnl]
    out = pd.DataFrame(rows)
    out.attrs["final_equity"] = equity
    out.attrs["trades"] = trades
    return out


def shifted(dev: np.ndarray, lo: int, hi: int, seed: int) -> np.ndarray:
    """Each coin's deviation over ``lo:hi`` rolled by its own random offset: as many signals, at the wrong hours."""
    rng = np.random.default_rng(seed)
    out = dev.copy()
    span = hi - lo
    for j in range(dev.shape[1]):
        out[lo:hi, j] = np.roll(dev[lo:hi, j], int(rng.integers(span // 20, span - span // 20)))
    return out


__all__ = ["BasisRule", "Panel", "deviation", "panel", "replay", "shifted"]
