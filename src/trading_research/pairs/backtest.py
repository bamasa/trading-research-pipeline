"""A pair traded walk-forward: fitted on a formation window, traded on the next.

The walk
--------
The bars are cut into consecutive windows: ``formation`` bars on which the
hedge (intercept ``alpha``, ratio ``beta``), the residual's mean and its
standard deviation are estimated, then ``trading`` bars on which they are used
unchanged. The walk then steps forward by ``trading`` bars and fits again, so
every bar is traded by exactly one fit, made on bars that all closed before it.
Nothing estimated on a trading window feeds a decision in it.

The rule
--------
At each bar close ``t`` of a trading window the z-score of the residual,
``(log a - alpha - beta * log b - mean) / sd``, is compared with the bands:

* flat, ``z <= -entry``: long the spread (long ``a``, short ``beta`` of ``b``);
  ``z >= entry``: short it. No entry beyond the stop band, none on the last bar
  of a window, and none in a window whose formation p-value is above
  ``gate_p`` when a gate is set;
* in a position: leave when the residual is back inside ``exit`` (``exit``),
  when it has run past ``stop`` against the position (``stop``), when it has
  been held ``max_hold`` bars (``time``), or on the window's last bar
  (``window_end``). After a stop, a time exit or a window end the rule waits for
  ``|z|`` to come back inside ``entry`` before it enters again.

Fills
-----
A decision taken on the close of bar ``t`` is filled at the open of bar
``t + 1``, both legs. The close that triggered the decision is never the
price paid, so the rule cannot buy the very move it reacted to.

Accounting
----------
In bp of the first leg's notional; the second leg's notional is ``|beta|`` of
it. The gross of a trade is ``side * (r_a - beta * r_b)``, with ``r`` each leg's
simple return from entry fill to exit fill. Each of the four fills pays the
taker fee and slippage on its leg's notional (:mod:`.costs`): ``2 * c`` for leg
``a`` and ``2 * c * |beta|`` for leg ``b``. A trade's day is the UTC day of its
exit fill.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from trading_research.pairs.costs import LegCosts
from trading_research.pairs.stats import DEFAULT_LAGS, PairsError, engle_granger, half_life


@dataclass(frozen=True)
class PairRule:
    """The walk and the bands. Lengths are in bars."""

    formation: int = 720
    trading: int = 168
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 4.0
    max_hold: int = 72
    #: Trade a window only when its formation Engle-Granger p-value is at most this.
    gate_p: float | None = None
    lags: int = DEFAULT_LAGS

    def __post_init__(self) -> None:
        if self.formation < 100:
            raise PairsError(f"formation must be at least 100 bars, got {self.formation}")
        if self.trading < 2:
            raise PairsError(f"trading must be at least 2 bars, got {self.trading}")
        if not 0 <= self.exit_z < self.entry_z < self.stop_z:
            raise PairsError(
                f"bands must satisfy 0 <= exit < entry < stop, got "
                f"{self.exit_z}, {self.entry_z}, {self.stop_z}"
            )
        if self.max_hold < 1:
            raise PairsError(f"max_hold must be at least one bar, got {self.max_hold}")


@dataclass
class PairsResult:
    """Trades, days, windows and every decision of one backtest, with its summary."""

    a: str
    b: str
    rule: PairRule
    costs: LegCosts
    trades: pd.DataFrame
    days: pd.DataFrame
    windows: pd.DataFrame
    decisions: pd.DataFrame
    summary: dict[str, Any]


def _windows(n: int, rule: PairRule) -> list[tuple[int, int, int]]:
    """``(formation start, trading start, trading end)``; the end leaves a bar to fill on."""
    out = []
    start = 0
    while start + rule.formation < n - 1:
        begin = start + rule.formation
        out.append((start, begin, min(begin + rule.trading, n - 1)))
        start += rule.trading
    return out


def backtest(
    bars: pd.DataFrame,
    a: str,
    b: str,
    rule: PairRule,
    costs: LegCosts,
) -> PairsResult:
    """Run the walk on aligned bars (:func:`trading_research.pairs.data.align`)."""
    for symbol in (a, b):
        if ("close", symbol) not in bars.columns:
            raise PairsError(f"no bars for {symbol}; have {sorted(set(bars['close'].columns))}")
    if a == b:
        raise PairsError("a pair needs two different symbols")
    la = np.log(bars[("close", a)].to_numpy(dtype=np.float64))
    lb = np.log(bars[("close", b)].to_numpy(dtype=np.float64))
    oa = bars[("open", a)].to_numpy(dtype=np.float64)
    ob = bars[("open", b)].to_numpy(dtype=np.float64)
    stamps = pd.DatetimeIndex(bars.index)
    opens = pd.DatetimeIndex(bars[("open_time", "")])
    n = len(bars)
    plan = _windows(n, rule)
    if not plan:
        raise PairsError(f"{n} bars cannot hold a formation of {rule.formation} and a bar to trade")
    c = costs.per_fill_bp
    trades: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []

    for w, (f0, t0, t1) in enumerate(plan):
        test = engle_granger(la[f0:t0], lb[f0:t0], lags=rule.lags)
        hedge = test.hedge
        residual = hedge.residual(la[f0:t0], lb[f0:t0])
        mean, sd = float(residual.mean()), float(residual.std())
        allowed = rule.gate_p is None or test.p_value <= rule.gate_p
        windows.append(
            {
                "window": w,
                "formation_start": stamps[f0],
                "formation_end": stamps[t0 - 1],
                "trading_start": stamps[t0],
                "trading_end": stamps[t1 - 1],
                "alpha": hedge.alpha,
                "beta": hedge.beta,
                "spread_sd_bp": sd * 1e4,
                "eg_t": test.t,
                "p_value": test.p_value,
                "half_life_bars": half_life(residual),
                "traded": allowed,
            }
        )
        z = (la[t0:t1] - hedge.alpha - hedge.beta * lb[t0:t1] - mean) / sd
        position, entered, armed = 0, -1, True
        z_in = math.nan
        for k, t in enumerate(range(t0, t1)):
            value = float(z[k])
            last = t == t1 - 1
            action = ""
            if position == 0:
                if not armed and abs(value) < rule.entry_z:
                    armed = True
                if armed and allowed and not last and abs(value) < rule.stop_z:
                    if value <= -rule.entry_z:
                        position, action = 1, "enter_long"
                    elif value >= rule.entry_z:
                        position, action = -1, "enter_short"
                    if position:
                        entered, z_in = t, value
            else:
                reason = ""
                if (position == 1 and value >= -rule.exit_z) or (
                    position == -1 and value <= rule.exit_z
                ):
                    reason = "exit"
                elif position * value <= -rule.stop_z:
                    reason = "stop"
                elif t - entered >= rule.max_hold:
                    reason = "time"
                elif last:
                    reason = "window_end"
                if reason:
                    trades.append(
                        _trade(
                            w,
                            a,
                            b,
                            position,
                            hedge.beta,
                            entered,
                            t,
                            z_in,
                            value,
                            reason,
                            oa,
                            ob,
                            stamps,
                            opens,
                            c,
                        )
                    )
                    action, position = reason, 0
                    armed = reason == "exit"
            decisions.append(
                {
                    "timestamp": stamps[t],
                    "window": w,
                    "z": value,
                    "position": position,
                    "action": action,
                }
            )
        windows[-1]["trades"] = sum(1 for tr in trades if tr["window"] == w)

    trade_frame = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    days = daily(trade_frame)
    result = PairsResult(
        a=a,
        b=b,
        rule=rule,
        costs=costs,
        trades=trade_frame,
        days=days,
        windows=pd.DataFrame(windows),
        decisions=pd.DataFrame(decisions),
        summary={},
    )
    result.summary = summarise(result)
    return result


TRADE_COLUMNS = (
    "window",
    "a",
    "b",
    "side",
    "beta",
    "entry_decision",
    "entry_fill",
    "exit_decision",
    "exit_fill",
    "bars_held",
    "z_entry",
    "z_exit",
    "reason",
    "ret_a_bp",
    "ret_b_bp",
    "gross_bp",
    "cost_a_bp",
    "cost_b_bp",
    "cost_bp",
    "net_bp",
    "day",
)


def _trade(
    window: int,
    a: str,
    b: str,
    side: int,
    beta: float,
    entered: int,
    left: int,
    z_in: float,
    z_out: float,
    reason: str,
    oa: np.ndarray,
    ob: np.ndarray,
    stamps: pd.DatetimeIndex,
    opens: pd.DatetimeIndex,
    per_fill_bp: float,
) -> dict[str, Any]:
    """One round trip: decided on closes ``entered`` and ``left``, filled at the next opens."""
    i, j = entered + 1, left + 1
    ret_a = (oa[j] / oa[i] - 1.0) * 1e4
    ret_b = (ob[j] / ob[i] - 1.0) * 1e4
    gross = side * (ret_a - beta * ret_b)
    cost_a = 2.0 * per_fill_bp
    cost_b = 2.0 * per_fill_bp * abs(beta)
    return {
        "window": window,
        "a": a,
        "b": b,
        "side": side,
        "beta": beta,
        "entry_decision": stamps[entered],
        "entry_fill": opens[i],
        "exit_decision": stamps[left],
        "exit_fill": opens[j],
        "bars_held": left - entered,
        "z_entry": z_in,
        "z_exit": z_out,
        "reason": reason,
        "ret_a_bp": ret_a,
        "ret_b_bp": ret_b,
        "gross_bp": gross,
        "cost_a_bp": cost_a,
        "cost_b_bp": cost_b,
        "cost_bp": cost_a + cost_b,
        "net_bp": gross - cost_a - cost_b,
        "day": opens[j].strftime("%Y-%m-%d"),
    }


def daily(trades: pd.DataFrame) -> pd.DataFrame:
    """Per UTC day of exit: trades, and gross, cost and net summed, in bp."""
    if trades.empty:
        return pd.DataFrame(columns=["day", "trades", "gross_bp", "cost_bp", "net_bp"])
    grouped = trades.groupby("day", sort=True)
    out = grouped[["gross_bp", "cost_bp", "net_bp"]].sum()
    out.insert(0, "trades", grouped.size())
    return out.reset_index()


def summarise(result: PairsResult) -> dict[str, Any]:
    """The numbers a page needs: per trade and per day, day-clustered t, headroom."""
    from trading_research.evaluation.significance import SignificanceError, assess
    from trading_research.walkthrough.verdict import headroom

    trades, rule = result.trades, result.rule
    out: dict[str, Any] = {
        "a": result.a,
        "b": result.b,
        **{f"rule_{k}": v for k, v in asdict(rule).items()},
        **result.costs.describe(),
        "windows": len(result.windows),
        "windows_traded": int(result.windows["traded"].sum()) if len(result.windows) else 0,
        "bars_traded": len(result.decisions),
        "trades": len(trades),
    }
    if trades.empty:
        return {**out, "note": "no trade"}
    for column in ("gross_bp", "cost_bp", "net_bp"):
        out[f"{column.removesuffix('_bp')}_per_trade_bp"] = float(trades[column].mean())
        out[f"{column.removesuffix('_bp')}_total_bp"] = float(trades[column].sum())
    out["days_with_trades"] = int(trades["day"].nunique())
    out["exits"] = trades["reason"].value_counts().to_dict()
    out["mean_bars_held"] = float(trades["bars_held"].mean())
    for value in ("net_bp", "gross_bp"):
        name = value.removesuffix("_bp")
        try:
            s = assess(trades, value=value, cluster="day", series=None)
            out[f"{name}_day_t"] = s.cluster_t
            out[f"{name}_days_positive_share"] = s.positive_clusters / s.clusters
        except SignificanceError:
            out[f"{name}_day_t"] = math.nan
            out[f"{name}_days_positive_share"] = math.nan
    room = headroom(trades.assign(filled=True), "taker")
    out["cost_to_break_even_bp_per_trade"] = room["cost_to_break_even_bp"]
    out["cost_cut_to_break_even_bp_per_trade"] = room["cost_cut_to_break_even_bp"]
    legs = float((2.0 * (1.0 + trades["beta"].abs())).sum())
    gross = float(trades["gross_bp"].sum())
    out["cost_to_break_even_bp_per_fill"] = gross / legs if gross > 0 else math.nan
    return out
