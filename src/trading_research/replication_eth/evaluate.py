"""From a span's probabilities to per-day results for one run at each fee.

A span is consecutive stored days concatenated, as the original concatenated
its test days; the strategy inputs are filled across it as the original filled
them across its days. A row trades only if it passes the volatility filter and
ends a window (its probabilities are finite).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from trading_research.market_making.events import NS_PER_S, day_start_ns
from trading_research.replication_eth.days import MIN_VOLATILITY
from trading_research.replication_eth.execution import close_by_time, signal_to_signal, taker_book
from trading_research.replication_eth.gate import (
    FLAT,
    UP,
    GateConfig,
    gate,
    one_per_event,
    strategy_inputs,
)
from trading_research.replication_eth.sweep import SIGNAL_TO_SIGNAL, Run

#: Fees per side the results are reported at: none, the original's effective
#: fee on turnover, and the venue's base taker fee.
FEES_BP = (0.0, 2.2, 5.5)


def signals(
    span: dict[str, np.ndarray], probs: np.ndarray, cfg: GateConfig
) -> tuple[np.ndarray, np.ndarray]:
    """``(decision_ns, side)`` of the span's traded rows, in time order."""
    rv_bp, spread_bp = strategy_inputs(
        span["rv_50"], span["mid_mean_50"], span["spread_rolling_10"]
    )
    valid = (span["volatility"] >= MIN_VOLATILITY) & np.isfinite(probs[:, 0])
    rows = np.flatnonzero(valid)
    labels = gate(probs[rows], rv_bp[rows], spread_bp[rows], cfg)
    labels = one_per_event(labels, span["ts"][rows] // NS_PER_S, cfg)
    traded = labels != FLAT
    return span["ts"][rows][traded], np.where(labels[traded] == UP, 1, -1)


def run_day(
    book_frame: pd.DataFrame,
    day: date,
    decision_ns: np.ndarray,
    side: np.ndarray,
    strategy: str,
    fee_bp: float,
) -> pd.DataFrame:
    book = taker_book(book_frame, day)
    if strategy == SIGNAL_TO_SIGNAL:
        return signal_to_signal(book, decision_ns, side, fee_bp)
    hold_s = float(strategy.removeprefix("close_by_time_").removesuffix("s"))
    return close_by_time(book, decision_ns, side, hold_s, fee_bp).trades


def day_rows(
    run: Run,
    days: list[date],
    books: dict[date, pd.DataFrame],
    decision_ns: np.ndarray,
    side: np.ndarray,
) -> list[dict[str, object]]:
    """One row per day and fee: signals, round trips, gross, fees, net, turnover."""
    out = []
    for day in days:
        start = day_start_ns(day)
        mine = (decision_ns >= start) & (decision_ns < start + 86_400 * NS_PER_S)
        for fee in FEES_BP:
            trades = run_day(books[day], day, decision_ns[mine], side[mine], run.strategy, fee)
            turnover = float(
                ((trades["entry_price"] + trades["exit_price"]) * trades["size"]).sum()
            )
            out.append(
                {
                    "run": run.gate.name,
                    "strategy": run.strategy,
                    "source": run.source,
                    "day": day.isoformat(),
                    "fee_bp_per_side": fee,
                    "signals": int(mine.sum()),
                    "round_trips": len(trades),
                    "fills": 2 * len(trades),
                    "gross_usdt": float(trades["gross"].sum()),
                    "fees_usdt": float(trades["fees"].sum()),
                    "net_usdt": float(trades["net"].sum()),
                    "turnover_usdt": turnover,
                    "mean_hold_s": float(
                        (
                            (trades["exit_snapshot_ns"] - trades["entry_snapshot_ns"]) / NS_PER_S
                        ).mean()
                    )
                    if len(trades)
                    else float("nan"),
                }
            )
    return out
