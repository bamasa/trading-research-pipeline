"""Signals to taker round trips, through the pipeline's executor.

Two ways of holding a position, both with the original's 0.01 ETH, both
crossing the spread on every fill (:func:`trading_research.backtest.taker.
execute_day`, 10 ms latency, the next 100 ms snapshot walked):

* ``signal_to_signal`` is what the original's best run used (its
  ``baseline_strategy``, MaxPos = OrderSize = 0.01 ETH): a signal from flat
  opens, the next opposite signal closes to flat (it does not reverse), a
  repeated signal is ignored, and an open position is closed at 23:59;
* ``close_by_time`` opens on a signal from flat and closes after a fixed hold;
  signals while a position is open are ignored.

Funding is left out: the original traded spot, and on 0.01 ETH a settlement
moves about 0.003 USDT. Entries keep the executor's hours (00:01 to 23:58).
"""

from __future__ import annotations

from datetime import date, time

import numpy as np
import pandas as pd

from trading_research.backtest.costs import FeeTier
from trading_research.backtest.taker import TakerBook, TakerDay, TakerSettings, execute_day
from trading_research.market_making.events import NS_PER_S, day_start_ns

CLIP = 0.01


def fee_tier(taker_bp: float) -> FeeTier:
    return FeeTier(f"taker_{taker_bp:g}bp", 0.0, taker_bp)


def taker_book(frame: pd.DataFrame, day: date) -> TakerBook:
    """The executor's book from a gridded day (``ts`` plus level columns)."""
    stamped = frame.assign(timestamp=pd.to_datetime(frame["ts"], utc=True))
    return TakerBook.from_frame(stamped, day, depth=10)


def _ns(day: date, moment: time) -> int:
    seconds = moment.hour * 3600 + moment.minute * 60 + moment.second
    return day_start_ns(day) + seconds * NS_PER_S


def round_trips(
    decision_ns: np.ndarray, side: np.ndarray, day: date, settings: TakerSettings
) -> list[tuple[int, int, int]]:
    """``(entry_ns, side, exit_ns)`` under ``signal_to_signal``; ``exit_ns`` is -1
    for a position still open at the day's end."""
    first, last = _ns(day, settings.no_entry_before), _ns(day, settings.no_entry_after)
    trips: list[tuple[int, int, int]] = []
    open_at, held = -1, 0
    for t, s in zip(decision_ns.tolist(), side.tolist(), strict=True):
        if held == 0:
            if first <= t <= last:
                open_at, held = t, s
        elif s == -held:
            trips.append((open_at, held, t))
            held = 0
    if held != 0:
        trips.append((open_at, held, -1))
    return trips


def signal_to_signal(
    book: TakerBook, decision_ns: np.ndarray, side: np.ndarray, taker_bp: float
) -> pd.DataFrame:
    """The day's round trips, one executor call per trip, its trade rows stacked."""
    settings = TakerSettings(clip=CLIP, fees=fee_tier(taker_bp))
    flatten = _ns(book.day, settings.flatten_at)
    rows = []
    for entry, s, exit_ns in round_trips(decision_ns, side, book.day, settings):
        hold = (exit_ns if exit_ns >= 0 else flatten + 1) - entry
        done = execute_day(book, np.array([entry]), np.array([s]), max(hold, 1), settings)
        rows.append(done.trades)
    if not rows:
        return execute_day(book, np.zeros(0, np.int64), np.zeros(0, np.int64), 1, settings).trades
    return pd.concat(rows, ignore_index=True)


def close_by_time(
    book: TakerBook, decision_ns: np.ndarray, side: np.ndarray, hold_s: float, taker_bp: float
) -> TakerDay:
    settings = TakerSettings(clip=CLIP, fees=fee_tier(taker_bp))
    return execute_day(book, decision_ns, side, int(hold_s * NS_PER_S), settings)
