"""Hand-made rows, prints and sessions for the paper-trading tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from trading_research.live.decide import ExecutionSettings, LoadedModel
from trading_research.live.events import BookRow, Event, Print
from trading_research.live.fees import FeeSchedule
from trading_research.live.ledger import Limits, PaperLedger
from trading_research.live.session import Session, SessionConfig

MS = 1_000_000
T0 = 1_767_225_600_000_000_000  # 2026-01-01T00:00:00Z


def row(ms: int, bid: float = 99.9, ask: float = 100.1, size: float = 1.0) -> BookRow:
    tick = 0.1
    bids = tuple(round(bid - i * tick, 1) for i in range(10))
    asks = tuple(round(ask + i * tick, 1) for i in range(10))
    return BookRow(T0 + ms * MS, T0 + ms * MS, ms, bids, (size,) * 10, asks, (size,) * 10)


def trade(ms: int, price: float, size: float, side: int) -> Print:
    return Print(T0 + ms * MS, T0 + ms * MS, price, size, side, f"t{ms}")


class FixedModel:
    """Predicts the same class probabilities everywhere."""

    def __init__(self, proba: tuple[float, float, float]) -> None:
        self.proba = np.asarray(proba, dtype=float)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return np.tile(self.proba, (len(x), 1))


def model(proba: tuple[float, float, float] = (0.1, 0.1, 0.8)) -> LoadedModel:
    return LoadedModel("fixed", FixedModel(proba), [], "0" * 64, None)


def schedule(maker: float = 0.0, taker: float = 10.0) -> FeeSchedule:
    return FeeSchedule("test", "test", "TEST", "given", given=(maker, taker))


def session(
    tmp: Path,
    *,
    mode: str = "taker",
    proba: tuple[float, float, float] = (0.1, 0.1, 0.8),
    every: int = 1_000,
    clip: float = 0.01,
    hold_s: float = 1.0,
    max_position: float = 1.0,
    max_loss: float = 1_000.0,
    stale_after_s: float = 5.0,
    fees: FeeSchedule | None = None,
    flatten_wait_s: float = 30.0,
) -> Session:
    limits = Limits(max_position=max_position, max_loss=max_loss, stale_after_s=stale_after_s)
    config = SessionConfig(
        venue="test",
        market="test",
        symbol="TEST",
        live=False,
        execution=ExecutionSettings(mode=mode, clip=clip, hold_ns=int(hold_s * 1e9)),
        limits=limits,
        every=every,
        run_dir=tmp / "run",
        tick=0.1,
        flatten_wait_s=flatten_wait_s,
    )
    return Session(config, model(proba), PaperLedger(fees or schedule(), limits))


def rows_every(start_ms: int, stop_ms: int, step_ms: int = 100, **kwargs: float) -> list[Event]:
    return [row(ms, **kwargs) for ms in range(start_ms, stop_ms, step_ms)]


def in_time(events: list[Event]) -> list[Event]:
    """Stamp order, a print before a row at the same stamp (it belongs to that row)."""
    return sorted(events, key=lambda e: (e.ts_ns, 0 if isinstance(e, Print) else 1))
