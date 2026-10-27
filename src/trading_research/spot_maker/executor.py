"""What both executors share: their settings, their day's run, and the 23:59 flatten.

:class:`MakerSettings` holds one run's coordinates (the registered values by
default; a neighbourhood cell, the 110 ms re-run or a fill-rule variant changes
one or two). :class:`DayRun` is what a run leaves: its fills, its counters, the
realised holds (MK-S) and the touch proxies' widths at its quote instants (the
quoters).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import time

from trading_research.search_engine.walk_tables import clock_ns
from trading_research.spot.fills import PrintTape
from trading_research.spot_maker import registration as rg
from trading_research.spot_maker import resting
from trading_research.spot_maker.money import FillLog

NS_PER_S = rg.NS_PER_S


@dataclass(frozen=True, kw_only=True)
class MakerSettings:
    """One run's coordinates."""

    requote_s: float
    skew_bp: float
    limit_clips: int
    hold_s: float = rg.SIGNAL_HOLD_S
    clip: float = rg.CLIP
    tick: float = rg.TICK
    latency_ns: int = rg.LATENCY_NS
    rule: str = "trade_through"
    posting_first: time = rg.POSTING_FIRST
    posting_last: time = rg.POSTING_LAST
    flatten_at: time = rg.FLATTEN_AT

    def __post_init__(self) -> None:
        if self.rule not in resting.RULES:
            raise ValueError(f"unknown fill rule {self.rule!r}")
        if not (self.requote_s > 0 and self.hold_s > 0 and self.clip > 0 and self.limit_clips > 0):
            raise ValueError("requote, hold, clip and limit must be positive")

    @property
    def hold_ns(self) -> int:
        return round(self.hold_s * NS_PER_S)

    @property
    def requote_ns(self) -> int:
        return round(self.requote_s * NS_PER_S)

    @property
    def limit(self) -> float:
        return self.limit_clips * self.clip

    def first_ns(self, tape: PrintTape) -> int:
        return tape.start_ns + clock_ns(self.posting_first)

    def last_ns(self, tape: PrintTape) -> int:
        return tape.start_ns + clock_ns(self.posting_last)

    def flatten_ns(self, tape: PrintTape) -> int:
        return tape.start_ns + clock_ns(self.flatten_at)


def settings_for(slot: str, **changes: object) -> MakerSettings:
    """A slot's registered settings, with ``changes`` (a cell, a latency, a rule)."""
    spec = rg.SLOTS[slot]
    base = {"requote_s": spec.requote_s, "skew_bp": spec.skew_bp, "limit_clips": spec.limit_clips}
    return MakerSettings(**{**base, **changes})  # type: ignore[arg-type]


@dataclass
class DayRun:
    """One run of one day."""

    log: FillLog = field(default_factory=FillLog)
    counters: Counter[str] = field(default_factory=Counter)
    holds_s: list[float] = field(default_factory=list)
    widths_bp: list[float] = field(default_factory=list)
    posted_btc: float = 0.0


def close_position(run: DayRun, tape: PrintTape, position: float, settings: MakerSettings,
                   kind: str = "flatten") -> int:  # fmt: skip
    """The 23:59 taker flatten of ``position`` (BTC, positive long); its fill time."""
    side = -1 if position > 0 else 1
    ts, price, fell_back = resting.flatten(tape, side, abs(position), settings.flatten_ns(tape),
                                           settings.latency_ns)  # fmt: skip
    run.log.add(ts, side, price, abs(position), False, kind)
    run.counters["flattens"] += 1
    run.counters["flattens_fell_back"] += int(fell_back)
    return ts
