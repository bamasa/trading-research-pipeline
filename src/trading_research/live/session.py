"""One paper session: events in; features, decisions, orders, fills and marks out.

The same loop runs a live stream and a replayed day. For every event:

* a **print** goes to the feature buffer and to the paper orders (taker sweeps,
  trade-through fills); a print stamped at or before a row already evaluated
  is counted as late;
* a **row** goes to the feature buffer, advances the orders (cancels landing,
  hold timers, the quoter), marks the account at the mid and, at a boundary,
  scores the model and hands its decision to the execution;
* a **gap** is logged and, with ``kill_on_gap``, latches the kill switch.

A boundary is every ``every``-th row of a UTC day, counted as ``prepare``
counts the rows it keeps, so a model trained on prepared rows decides where
it was trained to. Event time drives every rule; in a live session an order's
arrival also carries the time the loop took since the row arrived and the
feed's measured delay (the median of receipt minus venue stamp over recent
prints), so a live decision is never credited with arriving before it could
have.

The run directory gets ``status.json`` every few seconds while running and,
at the end, the tables, the manifest and the page
(:mod:`trading_research.live.page`).
"""

from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_research.features import REGISTRY
from trading_research.live.decide import Execution, ExecutionSettings, LoadedModel, decide
from trading_research.live.events import BookRow, Event, Gap, Print
from trading_research.live.features import (
    NS_PER_DAY,
    FeatureSet,
    PreparedSet,
    RegistrySet,
    StreamingFeatures,
)
from trading_research.live.ledger import Limits, PaperLedger


@dataclass
class SessionConfig:
    """Everything a session depends on besides its events."""

    venue: str
    market: str
    symbol: str
    live: bool
    execution: ExecutionSettings
    limits: Limits
    min_confidence: float = 0.5
    every: int = 50
    run_dir: Path = Path("runs/paper/run")
    status_every_s: float = 5.0
    #: After the kill switch latches, how long to wait for a print to flatten against.
    flatten_wait_s: float = 30.0
    tick: float | None = None


def feature_set_for(model: LoadedModel, symbol: str) -> FeatureSet | None:
    """The set that computes a model's features: the registry when it reads only
    registry features no generated column shadows, else ``prepare``'s frame."""
    if not model.features:
        return None
    from trading_research.features.generated import base_quantities

    shadowed = set(base_quantities(_one_row()).columns)
    registry = set(REGISTRY.names("book")) - shadowed
    if set(model.features) <= registry:
        return RegistrySet(names=list(model.features))
    return PreparedSet(names=list(model.features), symbol=symbol)


def _one_row() -> pd.DataFrame:
    data: dict[str, Any] = {"timestamp": pd.to_datetime([0], utc=True)}
    for side in ("bid", "ask"):
        data[f"{side}_price_0"] = [1.0]
        data[f"{side}_size_0"] = [1.0]
    return pd.DataFrame(data)


def _tick(row: BookRow) -> float:
    from trading_research.market_making.events import OffGridPrice, derive_tick

    try:
        return derive_tick(np.asarray([*row.bid_px, *row.ask_px]))
    except OffGridPrice:
        return float(min(np.diff(sorted({*row.bid_px, *row.ask_px}))))


@dataclass
class Session:
    config: SessionConfig
    model: LoadedModel
    ledger: PaperLedger
    gaps: list[Gap] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    #: (decision row stamp, receipt, side, p_sell, p_hold, p_buy, outcome, features ms, model ms, loop lag ms)
    decisions: list[tuple[Any, ...]] = field(default_factory=list)
    feed_ms: deque[float] = field(default_factory=lambda: deque(maxlen=2000))
    row_feed_ms: list[float] = field(default_factory=list)
    print_feed_ms: list[float] = field(default_factory=list)
    lag_ms: list[float] = field(default_factory=list)
    first_ns: int | None = None
    last_ns: int | None = None
    started_wall_ns: int = 0
    ended_wall_ns: int = 0

    def __post_init__(self) -> None:
        self.feature_set = feature_set_for(self.model, self.config.symbol)
        self.features = StreamingFeatures(
            [self.feature_set] if self.feature_set is not None else [], symbol=self.config.symbol
        )
        self.execution: Execution | None = None
        self.last_row: BookRow | None = None
        self.last_evaluated_ns = -1
        self.per_day: dict[int, int] = {}
        self._status_at = 0.0
        self._killed_wall: float | None = None
        self.rows = 0
        self.prints = 0

    def _count(self, what: str, n: int = 1) -> None:
        self.counts[what] = self.counts.get(what, 0) + n

    def _delay_ns(self, row: BookRow, now_wall: int) -> int:
        if not self.config.live:
            return 0
        offset = float(np.median(self.feed_ms)) if self.feed_ms else 0.0
        return max(0, now_wall - row.recv_ns) + int(max(offset, 0.0) * 1e6)

    def run(self, events: Iterable[Event | None]) -> dict[str, Any]:
        self.started_wall_ns = time.time_ns()
        self.config.run_dir.mkdir(parents=True, exist_ok=True)
        try:
            for event in events:
                if event is None:
                    self._heartbeat()
                elif isinstance(event, Print):
                    self._on_print(event)
                elif isinstance(event, BookRow):
                    self._on_row(event)
                elif isinstance(event, Gap):
                    self._on_gap(event)
                if (
                    self.config.live
                    and time.monotonic() - self._status_at > self.config.status_every_s
                ):
                    self.write_status("running")
                if self._done_after_kill():
                    break
        except KeyboardInterrupt:
            self._count("interrupted")
        self.ended_wall_ns = time.time_ns()
        if self.execution is not None and self.last_ns is not None:
            self.execution.finish(self.last_ns)
            if self.last_row is not None:
                self.ledger.mark(self.last_ns, self.last_row.mid)
        return self.summary()

    def _done_after_kill(self) -> bool:
        """Once killed: stop when flat, or ``flatten_wait_s`` after the kill if no print came."""
        if self.ledger.killed is None or self.execution is None:
            return False
        flat = self.ledger.account.position == 0 and not self.execution.working()
        waited = (self.last_ns or 0) - (
            self.ledger.killed_ns or 0
        ) > self.config.flatten_wait_s * 1e9
        if self.config.live and self._killed_wall is None:
            self._killed_wall = time.monotonic()
        if self.config.live and self._killed_wall is not None:
            waited = waited or time.monotonic() - self._killed_wall > self.config.flatten_wait_s
        return flat or waited

    # -- events ----------------------------------------------------------------

    def _heartbeat(self) -> None:
        if self.config.live and self.last_row is not None:
            stale = (time.time_ns() - self.last_row.recv_ns) / 1e9
            if stale > self.config.limits.stale_after_s:
                self.ledger.kill("stale_data", self.last_row.ts_ns)

    def _on_gap(self, gap: Gap) -> None:
        self.gaps.append(gap)
        self._count(f"gap_{gap.kind}")
        if self.config.limits.kill_on_gap and gap.kind in (
            "sequence",
            "trade_sequence",
            "reconnect",
        ):
            self.ledger.kill("sequence_gap" if gap.kind != "reconnect" else "reconnect", gap.ts_ns)

    def _on_print(self, p: Print) -> None:
        self.prints += 1
        if p.ts_ns <= self.last_evaluated_ns:
            self._count("late_prints")
        self.features.add_print(p)
        if self.config.live:
            delay = (p.recv_ns - p.ts_ns) / 1e6
            self.feed_ms.append(delay)
            self.print_feed_ms.append(delay)
        if self.execution is not None:
            self.execution.on_print(p)
        self._extend(p.ts_ns)

    def _extend(self, ts: int) -> None:
        self.first_ns = ts if self.first_ns is None else self.first_ns
        self.last_ns = ts if self.last_ns is None else max(self.last_ns, ts)

    def _on_row(self, row: BookRow) -> None:
        wall = time.time_ns()
        self.rows += 1
        if self.execution is None:
            tick = self.config.tick or _tick(row)
            self.execution = Execution(self.config.execution, self.ledger, tick)
        if self.last_row is not None:
            pause = (row.ts_ns - self.last_row.ts_ns) / 1e9
            if pause > self.config.limits.stale_after_s:
                self._count("stale_pauses")
                self.ledger.kill("stale_data", row.ts_ns)
        if self.config.live:
            self.row_feed_ms.append((row.recv_ns - row.ts_ns) / 1e6)
            self.lag_ms.append((wall - row.recv_ns) / 1e6)
        index = self.features.add_row(row)
        delay = self._delay_ns(row, wall)
        self.execution.on_row(row, delay)
        self.ledger.mark(row.ts_ns, row.mid)
        self.last_row = row
        self._extend(row.ts_ns)
        day = row.ts_ns // NS_PER_DAY
        in_day = self.per_day.get(day, 0)
        self.per_day[day] = in_day + 1
        if in_day % self.config.every == 0:
            self._decide(row, index, wall)

    def _decide(self, row: BookRow, index: int, wall: int) -> None:
        assert self.execution is not None
        self.last_evaluated_ns = row.ts_ns
        t0 = time.perf_counter()
        values = (
            self.features.compute(self.feature_set, index)
            if self.feature_set is not None
            else pd.Series(dtype=float)
        )
        t1 = time.perf_counter()
        proba = self.model.proba(values)
        t2 = time.perf_counter()
        if proba is None:
            side, outcome = 0, "incomplete_features"
            proba = np.full(3, np.nan)
        else:
            side = decide(proba, self.config.min_confidence)
            outcome = self.execution.on_decision(side, row, self._delay_ns(row, time.time_ns()))
        lag = (
            (time.time_ns() - row.recv_ns) / 1e6
            if self.config.live
            else (time.time_ns() - wall) / 1e6
        )
        self.decisions.append(
            (
                row.ts_ns,
                row.recv_ns,
                side,
                *proba.tolist(),
                outcome,
                (t1 - t0) * 1e3,
                (t2 - t1) * 1e3,
                lag,
            )
        )
        self._count(f"decision_{outcome}")

    # -- outputs ---------------------------------------------------------------

    def decisions_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.decisions,
            columns=[
                "ts_ns", "recv_ns", "side", "p_sell", "p_hold", "p_buy", "outcome",
                "features_ms", "model_ms", "decision_lag_ms",
            ],
        )  # fmt: skip

    def gaps_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [(g.ts_ns, g.recv_ns, g.kind, g.detail) for g in self.gaps],
            columns=["ts_ns", "recv_ns", "kind", "detail"],
        )

    def latency(self) -> dict[str, Any]:
        def stats(values: Iterable[float]) -> dict[str, float] | None:
            array = np.asarray(list(values), dtype=float)
            array = array[np.isfinite(array)]
            if not len(array):
                return None
            return {
                "n": len(array),
                "p50": float(np.percentile(array, 50)),
                "p90": float(np.percentile(array, 90)),
                "p99": float(np.percentile(array, 99)),
                "max": float(array.max()),
            }

        decisions = self.decisions_frame()
        return {
            "decision_lag_ms": stats(decisions["decision_lag_ms"]),
            "features_ms": stats(decisions["features_ms"]),
            "model_ms": stats(decisions["model_ms"]),
            "row_loop_lag_ms": stats(self.lag_ms),
            "row_feed_ms": stats(self.row_feed_ms),
            "print_feed_ms": stats(self.print_feed_ms),
        }

    def summary(self) -> dict[str, Any]:
        ex = self.execution
        span_s = ((self.last_ns or 0) - (self.first_ns or 0)) / 1e9
        wall_s = (self.ended_wall_ns - self.started_wall_ns) / 1e9 if self.ended_wall_ns else 0.0
        decisions = self.decisions_frame()
        out: dict[str, Any] = {
            "rows": self.rows,
            "prints": self.prints,
            "span_s": span_s,
            "wall_s": wall_s,
            "boundaries": len(decisions),
            "decisions_buy": int((decisions["side"] == 1).sum()),
            "decisions_sell": int((decisions["side"] == -1).sum()),
            "decision_outcomes": decisions["outcome"].value_counts().to_dict(),
            "gaps": len(self.gaps),
            "gap_kinds": self.gaps_frame()["kind"].value_counts().to_dict(),
            "counts": dict(self.counts),
            "pnl": self.ledger.summary(),
            "fill_rate": ex.fill_rate() if ex is not None else {},
            "execution_counts": dict(ex.counts) if ex is not None else {},
            "latency": self.latency(),
        }
        return out

    def write_status(self, state: str) -> None:
        self._status_at = time.monotonic()
        pnl = self.ledger.summary()
        status = {
            "state": state,
            "updated_ns": time.time_ns(),
            "rows": self.rows,
            "prints": self.prints,
            "boundaries": len(self.decisions),
            "gaps": len(self.gaps),
            "position": pnl["position"],
            "net": pnl["net"],
            "gross": pnl["gross"],
            "fees": pnl["fees"],
            "killed": pnl["killed"],
        }
        path = self.config.run_dir / "status.json"
        path.write_text(json.dumps(status, indent=2, default=str) + "\n", encoding="utf-8")
