"""The supervisor: several paper sessions run as one portfolio, watched and balanced.

Each strategy is a :class:`trading_research.live.session.Session` exactly as
``trp paper`` runs it — the same features, decisions, fill rules, fees and kill
switch. The supervisor adds what a single session cannot see:

* **Watching.** It sits between a session and its events: an event is handed
  to the session, and once the session has processed it the supervisor reads
  what changed — the book row's mid for the regime monitor, each round trip
  closed (the cash between two flat positions, over the entry's notional) for
  the drift monitor, gaps and the feed's age.
* **States.** Each strategy is ``running``, ``reduced`` or ``halted``. A regime
  alarm reduces it; a drift alarm, a dead feed or the session's own kill
  switch halts it; the portfolio's daily loss or drawdown halts all of them.
  ``halted`` latches: only a new run starts a strategy again, after someone
  has read why it stopped.
* **Balancing.** The balancer (:mod:`trading_research.production.allocator`)
  re-weights the strategies whenever a state changes or a strategy closes
  ``min_trades`` more round trips; the weight becomes a scale on the
  strategy's position limit, enforced on every order the session asks to send.
  A halted strategy gets a limit of nothing, and its session's kill switch is
  latched with the supervisor's reason, so it flattens what it holds.

The supervisor writes ``supervisor.json`` (states, transitions, allocations,
monitors, portfolio equity) as it goes, and :meth:`snapshot` is what the health
endpoint serves. No order leaves the process: the broker is the paper ledger.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from trading_research.live.events import BookRow, Event, Gap
from trading_research.live.ledger import PaperLedger
from trading_research.production.allocator import Allocation, allocate
from trading_research.production.monitors import DriftMonitor, FeedMonitor, RegimeMonitor
from trading_research.production.spec import Deployment, StrategySpec

STATES = ("running", "reduced", "halted")


@dataclass
class SupervisedLedger(PaperLedger):
    """The paper ledger with a position limit the supervisor can move."""

    scale: float = 1.0

    def allows(self, side: int, size: float) -> bool:
        if self.killed is not None:
            return False
        before = self.account.position
        after = before + side * size
        if abs(after) > abs(before) and abs(after) > self.limits.max_position * self.scale * (
            1 + 1e-12
        ):
            self.refused += 1
            return False
        return super().allows(side, size)


@dataclass
class Strand:
    """One strategy under supervision."""

    spec: StrategySpec
    ledger: SupervisedLedger
    drift: DriftMonitor
    regime: RegimeMonitor
    feed: FeedMonitor
    state: str = "running"
    reason: str | None = None
    seen_fills: int = 0
    flat_cash: float = 0.0
    entry_notional: float | None = None
    round_trips: int = 0
    trades_at_balance: int = 0
    last_ts_ns: int | None = None

    def equity(self) -> float:
        if self.ledger.marks:
            return float(self.ledger.marks[-1][3])
        return float(self.ledger.account.cash)


@dataclass
class Supervisor:
    deployment: Deployment
    run_dir: Path
    strands: dict[str, Strand] = field(default_factory=dict)
    transitions: list[dict[str, Any]] = field(default_factory=list)
    allocations: dict[str, Allocation] = field(default_factory=dict)
    day: str | None = None
    day_start_equity: float = 0.0
    peak_equity: float = 0.0
    portfolio_halt: str | None = None
    #: Why and when the balancer last ran.
    balanced: dict[str, Any] = field(default_factory=dict)
    status_every_s: float = 2.0
    started_wall_ns: int = field(default_factory=time.time_ns)
    _lock: threading.RLock = field(default_factory=threading.RLock)
    _written_at: float = 0.0

    # -- set-up ------------------------------------------------------------------

    def add(self, spec: StrategySpec, ledger: SupervisedLedger) -> None:
        monitors = self.deployment.monitors
        self.strands[spec.name] = Strand(
            spec=spec,
            ledger=ledger,
            drift=DriftMonitor(
                mu_bp=spec.expected.net_bp,
                sd_bp=spec.expected.sd_bp,
                k=monitors.drift.k,
                h=monitors.drift.h,
                min_trades=monitors.drift.min_trades,
            ),
            regime=RegimeMonitor(
                warmup=monitors.regime.warmup,
                k=monitors.regime.k,
                h=monitors.regime.h,
                sample_s=monitors.regime.sample_s,
            ),
            feed=FeedMonitor(stale_after_s=monitors.feed.stale_after_s),
        )
        self.rebalance("start", None)

    # -- the watch ---------------------------------------------------------------

    def watch(self, name: str, events: Iterable[Event | None]) -> Iterator[Event | None]:
        """Hand each event to the session, then read what it changed."""
        for event in events:
            yield event
            self.after(name, event)

    def after(self, name: str, event: Event | None) -> None:
        with self._lock:
            strand = self.strands[name]
            wall = time.time_ns()
            if isinstance(event, BookRow):
                strand.last_ts_ns = event.ts_ns
                strand.feed.row(event.ts_ns, wall)
                if strand.regime.observe(event.ts_ns, event.mid):
                    self.transition(strand, "reduced", "regime break", event.ts_ns)
                self._fills(strand)
                self._portfolio(event.ts_ns)
            elif isinstance(event, Gap):
                strand.feed.gap(event.kind)
            elif event is None and strand.feed.stale(wall):
                self.transition(strand, "halted", "feed stale", strand.last_ts_ns or 0)
            if strand.ledger.killed is not None and strand.state != "halted":
                self.transition(
                    strand, "halted", f"kill switch: {strand.ledger.killed}", strand.last_ts_ns or 0
                )
            if time.monotonic() - self._written_at > self.status_every_s:
                self.write()

    def _fills(self, strand: Strand) -> None:
        fills = strand.ledger.fills
        while strand.seen_fills < len(fills):
            fill = fills[strand.seen_fills]
            strand.seen_fills += 1
            notional, position_after, cash_after = fill[5], fill[12], fill[13]
            if strand.entry_notional is None:
                strand.entry_notional = float(notional)
            if position_after == 0:
                net = float(cash_after) - strand.flat_cash
                net_bp = net / strand.entry_notional * 1e4 if strand.entry_notional else 0.0
                strand.flat_cash = float(cash_after)
                strand.entry_notional = None
                strand.round_trips += 1
                if strand.drift.observe(net_bp) and strand.state != "halted":
                    self.transition(strand, "halted", "drift from the backtest", int(fill[0]))
                need = self.deployment.monitors.drift.min_trades
                if strand.round_trips - strand.trades_at_balance >= need:
                    self.rebalance("round trips", int(fill[0]))

    def _portfolio(self, ts_ns: int) -> None:
        equity = sum(s.equity() for s in self.strands.values())
        day = time.strftime("%Y-%m-%d", time.gmtime(ts_ns / 1e9))
        if day != self.day:
            self.day, self.day_start_equity = day, equity
        self.peak_equity = max(self.peak_equity, equity)
        limits = self.deployment.portfolio
        reason = None
        if equity - self.day_start_equity <= -limits.daily_loss:
            reason = "portfolio daily loss"
        elif self.peak_equity - equity >= limits.max_drawdown:
            reason = "portfolio drawdown"
        if reason is not None and self.portfolio_halt is None:
            self.portfolio_halt = reason
            for strand in self.strands.values():
                if strand.state != "halted":
                    self.transition(strand, "halted", reason, ts_ns)

    # -- states and balance ------------------------------------------------------

    def transition(self, strand: Strand, state: str, reason: str, ts_ns: int) -> None:
        if state not in STATES:
            raise ValueError(state)
        if strand.state == "halted" or strand.state == state:
            return
        self.transitions.append(
            {
                "strategy": strand.spec.name,
                "from": strand.state,
                "to": state,
                "reason": reason,
                "ts_ns": ts_ns,
                "wall_ns": time.time_ns(),
            }
        )
        strand.state, strand.reason = state, reason
        if state == "halted":
            strand.ledger.kill(f"supervisor: {reason}", ts_ns)
        self.rebalance(f"{strand.spec.name} {state}", ts_ns)

    def rebalance(self, why: str, ts_ns: int | None) -> None:
        self.balanced = {"why": why, "ts_ns": ts_ns}
        sd = {}
        for name, strand in self.strands.items():
            live = strand.drift.live_sd_bp
            enough = strand.drift.trades >= self.deployment.monitors.drift.min_trades
            sd[name] = live if enough and live > 0 else strand.spec.expected.sd_bp
            strand.trades_at_balance = strand.round_trips
        portfolio = self.deployment.portfolio
        self.allocations = allocate(
            sd,
            {name: s.state for name, s in self.strands.items()},
            capital=self.deployment.capital,
            max_gross=portfolio.max_gross,
            max_weight=portfolio.max_weight,
        )
        for name, allocation in self.allocations.items():
            self.strands[name].ledger.scale = allocation.scale

    # -- what it reports ---------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            wall = time.time_ns()
            strategies = {}
            for name, s in self.strands.items():
                allocation = self.allocations.get(name)
                strategies[name] = {
                    "state": s.state,
                    "reason": s.reason,
                    "symbol": s.spec.symbol,
                    "venue": s.spec.venue,
                    "execution": s.spec.execution,
                    "equity": s.equity(),
                    "position": float(s.ledger.account.position),
                    "fills": len(s.ledger.fills),
                    "round_trips": s.round_trips,
                    "weight": allocation.weight if allocation else 0.0,
                    "limit_scale": s.ledger.scale,
                    "drift": {
                        "statistic": s.drift.statistic,
                        "threshold": s.drift.h,
                        "live_mean_bp": s.drift.mean_bp,
                        "expected_bp": s.drift.mu_bp,
                    },
                    "regime": {
                        "statistic": s.regime.statistic,
                        "threshold": s.regime.h,
                        "ready": s.regime.ready,
                        "alarms": s.regime.alarms,
                    },
                    "feed": {
                        "rows": s.feed.rows,
                        "gaps": s.feed.gaps,
                        "age_s": s.feed.age_s(wall),
                    },
                }
            equity = sum(s.equity() for s in self.strands.values())
            return {
                "deployment": self.deployment.name,
                "deployment_sha256": self.deployment.sha256,
                "capital": self.deployment.capital,
                "equity": equity,
                "day_pnl": equity - self.day_start_equity,
                "drawdown": self.peak_equity - equity,
                "portfolio_halt": self.portfolio_halt,
                "balanced": dict(self.balanced),
                "uptime_s": (wall - self.started_wall_ns) / 1e9,
                "strategies": strategies,
                "transitions": list(self.transitions),
            }

    def ready(self) -> bool:
        """Whether any strategy may still trade on a live feed."""
        with self._lock:
            wall = time.time_ns()
            return any(
                s.state != "halted" and s.feed.rows > 0 and not s.feed.stale(wall)
                for s in self.strands.values()
            )

    def write(self) -> Path:
        self._written_at = time.monotonic()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        target = self.run_dir / "supervisor.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.snapshot(), indent=2, default=float), "utf-8")
        temporary.replace(target)
        return target
