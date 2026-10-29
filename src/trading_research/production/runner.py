"""Run a deployment: one paper session per strategy, one supervisor, one health server.

A strategy's session is built as ``trp paper`` builds one — the same model
loader, fee schedule, execution settings and limits — with the supervised
ledger in place of the plain one. Each session runs on its own thread over its
own events (a public stream, or a stored day replayed), every event passing
through the supervisor's watch; the health server reads the supervisor's
snapshot while they run. At the end every session writes its usual paper page
under ``<run>/<strategy>/`` and the supervisor writes ``supervisor.json``.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from trading_research.live.events import Event
from trading_research.live.session import Session, SessionConfig
from trading_research.production.health import HealthServer
from trading_research.production.spec import Deployment, DeploymentError, StrategySpec, check_pins
from trading_research.production.supervisor import SupervisedLedger, Supervisor


@dataclass
class RunResult:
    run_dir: Path
    snapshot: dict[str, Any]
    summaries: dict[str, dict[str, Any]] = field(default_factory=dict)
    pages: dict[str, Path] = field(default_factory=dict)
    port: int | None = None


def _schedule(spec: StrategySpec) -> Any:
    from trading_research.live.fees import schedule_for

    given = None
    if spec.fee_bp:
        maker, taker = (float(x) for x in spec.fee_bp.split(","))
        given = (maker, taker)
    as_of = int(pd.Timestamp(spec.fee_asof, tz="UTC").value) if spec.fee_asof else None
    return schedule_for(
        spec.venue, spec.market, spec.symbol, spec.fee_tier, given=given, as_of_ns=as_of
    )


def build_session(
    spec: StrategySpec, deployment: Deployment, run_dir: Path, *, live: bool
) -> tuple[Session, SupervisedLedger]:
    """The strategy's paper session with a ledger the supervisor can scale and stop."""
    from trading_research.live.decide import ExecutionSettings, load_model
    from trading_research.live.ledger import Limits

    loaded = load_model(spec.model)
    latency = int(spec.latency_ms * 1e6)
    settings = ExecutionSettings(
        mode=spec.execution,
        clip=spec.clip,
        latency_ns=latency,
        cancel_latency_ns=latency,
        hold_ns=int(spec.hold_s * 1e9),
    )
    limits = Limits(
        max_position=spec.clip * spec.max_position_clips,
        max_loss=spec.max_loss if spec.max_loss is not None else deployment.portfolio.daily_loss,
        stale_after_s=deployment.monitors.feed.stale_after_s,
        kill_on_gap=True,
    )
    config = SessionConfig(
        venue=spec.venue,
        market=spec.market,
        symbol=spec.symbol,
        live=live,
        execution=settings,
        limits=limits,
        min_confidence=spec.min_confidence,
        every=spec.every,
        run_dir=run_dir / spec.name,
    )
    ledger = SupervisedLedger(_schedule(spec), limits)
    return Session(config, loaded, ledger), ledger


def run(
    deployment: Deployment,
    run_dir: Path,
    *,
    days: dict[str, str] | None = None,
    minutes: float | None = None,
    speed: float = 0.0,
    root: Path = Path("data/paper"),
    serve: bool = True,
    port: int | None = None,
    command: list[str] | None = None,
) -> RunResult:
    """Replay ``days`` (strategy name -> stored day) or run live for ``minutes``."""
    from trading_research.live.page import write_run

    if (days is None) == (minutes is None):
        raise DeploymentError("give either the stored days to replay or the minutes to run live")
    problems = check_pins(deployment)
    if problems:
        raise DeploymentError("model pins do not hold: " + "; ".join(problems))
    live = minutes is not None
    sources: dict[str, Iterable[Event | None]] = {}
    described: dict[str, dict[str, Any]] = {}
    if days is not None:
        from trading_research.live.replay import open_day, paced

        missing = sorted({s.name for s in deployment.strategies} - set(days))
        if missing:
            raise DeploymentError(f"no stored day for {', '.join(missing)}")
        for spec in deployment.strategies:
            stored = open_day(
                days[spec.name], venue=spec.venue, market=spec.market, symbol=spec.symbol,
                root=root,
            )  # fmt: skip
            sources[spec.name] = paced(stored.events, speed)
            described[spec.name] = {"kind": stored.kind, "path": str(stored.path)}
    else:
        from trading_research.live.guard import refuse_keys
        from trading_research.live.stream import LiveStream, connections

        until = time.time_ns() + int(float(minutes or 0) * 60e9)
        for spec in deployment.strategies:
            plan = connections(spec.venue, spec.market, spec.symbol)
            refuse_keys({"strategy": spec.name}, [c.url for c in plan])
            stream = LiveStream(spec.venue, spec.market, spec.symbol)
            sources[spec.name] = stream.events(until_ns=until)
            described[spec.name] = {"kind": "live", "urls": [c.url for c in plan]}

    supervisor = Supervisor(deployment, run_dir)
    sessions: dict[str, Session] = {}
    for spec in deployment.strategies:
        session, ledger = build_session(spec, deployment, run_dir, live=live)
        sessions[spec.name] = session
        supervisor.add(spec, ledger)
    supervisor.write()

    server = None
    if serve:
        server = HealthServer(
            supervisor.snapshot,
            supervisor.ready,
            host=deployment.host,
            port=deployment.port if port is None else port,
        ).start()
    result = RunResult(run_dir=run_dir, snapshot={}, port=server.port if server else None)
    errors: dict[str, BaseException] = {}

    def one(name: str) -> None:
        try:
            result.summaries[name] = sessions[name].run(supervisor.watch(name, sources[name]))
        except BaseException as exc:  # reported below, once the other strategies finish
            errors[name] = exc
            supervisor.transition(
                supervisor.strands[name],
                "halted",
                f"error: {exc}",
                supervisor.strands[name].last_ts_ns or 0,
            )

    threads = [threading.Thread(target=one, args=(name,), name=f"strategy-{name}")
               for name in sessions]  # fmt: skip
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        if server is not None:
            server.stop()
    for name, summary in result.summaries.items():
        result.pages[name] = write_run(
            sessions[name], summary, command=command or [], source=described[name]
        )
    supervisor.write()
    result.snapshot = supervisor.snapshot()
    if errors:
        first = next(iter(errors.values()))
        raise RuntimeError(f"strategy {next(iter(errors))} failed: {first}") from first
    return result
