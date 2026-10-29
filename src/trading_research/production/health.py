"""Health, readiness and metrics over HTTP, from the standard library.

Three paths, for whatever runs the process (a container runtime, a load
balancer, a scraper):

``/health``
    200 while the process serves at all, with the supervisor's snapshot as
    JSON: liveness.
``/ready``
    200 while at least one strategy may trade on a fresh feed, else 503:
    readiness, so a balancer or an orchestrator stops counting on a deployment
    whose strategies have all halted or whose feeds have died.
``/metrics``
    The snapshot in Prometheus's text format: equity, position, fills, round
    trips, weight, state, and each monitor's statistic beside its threshold,
    one series per strategy.

The server runs on its own daemon thread and reads only the snapshot; it can
neither change a state nor send anything to a venue.
"""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

STATE_CODE = {"running": 0, "reduced": 1, "halted": 2}


def _number(value: Any) -> str:
    if value is None:
        return "NaN"
    number = float(value)
    if math.isnan(number):
        return "NaN"
    if math.isinf(number):
        return "+Inf" if number > 0 else "-Inf"
    return repr(number)


def metrics(snapshot: dict[str, Any]) -> str:
    """The snapshot as Prometheus text exposition."""
    lines = []

    def series(name: str, kind: str, doc: str, rows: list[tuple[dict[str, str], Any]]) -> None:
        lines.append(f"# HELP trp_{name} {doc}")
        lines.append(f"# TYPE trp_{name} {kind}")
        for labels, value in rows:
            body = ",".join(f'{k}="{v}"' for k, v in labels.items())
            lines.append(f"trp_{name}{{{body}}} {_number(value)}" if body else
                         f"trp_{name} {_number(value)}")  # fmt: skip

    deployment = {"deployment": str(snapshot.get("deployment", ""))}
    series("portfolio_equity", "gauge", "Net equity of every strategy, quote currency.",
           [(deployment, snapshot.get("equity"))])  # fmt: skip
    series("portfolio_day_pnl", "gauge", "Net since the start of the UTC day.",
           [(deployment, snapshot.get("day_pnl"))])  # fmt: skip
    series("portfolio_drawdown", "gauge", "Equity below its peak.",
           [(deployment, snapshot.get("drawdown"))])  # fmt: skip
    strategies = snapshot.get("strategies", {})

    def each(path: tuple[str, ...]) -> list[tuple[dict[str, str], Any]]:
        rows = []
        for name, body in strategies.items():
            value: Any = body
            for key in path:
                value = value.get(key) if isinstance(value, dict) else None
            rows.append(({**deployment, "strategy": name, "symbol": str(body.get("symbol"))},
                         value))  # fmt: skip
        return rows

    state_rows = [
        ({**deployment, "strategy": name}, STATE_CODE.get(body.get("state"), -1))
        for name, body in strategies.items()
    ]
    series("strategy_state", "gauge", "0 running, 1 reduced, 2 halted.", state_rows)
    series("strategy_equity", "gauge", "Net equity, quote currency.", each(("equity",)))
    series("strategy_position", "gauge", "Position, base units.", each(("position",)))
    series("strategy_fills_total", "counter", "Fills booked.", each(("fills",)))
    series("strategy_round_trips_total", "counter", "Round trips closed.", each(("round_trips",)))
    series("strategy_weight", "gauge", "Share of capital the balancer gives it.",
           each(("weight",)))  # fmt: skip
    series("drift_statistic", "gauge", "Drift CUSUM; alarms above its threshold.",
           each(("drift", "statistic")))  # fmt: skip
    series("drift_threshold", "gauge", "Drift CUSUM threshold.", each(("drift", "threshold")))
    series("regime_statistic", "gauge", "Regime CUSUM; alarms above its threshold.",
           each(("regime", "statistic")))  # fmt: skip
    series("regime_threshold", "gauge", "Regime CUSUM threshold.", each(("regime", "threshold")))
    series("feed_age_seconds", "gauge", "Seconds since the last book row.",
           each(("feed", "age_s")))  # fmt: skip
    series("feed_gaps_total", "counter", "Gaps the stream reported.", each(("feed", "gaps")))
    return "\n".join(lines) + "\n"


class HealthServer:
    """Serve ``/health``, ``/ready`` and ``/metrics`` from two callables."""

    def __init__(
        self,
        snapshot: Callable[[], dict[str, Any]],
        ready: Callable[[], bool],
        host: str = "127.0.0.1",
        port: int = 9108,
    ) -> None:
        self._snapshot = snapshot
        self._ready = ready
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                outer._answer(self)

            def log_message(self, *args: Any) -> None:
                del args  # requests are not logged: the snapshot is the record

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def port(self) -> int:
        return int(self.server.server_address[1])

    def start(self) -> HealthServer:
        self.thread.start()
        return self

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def _answer(self, request: BaseHTTPRequestHandler) -> None:
        path = request.path.split("?", 1)[0]
        if path == "/health":
            code, kind, body = 200, "application/json", json.dumps(self._snapshot(), default=float)
        elif path == "/ready":
            ok = self._ready()
            code, kind, body = (200 if ok else 503), "application/json", json.dumps({"ready": ok})
        elif path == "/metrics":
            code, kind, body = 200, "text/plain; version=0.0.4", metrics(self._snapshot())
        else:
            code, kind, body = 404, "text/plain", "not found\n"
        data = body.encode("utf-8")
        request.send_response(code)
        request.send_header("Content-Type", kind)
        request.send_header("Content-Length", str(len(data)))
        request.end_headers()
        request.wfile.write(data)
