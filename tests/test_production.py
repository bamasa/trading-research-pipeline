"""Production: the deployment file, the balancer, the monitors, the supervisor, health and the commands."""

from __future__ import annotations

import hashlib
import json
import pickle
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from trading_research.cli import app
from trading_research.live.fees import schedule_for
from trading_research.live.ledger import Limits
from trading_research.production.allocator import allocate
from trading_research.production.health import HealthServer, metrics
from trading_research.production.monitors import DriftMonitor, FeedMonitor, RegimeMonitor
from trading_research.production.spec import DeploymentError, check_pins, parse
from trading_research.production.supervisor import SupervisedLedger, Supervisor

runner = CliRunner()


def deployment(**changes: Any) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "name": "test",
        "capital": 1000,
        "portfolio": {"max_gross": 0.8, "max_weight": 0.6, "daily_loss": 40, "max_drawdown": 80},
        "monitors": {"drift": {"min_trades": 3}, "regime": {"warmup": 50, "sample_s": 0.5}},
        "strategies": [
            {"name": "a", "venue": "bybit", "market": "linear", "symbol": "BTCUSDT",
             "model": "always_hold", "execution": "taker", "clip": 0.001,
             "fee_tier": "base", "fee_asof": "2026-09-15", "expected": {"net_bp": 5, "sd_bp": 20}},
            {"name": "b", "venue": "bybit", "market": "linear", "symbol": "ETHUSDT",
             "model": "always_hold", "execution": "taker", "clip": 0.01,
             "fee_tier": "base", "fee_asof": "2026-09-15", "expected": {"net_bp": 0, "sd_bp": 10}},
        ],
    }  # fmt: skip
    raw.update(changes)
    return raw


# -- the deployment file -----------------------------------------------------------


def test_a_deployment_parses_with_its_defaults() -> None:
    spec = parse(deployment())
    assert [s.name for s in spec.strategies] == ["a", "b"]
    assert spec.monitors.drift.min_trades == 3 and spec.monitors.drift.h == 5.0
    assert spec.strategy("b").expected.sd_bp == 10 and spec.port == 9108


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"capital": 0}, "capital"),
        ({"strategies": []}, "non-empty"),
        ({"surprise": 1}, "unknown top-level"),
        ({"portfolio": {"max_weight": 1.5}}, "shares"),
        ({"monitors": {"vibes": {}}}, "unknown monitor"),
    ],
)
def test_a_deployment_refuses_what_it_cannot_run(change: dict[str, Any], message: str) -> None:
    with pytest.raises(DeploymentError, match=message):
        parse(deployment(**change))


def test_a_strategy_needs_its_backtest_reference_and_a_known_execution() -> None:
    raw = deployment()
    del raw["strategies"][0]["expected"]
    with pytest.raises(DeploymentError, match="expected"):
        parse(raw)
    raw = deployment()
    raw["strategies"][1]["execution"] = "market"
    with pytest.raises(DeploymentError, match="execution"):
        parse(raw)
    raw = deployment()
    raw["strategies"][1]["name"] = "a"
    with pytest.raises(DeploymentError, match="unique"):
        parse(raw)


def test_a_deployment_naming_a_credential_is_refused() -> None:
    raw = deployment()
    raw["strategies"][0]["symbol"] = "${BYBIT_API_KEY}"
    with pytest.raises(DeploymentError, match="public market data only"):
        parse(raw)


def test_a_model_by_path_must_be_the_file_it_pins(tmp_path: Path) -> None:
    from trading_research.models.base import AlwaysHold

    path = tmp_path / "model.pkl"
    path.write_bytes(pickle.dumps({"model": AlwaysHold(), "features": []}))
    raw = deployment()
    raw["strategies"][0]["model"] = str(path)
    assert "no model_sha256 pin" in check_pins(parse(raw))[0]
    raw["strategies"][0]["model_sha256"] = "0" * 64
    assert "the deployment pins" in check_pins(parse(raw))[0]
    raw["strategies"][0]["model_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert check_pins(parse(raw)) == []


# -- the balancer ------------------------------------------------------------------


def test_weights_are_inverse_to_volatility_and_sum_to_the_gross() -> None:
    plan = allocate({"a": 10, "b": 20}, {}, capital=1000, max_gross=0.9, max_weight=1.0)
    assert plan["a"].weight == pytest.approx(0.6) and plan["b"].weight == pytest.approx(0.3)
    assert plan["a"].capital == pytest.approx(600)
    assert plan["a"].scale == pytest.approx(0.6 / 0.45)


def test_the_cap_is_water_filled_and_a_capped_remainder_stays_undeployed() -> None:
    plan = allocate({"a": 1, "b": 10, "c": 10}, {}, capital=1, max_gross=0.9, max_weight=0.5)
    assert plan["a"].weight == pytest.approx(0.5)
    assert plan["b"].weight == pytest.approx(0.2) and plan["c"].weight == pytest.approx(0.2)
    both = allocate({"a": 1, "b": 1}, {}, capital=1, max_gross=1.0, max_weight=0.3)
    assert sum(x.weight for x in both.values()) == pytest.approx(0.6)


def test_a_reduced_strategy_counts_half_and_a_halted_one_nothing() -> None:
    plan = allocate({"a": 10, "b": 10, "c": 10}, {"b": "reduced", "c": "halted"},
                    capital=1, max_gross=0.9, max_weight=1.0)  # fmt: skip
    assert plan["a"].weight == pytest.approx(0.6) and plan["b"].weight == pytest.approx(0.3)
    assert plan["c"].weight == 0 and plan["c"].scale == 0
    with pytest.raises(ValueError, match="volatility"):
        allocate({"a": 0}, {}, capital=1, max_gross=1, max_weight=1)


# -- the monitors ------------------------------------------------------------------


def test_drift_stays_quiet_on_the_backtest_and_alarms_on_a_shortfall() -> None:
    rng = np.random.default_rng(3)
    quiet = DriftMonitor(mu_bp=5, sd_bp=20, min_trades=10)
    assert not any(quiet.observe(x) for x in rng.normal(5, 20, 300))
    short = DriftMonitor(mu_bp=5, sd_bp=20, min_trades=10)
    alarms = [short.observe(x) for x in rng.normal(-25, 20, 60)]
    first = alarms.index(True)
    assert 9 <= first < 30  # the tenth round trip at the earliest
    assert short.mean_bp < 0 and short.live_sd_bp == pytest.approx(20, rel=0.3)


def test_drift_waits_for_its_minimum_of_round_trips() -> None:
    monitor = DriftMonitor(mu_bp=5, sd_bp=1, min_trades=5)
    assert [monitor.observe(-100) for _ in range(5)] == [False] * 4 + [True]


def test_the_regime_monitor_flags_a_planted_variance_break_and_not_before() -> None:
    quiet, delays = 0, []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        calm, wild = rng.normal(0, 1e-4, 3000), rng.normal(0, 2e-4, 300)
        prices = 100 * np.exp(np.cumsum(np.concatenate([calm, wild])))
        monitor = RegimeMonitor(warmup=600, sample_s=1.0, h=12.0)
        alarms = [i for i, p in enumerate(prices) if monitor.observe(int(i * 1e9), p)]
        quiet += all(i >= 3000 for i in alarms)
        delays.append(next(i for i in alarms if i >= 3000) - 3000)
        assert monitor.ready
    assert quiet >= 19 and float(np.median(delays)) < 40


def test_the_feed_is_stale_after_its_threshold_of_wall_time() -> None:
    feed = FeedMonitor(stale_after_s=5)
    assert not feed.stale(10**9) and feed.age_s(0) == float("inf")
    feed.row(1, wall_ns=10**9)
    assert not feed.stale(5 * 10**9) and feed.stale(7 * 10**9)
    feed.gap("sequence")
    assert feed.gap_kinds == {"sequence": 1}


# -- the supervisor ----------------------------------------------------------------


def _ledger(max_position: float = 1.0) -> SupervisedLedger:
    schedule = schedule_for("bybit", "linear", "BTCUSDT", given=(2.0, 5.5))
    return SupervisedLedger(schedule, Limits(max_position=max_position, max_loss=1e9))


def test_the_supervised_limit_refuses_growth_beyond_its_scale_and_allows_reduction() -> None:
    ledger = _ledger()
    ledger.scale = 0.5
    assert ledger.allows(1, 0.5) and not ledger.allows(1, 0.6)
    ledger.fill(1, 1, 100.0, 0.5, maker=False, oid=1, reason="entry", path="t")
    assert not ledger.allows(1, 0.1) and ledger.allows(-1, 0.5)
    ledger.scale = 0.0
    assert not ledger.allows(1, 0.1) and ledger.allows(-1, 0.2)


def test_a_halt_latches_kills_the_session_and_takes_its_weight() -> None:
    spec = parse(deployment())
    supervisor = Supervisor(spec, Path("unused"))
    for strategy in spec.strategies:
        supervisor.add(strategy, _ledger())
    a, b = supervisor.strands["a"], supervisor.strands["b"]
    supervisor.transition(a, "reduced", "regime break", 1)
    assert a.ledger.scale < b.ledger.scale
    supervisor.transition(a, "halted", "drift from the backtest", 2)
    supervisor.transition(a, "running", "nothing", 3)
    assert a.state == "halted" and a.ledger.killed == "supervisor: drift from the backtest"
    assert a.ledger.scale == 0 and supervisor.allocations["b"].weight == pytest.approx(0.6)
    assert [t["to"] for t in supervisor.transitions] == ["reduced", "halted"]


def test_round_trips_feed_the_drift_monitor_and_its_alarm_halts() -> None:
    spec = parse(deployment())
    supervisor = Supervisor(spec, Path("unused"))
    supervisor.add(spec.strategies[0], _ledger())
    strand = supervisor.strands["a"]
    for i in range(4):
        strand.ledger.fill(10 * i, 1, 100.0, 0.1, maker=False, oid=2 * i, reason="e", path="t")
        strand.ledger.fill(10 * i + 5, -1, 99.0, 0.1, maker=False, oid=2 * i + 1, reason="x",
                           path="t")  # fmt: skip
    supervisor._fills(strand)
    assert strand.round_trips == 4 and strand.drift.mean_bp < -100
    assert strand.state == "halted" and strand.reason == "drift from the backtest"


def test_the_portfolio_daily_loss_halts_every_strategy() -> None:
    spec = parse(deployment())
    supervisor = Supervisor(spec, Path("unused"))
    for strategy in spec.strategies:
        supervisor.add(strategy, _ledger())
    supervisor._portfolio(0)
    supervisor.strands["a"].ledger.marks.append((1, 100.0, 0.0, -45.0, 0.0))
    supervisor._portfolio(1)
    assert supervisor.portfolio_halt == "portfolio daily loss"
    assert {s.state for s in supervisor.strands.values()} == {"halted"}


# -- health ------------------------------------------------------------------------


def test_health_ready_and_metrics_are_served(tmp_path: Path) -> None:
    spec = parse(deployment())
    supervisor = Supervisor(spec, tmp_path)
    for strategy in spec.strategies:
        supervisor.add(strategy, _ledger())
    server = HealthServer(supervisor.snapshot, supervisor.ready, port=0).start()
    base = f"http://127.0.0.1:{server.port}"
    try:
        health = json.loads(urllib.request.urlopen(f"{base}/health").read())
        assert health["deployment"] == "test" and set(health["strategies"]) == {"a", "b"}
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(f"{base}/ready")
        assert caught.value.code == 503
        text = urllib.request.urlopen(f"{base}/metrics").read().decode()
        assert 'trp_strategy_state{deployment="test",strategy="a"} 0' in text
        assert "# TYPE trp_drift_statistic gauge" in text
    finally:
        server.stop()


def test_metrics_write_nan_and_infinity_as_prometheus_reads_them() -> None:
    text = metrics({"deployment": "d", "equity": float("nan"), "strategies": {
        "a": {"symbol": "X", "state": "halted", "feed": {"age_s": float("inf")}}}})  # fmt: skip
    assert 'trp_portfolio_equity{deployment="d"} NaN' in text
    assert 'trp_feed_age_seconds{deployment="d",strategy="a",symbol="X"} +Inf' in text
    assert 'trp_strategy_state{deployment="d",strategy="a"} 2' in text


# -- the commands ------------------------------------------------------------------


def test_check_run_and_status_on_a_replayed_demo_day(tmp_path: Path) -> None:
    from trading_research.features import build
    from trading_research.models.linear import LogisticBaseline

    demo = tmp_path / "demo"
    assert runner.invoke(app, ["generate-demo-data", "-o", str(demo), "-n", "800"]).exit_code == 0
    book = pd.read_parquet(demo / "book.parquet")
    features = ["log_depth_ratio", "ofi_1", "ofi_20"]
    x = build(book, features)[features].dropna().reset_index(drop=True)
    y = pd.Series([-1, 0, 1] * (len(x) // 3 + 1))[: len(x)]
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.pkl").write_bytes(
        pickle.dumps({"model": LogisticBaseline().fit(x, y), "features": features})
    )
    raw = deployment()
    raw["strategies"][0].update(
        model=str(model), every=5, hold_s=10, min_confidence=0.34,
        model_sha256=hashlib.sha256((model / "model.pkl").read_bytes()).hexdigest(),
    )  # fmt: skip
    raw["strategies"][1]["symbol"] = "BTCUSDT"
    path = tmp_path / "deployment.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    checked = runner.invoke(app, ["prod", "check", str(path)])
    assert checked.exit_code == 0, checked.output
    assert "model pins hold" in checked.output

    run_dir = tmp_path / "run"
    result = runner.invoke(
        app,
        ["prod", "run", str(path), "--day", f"a={demo}", "--day", f"b={demo}", "--port", "0",
         "--run-dir", str(run_dir)],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    snapshot = json.loads((run_dir / "supervisor.json").read_text())
    assert snapshot["strategies"]["a"]["fills"] > 0 and snapshot["strategies"]["b"]["fills"] == 0
    assert snapshot["deployment_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ("a", "b"):
        assert (run_dir / name / "page.md").exists()
    status = runner.invoke(app, ["prod", "status", str(run_dir)])
    assert status.exit_code == 0 and "portfolio equity" in status.output

    raw["strategies"][0]["model_sha256"] = "0" * 64
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert runner.invoke(app, ["prod", "check", str(path)]).exit_code == 1
    refused = runner.invoke(app, ["prod", "run", str(path), "--day", f"a={demo}",
                                  "--day", f"b={demo}", "--no-serve", "--run-dir", str(run_dir)])  # fmt: skip
    assert refused.exit_code == 2 and "model pins do not hold" in refused.output


def test_run_takes_days_or_minutes_but_not_both(tmp_path: Path) -> None:
    path = tmp_path / "deployment.yaml"
    path.write_text(yaml.safe_dump(deployment()), encoding="utf-8")
    neither = runner.invoke(app, ["prod", "run", str(path)])
    assert neither.exit_code == 2
    both = runner.invoke(app, ["prod", "run", str(path), "--day", "a=x", "--minutes", "1"])
    assert both.exit_code == 2


def test_the_example_deployment_parses() -> None:
    example = Path(__file__).resolve().parent.parent / "deploy" / "deployment.example.yaml"
    spec = parse(yaml.safe_load(example.read_text(encoding="utf-8")))
    assert spec.strategies and spec.port == 9108


def test_every_alert_reads_a_series_the_metrics_serve() -> None:
    import re

    deploy = Path(__file__).resolve().parent.parent / "deploy"
    rules = yaml.safe_load((deploy / "alerts.yml").read_text(encoding="utf-8"))
    served = metrics({"deployment": "d", "strategies": {"a": {"symbol": "X"}}})
    names = set(re.findall(r"^# TYPE (trp_\w+)", served, re.MULTILINE))
    used = {
        name
        for group in rules["groups"]
        for rule in group["rules"]
        for name in re.findall(r"trp_\w+", rule["expr"])
    }
    assert used and used <= names, used - names
    compose = yaml.safe_load((deploy / "docker-compose.yml").read_text(encoding="utf-8"))
    assert compose["services"]["trp"]["command"][0] == "run"
