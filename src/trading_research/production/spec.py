"""A deployment: which frozen strategies run, with what capital, under which limits.

One YAML file describes a deployment, and nothing about it is decided at run
time::

    name: btc-reversion
    capital: 1000                 # quote currency the portfolio may put at risk
    portfolio:
      max_gross: 0.8              # share of capital deployed across strategies
      max_weight: 0.6             # share any one strategy may take
      daily_loss: 40              # quote currency, the whole portfolio, per UTC day
      max_drawdown: 80            # quote currency, from the peak of portfolio equity
    health: {host: 127.0.0.1, port: 9108}
    monitors:
      drift:  {k: 0.5, h: 5.0, min_trades: 10}
      regime: {warmup: 600, k: 0.5, h: 12.0, sample_s: 1.0}
      feed:   {stale_after_s: 5.0}
    strategies:
      - name: btc-taker
        venue: bybit
        market: linear
        symbol: BTCUSDT
        model: models/btc/model.pkl
        model_sha256: 3f2a...       # the file the backtest validated; any other is refused
        execution: taker
        clip: 0.001
        hold_s: 120
        expected: {net_bp: 6.9, sd_bp: 45.0}   # per round trip, from the held-out backtest

``load`` reads it, checks every field, pins each model to the sha256 its
backtest recorded, and puts the whole file through the paper step's key guard:
a deployment that names a credential or a private endpoint does not start.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from trading_research.live.guard import check_config

EXECUTIONS = ("taker", "passive", "maker")
VENUES = {"bybit": ("linear", "spot"), "binance": ("linear", "spot")}


class DeploymentError(ValueError):
    """The deployment file is incomplete, inconsistent or refused by the key guard."""


@dataclass(frozen=True)
class Expected:
    """What the backtest says one round trip earns: the reference the drift monitor holds."""

    net_bp: float
    sd_bp: float

    def __post_init__(self) -> None:
        if not self.sd_bp > 0:
            raise DeploymentError("expected.sd_bp must be positive")


@dataclass(frozen=True)
class StrategySpec:
    name: str
    venue: str
    market: str
    symbol: str
    model: str
    execution: str
    clip: float
    expected: Expected
    model_sha256: str | None = None
    hold_s: float = 120.0
    latency_ms: float = 10.0
    max_position_clips: float = 10.0
    #: Quote currency the strategy's own kill switch allows; the portfolio's daily loss if unset.
    max_loss: float | None = None
    every: int = 50
    min_confidence: float = 0.5
    fee_bp: str | None = None
    fee_tier: str | None = None
    fee_asof: str | None = None


@dataclass(frozen=True)
class PortfolioSpec:
    max_gross: float = 0.8
    max_weight: float = 0.6
    daily_loss: float = 50.0
    max_drawdown: float = 100.0


@dataclass(frozen=True)
class DriftSpec:
    k: float = 0.5
    h: float = 5.0
    min_trades: int = 10


@dataclass(frozen=True)
class RegimeSpec:
    warmup: int = 600
    k: float = 0.5
    h: float = 12.0
    sample_s: float = 1.0


@dataclass(frozen=True)
class FeedSpec:
    stale_after_s: float = 5.0


@dataclass(frozen=True)
class MonitorSpec:
    drift: DriftSpec = field(default_factory=DriftSpec)
    regime: RegimeSpec = field(default_factory=RegimeSpec)
    feed: FeedSpec = field(default_factory=FeedSpec)


@dataclass(frozen=True)
class Deployment:
    name: str
    capital: float
    strategies: tuple[StrategySpec, ...]
    portfolio: PortfolioSpec = field(default_factory=PortfolioSpec)
    monitors: MonitorSpec = field(default_factory=MonitorSpec)
    host: str = "127.0.0.1"
    port: int = 9108
    source: str | None = None
    sha256: str | None = None

    def strategy(self, name: str) -> StrategySpec:
        for spec in self.strategies:
            if spec.name == name:
                return spec
        raise KeyError(name)


def _section(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key) or {}
    if not isinstance(value, Mapping):
        raise DeploymentError(f"{key} must be a mapping")
    return value


def _build(cls: Any, raw: Mapping[str, Any], where: str) -> Any:
    names = set(cls.__dataclass_fields__)
    unknown = sorted(set(raw) - names)
    if unknown:
        raise DeploymentError(f"{where}: unknown field(s) {', '.join(unknown)}")
    try:
        return cls(**raw)
    except TypeError as exc:
        raise DeploymentError(f"{where}: {exc}") from exc


def _strategy(raw: Mapping[str, Any], index: int) -> StrategySpec:
    where = f"strategies[{index}]"
    if not isinstance(raw, Mapping):
        raise DeploymentError(f"{where} must be a mapping")
    body = dict(raw)
    expected = body.pop("expected", None)
    if not isinstance(expected, Mapping):
        raise DeploymentError(
            f"{where}: expected {{net_bp, sd_bp}} is required, the backtest's figure per round "
            "trip that the drift monitor holds the live result to"
        )
    body["expected"] = _build(Expected, expected, f"{where}.expected")
    spec: StrategySpec = _build(StrategySpec, body, where)
    if spec.venue not in VENUES or spec.market not in VENUES[spec.venue]:
        raise DeploymentError(f"{where}: venue/market {spec.venue}/{spec.market} is not supported")
    if spec.execution not in EXECUTIONS:
        raise DeploymentError(f"{where}: execution must be one of {', '.join(EXECUTIONS)}")
    if not spec.clip > 0 or not spec.hold_s > 0 or spec.every < 1:
        raise DeploymentError(f"{where}: clip and hold_s must be positive, every at least 1")
    return spec


def parse(raw: Mapping[str, Any], *, source: str | None = None) -> Deployment:
    """A deployment from its parsed YAML, every field checked."""
    found = check_config(raw)
    if found:
        raise DeploymentError(
            "a deployment runs on public market data only and is refused: " + "; ".join(found)
        )
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise DeploymentError("name is required")
    capital = raw.get("capital")
    if not isinstance(capital, (int, float)) or not capital > 0:
        raise DeploymentError("capital must be a positive number")
    entries = raw.get("strategies")
    if not isinstance(entries, list) or not entries:
        raise DeploymentError("strategies must be a non-empty list")
    strategies = tuple(_strategy(entry, i) for i, entry in enumerate(entries))
    names = [s.name for s in strategies]
    if len(set(names)) != len(names):
        raise DeploymentError("strategy names must be unique")
    portfolio: PortfolioSpec = _build(PortfolioSpec, _section(raw, "portfolio"), "portfolio")
    if not 0 < portfolio.max_weight <= 1 or not 0 < portfolio.max_gross <= 1:
        raise DeploymentError("portfolio.max_weight and max_gross are shares in (0, 1]")
    if not portfolio.daily_loss > 0 or not portfolio.max_drawdown > 0:
        raise DeploymentError("portfolio.daily_loss and max_drawdown must be positive")
    monitors_raw = _section(raw, "monitors")
    unknown = sorted(set(monitors_raw) - {"drift", "regime", "feed"})
    if unknown:
        raise DeploymentError(f"monitors: unknown monitor(s) {', '.join(unknown)}")
    monitors = MonitorSpec(
        drift=_build(DriftSpec, _section(monitors_raw, "drift"), "monitors.drift"),
        regime=_build(RegimeSpec, _section(monitors_raw, "regime"), "monitors.regime"),
        feed=_build(FeedSpec, _section(monitors_raw, "feed"), "monitors.feed"),
    )
    health = _section(raw, "health")
    unknown = sorted(
        set(raw) - {"name", "capital", "strategies", "portfolio", "monitors", "health"}
    )
    if unknown:
        raise DeploymentError(f"unknown top-level field(s) {', '.join(unknown)}")
    return Deployment(
        name=name,
        capital=float(capital),
        strategies=strategies,
        portfolio=portfolio,
        monitors=monitors,
        host=str(health.get("host", "127.0.0.1")),
        port=int(health.get("port", 9108)),
        source=source,
    )


def load(path: Path | str) -> Deployment:
    """Read, check and pin a deployment file."""
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    if not isinstance(raw, Mapping):
        raise DeploymentError(f"{target} is not a YAML mapping")
    deployment = parse(raw, source=str(target))
    return Deployment(
        **{**deployment.__dict__, "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
    )


def check_pins(deployment: Deployment) -> list[str]:
    """Each strategy's model loaded and compared with the sha256 its file pins (empty: all hold).

    A model named by path must carry a pin: the file the backtest validated is
    the only one a deployment may run. A model by name (``always_hold``) needs
    none.
    """
    from trading_research.live.decide import load_model

    problems = []
    for spec in deployment.strategies:
        try:
            loaded = load_model(spec.model)
        except (FileNotFoundError, ValueError) as exc:
            problems.append(f"{spec.name}: {exc}")
            continue
        if loaded.path is None:
            continue
        if spec.model_sha256 is None:
            problems.append(f"{spec.name}: model {spec.model} has no model_sha256 pin")
        elif loaded.sha256 != spec.model_sha256:
            problems.append(
                f"{spec.name}: model {spec.model} is {loaded.sha256[:12]}, the deployment pins "
                f"{spec.model_sha256[:12]}"
            )
    return problems
