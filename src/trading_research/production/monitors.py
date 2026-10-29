"""What a running strategy is watched for: drift from its backtest, a regime break, a dead feed.

Each monitor is online — one observation in, the statistic and whether it
alarms out — and keeps nothing but its running sums, so it costs the same on
the millionth observation as on the first.

**Drift.** A one-sided Page CUSUM on the round trips' net, standardised by the
backtest's figure: ``S = max(0, S + (mu0 - x) / sd - k)``. It grows while the
live strategy earns less than its backtest by more than ``k`` standard
deviations a trade and alarms at ``h``, so a strategy that has stopped earning
what it was validated to earn is stopped after a number of trades that scales
with how far it fell short, not after a fixed loss. Nothing alarms before
``min_trades`` round trips.

**Regime.** CUSUMs on the mid's log return, sampled every ``sample_s``
seconds of event time and standardised by the first ``warmup`` samples: two on
the mean (``±z - k``) and two on the variance, each the log-likelihood ratio of
a variance ``ratio`` times larger (or smaller) than the warm-up's,
``(z^2 (1 - 1/r) - ln r) / 2``. A ratio statistic drifts down while nothing
changes, so its false-alarm interval grows about as ``e^h``, where a
``z^2 - 1`` sum would ride the chi-square's long right tail into an alarm every
few hundred samples. It alarms when any sum crosses ``h`` and restarts them,
so a market that moved into a state unlike the one the session started in is
flagged once per change, the way the offline monitors of
:mod:`trading_research.validation.changepoint` flag a break in history.

**Feed.** Event time against wall time: a stream that has delivered no book
row for ``stale_after_s`` is dead whatever the account says.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class DriftMonitor:
    mu_bp: float
    sd_bp: float
    k: float = 0.5
    h: float = 5.0
    min_trades: int = 10
    statistic: float = 0.0
    trades: int = 0
    total_bp: float = 0.0
    sum_sq: float = 0.0

    def observe(self, net_bp: float) -> bool:
        """One round trip's net, in basis points; whether the monitor alarms."""
        self.trades += 1
        self.total_bp += net_bp
        self.sum_sq += net_bp * net_bp
        z = (self.mu_bp - net_bp) / self.sd_bp
        self.statistic = max(0.0, self.statistic + z - self.k)
        return self.trades >= self.min_trades and self.statistic > self.h

    @property
    def mean_bp(self) -> float:
        return self.total_bp / self.trades if self.trades else float("nan")

    @property
    def live_sd_bp(self) -> float:
        if self.trades < 2:
            return float("nan")
        mean = self.mean_bp
        return math.sqrt(max(self.sum_sq / self.trades - mean * mean, 0.0) * self.trades
                         / (self.trades - 1))  # fmt: skip


@dataclass
class RegimeMonitor:
    warmup: int = 600
    k: float = 0.5
    h: float = 12.0
    sample_s: float = 1.0
    ratio: float = 2.0
    samples: int = 0
    alarms: int = 0
    mean_up: float = 0.0
    mean_down: float = 0.0
    var_up: float = 0.0
    var_down: float = 0.0
    _last_ns: int | None = None
    _last_mid: float | None = None
    _sum: float = 0.0
    _sum_sq: float = 0.0
    _mu: float = 0.0
    _sd: float = 0.0

    def observe(self, ts_ns: int, mid: float) -> bool:
        """A mid at an event time; whether a regime alarm is raised by it."""
        if not mid > 0:
            return False
        if self._last_ns is None or self._last_mid is None:
            self._last_ns, self._last_mid = ts_ns, mid
            return False
        if ts_ns - self._last_ns < self.sample_s * 1e9:
            return False
        r = math.log(mid / self._last_mid)
        self._last_ns, self._last_mid = ts_ns, mid
        self.samples += 1
        if self.samples <= self.warmup:
            self._sum += r
            self._sum_sq += r * r
            if self.samples == self.warmup:
                self._mu = self._sum / self.warmup
                variance = self._sum_sq / self.warmup - self._mu * self._mu
                self._sd = math.sqrt(max(variance, 0.0))
            return False
        if self._sd <= 0:
            return False
        z = (r - self._mu) / self._sd
        self.mean_up = max(0.0, self.mean_up + z - self.k)
        self.mean_down = max(0.0, self.mean_down - z - self.k)
        up, down = self.ratio, 1.0 / self.ratio
        self.var_up = max(0.0, self.var_up + 0.5 * (z * z * (1 - 1 / up) - math.log(up)))
        self.var_down = max(0.0, self.var_down + 0.5 * (z * z * (1 - 1 / down) - math.log(down)))
        if self.statistic > self.h:
            self.alarms += 1
            self.mean_up = self.mean_down = self.var_up = self.var_down = 0.0
            return True
        return False

    @property
    def statistic(self) -> float:
        return max(self.mean_up, self.mean_down, self.var_up, self.var_down)

    @property
    def ready(self) -> bool:
        return self.samples > self.warmup and self._sd > 0


@dataclass
class FeedMonitor:
    stale_after_s: float = 5.0
    rows: int = 0
    gaps: int = 0
    last_event_ns: int | None = None
    last_wall_ns: int | None = None
    gap_kinds: dict[str, int] = field(default_factory=dict)

    def row(self, ts_ns: int, wall_ns: int) -> None:
        self.rows += 1
        self.last_event_ns = ts_ns
        self.last_wall_ns = wall_ns

    def gap(self, kind: str) -> None:
        self.gaps += 1
        self.gap_kinds[kind] = self.gap_kinds.get(kind, 0) + 1

    def age_s(self, wall_ns: int) -> float:
        """Seconds since the last book row arrived (infinite before the first)."""
        if self.last_wall_ns is None:
            return float("inf")
        return max(0.0, (wall_ns - self.last_wall_ns) / 1e9)

    def stale(self, wall_ns: int) -> bool:
        return self.rows > 0 and self.age_s(wall_ns) > self.stale_after_s
