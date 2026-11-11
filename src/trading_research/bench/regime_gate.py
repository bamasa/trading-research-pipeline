"""The regime gate of ``docs/preregistration/regime_gate_2026_11.md``.

The reversion rule is paused while the structural-break detector says the
equal-weighted index has stopped reverting (a ``+1`` alarm of its dependence
family) and resumed when it says the reversal is back (``-1``). Everything here
is the registration's rule as code: the stream, the detector's setting, the
state machine, the placebo shifts and the decision. Nothing is searched.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from trading_research.bench.tracks import Panel

HISTORY, ONLINE, FAMILY = 365, 90, "dependence"
TEN_MINUTES = 120  # rows of the five-second grid
PLACEBOS = 19


def index_returns(panel: Panel) -> pd.Series:
    """The equal-weighted index's non-overlapping ten-minute log returns, labelled at their end."""
    from trading_research.strategies.reversion import index_level

    log_mid = pd.DataFrame({s: np.log(m) for s, m in panel.mids.items()}).sort_index().ffill()
    log_mid = log_mid.dropna()
    level = pd.Series(index_level(log_mid.to_numpy(dtype=np.float64)), index=log_mid.index)
    return level.iloc[::TEN_MINUTES].diff().dropna()


def alarms(returns: pd.Series) -> list[tuple[pd.Timestamp, int]]:
    """The detector's dependence alarms on the stream: (label of the triggering return, direction)."""
    from trading_research.validation.structural_breaks import MonitorSpec, detect_breaks

    spec = MonitorSpec(history_len=HISTORY, online_len=ONLINE, statistics=(FAMILY,))
    table = detect_breaks(returns.reset_index(drop=True), spec, on_progress=None).table()
    labels = pd.DatetimeIndex(returns.index)
    positions = table["index"].to_numpy(dtype=np.int64)
    directions = table["direction"].to_numpy(dtype=np.int64)
    return [(labels[p], int(d)) for p, d in zip(positions, directions, strict=True)]


def pause_mask(labels: pd.DatetimeIndex, found: list[tuple[pd.Timestamp, int]]) -> np.ndarray:
    """True on every row the gate pauses: after a ``+1`` alarm and until the next ``-1``.

    A row is affected by an alarm only if it is labelled after the alarm's
    return, so the gate never acts on a value before it was known.
    """
    paused = np.zeros(len(labels), dtype=bool)
    state, last = False, 0
    for at, direction in sorted(found):
        cut = int(labels.searchsorted(at, side="right"))
        paused[last:cut] = state
        state, last = direction > 0, cut
    paused[last:] = state
    return paused


def shifted(mask: np.ndarray, start: int, j: int) -> np.ndarray:
    """The ``j``-th placebo: the scored part of the mask rolled by j/20 of its length."""
    out = mask.copy()
    scored = mask[start:]
    out[start:] = np.roll(scored, round(j * len(scored) / (PLACEBOS + 1)))
    return out


@dataclass
class GateResult:
    """The gate on one panel: what it paused, and what it earned against running ungated."""

    name: str
    alarms: list[tuple[str, int]] = field(default_factory=list)
    paused_share: float = 0.0
    ungated_pct: float = 0.0
    gated_pct: float = 0.0
    placebo_pct: list[float] = field(default_factory=list)

    @property
    def gain(self) -> float:
        return self.gated_pct - self.ungated_pct

    @property
    def placebo_gains(self) -> list[float]:
        return [p - self.ungated_pct for p in self.placebo_pct]

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["gain"], out["placebo_gains"] = self.gain, self.placebo_gains
        return out


def run_gate(name: str, panel: Panel, build: Callable[[], Any]) -> GateResult:
    """The rule fitted on the panel's fitted rows, then traded ungated, gated and under each placebo."""
    from trading_research.bench.score import _taker

    found = alarms(index_returns(panel))
    n = len(panel.grids)
    totals = np.zeros(2 + PLACEBOS)
    paused_rows = scored_rows = 0
    for symbol in panel.grids:
        strategy = build()
        market = panel.market(symbol)
        labels = pd.DatetimeIndex(pd.to_datetime(market.frame["timestamp"], utc=True))
        start = int(labels.searchsorted(panel.start))
        strategy.fit(market.frame.iloc[:start])
        decisions = np.asarray(strategy.decide(market.frame), dtype=np.int64).copy()
        decisions[:start] = 0
        mask = pause_mask(labels, found)
        paused_rows += int(mask[start:].sum())
        scored_rows += len(labels) - start
        masks = [np.zeros(len(labels), dtype=bool), mask] + [
            shifted(mask, start, j) for j in range(1, PLACEBOS + 1)
        ]
        hold, cooldown = int(strategy.horizon), int(getattr(strategy, "cooldown", 0) or 0)
        for k, m in enumerate(masks):
            filled = _taker(np.where(m, 0, decisions), market, hold, cooldown)
            totals[k] += float(filled["net_bp"].sum())
    pct = totals / n / 100.0
    return GateResult(
        name,
        alarms=[(at.strftime("%Y-%m-%dT%H:%M"), d) for at, d in found],
        paused_share=paused_rows / max(scored_rows, 1),
        ungated_pct=float(pct[0]),
        gated_pct=float(pct[1]),
        placebo_pct=[float(v) for v in pct[2:]],
    )


def verdict(results: list[GateResult]) -> dict[str, Any]:
    """The registration's decision over a set of panels: passes, or killed with the first failure."""
    gain = float(sum(r.gain for r in results))
    placebo = [float(sum(r.placebo_gains[j] for r in results)) for j in range(PLACEBOS)]
    if all(not r.alarms for r in results):
        status, first = "not tested", "no alarm"
    elif gain <= 0:
        status, first = "killed", "the gate's gain is not positive"
    elif any(p >= gain for p in placebo):
        status, first = "killed", "a placebo gained as much"
    else:
        status, first = "passes", ""
    return {"status": status, "first_kill": first, "gain_pct": gain,
            "placebo_gains_pct": placebo, "beaten": sum(p < gain for p in placebo),
            "panels_helped": sum(r.gain > 0 for r in results)}  # fmt: skip


__all__ = ["GateResult", "alarms", "index_returns", "pause_mask", "run_gate", "shifted", "verdict"]
