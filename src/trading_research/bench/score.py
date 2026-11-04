"""Scoring one strategy on one track: the same numbers for every strategy, read the same way.

For each instrument of the track the strategy is fitted on the history (the
rows before trading starts), decides on every row, and its decisions after the
start are run through the taker execution at Bybit's base fee (5.5 bp a side,
0.5 bp slippage, the spread paid on both legs), one position at a time, and no
sooner after a close than the strategy's ``cooldown`` rows if it declares one. From the attempts of every
instrument, pooled:

``gross``, ``cost``, ``net``
    bp per trade, before costs, the costs, and after them.
``t``
    The day-clustered *t* of net (hour-clustered on a track shorter than five
    days): trades on one day share a market, so they are not independent.
    Clusters never cross groups: each seed of a synthetic track is its own
    market.
``placebo``
    The share of nineteen placebos whose gross is below the strategy's. Each
    placebo is the same decisions moved in time (rotated within the scored
    rows by 1/20 to 19/20 of their length), traded the same way. An edge that
    a rotation keeps is a property of the market's drift, not of the timing.
``break-even fee``
    The taker fee per side at which net would be zero, with the spread and the
    slippage still paid. Set beside the fee ladder, it says which account, if
    any, the strategy would pay at, and the ladder's last rung (no fee at all)
    is the hypothetical case of a firm that pays none.

The verdict is a rule fixed here, before any board is computed, and is the
same for every strategy: see :func:`verdict`.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_research.bench.tracks import Markets, Track
from trading_research.plugins.market import MarketData, matches

#: The base tier's taker fee, bp a side: what every net here pays.
BASE_TAKER_BP = 5.5

#: Rungs of the fee ladder, bp a side: Bybit's base, VIP 1, the lowest
#: published taker rate (Pro 6, major coins), and none at all.
FEE_LADDER: tuple[tuple[str, float], ...] = (
    ("base", 5.5),
    ("vip1", 4.0),
    ("pro6_g1", 1.5),
    ("none (hypothetical)", 0.0),
)

PLACEBOS = 19
T_FOUND = 3.0
T_PAYS = 2.0
PLACEBO_BEATEN = 0.95


@dataclass
class TrackScore:
    """One strategy on one track."""

    track: str
    kind: str
    status: str  # "scored", "n/a" or "skipped"
    reason: str = ""
    instruments: int = 0
    groups: int = 0
    data_sha256: str = ""
    trades: int = 0
    clusters: int = 0
    cluster_unit: str = ""
    gross_bp: float = math.nan
    cost_bp: float = math.nan
    net_bp: float = math.nan
    gross_t: float = math.nan
    net_t: float = math.nan
    #: One-sided p of net > 0 from its clustered t (Student, clusters - 1 df),
    #: for a family correction across a board.
    net_p: float = math.nan
    placebo: float = math.nan
    breakeven_taker_bp: float = math.nan
    net_at: dict[str, float] = field(default_factory=dict)
    verdict: str = ""
    passed: bool | None = None
    seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in out.items()}


def missing_inputs(strategy: Any, frame: pd.DataFrame) -> list[str]:
    """Declared inputs the track's frame does not carry."""
    declared = tuple(getattr(strategy, "inputs", ()) or ())
    found = matches(declared, list(frame.columns))
    return [pattern for pattern, columns in found.items() if not columns]


def _taker(decisions: np.ndarray, market: MarketData, hold: int, cooldown: int = 0) -> pd.DataFrame:
    from trading_research.plugins.registry import resolve

    execution = resolve("execution", "taker", fees="base", cooldown=cooldown)
    attempts = execution.execute(decisions, market, hold=hold).attempts
    return attempts[attempts["filled"].astype(bool)]


def _clustered_t(values: pd.Series, keys: pd.Series) -> tuple[float, int]:
    means = values.groupby(keys.to_numpy()).mean()
    n = len(means)
    if n < 3:
        return math.nan, n
    sd = float(means.std(ddof=1))
    if sd == 0.0:
        return math.nan, n
    return float(means.mean() / (sd / math.sqrt(n))), n


def score_track(
    track: Track,
    build: Callable[[], Any],
    *,
    data_root: Path,
    markets: Markets | None = None,
) -> TrackScore:
    """Fit, decide, trade and score one strategy on every instrument of ``track``."""
    started = time.perf_counter()
    try:
        chosen = markets if markets is not None else track.markets(data_root)
    except FileNotFoundError as exc:
        return TrackScore(track.name, track.kind, "skipped", reason=str(exc))

    real: list[pd.DataFrame] = []
    placebo: list[list[pd.DataFrame]] = [[] for _ in range(PLACEBOS)]
    used = 0
    for market, start, group in chosen:
        strategy = build()
        missing = missing_inputs(strategy, market.frame)
        if missing:
            return TrackScore(
                track.name, track.kind, "n/a",
                reason=f"the track's frame carries nothing for {', '.join(missing)}",
            )  # fmt: skip
        try:
            strategy.fit(market.frame.iloc[:start])
            decisions = np.asarray(strategy.decide(market.frame), dtype=np.int64).copy()
        except Exception as exc:  # a strategy that cannot run on a track is n/a there
            return TrackScore(track.name, track.kind, "n/a", reason=f"{type(exc).__name__}: {exc}")
        decisions[:start] = 0
        hold = int(strategy.horizon)
        cooldown = int(getattr(strategy, "cooldown", 0) or 0)
        real.append(_taker(decisions, market, hold, cooldown).assign(group=group))
        scored = decisions[start:]
        for j in range(PLACEBOS):
            shift = round((j + 1) * len(scored) / (PLACEBOS + 1))
            moved = np.zeros_like(decisions)
            moved[start:] = np.roll(scored, shift)
            placebo[j].append(_taker(moved, market, hold, cooldown))
        used += 1

    out = TrackScore(track.name, track.kind, "scored", instruments=used)
    out.groups = len(chosen.parts)
    out.data_sha256 = chosen.fingerprint()
    real = [a for a in real if len(a)]
    attempts = pd.concat(real, ignore_index=True) if real else pd.DataFrame()
    out.seconds = round(time.perf_counter() - started, 1)
    if attempts.empty:
        out.verdict = "no trades"
        out.passed = None if track.kind == "recorded" or track.planted else True
        return out

    ns = attempts["ts"].to_numpy(dtype="int64")
    spans = attempts.assign(ns=ns).groupby("group")["ns"].agg(lambda v: v.max() - v.min())
    unit = "day" if float(spans.max()) / 86_400e9 >= 5 else "hour"
    floored = pd.to_datetime(ns, utc=True).floor("1D" if unit == "day" else "1h")
    # Trades cluster within a group (a seed, or the one span), never across.
    keys = pd.Series(attempts["group"].astype(str).to_numpy() + "|" + floored.astype(str))
    out.trades = len(attempts)
    out.cluster_unit = unit
    out.gross_bp = float(attempts["gross_bp"].mean())
    out.cost_bp = float(attempts["cost_bp"].mean())
    out.net_bp = float(attempts["net_bp"].mean())
    out.gross_t, out.clusters = _clustered_t(attempts["gross_bp"].reset_index(drop=True), keys)
    out.net_t, _ = _clustered_t(attempts["net_bp"].reset_index(drop=True), keys)
    if math.isfinite(out.net_t):
        from trading_research.market_making.verdicts import t_sf

        out.net_p = float(t_sf(out.net_t, out.clusters - 1))
    gross_placebos = [
        float(pd.concat([x for x in p if len(x)], ignore_index=True)["gross_bp"].mean())
        if any(len(x) for x in p)
        else 0.0
        for p in placebo
    ]
    out.placebo = float(np.mean([g < out.gross_bp for g in gross_placebos]))
    other = out.cost_bp - 2.0 * BASE_TAKER_BP  # spread and slippage, both legs
    out.breakeven_taker_bp = (out.gross_bp - other) / 2.0
    out.net_at = {name: out.gross_bp - other - 2.0 * fee for name, fee in FEE_LADDER}
    out.verdict, out.passed = verdict(track, out)
    return out


def verdict(track: Track, s: TrackScore) -> tuple[str, bool | None]:
    """The rule, the same for every strategy.

    Planted track: *found* when gross has a clustered *t* of at least 3 and
    beats at least 95% of the placebos; a miss is not a failure (a rule built
    for another effect is not expected to find this one), so it passes as
    ``None``. Null track: *quiet* unless gross has a *t* of 3 or more, which
    on a market with nothing in it means the strategy read the future or fitted
    noise. Recorded track: *pays the base fee* when net is positive with a *t*
    of at least 2 and gross beats 95% of the placebos; *an edge, smaller than
    the fee* when gross beats the placebos and is positive but net is not;
    *positive, not distinguishable from chance* when net is positive without
    the *t* or the placebo; otherwise *nothing beyond chance*.
    """
    beats = s.placebo >= PLACEBO_BEATEN
    if track.kind == "synthetic" and track.planted:
        if s.gross_t >= T_FOUND and beats:
            return f"found the planted {track.planted}", True
        return f"did not find the planted {track.planted}", None
    if track.kind == "synthetic":
        if s.gross_t >= T_FOUND:
            return "earned where nothing was planted: look-ahead or a fit to noise", False
        return "quiet, as it should be", True
    if s.net_bp > 0 and s.net_t >= T_PAYS and beats:
        return "pays the base fee", True
    if s.net_bp > 0:
        return "positive after the fee, not distinguishable from chance", False
    if beats and s.gross_bp > 0:
        return "an edge, smaller than the fee", False
    return "nothing beyond chance", False
