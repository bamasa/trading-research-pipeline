"""The forward track's one read, decided as its registration states.

``docs/preregistration/bench_forward_2026_11.md`` registered, before the
track's first scored day, which strategies are read (the twelve frozen in
:data:`~trading_research.bench.manifest.FORWARD_STRATEGIES`), what a pass is
(F1) and what the reversion should do given the market state (P1). This module
is that rule as code, committed before the read, so the read is one command
whose output is not chosen after the numbers are seen.

F1, for each of the twelve: **passes** when its verdict on the track is *pays
the base fee* and its Holm-adjusted *p* of net > 0 over the twelve (m = 12,
whether or not each scored) is below 0.05; otherwise **killed**.

P1, outside the family: the index's ten-minute autocorrelation on the scored
days at or below -0.05 predicts the reversion's net per trade above zero;
above -0.02 predicts zero or below; between the two predicts nothing. It is
**right**, **wrong** or **not tested** (the state fell between, or the
reversion made no trade, so it has no net to compare).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

from trading_research.bench.manifest import (
    FORWARD_FIT_FINGERPRINT,
    FORWARD_REGISTERED,
    FORWARD_STRATEGIES,
    FORWARD_TRACK,
)
from trading_research.bench.score import TrackScore
from trading_research.bench.tracks import TRACKS, Track

#: F1's family level.
ALPHA = 0.05
#: P1's thresholds on the index's ten-minute autocorrelation.
P1_POSITIVE_AT_OR_BELOW = -0.05
P1_NOT_POSITIVE_ABOVE = -0.02
#: The strategy P1 is about.
P1_STRATEGY = ("reversion", "strategy")


def unfrozen(specs: Sequence[Any]) -> list[str]:
    """The plugins that are not the registered twelve at their registered sources."""
    found = {(s.name, s.kind): s.fingerprint() for s in specs}
    problems = [
        f"{name} ({kind}): source {found[(name, kind)][:16] or 'unreadable'}, registered {sha[:16]}"
        for (name, kind), sha in FORWARD_STRATEGIES.items()
        if (name, kind) in found and found[(name, kind)] != sha
    ]
    problems += [f"{name} ({kind}): not registered" for name, kind in found
                 if (name, kind) not in FORWARD_STRATEGIES]  # fmt: skip
    problems += [f"{name} ({kind}): registered, not read" for name, kind in FORWARD_STRATEGIES
                 if (name, kind) not in found]  # fmt: skip
    return problems


def f1(scores: dict[tuple[str, str], TrackScore]) -> dict[str, dict[str, Any]]:
    """F1 for each of the twelve, with Holm over all twelve however many scored."""
    from trading_research.pairs.stats import holm_adjusted

    keys = list(FORWARD_STRATEGIES)
    p = [scores[k].net_p if k in scores else math.nan for k in keys]
    adjusted = holm_adjusted(p)
    out: dict[str, dict[str, Any]] = {}
    for key, holm in zip(keys, adjusted, strict=True):
        score = scores.get(key)
        holm_p = None if math.isnan(holm) else float(holm)
        passes = (
            score is not None
            and score.status == "scored"
            and score.passed is True
            and holm_p is not None
            and holm_p < ALPHA
        )
        out[key[0]] = {
            "kind": key[1],
            "status": "passes" if passes else "killed",
            "verdict": score.verdict if score and score.status == "scored" else
                       (f"{score.status}: {score.reason}" if score else "not read"),
            "holm_p": holm_p,
        }  # fmt: skip
    return out


def p1(state: float, net_bp: float) -> dict[str, Any]:
    """P1 from the scored days' state and the reversion's net per trade at the base fee."""
    if state <= P1_POSITIVE_AT_OR_BELOW:
        predicted = "above zero"
    elif state > P1_NOT_POSITIVE_ABOVE:
        predicted = "zero or below"
    else:
        return {"state": state, "net_bp": _num(net_bp), "predicted": None,
                "status": "not tested", "why": "the state fell between -0.05 and -0.02"}  # fmt: skip
    if not math.isfinite(net_bp):
        return {"state": state, "net_bp": None, "predicted": predicted,
                "status": "not tested", "why": "the reversion made no trade"}  # fmt: skip
    right = net_bp > 0 if predicted == "above zero" else net_bp <= 0
    return {"state": state, "net_bp": net_bp, "predicted": predicted,
            "status": "right" if right else "wrong", "why": ""}  # fmt: skip


def _num(value: float) -> float | None:
    return value if math.isfinite(value) else None


def read(
    cache: Any,
    *,
    specs: Sequence[Any] | None = None,
    track: Track | None = None,
    on_score: Callable[[TrackScore], None] | None = None,
) -> dict[str, Any]:
    """Score the twelve on the forward track and decide F1 and P1; the payload to commit.

    Raises :class:`~trading_research.bench.tracks.Sealed` before the track
    opens, :class:`FileNotFoundError` when its data is not here, and
    :class:`ValueError` when a plugin is not the one registered. ``track`` is
    for a test to read another panel track by the same rule.
    """
    from trading_research.bench import board_rows, code_version, run
    from trading_research.bench.state import index_autocorrelation
    from trading_research.plugins import get

    track = track or TRACKS[FORWARD_TRACK]
    specs = list(specs) if specs is not None else [get(n, k) for n, k in FORWARD_STRATEGIES]
    forward = track.name == FORWARD_TRACK
    problems = unfrozen(specs) if forward else []
    if problems:
        raise ValueError("not the registered strategies: " + "; ".join(problems))
    markets = cache.get(track)
    if markets is None:
        raise FileNotFoundError(cache.error(track))
    started = datetime.now(UTC)
    fitted = ""
    if forward:
        from trading_research.bench.tracks import fitted_window

        fitted = fitted_window(track.name, cache.data_root).fingerprint()
    cards = [run(spec, {}, [track], cache, on_score=on_score) for spec in specs]
    scores = {(c.name, c.kind): c.scores[0] for c in cards}
    state = index_autocorrelation(markets)
    reversion = scores.get(P1_STRATEGY)
    net = reversion.net_bp if reversion is not None and reversion.status == "scored" else math.nan
    decided = f1(scores)
    return {
        "track": track.name,
        "registered": FORWARD_REGISTERED,
        "registration": "docs/preregistration/bench_forward_2026_11.md",
        "read_at": started.strftime("%Y-%m-%d %H:%M UTC"),
        "code": code_version(),
        "data_sha256": markets.fingerprint(),
        "fitted_sha256": fitted,
        "fitted_matches_registration": fitted == FORWARD_FIT_FINGERPRINT if forward else None,
        "instruments": max((len(getattr(p, "mids", ())) for p in markets.parts), default=0),
        "f1": decided,
        "f1_passes": sorted(n for n, d in decided.items() if d["status"] == "passes"),
        "p1": p1(state, net),
        "rows": board_rows(cards),
    }


__all__ = ["ALPHA", "P1_NOT_POSITIVE_ABOVE", "P1_POSITIVE_AT_OR_BELOW", "f1", "p1", "read",
           "unfrozen"]  # fmt: skip
