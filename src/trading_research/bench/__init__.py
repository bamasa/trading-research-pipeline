"""The benchmark: fixed tracks, one scoring rule, a card per strategy and a board.

``trp bench list`` shows the tracks, ``trp bench run NAME`` scores one registered
strategy or model (a plugin of your own included: see ``docs/extending.md``) on
them and writes its card, and ``trp bench board`` scores every registered
strategy and writes the board. :mod:`.tracks` defines the markets and
:mod:`.score` the numbers and the verdict; this module runs them and writes
the pages.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trading_research.bench.score import FEE_LADDER, TrackScore, score_track
from trading_research.bench.tracks import RECORDED_HELP, TRACKS, Markets, Track, iter_tracks

DISCLAIMER = (
    "Every number here is a backtest on a fixed span, at a stated fee, with a simple "
    "taker fill on a five-second grid (one second for the book tracks). The recorded "
    "spans were chosen after this repository had read them. A pass on a track is not a "
    "forecast. To trade a strategy that passes, it still needs: a pre-registered test on "
    "data no one has looked at (a held-out block, then a forward one); an execution "
    "simulated at the depth and latency it would really meet, since the top of the book "
    "on a grid is optimistic; the fee tier the account actually has; and a paper run on "
    "live data before money. This repository's own studies, which did all of that, found "
    "nothing that pays Bybit's published taker fees."
)


@dataclass
class Card:
    """One strategy's scores on every track run."""

    name: str
    kind: str
    origin: str
    params: dict[str, Any]
    sha256: str | None
    scores: list[TrackScore] = field(default_factory=list)

    @property
    def marks(self) -> str:
        return "".join(_mark(s) for s in self.scores)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "kind": self.kind, "origin": self.origin,
            "params": self.params, "sha256": self.sha256,
            "scores": [s.as_dict() for s in self.scores],
        }  # fmt: skip


def _mark(s: TrackScore) -> str:
    if s.status != "scored":
        return "-"
    return {True: "✓", False: "✗", None: "·"}[s.passed]


def builder(spec: Any, params: dict[str, Any]) -> Callable[[], Any]:
    """A function returning a fresh strategy: a model is made into one at the base fee."""
    merged = {**spec.check_params, **params}
    if spec.kind == "model":
        from trading_research.backtest.costs import fee_tier
        from trading_research.plugins.model_strategy import ModelStrategy

        return lambda: ModelStrategy(spec.build(**merged), fees=fee_tier("base"))
    return lambda: spec.build(**merged)


class MarketCache:
    """Each track's markets, built once and shared by every strategy on a board."""

    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root
        self._built: dict[str, Markets | FileNotFoundError] = {}

    def get(self, track: Track) -> Markets | None:
        if track.name not in self._built:
            try:
                self._built[track.name] = track.markets(self.data_root)
            except FileNotFoundError as exc:
                self._built[track.name] = exc
        built = self._built[track.name]
        return None if isinstance(built, FileNotFoundError) else built

    def error(self, track: Track) -> str:
        built = self._built.get(track.name)
        return str(built) if isinstance(built, FileNotFoundError) else ""


def run(
    spec: Any,
    params: dict[str, Any],
    tracks: Iterable[Track],
    cache: MarketCache,
    on_score: Callable[[TrackScore], None] | None = None,
) -> Card:
    """Score one plugin on each track."""
    card = Card(spec.name, spec.kind, spec.origin, params, spec.fingerprint())
    build = builder(spec, params)
    for track in tracks:
        markets = cache.get(track)
        if markets is None:
            score = TrackScore(track.name, track.kind, "skipped", reason=cache.error(track))
        else:
            score = score_track(track, build, data_root=cache.data_root, markets=markets)
        card.scores.append(score)
        if on_score:
            on_score(score)
    return card


# --- pages ------------------------------------------------------------------------


def _n(value: float | None, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:+.{digits}f}" if digits else f"{value:,.0f}"


def _pct(value: float) -> str:
    return "-" if math.isnan(value) else f"{value:.0%}"


def card_page(card: Card) -> str:
    lines = [
        f"# {card.name}: benchmark card",
        "",
        f"A {card.kind} ({card.origin}), parameters {card.params or 'as registered'}, "
        f"source sha256 `{(card.sha256 or 'unavailable')[:16]}`. Written "
        f"{datetime.now(UTC):%Y-%m-%d %H:%M} UTC by `trp bench run`.",
        "",
        f"Marks, one per track in the order below: **{card.marks}** "
        "(✓ as a good strategy should, ✗ not, · a planted effect it was not built for, "
        "- not run).",
        "",
        "| track | verdict | trades | gross bp | net bp | t (net) | placebo | break-even taker fee |",
        "|---|---|--:|--:|--:|--:|--:|--:|",
    ]
    for s in card.scores:
        if s.status != "scored":
            lines.append(f"| {s.track} | {s.status}: {s.reason[:120]} | | | | | | |")
            continue
        lines.append(
            f"| {s.track} | {s.verdict} | {s.trades:,} | {_n(s.gross_bp)} | {_n(s.net_bp)} | "
            f"{_n(s.net_t)} | {_pct(s.placebo)} | {_n(s.breakeven_taker_bp)} bp |"
        )
    lines += ["", "## Net per trade at each fee, bp", "",
              "| track | " + " | ".join(f"{n} ({f:g} bp)" for n, f in FEE_LADDER) + " |",
              "|---|" + "--:|" * len(FEE_LADDER)]  # fmt: skip
    for s in card.scores:
        if s.status == "scored" and s.net_at:
            lines.append(
                f"| {s.track} | " + " | ".join(_n(s.net_at[n]) for n, _ in FEE_LADDER) + " |"
            )
    lines += ["", "## What a pass would still need", "", DISCLAIMER, ""]
    return "\n".join(lines)


def write_card(card: Card, output: Path) -> Path:
    folder = output / card.name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "card.json").write_text(json.dumps(card.as_dict(), indent=2) + "\n", encoding="utf-8")
    page = folder / "card.md"
    page.write_text(card_page(card), encoding="utf-8")
    return page


def board_rows(cards: list[Card]) -> list[dict[str, Any]]:
    rows = []
    for card in cards:
        row: dict[str, Any] = {"name": card.name, "kind": card.kind, "marks": card.marks}
        for s in card.scores:
            row[s.track] = s.as_dict()
        rows.append(row)
    return rows


__all__ = ["DISCLAIMER", "FEE_LADDER", "RECORDED_HELP", "TRACKS", "Card", "MarketCache",
           "TrackScore", "board_rows", "builder", "card_page", "iter_tracks", "run",
           "write_card"]  # fmt: skip
