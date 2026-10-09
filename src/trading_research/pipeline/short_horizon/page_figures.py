"""The pages' figures, each rendered in a light and a dark theme for the README's ``<picture>``.

Every chart is one of :mod:`trading_research.reporting`'s, drawn from what has
been computed: the placebo search and S25 against its own nulls from the nulls
file, the leaderboard and the marginals from theirs, S25's edge ladder, the
fill-rule comparison (A0 on O, and S-P and S25-Z on H when H's read records
their order streams), the regime map per execution path, and the
break-detector view once a block is read. Each is written as
``docs/images/short_horizon_2025_<name>_light.png`` and ``..._dark.png``
(``plots.both_themes``); a figure whose input is missing is listed as pending.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd

IMAGES = Path("docs/images")
PREFIX = "short_horizon_2025_"
#: How the pages file, under ``experiments/results``, reaches the images.
FROM_PAGES = "../../docs/images"
#: The regime map's charts: the slots on each execution path, and its units.
EXECUTION_PATHS: dict[str, tuple[tuple[str, ...], str]] = {
    "taker": (("S25", "S-C", "S-R"), "taker slots, daily net in bp of traded notional"),
    "entry": (("S-P",), "passive entry (S-P), daily net per 100 USDT of clip"),
    "maker": (("S-M",), "the market maker (S-M), daily net per 100 USDT of clip"),
}
Draw = Callable[[Path], Any]


def _search(c: Any, drawn: dict[str, tuple[Draw, str]], written: Any) -> None:
    """The placebo search, S25's nulls, the leaderboard and the marginals (stage D)."""
    from trading_research.reporting import search_figures as sf
    from trading_research.reporting.nulls import read_nulls

    if c.d is None:
        written.wait("figures of the search", "stage D is not complete")
        return
    nulls = read_nulls(c.study.results("nulls.csv"))
    drawn["placebo_search"] = (lambda p: sf.placebo_search_chart(nulls, p),
                               "The chosen cells' A against both null distributions, with p_search")  # fmt: skip
    if (nulls["subject"] == "S25").any():
        drawn["placebo_s25"] = (lambda p: sf.placebo_search_chart(nulls, p, subject="S25"),
                                "S25's A against its own nulls")  # fmt: skip
    board = c.board
    drawn["leaderboard"] = (
        lambda p: sf.leaderboard_chart(board, p),
        "Every cell by S, with S25 and the regions",
    )
    marginals = pd.read_csv(c.study.results("marginals.csv"), comment="#")
    drawn["marginals"] = (
        lambda p: sf.marginals_chart(marginals, p),
        "The median cell per axis value",
    )


def _fill_rules(c: Any, drawn: dict[str, tuple[Draw, str]], written: Any) -> None:
    """A0's order stream on O under the three fill rules, and S-P's and S25-Z's on H."""
    from trading_research.reporting.search_figures import fill_rule_chart

    found = (c.c or {}).get("a0_and_fill") or {}
    if not found.get("run"):
        written.wait("the fill-rule figure", "A0 and FILL have not run on O")
        return
    streams = {"A0 on O": pd.DataFrame(found["fill_summary"])}
    h = c.read.get("H")
    if h is not None:
        on_h = h.get("fill_rules") or {}
        if not on_h:
            written.wait("the fill-rule figure", "S-P's and S25-Z's streams are not in H's results")
        for name, rows in list(on_h.items())[:2]:
            streams[f"{name} on H"] = pd.DataFrame(rows)
    drawn["fill_rules"] = (lambda p: fill_rule_chart(streams, p),
                           "The fill ratio of the same orders under three fill rules")  # fmt: skip


def _daily(c: Any) -> dict[str, pd.Series]:
    """Each slot's daily series through the year: D's walk-forward, then each block read."""
    from datetime import date

    from trading_research.pipeline.short_horizon.page_numbers import d_daily

    parts: dict[str, list[pd.Series]] = {}
    if c.d is not None and c.c is not None:
        for slot, series in d_daily(c.study).items():
            parts.setdefault(slot, []).append(series)
    for block in c.read:
        for slot, numbers in (c.read[block].get("pages") or {}).get("slots", {}).items():
            daily = numbers["daily"]
            index = [date.fromisoformat(d) for d in daily["days"]]
            values = [float("nan") if v is None else float(v) for v in daily["value"]]
            parts.setdefault(slot, []).append(pd.Series(values, index=index, dtype="float64"))
    return {slot: pd.concat(found) for slot, found in parts.items()}


def _alarms(c: Any) -> list[pd.Timestamp]:
    """D's fold-wise alarms in their months, then the frozen detector's after D."""
    from trading_research.data import study_access as sa
    from trading_research.pipeline.short_horizon.page_numbers import d_alarms

    stamps = d_alarms(c.study) if c.d is not None else []
    after = pd.Timestamp(sa.BLOCKS["D"][1], tz="UTC") + pd.Timedelta(days=1)
    latest = [r for r in c.read.values() if r.get("pages")]
    if latest:
        frozen = [pd.Timestamp(a["timestamp"]) for a in latest[-1]["pages"]["alarms"]]
        stamps += [t for t in frozen if t >= after]
    return sorted(stamps)


def _regime_maps(c: Any, drawn: dict[str, tuple[Draw, str]], written: Any) -> None:
    from trading_research.pipeline.short_horizon.pages import held_out_days
    from trading_research.reporting.regime_map import regime_map_chart

    daily = _daily(c)
    if not daily:
        written.wait("the regime map's figures", "no slot has a daily series yet")
        return
    alarms, blocks = _alarms(c), held_out_days(c)
    for name, (slots, units) in EXECUTION_PATHS.items():
        mine = {s: daily[s] for s in slots if s in daily}
        if mine:
            title = f"The regime map, {units}, cut at alarms"
            draw = partial(regime_map_chart, mine, alarms, blocks=blocks, title=title)
            drawn[f"regime_map_{name}"] = (draw, f"The regime map for the {name} path")


def _break_view(c: Any, drawn: dict[str, tuple[Draw, str]], written: Any) -> None:
    """The stream from 2024-12-01 to the last block read, with every alarm, b,
    F- and F+, and each guarded slot's guard windows."""
    from trading_research.data import study_access as sa
    from trading_research.pipeline.short_horizon.page_numbers import close_path
    from trading_research.reporting.break_view import break_view_chart

    latest = [block for block, r in c.read.items() if r.get("pages")]
    if not latest or not close_path(c.study, latest[-1]).exists():
        written.wait("the break-detector view", "no block read has kept its hourly close")
        return
    block = latest[-1]
    close = pd.read_parquet(close_path(c.study, block))["close"]
    alarms = pd.DataFrame(c.read[block]["pages"]["alarms"])
    if alarms.empty:
        alarms = pd.DataFrame(columns=["timestamp", "statistic"])
    h = (c.read.get("H") or {}).get("pages", {}).get("slots", {})
    guards = {slot: int(n["window_h"]) for slot, n in h.items() if int(n.get("window_h", 0)) > 0}
    b = ((c.read.get("F") or {}).get("split") or {}).get("b")
    spans = {name: sa.BLOCKS[name] for name in ("W", "D", "H", "F", "P")}
    written.wait("the break-detector view's 1 s CUSUM restarts", "no read records them")
    drawn["break_view"] = (
        lambda p: break_view_chart(close, alarms, p, blocks=spans, p_opened="P" in c.read, b=b,
                                   guards=guards),
        "BTCUSDT's hourly close with the frozen detector's alarms, b and the guard windows",
    )  # fmt: skip


def render(c: Any, ladder: pd.DataFrame, written: Any) -> tuple[dict[str, str], list[Path]]:
    """Every figure the computed results can draw, in both themes: the README
    ``<picture>`` block of each (as the pages file reaches it) and the files."""
    from trading_research.reporting.plots import both_themes
    from trading_research.reporting.search_figures import edge_ladder_chart, picture

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        written.wait("figures", "matplotlib is not installed")
        return {}, []
    drawn: dict[str, tuple[Draw, str]] = {}
    _search(c, drawn, written)
    drawn["edge_ladder"] = (
        lambda p: edge_ladder_chart(ladder, p),
        "S25's edge ladder, one change at each step",
    )
    _fill_rules(c, drawn, written)
    _regime_maps(c, drawn, written)
    _break_view(c, drawn, written)
    figures, paths = {}, []
    for name, (draw, alt) in drawn.items():
        paths += both_themes(draw, c.study.repo / IMAGES / f"{PREFIX}{name}.png")
        figures[name] = picture(f"{PREFIX}{name}", alt, images=FROM_PAGES)
    return figures, paths
