"""The results page, written from the results files: one row per slot and measured row.

Registered ("Artefacts and headline sentences"): a page with one row per slot
and measured row (status, first kill against its reference, mean, t,
Newey-West t, median day, trades, MDE, placebo percentile, K-nbhd median,
break-even per side against every tier, the brackets and the calibration), the
edge ladder, and the headline sentences the read wrote, never edited. The page
is ``experiments/results/spot_2026_page.md``; it is written by every read and
again by the ``pages`` command, from the same files, to the same bytes.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from trading_research.spot import registration as sr
from trading_research.spot.context import SpotStudy
from trading_research.spot.plan import ROWS

PAGE_PATH = Path("experiments/results/spot_2026_page.md")
MEASURED = "measured beside the family; no status"


def _n(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int | float):
        return f"{value:.{digits}f}"
    return str(value)


def _first_kill(kills: Sequence[Mapping[str, Any]], name: str | None) -> str:
    if not name:
        return "none"
    found = next((k for k in kills if k["name"] == name), None)
    if found is None:
        return name
    return f"{name} at {_n(found['value'])} against {_n(found['reference'])}"


def _measured_first(kills: Sequence[Mapping[str, Any]]) -> str:
    fired = [k for k in kills if k.get("fired")]
    return _first_kill(kills, fired[0]["name"]) if fired else "none"


def block_lines(results: Mapping[str, Any]) -> list[str]:
    """The block's table, its edge ladder and its sentences."""
    block = results["block"]
    m = results.get("measurements", {})
    rows = m.get("rows", {})
    lines = [f"## {block} ({results.get('stamp')})", "", "### Headline sentences", ""]
    for found in (m.get("sentences") or {}).values():
        lines += [f"- {s}" for s in found] if isinstance(found, list) else []
    lines += ["", "### Slots and measured rows", "",
              "| Row | Status | First kill | Mean | t | NW t | Median day | Trades | MDE | "
              "Placebo pct | K-nbhd median | Break-even bp/side | Every tier | Brackets |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for name in ROWS:
        found = rows.get(name, {})
        status = results.get("statuses", {}).get(name)
        if status is not None:
            state, kill = (
                status["status"],
                _first_kill(status.get("kills", []), status.get("first_kill")),
            )
            mde = status.get("mde")
        else:
            state, mde = MEASURED, None
            kill = _measured_first(m.get("measured_rows_kills", {}).get(name, []))
        tiers = ", ".join(f"{t} {_n(v)}" for t, v in (found.get("tiers") or {}).items())
        brackets = ", ".join(f"{b} {_n(v)}" for b, v in (found.get("brackets") or {}).items())
        lines.append(
            f"| {name} | {state} | {kill} | {_n(found.get('mean'))} | {_n(found.get('t'))} | "
            f"{_n(found.get('newey_west_t'))} | {_n(found.get('median'))} | {found.get('trades', 'n/a')} | "
            f"{_n(mde)} | {_n(found.get('placebo_percentile'), 1)} | "
            f"{_n(found.get('neighbourhood_median'))} | {_n(found.get('fee_break_even_bp'))} | "
            f"{tiers or 'n/a'} | {brackets or 'n/a'} |"
        )
    calib = m.get("calibration", {})
    lines += ["", f"Thinned calibration: a_buy {_n(calib.get('a_buy'))} bp, "
              f"a_sell {_n(calib.get('a_sell'))} bp.", "", "### Edge ladder", "",
              "| Step | Zero fees | Verdict tier | Cheapest dated tier |", "|---|---|---|---|"]  # fmt: skip
    for step in m.get("edge_ladder", []):
        lines.append(f"| {step['label']} | {_n(step['zero_fees'])} | {_n(step['verdict'])} | "
                     f"{step['cheapest_tier'] or 'n/a'} {_n(step['at_cheapest'])} |")  # fmt: skip
    return [*lines, ""]


def publish(study: SpotStudy) -> dict[str, Any]:
    """Write the page from every results file there is; the paths and the sentences."""
    blocks = []
    for relative in sr.RESULTS_PATHS.values():
        path = study.repo / relative
        if path.exists():
            blocks.append(json.loads(path.read_text(encoding="utf-8")))
    if not blocks:
        return {"written": [], "sentences": {}}
    lines = ["# S25 on spot, 2026: results", "",
             "Written by `experiments/spot_2026.py` from the results files of the reads; "
             "every number beside the verdict tier is a measurement.", ""]  # fmt: skip
    for results in blocks:
        lines += block_lines(results)
    target = study.repo / PAGE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")
    sentences = {r["block"]: r.get("measurements", {}).get("sentences", {}) for r in blocks}
    return {"written": [PAGE_PATH.as_posix()], "sentences": sentences}
