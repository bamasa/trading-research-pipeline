"""The study's page, ``experiments/results/spot_maker_2026_page.md``, from its results files.

Per block read: the headline sentences as the results file holds them; one row
per slot of the verdict at the fees charged (status, first kill against its
reference, the mean of the daily ``net_bp_fill``, t, Newey-West t, the median
day, days with a fill and days positive, fills a day, the net per round trip
for MK-S, the net per day, the MDE), with the zero-fee figure only as a
reference column beside it; every kill of each slot against its reference; the
making part and the markouts; the runs beside the verdict (touch, delayed,
110 ms, drift, placebos, the nine cells); the tiers and the flattens; the
activity (fill ratio, capacity, drawdown, the proxies' width, MK-S's counters
and realised hold). Every number but the verdict is a measurement.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from trading_research.spot_maker import registration as rg
from trading_research.spot_maker.context import MakerStudy

INTRO = (
    "Written by `experiments/spot_maker_2026.py` from the results files of the reads; "
    "registered in `docs/preregistration/spot_maker_2026.md`. The verdict of every slot is "
    "`net_bp_fill`, the day's net over its turnover in bp, averaged over the days with a fill, "
    "at the fees a Regular account was charged on each day: 0 bp on a resting order's fill and "
    "10 bp on the 23:59 taker flatten. Resting orders fill only when a print trades through "
    "their price (the queue ahead taken as exhausted). Every other number is a measurement; the "
    "zero-fee figures are references beside the fee-inclusive ones, never a result."
)


def _f(value: Any, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _table(head: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head),
            *("| " + " | ".join(r) + " |" for r in rows)]  # fmt: skip


def _kill_text(kill: Mapping[str, Any] | None) -> str:
    if kill is None:
        return "none"
    return f"{kill['name']} at {_f(kill['value'])} against {_f(kill['reference'])}"


def fill_label(results: Mapping[str, Any]) -> str | None:
    """The calibration's label beside every verdict: None when it passed."""
    found = results.get("calibration")
    if not isinstance(found, Mapping):
        return rg.CALIBRATION_LABEL
    return None if found.get("passes") else str(found.get("label") or rg.CALIBRATION_LABEL)


def calibration_lines(results: Mapping[str, Any]) -> list[str]:
    found = results.get("calibration")
    label = fill_label(results)
    head = ["### The fill rule's calibration against a book", ""]
    if not isinstance(found, Mapping) or found.get("status") != "run":
        status = found.get("status") if isinstance(found, Mapping) else "not recorded"
        return [*head, f"{status}: every verdict above carries the label \"{label}\".", ""]
    rows = [[slot, str(found[slot]["days"]), _f(found[slot]["ordering_share"]),
             _f(found[slot]["bracket_share"]), "yes" if found[slot]["passes"] else "no"]
            for slot in rg.CALIBRATION_SLOTS]  # fmt: skip
    verdict = ("it passes; no verdict carries a label" if label is None
               else f"it does not pass; every verdict above carries the label \"{label}\"")
    return [*head, "Bybit linear BTCUSDT, 2024-02-01 to 25, with a book: the share of days on which "
            "the maker volume filled orders trade-through <= queue <= touch, and on which the "
            f"prints rule's making part (prints thinned to {found.get('thinned_to')} a day) lies "
            f"between the book's trade-through and touch making parts; {verdict}.", "",
            *_table(("Slot", "Days", "Ordered", "Inside the bracket", "Passes"), rows), ""]  # fmt: skip


def verdict_rows(results: Mapping[str, Any]) -> list[list[str]]:
    out = []
    label = fill_label(results)
    for slot in rg.FAMILY:
        status, s = results["statuses"][slot], results["summaries"].get(slot) or {}
        if not s:
            out.append([slot, status["status"], _kill_text(status.get("first_kill"))] + ["n/a"] * 10)
            continue
        trip = _f(s.get("net_bp_round_trip")) if slot == "MK-S" else "no round trips"
        out.append([
            slot, status["status"], _kill_text(status.get("first_kill")),
            _f(s["mean"]) + (f" ({label})" if label else ""),
            _f(s["t"]), _f(s["newey_west_t"]), _f(s["median"]),
            f"{s['days']} of {s['usable_days']}",
            f"{s.get('days_positive', 'n/a')} ({_f(s['share_positive'])})",
            f"{_f(s['fills_a_day'], 1)} ({_f(s.get('maker_fills_a_day'), 1)} / "
            f"{_f(s.get('taker_fills_a_day'), 1)})",
            trip, _f(s["net_bp_clip_day"]), _f(results.get("mde", {}).get(slot)),
            _f(s["gross_bp_fill"]),
        ])  # fmt: skip
    return out


VERDICT_HEAD = ("Slot", "Status", "First kill", "Net bp per fill (verdict)", "t", "NW t",
                "Median day", "Days with a fill", "Days positive (share)",
                "Fills a day (maker / taker)", "Net bp per round trip", "Net bp of a clip a day",
                "MDE", "Gross bp per fill, zero-fee reference")  # fmt: skip


def kill_rows(results: Mapping[str, Any]) -> list[list[str]]:
    out = []
    for slot in rg.FAMILY:
        for kill in results["statuses"][slot].get("kills", []):
            out.append([slot, kill["name"], _f(kill["value"], 4), _f(kill["reference"], 4),
                        "fired" if kill["fired"] else "no"])  # fmt: skip
    return out


def detail_rows(results: Mapping[str, Any]) -> dict[str, list[list[str]]]:
    tables: dict[str, list[list[str]]] = {"making": [], "runs": [], "fees": [], "activity": []}
    for slot in rg.FAMILY:
        s = results["summaries"].get(slot)
        if not s:
            continue
        marks = s.get("markouts") or {}
        tables["making"].append([slot, _f(s["making_part"]), *(_f(marks.get(k)) for k in
                                 ("1.0", "5.0", "30.0")), _f(s["spread_part"]),
                                 _f(s["adverse_part"])])  # fmt: skip
        cells = [c for c in s.get("cells") or [] if c is not None]
        tables["runs"].append([slot, _f(s.get("touch")), _f(s.get("delayed")), _f(s.get("latency")),
                               _f(s.get("all_long")), _f(s.get("all_short")), _f(s.get("flipped")),
                               _f(results.get("placebo_percentile", {}).get(slot), 1),
                               _f(float(np.median(cells)) if cells else None),
                               " / ".join(_f(c) for c in s.get("cells") or [])])  # fmt: skip
        totals = s.get("totals") or {}
        tables["fees"].append([slot, _f(s["mean"]), _f(s.get("tier_vip3")), _f(s.get("tier_vip9")),
                               _f(s.get("tier_zero_fees")), _f(s.get("net_bp_fill_pooled")),
                               _f(totals.get("fees_quote")), str(s.get("flattens", "n/a")),
                               _f(s.get("flatten_notional"))])  # fmt: skip
        tables["activity"].append([slot, _f(s["fill_ratio"], 4), _f(s["capacity"], 5),
                                   _f(s["max_drawdown_quote"]), _f(s.get("width_bp_median"), 4),
                                   _f(totals.get("net_quote")), _f(totals.get("turnover"), 0)])  # fmt: skip
    return tables


def counters_lines(results: Mapping[str, Any]) -> list[str]:
    s = results["summaries"].get("MK-S")
    if not s:
        return []
    c = s.get("counters") or {}
    return [f"MK-S over the block: {c.get('decisions', 0)} decisions; {c.get('entries_posted', 0)} "
            f"entries posted, {c.get('entries_filled', 0)} filled in any part "
            f"({c.get('entries_partial', 0)} in part), {c.get('entries_unfilled', 0)} cancelled "
            f"unfilled; skipped: {c.get('skipped_busy', 0)} busy, {c.get('skipped_hours', 0)} "
            f"outside the hours, {c.get('entries_no_proxy', 0)} without a proxy; "
            f"{c.get('exits_flattened', 0)} exits flattened; realised hold from the entry's first "
            f"fill to the exit's last: mean {_f(s.get('hold_s_mean'), 1)} s, median of the daily "
            f"medians {_f(s.get('hold_s_median_of_days'), 1)} s."]  # fmt: skip


def block_lines(block: str, results: Mapping[str, Any]) -> list[str]:
    lines = [f"## {block} ({results['read']})", "", "### Headline sentences", ""]
    lines += [f"- {h}" for h in results.get("headlines", [])] or ["- none written"]
    lines += ["", "### The verdict, net of the fees charged", ""]
    lines += _table(VERDICT_HEAD, verdict_rows(results))
    lines += ["", *calibration_lines(results)]
    if not results.get("summaries"):
        return lines
    tables = detail_rows(results)
    lines += ["", "### Every kill, in each slot's registered order (value against reference)", ""]
    lines += _table(("Slot", "Kill", "Value", "Reference", "Fired"), kill_rows(results))
    lines += ["", "### The making part and the markouts of the maker fills (bp per fill)", ""]
    lines += _table(("Slot", "Making part (5 s)", "1 s", "5 s", "30 s", "Spread part",
                     "Adverse part"), tables["making"])  # fmt: skip
    lines += ["", "### The runs beside the verdict (block means, bp per fill; measurements)", ""]
    lines += _table(("Slot", "Touch", "Delayed", "110 ms", "All long", "All short",
                     "Sign-flipped", "Placebo pct", "K-nbhd median", "The nine cells"),
                    tables["runs"])  # fmt: skip
    lines += ["", "### Tiers and flattens (bp per fill; only the flattens' fees change)", ""]
    lines += _table(("Slot", "Regular (verdict)", "VIP 3", "VIP 9", "Zero fees (reference)",
                     "Pooled net bp per fill", "Fees paid (FDUSD)", "Flattens",
                     "Flattened notional (FDUSD)"), tables["fees"])  # fmt: skip
    lines += ["", "### Activity", ""]
    lines += _table(("Slot", "Fill ratio", "Capacity (share of printed volume)",
                     "Max drawdown (FDUSD)", "Proxies' width, median of days (bp)",
                     "Net (FDUSD)", "Turnover (FDUSD)"), tables["activity"])  # fmt: skip
    act = results.get("activity") or {}
    lines += ["", *counters_lines(results), "",
              f"Days: {results.get('usable_days')} usable of {results.get('days')}; excluded "
              f"{results.get('excluded') or 'none'}. The venue printed {_f(act.get('prints_a_day'), 0)} "
              f"aggregated prints a day; the held unit's own P&L, never in a verdict, was "
              f"{_f(act.get('inventory_pnl_bp_mean'))} bp a day on average.", ""]  # fmt: skip
    return lines


def page(blocks: Mapping[str, Mapping[str, Any]]) -> str:
    lines = ["# S25-PF as a maker on Binance BTC/FDUSD, 2026: results", "", INTRO, ""]
    for block, results in blocks.items():
        lines += block_lines(block, results)
        if "after_P" in results:
            lines += ["## After P", ""] + [f"- {s}: {v}" for s, v in results["after_P"].items()]
    return "\n".join(lines).rstrip("\n") + "\n"


def write_page(study: MakerStudy) -> Path:
    blocks = {}
    for block, relative in rg.RESULTS_PATHS.items():
        path = Path(study.repo) / relative
        if path.exists():
            blocks[block] = json.loads(path.read_text("utf-8"))
    target = Path(study.repo) / rg.PAGE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(page(blocks), encoding="utf-8")
    return target
