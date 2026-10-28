"""The results page, written from the results files: one row per slot and measured row.

Registered ("Artefacts and headline sentences"): a page with one row per slot
and measured row (status, first kill against its reference, mean, t,
Newey-West t, median day, trades, MDE, placebo percentile, K-nbhd median,
break-even per side against every tier, the brackets and the calibration), the
edge ladder, and the headline sentences the read wrote, never edited. The page
is ``experiments/results/spot_2026_page.md``; it is written by every read and
again by the ``pages`` command, from the same files, to the same bytes.

The labels the registration requires ("Fees", "Metrics", "Status rules"), added
by the presentation amendment of 2026-10-27 after H was read, change no number:
every tier but the verdict tier is labelled with the status it requires, a zero
fee at a tier "conditional on <tier> eligibility", the result at zero fees
"conditional on a zero-fee venue"; a tier no sourced interval lists on any day of
the block is named, not valued; each row's 30-day traded volume stands beside its
tiers; and the page carries the registered caveat on zero taker fees.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from trading_research.spot import access as sa
from trading_research.spot import registration as sr
from trading_research.spot.context import SpotStudy
from trading_research.spot.costs import FeeTable, load_fee_table
from trading_research.spot.plan import ROWS

PAGE_PATH = Path("experiments/results/spot_2026_page.md")
MEASURED = "measured beside the family; no status"
#: The registered labels: the result at zero fees ("Metrics", ``gross_bp_trade``), and
#: the caveat beside it ("Status rules"; ``statuses.no_zero_taker_cell``).
ZERO_FEES = "conditional on a zero-fee venue"
CAVEAT = (
    "No number at a tier other than the verdict tier, at zero fees, under a bracket or the "
    "long-only variant, with the gap-excluded days put back, of a measured row, or in the "
    "inventory's P&L may be described as the strategy working. Such numbers are measurements. "
    "In particular, no account could trade either pair at zero taker fees on any day of W, D "
    "or H as far as the sources show, so a positive gross edge is a statement about a venue "
    "that did not exist for these pairs on those days."
)
VENUE = {sa.BINANCE: "Binance", sa.BYBIT: "Bybit"}
PAIR = {sa.BINANCE: "Binance BTC/FDUSD", sa.BYBIT: "Bybit spot BTCUSDT"}
_TIER_NAMES = {"regular": "Regular", "non_vip": "non-VIP", "vip2_to_vip9": "VIP 2 to 9",
               "spot_liquidity_provider": "Spot Liquidity Provider"}  # fmt: skip
#: {instrument: {tier: its taker fees on the block's days it covers}}: the tiers dated
#: for a block; None when the page is written without the fee table.
Dated = dict[str, dict[str, tuple[float, ...]]] | None


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


def _sides(records: Sequence[Mapping[str, Any]]) -> str:
    """A FILL-S summary's mean advantage per side, buy then sell."""
    by = {r.get("side"): r.get("mean_advantage_bp") for r in records}
    return f"buy {_n(by.get('buy'))}, sell {_n(by.get('sell'))}"


def tier_name(tier: str) -> str:
    """A tier's name as the venue writes it: ``vip3`` is VIP 3."""
    if tier in _TIER_NAMES:
        return _TIER_NAMES[tier]
    return f"VIP {tier[3:]}" if tier.startswith("vip") and tier[3:].isdigit() else tier


def dated_tiers(table: FeeTable | None, block: str) -> Dated:
    """Each venue's tiers that a sourced interval lists on a day of ``block``, with
    their taker fees on those days."""
    if table is None or block not in sa.BLOCKS:
        return None
    days = sa.block_days(block)
    out: dict[str, dict[str, tuple[float, ...]]] = {}
    for instrument in sa.INSTRUMENTS:
        found = out.setdefault(instrument, {})
        for tier in table.tiers(instrument):
            covered = [d for d in days if table.covered(instrument, d, tier)]
            if covered:
                fees = {table.day_fees(instrument, d, tier).taker_bp for d in covered}
                found[tier] = tuple(sorted(fees))
    return out


def tier_label(instrument: str | None, tier: str, fees: Sequence[float] = ()) -> str:
    """What a tier's number is conditional on: the verdict tier is the one an account
    holds without volume or holdings; a zero fee at another tier is conditional on its
    eligibility; any other tier requires its status."""
    if instrument is not None and tier == sr.VERDICT_TIER.get(instrument):
        return "verdict tier"
    named = " ".join(n for n in (VENUE.get(instrument or ""), tier_name(tier)) if n)
    if fees and max(fees) == 0.0:
        return f"conditional on {named} eligibility"
    return f"requires {named} status"


def tier_cell(name: str, tiers: Mapping[str, Any], dated: Dated) -> str:
    """The row's ``net_bp_trade`` at every tier dated for the block, each labelled,
    then at zero fees, labelled; a tier not dated for the block is left out."""
    instrument = sr.INSTRUMENT_OF[name]
    on_venue = None if dated is None else dated.get(instrument, {})
    parts = [
        f"{tier} {_n(value)} ({tier_label(instrument, tier, (on_venue or {}).get(tier, ()))})"
        for tier, value in tiers.items()
        if tier != "zero_fees" and (on_venue is None or tier in on_venue)
    ]
    if "zero_fees" in tiers:
        parts.append(f"zero fees {_n(tiers['zero_fees'])} ({ZERO_FEES})")
    return "; ".join(parts) or "n/a"


def volume_cell(name: str, volume: Any) -> str:
    """The row's 30-day traded volume, in millions of the pair's quote currency."""
    if not isinstance(volume, int | float):
        return "n/a"
    return f"{volume / 1e6:.2f}M {sa.INSTRUMENTS[sr.INSTRUMENT_OF[name]].quote}"


def tier_lines(block: str, rows: Mapping[str, Any], dated: Dated) -> list[str]:
    """What the tier labels mean, the tiers not dated for the block, and the volume."""
    lines = ["### Tiers and the 30-day volume", "",
             "Every tier but the verdict tier (Binance's Regular, Bybit's non-VIP, the tier an "
             "account holds without 30-day volume, coin holdings or programme membership) is a "
             "measurement, labelled with the status it requires. A VIP status depends on "
             "30-day volume and coin holdings at thresholds the registration does not "
             "transcribe, so no tier other than the verdict tier is said to be within a "
             "slot's reach; a zero fee at a tier would read \"conditional on <tier> "
             f"eligibility\". The result at zero fees is {ZERO_FEES}.", ""]  # fmt: skip
    if dated is not None:
        for instrument in sa.INSTRUMENTS:
            listed = {t for r, found in rows.items() if sr.INSTRUMENT_OF.get(r) == instrument
                      for t in (found.get("tiers") or {}) if t != "zero_fees"}  # fmt: skip
            missing = sorted(listed - set(dated.get(instrument, {})))
            if missing:
                lines += [f"Not dated for {block}: {PAIR[instrument]}'s {_listed(missing)}; no "
                          f"sourced interval lists them on any day of {block}, so no value is "
                          "given at them.", ""]  # fmt: skip
    lines += [f"Each row's 30-day traded volume is 30 times its mean daily turnover on {block}, "
              "in the pair's quote currency: a measurement, which never moves the verdict "
              "tier.", ""]  # fmt: skip
    return lines


def _listed(items: Sequence[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def calibration_cell(name: str, calib: Mapping[str, Any]) -> str:
    """The row's calibration: S25-PF's and S25-PFT's thinned a_buy and a_sell (what
    K-fill charges), S25-PB's and S25-BB's FILL-S mean advantage per side under the
    rule and the worse-of-two bracket."""
    if name in ("S25-PF", "S25-PFT"):
        return f"thinned a_buy {_n(calib.get('a_buy'))}, a_sell {_n(calib.get('a_sell'))}"
    found = calib.get("fill_s", {}).get(name)
    if not found:
        return "n/a"
    return "; ".join(f"{b} {_sides(found.get(b, []))}" for b in ("rule", "worse_of_two"))


def block_lines(results: Mapping[str, Any], table: FeeTable | None = None) -> list[str]:
    """The block's table, its edge ladder and its sentences; with the fee table, a tier
    not dated for the block is named and not valued."""
    block = results["block"]
    dated = dated_tiers(table, block)
    m = results.get("measurements", {})
    rows = m.get("rows", {})
    lines = [f"## {block} ({results.get('stamp')})", "", "### Headline sentences", ""]
    for found in (m.get("sentences") or {}).values():
        lines += [f"- {s}" for s in found] if isinstance(found, list) else []
    lines += ["", "### Slots and measured rows", "",
              "| Row | Status | First kill | Mean | t | NW t | Median day | Trades | MDE | "
              "Placebo pct | K-nbhd median | Break-even bp/side | Every tier, labelled | "
              "30-day traded volume | Brackets | Calibration |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    calib = m.get("calibration", {})
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
            state = m.get("measured_rows_not_assessed", {}).get(name, state)
            kill = _measured_first(m.get("measured_rows_kills", {}).get(name, []))
        tiers = tier_cell(name, found.get("tiers") or {}, dated)
        volume = volume_cell(name, found.get("volume_30_day_quote"))
        brackets = ", ".join(f"{b} {_n(v)}" for b, v in (found.get("brackets") or {}).items())
        lines.append(
            f"| {name} | {state} | {kill} | {_n(found.get('mean'))} | {_n(found.get('t'))} | "
            f"{_n(found.get('newey_west_t'))} | {_n(found.get('median'))} | {found.get('trades', 'n/a')} | "
            f"{_n(mde)} | {_n(found.get('placebo_percentile'), 1)} | "
            f"{_n(found.get('neighbourhood_median'))} | {_n(found.get('fee_break_even_bp'))} | "
            f"{tiers} | {volume} | {brackets or 'n/a'} | {calibration_cell(name, calib)} |"
        )
    both = calib.get("net_bp_trade_both_ways", {})
    lines += ["", f"Thinned calibration: a_buy {_n(calib.get('a_buy'))} bp, "
              f"a_sell {_n(calib.get('a_sell'))} bp; capacity median "
              f"{_n(calib.get('thinned_capacity_median'), 4)}.", ""]  # fmt: skip
    for slot in ("S25-BB", "S25-PB"):
        found = both.get(slot, {})
        lines.append(f"FILL-S {slot}, net_bp_trade: book walk {_n(found.get('book_walk'))}, "
                     f"print rule {_n(found.get('print_rule'))}.")  # fmt: skip
    for slot, found in calib.get("fill_s", {}).items():
        lines.append(f"FILL-S {slot}, mean advantage over the book walk (bp of mid): " + "; ".join(
            f"{b} {_sides(found.get(b, []))}" for b in found) + ".")  # fmt: skip
    lines += ["", *tier_lines(block, rows, dated), "### Edge ladder", "",
              f"| Step | Zero fees ({ZERO_FEES}) | Verdict tier | Cheapest dated tier, labelled |",
              "|---|---|---|---|"]  # fmt: skip
    for step in m.get("edge_ladder", []):
        lines.append(f"| {step['label']} | {_n(step['zero_fees'])} | {_n(step['verdict'])} | "
                     f"{ladder_tier(step, dated)} |")  # fmt: skip
    return [*lines, ""]


def ladder_tier(step: Mapping[str, Any], dated: Dated) -> str:
    """The ladder's cheapest dated tier, its value and its label."""
    tier = step.get("cheapest_tier")
    if not tier:
        return f"n/a {_n(step.get('at_cheapest'))}"
    slot = (str(step.get("step") or step.get("label") or "").split() or [""])[0]
    instrument = sr.INSTRUMENT_OF.get(slot)
    fees = (dated or {}).get(instrument or "", {}).get(tier, ())
    return f"{tier} {_n(step.get('at_cheapest'))} ({tier_label(instrument, tier, fees)})"


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
             "every number beside the verdict tier is a measurement.", "", CAVEAT, ""]  # fmt: skip
    config = study.repo / sr.CONFIG_PATH
    table = load_fee_table(config) if config.exists() else None
    for results in blocks:
        lines += block_lines(results, table)
    final = final_lines(study)
    if final:
        lines += final
    target = study.repo / PAGE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")
    sentences = {r["block"]: r.get("measurements", {}).get("sentences", {}) for r in blocks}
    return {"written": [PAGE_PATH.as_posix()], "sentences": sentences}


def final_lines(study: SpotStudy) -> list[str]:
    """The family's statuses after P by the registered table, while P is not yet read
    or once its deadline has passed unread (``reads.final_statuses``)."""
    from trading_research.spot.reads import final_statuses

    found = final_statuses(study)
    if not found:
        return []
    lines = ["## After P", ""]
    lines += [f"- {slot}: {status}" for slot, status in found.items()]
    return [*lines, ""]
