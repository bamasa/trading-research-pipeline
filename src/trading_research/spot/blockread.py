"""What a block's read computes from the frozen files: statuses, kills and measurements.

:func:`frozen_models` loads every row's frozen fits and thresholds as the frozen
configuration records them, each file checked against its sha256 (the fits, the
fitted-values file; S25-T's 2025 files against the 2025 configuration).
:func:`compute` reads the block's days (:func:`trading_research.spot.heldout.read_days`),
decides which venue is unreadable (more than a quarter of its days excluded),
assembles each row's daily series, applies the status rules
(:func:`trading_research.spot.statuses.on_h` or ``on_p``, with the thinned
calibration's ``a_buy`` and ``a_sell``), evaluates the measured rows' kills, and
gathers the measurements beside them: every row's series statistics, the
verdicts on Bybit with the gap-excluded days put back, both calibrations, the
excluded days by venue and reason, the venues' prints a day and the edge ladder.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import date
from typing import Any

import numpy as np

from trading_research.spot import access as sa
from trading_research.spot import assemble as asm
from trading_research.spot import heldout as ho
from trading_research.spot import loader as ld
from trading_research.spot import statuses as st
from trading_research.spot.context import SpotStudy
from trading_research.spot.costs import FeeTable
from trading_research.spot.fits import frozen_2025, load_model
from trading_research.spot.ledger import FITTED_VALUES_PATH, WEIGHTS_DIR, file_sha256
from trading_research.spot.plan import ROWS, fit_key
from trading_research.spot.registration import FAMILY, MEASURED_ROWS
from trading_research.spot.runs import RowModels
from trading_research.strategies.hold_first import HoldFirstRule

#: The edge ladder's steps, each labelled with the one thing that changed.
LADDER: tuple[tuple[str, str], ...] = (
    ("S25-T", "S25-T on spot books (no refit)"),
    ("S25-BB", "S25-BB (refit on spot books)"),
    ("S25-PB book walk", "S25-PB by the book walk (prints-only inputs, same venue)"),
    ("S25-PB", "S25-PB by the print rule"),
    ("S25-PF", "S25-PF (the other venue)"),
)


def frozen_models(study: SpotStudy, frozen: Any) -> dict[str, RowModels]:
    """Every row's frozen fits and thresholds; ``frozen.value(path)`` gives the frozen
    values (the frozen registration, or the values the amendment is about to write)."""
    from trading_research.spot import days as dy

    fitted_path = study.repo / FITTED_VALUES_PATH
    digest = frozen.value(("features", "fitted_values_sha256"))
    if not fitted_path.is_file() or file_sha256(fitted_path) != digest:
        raise RuntimeError(f"{FITTED_VALUES_PATH} is not at its frozen sha256 {digest}")
    touch = float(json.loads(fitted_path.read_text(encoding="utf-8"))["touch_size"])
    weights = frozen.value(("models", "weights_sha256"))
    thresholds = frozen.value(("models", "thresholds"))
    out = {}
    for name, row in ROWS.items():
        if not row.fitted:
            continue
        models = {}
        for hold, fee in row.fits:
            key = fit_key(name, hold, fee)
            if key in weights:
                models[(hold, fee)] = load_model(study.repo / WEIGHTS_DIR / key, weights[key])
        rules = {c: HoldFirstRule(**thresholds[name][c.key]) for c in row.cells}
        out[name] = RowModels(name, models, rules, dy.layout(row.store, touch))
    out["S25-T"] = frozen_2025(study)
    return out


def cheapest_tier(table: FeeTable, name: str, days: Sequence[date]) -> str | None:
    """The tier with the lowest taker fee that a dated source covers on every day."""
    best: tuple[float, str] | None = None
    for tier in table.tiers(name):
        if days and all(table.covered(name, d, tier) for d in days):
            fee = max(table.day_fees(name, d, tier).taker_bp for d in days)
            best = min(best, (fee, tier)) if best else (fee, tier)
    return None if best is None else best[1]


def _mean_of(rows: Sequence[Mapping[str, Any]], *keys: str) -> float | None:
    values = []
    for r in rows:
        node: Any = r
        for k in keys:
            node = None if node is None else node.get(k)
        values.append(np.nan if node is None else float(node))
    found = np.asarray(values, dtype=float)
    found = found[np.isfinite(found)]
    return float(found.mean()) if len(found) else None


def ladder(records: Mapping[str, Any], table: FeeTable) -> list[dict[str, Any]]:
    """The edge ladder: each step at zero fees, the verdict tier and the cheapest tier."""
    out = []
    for step, label in LADDER:
        name = step.split()[0]
        venue = ROWS[name].instrument
        days = asm.kept_days(records, venue)
        if step.endswith("book walk"):
            rows = [records[d].get("calibration", {}).get("pb_book", {}) for d in days]
        else:
            rows = [records[d]["rows"].get(name, {}) for d in days]
        cheapest = cheapest_tier(table, venue, [date.fromisoformat(d) for d in days])
        out.append({"step": step, "label": label, "zero_fees": _mean_of(rows, "tiers", "zero_fees"),
                    "verdict": _mean_of(rows, "value"), "cheapest_tier": cheapest,
                    "at_cheapest": None if cheapest is None else _mean_of(rows, "tiers", cheapest)})  # fmt: skip
    return out


def put_back_view(records: Mapping[str, Any], name: str) -> dict[str, Any]:
    """A Bybit verdict with the gap-excluded days put back (a measurement)."""
    kept = asm.kept_days(records, sa.BYBIT)
    days = sorted([*kept, *(d for d, r in records.items() if "put_back" in r)])
    values = []
    for d in days:
        r = records[d]
        found = (r["put_back"] if "put_back" in r else r["rows"]).get(name, {})
        values.append(np.nan if found.get("value") is None else float(found["value"]))
    stats = st.day_statistics(
        np.asarray(values, dtype=float), [date.fromisoformat(d) for d in days]
    )
    return {"days": len(days), "put_back": len(days) - len(kept), **asdict(stats)}


def compute(
    study: SpotStudy,
    plan: ho.ReadPlan,
    *,
    blind: bool,
    mde: Mapping[str, float],
    study_voids: Sequence[str] = (),
    h_statuses: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """The block's statuses (H, or P given H's statuses), the measured rows' kills and
    every measurement; ``rows`` holds each row's SlotDays."""
    records = ho.read_days(study, plan)
    excluded = {name: asm.excluded(records, name, plan.days) for name in ho.VENUE_ROWS}
    unreadable = [name for name, found in excluded.items() if ld.unreadable(found, len(plan.days))]
    rows = {r: asm.slot_days(records, r, mde.get(r)) for r in ROWS}
    folder = study.work / "reads" / plan.label
    calibration = asm.calibration([folder], asm.kept_days(records, sa.BYBIT))
    a_buy, a_sell = float(calibration["a_buy"]), float(calibration["a_sell"])
    family = {r: rows[r] for r in FAMILY}
    if h_statuses is None:
        found = st.on_h(family, blind=blind, unreadable=unreadable, study_voids=study_voids,
                        a_buy=a_buy, a_sell=a_sell)  # fmt: skip
    else:
        found = st.on_p(family, h_statuses, unreadable=unreadable, study_voids=study_voids,
                        a_buy=a_buy, a_sell=a_sell)  # fmt: skip
    measured = {r: [asdict(k) for k in st.measured_kills(rows[r], a_buy, a_sell)]
                for r in MEASURED_ROWS if ROWS[r].instrument not in unreadable}  # fmt: skip
    not_assessed = {r: st.unreadable_status(ROWS[r].instrument, plan.block)
                    for r in MEASURED_ROWS if ROWS[r].instrument in unreadable}  # fmt: skip
    calibration["thinned_capacity_median"] = asm.thinned_capacity_median(records)
    calibration["net_bp_trade_both_ways"] = asm.both_ways(records)
    measurements = {
        "rows": {r: asm.measurements(records, s) for r, s in rows.items()},
        "measured_rows_kills": measured,
        "measured_rows_not_assessed": not_assessed,
        "gap_days_put_back": {r: put_back_view(records, r) for r in ho.PUT_BACK_ROWS},
        "calibration": calibration,
        "excluded": {sa.instrument(n).key: {d.isoformat(): list(v) for d, v in e.items()}
                     for n, e in excluded.items()},
        "unreadable": [sa.instrument(n).key for n in unreadable],
        "prints_a_day": {sa.instrument(n).key: _mean_of([r["venues"][n] for r in records.values()],
                                                        "prints") for n in ho.VENUE_ROWS},
        "edge_ladder": ladder(records, study.fee_table),
        "days_planned": len(plan.days),
        "s25_t_rate": {"trades_per_day": st.trades_per_day(rows["S25-T"]), "registered": 240},
    }  # fmt: skip
    return {"assessments": found, "rows": rows, "measurements": measurements,
            "calibration_advantage": asm.rule_advantage(calibration)}  # fmt: skip


def sentences(
    found: Mapping[str, Any], after_p: Mapping[str, str] | None = None
) -> dict[str, list[str]]:
    """The registered headline sentences the outcome calls for."""
    rows = found["rows"]
    s25_pf = st.fee_break_even(rows["S25-PF"]) if rows["S25-PF"].days else math.nan
    s25_t = st.trades_per_day(rows["S25-T"]) if rows["S25-T"].days else math.nan
    return st.headline_sentences(found["assessments"], s25_pf_gross_bp_turnover=s25_pf,
                                 s25_t_trades_per_day=s25_t,
                                 calibration=found["calibration_advantage"], after_p=after_p)  # fmt: skip
