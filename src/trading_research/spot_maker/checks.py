"""Stage 0's checks: the readers' proof on a day and the fill rule on every resting order.

* :func:`reader_proof`: the loader's findings on a reader-test day, the print
  contract (``spot/prints.check``), both aggressor sides present and the tick
  equal to the registered 0.01.
* :func:`order_checks`: every resting order an audited run posted, re-filled
  from its side, price, size and live window: its fills inside its life and
  size, the fills the run recorded exactly those its own rule gives, and the
  same order's fills nested, delayed within trade-through within touch.
* :func:`audit_day`: each slot's run under each of the three rules, audited,
  with its void conditions (the account's identity among them).
* :func:`rule_checks`: probe orders at every 60 s instant, one a side at its
  proxy, under the three rules.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from trading_research.spot.fills import PrintTape
from trading_research.spot.prints import PrintContractError, check
from trading_research.spot_maker import copies as cp
from trading_research.spot_maker import day as dd
from trading_research.spot_maker import money as mn
from trading_research.spot_maker import registration as rg
from trading_research.spot_maker import resting
from trading_research.spot_maker.executor import DayRun, settings_for
from trading_research.spot_maker.touch import Proxies, quote_price

SIZE_TOL = 1e-12


def _nested(sums: dict[str, float]) -> bool:
    return (sums["delayed"] <= sums["trade_through"] + SIZE_TOL
            and sums["trade_through"] <= sums["touch"] + SIZE_TOL)  # fmt: skip


def rule_checks(tape: PrintTape, proxies: Proxies, step_s: float = 60.0) -> dict[str, Any]:
    """The fill rule on probe orders: at every instant an order of each side at its
    proxy, its fills under the three rules nested, inside its life and size."""
    step = round(step_s * rg.NS_PER_S)
    instants = np.arange(tape.start_ns + step, tape.start_ns + 86_340 * rg.NS_PER_S, step)
    asks, bids = proxies.at(instants)
    bad, orders = 0, 0
    for tau, ask, bid in zip(instants, asks, bids, strict=True):
        start, end = int(tau) + rg.LATENCY_NS, int(tau) + step + rg.LATENCY_NS
        for side in (1, -1):
            price = quote_price(side, float(ask), float(bid), 0.0, skew_bp=0.0, clip=rg.CLIP,
                                tick=rg.TICK)  # fmt: skip
            if not np.isfinite(price):
                continue
            orders += 1
            got = {r: resting.fill(tape, side, price, rg.CLIP, start, end, r) for r in resting.RULES}
            sums = {r: float(v[1].sum()) for r, v in got.items()}
            for idx, size in got.values():
                stamps = tape.ts[idx]
                bad += int(bool(((stamps < start) | (stamps >= end)).any())
                           or float(size.sum()) > rg.CLIP + SIZE_TOL)  # fmt: skip
            bad += int(not _nested(sums))
    return {"orders": orders, "violations": bad, "passes": bad == 0}


def order_checks(tape: PrintTape, run: DayRun, rule: str) -> dict[str, Any]:
    """Every resting order a run posted under ``rule``: its fills inside its life and
    size, the fills it recorded those ``rule`` gives, and its fills under the three
    rules nested."""
    if run.orders is None:
        raise ValueError("the run was not audited")
    bad = 0
    for side, price, size, start, end, stamps, sizes in run.orders:
        got = {r: resting.fill(tape, side, price, size, start, end, r) for r in resting.RULES}
        sums = {r: float(v[1].sum()) for r, v in got.items()}
        own_idx, own_sizes = got[rule]
        outside = bool(((stamps < start) | (stamps >= end)).any())
        over = float(sizes.sum()) > size + SIZE_TOL
        same = bool(np.array_equal(tape.ts[own_idx], stamps)
                    and np.allclose(own_sizes, sizes, rtol=0.0, atol=SIZE_TOL))  # fmt: skip
        bad += int(outside or over or not same or not _nested(sums))
    return {"orders": len(run.orders), "violations": bad}


def audit_day(ctx: dd.DayContext) -> dict[str, Any]:
    """Each slot's run under each rule, audited: its orders and its void conditions."""
    out: dict[str, Any] = {}
    for slot in rg.FAMILY:
        spec, found = rg.SLOTS[slot], {}
        for rule in resting.RULES:
            run = ctx.runner(slot)(settings_for(slot, rule=rule, audit=True), None, "both")
            fills = run.log.day(ctx.pricing)
            voids = mn.voids(fills, first_price=ctx.first, last_price=ctx.last, unit=spec.unit,
                             table=ctx.table, tier=ctx.pricing.tier)  # fmt: skip
            found[rule] = {**order_checks(ctx.tape, run, rule), "voids": voids}
        out[slot] = found
    return out


def reader_proof(loaded: cp.DayInput) -> dict[str, Any]:
    """The loader's findings and the contract's checks on one reader-test day."""
    frame = loaded.loaded.frame
    found: dict[str, Any] = {
        "status": "usable" if loaded.usable else "excluded", "exclusions": list(loaded.reasons),
        "rows": 0 if frame is None else len(frame), "tick": loaded.loaded.tick,
        "notes": list(loaded.loaded.notes), "rpi_rows": loaded.loaded.rpi_rows,
        "rows_outside_day": loaded.loaded.rows_outside_day, "sides_present": False,
    }  # fmt: skip
    contract = frame is not None
    if frame is not None:
        try:
            check(frame)
        except PrintContractError as exc:
            contract, found["contract_error"] = False, str(exc)
        aggressor = frame["aggressor"].to_numpy()
        found["sides_present"] = bool((aggressor == 1).any() and (aggressor == -1).any())
    found["contract"] = contract
    tick = loaded.loaded.tick
    found["tick_registered"] = tick is not None and abs(float(tick) - rg.TICK) < 1e-12
    return found
