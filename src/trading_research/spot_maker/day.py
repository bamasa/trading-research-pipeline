"""One day of the study: every run of every slot, as one record of plain values.

For each slot (:func:`slot_day`) the true run at the registered settings, and
the runs the kills and measurements read, each a full executor run on the
day's prints whose void conditions join the slot's day record:

* ``touch`` and ``delayed`` (the fill rule's other two forms), ``latency``
  (every post, requote and cancel 110 ms after its decision);
* ``cells``: the nine neighbourhood cells, requote x skew, in the
  registration's order (the own cell is the true run);
* ``all_long`` and ``all_short`` (K-drift): every signal direction replaced,
  or, for MK-Q, quoting the buy side only and the sell side only;
* ``flipped`` and ``random`` (MK-S and MK-SQ): every direction reversed, and
  200 draws of ``spot/runs.random_sides`` (``default_rng([seed, day ordinal])``).

A run's daily value is its ``net_bp_fill`` (NaN without a fill). The true
run's record also holds its money, markout sums, capacity sums, counters, the
realised holds (MK-S), the touch proxies' widths (the quoters), the fill ratio
and every tier.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from trading_research.spot.costs import FeeTable
from trading_research.spot.fills import PrintTape
from trading_research.spot.runs import quantiles, random_sides
from trading_research.spot_maker import money as mn
from trading_research.spot_maker import registration as rg
from trading_research.spot_maker.executor import DayRun, MakerSettings, settings_for
from trading_research.spot_maker.quoter import run_quoter
from trading_research.spot_maker.signal import DayDecisions
from trading_research.spot_maker.signal_maker import run_signal_maker
from trading_research.spot_maker.touch import Proxies

TIERS: tuple[str, ...] = ("vip3", "vip9")


class DayContext:
    """What every run of one day reads."""

    def __init__(self, tape: PrintTape, proxies: Proxies, decisions: DayDecisions,
                 table: FeeTable, tier: str = "regular") -> None:  # fmt: skip
        self.tape, self.proxies, self.decisions, self.table = tape, proxies, decisions, table
        self.pricing = mn.Pricing(table, tier)
        self.first, self.last = float(tape.price[0]), float(tape.price[-1])

    def runner(self, slot: str) -> Callable[[MakerSettings, np.ndarray | None, str], DayRun]:
        """``run(settings, sides, mode)`` for the slot; ``sides`` replaces the signal's."""
        d = self.decisions
        if slot == "MK-S":
            return lambda s, sides, _: run_signal_maker(
                self.tape, self.proxies, d.grid_ns, d.decision_ns,
                d.decision_side if sides is None else sides, s)  # fmt: skip
        if slot == "MK-Q":
            return lambda s, _, mode: run_quoter(self.tape, self.proxies, s, mode=mode)
        settings = settings_for(slot)
        keep = (d.decision_ns >= settings.first_ns(self.tape)) & (
            d.decision_ns <= settings.last_ns(self.tape)
        )
        return lambda s, sides, mode: run_quoter(
            self.tape, self.proxies, s, d.decision_ns[keep],
            d.decision_side[keep] if sides is None else sides, mode)  # fmt: skip

    def sides(self, slot: str) -> np.ndarray:
        """The signal's sides the slot reads: every decision (MK-S), those inside the
        posting hours (MK-SQ), none (MK-Q)."""
        d = self.decisions
        if slot == "MK-S":
            return d.decision_side
        if slot == "MK-SQ":
            s = settings_for(slot)
            inside = (d.decision_ns >= s.first_ns(self.tape)) & (d.decision_ns <= s.last_ns(self.tape))
            return d.decision_side[inside]
        return np.zeros(0, dtype=np.int64)


def _value(ctx: DayContext, run: DayRun, unit: float, voids: set[str]) -> tuple[float, mn.DayFills]:
    fills = run.log.day(ctx.pricing)
    voids.update(mn.voids(fills, first_price=ctx.first, last_price=ctx.last, unit=unit,
                          table=ctx.table, tier=ctx.pricing.tier))  # fmt: skip
    return mn.money(fills, ctx.first, rg.CLIP)["net_bp_fill"], fills


def _true_record(ctx: DayContext, run: DayRun, fills: mn.DayFills) -> dict[str, Any]:
    out: dict[str, Any] = dict(mn.money(fills, ctx.first, rg.CLIP))
    out.update(mn.markout_sums(fills, ctx.proxies, rg.MARKOUTS_S))
    out["capacity_slot_btc"], out["capacity_venue_btc"] = mn.capacity(fills, ctx.tape)
    out["counters"] = dict(run.counters)
    out["posted_btc"] = run.posted_btc
    out["filled_maker_btc"] = float(fills.size[fills.maker].sum())
    out["hold_s"] = quantiles(np.asarray(run.holds_s))
    out["hold_s_sum"], out["hold_s_n"] = float(np.sum(run.holds_s)), float(len(run.holds_s))
    out["width_bp_median"] = float(np.median(run.widths_bp)) if run.widths_bp else math.nan
    out["tiers"] = mn.tier_net_bp(fills, ctx.table, TIERS)
    return out


def slot_day(ctx: DayContext, slot: str) -> dict[str, Any]:
    """Every run of ``slot`` on the day; ``void`` holds every void condition any fired."""
    spec, voids = rg.SLOTS[slot], set[str]()
    run, base = ctx.runner(slot), settings_for(slot)

    def value(settings: MakerSettings, sides: np.ndarray | None = None, mode: str = "both") -> float:
        return _value(ctx, run(settings, sides, mode), spec.unit, voids)[0]

    true_run = run(base, None, "both")
    true_value, fills = _value(ctx, true_run, spec.unit, voids)
    runs: dict[str, Any] = {
        "touch": value(settings_for(slot, rule="touch")),
        "delayed": value(settings_for(slot, rule="delayed")),
        "latency": value(settings_for(slot, latency_ns=rg.LATENCY_RERUN_NS)),
        "cells": [true_value if (r, k) == (spec.requote_s, spec.skew_bp)
                  else value(settings_for(slot, requote_s=r, skew_bp=k)) for r, k in spec.cells()],
    }  # fmt: skip
    sides = ctx.sides(slot)
    if spec.signal:
        ones = np.ones(len(sides), dtype=np.int64)
        runs["all_long"], runs["all_short"] = value(base, ones), value(base, -ones)
        runs["flipped"] = value(base, -sides)
        day = ctx.tape.day
        runs["random"] = [value(base, random_sides(len(sides), k, day))
                          for k in range(rg.RANDOM_DRAWS)]  # fmt: skip
    else:
        runs["all_long"] = value(base, None, "bid_only")
        runs["all_short"] = value(base, None, "ask_only")
    return {"value": true_value, "true": _true_record(ctx, true_run, fills), "runs": runs,
            "signal_decisions": len(sides), "void": sorted(voids)}  # fmt: skip


def day_record(ctx: DayContext, *, warm: bool, slots: tuple[str, ...] = rg.FAMILY) -> dict[str, Any]:
    """The day's record: its activity and the held unit's P&L, and every slot's runs."""
    d = ctx.decisions
    return {
        "usable": True, "warm": warm, "prints": len(ctx.tape),
        "first_price": ctx.first, "last_price": ctx.last,
        "inventory_pnl_bp": (ctx.last / ctx.first - 1.0) * 1e4,
        "signal": {"decisions": len(d.decision_ns), "scored_rows": d.scored_rows,
                   "long": int((d.decision_side > 0).sum()),
                   "short": int((d.decision_side < 0).sum()), "grid_rows": len(d.grid_ns)},
        "slots": {slot: slot_day(ctx, slot) for slot in slots},
    }  # fmt: skip
