"""The second market-making round on its held-out blocks, each read once.

Runs everything ``docs/preregistration/market_making_round2.md`` assigns to a
held-out read, with every value frozen by its D amendment
(``configs/mm_prereg_round2.yaml``) and nothing chosen here:

* ``--read first``: H for the admitted basket together with P for BICOUSDT;
* ``--read basket_P``: P for the admitted basket, after H's results are
  committed;
* ``--read Q --symbols ...``: Q for the instruments named, after their P.

On every basket instrument: S0, S1, S1-touch and G(base) with their kept fills,
the same under pessimistic cancellation attribution (K-pess) and under the
joint pessimistic queue bracket (K-queue), every cell of S1's and of the gate's
D neighbourhood (K-nbhd), 50 shifted-gate placebos, B3's strategy at each
professional tier with its brackets, neighbourhood and the base taker fee, the
fee grid and the robustness axes. On BICOUSDT: S0, its S1 and S1-touch, B4's
G(base) with its brackets, neighbourhood, 50 placebos and the fee grid.

This script only reads and simulates. Its outputs land in
``artifacts/mm/round2/heldout/<read>/`` (gitignored): the rows of every
simulated configuration, the kept fills of the main strategies, the gate's
persistence (K4) per instrument-day, the market's room by hour, the gates'
open shares, and the run's record. ``experiments/market_making_round2_verdicts.py``
reads those and no market data.

The block is opened through :meth:`trading_research.market_making.round2.Ledger.open`,
which checks the frozen configuration and the clean committed tree and writes
the ledger entry before any file of the block is read. ``--dry-run`` runs the
same code on the last days of D under the development access, with fewer
placebo seeds and fee points, to exercise every path before a block is read.

Choices the registration does not spell out, fixed here before any held-out
block is read:

* the shifted-gate placebo shifts each instrument's whole state tape over the
  block (the real gate's open and closed periods, every day's closed first
  minutes included) by one offset per seed, the same for every instrument; the
  simulator, the base and the clip are unchanged;
* a gate at a fee other than the base (B3's tiers, the fee grid) reads that
  fee in its trailing room and in its hour mask; the mask at a fee is
  recomputed from D's hourly markouts, which must give the frozen masks at the
  base fee;
* B3's K-nbhd recomputes the tier's strategy with each cell of its
  neighbourhood at that tier (each instrument's S1 cells, or the gate's);
* K4 reads the gate's state at each print's time from the real state tape;
  for the ``hours`` kind no first minutes are left out.

Usage::

    .venv/bin/python -m experiments.market_making_round2_heldout --dry-run --workers 8
    .venv/bin/python -m experiments.market_making_round2_heldout --read first --workers 8
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from experiments import _round2 as r2
from experiments._common import RESULTS
from trading_research.market_making import gate, round2, round2_run
from trading_research.market_making.events import NS_PER_DAY, day_start_ns
from trading_research.market_making.heldout import Spec
from trading_research.market_making.queue import ArrivalGrowth, CancelAttribution
from trading_research.market_making.quoters import TouchQuoter
from trading_research.market_making.signals import SignalTape

OUT_ROOT = r2.OUT_ROOT / "heldout"
BICO = round2.B4_SYMBOL

#: The registered fee grid, the robustness axes and the placebo seeds.
FEE_GRID = (-1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
SEEDS = tuple(range(50))
PESSIMISTIC = {"cancel_attribution": CancelAttribution.PESSIMISTIC}
JOINT = {**PESSIMISTIC, "arrival_growth": ArrivalGrowth.ALL_AHEAD}


@dataclass(frozen=True)
class Frozen:
    """Every value a held-out read uses, from the frozen registration."""

    registration: round2.Registration
    admitted: tuple[str, ...]
    instruments: dict[str, r2.Instrument]
    s1: dict[str, dict[str, float]]
    s1_near: dict[str, list[dict[str, float]]]
    gate: gate.GateCell | None
    gate_near: list[gate.GateCell]
    b3: dict[str, str]
    b4: gate.GateCell
    b4_near: list[gate.GateCell]
    hourly: dict[str, pd.DataFrame]
    busy: dict[str, float]

    @classmethod
    def load(cls) -> Frozen:
        reg = round2.Registration.load().require_frozen()
        admitted = reg.admitted
        symbols = (*admitted, BICO)
        instruments = {s: r2.Instrument(s, reg.clip(s), reg.sigma_ref(s)) for s in symbols}
        s1 = {s: reg.s1_cell(s) for s in symbols}
        near = reg.value(("d_search", "S1", "neighbourhood"))
        s1_near = {s: [r2.s1_cell(c[:4]) for c in near[s]["cells"]] for s in admitted}
        chosen = (
            gate.GateCell.from_record(reg.value(("d_search", "gate", "chosen")))
            if admitted
            else None
        )
        b4 = reg.gate_cell(BICO)
        hours = pd.read_csv(RESULTS / "mm_round2_hours_D.csv")
        hourly = {
            s: hours[hours["symbol"] == s][["hour", "volume", "markout_bp"]].reset_index(drop=True)
            for s in symbols
        }
        for s in symbols:
            if r2.masks(hourly[s], r2.tier("base").maker_bp) != reg.hours_masks(s):
                raise SystemExit(f"{s}: D's hourly markouts do not give the frozen masks")
        table = pd.DataFrame(reg.value(("admission", "table")))
        b3 = (
            {}
            if not admitted
            else {
                t: v["strategy"] for t, v in reg.value(("d_search", "b3_choice", "chosen")).items()
            }
        )
        return cls(
            registration=reg,
            admitted=admitted,
            instruments=instruments,
            s1=s1,
            s1_near=s1_near,
            gate=chosen,
            gate_near=_cells(reg.value(("d_search", "gate", "chosen_neighbourhood")))
            if admitted
            else [],
            b3=b3,
            b4=b4,
            b4_near=_cells(reg.value(("d_search", "b4", "chosen_neighbourhood"))),
            hourly=hourly,
            busy=dict(zip(table["symbol"], table.get("prints_per_day", 0), strict=True)),
        )

    def cell_for(self, symbol: str) -> gate.GateCell:
        cell = self.b4 if symbol == BICO else self.gate
        assert cell is not None
        return cell


def _cells(record: dict[str, Any]) -> list[gate.GateCell]:
    """The gate cells of a frozen neighbourhood record."""
    out = []
    for row in record["cells"]:
        base, kind, window, margin = row[:4]
        out.append(
            gate.GateCell(
                str(base), str(kind), None if window is None else int(window), float(margin)
            )
        )
    return out


# ---------------------------------------------------------------------------
# What runs on each instrument
# ---------------------------------------------------------------------------


@dataclass
class Plan:
    """One instrument's specs on its block, and the gate tapes they read."""

    symbol: str
    frozen: Frozen
    markouts: dict[date, gate.PrintMarkouts]
    offsets: dict[int, int]
    span: tuple[int, int]
    fee_grid: Sequence[float]
    specs: list[Spec]

    @property
    def instrument(self) -> r2.Instrument:
        return self.frozen.instruments[self.symbol]

    def state(self, cell: gate.GateCell, maker_bp: float) -> SignalTape:
        return r2.state(self.markouts, cell, maker_bp, self.frozen.hourly[self.symbol])

    def add(self, label: str, quoter: Any, group: str, settings: Any, tape: SignalTape | None = None,
            *, keep: bool = False) -> None:  # fmt: skip
        tapes = None if tape is None else {tape.name: tape}
        self.specs.append(r2.spec(label, quoter, settings, group, tapes, keep=keep))

    def s1(self, label: str, group: str, cell: dict[str, float], fees: Any = None,
           *, touch: bool = False, keep: bool = False, **changes: Any) -> None:  # fmt: skip
        fees = fees or r2.tier("base")
        make = r2.s1_touch if touch else r2.s1
        settings = r2.config(self.instrument, cell["soft_limit_clips"], fees, **changes)
        self.add(label, make(self.instrument, cell, fees.maker_bp), group, settings, keep=keep)

    def g(self, label: str, group: str, cell: gate.GateCell, fees: Any = None,
          *, tape: SignalTape | None = None, keep: bool = False, **changes: Any) -> None:  # fmt: skip
        fees = fees or r2.tier("base")
        s1_cell = self.frozen.s1[self.symbol]
        settings = r2.config(self.instrument, s1_cell["soft_limit_clips"], fees, **changes)
        quoter = r2.gated(cell, self.instrument, s1_cell, fees.maker_bp)
        self.add(label, quoter, group, settings, tape or self.state(cell, fees.maker_bp), keep=keep)


def main_specs(plan: Plan) -> None:
    """S0, S1, S1-touch and G(base), kept; the same under both brackets."""
    s1_cell = plan.frozen.s1[plan.symbol]
    cell = plan.frozen.cell_for(plan.symbol)
    plan.add("S0", TouchQuoter(), "main", r2.config(plan.instrument, 6.0), keep=True)
    plan.s1("S1", "main", s1_cell, keep=True)
    plan.s1("S1-touch", "main", s1_cell, touch=True, keep=True)
    plan.g("G", "main", cell, keep=True)
    for tag, changes in (("pess", PESSIMISTIC), ("queue", JOINT)):
        plan.s1(f"S1|{tag}", tag, s1_cell, **changes)
        plan.s1(f"S1-touch|{tag}", tag, s1_cell, touch=True, **changes)
        plan.g(f"G|{tag}", tag, cell, **changes)


def neighbourhood_specs(plan: Plan) -> None:
    """Every other cell of the chosen cells' D neighbourhoods (K-nbhd)."""
    f, s = plan.frozen, plan.symbol
    cell = f.cell_for(s)
    if s != BICO:
        for near in f.s1_near[s]:
            if near != f.s1[s]:
                plan.s1(f"S1|nbhd|{r2.s1_label(near)}", "nbhd", near)
    for near_cell in f.b4_near if s == BICO else f.gate_near:
        if near_cell != cell:
            plan.g(f"G|nbhd|{near_cell.label}", "nbhd", near_cell)


def placebo_specs(plan: Plan) -> None:
    """G(base) on the real gate's state tape shifted around the block."""
    cell = plan.frozen.cell_for(plan.symbol)
    real = plan.state(cell, r2.tier("base").maker_bp)
    for seed, offset in plan.offsets.items():
        shifted = gate.shift_tape(real, offset, plan.span)
        plan.g(f"G|shift={seed:02d}", "placebo", cell, tape=shifted)


def tier_specs(plan: Plan) -> None:
    """B3's strategy at each professional tier: as chosen on D, under both
    brackets, over its neighbourhood, and with the taker fee at the base."""
    f, s = plan.frozen, plan.symbol
    s1_cell = f.s1[s]
    assert f.gate is not None
    for name, pick in f.b3.items():
        fees = r2.tier(name)
        variants: list[tuple[str, Any, dict[str, Any]]] = [
            ("", fees, {}),
            ("|pess", fees, PESSIMISTIC),
            ("|queue", fees, JOINT),
            ("|taker=5.5", r2.tier(name, taker_bp=5.5), {}),
        ]
        for tag, tier, changes in variants:
            label = f"B3|{name}{tag}"
            if pick == "S1":
                plan.s1(label, "tier", s1_cell, tier, keep=not tag, **changes)
            else:
                plan.g(label, "tier", f.gate, tier, keep=not tag, **changes)
        if pick == "S1":
            for near in f.s1_near[s]:
                if near != s1_cell:
                    plan.s1(f"B3|{name}|nbhd|{r2.s1_label(near)}", "tier", near, fees)
        else:
            for near_cell in f.gate_near:
                if near_cell != f.gate:
                    plan.g(f"B3|{name}|nbhd|{near_cell.label}", "tier", near_cell, fees)


def fee_specs(plan: Plan) -> None:
    """S1 and G(base) over the fee grid, the gate reading the fee it pays."""
    s1_cell = plan.frozen.s1[plan.symbol]
    cell = plan.frozen.cell_for(plan.symbol)
    for fee in plan.fee_grid:
        if fee == r2.tier("base").maker_bp:
            continue  # the main run
        tier = r2.grid_tier(fee)
        plan.s1(f"S1|fee={fee:+.2f}", "fee", s1_cell, tier)
        plan.g(f"G|fee={fee:+.2f}", "fee", cell, tier)


#: The robustness axes, one at a time, beside the defaults and the brackets.
ROBUST: tuple[tuple[str, dict[str, Any]], ...] = (
    ("cancel=optimistic", {"cancel_attribution": CancelAttribution.OPTIMISTIC}),
    ("growth=none", {"arrival_growth": ArrivalGrowth.NONE}),
    ("growth=all_ahead", {"arrival_growth": ArrivalGrowth.ALL_AHEAD}),
    *(
        (
            f"latency={ms}ms",
            {"order_latency_ns": ms * 1_000_000, "cancel_latency_ns": ms * 1_000_000},
        )
        for ms in (1, 50, 100, 250)
    ),
    *((f"feed={ms}ms", {"feed_latency_ns": ms * 1_000_000}) for ms in (20, 50)),
)


def robust_specs(plan: Plan) -> None:
    """S1 and G(base) on each robustness cell; pessimistic attribution is the
    K-pess bracket, already run."""
    s1_cell = plan.frozen.s1[plan.symbol]
    cell = plan.frozen.cell_for(plan.symbol)
    for tag, changes in ROBUST:
        plan.s1(f"S1|{tag}", "robust", s1_cell, **changes)
        plan.g(f"G|{tag}", "robust", cell, **changes)


def build(plan: Plan, *, robust: bool = True) -> Plan:
    """Every spec of one instrument: the basket's set, or B4's on BICOUSDT."""
    main_specs(plan)
    neighbourhood_specs(plan)
    placebo_specs(plan)
    fee_specs(plan)
    if plan.symbol != BICO:
        tier_specs(plan)
        if robust:
            robust_specs(plan)
    return plan


def per_job(prints_per_day: float) -> int:
    if prints_per_day > 200_000:
        return 3
    if prints_per_day > 60_000:
        return 5
    return 8


# ---------------------------------------------------------------------------
# The read
# ---------------------------------------------------------------------------


def open_read(
    args: argparse.Namespace, frozen: Frozen
) -> tuple[round2.Access, dict[str, list[date]], str]:
    """The access, each instrument's days, and the output name."""
    if args.dry_run:
        days = round2.block_days("D")[-args.dry_days :]
        return round2.development_access(), dict.fromkeys((*frozen.admitted, BICO), days), "dry"
    ledger = round2.Ledger()
    symbols = [s for s in args.symbols.split(",") if s] if args.symbols else []
    access = ledger.open(args.read, symbols=symbols, force=args.force)
    r2.report(f"ledger: {json.dumps(ledger.entries()[args.read][-1])}")
    block = {"first": None, "basket_P": "P", "Q": "Q"}[args.read]
    if block is None:
        days = {s: round2.block_days("H") for s in access.symbols("H")}
        days |= {s: round2.block_days("P") for s in access.symbols("P")}
    else:
        days = {s: round2.block_days(block) for s in access.symbols(block)}
    return access, days, args.read


def persistence(plan: Plan) -> pd.DataFrame:
    """K4's ingredients per day: the market's markout of prints that arrived
    while the real gate was open, and while it was closed."""
    cell = plan.frozen.cell_for(plan.symbol)
    tape = plan.state(cell, r2.tier("base").maker_bp)
    skip = float(cell.window_min or 0)
    rows = []
    for day, prints in sorted(plan.markouts.items()):
        row = gate.room_persistence(prints, tape, skip_first_min=skip)
        rows.append({"symbol": plan.symbol, "day": day.isoformat(), "cell": cell.label, **row})
    return pd.DataFrame(rows)


def open_shares(plan: Plan) -> dict[str, float]:
    """The real gate's share of quoting time open, and each placebo's."""
    cell = plan.frozen.cell_for(plan.symbol)
    real = plan.state(cell, r2.tier("base").maker_bp)
    starts = [day_start_ns(d) for d in sorted(plan.markouts)]
    out = {"real": gate.open_share(real, starts)}
    for seed, offset in plan.offsets.items():
        out[f"shift={seed:02d}"] = gate.open_share(gate.shift_tape(real, offset, plan.span), starts)
    return out


def peak_mb(who: int = resource.RUSAGE_SELF) -> float:
    peak = resource.getrusage(who).ru_maxrss
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--read", default="first", choices=sorted(round2.READS))
    parser.add_argument("--symbols", default="", help="Q's instruments, comma-separated")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true", help="the last days of D, no ledger")
    parser.add_argument("--dry-days", type=int, default=3)
    parser.add_argument("--dry-seeds", type=int, default=3)
    parser.add_argument("--force", action="store_true", help="a second read, stamped as such")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise SystemExit("between 1 and 8 workers")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    started = time.perf_counter()
    stages: dict[str, float] = {}
    frozen = Frozen.load()
    access, days, name = open_read(args, frozen)
    out = OUT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    seeds = SEEDS[: args.dry_seeds] if args.dry_run else SEEDS
    fee_grid = (FEE_GRID[0], FEE_GRID[-3], FEE_GRID[-1]) if args.dry_run else FEE_GRID
    r2.report(f"read {name}: {', '.join(f'{s} {d[0]}..{d[-1]}' for s, d in days.items())}")

    t = time.perf_counter()
    plans = []
    for symbol in sorted(days, key=lambda s: -float(frozen.busy.get(s, 0.0))):
        block = days[symbol]
        scored, missing = round2_run.markouts(
            symbol, block, access=access, book_roots=r2.BOOK_ROOTS, trades_root=r2.TRADES_ROOT,
            out=r2.MARKOUTS_ROOT, workers=args.workers,
        )  # fmt: skip
        if missing:
            r2.report(f"  {symbol}: {len(missing)} days not scored: {sorted(missing)}")
        span = (day_start_ns(block[0]), day_start_ns(block[-1]) + NS_PER_DAY)
        offsets = {seed: gate.placebo_offset(seed, span[1] - span[0]) for seed in seeds}
        plan = Plan(symbol, frozen, scored, offsets, span, fee_grid, [])
        plans.append(build(plan))
        r2.report(f"  {symbol}: {len(plan.specs)} specs a day on {len(block)} days")
    stages["tapes"] = time.perf_counter() - t

    t = time.perf_counter()
    r2.report("simulating")
    rows, folders = round2_run.run_plans(
        [(p.symbol, days[p.symbol], p.specs, per_job(float(frozen.busy.get(p.symbol, 0.0))))
         for p in plans],
        access=access, book_roots=r2.BOOK_ROOTS, trades_root=r2.TRADES_ROOT,
        funding_root=r2.FUNDING_ROOT, out_root=out / "runs", workers=args.workers,
        market_horizons_s=r2.MARKOUT_HORIZONS_S, on_progress=r2.report,
    )  # fmt: skip
    stages["simulation"] = time.perf_counter() - t
    rows["stamp"] = access.stamp
    rows.to_parquet(out / "rows.parquet", index=False)
    pd.concat([persistence(p) for p in plans], ignore_index=True).to_parquet(
        out / "persistence.parquet", index=False
    )
    hours = []
    for p in plans:
        frame = gate.hourly_markouts(p.markouts.values())
        frame.insert(0, "symbol", p.symbol)
        hours.append(frame)
    pd.concat(hours, ignore_index=True).to_parquet(out / "hours.parquet", index=False)
    shares = {p.symbol: open_shares(p) for p in plans}
    stages["total"] = time.perf_counter() - started
    meta: dict[str, Any] = {
        "read": name,
        "stamp": access.stamp,
        "days": {s: [d.isoformat() for d in ds] for s, ds in days.items()},
        "config_sha256": frozen.registration.sha256,
        "admitted": list(frozen.admitted),
        "seeds": list(seeds),
        "offsets_ns": {p.symbol: {str(k): v for k, v in p.offsets.items()} for p in plans},
        "fee_grid": list(fee_grid),
        "open_share": shares,
        "folders": {s: str(f) for s, f in folders.items()},
        "stages_s": stages,
        "workers": args.workers,
        "specs": {p.symbol: len(p.specs) for p in plans},
        "peak_rss_mb_worker": float(rows["worker_peak_rss_mb"].max()) if len(rows) else np.nan,
        "peak_rss_mb_any_child": peak_mb(resource.RUSAGE_CHILDREN),
        "peak_rss_mb_driver": peak_mb(),
    }
    if not args.dry_run:
        meta["ledger"] = round2.Ledger().entries()[args.read]
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")
    r2.report(
        json.dumps({k: meta[k] for k in ("stages_s", "specs", "peak_rss_mb_worker")}, indent=1)
    )
    r2.report("done")


if __name__ == "__main__":
    main()
