"""The fill rule's calibration against a book: the three slots on Bybit spot BTCUSDT's D days.

Registered by the amendments of 2026-10-27 and 2026-10-28 (``calibration`` in
the configuration). Every verdict of the study rests on a prints-only rule: the
touch is the last print of each side and a resting order fills only on a print
through its price. On D's days, where the spot study holds Bybit spot
BTCUSDT's book (ten levels) and prints, each slot is run

* **with the book**, in the market-making simulator
  (``market_making/simulator.simulate_day``) under its trade-through, queue and
  touch rules, the slot's rule as a quoter (:mod:`.book_quoters`); its tape the
  day's reachable prints on the book's tick;
* **with prints only**, by this study's executors on the day's prints thinned
  to BTC/FDUSD's count of aggregated prints that day (seeds 0 to 9,
  ``spot/calibration.thin_prints``), the proxies cold, the tick the book's.

The signal (MK-S, MK-SQ) is S25-PF, frozen, on the day's seed-0 thinned
prints, cold; the same decisions in every run of the day. Per slot, day and
rule: the maker fills, their volume and the making part (the mean 5 s markout
of the maker fills, bp per fill). The test (:func:`verdict`), for every slot,
on at least 80% of the days the simulator does not exclude with a maker fill
under every rule: the maker volume orders trade-through <= queue <= touch, and
the prints rule's making part lies between the book's trade-through and touch
making parts. When it fails, or cannot run, every verdict carries the label
"uncalibrated fill rule" on the page; nothing else changes. The spot study's
files are read in place, inside its own development access (D only).
"""

from __future__ import annotations

import json
import math
import multiprocessing
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_research.backtest.costs import FeeTier
from trading_research.market_making import events as ev
from trading_research.market_making.queue import FillRule
from trading_research.market_making.simulator import SimConfig, simulate_day
from trading_research.spot import access as sa
from trading_research.spot.archives import BYBIT_BOOK, BYBIT_PRINTS, day_path
from trading_research.spot.calibration import thin_prints
from trading_research.spot.fills import PrintTape
from trading_research.spot.state import write_json
from trading_research.spot_maker import money as mn
from trading_research.spot_maker import quoter as qt
from trading_research.spot_maker import registration as rg
from trading_research.spot_maker.book_quoters import BookQuoter, BookSignalMaker
from trading_research.spot_maker.executor import settings_for
from trading_research.spot_maker.signal_maker import run_signal_maker
from trading_research.spot_maker.touch import Proxies

BOOK_RULES: dict[str, FillRule] = {"trade_through": FillRule.THROUGH, "queue": FillRule.QUEUE,
                                   "touch": FillRule.TOUCH}  # fmt: skip
FEES = FeeTier("calibration", 0.0, 10.0)
COLD_MINUTES = 61
#: A spot day has no funding: the simulator's flag for it excludes nothing here.
IGNORED_FLAGS = frozenset({"no_funding"})


def calibration_days() -> list[date]:
    return sa.block_days("D")


class _ZeroFees:
    """A fee table charging nothing: the calibration reads markouts, never money."""

    def fees_bp(self, instrument: str, tier: str, ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        del instrument, tier
        return np.zeros(len(ns)), np.zeros(len(ns))


PRICING = mn.Pricing(_ZeroFees(), "regular")  # type: ignore[arg-type]


def load_book_day(spot_checkout: Path, day: date) -> tuple[ev.DayEvents, pd.DataFrame, int]:
    """The day's book and reachable prints on the book's tick as the simulator's events,
    every print of the day, and the count of reachable prints off the book's tick."""
    access = sa.development_access()
    access.require(sa.BYBIT, [day], sa.BOOK, what="load")
    access.require(sa.BYBIT, [day], sa.PRINTS, what="load")
    root = Path(spot_checkout) / "data" / "spot"
    prints = ev.within_day(pd.read_parquet(day_path(BYBIT_PRINTS, day, root)), day)
    book_ts, planes, gaps, _ = ev._read_book(day_path(BYBIT_BOOK, day, root), day, 10)
    distinct = np.concatenate([pd.unique(planes[s, "price"][:, k]) for s in ("bid", "ask")
                               for k in range(planes[s, "price"].shape[1])])  # fmt: skip
    tick = ev.derive_tick(distinct[np.isfinite(distinct) & (distinct > 0)])
    reachable = prints[~prints["rpi"].to_numpy(dtype=bool)]
    price = reachable["price"].to_numpy(dtype=np.float64)
    on_grid = np.abs(np.round(price / tick) * tick - price) <= 1e-6 * tick
    events = ev._from_frames(rg.CALIBRATION_SYMBOL, day, book_ts, planes,
                             reachable[on_grid].reset_index(drop=True), gaps, None, 8)  # fmt: skip
    return events, prints.reset_index(drop=True), int((~on_grid).sum())


def _making(sums: float, count: float) -> float:
    return sums / count if count > 0 else math.nan


def book_runs(events: ev.DayEvents, slot: str, decision_ns: np.ndarray,
              decision_side: np.ndarray) -> dict[str, Any]:  # fmt: skip
    """The slot under each of the book's rules: maker fills, volume and the making part."""
    settings = settings_for(slot, tick=events.spec.tick)
    start = ev.day_start_ns(events.day)
    tape = PrintTape(events.day, np.array([start]), np.array([1.0]), np.array([1.0]), np.array([1]))
    first, last, flat = settings.first_ns(tape), settings.last_ns(tape), settings.flatten_ns(tape)
    inside = (decision_ns >= first) & (decision_ns <= last)
    config = SimConfig(clip_btc=settings.limit if slot == "MK-SQ" else settings.clip, fees=FEES,
                       soft_limit_clips=1.0, warmup_s=60.0)  # fmt: skip
    out: dict[str, Any] = {}
    for name, rule in BOOK_RULES.items():
        if slot == "MK-S":
            quoter: Any = BookSignalMaker(settings, decision_ns, decision_side, first, last, flat)
        else:
            ns = decision_ns[inside] if slot == "MK-SQ" else np.zeros(0, dtype=np.int64)
            side = decision_side[inside] if slot == "MK-SQ" else np.zeros(0, dtype=np.int64)
            quoter = BookQuoter(settings, qt.instants(tape, settings, ns if len(ns) else None),
                                ns, side, name=slot)  # fmt: skip
        result = simulate_day(events, quoter, config.with_(fill_rule=rule))
        maker = result.fills[result.fills["maker"].to_numpy(dtype=bool)]
        marks = maker["markout_5s"].to_numpy(dtype=np.float64) if len(maker) else np.zeros(0)
        marks = marks[np.isfinite(marks)]
        out[name] = {"maker_fills": len(maker), "maker_btc": float(maker["size"].sum()),
                     "making_sum": float(marks.sum()), "making_n": len(marks),
                     "making_part": _making(float(marks.sum()), len(marks)),
                     "excluded": sorted(set(result.flags) - IGNORED_FLAGS)}  # fmt: skip
    return out


def prints_runs(frames: Sequence[pd.DataFrame], day: date, slot: str, tick: float,
                decisions: Any) -> dict[str, Any]:  # fmt: skip
    """The slot by this study's executors on each seed's thinned prints."""
    settings = settings_for(slot, tick=tick)
    fills, volume, sums, count = 0, 0.0, 0.0, 0.0
    for frame in frames:
        tape = PrintTape.from_frame(frame, day)
        proxies = Proxies.from_tape(tape)
        ns, side = decisions.decision_ns, decisions.decision_side
        if slot == "MK-S":
            run = run_signal_maker(tape, proxies, decisions.grid_ns, ns, side, settings)
        else:
            keep = (ns >= settings.first_ns(tape)) & (ns <= settings.last_ns(tape))
            signal = slot == "MK-SQ"
            run = qt.run_quoter(tape, proxies, settings, ns[keep] if signal else None,
                                side[keep] if signal else None)  # fmt: skip
        day_fills = run.log.day(PRICING)
        found = mn.markout_sums(day_fills, proxies, (rg.MAKING_S,))
        fills += int(day_fills.maker.sum())
        volume += float(day_fills.size[day_fills.maker].sum())
        sums, count = sums + found["markout_5s_sum"], count + found["markout_5s_n"]
    seeds = max(len(frames), 1)
    return {"maker_fills": fills / seeds, "maker_btc": volume / seeds, "making_sum": sums,
            "making_n": count, "making_part": _making(sums, count)}  # fmt: skip


def calibration_day(spot_checkout: Path, day: date, count: int, signal: Any) -> dict[str, Any]:
    """One day of the calibration: every slot under the book's three rules and the prints rule."""
    from trading_research.spot_maker.signal import day_decisions

    head: dict[str, Any] = {"day": day.isoformat(), "thinned_to": int(count)}
    try:
        events, prints, off_grid = load_book_day(spot_checkout, day)
    except (ev.DayUnavailable, ev.OffGridPrice, FileNotFoundError, ValueError) as exc:
        return {**head, "usable": False, "reason": f"{type(exc).__name__}: {exc}"[:200]}
    frames = [
        thin_prints(prints, count, seed).reset_index(drop=True) for seed in rg.CALIBRATION_SEEDS
    ]
    start = int(pd.Timestamp(day, tz="UTC").value)
    decisions = day_decisions(signal, frames[0], None, start + COLD_MINUTES * 60 * rg.NS_PER_S)
    out: dict[str, Any] = {**head, "usable": True, "prints": len(prints), "off_grid": off_grid,
                           "tick": events.spec.tick, "sequence_gaps": events.sequence_gaps,
                           "decisions": len(decisions.decision_ns), "slots": {}}  # fmt: skip
    for slot in rg.CALIBRATION_SLOTS:
        out["slots"][slot] = {
            "book": book_runs(events, slot, decisions.decision_ns, decisions.decision_side),
            "prints": prints_runs(frames, day, slot, events.spec.tick, decisions)}  # fmt: skip
    return out


def day_tests(record: Mapping[str, Any], slot: str) -> tuple[bool, bool] | None:
    """(ordering, bracket) on a day, or None when the day does not count."""
    if not record.get("usable"):
        return None
    found = record["slots"][slot]
    book, prints = found["book"], found["prints"]
    if any(book[r]["excluded"] or not book[r]["maker_fills"] for r in rg.CALIBRATION_RULES):
        return None
    n = [book[r]["maker_btc"] for r in rg.CALIBRATION_RULES]
    ordering = n[0] <= n[1] + 1e-12 and n[1] <= n[2] + 1e-12
    lo, hi = sorted((book["trade_through"]["making_part"], book["touch"]["making_part"]))
    making = prints["making_part"]
    bracket = making is not None and math.isfinite(making) and lo <= making <= hi
    return bool(ordering), bool(bracket)


def verdict(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The registered test over the days, per slot and in all; the label it sets."""
    out: dict[str, Any] = {"days": len(records),
                           "usable_days": sum(1 for r in records if r.get("usable"))}  # fmt: skip
    passes = bool(records)
    for slot in rg.CALIBRATION_SLOTS:
        tests = [t for t in (day_tests(r, slot) for r in records) if t is not None]
        n = len(tests)
        ordering = sum(t[0] for t in tests) / n if n else math.nan
        bracket = sum(t[1] for t in tests) / n if n else math.nan
        ok = bool(
            n and ordering >= rg.CALIBRATION_MIN_SHARE and bracket >= rg.CALIBRATION_MIN_SHARE
        )
        out[slot] = {"days": n, "ordering_share": ordering, "bracket_share": bracket, "passes": ok}
        passes = passes and ok
    out["passes"] = passes
    out["label"] = None if passes else rg.CALIBRATION_LABEL
    return out


def not_run(reason: str) -> dict[str, Any]:
    return {"passes": False, "label": rg.CALIBRATION_LABEL, "status": rg.CALIBRATION_NOT_RUN,
            "reason": reason}  # fmt: skip


def calibration_job(repo: Path, day: date, spot_checkout: Path, count: int,
                    binding: Any) -> dict[str, Any]:  # fmt: skip
    """One day in a worker: the frozen signal loaded from this checkout's copies."""
    from trading_research.spot_maker.signal import load_signal

    return calibration_day(spot_checkout, day, count, load_signal(repo, binding))


def run_calibration(repo: Path, spot_checkout: Path | None, counts: Mapping[str, int],
                    binding: Any, folder: Path, workers: int = 1) -> dict[str, Any]:  # fmt: skip
    """Every D day with a BTC/FDUSD count (kept under ``folder``, so a stopped stage
    resumes) and the verdict; ``counts`` BTC/FDUSD's aggregated prints by day."""
    days = [d for d in calibration_days() if d.isoformat() in counts]
    root = None if spot_checkout is None else Path(spot_checkout) / "data" / "spot"
    if root is None or not any(day_path(BYBIT_BOOK, d, root).exists() for d in days):
        return not_run("no book day of D under the spot study's checkout")
    folder.mkdir(parents=True, exist_ok=True)
    todo = [d for d in days if not (folder / f"{d}.json").exists()]
    args = [(Path(repo), d, Path(spot_checkout), int(counts[d.isoformat()]), binding) for d in todo]  # type: ignore[arg-type]
    if workers > 1 and len(todo) > 1:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(workers, mp_context=context) as pool:
            for record in pool.map(calibration_job, *zip(*args, strict=True)):
                write_json(folder / f"{record['day']}.json", record)
    else:
        for arg in args:
            record = calibration_job(*arg)
            write_json(folder / f"{record['day']}.json", record)
    records = [json.loads((folder / f"{d}.json").read_text("utf-8")) for d in days]
    return {
        **verdict(records),
        "status": "run",
        "per_day": {r["day"]: _day_line(r) for r in records},
    }


def _day_line(record: Mapping[str, Any]) -> dict[str, Any]:
    """The day's fills, volume and making part per slot and rule, for the record."""
    if not record.get("usable"):
        return {"usable": False, "reason": record.get("reason")}
    out: dict[str, Any] = {"usable": True, "thinned_to": record["thinned_to"]}
    for slot, found in record["slots"].items():
        rules = {r: [found["book"][r]["maker_fills"], found["book"][r]["maker_btc"],
                     found["book"][r]["making_part"]] for r in rg.CALIBRATION_RULES}  # fmt: skip
        rules["prints"] = [
            found["prints"]["maker_fills"],
            found["prints"]["maker_btc"],
            found["prints"]["making_part"],
        ]
        out[slot] = rules
    return out
