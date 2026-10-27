"""The fill rule's calibration against a book: the quoters on Bybit linear BTCUSDT days with a book.

Registered by the amendment of 2026-10-27 (``calibration`` in the
configuration). Every verdict of the study rests on a prints-only rule: the
touch is the last print of each side and a resting order fills only on a print
through its price. On days where a book exists, the same quoting rule is run

* **with the book**, in the market-making simulator
  (``market_making/simulator.simulate_day``), under its trade-through, queue
  and touch rules, the quoter being :class:`BookQuoter`: the slot's requote
  instants, skew and limit, quoting the book's best bid and ask, post-only
  against the other side, no replenishment between instants (an order the
  simulator keeps at an unchanged price keeps its place);
* **with prints only**, by this study's quoter (``spot_maker/quoter``) on the
  same day's prints thinned to BTC/FDUSD's median count a day on D, seeds 0
  to 9, the proxies cold.

MK-SQ's decisions are S25-PF's, frozen, on the day's seed-0 thinned prints,
cold. Per day and rule: the maker fills, their volume and the making part (the
mean 5 s markout of the maker fills, bp per fill). The test (:func:`verdict`):
on at least 80% of the days, for both slots, the maker volume filled orders
trade-through <= queue <= touch (the book's touch rule fills an order whole at
one print, so fills are not counted alike across rules; volume is), and the
prints rule's making part lies between the
trade-through and the touch rules' making parts. When it fails, or the
calibration cannot run, every verdict carries the label "uncalibrated fill
rule" on the page; nothing else changes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_research.backtest.costs import FeeTier
from trading_research.market_making.events import DayEvents, DayUnavailable, load_day
from trading_research.market_making.queue import FillRule
from trading_research.market_making.quoters import NO_QUOTE, MarketView, Quote, Quotes
from trading_research.market_making.simulator import SimConfig, simulate_day
from trading_research.spot.calibration import thin_prints
from trading_research.spot.fills import PrintTape
from trading_research.spot_maker import money as mn
from trading_research.spot_maker import quoter as qt
from trading_research.spot_maker import registration as rg
from trading_research.spot_maker.executor import MakerSettings, settings_for
from trading_research.spot_maker.touch import Proxies

BOOK_RULES: dict[str, FillRule] = {"trade_through": FillRule.THROUGH, "queue": FillRule.QUEUE,
                                   "touch": FillRule.TOUCH}  # fmt: skip
FEES = FeeTier("calibration", 0.0, 10.0)
COLD_MINUTES = 61


def calibration_days() -> list[date]:
    start, end = rg.CALIBRATION_DAYS
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


@dataclass
class BookQuoter:
    """A slot's quoting rule in the simulator's interface; prices in ticks."""

    settings: MakerSettings
    instants: np.ndarray
    signal_ns: np.ndarray
    signal_side: np.ndarray
    name: str = "MK-Q"
    uses_future: bool = False
    _k: int = field(default=-1, init=False)
    _bid: tuple[int | None, float] = field(default=(None, 0.0), init=False)
    _ask: tuple[int | None, float] = field(default=(None, 0.0), init=False)
    _position: float = field(default=0.0, init=False)

    def _requote(self, view: MarketView, position: float, instant: int) -> None:
        s, tick = self.settings, view.tick
        sigma = int(qt.signal_sides(np.array([instant]), self.signal_ns, self.signal_side,
                                    s.hold_ns)[0])  # fmt: skip
        want_bid, want_ask = s.clip * (2 if sigma > 0 else 1), s.clip * (2 if sigma < 0 else 1)
        shift = 1.0 - s.skew_bp * 1e-4 * position / s.clip
        bid = math.floor(min(view.best_bid * tick * shift, (view.best_ask - 1) * tick) / tick + 1e-6)
        ask = math.ceil(max(view.best_ask * tick * shift, (view.best_bid + 1) * tick) / tick - 1e-6)
        self._bid = (bid, min(want_bid, s.limit - position))
        self._ask = (ask, min(want_ask, s.limit + position))
        self._position = position

    def quotes(self, view: MarketView, position: float) -> Quotes:
        k = int(np.searchsorted(self.instants, view.ts, side="right")) - 1
        if k < 0:
            return Quotes(NO_QUOTE, NO_QUOTE)
        if k != self._k:
            self._k = k
            self._requote(view, position, int(self.instants[k]))
        moved = position - self._position
        self._position = position
        if moved > 0:
            self._bid = (self._bid[0], self._bid[1] - moved)
        elif moved < 0:
            self._ask = (self._ask[0], self._ask[1] + moved)
        bid, ask = (Quote(p, q) if p is not None and q > rg.REMAINDER_BTC else NO_QUOTE
                    for p, q in (self._bid, self._ask))  # fmt: skip
        return Quotes(bid, ask)


def events_tape(events: DayEvents) -> PrintTape:
    return PrintTape(events.day, events.trade_ts, events.trade_px, events.trade_sz,
                     events.trade_aggressor)  # fmt: skip


def prints_frame(events: DayEvents) -> pd.DataFrame:
    """The day's prints in the spot print contract (what the features and the thinning read)."""
    n = len(events.trade_ts)
    return pd.DataFrame({
        "timestamp": pd.to_datetime(np.asarray(events.trade_ts, dtype=np.int64), utc=True),
        "price": np.asarray(events.trade_px, dtype=np.float64),
        "size": np.asarray(events.trade_sz, dtype=np.float64),
        "aggressor": np.asarray(events.trade_aggressor, dtype=np.int8),
        "trade_id": np.arange(n, dtype=np.int64), "rpi": np.zeros(n, dtype=bool),
        "source": pd.Series(["bybit-linear-trades"] * n, dtype="string"),
    })  # fmt: skip


def frame_tape(frame: pd.DataFrame, day: date) -> PrintTape:
    return PrintTape.from_frame(frame, day)


def _making(sums: float, count: float) -> float:
    return sums / count if count > 0 else math.nan


def book_runs(events: DayEvents, slot: str, signal_ns: np.ndarray,
              signal_side: np.ndarray) -> dict[str, Any]:  # fmt: skip
    """The slot under each of the book's rules: maker fills and the making part."""
    settings = settings_for(slot, tick=rg.CALIBRATION_TICK)
    tape = events_tape(events)
    instants = qt.instants(tape, settings, signal_ns if len(signal_ns) else None)
    config = SimConfig(clip_btc=settings.limit if slot == "MK-SQ" else settings.clip, fees=FEES,
                       soft_limit_clips=1.0, warmup_s=60.0)  # fmt: skip
    out: dict[str, Any] = {}
    for name, rule in BOOK_RULES.items():
        quoter = BookQuoter(settings, instants, signal_ns, signal_side, name=slot)
        result = simulate_day(events, quoter, config.with_(fill_rule=rule))
        maker = result.fills[result.fills["maker"].to_numpy(dtype=bool)]
        marks = maker["markout_5s"].to_numpy(dtype=np.float64) if len(maker) else np.zeros(0)
        marks = marks[np.isfinite(marks)]
        out[name] = {"maker_fills": len(maker), "maker_btc": float(maker["size"].sum()),
                     "making_sum": float(marks.sum()),
                     "making_n": len(marks), "making_part": _making(marks.sum(), len(marks)),
                     "excluded": list(result.flags)}  # fmt: skip
    return out


def prints_runs(frames: Sequence[pd.DataFrame], day: date, slot: str, signal_ns: np.ndarray,
                signal_side: np.ndarray) -> dict[str, Any]:  # fmt: skip
    """The slot by this study's prints-only quoter on each seed's thinned prints."""
    settings = settings_for(slot, tick=rg.CALIBRATION_TICK)
    fills, volume, sums, count = 0, 0.0, 0.0, 0.0
    pricing = mn.Pricing(_ZERO_TABLE, "regular")
    for frame in frames:
        tape = frame_tape(frame, day)
        proxies = Proxies.from_tape(tape)
        keep = (signal_ns >= settings.first_ns(tape)) & (signal_ns <= settings.last_ns(tape))
        run = qt.run_quoter(tape, proxies, settings, signal_ns[keep] if slot == "MK-SQ" else None,
                            signal_side[keep] if slot == "MK-SQ" else None)  # fmt: skip
        day_fills = run.log.day(pricing)
        found = mn.markout_sums(day_fills, proxies, (rg.MAKING_S,))
        fills += int(day_fills.maker.sum())
        volume += float(day_fills.size[day_fills.maker].sum())
        sums, count = sums + found["markout_5s_sum"], count + found["markout_5s_n"]
    seeds = max(len(frames), 1)
    return {"maker_fills": fills / seeds, "maker_btc": volume / seeds, "making_sum": sums,
            "making_n": count,
            "making_part": _making(sums, count)}  # fmt: skip


class _ZeroFees:
    """A fee table charging nothing: the calibration reads markouts, never money."""

    intervals: Mapping[str, Any] = {}

    def fees_bp(self, instrument: str, tier: str, ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        del instrument, tier
        return np.zeros(len(ns)), np.zeros(len(ns))


_ZERO_TABLE: Any = _ZeroFees()


def calibration_day(book_data: Path, day: date, count: int, signal: Any) -> dict[str, Any]:
    """One day of the calibration: both slots under the book's three rules and the prints rule."""
    from trading_research.spot_maker.signal import day_decisions

    head: dict[str, Any] = {"day": day.isoformat(), "thinned_to": int(count)}
    root = Path(book_data)
    try:
        events = load_day(rg.CALIBRATION_SYMBOL, day, book_roots=[root / "book", root / "book_fresh"],
                          trades_root=root / "trades", funding_root=root / "funding")  # fmt: skip
    except (DayUnavailable, FileNotFoundError) as exc:
        return {**head, "usable": False, "reason": f"{type(exc).__name__}: {exc}"[:200]}
    full = prints_frame(events)
    frames = [thin_prints(full, count, seed).reset_index(drop=True) for seed in rg.CALIBRATION_SEEDS]
    start = int(pd.Timestamp(day, tz="UTC").value)
    decisions = day_decisions(signal, frames[0], None, start + COLD_MINUTES * 60 * rg.NS_PER_S)
    out: dict[str, Any] = {**head, "usable": True, "prints": len(full),
                           "decisions": len(decisions.decision_ns), "slots": {}}  # fmt: skip
    for slot in rg.CALIBRATION_SLOTS:
        ns = decisions.decision_ns if slot == "MK-SQ" else np.zeros(0, dtype=np.int64)
        side = decisions.decision_side if slot == "MK-SQ" else np.zeros(0, dtype=np.int64)
        out["slots"][slot] = {"book": book_runs(events, slot, ns, side),
                              "prints": prints_runs(frames, day, slot, ns, side)}  # fmt: skip
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
        ok = bool(n and ordering >= rg.CALIBRATION_MIN_SHARE and bracket >= rg.CALIBRATION_MIN_SHARE)
        out[slot] = {"days": n, "ordering_share": ordering, "bracket_share": bracket, "passes": ok}
        passes = passes and ok
    out["passes"] = passes
    out["label"] = None if passes else rg.CALIBRATION_LABEL
    return out


def not_run(reason: str) -> dict[str, Any]:
    return {"passes": False, "label": rg.CALIBRATION_LABEL, "status": rg.CALIBRATION_NOT_RUN,
            "reason": reason}  # fmt: skip


def calibration_job(repo: Path, day: date, book_data: Path, count: int, binding: Any) -> dict[str, Any]:
    """One day in a worker: the frozen signal loaded from this checkout's copies."""
    from trading_research.spot_maker.signal import load_signal

    return calibration_day(book_data, day, count, load_signal(repo, binding))


def run_calibration(repo: Path, book_data: Path | None, count: int, binding: Any, folder: Path,
                    workers: int = 1) -> dict[str, Any]:  # fmt: skip
    """Every calibration day (kept under ``folder``, so a stopped stage resumes) and the verdict."""
    import json
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    from trading_research.spot.state import write_json

    days = calibration_days()
    if book_data is None or not any(
        (Path(book_data) / "book" / rg.CALIBRATION_SYMBOL / f"{d}.parquet").exists() for d in days
    ):
        return not_run("no book day of the calibration's days under the book data given")
    folder.mkdir(parents=True, exist_ok=True)
    todo = [d for d in days if not (folder / f"{d}.json").exists()]
    args = [(Path(repo), d, Path(book_data), int(count), binding) for d in todo]
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
    return {**verdict(records), "status": "run", "thinned_to": int(count),
            "per_day": {r["day"]: _day_line(r) for r in records}}  # fmt: skip


def _day_line(record: Mapping[str, Any]) -> dict[str, Any]:
    """The day's fills and making parts per slot and rule, for the record."""
    if not record.get("usable"):
        return {"usable": False, "reason": record.get("reason")}
    out: dict[str, Any] = {"usable": True}
    for slot, found in record["slots"].items():
        rules = {r: [found["book"][r]["maker_fills"], found["book"][r]["maker_btc"],
                     found["book"][r]["making_part"]] for r in rg.CALIBRATION_RULES}  # fmt: skip
        rules["prints"] = [found["prints"]["maker_fills"], found["prints"]["maker_btc"],
                           found["prints"]["making_part"]]
        out[slot] = rules
    return out
