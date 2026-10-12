"""A taker slot's held-out runs: the K-latency re-run, every run's stale flag, the one-day shift."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from _engine_books import CLIP, DAY, make_book
from trading_research.backtest.taker import TakerSettings
from trading_research.pipeline.short_horizon import heldout as ho
from trading_research.search_engine.catalogue import parse_key
from trading_research.search_engine.cells import score_cell
from trading_research.search_engine.null_b import entries_of, replay
from trading_research.search_engine.tapes import hold_first_tape
from trading_research.search_engine.walk_tables import WalkTable, execute_rows, hold_table
from trading_research.strategies.hold_first import fit_hold_first

CONFIG = "BTCUSDT|C|gbm|micro|hold_s=5|label_fee_bp=0"
KEY = "BTCUSDT|C|gbm|micro|open|hold_s=5|trades_per_day=240|label_fee_bp=0|regime=none"
NEAR = KEY.replace("trades_per_day=240", "trades_per_day=100")
HOLD = 5_000_000_000


def _day(seed: int, day: date, *, late_stretch: bool = True) -> tuple[Any, Any]:
    """A day's context for the slot (its tape, tables at 10 and 110 ms) and its rules."""
    book = make_book(seed=seed, thin_share=0.05, day=day, late_stretch=late_stretch)
    walk = WalkTable.from_book(book, CLIP)
    validation = np.random.default_rng(100).dirichlet([1.0, 3.0, 1.0], 20_000)
    rules = {
        float(n): fit_hold_first(validation, days=1.0, trades_per_day=n * 20) for n in (100, 240)
    }
    proba = np.random.default_rng(seed + 1).dirichlet([1.0, 3.0, 1.0], len(book.ts))
    gates = {"quiet_out": np.ones(len(book.ts), bool), "tight_only": np.ones(len(book.ts), bool)}
    tape = hold_first_tape(day, CONFIG, book.ts, proba, rules[240.0], gates, np.zeros(0, np.int64))
    ctx = ho.DayContext(day, book, book.ts, {HOLD: hold_table(walk, HOLD)},
                        {HOLD: hold_table(walk, HOLD, TakerSettings(latency_ns=110_000_000))},
                        {CONFIG: tape}, {CONFIG: proba[:, 2] - proba[:, 0]},
                        {5.0: np.random.default_rng(seed).normal(size=len(book.ts))})  # fmt: skip
    return ctx, {CONFIG: rules}


def test_k_latency_reruns_the_slot_on_the_110_ms_tables() -> None:
    ctx, rules = _day(0, DAY)
    slot = ho.TakerSlot("S-C", KEY, [NEAR], [DAY], 2)
    slot.add(0, ctx, rules)
    cell, rule = parse_key(KEY), rules[CONFIG][240.0]
    again = score_cell(ctx.tapes[CONFIG], ctx.late[HOLD], cell, rule)
    assert slot.latency[0] == pytest.approx(ho._bp(again))
    true = score_cell(ctx.tapes[CONFIG], ctx.tables[HOLD], cell, rule)
    replayed = execute_rows(ctx.late[HOLD], true.rows, true.side)  # the 10 ms entries only
    assert again.trades != replayed.trades


@pytest.mark.parametrize("which", ["late", "neighbour"])
def test_a_day_any_run_flags_stale_is_stale_for_the_slot(
    which: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    from trading_research.search_engine import cells

    ctx, rules = _day(0, DAY)
    real = cells.score_cell

    def flagging(tape: Any, table: Any, cell: Any, rule: Any = None) -> Any:
        found = real(tape, table, cell, rule)
        hit = table is ctx.late[HOLD] if which == "late" else cell.key == NEAR
        return replace(found, stale=True) if hit else found

    monkeypatch.setattr(cells, "score_cell", flagging)
    slot = ho.TakerSlot("S-C", KEY, [NEAR], [DAY], 2)
    slot.add(0, ctx, rules)
    assert slot.stale == {0} and not real(ctx.tapes[CONFIG], ctx.tables[HOLD], parse_key(KEY),
                                          rules[CONFIG][240.0]).stale  # fmt: skip


def test_the_one_day_shift_carries_the_signal_tape_not_the_entries_taken() -> None:
    first, rules = _day(0, DAY)
    second, _ = _day(1, DAY + timedelta(days=1))
    cell, rule = parse_key(KEY), rules[CONFIG][240.0]
    signal = ho.signal_tape(first.tapes[CONFIG], first.tables[HOLD], cell, rule)
    same = ho.shift_on(first.tables[HOLD], signal)
    np.testing.assert_array_equal(
        same.rows, score_cell(first.tapes[CONFIG], first.tables[HOLD], cell, rule).rows
    )
    table = second.tables[HOLD]
    shifted = ho.shift_on(table, signal)
    rows = np.searchsorted(table.walk.ts, table.walk.start_ns + signal.tod_ns, side="left")
    inside = rows < table.rows
    reference = execute_rows(table, rows[inside], signal.side[inside])
    np.testing.assert_array_equal(shifted.rows, reference.rows)
    taken = score_cell(first.tapes[CONFIG], first.tables[HOLD], cell, rule)
    old = replay(table, entries_of(first.tables[HOLD], taken))
    assert len(signal.side) > taken.trades and shifted.trades != old.trades


def _taker_with_shifts(days: list[date]) -> ho.TakerSlot:
    slot = ho.TakerSlot("S-C", KEY, [], days, 2)
    n = len(days)
    slot.signals = {
        i: ho.SignalTape(np.zeros(0, np.int64), np.zeros(0, np.int64), None) for i in range(n)
    }
    slot.shifted_from = {i: (i - 1) % n for i in range(n)}
    return slot


def test_the_shift_steps_over_a_stale_day_and_over_a_day_its_own_run_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    days = [date(2025, 7, 8) + timedelta(days=i) for i in range(6)]
    slot = _taker_with_shifts(days)
    built: list[date] = []
    monkeypatch.setattr(ho, "hold_table_of", lambda _block, day, _hold: built.append(day) or day)
    flagged = {days[3]}  # day 3 read again from day 1 walks a stale snapshot
    monkeypatch.setattr(
        ho, "shift_on", lambda table, _signal: SimpleNamespace(trades=0, stale=table in flagged)
    )
    stale, _ = ho.step_over_stale(SimpleNamespace(), [slot], [], days, {days[2]}, None)  # type: ignore[arg-type]
    assert stale == {days[2], days[3]}
    assert slot.shifted_from[4] == 1 and slot.shifted_from[3] == 1 and slot.shifted_from[0] == 5
    assert built == [days[3], days[4]]
    last = _taker_with_shifts(days)
    stale, _ = ho.step_over_stale(SimpleNamespace(), [last], [], days, {days[5]}, None)  # type: ignore[arg-type]
    assert last.shifted_from[0] == 4  # circularly over the kept days


def test_a_passive_slots_shift_is_simulated_again_from_the_previous_kept_day(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pandas as pd

    from trading_research.pipeline.short_horizon.passive import Decisions, SimCell

    days = [date(2025, 7, 8) + timedelta(days=i) for i in range(4)]
    own = [
        Decisions(np.array([10 + i], np.int64), np.array([1]), np.array([1.0])) for i in range(4)
    ]
    cell = SimCell("S-P", KEY, "entry", 5.0)
    p = ho.PassiveSlot("S-P", cell, [], decisions={KEY: own})
    seen: list[Any] = []

    def simulate(block: Any, jobs: list[Any]) -> list[Any]:
        seen.extend(jobs)
        return list(jobs)

    def read(job: Any) -> pd.DataFrame:
        return pd.DataFrame({"run_slot": ["S-P"], "variant": ["shifted"], "day": [job.day.isoformat()],
                             "excluded": [False], "net_per_100": [7.0]})  # fmt: skip

    monkeypatch.setattr(ho, "simulate_jobs", simulate)
    monkeypatch.setattr(pd, "read_parquet", read)
    rows = pd.DataFrame({"run_slot": ["S-P"] * 4, "variant": ["shifted"] * 4, "excluded": [False] * 4,
                         "day": [d.isoformat() for d in days], "net_per_100": [1.0] * 4})  # fmt: skip
    block = SimpleNamespace(study=SimpleNamespace(repo="repo", symbol="BTCUSDT"), access=None, threshold=9,
                            flags_ns=np.zeros(0, np.int64), fitted={"sigma_ref": 1.0}, work=Path("work"))  # fmt: skip
    stale, out = ho.step_over_stale(block, [], [p], days, {days[1]}, rows)  # type: ignore[arg-type]
    assert stale == {days[1]} and [job.day for job in seen] == [days[2]]
    run = seen[0].runs[0]
    assert run.variant == "shifted" and seen[0].out.name == days[0].isoformat()
    assert run.decisions.stamps_ns.tolist() == [10 - midnight(days[0]) + midnight(days[2])]
    assert out.sort_values("day")["net_per_100"].tolist() == [1.0, 1.0, 7.0, 1.0]


def midnight(day: date) -> int:
    from trading_research.pipeline.short_horizon.plan import midnight_ns

    return int(midnight_ns(day))
