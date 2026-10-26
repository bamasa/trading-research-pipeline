"""The block's first kept day takes the last kept day's tape; a stale shift excludes it."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from trading_research.spot import access as sa
from trading_research.spot import heldout as ho

DAYS = [date(2026, 8, 8) + timedelta(days=k) for k in range(4)]


def _records() -> dict[str, dict[str, Any]]:
    out = {}
    for d in DAYS:
        row = {"value": 1.0, "void": [], "signal": {"tod_ns": [1], "side": [1]}}
        out[d.isoformat()] = {"day": d.isoformat(), "kept": {sa.BYBIT: True},
                              "venues": {sa.BYBIT: {"reasons": []}},
                              "rows": {r: dict(row) for r in ho.VENUE_ROWS[sa.BYBIT]}}  # fmt: skip
    return out


def _patch(monkeypatch: pytest.MonkeyPatch, stale_on: set[str]) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []

    def shift(name, table, start, source, voids=None, fees=None):  # type: ignore[no-untyped-def]
        calls.append((table, source["day"]))
        return 0.5, table in stale_on and ho.ROWS[name].grid == "book"

    monkeypatch.setattr(ho, "_shift", shift)
    monkeypatch.setattr(ho.dy, "tape", lambda *_: None)
    monkeypatch.setattr(ho, "own_tables", _tables)
    return calls


def _tables(*args: Any) -> dict[str, str]:
    """Each row's table named by its day: (plan, inputs, fee table, venue)."""
    _, inputs, _, venue = args
    return dict.fromkeys(ho.VENUE_ROWS[venue], inputs.day.isoformat())


def _load(day: date) -> Any:
    return SimpleNamespace(day=day)


def test_a_stale_first_shift_excludes_the_day_and_the_next_takes_the_last_tape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _records()
    first, second, last = DAYS[0].isoformat(), DAYS[1].isoformat(), DAYS[-1].isoformat()
    calls = _patch(monkeypatch, {first})
    ho.first_shift(None, records, sa.BYBIT, records[last], _load, None, tmp_path)  # type: ignore[arg-type]
    assert not records[first]["kept"][sa.BYBIT]
    assert records[first]["venues"][sa.BYBIT]["reasons"] == [ho.STALE]
    assert records[second]["kept"][sa.BYBIT]
    assert records[second]["rows"]["S25-BB"]["shifted_from"] == last  # circular, redone
    assert {source for _, source in calls} == {last}
    written = json.loads((tmp_path / "days" / f"{second}.json").read_text(encoding="utf-8"))
    assert written["first_shift_" + sa.BYBIT] == last
    calls.clear()  # resumed: nothing is computed again
    ho.first_shift(None, records, sa.BYBIT, records[last], _load, None, tmp_path)  # type: ignore[arg-type]
    assert calls == []


def test_a_first_shift_that_keeps_its_day_ends_the_cascade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _records()
    first, last = DAYS[0].isoformat(), DAYS[-1].isoformat()
    _patch(monkeypatch, set())
    ho.first_shift(None, records, sa.BYBIT, records[last], _load, None, tmp_path)  # type: ignore[arg-type]
    assert records[first]["kept"][sa.BYBIT]
    assert records[first]["rows"]["S25-PB"]["shifted_from"] == last
    assert "shifted_from" not in records[DAYS[1].isoformat()]["rows"]["S25-BB"]


def test_null_b_gives_a_stale_shifted_run_no_value_and_counts_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trading_research.spot import development as dev
    from trading_research.spot.plan import REGISTERED

    records = _records()
    for r in records.values():
        r["kept"][sa.BINANCE] = False
    stale_day = DAYS[2].isoformat()
    monkeypatch.setattr(ho, "loader", lambda *_: _load)
    monkeypatch.setattr(ho, "own_tables", _tables)

    def shift(*args: Any) -> tuple[float, bool]:
        name, table = args[0], args[1]
        return 1.0, table == stale_day and ho.ROWS[name].grid == "book"

    monkeypatch.setattr(ho, "_shift", shift)
    study = SimpleNamespace(fee_table=None, scale=SimpleNamespace(
        null_b_shifts=lambda n: range(1, n)))  # fmt: skip
    rows = {n: SimpleNamespace(value=[2.0]) for names in ho.VENUE_ROWS.values() for n in names}
    found = dev.null_b(study, None, records, rows)  # type: ignore[arg-type]
    assert found["S25-BB"]["stale_runs"] == 3 and found["S25-PB"]["stale_runs"] == 0
    assert found["S25-BB"]["median_shift"] == 1.0
    assert REGISTERED.null_b_shifts(61) == range(7, 55)


def test_null_b_keeps_the_registered_shifts_when_a_test_day_is_excluded() -> None:
    from trading_research.spot.plan import DRY_RUN, REGISTERED

    assert REGISTERED.null_b_shifts(60) == range(7, 55)
    assert REGISTERED.null_b_shifts(40) == range(7, 40)
    assert len(DRY_RUN.null_b_shifts(6)) >= 1
