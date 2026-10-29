"""A block's read day by day: every row, the one-day shift, the calibration, the resumption."""

from __future__ import annotations

import math
from datetime import date
from typing import Any

import numpy as np
import pytest

from test_spot_day_fixture import DAYS, stub_models, study_with_days
from trading_research.spot import access as sa
from trading_research.spot import assemble as asm
from trading_research.spot import blockread as br
from trading_research.spot import heldout as ho
from trading_research.spot import synthetic as syn
from trading_research.spot.statuses import check_status


@pytest.fixture(autouse=True)
def _offline():  # type: ignore[no-untyped-def]
    with syn.no_network():
        yield


@pytest.fixture(scope="module")
def read(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    study: Any = study_with_days(tmp_path_factory.mktemp("repo"), gap_days=frozenset({DAYS[1]}))
    plan = ho.ReadPlan(label="test", block="H", days=DAYS, access=sa.reader_test_access("dry run"),
                       models=stub_models(), max_gaps=0, depth=10)  # fmt: skip
    with syn.no_network():
        records = ho.read_days(study, plan)
    return {"study": study, "plan": plan, "records": records}


def test_every_kept_day_has_every_row_and_its_shift(read: dict[str, Any]) -> None:
    records = read["records"]
    first, gap, last = (records[d.isoformat()] for d in DAYS)
    assert first["kept"] == {sa.BINANCE: True, sa.BYBIT: True}
    assert gap["kept"][sa.BINANCE] and not gap["kept"][sa.BYBIT]
    assert "put_back" in gap and set(gap["put_back"]) == {"S25-BB", "S25-T"}
    assert set(last["rows"]) == {"S25-PF", "S25-PFT", "S25-BB", "S25-T", "S25-PB"}
    assert last["rows"]["S25-BB"]["shifted_from"] == DAYS[0].isoformat()  # steps over the gap day
    assert gap["rows"]["S25-PF"]["shifted_from"] == DAYS[0].isoformat()
    assert first["rows"]["S25-PF"]["shifted_from"] == DAYS[2].isoformat()  # circular
    assert first["calibration"]["thinned"]["binance_prints"] > 0


def test_a_finished_day_is_read_back_not_computed_again(read: dict[str, Any], monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def refused(*_: object, **__: object) -> None:
        raise AssertionError("computed again")

    monkeypatch.setattr(ho, "read_day", refused)
    with syn.no_network():
        assert ho.read_days(read["study"], read["plan"]) == read["records"]


def test_series_hold_nan_on_days_without_a_value(read: dict[str, Any]) -> None:
    records = read["records"]
    s = asm.slot_days(records, "S25-BB", mde=1.0)
    assert [d.isoformat() for d in s.days] == asm.kept_days(records, sa.BYBIT)
    assert s.random.shape == (200, 2) and s.neighbourhood is not None
    pf = asm.slot_days(records, "S25-PF")
    assert len(pf.days) == 3 and pf.worse_of_two is not None and pf.fills is not None
    excluded = asm.excluded(records, sa.BYBIT, DAYS)
    assert excluded == {DAYS[1]: ("sequence_gaps",)}
    assert asm.excluded(records, sa.BINANCE, [*DAYS, date(2026, 3, 6)]) == {
        date(2026, 3, 6): ("not read",)
    }


def test_a_venue_with_a_third_of_its_days_out_is_unreadable(read: dict[str, Any]) -> None:
    with syn.no_network():
        found = br.compute(read["study"], read["plan"], blind=True, mde={})
    statuses = {s: a.status for s, a in found["assessments"].items()}
    assert statuses["S25-BB"] == statuses["S25-T"] == "not assessed (H unreadable on Bybit)"
    assert all(check_status(v) for v in statuses.values())
    m = found["measurements"]
    assert m["unreadable"] == ["bybit_spot_BTCUSDT"] and "S25-PFT" in m["measured_rows_kills"]
    assert m["gap_days_put_back"]["S25-BB"]["put_back"] == 1
    assert next(step["step"] for step in m["edge_ladder"]) == "S25-T"
    sentences = br.sentences(found)
    assert len(sentences["each_slot_on_H"]) == 3 and sentences["S25-T"]
    assert math.isfinite(m["rows"]["S25-PF"]["trades_per_day"])
    assert np.isfinite(found["rows"]["S25-PF"].trades).all()


def test_on_p_only_h_candidates_are_assessed_and_the_table_gives_the_final_status(
    read: dict[str, Any],
) -> None:
    from trading_research.spot import reads

    on_h = {"S25-PF": "candidate", "S25-BB": "killed", "S25-T": "inconclusive"}
    with syn.no_network():
        found = br.compute(read["study"], read["plan"], blind=True, mde={}, h_statuses=on_h)
    statuses = {s: a.status for s, a in found["assessments"].items()}
    assert statuses["S25-BB"] == statuses["S25-T"] == "not taken to the block"
    assert statuses["S25-PF"] not in ("not taken to the block",)
    h = {"statuses": {s: {"status": v} for s, v in on_h.items()}}
    final = reads._final("P", found["assessments"], True, h)
    assert final["S25-BB"] == "killed" and final["S25-T"] == "inconclusive"
    assert final["S25-PF"].startswith(
        ("candidate, not confirmed", "confirmed", "killed on P", "void")
    )
