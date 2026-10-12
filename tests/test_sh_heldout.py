"""A block read's measurements wired to the slots: the twin, the flag sets, the measured
cells, the series' extras, the put-back view of the gap days."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from trading_research.pipeline.regime import NS_PER_HOUR, usable_hours
from trading_research.pipeline.short_horizon import heldout as ho
from trading_research.pipeline.short_horizon.passive import SimCell
from trading_research.search_engine.tapes import GUARD_WINDOWS_H, CandidateTape

DAYS = [date(2025, 8, 21) + timedelta(days=i) for i in range(4)]
C_KEY = "BTCUSDT|C|gbm|micro|open|hold_s=50|trades_per_day=240|label_fee_bp=0|regime=pull_4h"
R_KEY = "BTCUSDT|R|tcn|tensor+micro|open|hold_s=50|k=1|beta_bp=0|regime=none"
S25 = "BTCUSDT|C|gbm|micro|open|hold_s=50|trades_per_day=240|label_fee_bp=0|regime=none"


def test_the_twin_and_the_other_regimes_of_a_key() -> None:
    assert ho.twin_key(C_KEY) == S25 and ho.twin_key(S25) == S25
    assert ho.with_regime(S25, "pull_120h") == C_KEY.replace("pull_4h", "pull_120h")


def _tape(stamps: np.ndarray, flags: np.ndarray) -> CandidateTape:
    from trading_research.pipeline.execution import guarded_rows

    n = len(stamps)
    return CandidateTape(
        DAYS[0], "cfg", "C", np.arange(n), stamps, np.ones(n, dtype=np.int8),
        {"p_long": np.full(n, 0.6), "p_short": np.full(n, 0.2)},
        {"quiet_out": np.ones(n, dtype=bool), "tight_only": np.ones(n, dtype=bool)},
        {w: guarded_rows(stamps, flags, w * NS_PER_HOUR) for w in GUARD_WINDOWS_H},
        {"theta_long": 0.5, "theta_short": 0.5},
    )  # fmt: skip


def test_a_tape_reguarded_under_other_flags_changes_that_window_only() -> None:
    stamps = np.arange(10, dtype=np.int64) * NS_PER_HOUR
    tape = _tape(stamps, np.array([1 * NS_PER_HOUR]))
    other = ho.reguarded(tape, np.array([6 * NS_PER_HOUR]), 4)
    assert tape.guards[4].tolist() == [
        False,
        False,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
    ]
    assert other.guards[4].tolist() == [False] * 7 + [True, True, True]
    assert other.guards[24].tolist() == tape.guards[24].tolist() and other.rows is tape.rows
    assert not ho.reguarded(tape, np.zeros(0, dtype=np.int64), 24).guards[24].any()


def test_flag_sets_draw_as_many_flags_as_the_block_has_inside_its_days() -> None:
    inside = usable_hours(DAYS)[[3, 50, 70]]
    outside = np.array([usable_hours([DAYS[0] - timedelta(days=5)])[0]])
    block = SimpleNamespace(
        flags_ns=np.r_[outside, inside],
        study=SimpleNamespace(scale=SimpleNamespace(taker_flag_sets=7)),
    )
    count, sets = ho.flag_sets(block, DAYS)  # type: ignore[arg-type]
    assert count == 3 and len(sets) == 7
    hours = set(usable_hours(DAYS).tolist())
    for found in sets:
        assert len(found) == 3 == len(set(found.tolist())) and set(found.tolist()) <= hours
    assert ho.flag_sets(block, []) == (0, [])  # type: ignore[arg-type]
    assert (
        len(
            ho.flag_sets(
                SimpleNamespace(flags_ns=np.zeros(0, dtype=np.int64), study=block.study), DAYS
            )[1][0]
        )
        == 0
    )  # type: ignore[arg-type]


def _block(finalists: dict[str, dict[str, Any]]) -> Any:
    return SimpleNamespace(finalist=lambda slot: finalists.get(slot))


def test_the_passive_slots_carry_their_measured_cells_and_the_twin() -> None:
    sp_cells = [C_KEY, C_KEY.replace("trades_per_day=240", "trades_per_day=120")]
    found = ho.passive_slots(_block({"S-P": {"key": C_KEY, "cells": sp_cells},
                                     "S-M": {"key": C_KEY, "guard": "pull_4h"},
                                     "S25-Z": {"key": S25, "cells": [S25]}}))  # fmt: skip
    by_slot = {p.slot: p for p in found}
    sp, sm, sz = by_slot["S-P"], by_slot["S-M"], by_slot["S25-Z"]
    assert sp.window_h == 4 and [c.key for c in sp.neighbours] == sp_cells[1:]
    assert [name for name, _ in sp.measured] == [
        "regime/none",
        "regime/pull_24h",
        "regime/pull_120h",
    ]
    assert [c.policy for _, c in sp.measured] == ["none", "pull_24h", "pull_120h"]
    assert sp.measured[0][1].key == S25 and len(sp.cells) == 5
    assert [name for name, _ in sm.measured] == ["regime/none", "regime/widen_4h", "regime/pull_24h",
                                                 "regime/widen_24h", "regime/pull_120h", "regime/widen_120h"]  # fmt: skip
    twin = sm.measured[0][1]
    assert twin == SimCell("S-M", S25, "maker", 50.0, lean=1.0, policy="none")
    assert all(c.lean == 1.0 and c.key == C_KEY for _, c in sm.measured[1:])
    assert sm.cell.policy == "pull_4h" and [c.lean for c in sm.neighbours] == [0.5, 2.0]
    assert sz.window_h == 0 and sz.measured == [] and sz.cell.fees == "zero"
    unguarded = ho.passive_slots(_block({"S-M": {"key": S25, "guard": "none"}}))[0]
    assert [name for name, _ in unguarded.measured] == [f"regime/{p}" for p in ho.MAKER_GUARDS]


def _rows(days: list[date], variants: dict[str, float]) -> pd.DataFrame:
    rows = []
    for variant, value in variants.items():
        for k, day in enumerate(days):
            rows.append({"run_slot": "S-P", "variant": variant, "day": day.isoformat(), "excluded": False,
                         "net_per_100": value + k, "trades": 2, "net_usdt": 1.0, "passive_notional": 10.0,
                         "making_usdt": 0.0, "net_bp_trade": 1.0, "gross_bp_trade": 1.0})  # fmt: skip
    return pd.DataFrame(rows)


def test_the_passive_series_carry_the_twin_the_flag_sets_and_each_measurement() -> None:
    p = ho.PassiveSlot(
        "S-P", SimCell("S-P", C_KEY, "entry", 50.0, policy="pull_4h"), [], ic=[0.1] * 4
    )
    rows = _rows(DAYS, {"true": 1.0, "regime/none": 2.0, "regime/pull_24h": 3.0, "exit/passive": -1.0,
                        "flagset/0": 5.0, "flagset/1": 6.0, "flipped": -1.0})  # fmt: skip
    s = ho.passive_series(p, rows, DAYS, 0, 4, False)
    assert s.value.tolist() == [1.0, 2.0, 3.0, 4.0] and s.window_h == 4
    assert set(s.extra) == {"regime/none", "regime/none/trades", "regime/pull_24h", "regime/pull_24h/trades",
                            "exit/passive", "exit/passive/trades", "twin", "flagsets"}  # fmt: skip
    assert s.extra["twin"].tolist() == [2.0, 3.0, 4.0, 5.0] and s.extra["exit/passive"][0] == -1.0
    assert s.extra["flagsets"].shape == (2, 4) and s.extra["flagsets"][1].tolist() == [
        6.0,
        7.0,
        8.0,
        9.0,
    ]
    assert s.extra["regime/none/trades"].tolist() == [2.0] * 4
    bare = ho.passive_series(p, _rows(DAYS, {"true": 1.0}), DAYS, 0, 4, False)
    assert bare.extra == {} and np.isnan(bare.flipped).all()


def test_the_put_back_view_reads_the_gap_days_as_usable_with_the_threshold_lifted() -> None:
    study = SimpleNamespace(dir=lambda *parts: Path("work").joinpath(*parts))
    block = ho.Block(study, "F", None, tuple(DAYS), values={("data", "max_sequence_gaps"): 3})  # type: ignore[arg-type]
    block.__dict__["validated"] = {"reasons": {DAYS[0].isoformat(): [], DAYS[1].isoformat(): ["sequence_gaps"],
                                               DAYS[2].isoformat(): [], DAYS[3].isoformat(): ["sequence_gaps", "no_book_file"]},
                                   "usable": [DAYS[0], DAYS[2]], "gap_days": [DAYS[1]], "unreadable": False}  # fmt: skip
    block.__dict__["detector"] = {"flags_ns": [1]}
    assert (
        block.threshold == 3 and block.usable == [DAYS[0], DAYS[2]] and block.gap_days == [DAYS[1]]
    )
    view = block.put_back()
    assert (
        view.threshold == ho.GAPS_UNBOUNDED
        and view.usable == DAYS[:3]
        and view.gap_days == [DAYS[1]]
    )
    assert (
        view.validated is block.validated
        and view.detector is block.detector
        and view.work == block.work
    )
    assert not block.gaps_put_back and view.name == "F"


def test_a_taker_slot_reports_extras_only_for_a_window_or_a_control_fit() -> None:
    plain = ho.TakerSlot("S25", S25, [], DAYS, 2)
    assert plain.window_h == 0 and plain.extra() == {}
    guarded = ho.TakerSlot("S-C", C_KEY, [S25], DAYS, 2, [np.zeros(0, dtype=np.int64)] * 3)
    assert guarded.window_h == 4 and set(guarded.extra()) == {"guarded", "twin", "flagsets"}
    assert guarded.extra()["flagsets"].shape == (3, 4) and np.isnan(guarded.extra()["twin"]).all()
    network = ho.TakerSlot("S-R", R_KEY, [], DAYS, 2)
    network.has_control = True
    assert set(network.extra()) == {"control_ic_label", "control_ic_move"}


def test_run_passive_without_kills_is_the_true_run_of_each_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trading_research.pipeline.short_horizon import heldout_passive as hp

    seen: list[hp.VariantJob] = []
    monkeypatch.setattr(hp, "run_variants", lambda job: seen.append(job))
    monkeypatch.setattr(pd, "read_parquet", lambda _: pd.DataFrame({"variant": ["true"]}))
    block = SimpleNamespace(study=SimpleNamespace(scale=SimpleNamespace(random_direction_draws=2), workers=1,
                                                  repo="repo", symbol="BTCUSDT"), work=SimpleNamespace(),
                            access=None, flags_ns=np.zeros(0, dtype=np.int64), threshold=9, name="F",
                            fitted={"sigma_ref": 1.0})  # fmt: skip
    block.work = Path("work")
    own = SimpleNamespace(stamps_ns=np.zeros(0, dtype=np.int64), side=np.zeros(0, dtype=np.int64),
                          strength_bp=np.zeros(0))  # fmt: skip
    p = ho.PassiveSlot("S-P", SimCell("S-P", C_KEY, "entry", 50.0, policy="pull_4h"), [],
                       [("regime/none", SimCell("S-P", S25, "entry", 50.0))],
                       decisions={C_KEY: [own, own], S25: [own, own]}, flagged=[[own, own]])  # fmt: skip
    ho.run_passive(block, [p], DAYS[:2], [np.zeros(0, dtype=np.int64)], kills=False)  # type: ignore[arg-type]
    assert [[r.variant for r in job.runs] for job in seen] == [["true"], ["true"]]
    assert all(job.out.name == "passive_gap_days" and job.max_sequence_gaps == 9 for job in seen)
    seen.clear()
    ho.run_passive(block, [p], DAYS[:2], [np.zeros(0, dtype=np.int64)], kills=True)  # type: ignore[arg-type]
    names = [r.variant for r in seen[0].runs]
    assert names[:2] == ["true", "flipped"] and "regime/none" in names and "flagset/0" in names
    assert "exit/passive" in names and "fill/touch" not in names and seen[0].out.name == "passive"
