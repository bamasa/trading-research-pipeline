"""A block's read around its statuses: unreadability, R void, the MDE per block, the measurements."""

from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from trading_research.pipeline.short_horizon import heldout as ho
from trading_research.pipeline.short_horizon import reads

H_DAYS = [date(2025, 7, 8) + timedelta(days=i) for i in range(44)]


def _validated(usable: list[date], gaps: list[date] = ()) -> dict[str, Any]:  # type: ignore[assignment]
    return {"reasons": {}, "usable": list(usable), "gap_days": list(gaps), "unreadable": False}


def test_days_flagged_stale_count_once_toward_unreadability() -> None:
    block = SimpleNamespace(
        days_planned=tuple(H_DAYS), validated=_validated(H_DAYS[5:], H_DAYS[:1])
    )
    assert not ho.unreadable(block, set(H_DAYS[5:11]))  # type: ignore[arg-type]
    assert ho.unreadable(block, set(H_DAYS[5:12]))  # type: ignore[arg-type]
    assert not ho.unreadable(block, set(H_DAYS[:5]))  # type: ignore[arg-type]  # excluded already


def test_a_block_unreadable_once_its_stale_days_are_counted_assesses_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trading_research.pipeline.short_horizon import heldout

    monkeypatch.setattr(heldout.Block, "validated", property(lambda _: _validated(H_DAYS)))
    stale = [d.isoformat() for d in H_DAYS[:12]]
    monkeypatch.setattr(
        heldout, "compute_block", lambda *_, **__: ({}, {"unreadable": True, "stale_days": stale})
    )
    study = SimpleNamespace(scale=SimpleNamespace(days=lambda _: H_DAYS, name="registered"))
    session = SimpleNamespace(access=None, key=("H", 1), stamp="now")
    found = reads.compute_results(study, "H", session)  # type: ignore[arg-type]
    assert found["unreadable"] and found["stale_days"] == stale and found["taken_to_F"] == []
    assert set(found["statuses"].values()) == {"not assessed (H unreadable)"}


F_DAYS = [date(2025, 8, 21) + timedelta(days=i) for i in range(80)]
P_DAYS = [date(2025, 11, 17) + timedelta(days=i) for i in range(40)]
ON_D = {"timestamp": "2025-03-01T05:00:00+00:00", "statistic": "scale"}
ON_H = {"timestamp": "2025-07-20T00:00:00+00:00", "statistic": "dependence"}
ON_F = {"timestamp": f"{F_DAYS[50]}T12:00:00+00:00", "statistic": "scale"}


def _registration() -> Any:
    values = {("detector", "alarms_D"): [{"timestamp": ON_D["timestamp"], "channel": "scale"}],
              ("detector", "r_untestable_declared"): False}  # fmt: skip
    return SimpleNamespace(value=lambda path: values[path])


def _slot(days: list[date], value: np.ndarray) -> Any:
    from trading_research.pipeline.short_horizon.statuses import SlotDays

    n, rng = len(days), np.random.default_rng(0)
    return SlotDays("S-C", "taker", list(days), value, np.full(n, 10), gross=np.full(n, 10.0),
                    turnover=np.full(n, 1_000.0), ic=np.full(n, 0.05), random=rng.normal(-1, 1, (20, n)),
                    flipped=np.full(n, -2.0), shifted=np.full(n, -1.0), long_only=np.full(n, -1.0),
                    short_only=np.full(n, -1.0), neighbours=np.full((8, n), 1.0), latency=value - 0.1)  # fmt: skip


def _write(study: Any, block: str, results: dict[str, Any]) -> None:
    import json

    path = reads.results_path(study, block)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results), encoding="utf-8")


def test_r_is_void_when_any_earlier_runs_alarms_are_not_reproduced_on_their_own_dates() -> None:
    december = {"timestamp": "2024-12-15T00:00:00+00:00", "statistic": "scale"}
    run = [december, ON_D, ON_H]
    assert not reads.r_void(_registration(), run)  # a December alarm is not one of D's
    assert not reads.r_void(_registration(), run, {"H": [ON_H]})
    assert reads.r_void(_registration(), run, {"H": [ON_H | {"statistic": "scale"}]})
    assert reads.r_void(_registration(), run, {"H": []})
    assert reads.r_void(_registration(), [december, ON_H])


def test_a_void_r_is_never_tested_and_f_is_assessed_on_all_of_f(tmp_path: Any) -> None:
    study = SimpleNamespace(repo=tmp_path)
    value = np.where(np.array(F_DAYS) < F_DAYS[50], 4.0, -4.0) + 0.01 * np.arange(80)
    series = {"S-C": _slot(F_DAYS, value)}
    h = {"taken_to_F": ["S-C"], "statuses": {"S-C": "candidate"}, "R_void": False,
         "info": {"alarms_in_block": [ON_H]}}  # fmt: skip
    _write(study, "H", h)
    found = reads.statuses_f(study, series, [ON_D, ON_H, ON_F], _registration())  # type: ignore[arg-type]
    assert not found["R_void"] and found["R"]["S-C"]["outcome"] == "supported"
    _write(study, "H", h | {"info": {"alarms_in_block": []}})  # H's run recorded no alarm
    found = reads.statuses_f(study, series, [ON_D, ON_H, ON_F], _registration())  # type: ignore[arg-type]
    assert found["R_void"] and found["R"]["S-C"]["outcome"] == "void"
    assert found["split"]["on_all_of_F"] and found["split"]["reason"] == "R void"
    assert found["after_F"]["S-C"].endswith(", assessed on all of F")


def test_r_found_void_on_p_scores_p_with_the_full_bar(tmp_path: Any) -> None:
    study = SimpleNamespace(repo=tmp_path)
    _write(study, "H", {"info": {"alarms_in_block": [ON_H]}})
    _write(study, "F", {"qualifying_for_P": ["S-C"], "R": {"S-C": {"outcome": "supported"}},
                        "after_F": {"S-C": "candidate"}, "statuses": {"S-C": "candidate"},
                        "info": {"alarms_in_block": [ON_F]}})  # fmt: skip
    series = {"S-C": _slot(P_DAYS, 3.0 + np.random.default_rng(1).normal(0, 1, 40))}
    kept = reads.statuses_p(study, series, alarms=[ON_D, ON_H, ON_F], registration=_registration())  # type: ignore[arg-type]
    assert not kept["R_void"] and kept["statuses"]["S-C"] == "candidate; the break did not last"
    moved = ON_F | {"timestamp": f"{F_DAYS[51]}T12:00:00+00:00"}
    void = reads.statuses_p(study, series, alarms=[ON_D, ON_H, moved], registration=_registration())  # type: ignore[arg-type]
    assert void["R_void"] and void["statuses"]["S-C"] == "confirmed"
