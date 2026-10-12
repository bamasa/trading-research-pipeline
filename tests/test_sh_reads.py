"""A block's read around its statuses: unreadability, R void, the MDE per block, the measurements."""

from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace
from typing import Any

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
