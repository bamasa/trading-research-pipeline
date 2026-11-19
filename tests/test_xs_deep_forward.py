"""The xs_deep test block stays sealed until its date, and is read only with the frozen sources."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from experiments import xs_deep_forward as fwd


def test_sealed_before_it_opens(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(fwd, "today", lambda: date(2026, 12, 14))
    with pytest.raises(SystemExit, match="sealed until 2026-12-15"):
        fwd.fetch(tmp_path)
    with pytest.raises(SystemExit, match="sealed"):
        fwd.read(tmp_path)


def test_read_refuses_sources_that_are_not_frozen(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(fwd, "today", lambda: date(2026, 12, 15))
    monkeypatch.setattr(fwd, "FROZEN", dict.fromkeys(fwd.SOURCES, "0" * 64))
    with pytest.raises(SystemExit, match="not the frozen sources"):
        fwd.read(tmp_path)


def test_the_blocks_are_the_registrations() -> None:
    assert (fwd.FIT_LAST - fwd.FIT_FIRST).days + 1 == 14
    assert (fwd.SCORED_LAST - fwd.SCORED_FIRST).days + 1 == 14
    assert date(2026, 11, 30) == fwd.SCORED_FIRST and fwd.OPENS > fwd.SCORED_LAST
