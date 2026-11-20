"""xvenue's held-out months are fetched and read only with the frozen sources."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments import xvenue_funding as xv


def test_the_study_sources_are_the_frozen_ones() -> None:
    assert xv.unfrozen() == []


def test_read_refuses_other_sources(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(xv, "FROZEN", dict.fromkeys(xv.SOURCES, "0" * 64))
    with pytest.raises(SystemExit, match="not the frozen sources"):
        xv.read(tmp_path)


def test_btc_and_eth_are_not_in_the_universe() -> None:
    assert "BTCUSDT" not in xv.COINS and "ETHUSDT" not in xv.COINS and len(xv.COINS) == 15
