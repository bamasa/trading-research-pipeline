"""carry_ml's held-out months are fetched and read only with the frozen sources, and read once."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from experiments import carry_ml


def test_the_study_sources_are_the_frozen_ones() -> None:
    assert carry_ml.unfrozen() == []


def test_fetch_and_read_refuse_other_sources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(carry_ml, "FROZEN", dict.fromkeys(carry_ml.SOURCES, "0" * 64))
    with pytest.raises(SystemExit, match="code freeze"):
        carry_ml.fetch(tmp_path)
    with pytest.raises(SystemExit, match="not the frozen sources"):
        carry_ml.read(tmp_path)


def test_btc_and_eth_are_never_candidates_on_h() -> None:
    for month in pd.period_range(carry_ml.H_FIRST, carry_ml.H_LAST, freq="M"):
        assert not carry_ml.readable("BTCUSDT", month)
        assert not carry_ml.readable("ETHUSDT", month)
        assert carry_ml.readable("FILUSDT", month)
    assert not carry_ml.readable("FILUSDT", carry_ml.H_LAST + 1)
