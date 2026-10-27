"""The frozen signal: copies bound by hash, θ checked, decisions on a day of prints."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_research.market_making.prereg import HeldOutLocked
from trading_research.spot_maker import blocks as bk
from trading_research.spot_maker import registration as rg
from trading_research.spot_maker import signal as sg
from trading_research.spot_maker import standin

DAYS = [date(2026, 5, 1), date(2026, 5, 2)]


def test_copy_check_load_and_decide(tmp_path: Path) -> None:
    source, repo = tmp_path / "source", tmp_path / "repo"
    binding = standin.write_source(source, DAYS)
    copied = sg.copy_signal(source, repo, binding)
    assert set(copied) == {rg.SIGNAL_SLOT_FILE, rg.SIGNAL_FIT}
    frozen = sg.load_signal(repo, binding)
    prints = [pd.read_parquet(bk.day_file(source, d)) for d in DAYS]
    start = int(prints[1]["timestamp"].iloc[0].normalize().value)
    found = sg.day_decisions(frozen, prints[1], prints[0], start)
    assert len(found.decision_ns) > 10 and set(np.unique(found.decision_side)) <= {-1, 1}
    assert np.isin(found.decision_ns, found.grid_ns).all()
    cold = sg.day_decisions(frozen, prints[1], None, start + 61 * 60 * 10**9)
    assert cold.decision_ns.min() >= start + 61 * 60 * 10**9


def test_a_wrong_hash_or_theta_is_refused(tmp_path: Path) -> None:
    source, repo = tmp_path / "source", tmp_path / "repo"
    binding = standin.write_source(source, DAYS[:1])
    with pytest.raises(HeldOutLocked):
        sg.copy_signal(source, repo, sg.SignalBinding("0" * 64, binding.fit_sha256, binding.theta))
    sg.copy_signal(source, repo, binding)
    with pytest.raises(HeldOutLocked):
        sg.load_signal(repo, sg.SignalBinding(binding.slot_sha256, binding.fit_sha256,
                                              {"theta_long": 0.5, "theta_short": 0.5}))  # fmt: skip
    (repo / rg.SIGNAL_DIR / rg.SIGNAL_FIT).write_bytes(b"changed")
    with pytest.raises(HeldOutLocked):
        sg.load_signal(repo, binding)
    assert sg.SignalBinding().slot_sha256 == rg.SIGNAL_SLOT_SHA256
