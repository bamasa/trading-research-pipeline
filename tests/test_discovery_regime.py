"""The regime policy as an axis of the discovery search, flags from the break detector."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_research.backtest.costs import TakerCosts
from trading_research.pipeline.discovery import (
    _guard,
    _trade_all,
    _with_neighbourhood,
    panel_flags,
    run,
)
from trading_research.pipeline.regime import RegimePolicy
from trading_research.validation.structural_breaks import MonitorSpec

H = 3_600 * 1_000_000_000
SMALL = MonitorSpec(
    history_len=60,
    online_len=24,
    min_history=50,
    statistics=("scale", "dependence"),
    threshold={"scale": 8.0, "dependence": 7.0},
)


def _panel(days: int = 12, step_s: int = 60, seed: int = 2) -> tuple[pd.DataFrame, dict]:
    """Six instruments sharing a market factor that turns five times as rough on day 8."""
    rng = np.random.default_rng(seed)
    periods = days * 86_400 // step_s
    index = pd.date_range("2024-02-01", periods=periods, freq=f"{step_s}s", tz="UTC")
    rough = np.where(index >= pd.Timestamp("2024-02-08 06:00", tz="UTC"), 5.0, 1.0)
    common = np.cumsum(rng.normal(0, 2e-4, len(index)) * rough)
    panel, books = {}, {}
    for k in range(6):
        mid = 100.0 * np.exp(common + np.cumsum(rng.normal(0, 3e-4, len(index))))
        bid, ask = mid - 0.01, mid + 0.01
        panel[f"S{k}"] = pd.Series(np.log(mid), index=index)
        books[f"S{k}"] = pd.DataFrame({"bid_price_0": bid, "ask_price_0": ask}, index=index)
    return pd.DataFrame(panel), books


def test_a_widen_policy_is_refused_before_anything_is_read(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no spread of its own"):
        run(book_root=tmp_path, fetch=False, regimes=("none", "widen_4h"))
    with pytest.raises(ValueError, match="'none' or"):
        run(book_root=tmp_path, fetch=False, regimes=("sometimes",))


def test_the_guard_blanks_rows_after_each_flag() -> None:
    panel, _ = _panel(days=1)
    start = panel.index[0].value
    mask = _guard(panel, np.array([start + 2 * H]), RegimePolicy.parse("pull_4h"))
    times = (panel.index[mask] - panel.index[0]) / pd.Timedelta(1, unit="h")
    assert times.min() > 2.0 and times.max() <= 6.0
    assert mask.sum() == 4 * 60


def test_a_guarded_search_cell_trades_less_and_none_is_unchanged() -> None:
    panel, books = _panel(days=3)
    values = panel.to_numpy()
    costs = TakerCosts(fee_bp_per_side=5.5, slippage_bp=0.5)
    block = slice(0, len(panel))
    plain, _ = _trade_all(panel, books, values, costs, 30, 30, 60.0, block)
    off, _ = _trade_all(
        panel, books, values, costs, 30, 30, 60.0, block, guard=np.zeros(len(panel), dtype=bool)
    )
    np.testing.assert_array_equal(plain, off)
    half = np.arange(len(panel)) >= len(panel) // 2
    pulled, _ = _trade_all(panel, books, values, costs, 30, 30, 60.0, block, guard=half)
    assert 0 < len(pulled) < len(plain)


def test_regimes_are_not_mixed_across_neighbourhoods() -> None:
    rows = [
        {"lookback_s": lb, "hold_s": h, "rate": 20.0, "regime_h": w, "net_per_trade_bp": v}
        for lb in (100, 200)
        for h in (100, 200)
        for w, v in ((0.0, 5.0), (24.0, -40.0))
    ]
    scored = _with_neighbourhood(pd.DataFrame(rows))
    assert (scored.loc[scored["regime_h"] == 0.0, "neighbourhood_bp"] == 5.0).all()


pytest.importorskip("structural_break")


def test_the_panel_flags_profile_reads_only_the_search_block() -> None:
    panel, _ = _panel()
    cut = 7 * 1440
    flags, alarms = panel_flags(panel, cut, SMALL)
    assert len(flags) == len(alarms) >= 1
    assert (alarms["timestamp"] > pd.Timestamp("2024-02-08 06:00", tz="UTC")).any()
    changed = panel.copy()
    changed.iloc[cut:] = changed.iloc[cut:] * 1.05 + 0.3
    _, moved = panel_flags(changed, cut, SMALL)
    edge = panel.index[cut]
    kept = list(moved["timestamp"][moved["timestamp"] <= edge])
    assert kept == list(alarms["timestamp"][alarms["timestamp"] <= edge])


def test_a_span_too_short_for_the_detector_says_so() -> None:
    panel, _ = _panel(days=2)
    with pytest.raises(RuntimeError, match="cannot run on this span"):
        panel_flags(panel, 1440, SMALL)


def test_the_whole_run_searches_the_regime_axis(monkeypatch: pytest.MonkeyPatch) -> None:
    """End to end on a synthetic panel: the flags stage, the axis, the held-out read."""
    from trading_research.pipeline import discovery

    panel, books = _panel(days=4, step_s=5)
    audit = pd.DataFrame(
        {"source": ["cross-section", "own book"], "median": [0.0, 0.0], "strongest": [0.2, 0.1]}
    )
    monkeypatch.setattr(discovery, "_load_panel", lambda *_, **__: (panel, books))
    monkeypatch.setattr(discovery, "_source_audit", lambda *_, **__: audit)
    monkeypatch.setattr(discovery, "MIN_TRADES_PER_DAY", 0.0)
    lines: list[str] = []
    result = run(fetch=False, regimes=("none", "pull_4h"), detector=SMALL, report=lines.append)
    names = [stage.name for stage in result.stages]
    assert names[:3] == ["load", "regime flags", "audit sources"]
    search = next(s for s in result.stages if s.name == "search")
    assert search.table is not None
    assert set(search.table["regime_h"]) == {0.0, 4.0}
    assert "regime " in search.decision
    assert result.held_out["trades"] > 0


def test_the_command_refuses_a_policy_the_mode_cannot_carry(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from trading_research.cli import app

    args = ["reproduce", "--no-fetch", "--book", str(tmp_path), "--regimes", "none,widen_4h"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 2
    assert "no spread of its own" in result.output
