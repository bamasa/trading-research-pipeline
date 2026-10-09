"""The study's simulator settings, S-M's plan and sigma_ref, and the passive cells."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from datetime import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from trading_research.market_making.quoters import (
    MarketView,
    Quotes,
    RegimeGuard,
    SkewQuoter,
    TouchQuoter,
)
from trading_research.market_making.signal_entry import SignalEntryQuoter
from trading_research.market_making.simulator import simulate_day
from trading_research.market_making.synthetic import hand_built_market, random_market
from trading_research.pipeline import regime
from trading_research.pipeline.market_making import SignalLean, make_quoter, signal_tapes
from trading_research.pipeline.regime import RegimePolicy
from trading_research.pipeline.short_horizon_execution import (
    CLIP_BTC,
    LEANS,
    S1_FROZEN,
    flag_tape,
    lean_plan,
    sigma_ref_from_days,
    signal_entry_cell,
    study_config,
)

REGISTRATION = Path(__file__).resolve().parents[1] / "configs" / "short_horizon_2025.yaml"


def test_the_settings_are_the_registered_ones() -> None:
    common = yaml.safe_load(REGISTRATION.read_text(encoding="utf-8"))["common"]
    passive, maker = common["passive_fill"], common["market_maker"]
    config = study_config(3)
    assert config.clip_btc == common["clip_btc"] == CLIP_BTC
    assert config.clip_notional is None
    assert config.order_latency_ns == passive["order_latency_ms"] * 1_000_000
    assert config.cancel_latency_ns == passive["cancel_latency_ms"] * 1_000_000
    assert config.feed_latency_ns == passive["feed_latency_ms"]
    assert config.soft_limit_clips == passive["soft_limit_clips"] == maker["soft_limit_clips"]
    assert config.warmup_s == passive["warmup_s"]
    assert config.stop_quoting_at == time.fromisoformat(passive["stop_quoting_at"])
    assert config.flatten_at == time.fromisoformat(passive["flatten_at"])
    assert list(config.markout_horizons_s) == passive["markout_horizons_s"]
    assert config.decomposition_horizon_s == passive["decomposition_horizon_s"]
    assert config.suspend_after_pause_s == passive["suspend_after_pause_s"]
    assert config.vol_half_life_s == passive["vol_half_life_s"]
    assert config.touch_window_s == passive["touch_window_s"]
    assert config.flatten_slippage_bp == passive["flatten_slippage_bp"]
    for name in ("queue_priority", "cancel_attribution", "arrival_growth", "crossed_policy"):
        assert getattr(config, name).value == passive[name]
    assert config.max_sequence_gaps == 3
    assert maker["S1"] == S1_FROZEN
    assert list(LEANS) == maker["lean"]
    assert study_config(3, fill_rule="touch").fill_rule == "touch"


@dataclass
class Recorder:
    """The touch quoter, recording the volatility every decision reads."""

    name: str = "recorder"
    uses_future: bool = False
    vols: list[float] = field(default_factory=list)

    def quotes(self, view: MarketView, position: float) -> Quotes:
        self.vols.append(view.vol_bp_1m)
        return TouchQuoter().quotes(view, position)


def test_sigma_ref_is_the_median_volatility_the_rows_read() -> None:
    config = study_config(0, warmup_s=0.0)
    days = [random_market(seed, n_snapshots=900, max_step=2, lot=0.001) for seed in (1, 2)]
    read: list[float] = []
    for events in days:
        recorder = Recorder()
        simulate_day(events, recorder, config)
        assert len(recorder.vols) == len(events.book_ts)
        read += recorder.vols
    values = np.array(read)
    expected = float(np.median(values[np.isfinite(values)]))
    assert sigma_ref_from_days(iter(days), config) == expected
    with pytest.raises(ValueError, match="at least one day"):
        sigma_ref_from_days([], config)


def test_the_lean_plan_wires_s1_the_lean_and_both_guards() -> None:
    config = study_config(0)
    pulled = lean_plan(config, 3.0, lean=2.0, hold_s=50.0, policy=RegimePolicy.parse("pull_4h"))
    assert pulled.guard_window_min == 240.0
    quoter = make_quoter(pulled)
    assert isinstance(quoter, RegimeGuard) and quoter.action == "pull"
    assert quoter.window_min == 240.0
    lean = quoter.base
    assert isinstance(lean, SignalLean) and (lean.lam, lean.window_s) == (2.0, 50.0)
    assert lean.base == SkewQuoter(2.0, 0.5, 4.0, 3.0, 2.0)
    widened = lean_plan(config, 3.0, lean=1.0, hold_s=20.0, policy=RegimePolicy.parse("widen_24h"))
    assert make_quoter(widened).action == "widen"
    assert make_quoter(widened).window_min == 1440.0
    plain = lean_plan(config, 3.0, lean=1.0, hold_s=20.0)
    assert isinstance(make_quoter(plain), SignalLean)
    unguarded = regime.maker_plan(widened, RegimePolicy())
    assert unguarded.regime_policy == "none" and isinstance(make_quoter(unguarded), SignalLean)
    assert regime.maker_plan(plain, RegimePolicy.parse("pull_4h")) == lean_plan(
        config, 3.0, lean=1.0, hold_s=20.0, policy=RegimePolicy.parse("pull_4h")
    )
    with pytest.raises(ValueError, match="window"):
        RegimePolicy("pull", 0)
    with pytest.raises(ValueError, match="clip_btc"):
        lean_plan(config.with_(clip_btc=None, clip_notional=100.0), 3.0, lean=1.0, hold_s=20.0)


def _wide_market():
    """A minute of books with levels five ticks apart and a touch that moves every
    second, so S1's half-spread (about 10 bp) lands inside the visible book."""
    snapshots, prints = [], []
    for row in range(600):
        bid = 10_000 + (row // 10) % 2
        bids = [(bid - 5 * level, 0.05) for level in range(10)]
        asks = [(bid + 2 + 5 * level, 0.05) for level in range(10)]
        snapshots.append((100.0 * row, bids, asks))
        if row % 7 == 0:
            prints.append((100.0 * row + 50, bid - 10, 0.02, -1))
            prints.append((100.0 * row + 60, bid + 12, 0.02, 1))
    return hand_built_market(snapshots, prints, funding=[(0.0, 1e-4)], lot=0.001)


def test_s_m_pulls_its_quotes_inside_the_guard_window_and_keeps_the_hard_limit() -> None:
    events = _wide_market()
    config = study_config(0, warmup_s=0.0)
    sigma = sigma_ref_from_days([events], config)
    flag_ns = int(events.book_ts[200])
    window_s = 15.0
    tapes = flag_tape(np.array([flag_ns, flag_ns]))
    stamps = events.book_ts[::50]
    tapes.update(signal_tapes(stamps, np.where(np.arange(len(stamps)) % 2 == 0, 3.0, -3.0)))
    for action in ("pull", "widen"):
        policy = "guard_pull" if action == "pull" else "guard_widen"
        guard = RegimePolicy(action, window_s / 3600)  # a 15 s window on a minute of books
        plan = lean_plan(config, sigma, lean=1.0, hold_s=5.0, policy=guard)
        assert plan.regime_policy == policy
        result = simulate_day(events, make_quoter(plan), config, tapes)
        decided = result.orders["decided_ns"].to_numpy()
        inside = (decided > flag_ns) & (decided <= flag_ns + window_s * 1e9)
        assert len(decided) > 0
        if policy == "guard_pull":
            assert not inside.any()
        else:
            assert inside.any()
        assert result.counters["max_abs_position"] <= 2 * CLIP_BTC + 1e-12
        assert (result.orders["size"] <= CLIP_BTC + 1e-12).all()


def test_signal_entry_cells_and_flag_tapes() -> None:
    cell = signal_entry_cell("S-P", study_config(0), hold_s=50.0, cancel_on_flag=True)
    first, second = cell.make_quoter(), cell.make_quoter()
    assert isinstance(first, SignalEntryQuoter) and first is not second
    assert (first.hold_s, first.passive_exit) == (50.0, False)
    assert cell.config.cancel_entries_on_flag
    plain = signal_entry_cell("S-P", study_config(0), hold_s=50.0)
    assert not plain.config.cancel_entries_on_flag
    assert pickle.loads(pickle.dumps(cell.make_quoter))().hold_s == 50.0
    tape = flag_tape(np.array([3_000_000_000, 1_000_000_000, 3_000_000_000]))["flag_s"]
    assert tape.ts.tolist() == [1_000_000_000, 3_000_000_000]
    assert tape.values.tolist() == [1.0, 3.0]
