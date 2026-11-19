"""The cross-section study's pieces on planted panels: no look-ahead, honest costs, a planted edge found."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from trading_research.xs_deep import evaluate, features
from trading_research.xs_deep.panel import Panel, day_bars

N, DAYS = 12, 3


def planted(strength: float, seed: int = 0) -> Panel:
    """Relative one-minute moves that partly reverse over the next ten minutes."""
    rng = np.random.default_rng(seed)
    t = DAYS * 1440
    shock = rng.normal(0, 5, (t, N))
    rel = shock.copy()
    for lag in range(1, 11):
        rel[lag:] -= strength * shock[:-lag] / 10
    market = rng.normal(0, 3, (t, 1))
    logm = np.cumsum((rel + market) * 1e-4, axis=0) + np.log(100.0)
    times = pd.date_range("2024-02-01 00:01", periods=t, freq="1min", tz="UTC")
    mid = np.exp(logm)
    return Panel(
        times=times,
        symbols=tuple(f"S{i}" for i in range(N)),
        mid=mid,
        spread_bp=np.full((t, N), 2.0),
        imbalance=rng.uniform(-1, 1, (t, N)),
    )


def test_inputs_do_not_read_later_bars() -> None:
    panel = planted(0.5)
    before = features.build(panel)
    later = Panel(
        panel.times, panel.symbols, panel.mid.copy(), panel.spread_bp.copy(), panel.imbalance.copy()
    )
    t = 2000
    later.mid[t + 1 :] *= 1.5
    later.spread_bp[t + 1 :] = 9.0
    later.imbalance[t + 1 :] = 0.0
    after = features.build(later)
    np.testing.assert_array_equal(before.tab[: t + 1], after.tab[: t + 1])
    np.testing.assert_array_equal(before.series[: t + 1], after.series[: t + 1])
    # A target whose exit is bar t is unchanged.
    np.testing.assert_allclose(before.fwd[t - 11], after.fwd[t - 11])


def test_target_skips_the_decision_bar_and_cost_is_fee_plus_half_spreads() -> None:
    panel = planted(0.0)
    frame = features.build(panel)
    logm = np.log(panel.mid)
    t = 500
    expected = (logm[t + 11] - logm[t + 1]) * 1e4
    np.testing.assert_allclose(frame.fwd[t], expected)
    np.testing.assert_allclose(frame.target[t].mean(), 0.0, atol=1e-9)
    np.testing.assert_allclose(frame.cost[t], 2 * 5.5 + 2.0)


def test_training_rows_never_reach_the_scored_days() -> None:
    panel = planted(0.0)
    frame = features.build(panel)
    days = [date(2024, 2, 1) + timedelta(days=k) for k in range(DAYS)]
    at = evaluate.rows(panel.times, frame.usable(), days[:2], 5, train=True)
    exit_day = (panel.times[at + 11] - np.timedelta64(1, "m")).date
    assert set(exit_day) <= set(days[:2])
    scored = evaluate.rows(panel.times, frame.usable(), days[2:], 10, train=False)
    assert (np.asarray(panel.times[scored].minute) % 10 == 0).all()


@pytest.mark.parametrize("name", ["reversal", "ridge"])
def test_a_planted_reversal_is_ranked_and_a_null_is_not(name: str) -> None:
    days = [date(2024, 2, 1) + timedelta(days=k) for k in range(DAYS)]
    edge, null = [], []
    for strength, sink in ((0.8, edge), (0.0, null)):
        panel = planted(strength, seed=1)
        frame = features.build(panel)
        at, scores = evaluate.fold(name, frame, panel.times, days[:2], days[2:])
        sink.append(evaluate.decisions(frame, at, scores)["ic"].mean())
    assert edge[0] > 0.1
    assert abs(null[0]) < 0.1  # about four standard errors of 142 decision times


def test_a_day_of_snapshots_becomes_1440_bars_carried_five_minutes() -> None:
    stamps = pd.to_datetime(["2024-02-01 00:00:30", "2024-02-01 00:20:00"], utc=True)
    book = pd.DataFrame(
        {
            "timestamp": stamps,
            "bid_price_0": [99.0, 100.0],
            "bid_size_0": [3.0, 1.0],
            "ask_price_0": [101.0, 102.0],
            "ask_size_0": [1.0, 1.0],
        }
    )
    bars = day_bars(book, date(2024, 2, 1))
    assert len(bars) == 1440
    assert bars["mid"].iloc[0] == 100.0  # the 00:01 bar
    assert bars["imbalance"].iloc[0] == pytest.approx(0.5)
    assert np.isfinite(bars["mid"].iloc[5]) and np.isnan(bars["mid"].iloc[6])
    assert bars["mid"].iloc[19] == 101.0  # 00:20, inclusive


def test_holm_is_monotone_and_capped() -> None:
    adj = evaluate.holm({"a": 0.01, "b": 0.04, "c": 0.5})
    assert adj["a"] == pytest.approx(0.03)
    assert adj["b"] == pytest.approx(0.08)
    assert adj["c"] == pytest.approx(0.5)


def test_the_causal_convolution_is_conv1d_and_never_reads_later_bars() -> None:
    torch = pytest.importorskip("torch")
    from trading_research.xs_deep.nets import CausalConv

    conv, ref = CausalConv(4, 8, 4), torch.nn.Conv1d(4, 8, 3, dilation=4)
    with torch.no_grad():
        ref.weight.copy_(conv.linear.weight.view(8, 3, 4).permute(0, 2, 1))
        ref.bias.copy_(conv.linear.bias)
    x = torch.randn(5, 60, 4)
    padded = torch.nn.functional.pad(x.transpose(1, 2), (8, 0))
    torch.testing.assert_close(conv(x), ref(padded).transpose(1, 2), atol=1e-5, rtol=1e-5)
    later = x.clone()
    later[:, 40:] += 1.0
    torch.testing.assert_close(conv(later)[:, :40], conv(x)[:, :40])


def test_only_the_cross_sectional_network_reads_the_other_instruments() -> None:
    torch = pytest.importorskip("torch")
    from trading_research.xs_deep.nets import build

    tab, seq = torch.randn(2, 6, 12), torch.randn(2, 6, 60, 4)
    mask = torch.ones(2, 6, dtype=torch.bool)
    mask[:, 5] = False
    other = seq.clone()
    other[:, 1] += 2.0  # a usable instrument moves
    hidden = seq.clone()
    hidden[:, 5] += 2.0  # a masked one moves
    for name, reads_others in (("tcn", False), ("xs_transformer", True)):
        net = build(name, 32).eval()
        with torch.no_grad():
            base, moved, masked = net(tab, seq, mask), net(tab, other, mask), net(tab, hidden, mask)
        assert (not torch.allclose(base[:, 0], moved[:, 0])) is reads_others
        torch.testing.assert_close(base[:, :5], masked[:, :5])
