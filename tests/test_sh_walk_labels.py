"""The book walk and the time-based labels, against explicit per-row loops."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from _sh_synthetic import make_rows
from trading_research.backtest.walk import walk, walk_cost_bp, walk_rows
from trading_research.features.timed import stamps_ns
from trading_research.labels.taker_vwap import (
    WalkedTakerLabel,
    control_move_bp,
    entry_exit_snapshots,
    forward_move_bp,
    purge_rows,
)


@pytest.fixture(scope="module")
def rows() -> pd.DataFrame:
    return make_rows(500, seed=3)


def _loop_walk(prices: list[float], sizes: list[float], clip: float) -> tuple[float, bool]:
    left, paid = clip, 0.0
    for price, size in zip(prices, sizes, strict=True):
        take = min(left, size)
        paid += take * price
        left -= take
        if left <= 0:
            return paid / clip, False
    return (paid + left * prices[-1]) / clip, True


def test_walk_matches_a_loop_at_the_touch_through_levels_and_beyond() -> None:
    prices = np.array([[100.0, 100.5, 101.0], [100.0, 100.5, 101.0], [100.0, 100.5, 101.0]])
    sizes = np.array([[2.0, 1.0, 1.0], [0.5, 1.0, 1.0], [0.5, 0.5, 0.5]])
    result = walk(prices, sizes, 1.5)
    for i in range(3):
        vwap, beyond = _loop_walk(list(prices[i]), list(sizes[i]), 1.5)
        assert result.vwap[i] == pytest.approx(vwap, rel=1e-15)
        assert bool(result.exceeds[i]) is beyond
    assert result.levels.tolist() == [1, 2, 3]
    # 0.5 at 100, 0.5 at 100.5, 0.5 at 101 (all visible), nothing beyond.
    assert result.vwap[2] == pytest.approx((50.0 + 50.25 + 50.5) / 1.5)


def test_walk_refuses_prices_that_are_not_raw() -> None:
    with pytest.raises(ValueError, match="raw book prices"):
        walk(np.array([[-0.3, 0.1]]), np.array([[1.0, 1.0]]), 0.5)
    with pytest.raises(ValueError, match="positive"):
        walk(np.array([[1.0]]), np.array([[1.0]]), 0.0)


def test_walk_cost_is_spread_and_depth_in_bp(rows) -> None:
    cost = walk_cost_bp(rows, 0.01)
    mid = (rows["bid_price_0"] + rows["ask_price_0"]).to_numpy() / 2
    spread_bp = (rows["ask_price_0"] - rows["bid_price_0"]).to_numpy() / mid * 1e4
    assert np.all(cost >= spread_bp - 1e-9)
    big = walk_cost_bp(rows, 2.0)
    assert np.all(big >= cost - 1e-9)


def _loop_label(rows: pd.DataFrame, label: WalkedTakerLabel) -> np.ndarray:
    stamps = stamps_ns(rows)
    bids = [[rows[f"bid_price_{i}"].iloc[j] for i in range(10)] for j in range(len(rows))]
    bsz = [[rows[f"bid_size_{i}"].iloc[j] for i in range(10)] for j in range(len(rows))]
    asks = [[rows[f"ask_price_{i}"].iloc[j] for i in range(10)] for j in range(len(rows))]
    asz = [[rows[f"ask_size_{i}"].iloc[j] for i in range(10)] for j in range(len(rows))]
    out = np.full(len(rows), np.nan)
    for t in range(len(rows)):
        e = next((j for j in range(len(rows)) if stamps[j] >= stamps[t] + 10_000_000), None)
        if e is None:
            continue
        hold_ns = round(label.hold_s * 1e9)
        x = next((j for j in range(len(rows)) if stamps[j] >= stamps[e] + hold_ns), None)
        if x is None:
            continue
        buy_e, beyond = _loop_walk(asks[e], asz[e], label.clip)
        sell_e, beyond_sell = _loop_walk(bids[e], bsz[e], label.clip)
        if beyond or beyond_sell:
            continue
        sell_x, _ = _loop_walk(bids[x], bsz[x], label.clip)
        buy_x, _ = _loop_walk(asks[x], asz[x], label.clip)
        mid = (bids[e][0] + asks[e][0]) / 2
        long_bp = (sell_x - buy_e) / mid * 1e4 - 2 * label.label_fee_bp
        short_bp = (sell_e - buy_x) / mid * 1e4 - 2 * label.label_fee_bp
        out[t] = 1.0 if long_bp > 0 else (-1.0 if short_bp > 0 else 0.0)
    return out


@pytest.mark.parametrize("fee", [0.0, 5.5])
def test_vectorised_label_equals_the_loop(rows, fee: float) -> None:
    label = WalkedTakerLabel(hold_s=2.0, clip=0.6, label_fee_bp=fee)
    np.testing.assert_array_equal(label(rows).to_numpy(), _loop_label(rows, label))


def test_snapshots_are_first_at_or_after_on_stamps() -> None:
    ms = 1_000_000
    stamps = np.array([0, 5, 10, 120, 2_000, 2_150], dtype=np.int64) * ms
    entry, exit_ = entry_exit_snapshots(stamps, hold_s=2.0)
    assert entry.tolist() == [2, 3, 3, 4, 5, 6]
    # entry 2 (10 ms) + 2 s -> 2010 ms -> row 5; entry 3 (120 ms) -> 2120 -> row 5.
    assert exit_.tolist() == [5, 5, 5, 6, 6, 6]


def test_forward_and_control_moves_equal_loops(rows) -> None:
    stamps = stamps_ns(rows)
    mid = ((rows["bid_price_0"] + rows["ask_price_0"]) / 2).to_numpy()
    forward = forward_move_bp(rows, 1.5).to_numpy()
    control = control_move_bp(rows, 1.5).to_numpy()
    for t in range(len(rows)):
        e = np.searchsorted(stamps, stamps[t] + 10_000_000)
        x = np.searchsorted(stamps, stamps[e] + 1_500_000_000) if e < len(rows) else len(rows)
        expected = (mid[x] / mid[e] - 1) * 1e4 if x < len(rows) else np.nan
        np.testing.assert_allclose(forward[t], expected, rtol=1e-12)
        later = np.searchsorted(stamps, stamps[t] + 1_500_000_000)
        if later >= len(rows) or t < 19:
            assert np.isnan(control[t])
            continue
        now_mean = mid[t - 19 : t + 1].mean()
        later_mean = mid[later - 19 : later + 1].mean()
        expected = math.log(later_mean / now_mean) * 1e4
        np.testing.assert_allclose(control[t], expected, rtol=1e-9, atol=1e-8)


def test_the_label_reads_no_further_than_its_exit_row(rows) -> None:
    label = WalkedTakerLabel(hold_s=1.0, clip=0.2)
    out = label.outcomes(rows)
    full = label(rows).to_numpy()
    cut = 350
    part = label(rows.iloc[:cut].copy()).to_numpy()
    decided = out["exit_index"].to_numpy()[:cut] < cut
    np.testing.assert_array_equal(full[:cut][decided], part[decided])
    assert np.isnan(part[~decided]).all()
    reach = out["exit_index"].to_numpy() - np.arange(len(rows))
    assert reach[decided.nonzero()[0]].max() <= label.purge_rows()
    assert purge_rows(1.0) == 13
    assert label.horizon_s == pytest.approx(1.01)


def test_a_fee_turns_marginal_trades_into_holds(rows) -> None:
    free = WalkedTakerLabel(hold_s=5.0, clip=0.01)(rows)
    costly = WalkedTakerLabel(hold_s=5.0, clip=0.01, label_fee_bp=5.5)(rows)
    assert (costly == 0).sum() > (free == 0).sum()
    assert set(free.dropna().unique()) <= {-1.0, 0.0, 1.0}


def test_an_entry_beyond_the_visible_book_is_unlabelled(rows) -> None:
    label = WalkedTakerLabel(hold_s=1.0, clip=50.0)
    out = label.outcomes(rows)
    assert out["entry_exceeds"].any()
    assert label(rows)[out["entry_exceeds"]].isna().all()
    assert walk_rows(rows, "ask", 50.0).exceeds.all()
