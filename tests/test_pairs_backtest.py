"""The pairs backtest: alignment refusals, no look-ahead, both legs charged, the summary."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_research.pairs.backtest import PairRule, backtest
from trading_research.pairs.costs import leg_costs
from trading_research.pairs.data import AlignmentError, align, load
from trading_research.pairs.stats import PairsError
from trading_research.synthetic.pairs import PairSpec, generate

A, B = "SYNTHAUSDT", "SYNTHBUSDT"


@pytest.fixture(scope="module")
def planted() -> dict[str, pd.DataFrame]:
    return generate(PairSpec(seed=3)).bars


@pytest.fixture(scope="module")
def bars(planted: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return align(planted, "1h")


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def test_align_puts_every_symbol_on_one_clock(bars: pd.DataFrame) -> None:
    assert list(bars["close"].columns) == [A, B]
    assert bars.index.is_unique and bars.index.is_monotonic_increasing
    assert (np.diff(bars.index.asi8) == pd.Timedelta(1, unit="h").value).all()


def test_a_gap_is_refused(planted: dict[str, pd.DataFrame]) -> None:
    gappy = {s: f.drop(index=100).reset_index(drop=True) for s, f in planted.items()}
    with pytest.raises(AlignmentError, match="gap"):
        align(gappy, "1h")


def test_misaligned_symbols_are_refused(planted: dict[str, pd.DataFrame]) -> None:
    one = {A: planted[A].drop(index=50).reset_index(drop=True), B: planted[B]}
    with pytest.raises(AlignmentError, match="not on the same bars"):
        align(one, "1h")


def test_duplicates_bad_prices_and_a_lone_symbol_are_refused(
    planted: dict[str, pd.DataFrame],
) -> None:
    doubled = pd.concat([planted[A].iloc[:5], planted[A].iloc[4:5]], ignore_index=True)
    with pytest.raises(AlignmentError, match="more than once"):
        align({A: doubled, B: planted[B]}, "1h")
    bad = planted[A].copy()
    bad.loc[10, "open"] = 0.0
    with pytest.raises(AlignmentError, match="non-positive"):
        align({A: bad, B: planted[B]}, "1h")
    with pytest.raises(AlignmentError, match="two symbols"):
        align({A: planted[A]}, "1h")


def test_load_reads_what_the_generator_writes(tmp_path: Path, bars: pd.DataFrame) -> None:
    generate(PairSpec(seed=3)).write(tmp_path)
    assert (tmp_path / A / "klines-1h" / "2020-01-06.parquet").exists()
    assert (tmp_path / "truth.json").exists()
    loaded = load(tmp_path, [A, B], "1h")
    pd.testing.assert_frame_equal(loaded, bars, check_freq=False)
    with pytest.raises(AlignmentError, match="no 1h bars"):
        load(tmp_path, [A, "NOPEUSDT"], "1h")
    with pytest.raises(AlignmentError, match="unknown bar interval"):
        load(tmp_path, [A, B], "hourly")


# ---------------------------------------------------------------------------
# no look-ahead
# ---------------------------------------------------------------------------


def _scramble(bars: pd.DataFrame, cut: int, seed: int = 0) -> pd.DataFrame:
    changed = bars.copy()
    rng = np.random.default_rng(seed)
    for field in ("open", "close"):
        for symbol in (A, B):
            values = changed[(field, symbol)].to_numpy().copy()
            values[cut:] *= np.exp(rng.normal(0.0, 0.05, len(values) - cut))
            changed[(field, symbol)] = values
    return changed


@pytest.mark.parametrize("cut", [900, 1500, 2400])
def test_perturbing_later_prices_changes_no_earlier_decision(bars: pd.DataFrame, cut: int) -> None:
    rule, costs = PairRule(), leg_costs()
    before = backtest(bars, A, B, rule, costs)
    after = backtest(_scramble(bars, cut), A, B, rule, costs)
    when = bars.index[cut]
    early = before.decisions[before.decisions["timestamp"] < when].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        early, after.decisions[after.decisions["timestamp"] < when].reset_index(drop=True)
    )
    assert len(early) > 0
    fitted_before = before.windows[before.windows["formation_end"] < when]
    pd.testing.assert_frame_equal(
        fitted_before.drop(columns=["trades"]),
        after.windows.loc[fitted_before.index].drop(columns=["trades"]),
    )
    # The decisions are not blind to the perturbation: later ones do move.
    late = before.decisions["timestamp"] >= when
    assert not np.allclose(before.decisions.loc[late, "z"], after.decisions.loc[late, "z"])


def test_fills_are_at_the_next_bars_open(bars: pd.DataFrame) -> None:
    result = backtest(bars, A, B, PairRule(), leg_costs())
    position = {stamp: k for k, stamp in enumerate(bars.index)}
    opens = bars[("open_time", "")]
    for trade in result.trades.itertuples():
        i = position[trade.entry_decision] + 1
        j = position[trade.exit_decision] + 1
        assert trade.entry_fill == opens.iloc[i] and trade.exit_fill == opens.iloc[j]
        ret_a = (bars[("open", A)].iloc[j] / bars[("open", A)].iloc[i] - 1) * 1e4
        ret_b = (bars[("open", B)].iloc[j] / bars[("open", B)].iloc[i] - 1) * 1e4
        assert trade.gross_bp == pytest.approx(trade.side * (ret_a - trade.beta * ret_b))


# ---------------------------------------------------------------------------
# costs
# ---------------------------------------------------------------------------


def test_both_legs_pay_each_fill(bars: pd.DataFrame) -> None:
    costs = leg_costs(fee_bp=4.0, slippage_bp=1.0)
    trades = backtest(bars, A, B, PairRule(), costs).trades
    assert len(trades) > 10
    assert np.allclose(trades["cost_a_bp"], 2 * 5.0)
    assert np.allclose(trades["cost_b_bp"], 2 * 5.0 * trades["beta"].abs())
    assert np.allclose(trades["cost_bp"], trades["cost_a_bp"] + trades["cost_b_bp"])
    assert np.allclose(trades["net_bp"], trades["gross_bp"] - trades["cost_bp"])
    assert np.allclose(trades["cost_bp"], [costs.round_trip_bp(b) for b in trades["beta"]])


def test_a_fee_changes_the_net_and_nothing_else(bars: pd.DataFrame) -> None:
    free = backtest(bars, A, B, PairRule(), leg_costs(fee_bp=0.0, slippage_bp=0.0)).trades
    paid = backtest(bars, A, B, PairRule(), leg_costs(fee_bp=7.0, slippage_bp=0.0)).trades
    assert np.allclose(free["net_bp"], free["gross_bp"])
    assert np.allclose(free["gross_bp"], paid["gross_bp"])
    assert np.allclose(free["net_bp"] - paid["net_bp"], 14.0 * (1 + paid["beta"].abs()))


def test_fees_come_from_the_repositorys_tables() -> None:
    assert leg_costs().per_fill_bp == pytest.approx(5.0 + 0.5)
    assert "BINANCE_UM_TAKER_BP" in leg_costs().source
    vip1 = leg_costs(tier="vip1", slippage_bp=0.0)
    assert vip1.per_fill_bp == pytest.approx(4.0) and "vip1" in vip1.source
    with pytest.raises(KeyError):
        leg_costs(tier="no_such_tier")
    with pytest.raises(ValueError, match="not both"):
        leg_costs(fee_bp=1.0, tier="vip1")


# ---------------------------------------------------------------------------
# the rule and the summary
# ---------------------------------------------------------------------------


def test_trades_stay_inside_their_window_and_respect_the_hold(bars: pd.DataFrame) -> None:
    rule = PairRule(max_hold=5)
    result = backtest(bars, A, B, rule, leg_costs())
    trades, windows = result.trades, result.windows.set_index("window")
    assert set(trades["reason"]) <= {"exit", "stop", "time", "window_end"}
    assert (trades["bars_held"] <= rule.max_hold).all() and "time" in set(trades["reason"])
    for trade in trades.itertuples():
        w = windows.loc[trade.window]
        assert w["trading_start"] <= trade.entry_decision < trade.exit_decision
        assert trade.exit_decision <= w["trading_end"]


def test_the_gate_skips_windows_whose_formation_does_not_cointegrate() -> None:
    walks = align(generate(PairSpec(seed=4, variant="independent")).bars, "1h")
    gated = backtest(walks, A, B, PairRule(gate_p=0.01), leg_costs())
    skipped = gated.windows.loc[~gated.windows["traded"], "window"]
    assert len(skipped) > 0
    assert not gated.trades["window"].isin(skipped).any()


def test_the_rule_refuses_bad_settings(bars: pd.DataFrame) -> None:
    with pytest.raises(PairsError, match="bands"):
        PairRule(entry_z=1.0, exit_z=1.5)
    with pytest.raises(PairsError, match="formation"):
        PairRule(formation=50)
    with pytest.raises(PairsError, match="cannot hold"):
        backtest(bars.iloc[:300], A, B, PairRule(formation=400), leg_costs())
    with pytest.raises(PairsError, match="no bars"):
        backtest(bars, A, "NOPE", PairRule(), leg_costs())


def test_the_summary_states_day_t_and_the_cost_at_which_net_is_zero(bars: pd.DataFrame) -> None:
    result = backtest(bars, A, B, PairRule(), leg_costs())
    s, trades = result.summary, result.trades
    assert s["trades"] == len(trades) > 0
    assert s["gross_per_trade_bp"] > 0 and s["gross_day_t"] > 3
    assert s["net_per_trade_bp"] == pytest.approx(trades["net_bp"].mean())
    assert s["cost_to_break_even_bp_per_trade"] == pytest.approx(trades["gross_bp"].mean())
    per_fill = trades["gross_bp"].sum() / (2 * (1 + trades["beta"].abs())).sum()
    assert s["cost_to_break_even_bp_per_fill"] == pytest.approx(per_fill)
    at_break_even = backtest(
        bars, A, B, PairRule(), leg_costs(fee_bp=per_fill, slippage_bp=0.0)
    ).trades
    assert at_break_even["net_bp"].sum() == pytest.approx(0.0, abs=1e-6)
    assert 0.0 <= s["net_days_positive_share"] <= 1.0
    days = result.days
    assert days["trades"].sum() == len(trades)
    assert days["net_bp"].sum() == pytest.approx(trades["net_bp"].sum())


def test_independent_walks_earn_nothing_significant() -> None:
    walks = align(generate(PairSpec(seed=9, variant="independent")).bars, "1h")
    s = backtest(walks, A, B, PairRule(), leg_costs()).summary
    assert not s["net_day_t"] >= 3
    assert math.isfinite(s["net_per_trade_bp"])
