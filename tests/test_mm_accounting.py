"""Money: average cost, fees, funding, and the identity that ties them to cash."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from trading_research.backtest.costs import (
    BYBIT_BASE,
    BYBIT_LINEAR_TIERS,
    FeeTier,
    breakeven_maker_bp,
    fee_tier,
)
from trading_research.market_making import accounting
from trading_research.market_making.accounting import Account, IdentityError
from trading_research.market_making.quoters import MarketView, Quote, Quotes, TouchQuoter
from trading_research.market_making.simulator import SimConfig, simulate_day
from trading_research.market_making.synthetic import (
    hand_built_market,
    known_answer_market,
    random_market,
)

CONFIG = SimConfig(clip_notional=1e12, warmup_s=0.0)
#: A clip of two on the random markets (mid near 100), whatever their touch.
FUZZ = SimConfig(clip_notional=250.0, clip_touch_share=1e9, warmup_s=0.0)


def test_pnl_identity_holds_over_random_fills() -> None:
    """500 fills with flips through zero, maker and taker fees, rebates and
    funding: the identity holds after every one, at any mark."""
    rng = np.random.default_rng(3)
    account = Account()
    for step in range(500):
        side = int(rng.choice([-1, 1]))
        price = float(rng.uniform(90, 110))
        size = float(rng.integers(1, 9))
        fee = float(rng.choice([2.0, 5.5, -0.5, 0.0]))
        account.fill(side, price, size, fee, maker=bool(rng.random() < 0.7))
        if step % 25 == 0:
            account.settle_funding(float(rng.normal(0, 3e-4)), float(rng.uniform(90, 110)))
        for mark in (price, float(rng.uniform(50, 150))):
            account.check_identity(mark)
    assert account.maker_turnover > 0 and account.taker_turnover > 0
    assert account.funding != 0.0


def test_a_broken_ledger_raises() -> None:
    account = Account()
    account.fill(1, 100.0, 2.0, 2.0, maker=True)
    account.check_identity(101.0)
    account.cash += 1.0
    with pytest.raises(IdentityError):
        account.check_identity(101.0)


def test_the_simulator_stops_at_the_event_that_breaks_the_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = accounting.Account.fill

    def leaky(self: Account, *args: object, **kwargs: object) -> float:
        fee = original(self, *args, **kwargs)  # type: ignore[arg-type]
        self.cash += 0.01
        return fee

    monkeypatch.setattr(accounting.Account, "fill", leaky)
    with pytest.raises(IdentityError):
        simulate_day(random_market(0), TouchQuoter(), FUZZ)


def test_a_flip_through_zero_realises_the_closed_part() -> None:
    account = Account()
    account.fill(1, 100.0, 2.0, 0.0, maker=True)
    account.fill(-1, 110.0, 5.0, 0.0, maker=True)
    assert account.realised == pytest.approx(20.0)
    assert account.position == -3.0
    assert account.avg_price == 110.0
    account.fill(1, 105.0, 3.0, 0.0, maker=True)
    assert account.realised == pytest.approx(35.0)
    assert account.position == 0.0
    assert account.cash == pytest.approx(35.0)


def test_maker_rebate_is_a_negative_fee() -> None:
    account = Account()
    fee = account.fill(1, 100.0, 1.0, -0.5, maker=True)
    assert fee == pytest.approx(-0.005)
    assert account.fees == pytest.approx(-0.005)
    assert account.cash == pytest.approx(-100.0 + 0.005)

    market = known_answer_market(move_ticks=0, n_prints=40)
    rebate = CONFIG.with_(fees=FeeTier("rebate", -0.5, 5.5))
    fills = simulate_day(market, TouchQuoter(), rebate).fills
    assert (fills.loc[fills["maker"], "fee"] < 0).all()
    assert (fills.loc[~fills["maker"], "fee"] > 0).all()


def test_funding_sign_longs_pay_when_the_rate_is_positive() -> None:
    long, short = Account(), Account()
    long.fill(1, 100.0, 2.0, 0.0, maker=True)
    short.fill(-1, 100.0, 2.0, 0.0, maker=True)
    assert long.settle_funding(1e-4, 100.0) == pytest.approx(-0.02)
    assert short.settle_funding(1e-4, 100.0) == pytest.approx(0.02)
    assert long.settle_funding(-1e-4, 100.0) == pytest.approx(0.02)
    long.check_identity(100.0)
    short.check_identity(100.0)


def test_flat_at_settlement_pays_nothing() -> None:
    assert Account().settle_funding(0.01, 100.0) == 0.0
    assert Account().settle_funding(0.01, float("nan")) == 0.0


@dataclass(frozen=True)
class BidOnce:
    """Bids at 100 until the first second has passed, then quotes nothing."""

    start: int
    name: str = "bid-once"
    uses_future: bool = False

    def quotes(self, view: MarketView, position: float) -> Quotes:
        bid = 100 if view.ts - self.start < 1_000_000_000 and position == 0 else None
        return Quotes(Quote(bid, view.clip), Quote(None, 0.0))


def test_funding_is_applied_at_settlement_boundaries() -> None:
    """Flat at 00:00, long one unit at 08:00: only the second settlement pays,
    and it pays on the position and mark of that instant."""
    book = ([(100 - i, 10.0) for i in range(3)], [(102 + i, 10.0) for i in range(3)])
    market = hand_built_market(
        [(500.0, *book), (1_000.0, *book), (9 * 3_600_000.0, *book)],
        [(600.0, 99, 1.0, -1)],
        funding=[(0.0, 5e-4), (8 * 3_600_000.0, 3e-4), (16 * 3_600_000.0, -2e-4)],
    )
    result = simulate_day(market, BidOnce(market.day_start_ns), CONFIG)
    mid = (100 + 102) / 2 * 0.01
    assert result.counters["funding_settlements"] == 3
    assert result.counters["funding_settlements_with_position"] == 2
    assert result.decomposition["funding"] == pytest.approx(-1.0 * mid * (3e-4 - 2e-4))
    eight = result.equity[result.equity["ts"] == market.day_start_ns + 8 * 3600 * 10**9]
    assert eight["funding"].iloc[0] == 0.0  # recorded at 08:00, before the settlement
    after = result.equity[result.equity["ts"] == market.day_start_ns + (8 * 3600 + 60) * 10**9]
    assert after["funding"].iloc[0] == pytest.approx(-mid * 3e-4)


def test_decomposition_sums_to_net() -> None:
    for seed in range(20):
        market = random_market(seed, n_snapshots=400, prints_per_snapshot=1.5, funding=True)
        result = simulate_day(market, TouchQuoter(), FUZZ)
        assert result.counters["fills"] > 20
        parts = result.decomposition
        assembled = (
            parts["spread"] + parts["adverse"] + parts["inventory"] - parts["fees"]
        ) + parts["funding"]
        assert assembled == pytest.approx(parts["net"], abs=1e-9)
        assert parts["net"] == pytest.approx(result.equity["equity"].iloc[-1], abs=1e-9)
        assert parts["making"] == pytest.approx(parts["spread"] + parts["adverse"] - parts["fees"])


def test_fee_breakeven_matches_a_rerun_at_that_fee() -> None:
    """The touch quoter's decisions do not read the fee, so the break-even from
    one run is exact: re-run at it and the net is zero."""
    market = random_market(11, n_snapshots=600, prints_per_snapshot=1.5)
    base = simulate_day(market, TouchQuoter(), FUZZ)
    turnover = base.counters["maker_turnover"]
    fee = breakeven_maker_bp(base.decomposition["net"], turnover, BYBIT_BASE.maker_bp)
    again = simulate_day(market, TouchQuoter(), FUZZ.with_(fees=FeeTier("even", fee, 5.5)))
    assert again.decomposition["net"] == pytest.approx(0.0, abs=1e-9 * turnover)
    assert again.counters["maker_turnover"] == pytest.approx(turnover)


def test_fee_tiers_are_looked_up_not_assumed() -> None:
    assert fee_tier("base") is BYBIT_BASE
    assert BYBIT_BASE.maker_bp == 2.0 and BYBIT_BASE.taker_bp == 5.5
    assert all(t.maker_bp <= t.taker_bp for t in BYBIT_LINEAR_TIERS)
    with pytest.raises(KeyError, match="no fee tier"):
        fee_tier("vip9")
    with pytest.raises(ValueError):
        FeeTier("upside-down", 6.0, 5.5)
    with pytest.raises(ValueError):
        breakeven_maker_bp(1.0, 0.0, 2.0)


def test_funding_is_charged_on_the_position_before_a_print_at_the_same_instant() -> None:
    """Long one unit at 08:00 exactly, when a sale fills a second bid at the same
    instant: funding comes first, on one unit, and a positive rate makes the
    long pay."""
    book = ([(100 - i, 10.0) for i in range(3)], [(102 + i, 10.0) for i in range(3)])
    eight = 8 * 3_600_000.0

    @dataclass(frozen=True)
    class BidAlways:
        name: str = "bid-always"
        uses_future: bool = False

        def quotes(self, view: MarketView, position: float) -> Quotes:
            return Quotes(Quote(100, 1.0), Quote(None, 0.0))

    market = hand_built_market(
        [(500.0, *book), (1_000.0, *book)]
        + [(eight - 2_000.0 + 100.0 * i, *book) for i in range(25)],
        [(600.0, 99, 1.0, -1), (eight, 99, 1.0, -1)],
        funding=[(eight, 1e-4)],
    )
    config = SimConfig(
        clip_notional=1e12, clip_touch_share=1e9, warmup_s=0.0, suspend_after_pause_s=1e6
    )
    result = simulate_day(market, BidAlways(), config)
    assert result.decomposition["funding"] == pytest.approx(-1.0 * 1.01 * 1e-4)


def test_fees_and_net_recomputed_from_the_fills_agree() -> None:
    """Every fee is the maker or taker rate on the fill's notional; the net is
    the fills' cash plus funding; the fills alone leave the day flat; and a
    taker always crosses the mid of the book it walked."""
    for seed in range(30):
        market = random_market(seed, n_snapshots=400, prints_per_snapshot=1.5, funding=True)
        for maker_bp in (2.0, -0.5):
            config = FUZZ.with_(soft_limit_clips=2.0, fees=FeeTier("t", maker_bp, 5.5))
            result = simulate_day(market, TouchQuoter(), config)
            fills = result.fills
            notional = fills["price"] * fills["size"]
            expected = np.where(fills["maker"], maker_bp, 5.5) * 1e-4 * notional
            np.testing.assert_allclose(fills["fee"], expected, rtol=1e-12)
            cash = float((-fills["side"] * notional).sum() - fills["fee"].sum())
            assert result.decomposition["net"] == pytest.approx(
                cash + result.decomposition["funding"], abs=1e-9
            )
            assert abs((fills["side"] * fills["size"]).sum()) < 1e-9
            taker = fills[~fills["maker"]]
            assert (taker["side"] * (taker["price"] - taker["mid_ref"]) > 0).all()
