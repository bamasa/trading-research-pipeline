"""The spot account: inventory shorts, the inventory's own P&L, K-inv both ways, K-cap, the day's bp."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from _engine_books import decisions as book_decisions
from _engine_books import make_book
from test_spot_prints_fixture import CLIP, grid_of, make_prints
from trading_research.backtest.taker import TakerBook
from trading_research.market_making.accounting import IdentityError
from trading_research.market_making.events import NS_PER_S
from trading_research.spot import account as acc
from trading_research.spot.account import (
    CapError,
    Fills,
    InventoryError,
    account_day,
    book_fills,
    day_summary,
    first_last_print,
    long_only,
    print_fills,
    with_extra_fees,
)
from trading_research.spot.costs import book_settings, execute_book_day, load_fee_table
from trading_research.spot.print_taker import PrintDay, execute_print_day, print_settings

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "spot_2026.yaml"
TABLE = load_fee_table(CONFIG)


def print_day(seed: int = 0) -> tuple[PrintDay, tuple[float, float], np.ndarray, np.ndarray]:
    prints = make_prints(seed)
    rows = grid_of(prints)
    rng = np.random.default_rng(seed)
    pick = rows[np.sort(rng.choice(len(rows), 3000, replace=False))]
    side = rng.choice(np.array([-1, 1]), len(pick))
    settings = print_settings(TABLE, "binance_spot_BTCFDUSD")
    day = execute_print_day(prints, rows, pick, side, 50 * NS_PER_S, settings)
    return day, first_last_print(prints), pick, side


def test_a_print_day_holds_on_the_spot_account_both_ways() -> None:
    day, (first, last), _, _ = print_day()
    fills = print_fills(day)
    assert len(fills.ts) == 2 * len(day.trades) and np.all(np.diff(fills.ts) >= 0)
    account = account_day(fills, day.net, first, last, CLIP)
    assert account.void == () and not account.k_inv
    account.require()
    assert account.inventory_pnl == CLIP * (last - first)
    assert account.inventory_pnl_bp == (last - first) / first * 1e4
    quote, received = account.checks
    assert (quote.base_start, received.base_start) == (CLIP, 2 * CLIP)
    assert quote.quote_start == received.quote_start == 3 * CLIP * first
    assert quote.min_base == 0.0 and quote.base_end == CLIP  # every short sold the held clip
    buys = fills.side > 0
    assert received.base_fees == pytest.approx((fills.size * fills.fee_bp * 1e-4)[buys].sum())
    assert received.base_end == pytest.approx(2 * CLIP - received.base_fees, abs=1e-15)
    assert quote.quote_end - quote.quote_start == pytest.approx(day.net, abs=1e-9)
    for base, cash in account.residuals():
        assert abs(base) < 1e-15 and abs(cash) < 1e-9


def test_a_book_day_holds_on_the_spot_account() -> None:
    b = make_book(2, day=date(2026, 5, 4))
    book = TakerBook(b.day, b.ts, b.bid_px, b.bid_sz, b.ask_px, b.ask_sz)
    rows, side = book_decisions(book, 500, seed=2)
    settings = book_settings(TABLE, "bybit_spot_BTCUSDT", book.day)
    day = execute_book_day(book, book.ts[rows], side, 20 * NS_PER_S, settings, TABLE)
    fills = book_fills(day, settings)
    assert (fills.fee_bp == 10.0).all() and len(fills.ts) == 2 * len(day.trades)
    mid = book.mid
    account = account_day(fills, day.net, float(mid[0]), float(mid[-1]), CLIP)
    assert account.void == ()


def fills_of(rows: list[tuple[int, float]]) -> Fills:
    """Fills of one clip from ``(side, price)``, a second apart, at 10 bp."""
    n = len(rows)
    return Fills(np.arange(n, dtype=np.int64) * NS_PER_S, np.array([r[0] for r in rows]),
                 np.array([r[1] for r in rows]), np.full(n, CLIP), np.full(n, 10.0))  # fmt: skip


def net_of(fills: Fills) -> float:
    notional = fills.price * fills.size
    return float((-fills.side * notional).sum() - (notional * fills.fee_bp * 1e-4).sum())


def test_two_shorts_at_once_void_the_slot() -> None:
    fills = fills_of([(-1, 100.0), (-1, 100.0), (1, 99.0), (1, 99.0)])
    account = account_day(fills, net_of(fills), 100.0, 99.0, CLIP)
    assert account.k_cap and account.k_inv and not account.k_acct
    quote, received = account.checks
    assert quote.k_inv and quote.first_negative == 1 and not received.k_inv
    with pytest.raises(CapError):
        account.require()
    assert account.void == ("K-cap", "K-inv")


def test_a_long_the_quote_cannot_pay_for_fires_k_inv() -> None:
    fills = fills_of([(1, 310.0), (-1, 311.0)])  # quote worth three clips at 100
    account = account_day(fills, net_of(fills), 100.0, 311.0, CLIP)
    assert account.k_inv and not account.k_cap
    with pytest.raises(InventoryError, match="quote booking"):
        account.require()


def test_a_net_the_account_cannot_explain_fires_k_acct() -> None:
    fills = fills_of([(1, 100.0), (-1, 101.0)])
    account = account_day(fills, net_of(fills) + 0.01, 100.0, 101.0, CLIP)
    assert account.k_acct and account.void == ("K-acct",)
    with pytest.raises(IdentityError):
        account.require()


def test_long_only_drops_shorts_before_the_one_position_rule() -> None:
    prints = make_prints(1)
    rows = grid_of(prints)
    rng = np.random.default_rng(1)
    pick = rows[np.sort(rng.choice(len(rows), 3000, replace=False))]
    side = rng.choice(np.array([-1, 1]), len(pick))
    settings = print_settings(TABLE, "binance_spot_BTCFDUSD")
    slot = execute_print_day(prints, rows, pick, side, 50 * NS_PER_S, settings)
    ts, s = long_only(pick, side)
    assert (s == 1).all() and len(ts) == (side > 0).sum()
    longs = execute_print_day(prints, rows, ts, s, 50 * NS_PER_S, settings)
    taken_long = set(slot.trades.loc[slot.trades["side"] > 0, "decision_ns"])
    assert set(longs.trades["decision_ns"]) - taken_long  # longs the slot skipped behind a short
    assert (longs.trades["side"] == 1).all()


def test_the_days_money_in_bp() -> None:
    day, _, _, _ = print_day(3)
    t = day.trades
    summary = day_summary(t)
    assert summary.trades == len(t) == summary.trades_long + summary.trades_short
    assert summary.net_bp == pytest.approx(day.day_bp("net"), rel=1e-12)
    assert summary.gross_bp == pytest.approx(day.day_bp("gross"), rel=1e-12)
    assert summary.turnover == pytest.approx(day.turnover, rel=1e-12)
    assert summary.gross_bp_turnover == pytest.approx(t["gross"].sum() / day.turnover * 1e4)
    long = t[t["side"] > 0]
    assert summary.net_bp_long == pytest.approx(long["net"].sum() / long["notional"].sum() * 1e4)
    assert summary.net_per_100_clip == pytest.approx((t["net"] / t["notional"] * 100).sum())
    assert summary.fees == pytest.approx(summary.gross - summary.net, rel=1e-9)
    # the fee break-even per side: the gross over turnover equals the fee that zeroes the net
    zero = summary.gross - summary.gross_bp_turnover * 1e-4 * summary.turnover
    assert zero == pytest.approx(0.0, abs=1e-9)
    empty = day_summary(t.iloc[:0])
    assert empty.trades == 0 and np.isnan(empty.net_bp) and empty.net_per_100_clip == 0.0
    shorts = day_summary(t[t["side"] < 0])
    assert np.isnan(shorts.net_bp_long) and shorts.trades_long == 0


def test_fills_are_refused_out_of_order() -> None:
    with pytest.raises(ValueError, match="time order"):
        Fills(np.array([2, 1]), np.array([1, -1]), np.ones(2), np.ones(2), np.ones(2))
    with pytest.raises(ValueError):
        Fills(np.array([1, 2]), np.array([1, 0]), np.ones(2), np.ones(2), np.ones(2))


def test_k_fills_extra_charge_falls_on_each_fills_side() -> None:
    day, _, _, _ = print_day(4)
    t = day.trades
    charged = with_extra_fees(t, 1.5, -2.0)  # a negative advantage charges nothing
    entry, exit_ = t["entry_price"] * t["size"], t["exit_price"] * t["size"]
    want = np.where(t["side"] > 0, entry * 1.5e-4, exit_ * 1.5e-4)
    np.testing.assert_allclose(charged["extra"], want, rtol=1e-12)
    np.testing.assert_allclose(charged["net"], t["net"] - want, rtol=1e-12, atol=1e-15)
    assert day_summary(charged).net_bp < day_summary(t).net_bp
    assert (with_extra_fees(t, 0.0, 0.0)["net"] == t["net"]).all()


def _round_trips(sides: np.ndarray, entry: np.ndarray, exit_: np.ndarray, fee_bp: float = 10.0):
    """One clip in and out per side, in time order, and the trips' net in quote."""
    n = len(sides)
    side = np.empty(2 * n, dtype=np.int64)
    side[0::2], side[1::2] = sides, -sides
    price = np.empty(2 * n)
    price[0::2], price[1::2] = entry, exit_
    fills = acc.Fills(np.arange(2 * n, dtype=np.int64), side, price, np.full(2 * n, 0.010),
                      np.full(2 * n, fee_bp))  # fmt: skip
    net = sides * (exit_ - entry) * 0.010 - (entry + exit_) * 0.010 * fee_bp * 1e-4
    return fills, float(net.sum())


@pytest.mark.parametrize("trips", [1_001, 1_726, 4_312])
def test_k_inv_never_fires_on_an_honest_heavy_day(trips: int) -> None:
    """Fees alone exhaust three clips of quote after about 1,000 round trips at 10 bp;
    they are a separate ledger, so the largest feasible day (20 s holds from 00:01 to
    23:58) fires nothing."""
    px = np.full(trips, 60_000.0)
    fills, net = _round_trips(np.ones(trips, dtype=np.int64), px, px)
    day = acc.account_day(fills, net, 60_000.0, 60_000.0)
    assert day.void == ()


def test_k_inv_never_fires_on_a_losing_day_or_after_many_longs_then_a_short() -> None:
    entry = np.full(850, 60_000.0)
    fills, net = _round_trips(np.ones(850, dtype=np.int64), entry, entry * (1 - 5e-4))
    assert acc.account_day(fills, net, 60_000.0, 60_000.0).void == ()
    sides = np.r_[np.ones(1_001, dtype=np.int64), -1]
    px = np.full(len(sides), 60_000.0)
    fills, net = _round_trips(sides, px, px)
    assert acc.account_day(fills, net, 60_000.0, 60_000.0).void == ()
