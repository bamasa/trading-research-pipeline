"""The registered short side: shorts sell held inventory, and the account is checked both ways.

As registered under "The short side": each day starts holding one clip of BTC
and quote worth three clips at the day's first print. A long buys a clip with
quote and sells it at its exit; a short sells the held clip and buys it back.
A round trip's P&L, fees and turnover are those of a long or a short of one
clip, so the executors' trades are the slot's trades unchanged, and nothing
here changes a trade.

What this module adds, from the executors' fills:

* **The inventory's own P&L**: the held clip marked from the day's first print
  to its last. A separate line, never inside ``net_bp_trade``, a kill or a
  status.
* **K-inv**: either coin's balance below zero after any fill, replayed twice:
  ``quote``, every fee booked in quote (the registered convention); and
  ``received``, each fee deducted in the asset received (a buy's in BTC at
  ``size x fee_bp x 1e-4``, a sell's in quote), the day starting with two clips
  of BTC. With these balances it fires only through an accounting fault.
* **K-cap**: the traded position beyond one clip either way after any fill.
* **K-acct** on the spot account: at the day's end the BTC balance is the
  start's (less, when fees are deducted in BTC, those fees) and the quote's
  change is the trades' net (plus, then, those fees' value at their fills).
* **The long-only variant**: short decisions dropped before the one-position
  rule (:func:`long_only`), so its trades differ from the slot's.
* **K-fill's extra charge** (:func:`with_extra_fees`): on the worse-of-two
  bracket's trades, every buy also charged ``max(0, a_buy)`` and every sell
  ``max(0, a_sell)`` bp of its notional; the fills themselves do not change.
* **The day's money in bp** (:func:`day_summary`): ``net_bp_trade`` (the day's
  net over its filled entry notional), the same before fees, the gross over
  turnover (the fee break-even per side), the two sides apart, and net per 100
  quote of clip, all from the trades' float64 raw prices.

Borrowing to short is not modelled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading_research.backtest.taker import TakerDay, TakerSettings
from trading_research.market_making.accounting import IdentityError
from trading_research.spot.costs import fill_fee
from trading_research.spot.print_taker import DayPrints, PrintDay

BOOKINGS: tuple[str, ...] = ("quote", "received")
#: Clips of BTC held at the day's start, per booking.
BASE_CLIPS: dict[str, int] = {"quote": 1, "received": 2}
#: Quote held at the day's start, in clips at the day's first print.
QUOTE_CLIPS = 3


class InventoryError(AssertionError):
    """A balance below zero (K-inv)."""


class CapError(AssertionError):
    """A traded position beyond one clip (K-cap)."""


@dataclass(frozen=True)
class Fills:
    """A day's fills in time order: +1 bought, -1 sold, each at its fee in bp."""

    ts: np.ndarray
    side: np.ndarray
    price: np.ndarray
    size: np.ndarray
    fee_bp: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.ts)
        if any(len(a) != n for a in (self.side, self.price, self.size, self.fee_bp)):
            raise ValueError("one side, price, size and fee per fill")
        if n > 1 and np.any(np.diff(self.ts) < 0):
            raise ValueError("fills must be in time order")
        if not np.isin(self.side, (-1, 1)).all():
            raise ValueError("a fill's side is +1 (bought) or -1 (sold)")


def _interleave(entry: np.ndarray, exit_: np.ndarray) -> np.ndarray:
    out = np.empty(2 * len(entry), dtype=np.result_type(entry, exit_))
    out[0::2], out[1::2] = entry, exit_
    return out


def book_fills(day: TakerDay, settings: TakerSettings) -> Fills:
    """The book executor's fills: each entry at its walked snapshot, each exit at
    the later of its snapshot and its arrival, both at the settings' taker fee."""
    t = day.trades
    side = t["side"].to_numpy(dtype=np.int64)
    exit_ts = np.maximum(
        t["exit_snapshot_ns"].to_numpy(dtype=np.int64),
        t["exit_arrival_ns"].to_numpy(dtype=np.int64),
    )
    return Fills(
        _interleave(t["entry_snapshot_ns"].to_numpy(dtype=np.int64), exit_ts),
        _interleave(side, -side),
        _interleave(
            t["entry_price"].to_numpy(dtype=np.float64), t["exit_price"].to_numpy(dtype=np.float64)
        ),
        _interleave(t["size"].to_numpy(dtype=np.float64), t["size"].to_numpy(dtype=np.float64)),
        np.full(2 * len(t), settings.fees.taker_bp),
    )


def print_fills(day: PrintDay) -> Fills:
    """The print executor's fills, each at the fee its instant paid."""
    t = day.trades
    side = t["side"].to_numpy(dtype=np.int64)
    return Fills(
        _interleave(
            t["entry_print_ns"].to_numpy(dtype=np.int64), t["exit_fill_ns"].to_numpy(dtype=np.int64)
        ),
        _interleave(side, -side),
        _interleave(
            t["entry_price"].to_numpy(dtype=np.float64), t["exit_price"].to_numpy(dtype=np.float64)
        ),
        _interleave(t["size"].to_numpy(dtype=np.float64), t["size"].to_numpy(dtype=np.float64)),
        _interleave(
            t["entry_fee_bp"].to_numpy(dtype=np.float64),
            t["exit_fee_bp"].to_numpy(dtype=np.float64),
        ),
    )


def first_last_print(prints: DayPrints) -> tuple[float, float]:
    """The day's first and last print prices, in archive order: the inventory's marks."""
    if not len(prints.ts):
        raise ValueError("a day without a print has no first print to mark the inventory at")
    return float(prints.price[0]), float(prints.price[-1])


@dataclass(frozen=True)
class InventoryCheck:
    """One booking's replay of the day's fills on the spot account."""

    booking: str
    base_start: float
    quote_start: float
    base_end: float
    quote_end: float
    min_base: float
    min_quote: float
    #: The fees deducted in BTC, in BTC and valued at their fills' prices (0 in quote).
    base_fees: float
    base_fee_value: float
    #: The first fill after which a balance was below zero, or -1.
    first_negative: int

    @property
    def k_inv(self) -> bool:
        return self.first_negative >= 0


def replay(fills: Fills, booking: str, first_price: float, clip: float) -> InventoryCheck:
    """Replay ``fills`` on an account starting with the registered balances."""
    if booking not in BOOKINGS:
        raise ValueError(f"booking is one of {BOOKINGS}, got {booking!r}")
    base0, quote0 = BASE_CLIPS[booking] * clip, QUOTE_CLIPS * clip * first_price
    notional = fills.price * fills.size
    fee = notional * fills.fee_bp * 1e-4
    buy = fills.side > 0
    in_base = buy & (booking == "received")
    base_step = fills.side * fills.size - np.where(in_base, fills.size * fills.fee_bp * 1e-4, 0.0)
    quote_step = -fills.side * notional - np.where(in_base, 0.0, fee)
    base = base0 + np.cumsum(base_step)
    quote = quote0 + np.cumsum(quote_step)
    negative = (base < -1e-12 * clip) | (quote < -1e-9 * max(quote0, 1.0))
    return InventoryCheck(
        booking,
        base0,
        quote0,
        float(base[-1]) if len(base) else base0,
        float(quote[-1]) if len(quote) else quote0,
        float(min(base.min(), base0)) if len(base) else base0,
        float(min(quote.min(), quote0)) if len(quote) else quote0,
        float((fills.size * fills.fee_bp * 1e-4)[in_base].sum()),
        float(fee[in_base].sum()),
        int(np.flatnonzero(negative)[0]) if negative.any() else -1,
    )


@dataclass(frozen=True)
class AccountDay:
    """The spot account's day: the inventory's own P&L and the void checks."""

    clip: float
    first_price: float
    last_price: float
    checks: tuple[InventoryCheck, ...]
    max_abs_position: float
    #: The trades' net in quote, which the account's quote change must explain.
    net: float

    @property
    def inventory_pnl(self) -> float:
        """The held clip marked from the first print to the last, in quote."""
        return self.clip * (self.last_price - self.first_price)

    @property
    def inventory_pnl_bp(self) -> float:
        """The same in bp of the held clip at the first print."""
        return (self.last_price - self.first_price) / self.first_price * 1e4

    @property
    def k_inv(self) -> bool:
        return any(c.k_inv for c in self.checks)

    @property
    def k_cap(self) -> bool:
        return self.max_abs_position > self.clip * (1.0 + 1e-9)

    def residuals(self) -> tuple[tuple[float, float], ...]:
        """Per booking: the BTC balance's end less its start less the fees taken in BTC,
        and the quote's change less those fees' value less the trades' net."""
        return tuple(
            (
                c.base_end - c.base_start + c.base_fees,
                c.quote_end - c.quote_start - c.base_fee_value - self.net,
            )
            for c in self.checks
        )

    @property
    def k_acct(self) -> bool:
        scale = max(1.0, max(c.quote_start for c in self.checks))
        return any(
            abs(base) > 1e-12 * self.clip or abs(quote) > 1e-9 * scale
            for base, quote in self.residuals()
        )

    @property
    def void(self) -> tuple[str, ...]:
        """The void conditions this account fires, in the registration's order."""
        fired = (("K-acct", self.k_acct), ("K-cap", self.k_cap), ("K-inv", self.k_inv))
        return tuple(name for name, hit in fired if hit)

    def require(self) -> None:
        """Raise on the first void condition, as the executors raise on their identity."""
        if self.k_acct:
            raise IdentityError(f"the spot account does not explain the trades: {self.residuals()}")
        if self.k_cap:
            raise CapError(f"a position of {self.max_abs_position!r} beyond the clip {self.clip!r}")
        if self.k_inv:
            bad = next(c for c in self.checks if c.k_inv)
            raise InventoryError(
                f"{bad.booking} booking: a balance below zero after fill {bad.first_negative}"
            )


def account_day(
    fills: Fills, net: float, first_price: float, last_price: float, clip: float = 0.010
) -> AccountDay:
    """The day's spot account from its fills, the trades' net in quote and the day's
    first and last print prices."""
    if not (first_price > 0 and last_price > 0):
        raise ValueError("the inventory is marked at raw print prices: positive")
    position = np.cumsum(fills.side * fills.size)
    biggest = float(np.abs(position).max()) if len(position) else 0.0
    checks = tuple(replay(fills, booking, first_price, clip) for booking in BOOKINGS)
    return AccountDay(
        float(clip), float(first_price), float(last_price), checks, biggest, float(net)
    )


def long_only(decision: np.ndarray, side: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The long-only variant's decisions: every short dropped before the one-position
    rule, so a long the slot skipped as busy behind a short may be taken here."""
    side = np.asarray(side)
    keep = side > 0
    return np.asarray(decision)[keep], side[keep]


@dataclass(frozen=True)
class DaySummary:
    """One day's money from a day's trades, in quote and in bp of float64 raw prices.

    ``net_bp`` is ``net_bp_trade``'s daily value: the day's net over its filled entry
    notional, x 1e4; ``gross_bp`` the same before fees; ``gross_bp_turnover`` the
    gross over both sides' notional, the fee break-even per side. The by-side values
    are the same ratio over the day's longs or shorts. A day or side without a trade
    has NaN ratios, no value in a daily series.
    """

    trades: int
    trades_long: int
    trades_short: int
    net: float
    gross: float
    fees: float
    entry_notional: float
    turnover: float
    net_bp: float
    gross_bp: float
    gross_bp_turnover: float
    net_bp_long: float
    net_bp_short: float
    #: Each trade's net over its own entry notional x 100, summed: per 100 quote of clip.
    net_per_100_clip: float


def _ratio(top: float, bottom: float) -> float:
    return top / bottom * 1e4 if bottom > 0 else float("nan")


def day_summary(trades: pd.DataFrame) -> DaySummary:
    """The day's summary from either executor's trades (``side``, ``size``,
    ``entry_price``, ``exit_price``, ``notional``, ``gross``, ``fees``, ``net``)."""
    side = trades["side"].to_numpy(dtype=np.int64)
    notional = trades["notional"].to_numpy(dtype=np.float64)
    net = trades["net"].to_numpy(dtype=np.float64)
    gross = trades["gross"].to_numpy(dtype=np.float64)
    size = trades["size"].to_numpy(dtype=np.float64)
    turnover = float(((trades["entry_price"] + trades["exit_price"]).to_numpy() * size).sum())
    long, short = side > 0, side < 0
    return DaySummary(
        len(trades),
        int(long.sum()),
        int(short.sum()),
        float(net.sum()),
        float(gross.sum()),
        float(trades["fees"].sum()),
        float(notional.sum()),
        turnover,
        _ratio(float(net.sum()), float(notional.sum())),
        _ratio(float(gross.sum()), float(notional.sum())),
        _ratio(float(gross.sum()), turnover),
        _ratio(float(net[long].sum()), float(notional[long].sum())),
        _ratio(float(net[short].sum()), float(notional[short].sum())),
        float((net / notional * 100.0).sum()) if len(trades) else 0.0,
    )


def with_extra_fees(trades: pd.DataFrame, buy_bp: float, sell_bp: float) -> pd.DataFrame:
    """K-fill's run: ``trades`` with every buy also charged ``max(0, buy_bp)`` and every
    sell ``max(0, sell_bp)`` bp of its fill's notional, as an extra fee (``extra``);
    ``fees``, ``net`` and ``net_bp`` include it. A long buys at its entry and sells at
    its exit, a short the reverse."""
    a_buy, a_sell = max(0.0, float(buy_bp)), max(0.0, float(sell_bp))
    side = trades["side"].to_numpy(dtype=np.int64)
    size = trades["size"].to_numpy(dtype=np.float64)
    entry = trades["entry_price"].to_numpy(dtype=np.float64) * size
    exit_ = trades["exit_price"].to_numpy(dtype=np.float64) * size
    extra = fill_fee(entry, np.where(side > 0, a_buy, a_sell)) + fill_fee(
        exit_, np.where(side > 0, a_sell, a_buy)
    )
    out = trades.copy()
    out["extra"] = extra
    out["fees"] = trades["fees"].to_numpy(dtype=np.float64) + extra
    out["net"] = trades["net"].to_numpy(dtype=np.float64) - extra
    out["net_bp"] = out["net"] / out["notional"] * 1e4
    return out
