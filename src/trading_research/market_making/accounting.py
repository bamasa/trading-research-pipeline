"""Position, cash, fees and funding, with an identity that has to hold.

Average cost. A fill that adds to the position moves the average price; one
that reduces it realises the difference between the fill and the average on
the part closed; one that goes through zero realises the closed part and opens
the rest at the fill price. Cash is kept independently of all of that, from the
fills, fees and funding payments alone.

The two bookkeepings must agree:

    realised + unrealised - fees + funding  ==  cash + position * mark

Both sides are measured from the start of the day, which starts flat with no
cash. The simulator checks this after every change to the account, so a
bookkeeping error stops the run at the event that caused it instead of
surfacing as a number that is merely wrong.
"""

from __future__ import annotations

from dataclasses import dataclass


class IdentityError(AssertionError):
    """The two bookkeepings disagree."""


@dataclass(slots=True)
class Account:
    """One instrument's account, in quote currency and base units."""

    position: float = 0.0
    cash: float = 0.0
    avg_price: float = 0.0
    realised: float = 0.0
    #: Positive is paid; a rebate makes it negative.
    fees: float = 0.0
    #: Positive is received.
    funding: float = 0.0
    maker_turnover: float = 0.0
    taker_turnover: float = 0.0

    def fill(self, side: int, price: float, size: float, fee_bp: float, *, maker: bool) -> float:
        """Book a fill of ``size`` at ``price``; ``side`` is +1 bought, -1 sold.

        Returns the fee charged, in quote currency (negative for a rebate).
        """
        if size <= 0:
            raise ValueError(f"fill size must be positive, got {size}")
        notional = price * size
        fee = notional * fee_bp * 1e-4
        self.fees += fee
        self.cash -= side * notional + fee
        if maker:
            self.maker_turnover += notional
        else:
            self.taker_turnover += notional

        position = self.position
        if position == 0.0 or (position > 0) == (side > 0):
            new_position = position + side * size
            self.avg_price = (self.avg_price * abs(position) + notional) / abs(new_position)
        else:
            closed = min(size, abs(position))
            self.realised += closed * (price - self.avg_price) * (1.0 if position > 0 else -1.0)
            new_position = position + side * size
            if size > abs(position):
                self.avg_price = price
        # A position that is zero up to rounding is zero: a residue of 1e-17
        # would otherwise carry an average price from a closed trade.
        if abs(new_position) <= 1e-12 * max(size, abs(position)):
            new_position = 0.0
            self.avg_price = 0.0
        self.position = new_position
        return fee

    def settle_funding(self, rate: float, mark: float) -> float:
        """Settle one funding payment on the position held now.

        ``rate`` is the venue's rate for the settlement: positive means longs
        pay shorts. Returns the payment received (negative when paid).
        """
        if self.position == 0.0:
            return 0.0
        payment = -self.position * mark * rate
        self.funding += payment
        self.cash += payment
        return payment

    def unrealised(self, mark: float) -> float:
        if self.position == 0.0:
            return 0.0
        return self.position * (mark - self.avg_price)

    def equity(self, mark: float) -> float:
        """Cash plus the position at ``mark``: the day's profit so far."""
        if self.position == 0.0:
            return self.cash
        return self.cash + self.position * mark

    def check_identity(self, mark: float, *, tol: float = 1e-9) -> None:
        """Raise :class:`IdentityError` unless the two bookkeepings agree.

        The tolerance is relative to the turnover, the scale at which the cash
        sums accumulate rounding.
        """
        books = self.realised + self.unrealised(mark) - self.fees + self.funding
        cash = self.equity(mark)
        scale = max(1.0, self.maker_turnover + self.taker_turnover)
        if not abs(books - cash) <= tol * scale:
            raise IdentityError(
                f"realised + unrealised - fees + funding = {books!r} but "
                f"cash + position x mark = {cash!r} (position {self.position!r}, "
                f"mark {mark!r})"
            )
