"""An own order, from the decision that sent it to the moment it is done.

Four states, and the one that matters most is the third. A cancel takes time to
reach the venue, and until it does the order is still in the book and can still
be filled — usually by exactly the move the cancel was trying to avoid. A
simulator that removed the order at the decision would protect it from the
fills a real participant cannot escape.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class OrderState(IntEnum):
    #: Sent, not yet at the venue: cannot fill.
    PENDING_NEW = 1
    #: Resting in the book.
    LIVE = 2
    #: A cancel is on its way: the order CAN still fill until it lands.
    PENDING_CANCEL = 3
    #: Filled, cancelled or rejected.
    DONE = 4


#: How an order finished, recorded once it is done.
FILLED, CANCELLED, REJECTED, OPEN = "filled", "cancelled", "rejected", "open"


@dataclass(slots=True, eq=False)
class Order:
    """One own limit order. Prices in ticks, sizes in base units, times in ns.

    Compared by identity: two orders are the same order only if they are the
    same object, however alike their fields.
    """

    oid: int
    #: +1 bid, -1 ask.
    side: int
    price: int
    size: float
    remaining: float
    decided_ns: int
    #: When the order reaches the venue: the decision plus the order latency.
    live_ns: int
    #: When a cancel sent for it lands, if one was sent.
    cancel_ns: int | None = None
    #: Estimated size in front of it at its price, in base units.
    queue_ahead: float = 0.0
    #: Visible size at its price in the last snapshot; None when that level was
    #: beyond the visible book, so there is nothing to compare the next one with.
    level_at_last_snapshot: float | None = None
    #: Print volume at its price, on the side that hits it, since that snapshot.
    traded_at_price_since_snapshot: float = 0.0
    #: The snapshot its queue was initialised from.
    arrival_snapshot_ns: int = 0
    #: True until the first snapshot after it went live has been applied.
    in_arrival_interval: bool = False
    #: Joined at the front by assumption (the ladder's upper-bound rung).
    front: bool = False
    #: Whether the latest snapshot showed the opposite touch at or through it.
    crossed: bool = False
    state: OrderState = OrderState.PENDING_NEW
    done_ns: int | None = None
    outcome: str = OPEN
    #: Own position when it was decided, for the limit tests.
    position_at_decision: float = 0.0

    @property
    def can_fill(self) -> bool:
        return self.state is OrderState.LIVE or self.state is OrderState.PENDING_CANCEL

    @property
    def working(self) -> bool:
        """Not done and not being cancelled: the order a quote is compared with."""
        return self.state is not OrderState.DONE and self.cancel_ns is None
