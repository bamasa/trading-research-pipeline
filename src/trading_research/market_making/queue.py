"""Where an own order stands in the queue at its price, and when prints reach it.

The data is a ten-level book photographed every hundred milliseconds and the
tape of prints with the aggressor's side. Neither contains the order being
simulated, and neither says who cancelled what inside a hundred milliseconds.
So the state of an order is one estimated number — the size in front of it,
``queue_ahead`` — and the rules below say how that number moves. Each one is a
choice where the data cannot decide, made in the direction stated and
bracketed where it matters.

Fills come only from prints
---------------------------
A print at exactly the order's price, from the side that hits it (a seller for
a bid), consumes the queue in front first; any excess fills the order, up to
what remains of it; whatever is left goes to orders behind. A print strictly
through the price (a sale below the bid) means the whole level was exhausted:
nothing is ahead any more and the order fills at its own limit, up to the size
of that print. Prints on the other side, or at prices that do not reach the
order, change nothing. A snapshot never fills anything on its own.

Snapshots correct the estimate
------------------------------
Between two snapshots the level at the order's price should have shrunk by the
prints at that price. If it shrank by more, the difference was cancelled, and
the cancellation is attributed by a rule: in proportion to the size ahead and
behind (``proportional``, the default), all behind (``pessimistic``) or all
ahead (``optimistic``). If it grew, the growth is behind the order — except in
the snapshot interval in which the order arrived, where part of the growth may
have joined before it. That part is the time share of the interval that had
passed when the order arrived (``pro_rata_time``, the default), bracketed by
none of it (``none``) and all of it (``all_ahead``). After every snapshot the
estimate is clamped to ``[0, visible size]``.

A level inside the visible ten but absent from the snapshot is empty, so
nothing is ahead. A level beyond the tenth is unobservable: the estimate is left
alone until it can be seen again, and an order that arrived there joins the tail
of whatever is visible when it first can be.
"""

from __future__ import annotations

import math
from enum import StrEnum

from trading_research.market_making.orders import Order


class CancelAttribution(StrEnum):
    """Whose orders a shortfall at the level belonged to."""

    PESSIMISTIC = "pessimistic"  # all behind the order
    PROPORTIONAL = "proportional"  # ahead and behind in proportion
    OPTIMISTIC = "optimistic"  # all ahead of the order


class ArrivalGrowth(StrEnum):
    """How much growth of the level during the arrival interval is ahead."""

    NONE = "none"
    PRO_RATA_TIME = "pro_rata_time"
    ALL_AHEAD = "all_ahead"


class QueuePriority(StrEnum):
    """Where an arriving order joins. FRONT is an upper bound, for the ladder only."""

    TAIL = "tail"
    FRONT = "front"


class CrossedPolicy(StrEnum):
    """What a snapshot showing the opposite touch at or through the order does."""

    TRADE_TAPE = "trade_tape"  # nothing: a fill needs a print
    ASSUME_FILLED = "assume_filled"  # fill in full: a diagnostic, never a result


#: Fill paths.
QUEUE, THROUGH, CROSSED, FLATTEN = "queue", "through", "crossed", "flatten"


def on_arrival(
    order: Order, level_size: float | None, *, inside_spread: bool, priority: QueuePriority
) -> None:
    """Initialise the queue when the order goes live.

    ``level_size`` is the visible size at the order's price in the latest
    snapshot: ``0.0`` when the price is inside the visible range but empty,
    ``None`` when it lies beyond the visible levels. An order better than its
    own side's touch is alone at a new level, so nothing is ahead.
    """
    if priority is QueuePriority.FRONT:
        order.front = True
        ahead = 0.0
    elif inside_spread:
        ahead = 0.0
    elif level_size is None:
        ahead = math.inf
    else:
        ahead = level_size
    order.queue_ahead = ahead
    order.level_at_last_snapshot = 0.0 if inside_spread else level_size
    order.traded_at_price_since_snapshot = 0.0
    order.in_arrival_interval = True


def on_print(order: Order, price: int, size: float, aggressor: int) -> tuple[float, str]:
    """Apply one print to one order; return the size filled and the path.

    The order's ``remaining`` is reduced by the fill. The caller passes as
    ``size`` what is left of the print after any own order with better price
    priority took its share.
    """
    if aggressor != -order.side or size <= 0.0:
        return 0.0, ""
    beyond = (order.price - price) * order.side
    if beyond < 0:
        return 0.0, ""
    if beyond > 0:
        order.queue_ahead = 0.0
        filled = min(order.remaining, size)
        order.remaining -= filled
        return filled, THROUGH
    order.traded_at_price_since_snapshot += size
    if size <= order.queue_ahead:
        order.queue_ahead -= size
        return 0.0, ""
    excess = size - order.queue_ahead
    order.queue_ahead = 0.0
    filled = min(order.remaining, excess)
    order.remaining -= filled
    return filled, QUEUE


def on_snapshot(
    order: Order,
    level_size: float | None,
    *,
    rule: CancelAttribution,
    growth: ArrivalGrowth,
    snapshot_ns: int,
) -> None:
    """Correct the queue estimate against a new snapshot.

    ``level_size`` follows :func:`on_arrival`'s convention. The previous
    snapshot is the one recorded on the order; in its arrival interval that is
    the snapshot its queue was initialised from.
    """
    if level_size is None:
        order.level_at_last_snapshot = None
        order.traded_at_price_since_snapshot = 0.0
        order.in_arrival_interval = False
        return

    ahead = order.queue_ahead
    previous = order.level_at_last_snapshot
    if order.front:
        ahead = 0.0
    elif previous is None:
        # First sight of this level: join the tail of what is visible now.
        ahead = min(ahead, level_size)
    else:
        expected = max(0.0, previous - order.traded_at_price_since_snapshot)
        change = level_size - expected
        if change < 0.0:
            cancelled = -change
            if rule is CancelAttribution.OPTIMISTIC:
                ahead -= cancelled
            elif rule is CancelAttribution.PROPORTIONAL:
                ahead -= cancelled * min(1.0, ahead / expected)
        elif change > 0.0 and order.in_arrival_interval:
            if growth is ArrivalGrowth.ALL_AHEAD:
                ahead += change
            elif growth is ArrivalGrowth.PRO_RATA_TIME:
                span = snapshot_ns - order.arrival_snapshot_ns
                if span > 0:
                    share = (order.live_ns - order.arrival_snapshot_ns) / span
                    ahead += change * min(1.0, max(0.0, share))
    order.queue_ahead = min(max(ahead, 0.0), level_size)
    order.level_at_last_snapshot = level_size
    order.traded_at_price_since_snapshot = 0.0
    order.in_arrival_interval = False
