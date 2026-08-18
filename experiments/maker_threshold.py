"""What maker execution would have to deliver to close the gap.

Everything else in this project assumes taking liquidity, and says so: a
directional model that wants to act now must cross the spread. The one lever
left after the retraining sweep is posting instead of crossing, and the standing
objection to modelling it is adverse selection — a resting order fills
preferentially when the market is about to move through it.

We have enough to bound it. ``aggTrades`` carries ``is_buyer_maker``, so the
aggressor side of every print is known, and ``bookTicker`` carries the size
resting at the touch. Together those give a first-order queue model: post at the
best bid, wait behind the size already there, and fill once enough sell-aggressor
volume has traded through it.

Why the queue matters more than it sounds
-----------------------------------------
An earlier version of this script assumed the front of the queue. It reported a
95% fill rate, a median wait of 0.3 s and essentially zero adverse selection —
which is not a finding but a tautology: an order that fills instantly, before
anything has happened, cannot have been selected against. The queue is not a
detail of the maker case, it *is* the maker case, and the asymmetry it creates
is the entire question:

- when price falls, the queue ahead is consumed and the order fills — into a
  move that is already going against it;
- when price rises, the touch moves up, the order is left behind, and the
  favourable move is missed.

Fills are therefore drawn from the losing half of the distribution. What follows
measures how much that costs, in basis points, rather than assuming a figure.

The model is still optimistic in three ways, all stated: the queue ahead is
taken as the size at the touch and never grows, cancellations ahead of the order
are ignored, and the order's own size is treated as negligible. A real queue is
worse than this one.

    uv run python -m experiments.maker_threshold --symbols BTCUSDT --days 5
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments._common import COSTS, emit, parser, round_trip

#: Binance USD-M maker fee at the standard tier, in basis points.
MAKER_BP = 2.0

#: How long a posted order is left resting before the decision is abandoned.
#: Swept rather than fixed: this is the whole trade-off. Waiting longer fills
#: more orders, and the extra fills are the ones the market had to move to
#: reach, so they are the worse ones.
PATIENCE_S = (1.0, 2.0, 5.0, 10.0, 30.0, 60.0)

#: Holding period after a fill, matching the horizon used elsewhere (2 min).
HORIZON_S = 120.0

#: One decision every N book rows. The book updates far faster than any model
#: would act, and scanning every row buys nothing but time.
DECISION_STRIDE = 500


def simulate_day(book: pd.DataFrame, trades: pd.DataFrame, patience_s: float) -> pd.DataFrame:
    """Post a passive buy at each decision moment and record what happened."""
    book_ns = book["timestamp"].to_numpy("datetime64[ns]").astype("int64")
    bid = book["bid_price_0"].to_numpy("float64")
    bid_size = book["bid_size_0"].to_numpy("float64")
    ask = book["ask_price_0"].to_numpy("float64")
    mid = (bid + ask) / 2.0

    trade_ns = trades["timestamp"].to_numpy("datetime64[ns]").astype("int64")
    price = trades["price"].to_numpy("float64")
    quantity = trades["quantity"].to_numpy("float64")
    # is_buyer_maker: the buyer rested, so the aggressor was a seller. Those are
    # the prints that consume a resting bid.
    sell_aggressor = trades["is_buyer_maker"].to_numpy(dtype=bool)

    patience_ns = int(patience_s * 1e9)
    horizon_ns = int(HORIZON_S * 1e9)

    rows = []
    for i in range(0, len(book), DECISION_STRIDE):
        posted = bid[i]
        queue_ahead = bid_size[i]
        start = book_ns[i]

        exit_at = np.searchsorted(book_ns, start + horizon_ns, side="right") - 1
        if exit_at <= i or exit_at >= len(book):
            continue
        unconditional_bp = (mid[exit_at] / mid[i] - 1.0) * 1e4

        lo = np.searchsorted(trade_ns, start, side="left")
        hi = np.searchsorted(trade_ns, start + patience_ns, side="right")
        window = slice(lo, hi)

        # Volume that would have to clear before this order trades: prints from
        # sell aggressors at or below the posted price.
        consuming = sell_aggressor[window] & (price[window] <= posted)
        cumulative = np.cumsum(np.where(consuming, quantity[window], 0.0))
        through = np.flatnonzero(cumulative > queue_ahead)

        if through.size == 0:
            rows.append(
                {
                    "filled": False,
                    "unconditional_bp": unconditional_bp,
                    "spread_bp": (ask[i] - bid[i]) / mid[i] * 1e4,
                }
            )
            continue

        fill_ns = trade_ns[window][through[0]]
        at_exit = np.searchsorted(book_ns, fill_ns + horizon_ns, side="right") - 1
        if at_exit >= len(book) or at_exit < 0:
            continue

        rows.append(
            {
                "filled": True,
                "wait_s": (fill_ns - start) / 1e9,
                # Earned from the price actually obtained, held one horizon.
                "realised_bp": (mid[at_exit] / posted - 1.0) * 1e4,
                # The same horizon from the decision moment, mid to mid: the
                # move a model's measured edge is quoted on.
                "unconditional_bp": unconditional_bp,
                "spread_bp": (ask[i] - bid[i]) / mid[i] * 1e4,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parser(__doc__ or "").parse_args()
    args.days = args.days or 10
    summary = []

    for symbol in args.symbols:
        book_files = sorted((args.raw_root / symbol / "bookTicker").glob("*.parquet"))[: args.days]
        # Keyed by patience, because each day is read once and simulated at
        # every setting: the parquet read dominates, the simulation does not.
        parts: dict[float, list[pd.DataFrame]] = {p: [] for p in PATIENCE_S}
        for book_file in book_files:
            trade_file = args.raw_root / symbol / "aggTrades" / book_file.name
            if not trade_file.exists():
                continue
            book = pd.read_parquet(book_file)
            trades = pd.read_parquet(trade_file)
            for patience in PATIENCE_S:
                parts[patience].append(simulate_day(book, trades, patience))
            print(
                f"  {symbol} {book_file.stem}: {len(parts[PATIENCE_S[0]][-1]):,} decisions",
                flush=True,
            )
            del book, trades

        taker_cost = round_trip(symbol)
        # Post to enter, cross to exit. The exit cannot be assumed passive: a
        # position held to a horizon has to be closed whether or not anyone
        # comes to trade with it.
        maker_entry_taker_exit = MAKER_BP + COSTS.fee_bp_per_side + COSTS.slippage_bp

        for patience in PATIENCE_S:
            if not parts[patience]:
                continue
            result = pd.concat(parts[patience], ignore_index=True)
            filled = result[result["filled"]]
            if filled.empty:
                continue

            # Adverse selection is a *conditioning* effect, so it has to be
            # measured by comparing populations rather than two numbers on the
            # same rows. An earlier version compared the filled orders' realised
            # move with those same orders' unconditional move and found nothing
            # — which it had to, since that difference is only the entry price.
            everywhere = float(result["unconditional_bp"].mean())
            when_filled = float(filled["unconditional_bp"].mean())
            # Positive means fills arrive at worse moments than average: the
            # market was already moving against the order when it traded.
            adverse_bp = everywhere - when_filled

            summary.append(
                {
                    "symbol": symbol,
                    "patience_s": patience,
                    "decisions": len(result),
                    "fill_rate": float(result["filled"].mean()),
                    "median_wait_s": float(filled["wait_s"].median()),
                    "adverse_selection_bp": adverse_bp,
                    "taker_round_trip_bp": taker_cost,
                    "maker_then_taker_bp": maker_entry_taker_exit,
                    "cost_saved_bp": taker_cost - maker_entry_taker_exit,
                    "net_gain_bp": (taker_cost - maker_entry_taker_exit) - adverse_bp,
                }
            )

    emit(pd.DataFrame(summary), "maker_threshold")


if __name__ == "__main__":
    main()
