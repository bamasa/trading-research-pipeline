"""Posting instead of crossing, measured on the venue's own prints.

This is the last untested lever. Every corrected search in this project ends the
same way: a gross edge of three to eight basis points, and a taker round trip of
twelve to sixteen. The signal is real and small; the cost is real and larger.
Posting is the only move that attacks the cost rather than trying to grow the
signal, and on Bybit it attacks it hard — a maker fee is 0.02% against 0.055%,
and a passive fill earns the spread instead of paying it. Four basis points a
round trip against fourteen.

If adverse selection were free, that arithmetic would already be a strategy.
It is not free, and the point of this script is to find out what it costs
instead of assuming a number for it.

How the fill is decided
-----------------------
:mod:`trading_research.backtest.maker` holds the mechanism; this script feeds
it. The book comes from the order-book archives already downloaded, and the
aggressive flow from Bybit's trade prints, whose ``side`` column names the
aggressor. Per five-second row:

* **sell volume at the bid** — size of prints where a seller crossed at or below
  the best bid. This is what consumes a resting buy order.
* **buy volume at the ask** — the mirror, consuming a resting sell order.

A signal to buy posts at the bid, joins the queue behind the size already
resting there, and fills only once that much selling has traded through. If the
bid instead ticks up — the move the signal wanted — the order is left behind and
usually times out unfilled.

What is varied
--------------
The timeout (how long the order rests), whether it joins the touch or posts
behind it, the holding period, and whether the exit is posted or crossed. Each
combination is run on the search block and then, unchanged, on the held-out
block, exactly as the taker searches were.

Two numbers matter equally
--------------------------
``net_per_fill_bp`` says whether the fills that happen are profitable.
``net_per_attempt_bp`` says whether the *strategy* is, because a signal that
never fills is a signal that earned nothing. Reporting only the first is how a
passive backtest flatters itself: filter hard enough and the surviving fills
look wonderful, while the strategy sits on its hands.
"""

from __future__ import annotations

import argparse
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from experiments._common import RESULTS, emit
from experiments.grand_search import (
    COSTS,
    GRID_SECONDS,
    ROWS_PER_DAY,
    Config,
    Data,
    build_features,
    normalise,
    to_grid,
)
from trading_research.backtest.evaluate import choose_confidence, decide
from trading_research.backtest.maker import MakerCosts, PostingRules, score, simulate
from trading_research.data.bybit_trades import load as load_trades
from trading_research.pipeline.stages import build_model
from trading_research.validation.changepoint import segment

#: Bybit USDT perpetuals: 0.02% maker, 0.055% taker.
MAKER_COSTS = MakerCosts(fee_bp_per_side=2.0, taker_fee_bp_per_side=5.5, slippage_bp=0.5)

#: The arrangements worth separating. Timeouts in rows of five seconds, so 12 is
#: a minute and 120 is ten.
TIMEOUTS = (12, 36, 120)
JOIN = (True, False)
PASSIVE_EXIT = (True, False)
HOLDS = (24, 60)

#: The signal. Deliberately one of the plainest configurations in the project —
#: the question here is execution, and a fancier model would make it harder to
#: tell which half was responsible.
SIGNAL = Config(
    plane="micro",
    model="logistic",
    target="direction",
    horizon=24,
    hold=24,
    cooldown=0,
    exit="clock",
    objective="net_per_trade_bp",
    refit="2d",
    train_days=10,
    gate="open",
)


def aggressive_flow(
    trades: pd.DataFrame, timestamps: pd.Series, bid: np.ndarray, ask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per row, the volume that would consume a resting bid and a resting ask.

    A print is assigned to the row whose interval contains it, and counted only
    if it happened at a price a resting order would have been standing at. A
    sell that traded *below* the best bid swept through it, so it counts too;
    one that traded above it never touched the queue.
    """
    edges = timestamps.to_numpy()
    slot = np.searchsorted(edges, trades["timestamp"].to_numpy(), side="right") - 1
    inside = (slot >= 0) & (slot < len(edges))
    slot, aggressor = slot[inside], trades["aggressor"].to_numpy()[inside]
    price, size = trades["price"].to_numpy()[inside], trades["size"].to_numpy()[inside]

    sell_at_bid = np.zeros(len(edges))
    buy_at_ask = np.zeros(len(edges))
    selling = (aggressor == -1) & (price <= bid[slot])
    buying = (aggressor == 1) & (price >= ask[slot])
    np.add.at(sell_at_bid, slot[selling], size[selling])
    np.add.at(buy_at_ask, slot[buying], size[buying])
    return sell_at_bid, buy_at_ask


def signals(data: Data, config: Config, start: int, stop: int) -> np.ndarray:
    """Walk-forward decisions over ``[start, stop)``, refit on the stated cadence.

    Identical in construction to the taker searches: fit on the front of the
    training window, choose the threshold on its tail, apply forward, never look
    at the rows being traded.
    """
    features = data.frames[config.norm_window]
    columns = data.plane[config.plane]
    target = data.target(config.target, config.horizon)
    forward = data.forward(config.hold)
    usable = features[columns].notna().all(axis=1).to_numpy()
    labelled = usable & target.notna().to_numpy()
    y = target.to_numpy()

    out = np.zeros(stop - start, dtype=int)
    stride = int(float(config.refit.removesuffix("d")) * ROWS_PER_DAY)
    for point in range(start, stop, stride):
        train_from = max(0, point - int(config.train_days * ROWS_PER_DAY))
        train_to = point - max(config.horizon, config.hold)
        if train_to - train_from < 20_000:
            continue
        mask = np.zeros(len(features), dtype=bool)
        mask[train_from:train_to] = True
        mask &= labelled
        index = np.flatnonzero(mask)
        if len(index) < 10_000:
            continue
        cut = int(len(index) * 0.75)
        fit_index, inner = index[:cut][-120_000:], index[cut:]
        if len(fit_index) < 5_000 or len(inner) < 3_000 or len(np.unique(y[fit_index])) < 2:
            continue

        model = build_model(config.model)
        model.fit(features.iloc[fit_index][columns], pd.Series(y[fit_index], index=fit_index))
        confidence, _ = choose_confidence(
            model.predict_proba(features.iloc[inner][columns]),
            pd.Series(forward[inner]),
            pd.Series(data.spread_bp[inner]),
            COSTS,
            objective=config.objective,
        )
        block = slice(point, min(stop, point + stride))
        rows = np.arange(block.start, block.stop)
        ok = usable[rows]
        if ok.sum() < 100:
            continue
        proba = model.predict_proba(features.iloc[rows[ok]][columns])
        decided = decide(proba, min_confidence=confidence)
        out[rows[ok] - start] = decided
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BICOUSDT")
    parser.add_argument("--book", type=Path, default=Path("data/book"))
    parser.add_argument("--trades", type=Path, default=Path("data/trades"))
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    files = sorted((args.book / args.symbol).glob("*.parquet"))
    book = to_grid(pd.concat([pd.read_parquet(p) for p in files], ignore_index=True))
    bid = book["bid_price_0"].to_numpy()
    ask = book["ask_price_0"].to_numpy()
    bid_size = book["bid_size_0"].to_numpy()
    ask_size = book["ask_size_0"].to_numpy()
    mid = (bid + ask) / 2.0
    spread_bp = (ask - bid) / mid * 1e4

    trades = load_trades(args.symbol, args.trades)
    sell_at_bid, buy_at_ask = aggressive_flow(trades, book["timestamp"], bid, ask)
    print(
        f"{args.symbol}: {len(book):,} rows, {len(trades):,} prints; "
        f"{(sell_at_bid > 0).mean():.1%} of rows saw selling at the bid"
    )

    raw = build_features(book)
    frame = normalise(raw, window=4000).astype("float32")
    frame["mid"] = mid
    cut = int(len(book) * 0.65)
    data = Data({4000: frame}, spread_bp, mid, search_end=cut)
    _ = segment(mid, spread_bp, minimum_rows=40_000)

    print("generating signals...")
    decision = np.zeros(len(book), dtype=int)
    decision[:cut] = signals(data, SIGNAL, 0, cut)
    decision[cut:] = signals(data, SIGNAL, cut, len(book))
    print(f"  {(decision != 0).sum():,} rows carry a signal")

    tick = float(np.min(np.diff(np.unique(bid))[np.diff(np.unique(bid)) > 0]))
    rows = []
    for timeout, join, passive, hold in product(TIMEOUTS, JOIN, PASSIVE_EXIT, HOLDS):
        rules = PostingRules(
            timeout_rows=timeout, join_touch=join, hold_rows=hold, passive_exit=passive
        )
        for name, block in (("search", slice(0, cut)), ("final", slice(cut, len(book)))):
            attempts = simulate(
                decision[block],
                bid[block],
                ask[block],
                bid_size[block],
                ask_size[block],
                sell_at_bid[block],
                buy_at_ask[block],
                rules,
                tick=tick,
            )
            result = score(attempts, MAKER_COSTS, spread_bp[block])
            days = (block.stop - block.start) / ROWS_PER_DAY
            rows.append(
                {
                    "block": name,
                    "timeout_s": timeout * GRID_SECONDS,
                    "join_touch": join,
                    "passive_exit": passive,
                    "hold_s": hold * GRID_SECONDS,
                    "days": round(days),
                    **result,
                }
            )

    table = pd.DataFrame(rows)
    table.insert(0, "symbol", args.symbol)
    RESULTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS / f"maker_{args.symbol}.csv", index=False)

    columns = [
        "block",
        "timeout_s",
        "join_touch",
        "passive_exit",
        "hold_s",
        "attempts",
        "fill_rate",
        "gross_per_fill_bp",
        "cost_per_fill_bp",
        "net_per_fill_bp",
        "net_per_attempt_bp",
    ]
    emit(
        table[columns].sort_values(["block", "net_per_fill_bp"], ascending=[True, False]),
        f"maker_{args.symbol}_summary",
    )


if __name__ == "__main__":
    main()
