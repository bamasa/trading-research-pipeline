# Execution assumptions

How each execution mode in this repository turns a decision into a trade, what
it charges, and what it does not simulate. Every assumption is listed with the
direction it pushes a result: "optimistic" flatters the strategy.

There are three modes, and they answer different questions:

| Mode | Code | Clock | Question |
|---|---|---|---|
| Taker | [`backtest/costs.py`](../src/trading_research/backtest/costs.py), [`backtest/execution.py`](../src/trading_research/backtest/execution.py) | the 5 s grid | does a directional edge clear the cost of crossing the spread? |
| Passive entry | [`backtest/maker.py`](../src/trading_research/backtest/maker.py) | the 5 s grid | does posting the entry instead of crossing help a directional signal? |
| Market maker | [`market_making/`](../src/trading_research/market_making/) | event time | does quoting both sides, with inventory limits, earn more than it pays? |

## Taker

Every entry and every exit crosses the spread. A round trip costs
`2 x fee + spread + 2 x slippage` in basis points of notional, with the spread
charged once (half on each leg) and taken from the book at the time of entry.
Fees are the venue's taker rate (5 bp on Binance USD-M, 5.5 bp on Bybit), and
slippage is a flat 0.5 bp a side. Signals are thinned to one position at a time
with a cooldown after each close, so a sustained view pays for one round trip
rather than hundreds.

What it does not model, all optimistic:

- **Queue position, partial fills and impact.** A taker order is assumed to fill
  in full at the touch plus the flat slippage, however large.
- **Latency.** Nothing between the decision and the fill beyond the slippage
  allowance.
- **Concurrent exposure.** Each signal is an independent round trip; there is
  no netting and no limit across instruments.

The one deliberate pessimism: every leg pays the taker fee, including exits that
could in principle have been posted.

## Passive entry

The entry is posted at (or a tick behind) the touch and fills only once the
opposing volume printed at that price exceeds the size that was resting there
when it was posted; it is cancelled after a timeout. The exit is either posted
under the same rule or crossed at the end of the hold. Maker fee 2.0 bp, taker
fee 5.5 bp, 0.5 bp slippage on a crossing leg only.

Three optimisms, stated in the module and kept, because the mode loses anyway:

- the queue ahead is the visible size at posting and never grows;
- cancellations ahead of the order are ignored;
- the order's own size is negligible, so it always fills in full.

## Market maker

Two-sided quoting in event time on Bybit's ten-level book (100 ms snapshots)
and trade prints, one instrument-day at a time. The full rule set — twenty
rules, each with its bias and the test that pins it — is in
[`market_making_simulator.md`](market_making_simulator.md). In one line each:

| Rule | Assumption | Bias |
|---|---|---|
| Fills | only from prints at or through the order's price, from the side that hits it; never from a snapshot | neutral; through-fills sized by the print are mildly pessimistic |
| Queue | joins the tail of the visible size; growth in the arrival interval split by time; later growth behind | neutral; the 100 ms staleness is bracketed (`none` / `all_ahead`) |
| Cancellations | attributed in proportion to the size ahead | optimistic against queue-reactive evidence; a verdict must survive `pessimistic` |
| Crossed snapshots | fill nothing without a print | optimistic in profit; counted and bracketed by `assume_filled` |
| Latency | 10 ms to arrive and to cancel; a cancel in flight can still be filled; feed latency 0 | feed latency 0 is optimistic; swept |
| Post-only | an order at or through the opposite touch on arrival is rejected | pessimistic |
| Size | a tenth of the trailing touch at most; own impact ignored | optimistic, bounded by the clip |
| Inventory | soft limit stops the growing side; hard limit is never crossed; past the soft limit a taker order flattens back | conservative |
| Fees | maker on passive fills, taker on flattens, by tier | neutral; swept |
| Funding | settled on the position held at each settlement, mid as the mark | neutral; the 00:00 settlement is never held through |
| Day | flat at the start and the end; quoting stops after a 5 s feed pause | conservative |

The study these serve, and the settings a verdict depends on, are fixed in
[`preregistration/market_making.md`](preregistration/market_making.md).
