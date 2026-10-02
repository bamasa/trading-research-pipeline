# Limitations

What this project does not establish. Written plainly, because the honest
version is more useful than the flattering one — and because a reader who finds
the limitations only after trusting a number has been misled by omission.

## No profitability claim

Nothing here is evidence that any strategy is or was profitable. Every result is
computed with hindsight, over a fixed historical window, under the assumptions
listed in [`methodology.md`](methodology.md). A backtest is a measurement of how
a rule would have scored on one particular past, not a forecast.

## The cost floor dominates

At taker rates the round trip costs about 11 basis points before anything is
predicted. On February–March 2024 BTCUSDT futures, only around 1% of moments
have a ten-second move large enough to clear that — and that figure assumes the
*direction is predicted perfectly*.

This is arithmetic, not modelling. It bounds what any classifier on this horizon
can achieve, and it is the single most important number in the repository. A
short-horizon result that ignores it is not a weaker version of the same claim;
it is a different claim about a market where trading is free.

## Synthetic results are circular

The generator injects a signal and the pipeline recovers it. That demonstrates
the pipeline works and nothing else. Every synthetic artefact is labelled so it
cannot be quoted as though it said something about markets.

## Only the touch is observed

Free exchange archives publish the best bid and ask, and nothing behind them.
Depth imbalance, book slope and concentration cannot be computed, so the feature
set is narrower than a full-depth study would use. The collector will close this
gap going forward; it cannot close it retrospectively.

## The book is sampled, not streamed

Data is reduced to the last update in each 100 ms interval. Intra-interval
updates are discarded, which caps how fast a strategy can be assumed to react
and makes the results silent about anything faster.

## Execution is simplified

No queue position, no partial fills, no market impact, no position netting, no
limit on concurrent exposure, and no latency beyond a flat slippage allowance.
Every one of these pushes results in the optimistic direction. The accounting is
honest about direction and cost per trade; it is not a simulation of a trading
system.

That describes the taker backtest. The market-making simulator is a separate
mode that does model queue position, partial fills, latency, inventory limits
and funding, in event time; what it cannot observe — queue dynamics inside a
100 ms snapshot, its own impact on others — is replaced by stated rules, each
with its direction, in [`market_making_simulator.md`](market_making_simulator.md).

## Running the models needs one process each

The TCN now runs the full seven-fold schedule, but only because each model gets
its own interpreter.

The real cause was process layout. XGBoost and PyTorch each bundle an OpenMP
runtime, and in one macOS process they do not coexist: the network fits 40k rows
in 20 seconds alone, in 65 seconds inside the fold pipeline, and never returns
once XGBoost has run in the same interpreter. Not slow — stuck. Separately,
PyTorch's Metal backend hangs on tensors this size, so it runs on CPU.

The staged pipeline avoids the conflict by construction, since each stage is its
own invocation. The comparison scripts in this study were not, which is how a
layout problem spent an afternoon looking like a hardware one.

What remains a genuine limit is training size: the network sees the tail of each
training block — 60k rows of roughly 210k — rather than all of it. Given the
gross edge it reaches, 0.91 bp against an 11 bp round trip, more data would have
to change the result by an order of magnitude to matter.

What that means for reading its result: it is a fair single observation, not a
weaker version of the others. It cannot be compared fold-for-fold with the
tabular models, and it says nothing about what a properly resourced sequence
model would do. What bounds the answer is elsewhere — the expected edge per
trade is roughly the information coefficient times the volatility of the move,
and that product sits near 0.4 bp at every horizon against an 11 bp round trip.
A sequence model can raise the coefficient. It cannot raise it twenty-five-fold.

## One period, two instruments

Thirty-eight days of BTCUSDT and fifty-nine of XRPUSDT, from February and March
2024 — a period containing a strong rally. Results from a single regime do not
generalise, and the walk-forward folds overlap heavily, so the seven folds are
not seven independent observations. They show stability across nearby windows,
which is weaker than it looks.

## The period was favourable, not unlucky

The obvious objection to a negative result is that the window was quiet, and a
strategy that needs a 12 bp move cannot find one in a flat market. It does not
hold here. Realised daily volatility was measured across all 36 months of
2023–2025 on the same instruments: the median month came to 219 bp, the range
75–633 bp, and the window used here to 453 bp — **higher than 97% of the other
months in three years**.

So the period was among the most favourable available, which makes the negative
result stronger rather than weaker. A quieter month would have less to trade,
not more, and the cost line would not move.

## Class balance is extreme

With the threshold at the cost floor, roughly 99% of moments are HOLD. Accuracy
is meaningless at that balance, and metrics computed over so few positive cases
are unstable regardless of the model.

## What the leakage checks do not cover

Truncation invariance says a feature does not read later *rows*. It cannot say
the data in those rows was correctly timed to begin with. A feature stamped with
the moment a message was received rather than when the event happened is causal
with respect to row order and still unusable in practice. That is a
data-contract question, not something a test on the frame can settle.

## Not investment advice

This is research and engineering code published as a portfolio and a teaching
artefact. It is not a trading system, not a recommendation, and not a claim
about future returns.
