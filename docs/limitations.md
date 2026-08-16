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

## The sequence model is under-evaluated

The TCN is reported on a single fold, not the full seven-fold schedule that
every other model runs. This is a compute limit rather than a choice about
method, and it is worth stating rather than hiding behind a smaller table.

Two causes, both since identified. PyTorch's Metal backend hangs on tensors of
the size a full training block produces with these operations, so the network
runs on CPU. And XGBoost and PyTorch each bundle an OpenMP runtime which, in one
macOS process, do not coexist: measured here, the network fits 40k rows in 20
seconds alone and never returns once XGBoost has run in the same interpreter.

The second is why a script looping over models in one process appeared to be
merely slow. The staged pipeline avoids it by construction — each stage is its
own invocation — but the comparison runs in this study were single scripts, and
the network trains on the tail of each training block rather than all of it.

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
