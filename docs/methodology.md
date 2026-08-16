# Methodology

How a result is produced here, and why each step is arranged the way it is. The
short version: almost every decision in this document exists to stop a number
from looking better than it is.

## The order of operations

```
raw data → features → labels → split → fit → choose threshold → evaluate
                                  ↑                    ↑            ↑
                              purge from          validation      test,
                            label horizon           only         once
```

Features and labels are computed on the **whole continuous series**, before any
splitting. This sounds backwards and is not: every feature is verified causal,
so computing it across the full series gives each row exactly the value it would
have had in real time. Computing per block instead would restart every rolling
window at each boundary and throw away the warm-up rows for no benefit.

The split then applies to the finished frame, and the purge removes the rows
whose labels reach across a boundary.

## Features

Eleven quantities at the touch — spread, queue imbalance, microprice deviation,
depth, order flow imbalance, short-horizon returns, realised volatility. All
standard definitions from the open literature.

Deliberately few. A handful of quantities that can each be checked by hand is
worth more than a wide set nobody can reason about, and on a sample of this size
a wide set mostly finds ways to fit the particular fortnight it was given.

Two properties are enforced mechanically rather than by review:

**Causality.** Each feature declares how far back it looks. The test suite
truncates the data at a cut and asserts that no value before the cut changed.
This catches centred windows, backward fills and whole-sample normalisation —
three ways look-ahead gets in, none of which makes anything visibly fail. See
[`src/trading_research/validation/leakage.py`](../src/trading_research/validation/leakage.py).

**Scale invariance.** A feature meant to transfer between instruments must not
change when the instrument's units do. Every feature is classified as scale-free
or unit-dependent, and adding one forces that decision. This was not
precautionary: raw order flow imbalance is in size units, and XRPUSDT rests
about 10^5 units at the touch against single digits for BTCUSDT. A model fitted
on one would read the other as permanently extreme, so a transfer experiment
using it would have measured a unit mismatch and reported it as a finding about
markets.

## Labels

Forward mid return over a fixed horizon, cut into sell / hold / buy by a
threshold.

The threshold is set to the **round-trip cost**, not to a tuned value. Below
that, "the price moved" and "money could have been made" are different
statements, and a model that predicts the first perfectly still loses money.

The label declares its horizon, and the split takes its purge from that
declaration rather than from a configuration field — so changing the horizon
cannot silently invalidate the split.

`threshold_for_share` exists for the cases where trade frequency should be fixed
first. Choosing a threshold directly conflates *how often to trade* with *when
to trade*, and two configurations compared at different trade frequencies are
mostly a comparison of the frequencies.

## Validation

Rolling walk-forward over whole days: 14 train, 7 validation, 7 test, stepped
one day, seven folds. Twenty-eight days per fold, thirty-four days in total.

**Days, not row fractions.** Activity, spread and volatility all have a daily
shape, so day boundaries produce blocks comparable to each other. A cut at row
60% lands mid-afternoon and the resulting periods differ for reasons that have
nothing to do with the model.

**Purging.** A label at *t* reads the price at *t + H*, so the last *H* rows of
a training block were decided by prices inside validation. Training on them
shows the model validation prices however carefully the blocks were cut, and
nothing about the resulting numbers looks wrong. Those rows are dropped.

**Gaps are refused, not skipped.** Treating 1 March as following 27 February
would make one fold cover less market than the others while still being labelled
fourteen days.

**Test is used once.** The confidence threshold is chosen on validation and
applied unchanged to test. Choosing it on test is the most common way a
short-horizon result is overstated.

## Models

Four, in increasing order of capacity:

| Model | Purpose |
|---|---|
| `always_hold` | Never trades. Under taker costs this is a genuinely strong strategy |
| `class_prior` | Predicts training frequencies, ignores features |
| `logistic` | Linear, readable coefficients |
| `xgboost` | Finds interactions the linear model cannot |

`always_hold` is not a formality. Trading is expensive, so doing nothing has a
real expected value, and a model that does not beat it net of costs has not
earned its complexity. `class_prior` separates skill that comes from the
features from skill that comes from knowing most moments are HOLD.

Hyperparameters are modest and fixed. A large search would make the comparison a
comparison of search budgets.

## Costs

Taker on both legs: the published Binance USD-M rate of 0.05% per side, plus the
spread once, plus slippage per side. Roughly 11 basis points per round trip.

Maker economics are not modelled. A resting order fills preferentially when the
market is about to move through it, and estimating that adverse selection needs
queue-position data this project does not have. Quoting maker costs without it
would flatter every result.

The same cost model is used when labelling, when deciding, and when computing
profit and loss. Backtests routinely overstate results by labelling against a
mid-price move the spread would have eaten.

## What the accounting does not model

Stated plainly, because these all push in the optimistic direction:

- Each signal is an independent round trip held for the label horizon. There is
  no position netting and no limit on concurrent exposure, which overstates
  achievable size.
- No queue position, no partial fills, no market impact. A real order changes
  the book it trades against.
- No latency between the decision and the order arriving, beyond the flat
  slippage allowance.
- Signals are not thinned. Consecutive observations produce highly overlapping
  trades, so trade counts are larger than any real system would run.

These are simplifications of *execution*. They do not affect whether a
directional edge exists at all, which is what the cost floor decides — and if an
edge does not survive here, no execution sophistication rescues it.

## Reproducibility

One seed per run. The configuration, the schema version, the fold schedule and
the cost model are written into the run manifest, so a reported number can be
traced back to the assumptions that produced it.
