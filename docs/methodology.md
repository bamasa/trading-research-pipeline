# Methodology

How a result is produced here, and why each step is arranged the way it is. The
short version: almost every decision in this document exists to stop a number
from looking better than it is.

## The pipeline as stages

Five stages, each a command, each reading and writing files:

```
prepare  raw book + trades      ->  feature matrix, one file per day
select   training window        ->  chosen feature list
train    training window        ->  fitted model
predict  any period             ->  class probabilities
backtest probabilities          ->  trades and profit
```

Slower than one in-memory run, and the right trade. A stage re-runs without
repeating what came before, which matters when feature building takes minutes
and fitting takes seconds. More importantly every intermediate is inspectable: a
pipeline that only emits a final number is one whose middle nobody checks, and
the middle is where the mistakes in this kind of work live.

Each stage writes a manifest recording its inputs, parameters and outputs, so a
result traces back to what produced it rather than to a memory of how it was
run.

**What varies is chosen independently.** Instrument, feature set, target horizon
and model are separate flags — `--symbol`, `--features`, `--horizon`, `--model` —
so comparing combinations is the normal case rather than a special one. Models
resolve through a registry, so adding one is an entry in a table rather than a
branch in the caller:

```
--model always_hold | class_prior | logistic | xgboost | tcn
```

**The ordering guarantee.** Stages that fit anything — `select` and `train` —
take an explicit date range and load only those files. Later data is not merely
unused; it is never in memory. That is stronger than filtering after loading,
because code cannot accidentally use what was never there. `predict` and
`backtest` fit nothing at all.

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

## How often to retrain

The walk-forward above fits a model once per fold and uses it for a whole test
week. That is a choice, not a fact, and by the seventh day the model is acting
on a fortnight that ended a week ago. The alternative is to refit as you go, and
three numbers describe any such scheme:

| Parameter | Meaning | Trade-off |
|---|---|---|
| `train_days` | How much history the fit sees | Sample size against staleness |
| `apply_days` | How long a fit is used before replacement | 1 means daily retraining |
| `step_days` | How far the window moves each time | Equal to `apply_days` gives contiguous coverage, which is what a live system produces |

None has an obvious value, so all three are searched — `trading-research
retrain-search`. The span is split chronologically: the schedule is chosen on
the first half and applied once to the second. A schedule is only eligible if it
produced at least `--min-windows` refits on validation, because the best average
over two windows is one lucky window with a decimal point.

**Where the confidence threshold comes from.** A retraining scheme has no
separate validation block — it has a training window and the days it trades. So
the *tail of the training window* is reserved for the threshold sweep: the model
fits on the front, the threshold is swept on the tail, and the days being traded
see neither. That costs sample size and is the only arrangement in which
everything configuring the trade decision comes from strictly before the first
day it trades.

**Windows that cannot be scored are counted, not hidden.** A training block with
one class, or too few complete rows, is recorded as a skipped window rather than
aborting the sweep — a schedule that fails half its windows is a finding about
that schedule, and an exception would lose it.

The search itself knows nothing about instruments, features or models: it is
handed a callable that fits on one date range and scores another
(`trading_research.validation.retrain`), and the pipeline builds that callable
(`trading_research.pipeline.retraining`). Another instrument is a different
prepared directory; another model is a different `--model`.

## Models

Five, in increasing order of capacity:

| Model | Purpose |
|---|---|
| `always_hold` | Never trades. Under taker costs this is a genuinely strong strategy |
| `class_prior` | Predicts training frequencies, ignores features |
| `logistic` | Linear, readable coefficients |
| `xgboost` | Finds interactions the linear model cannot |
| `tcn` | Sees a window of past rows rather than one row |

The sequence model is there because a per-row model only gets whatever the
feature engineering managed to compress into that row, and shape — an imbalance
that has been building versus one that just appeared — is not fully captured by
rolling statistics. Dilated causal convolutions were chosen over a recurrent
network because causality is then structural: the kernel cannot reach forward
even if the data handed to it were misaligned. On a problem where look-ahead is
the main hazard, an architecture that makes it impossible beats one that merely
permits avoiding it.

`always_hold` is not a formality. Trading is expensive, so doing nothing has a
real expected value, and a model that does not beat it net of costs has not
earned its complexity. `class_prior` separates skill that comes from the
features from skill that comes from knowing most moments are HOLD.

Hyperparameters are modest and fixed. A large search would make the comparison a
comparison of search budgets.

## Probability calibration

`--calibrate isotonic` or `sigmoid` wraps any model, fitting the calibrator on
the tail of the training block — data the model itself did not see, since a
calibrator trained on the model's own fit learns to correct an overconfidence
that only exists there.

Both methods are **monotonic**, and that decides what calibration can do here.
A strategy that trades whenever a score clears a threshold, and finds the
threshold by sweeping it on validation, behaves identically before and after:
the same trades are selected at a different numeric threshold. Observed
directly — the same model calibrated and not produces the same ordering and a
different confidence scale.

It earns its place where the number is used as a number: sizing by expected
value, comparing models on different scales, or any rule combining a
probability with a payoff. Not for this pipeline's current decision rule, and
saying so is more useful than adding it and implying otherwise.

## Exit rules

A position closes on the first of take-profit, stop-loss, and the clock. The
clock alone is the default, and on this data it is also the best of the three:

| Rule | Gross bp/trade | Hit rate |
|---|---:|---:|
| clock at H/4 | 0.73 | 58% |
| clock at H/2 | 0.56 | 51% |
| **clock at H** | **1.47** | 54% |
| clock at 2H | 1.23 | 52% |
| clock at 4H | −0.90 | 50% |
| take-profit 5 bp | 0.60 | **69%** |
| take-profit 50 bp | 2.00 | 54% |
| stop-loss 5 bp | 1.00 | 42% |
| stop-loss 20 bp | 1.70 | 54% |

Holding for exactly the label horizon is right, which is unsurprising once
stated: the model predicted a move over H, so a position held for less is
closed before the prediction resolves and one held for longer is exposed to
something the model said nothing about.

A tight take-profit is the clearest trap in the table — 69% of trades win and
the average outcome halves, because it caps the winners while the losers run
their full course. A tight stop does the mirror image, converting ordinary
noise into realised losses.

These figures come from sweeping the rules over the test block, so they are an
optimistic bound rather than a result. Choosing an exit rule properly means
choosing it on validation, like the confidence threshold.

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

The market-making simulator is a separate execution mode that does model a
single netted position with limits, queue position, partial fills and latency,
though not its own impact on the book: see
[`execution_assumptions.md`](execution_assumptions.md) for what each mode does
and does not simulate.

## Trade thinning

Acting on every signal charges a full round trip for each one, and consecutive
observations carry almost the same view: on a 100 ms grid a two-minute horizon
opens twelve hundred overlapping positions for a single opinion.

Two rules fix that. **One position at a time** — a signal arriving while a
position is open is ignored. **A cooldown after closing** — having just traded
on a view, wait before acting on it again, or the same slow-moving signal
reopens immediately and the position is effectively never closed, only
re-charged.

Neither creates an edge. The total is what it is; thinning redistributes it
across fewer trades. What it makes measurable is edge per *opinion*, which is
the honest question — and on real data it moves the gross figure from −0.15 bp
per trade to +1.46 while cutting trade count eighteenfold.

These are simplifications of *execution*. They do not affect whether a
directional edge exists at all, which is what the cost floor decides — and if an
edge does not survive here, no execution sophistication rescues it.

## Reproducibility

One seed per run. The configuration, the schema version, the fold schedule and
the cost model are written into the run manifest, so a reported number can be
traced back to the assumptions that produced it.
