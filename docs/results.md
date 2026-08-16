# Results

BTCUSDT and XRPUSDT, Binance USD-M futures, February–March 2024. Best bid and
ask sampled to a 100 ms grid, taker execution on both legs.

**Nothing here was profitable.** Across two instruments, four horizons, two
feature sets and four models, no walk-forward fold was positive after costs.
What follows is the evidence, and — more usefully — the mechanism.

---

## What was assumed

Read these before the numbers. Every one of them moves the result.

| | |
|---|---|
| Data | Binance USD-M `bookTicker` + `aggTrades`, public archive |
| Period | BTCUSDT 2024-02-01 … 2024-03-09 (38 days), XRPUSDT … 2024-03-30 (59 days) |
| Sampling | last update per 100 ms interval |
| Execution | taker on entry and exit |
| Fee | 5 bp per side (published USD-M taker rate) |
| Slippage | 0.5 bp per side |
| Round trip | **11.02 bp** (BTCUSDT), **12.68 bp** (XRPUSDT) |
| Validation | 14 train / 7 validation / 7 test days, stepped 1 day, 7 folds |
| Label threshold | the round-trip cost |

Maker execution is not modelled. A resting order fills preferentially when the
market is about to move through it, and estimating that adverse selection needs
queue-position data this project does not have. Quoting maker economics without
it would flatter every number below.

---

## 1. The ceiling, before any model

Share of moments whose future move exceeds the round-trip cost — an upper bound
that already assumes the direction is predicted perfectly.

| Horizon | BTCUSDT | XRPUSDT |
|---|---:|---:|
| 1 s | 0.01% | 0.02% |
| 5 s | 0.10% | 0.16% |
| 10 s | 0.31% | 0.41% |
| 30 s | 1.81% | 1.93% |
| 1 min | 4.85% | 5.28% |
| 5 min | 25.45% | 28.38% |
| 10 min | 39.39% | 42.65% |

Measured over the same ten days for both instruments, because comparing
instruments over different windows measures the window. An earlier version of
this table did exactly that — 38 days of BTCUSDT against 59 of XRPUSDT — and
made XRPUSDT look twice as tradeable. On matched days the gap is small.

## 2. Prediction works, and decays

Correlation of each feature with the forward return. BTCUSDT, 5.9M observations.

| Horizon | queue imbalance | microprice dev. | OFI |
|---|---:|---:|---:|
| 0.5 s | 0.242 | 0.190 | 0.164 |
| **1 s** | **0.270** | 0.207 | 0.180 |
| 10 s | 0.192 | 0.147 | 0.109 |
| 1 min | 0.081 | 0.062 | 0.040 |
| 5 min | 0.038 | 0.029 | 0.021 |
| 10 min | 0.024 | 0.019 | 0.013 |

The signal is real and peaks around one second. Part of the sub-second figure is
mechanical — the mid oscillates between bid and ask, and the microprice is
simply a better estimate of the underlying price. That component is not
tradeable: capturing it means crossing the spread that creates it.

## 3. Why no horizon works

Expected gross edge per trade is roughly the information coefficient times the
volatility of the move over that horizon. Both change; their product does not.

| Horizon | IC | σ move (bp) | edge = IC × σ | edge / cost |
|---|---:|---:|---:|---:|
| 1 s | 0.270 | 0.59 | 0.16 | 0.01× |
| 10 s | 0.192 | 2.10 | 0.40 | 0.04× |
| 1 min | 0.081 | 5.05 | 0.41 | 0.04× |
| 5 min | 0.038 | 11.12 | 0.42 | 0.04× |
| 20 min | 0.017 | 21.49 | 0.36 | 0.03× |

**Signal decays at almost exactly the rate volatility grows.** The expected edge
per trade sits near 0.4 bp at every horizon from one second to twenty minutes,
while the fee stays at 11 bp.

This is the central finding, and it is arithmetic rather than modelling. It
bounds what any classifier on this data can achieve, and it says that choosing a
different horizon is not the lever.

---

## 4. Walk-forward results

### Ten-second horizon, hand-picked features

| Instrument | Model | Trades | Hit | Gross/trade | Net/trade | Folds + |
|---|---|---:|---:|---:|---:|---:|
| BTCUSDT | always_hold | 0 | — | — | 0.00 | 0/7 |
| BTCUSDT | logistic | 508 | 53.0% | −1.72 | −14.41 | 0/7 |
| BTCUSDT | xgboost | 2283 | 51.1% | +0.21 | −10.83 | 0/7 |
| XRPUSDT | logistic | 472 | 55.4% | −7.71 | −37.45 | 0/7 |
| XRPUSDT | xgboost | 2241 | 48.1% | +0.26 | −20.61 | 0/7 |

`class_prior` never traded: its directional probabilities sit near 0.5% and
clear no confidence threshold. That is what it is in the comparison for.

XRPUSDT is the instructive row. Its logistic model predicts direction *more*
often than BTCUSDT's — 55.4% against 53.0% — and loses more than twice as much
per trade. Small wins, large losses.

### Horizon sweep, BTCUSDT, hand-picked features

| Horizon | Model | Hit | Gross/trade | Net/trade | Folds + |
|---|---|---:|---:|---:|---:|
| 30 s | logistic | 51.5% | 0.91 | −10.70 | 0/7 |
| 1 min | logistic | 48.1% | 0.01 | −11.81 | 0/7 |
| 2 min | logistic | 52.9% | 5.18 | −6.19 | 0/7 |
| 5 min | logistic | 50.2% | 5.15 | −6.25 | 0/7 |
| 2 min | xgboost | 45.4% | 0.94 | −11.57 | 0/7 |

The best figure in the study is 5.18 bp at two minutes — still half of what it
needed. See §6 before taking it at face value.

### Sequence model against the tabular ones

One fold, one test block, all three models on identical data. The network is
reported on one fold rather than seven — see
[`limitations.md`](limitations.md).

| Model | Cooldown | Trades | Hit | Gross/trade | Net/trade |
|---|---:|---:|---:|---:|---:|
| tcn | 0 | 78 | **55%** | **+1.63** | −10.06 |
| tcn | 120 | 52 | 54% | −1.69 | −13.46 |
| logistic | 0 | 91 | 51% | −0.28 | −11.46 |
| logistic | 120 | 57 | 53% | **+1.81** | **−9.36** |
| xgboost | 0 | 220 | 48% | −1.64 | −12.65 |
| xgboost | 120 | 86 | **56%** | −1.29 | −12.31 |

**Seeing a window did not change the picture.** The network's best gross figure
is 1.63 bp against logistic regression's 1.81 — a difference well inside the
noise of 78 trades, and both an order of magnitude short of the 11.02 bp a
round trip costs.

That is what the cost arithmetic predicted. Expected edge per trade is roughly
the information coefficient times the volatility of the move, and no
architecture changes that product; a sequence model can raise the coefficient,
not multiply it by ten.

Two smaller observations. Thinning helps the linear model and hurts the network
— on a single fold that is noise rather than a finding, and it is recorded to
stop it being read as one. And accuracy misleads once more: the best hit rate in
the table belongs to the configuration with the second-worst gross edge.

### Wide generated features against hand-picked, two-minute horizon

194 generated features — lags, differences, rolling and exponential statistics,
z-scores over five windows, trade-flow aggregates and calendar terms — cut to 40
by a selector fitted inside each fold's training block.

| Features | Model | Trades | Hit | Gross/trade | Net/trade | Folds + |
|---|---|---:|---:|---:|---:|---:|
| 10 hand-picked | logistic | 383 | 51.5% | **+2.24** | −8.22 | 0/7 |
| 10 hand-picked | xgboost | 430 | 48.2% | +0.73 | −10.45 | 0/7 |
| 40 of 194 | logistic | 438 | **56.6%** | **−0.67** | −12.04 | 0/7 |
| 40 of 194 | xgboost | 1882 | 51.2% | −0.83 | −10.27 | 0/7 |

Three things worth reading twice.

**More features made it worse.** Ten hand-picked columns beat forty selected
from a hundred and ninety-four. Expanding the feature space without expanding
the sample gives the selector more ways to fit the particular fortnight it saw.

**Higher accuracy, lower profit.** The wide set predicts direction more often —
56.6% against 51.5% — and earns less. It catches many small moves and misses
large ones. Accuracy is not a proxy for profitability, and here it points the
wrong way.

**Gradient boosting traded four times as often and lost five times as much.**
More candidate features give it more ways to find structure that is not there.

---

## 5. Trade thinning

Every result above charges a full round trip for each signal acted on.
Consecutive observations carry almost the same view, so that pays for one
opinion many times: on a 100 ms grid a two-minute horizon opens twelve hundred
overlapping positions for a single sustained signal.

Thinning fixes it with two rules — one position at a time, and an optional
cooldown after closing. Two-minute horizon, hand-picked features:

**BTCUSDT**, one model over five test days:

| Rule | Trades | Gross/trade | Net/trade | Hit |
|---|---:|---:|---:|---:|
| every signal | 2,255 | **−0.15** | −11.19 | 49% |
| one position at a time | 230 | **+1.13** | −9.98 | 51% |
| + cooldown 24 | 189 | +1.02 | −10.09 | 53% |
| + cooldown 120 | 124 | **+1.46** | −9.69 | **54%** |

**XRPUSDT**, seven folds:

| Model | Cooldown | Trades | Gross/trade | Net/trade | Folds + |
|---|---:|---:|---:|---:|---:|
| logistic | 0 | 121 | −1.75 | −14.84 | 0/7 |
| logistic | 120 | 83 | −1.24 | −14.21 | 0/7 |
| xgboost | 0 | 960 | −0.27 | −13.08 | 0/7 |
| xgboost | 24 | 614 | **+0.26** | −12.54 | 0/7 |
| xgboost | 120 | 280 | **+0.92** | −11.85 | 0/7 |

Thinning does what it should: trading eighteen times less turns a negative gross
edge into a positive one, and the hit rate rises with it. The signals that
survive really are the better ones.

It does not close the gap. The best gross figure here is 1.46 bp against an
11 bp round trip, and no fold on either instrument was positive.

One detail worth noting for anyone reading model comparisons: **the ranking
flips between instruments.** Logistic regression is the better model on BTCUSDT
and the worse one on XRPUSDT. Whatever separates them is smaller than the
difference between two markets in the same month.

## 6. Stability

The two-minute, hand-picked, logistic configuration produced a gross edge of
**5.18 bp** per trade in the horizon sweep and **2.24 bp** in the wide-feature
run. Same code, same model, same horizon, same instrument.

The only difference: the later run trims the data to exactly the days the folds
use, so four of thirty-eight days at the end were excluded from the statistics.

**The edge more than halved from a 10% change in sample boundaries** — a larger
difference than between any two models in this study.

The folds are also not seven independent observations. Each shifts one day
against a twenty-eight-day window, so consecutive folds share about 96% of their
data. Seven agreeing folds are close to one observation repeated.

---

## 7. Selection: three criteria, two of them wrong

The ranking stage was compared three ways on four days of BTCUSDT.

| Criterion | Top-ranked features |
|---|---|
| Mutual information, 3-class label | spread volatility, then time-of-day 4th |
| Mutual information, direction only | spread volatility, **time-of-day 2nd and 3rd** |
| Correlation with signed return | queue imbalance, microprice deviation |

The first rewards whatever separates "a big move happened" from "nothing
happened", which is volatility, not direction. The second was meant to fix that
and made it worse: restricting to moving rows shrinks the sample, and
nearest-neighbour mutual information is biased upward on small samples.

Time-of-day scoring highly is the tell. It predicts *when* the market moves and
says nothing about *which way*; a model leaning on it trades the volatile hours
and pays full costs for the privilege. Only the third criterion put the
directional features on top and dropped the calendar terms entirely.

---

## 8. Sensitivity to the fee

The fee is the obvious lever, so it is worth checking properly rather than
assuming. Costs are linear in the fee, so net profit per trade at any tier
follows from the gross edge and the trade count already measured.

Net basis points per trade, two-minute horizon, BTCUSDT:

| Features / model | 5.0 bp | 3.0 bp | 1.7 bp | 0.0 bp |
|---|---:|---:|---:|---:|
| 10 hand-picked / logistic | −8.78 | −4.78 | **−2.18** | **+1.22** |
| 10 hand-picked / xgboost | −10.29 | −6.29 | −3.69 | −0.29 |
| 40 of 194 / logistic | −11.70 | −7.70 | −5.10 | −1.70 |
| 40 of 194 / xgboost | −11.85 | −7.85 | −5.25 | −1.85 |

Fee at which each would break even:

| Features / model | Gross/trade | Break-even fee |
|---|---:|---|
| 10 hand-picked / logistic | +2.24 | **0.61 bp per side** |
| 10 hand-picked / xgboost | +0.73 | unreachable |
| 40 of 194 / logistic | −0.67 | unreachable |
| 40 of 194 / xgboost | −0.83 | unreachable |

**The fee tier does not close the gap.** The best configuration needs 0.61 bp
per side; Binance's top volume tiers reach roughly 1.7, nearly three times
that. Even at a *zero* fee it earns only 1.22 bp per trade, because the spread
and slippage cost about 1 bp whatever the tier. Three of the four
configurations lose money at any fee including zero — their gross edge is
negative, or smaller than the spread and slippage alone.

This corrects an earlier reading of the same data. Taking the 5.18 bp gross
figure from the horizon sweep, the break-even fee works out at 2.08 bp per side
— reachable at a high volume tier. But that figure is the unstable one: §6
shows it falling to 2.24 bp once the sample is trimmed to the days the folds
actually use. The lower number is the one to plan against.

## 9. What would have to change

- **Book depth.** One level is observed here because that is all any exchange
  publishes for free. Level imbalance, book slope and concentration need a
  collector recording the stream forward in time.
- **Trade thinning.** Every signal is treated as an independent round trip. A
  real system imposes a cooldown and does not hold overlapping positions,
  which raises edge per trade at the cost of trade count.
- **Maker execution.** Changes the arithmetic entirely — no spread crossed, and
  a lower or negative fee. It cannot be evaluated honestly without
  queue-position data, and is the only lever here large enough to matter.

Three things that would **not** close it, on this evidence: a longer horizon,
since the edge is horizon-invariant (§3); more features, which measurably made
it worse (§4); trade thinning, which improves edge per trade but by about one
basis point (§5); and a better fee tier, which falls short by a factor of three
(§8).

---

## Reading this

No result here is evidence that any strategy is or was profitable. These are
backtests computed with hindsight, over one period, in one regime, with
simplified execution — no queue position, no partial fills, no market impact,
no limit on concurrent exposure. Every one of those simplifications flatters the
result, and it is still negative.

See [`limitations.md`](limitations.md) and [`methodology.md`](methodology.md).
