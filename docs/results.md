# Results

BTCUSDT and XRPUSDT, Binance USD-M futures, February–March 2024. Best bid and
ask sampled to a 100 ms grid, taker execution on both legs.

**Nothing here was profitable.** Across two instruments, four horizons, two
feature sets and four models, no walk-forward fold was positive after costs. The
nearest approach is XRP with daily retraining (§10), which earns 12.31 bp gross
against a 12.68 bp cost — short by 0.37 bp, over 141 trades, and still negative.
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
| Retraining sweep (§10) | schedule searched on the first half of the span, applied once to the second |
| Label threshold | the round-trip cost |

Maker execution is not modelled anywhere in §0–§10: every entry and every exit
crosses the spread and pays the taker fee. §11 takes it up separately, with a
queue model built from the aggressor side of each print, and treats the result
as a bound rather than as a strategy.

---

## 0. The result, in two figures

![Gross against net](../assets/bt_equity.png)

![What a trade earned](../assets/bt_outcomes.png)

The first shows a model whose calls are right often enough to climb gross, and a
fee that takes it 1,200 bp the other way. The second shows why: the distribution
of what a trade earns sits almost entirely inside the cost line, so most trades
could never have paid for themselves however the direction turned out.

Everything below is these two pictures in detail.

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

![Room to trade](../assets/breakeven_share.png)

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

![Signal decay](../assets/signal_decay.png)

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
needed. See §7 before taking it at face value.

### Sequence model against the tabular ones

Full seven-fold schedule, all three models, two-minute horizon, BTCUSDT. Each
model in its own process — see [`limitations.md`](limitations.md) for why that
is not optional.

| Model | Cooldown | Trades/fold | Hit | Gross/trade | Net/trade | Folds + |
|---|---:|---:|---:|---:|---:|---:|
| logistic | 0 | 174 | 50% | −0.10 | −11.25 | 0/7 |
| **logistic** | 24 | 143 | 51% | **+1.81** | **−9.32** | 0/7 |
| logistic | 120 | 104 | 53% | +0.81 | −10.33 | 0/7 |
| tcn | 0 | 122 | 48% | −1.85 | −13.22 | 0/7 |
| tcn | 24 | 91 | 47% | −1.75 | −13.22 | 0/7 |
| tcn | 120 | 65 | 46% | −1.81 | −13.29 | 0/7 |
| xgboost | 0 | 233 | 47% | −0.63 | −11.64 | 0/7 |
| xgboost | 24 | 194 | 46% | −1.73 | −12.75 | 0/7 |
| xgboost | 120 | 132 | 46% | −1.49 | −12.51 | 0/7 |

![Every model against the cost](../assets/model_comparison.png)

**Nothing was profitable: zero positive folds out of twenty-one.**

**The simplest model won.** Logistic regression reaches 1.81 bp gross per trade
and is the only one of the three positive at any cooldown. Gradient boosting and
the network are negative everywhere.

**The network did not earn its complexity.** It trains about thirty times slower
than the linear model, needs its own process, and is negative at every cooldown.
Seeing a window of history is worth something in principle; it is not worth
enough to matter against an 11 bp round trip.

The network's figures move between runs — it is initialised randomly and the
schedule is not seeded end to end — by more than the distance between the models
in this table. An earlier run of the same script put it at +0.39 bp rather than
−1.75. That instability is itself the finding: a model whose result swings by
two basis points across runs cannot be said to have found a 0.4 bp edge.

A note on reading single folds. An earlier version of this section reported the
network on one fold, where it showed +1.63 bp gross with no cooldown. The full
schedule gives −3.27 for that same configuration. The spread between folds is
larger than the spread between models, which is worth remembering before any
result here is quoted from one window.

### Wide generated features against hand-picked, two-minute horizon

194 generated features — lags, differences, rolling and exponential statistics,
z-scores over five windows, trade-flow aggregates and calendar terms — cut to 40
by a selector fitted inside each fold's training block.

Best cooldown per configuration, seven folds, BTCUSDT:

| Features | Model | Trades/fold | Hit | Gross/trade | Net/trade | Folds + |
|---|---|---:|---:|---:|---:|---:|
| 10 hand-picked | logistic | 143 | 51.4% | **+1.81** | −9.32 | 0/7 |
| 10 hand-picked | xgboost | 233 | 46.6% | −0.63 | −11.64 | 0/7 |
| 40 of 194 | logistic | 126 | **51.8%** | −2.88 | −14.04 | 0/7 |
| 40 of 194 | xgboost | 154 | **52.0%** | +0.80 | −10.25 | 0/7 |

Three things worth reading twice.

**More features made the good model worse.** Ten hand-picked columns under
logistic regression give the best gross edge in the table; the same model on
forty columns selected from a hundred and ninety-four gives the worst. Expanding
the feature space without expanding the sample gives the selector more ways to
fit the particular fortnight it saw, and the linear model has no capacity to
spare for the noise it is handed.

**Higher accuracy, lower profit.** The wide set predicts direction more often
than the hand-picked set does under the same model — 51.8% against 51.4% for
logistic, 52.0% against 46.6% for the tree — and the linear model earns far
less for it. It catches many small moves and misses large ones. Accuracy is not
a proxy for profitability, and here it points the wrong way.

**The tree moved the other way, and it does not rescue anything.** Gradient
boosting improves with the wider set — from −0.63 to +0.80 bp gross — which is
what a model built for many weak inputs should do. It is still an order of
magnitude short of the 11.02 bp it has to clear, and it lands below the
hand-picked linear model it was supposed to beat. The two directions cancel to
the same conclusion: the feature set is not the binding constraint.

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

![The effect of thinning](../assets/thinning.png)

Thinning does what it should: trading eighteen times less turns a negative gross
edge into a positive one, and the hit rate rises with it. The signals that
survive really are the better ones.

It does not close the gap. The best gross figure here is 1.46 bp against an
11 bp round trip, and no fold on either instrument was positive.

One detail worth noting for anyone reading model comparisons: **the ranking
flips between instruments.** Logistic regression is the better model on BTCUSDT
and the worse one on XRPUSDT. Whatever separates them is smaller than the
difference between two markets in the same month.

## 6. Why "trade only the best signals" looks profitable

The most tempting idea in this study, and the one that took the most care to
answer. If the edge per trade is too small, trade less and pick better: act only
on the signals the model is most confident about.

![Selection leak](../assets/selection_leak.png)

Both lines are the same strategy, the same code and the same data. The only
difference is *when* the cutoff was decided.

| Selectivity | Cutoff chosen on test | Cutoff chosen on validation |
|---|---:|---:|
| all signals | −10.54 | −10.54 |
| top 1,000 | −9.92 | −10.47 |
| top 300 | −7.25 | −9.90 |
| top 100 | −7.05 | −7.29 |
| top 30 | **+3.13** | −7.36 |

Choosing the cutoff after seeing how each one scored turns a losing strategy
into a profitable one at the tightest setting. Choosing it on validation and
applying it once to test — the only version whose answer means anything —
produces a loss at every setting, and the gap between the two columns widens as
the selection gets tighter, which is what selection looks like from the outside.

Two further checks, in case the first looks like bad luck rather than selection.
Ranking raw signal strength instead of model confidence, with no model and no
fitted threshold anywhere, the mean outcome barely moves with selectivity at
all: 0.71 bp in the top 10% of moments, 0.92 in the top 1%, 0.52 in the top
0.1%, 0.91 in the top 0.01%. Every one of those is under a basis point against
an 11 bp cost — the strongest signals are not the more profitable ones. And the
validation-chosen cutoff jumps around across folds that share 96% of their data,
which is what fitting noise looks like from the outside.

The reason this section exists: a strategy trading a handful of times a month
cannot be distinguished from luck on this much data. Per-trade dispersion is
about 8 bp against an edge under 1 bp, so establishing the edge is real takes
thousands of trades. At three trades a month, a year of results is noise
whichever way it comes out.

## 7. Stability

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

## 8. Selection: three criteria, two of them wrong

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

## 9. Sensitivity to the fee

The fee is the obvious lever, so it is worth checking properly rather than
assuming. Costs are linear in the fee, so net profit per trade at any tier
follows from the gross edge and the trade count already measured.

Net basis points per trade, two-minute horizon, BTCUSDT, best cooldown per model:

| Model | 5.0 bp | 3.0 bp | 1.7 bp | 0.0 bp |
|---|---:|---:|---:|---:|
| logistic, cd=24 | −9.32 | −5.32 | **−2.72** | **+0.68** |
| logistic, cd=120 | −10.33 | −6.33 | −3.73 | −0.33 |
| xgboost, cd=0 | −11.64 | −7.64 | −5.04 | −1.64 |
| tcn, cd=24 | −13.22 | −9.22 | −6.62 | −3.22 |

Fee at which each would break even:

| Model | Gross/trade | Spread + slippage | Break-even fee |
|---|---:|---:|---|
| logistic, cd=24 | +1.81 | 1.13 | **0.34 bp per side** |
| logistic, cd=120 | +0.81 | 1.14 | unreachable |
| xgboost, cd=0 | −0.63 | 1.02 | unreachable |
| tcn, cd=24 | −1.75 | 1.47 | unreachable |

**The fee tier does not close the gap.** The best configuration needs 0.34 bp
per side; Binance's top volume tiers reach roughly 1.7, five times that. Even at
a *zero* fee it earns 0.68 bp per trade, because the spread and slippage cost
about 1.1 bp whatever the tier. Every other configuration loses money at any fee
including zero — its gross edge is negative, or smaller than the spread and
slippage alone.

That last column is the one worth dwelling on. A fee is negotiable; the spread
is not. Roughly a basis point of the cost survives any tier a taker could reach,
and against a gross edge under two, that alone decides the question.

This corrects an earlier reading of the same data. Taking the 5.18 bp gross
figure from the horizon sweep, the break-even fee works out at 2.08 bp per side
— reachable at a high volume tier. But that figure is the unstable one: §7
shows it falling to 2.24 bp once the sample is trimmed to the days the folds
actually use. The lower number is the one to plan against.

## 10. How often to retrain

Everything above fits a model once per fold and uses it for a whole test week.
That is an assumption, and the most favourable reading of the negative result so
far is that the model was simply stale. So the schedule was made a parameter:
train on the last *W* days, act for the next *A*, move forward, refit. All three
numbers were searched on validation and the winner applied once to test.

![Retraining schedules, XRP](../assets/retrain_xrp.png)

Every point is below the line, on both instruments and all three models. But the
shape is real and it is not noise: **more history helps up to about a fortnight
and then hurts**, and **refitting daily beats refitting weekly** at every window
length. The peak at 14 days is the trade-off stated in the methodology showing
up in the data — below it there is not enough to fit, above it the oldest days
describe a different market.

| Instrument | Model | Chosen on validation | Gross/trade | Net/trade | Trades | Positive windows |
|---|---|---|---:|---:|---:|---:|
| BTC | logistic | train21 / apply1 | +6.69 bp | **−4.72 bp** | 111 | 21% |
| BTC | xgboost | train14 / apply1 | −1.17 bp | −12.26 bp | 336 | 8% |
| BTC | tcn | train7 / apply3 | +0.37 bp | −10.80 bp | 353 | 0% |
| XRP | logistic | train14 / apply1 | +12.31 bp | **−0.44 bp** | 141 | 47% |
| XRP | xgboost | train7 / apply7 | +7.44 bp | −5.30 bp | 223 | 33% |
| XRP | tcn | train21 / apply3 | +5.81 bp | −6.82 bp | 69 | 40% |

Round trip is 11.02 bp on BTC and 12.68 bp on XRP. Schedules are chosen on the
first half of each span and applied once to the second; the ten days before the
span chose the features and are never scored.

**This is the closest the project comes to break-even.** XRP with daily
retraining earns 12.31 bp gross against a 12.68 bp cost — a shortfall of 0.37 bp
rather than the ~10 bp gap everywhere else in this document. Refitting daily on
a fortnight of history raises gross edge per trade by roughly an order of
magnitude over the fixed-model walk-forward in §4.

It is still not a profit, and the honest reading is narrower than it looks:

- **The sign is negative.** Not marginally positive, not zero. Negative.
- **141 trades.** The window-to-window spread puts the standard error near
  3.8 bp, so −0.44 bp is indistinguishable from zero *and* from −8 bp. The
  result rules nothing in.
- **Fewer than half its windows made money** — 7 of 15. The aggregate is carried
  by three good days out of fifteen.
- **The gain is real but bounded.** It comes from trading far less (141 trades
  over 25 days, against thousands in §4) on much better signals. That is the
  same lever as §5 and §6, applied harder, and it runs out here.

The linear model wins on both instruments. A boosted tree refitted daily on a
week of data is fitting noise faster than the extra freshness is worth, and the
network never had the sample size to justify itself.

## 11. Posting instead of crossing

Every number above crosses the spread. The last lever left is posting, and the
standing reason not to model it is adverse selection: a resting order fills
preferentially when the market is about to move through it. That objection is
correct, and it is measurable here — `aggTrades` gives the aggressor side of
every print and `bookTicker` gives the size at the touch, which together support
a first-order queue model.

A passive buy is posted at the best bid, waits behind the size already resting,
and fills once enough sell-aggressor volume has cleared it. Adverse selection is
then the difference between the forward move at all decision moments and the
forward move at the moments that actually filled.

![What posting costs and saves](../assets/maker_tradeoff.png)

| Instrument | Patience | Filled | Median wait | Adverse selection | Cost saved | Net |
|---|---:|---:|---:|---:|---:|---:|
| BTCUSDT | 1 s | 13% | 0.4 s | 0.79 bp | 3.52 bp | **+2.73 bp** |
| BTCUSDT | 10 s | 45% | 2.4 s | 0.88 bp | 3.52 bp | +2.64 bp |
| BTCUSDT | 60 s | 73% | 6.0 s | 1.09 bp | 3.52 bp | +2.43 bp |
| XRPUSDT | 1 s | 1.7% | 0.5 s | 0.16 bp | 5.18 bp | **+5.02 bp** |
| XRPUSDT | 10 s | 14% | 4.4 s | 2.18 bp | 5.18 bp | +3.00 bp |
| XRPUSDT | 60 s | 49% | 20.4 s | 2.84 bp | 5.18 bp | +2.34 bp |

The trade-off is exactly the one the objection predicts, and it is visible in
both directions: waiting longer fills more orders, and the extra fills are the
ones the market had to move to reach. XRPUSDT shows it most sharply — adverse
selection rises seventeen-fold, from 0.16 bp to 2.84 bp, as patience goes from
one second to a minute.

**But it never overtakes what posting saves.** Entering passively and exiting by
crossing costs 7.5 bp against 11.02 and 12.68 taker, and the saving exceeds the
adverse selection at every setting on both instruments. Net of both, posting is
worth between +2.3 and +5.0 bp per trade.

Set against §10, where daily retraining left XRPUSDT 0.37 bp short, that is more
than enough to close the gap on paper.

### Why this is a bound and not a result

The gap it would close is 0.37 bp; the margin it claims is 2.3 bp. That is a
comfortable-looking distance, and four things sit inside it:

- **The adverse selection is measured at unconditional moments**, not at the
  moments a model wants to trade. Those are precisely the moments the market is
  about to move, which is the worst case for a resting order — you are run over
  or you are missed. The figure here is a floor on the real one.
- **The fill rate destroys the sample.** XRPUSDT at ten seconds fills 14% of
  orders. The strategy that took 141 trades as a taker would take about twenty
  as a maker, and twenty trades establish nothing.
- **The queue model is optimistic.** The size ahead is taken as the touch size
  and never grows, cancellations ahead are ignored, and the order's own size is
  treated as negligible. Every one of those flatters the fill.
- **Only the entry is passive.** A position held to a horizon must be closed
  whether or not anyone comes to trade with it, so the exit still crosses. A
  fully passive strategy needs the same analysis on the way out, where being
  unable to fill is a risk rather than a missed opportunity.

What this section establishes is narrower than "maker execution works": it is
that the standard reason for dismissing it — adverse selection eats the saving —
does not hold on this data at this horizon. Whether the remaining margin
survives conditioning on a signal is a question this data can ask but the
experiment here does not answer.

## 12. How to leave a trade

Everything above holds a position for the label horizon and then leaves. That is
a default, not a decision: the entry side is tuned — features selected, a
confidence threshold swept, a retraining schedule searched — while the exit was
one constant, set equal to the horizon the label happened to use.

Six rules, searched on validation and applied once to test. Four read the price
(a fixed clock, take-profit, stop-loss, trailing stop) and two read the model's
ongoing opinion — leave when confidence in the position decays, or when the
prediction flips. The last two are the only place in this pipeline where a
prediction made *after* the entry is used for anything.

![Every way of leaving a trade](../assets/exit_policies.png)

The model is fitted once per fold and every policy scored against the same
predictions, so the table compares exits rather than fits.

### What transferred

| Instrument | Model | Chosen on validation | Test net/trade | Untouched clock | Gain |
|---|---|---|---:|---:|---:|
| BTC | logistic | hold 30 s | −7.07 | −7.58 | +0.52 |
| BTC | xgboost | hold 4 min | −11.40 | −11.15 | **−0.25** |
| BTC | tcn | take-profit 22 bp | −6.04 | −7.68 | +1.64 |
| XRP | logistic | exit on flip | −12.28 | −12.94 | +0.66 |
| XRP | xgboost | hold 30 s | −16.31 | −17.42 | +1.11 |
| XRP | tcn | hold 30 s | −14.63 | −11.17 | **−3.45** |

Four of six improved, two got worse, and the spread of outcomes is wider than
the median gain. **Choosing the exit on validation transfers weakly.** Against a
gap of seven to twelve basis points, a lever worth about half a point either way
is not the answer, and the two negative rows are the honest measure of how
reliable the positive ones are.

### What it did establish

**The label horizon is the wrong holding period.** The winner is a *shorter*
clock in three of six cases — thirty seconds against the two minutes the label
uses — and shortening the clock is the single largest improvement in the
validation table: 2.97 bp gross to 4.82. This follows from §3 rather than
contradicting it: signal decays fast, so most of what a prediction is worth is
realised early, while the cost of the round trip does not care how long the
position was open. Holding for the horizon the label was built on is an
assumption that this project made without examining, and it was costing about
two basis points a trade.

**A tight take-profit is a trap, and it is measurable.** At 4 bp, 75% of trades
win — against 54% for the untouched clock — and gross edge per trade falls from
2.97 bp to 1.27. It caps the winners and lets the losers run their course. Any
backtest reporting hit rate rather than expectancy would call this an
improvement.

**A wide stop-loss helps a little; a tight one does not.** At 22 bp, twice the
round trip, it truncates the left tail without touching much else: 4.06 bp gross
against 2.97. At 4 bp it fires on ordinary noise and the hit rate collapses to
33%.

**Trailing stops did nothing here.** They neither cap winners nor cut losers
cleanly; at 100 ms the path is mostly noise to shave against, and every setting
tested lands within a basis point of the baseline.

**Signal exits are the most interesting and the least conclusive.** Leaving when
the model's confidence decays is the second-best policy on validation (4.55 bp
against 2.97), which is what you would expect if the model knows something about
when its own call has expired. It did not survive to test on either instrument.

### A bug, and what it cost

The first run of this search produced the only positive test result in the
project: +3.37 bp per trade, BTC with the network and a stop-loss. It was wrong.
`Trade.move_bp` is the price change and `score` applies the direction — but
every path-dependent exit was storing a value that already had the direction
applied, so it was applied twice. Every short that exited early had its profit
inverted; every long was correct; the totals stayed plausible.

It is recorded here because of how it presented. It was invisible in every
summary, it survived a full test suite, and it appeared as a *good* result,
which is the direction that gets published. What caught it was checking a
number that was too good rather than a number that looked wrong. The regression
test is now four lines: the same profitable short, exited four different ways,
must be profitable each time.

## 13. What would have to change

- **Book depth.** One level is observed here because that is all any exchange
  publishes for free. Level imbalance, book slope and concentration need a
  collector recording the stream forward in time.
- **Trade thinning.** Every signal is treated as an independent round trip. A
  real system imposes a cooldown and does not hold overlapping positions,
  which raises edge per trade at the cost of trade count.
- **Maker execution.** Measured in §11 rather than assumed: worth +2.3 to
  +5.0 bp per trade after the adverse selection it causes, which is more than
  the 0.37 bp §10 was short by. The caveats there are the work still to do —
  chiefly that adverse selection was measured at unconditional moments, not at
  the moments a model wants to trade.

Things that would **not** close it, on this evidence: a longer horizon, since
the edge is horizon-invariant (§3); more features, which measurably made it
worse (§4); trade thinning, which improves edge per trade but by about one basis
point (§5); trading only the most confident signals, which is selection rather
than edge (§6); and a better fee tier, which falls short by a factor of five
(§9).

Retraining frequency (§10) is the one lever that moved the number materially —
it closed all but 0.37 bp of a 12.68 bp gap on XRP — and it still did not close
it. That it got as far as it did is the strongest argument for the two levers
above: a strategy this close to the line is one that maker execution or a real
volume tier could plausibly push across.

---

## Reading this

No result here is evidence that any strategy is or was profitable. These are
backtests computed with hindsight, over one period, in one regime, with
simplified execution — no queue position, no partial fills, no market impact,
no limit on concurrent exposure. Every one of those simplifications flatters the
result, and it is still negative.

See [`limitations.md`](limitations.md) and [`methodology.md`](methodology.md).
