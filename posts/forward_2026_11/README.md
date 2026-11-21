# A fortnight that had not happened yet

A post: on 5 November 2026 the benchmark froze twelve strategies, wrote down
what would count as a pass, and waited for 6 to 19 November to happen. A gate
and nine model variants were registered during the fortnight, before any of its
data was downloaded. On 21 November the repository read those fourteen days
once.
Nothing passed. Every number is in three files written by the read:
[`bench_forward_2026_11.json`](../../experiments/results/bench_forward_2026_11.json),
[`regime_gate_G2.json`](../../experiments/results/regime_gate_G2.json) and
[`reversion_meta_forward_2026_11.json`](../../experiments/results/reversion_meta_forward_2026_11.json);
the tests are registered in
[`bench_forward_2026_11.md`](../../docs/preregistration/bench_forward_2026_11.md),
[`regime_gate_2026_11.md`](../../docs/preregistration/regime_gate_2026_11.md) and
[`reversion_meta_forward_2026_11.md`](../../docs/preregistration/reversion_meta_forward_2026_11.md).
[`tests/test_forward_post.py`](../../tests/test_forward_post.py) fails if this
page and those files disagree.

## Why a forward test, in one paragraph

Every backtest in this repository was run on days its author had already seen.
Choosing the span, the parameters, even which idea to try next is fitting, and
no statistic undoes it. A forward test is the one check that does: the rules
are committed in public before the days exist, so nothing in them can have
been fitted to those days. The commit timestamps on GitHub are the proof.

## What was frozen before the read

1. **Twelve strategies** (5 November): the benchmark's whole version-1 board,
   from book imbalance to XGBoost, each frozen by the sha256 of its source.
   Pass: net per trade above zero at Bybit's base taker fee (5.5 bp a side),
   day-clustered *t* of at least 2, gross above 95% of nineteen time-shifted
   placebos, Holm *p* below 0.05 over the twelve. The stated prior: none
   passes.
2. **A prediction** (5 November): the ten-minute cross-sectional reversion, this
   project's one conditional edge, makes money when the market's index reverts
   (ten-minute autocorrelation −0.05 or below) and does not when it is flat
   (above −0.02). Between the two, no prediction.
3. **A gate** (11 November, before any scored day was downloaded): pause the reversion when a structural-break
   detector says the market stopped reverting, resume when it says reverting
   is back. It had already failed on ten historical episodes.
4. **Nine model variants** (15 November, the same): the same reversion with a model
   choosing which of its trades to take (logistic regression, XGBoost,
   LightGBM, a transformer, LSTM, GRU, TCN) or with a kill switch. Expected: all
   killed under Holm.

Fitted on 9 October to 5 November 2026, scored on 6 to 19 November, 26 Bybit
perpetuals, top of book every five seconds. All 26 had all 14 scored days.

## What the fortnight did

| | registered | result |
|---|---|---|
| The twelve strategies (F1) | pass if one pays the fee | **all twelve killed**; net from −9.4 to −16.4 bp a trade |
| The reversion's prediction (P1) | right or wrong if the state is clear | **not tested**: the state was −0.0237, between the thresholds |
| The break-detector gate (G2) | pass if it beats nineteen random pauses | **killed**: it never paused, gain 0.00% |
| Nine model variants (M1) | pass if one pays the fee | **all nine killed**; best −0.49 bp a trade |

**The twelve.** Six of them found something real: gross above their placebos,
up to +4.5 bp a trade (`xgboost`). The round trip costs about 14 bp. The best
net was −9.4 bp a trade on 1,389 trades; `order_flow` lost −13.0 bp on 136,485
trades, a *t* of −201. Every Holm *p* was 1.000.

**The prediction.** The market sat at −0.0237, just past the −0.02 line,
inside the zone where nothing was promised. The reversion made 816
trades at −12.19 bp each. That is what the flat state predicts, but the
registration said in advance that this zone tests nothing, so it is not
counted as right.

**The gate.** The detector raised one alarm in the whole fortnight, on
9 November: *the market is reverting more*. That means "keep trading" to a rule
that was already trading. The gate never paused, kept all of the rule's
−3.83% loss on capital, and its nineteen placebos were the same. Killed by the
registered rule, and the honest reading is narrower: the fortnight never asked
the gate to decide.

**The models.** This is the most interesting failure. Every one of the eight
model-chosen variants lifted the rule's gross, from +3.1 to between +5.2 and
+15.1 bp a trade, on days the models had never seen. So they did pick the
better trades. None picked well enough to pay the fee:

| variant | trades | gross bp | net bp |
|---|---:|---:|---:|
| plain rule | 816 | +3.1 | −12.2 |
| logistic regression | 554 | +15.1 | **−0.5** |
| XGBoost + kill switch | 226 | +12.3 | −3.2 |
| GRU | 561 | +11.2 | −4.1 |
| LightGBM | 683 | +10.3 | −5.1 |
| LSTM | 553 | +9.5 | −6.1 |
| TCN | 725 | +6.0 | −9.5 |
| XGBoost | 683 | +5.8 | −9.7 |
| transformer | 701 | +5.2 | −10.2 |
| kill switch alone | 159 | −13.9 | −29.5 |

The simplest model came within half a basis point of break-even; the
transformer was the weakest model. The kill switch on its own was the worst
thing on the board: it stopped and restarted the rule at the wrong moments.

## How to read it

- **Failures are the result.** Twenty-one strategies and a gate were frozen;
  none passed; the registrations expected that for F1, G2 and M1 in advance.
- **Only the twelve were frozen before the days existed.** The gate and the
  models were registered on 11 and 15 November, while the fortnight was
  running and before any of it was downloaded. The market's state by then was
  not known to the repository, but it was not unknowable. A
  forward test that confirms the prior is doing its job.
- **Fourteen days is short.** With 13 or 14 trading days, a difference of a few
  basis points is noise: the best model's day-clustered *t* was +0.02. "Killed"
  means it did not pay on this fortnight, not that it can never work.
- **The market did not pick a side.** The reversion's state fell in the one zone
  where nothing was predicted, so the conditional edge was neither confirmed
  nor refuted. The next sealed fortnight is already registered.
- **Fees decide.** Most of these strategies see something, gross. A taker at
  the base fee pays about 14 bp a round trip, and nothing here makes that much
  on a trade.

## LinkedIn text

> **I froze 21 trading strategies before reading the fortnight they would trade. All 21 failed.**
>
> On 5 November, before the days existed, I committed in public twelve strategies for 26 crypto perpetuals, a rule for what counts as a pass, and a prediction. During the fortnight, before downloading any of it, I added a market-state gate and nine variants where a model (logistic regression up to a transformer) chooses which trades to take. On 21 November I read 6–19 November once.
>
> Result, after Bybit's base fee: all twelve strategies lost, from −9.4 to −16.4 bp a trade. Six of them had a real gross edge, up to +4.5 bp, against a 14 bp round trip. The gate never paused. The prediction was not tested: the market landed in the exact zone where I had said it would tell me nothing.
>
> The most interesting part: every model lifted the rule's gross on days it had never seen, from +3.1 to as much as +15.1 bp a trade. The simplest one, logistic regression, got to −0.49 bp. Close, and still a loss.
>
> Fourteen days is short, and "killed" means killed on this fortnight. But this is the only kind of test in the project where nothing could have been fitted to the answer, and I wrote down that I expected this before reading it.
>
> Code, registrations and every number: github.com/bamasa/trading-research-pipeline/tree/main/posts/forward_2026_11
