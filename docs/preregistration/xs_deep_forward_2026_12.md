# Registration: machine learning and deep learning on the cross-section of 26 perpetuals, on a December fortnight (X1, X2, X3)

| | |
|---|---|
| **Committed** | 20 November 2026, before any of the models below was fitted or run on any data, and before any day of the test block (30 November to 13 December 2026) had happened |
| **What it registers** | Six ways of ranking 26 USDT perpetuals every ten minutes, one rule and five fitted models, their settings fixed here; a development replay on 2024 data, reported as a measurement; and three statuses read once on the test block |
| **Searched** | **Nothing.** Every setting below is written down before the first fit. No setting is compared, tuned or chosen on any data |
| **How it changes** | Only by a dated amendment appended at the end, which cannot change a status already read |

## Why

Every model this repository has fitted to the 26 perpetuals looked at one
instrument at a time: the benchmark's models predicting its next move, and the
nine models (four of them networks) choosing the reversion rule's trades
([`reversion_meta`](../../experiments/results/reversion_meta/README.md)). None
paid its taker cost, and the networks added least. None of them saw the other
25 instruments at the moment of the decision, though the one effect found here,
[ten-minute reversion](../findings.md#cross-sectional-reversion-at-ten-minutes),
is a property of the cross-section. This study asks two questions apart:

- **does any ranking pay its cost** on a fortnight nobody has seen (X1), and
- **does a network that reads all 26 instruments at once rank better** than a
  linear model (X2) and than the same network reading one instrument (X3).

The second question is about the ranking's quality, before costs. It can be
answered even if nothing pays.

## Data and target

Bybit's public top-of-book for the 26 instruments of the recorded tracks
([`bench/tracks.py`](../../src/trading_research/bench/tracks.py), `FORWARD_SYMBOLS`
on the test block), reduced to one-minute bars: the last snapshot of each
minute gives the mid price, the spread in bp and the touch imbalance
`(bid size - ask size) / (bid size + ask size)`.

- **Decision times:** every minute whose index is a multiple of ten.
- **Entry and exit:** at the mid one minute after the decision and eleven
  minutes after it. The minute between the decision and the entry keeps the
  bid-ask bounce of the last bar out of the target.
- **Target:** the instrument's log return from entry to exit, in bp, minus the
  mean of the 26 over the same minutes. Models are fitted on its cross-sectional
  z-score at each time, clipped at 3.
- **Training rows:** every fifth minute of the training window whose exit falls
  inside it.

## The six rankers

| name | kind | reads |
|---|---|---|
| `reversal` | rule, not fitted | minus the instrument's relative return over the last ten minutes |
| `ridge` | linear, ML | twelve features (below), standardised; penalty 10 |
| `lightgbm` | gradient boosting, ML | the same twelve; 300 rounds, learning rate 0.03, 15 leaves, at least 200 rows a leaf, 80% of features a tree |
| `mlp` | network, DL | the same twelve; two hidden layers of 64, ReLU, dropout 0.1 |
| `tcn` | network, DL | the last 60 minutes of four series per instrument; causal convolutions, 32 channels, kernel 3, dilations 1, 2, 4, 8, residual; one instrument at a time |
| `xs_transformer` | network, DL | the same encoder for each instrument, then two layers of self-attention across the 26 instruments at the same time (4 heads, width 32, no order among them), so each instrument's score depends on the others |

The twelve features: the relative return over 1, 5, 10, 30 and 60 minutes; the
index's return over 10 and 60 minutes; the standard deviation of the
instrument's one-minute relative return and of the index's one-minute return,
over 60 minutes; the spread; the imbalance; its mean over 10 minutes. The four
series: the one-minute relative return, the index's one-minute return, the
spread and the imbalance. Inputs are standardised with the training window's
means and deviations and clipped at 5.

Every network: mean squared error, Adam at learning rate 0.001 and weight decay
0.0001, 6 epochs, batches of 64 decision times, seed 0, on the CPU. The
networks are fitted once a window; no early stopping, no ensemble.

## The trade and its cost

At each decision time each ranker goes long the three instruments it ranks
highest and short the three it ranks lowest, equal notional, entering and
exiting as above. A leg's net is its side times its log return, minus Bybit's
base taker fee (5.5 bp) twice, minus half the spread at entry and half at exit.
This is the benchmark's base fee; the spread is the recorded one.

## Blocks

- **Development:** the recorded spans, 1 February to 10 March and 12 March to
  20 April 2024. They have been read by earlier studies; here they are a
  walk-forward replay and a measurement: fitted on 14 days, scored on the next
  7, moved forward 7 days, within each span. Reported with no status.
- **Test:** fitted on 16 to 29 November 2026, scored on **30 November to
  13 December 2026**. Downloaded once its last day's archive is published, not
  before 15 December 2026. The fitted days include the benchmark's forward
  fortnight, which is fetched only after that track's own read is committed.
  An instrument with a missing day is dropped and named; the test is read with
  at least 20.

## X1, X2, X3

The days are the observations: a day's number is the mean over its legs (X1)
or its decision times (X2, X3). One-sided *t*-tests over the 14 days.

- **X1, a ranking pays.** For each of the six: **passes** when its mean daily
  net per leg is positive and its Holm-adjusted *p*, over the six, is below
  0.05. Otherwise **killed**.
- **X2, deep learning ranks better than linear.** The information coefficient
  of a decision time is the Spearman correlation across instruments of the
  ranking and the target. **Passes** when `xs_transformer`'s mean daily
  coefficient minus `ridge`'s is positive with Holm-adjusted *p* below 0.05,
  over X2 and X3. Otherwise **killed**.
- **X3, reading the cross-section helps.** The same, `xs_transformer` minus `tcn`.

Reported beside, with no status: each ranker's gross per leg, its mean daily
coefficient, its net at no fee, the correlation of the rankers' scores, and the
reversion rule's state (the index's ten-minute autocorrelation) over the
fortnight.

## Expected outcome, stated now

**X1: all six killed.** Six legs every ten minutes pay about 13 bp each in fee
and spread; the reversion effect measured here was about 20 bp gross on its
strongest trades, not on three a side every ten minutes. **X2 and X3: killed,**
uncertain. Fourteen days resolve a difference in coefficient of about 0.02 at
best, and the networks did not beat simpler models on these instruments
before. If the reversion state holds in the fortnight, every ranker's
coefficient should be positive and `reversal`'s among the highest.

The sources are frozen by sha256 in `experiments/xs_deep_forward.py` by a code
freeze amendment committed before 30 November 2026.
