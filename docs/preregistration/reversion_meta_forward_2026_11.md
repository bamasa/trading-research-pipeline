# Registration: models choosing the reversion rule's trades, on the forward fortnight (M1)

| | |
|---|---|
| **Committed** | 15 November 2026, before the benchmark's forward track opens (21 November 2026) and before any of its scored days (6 to 19 November 2026) was downloaded |
| **What it registers** | Nine strategies, their sources frozen by sha256, scored once on the forward track after that track's own read (F1, P1) is committed |
| **Searched** | **Nothing on the forward track.** The strategies were built and looked at on the two recorded tracks (February and April 2024), which are development data for this test |
| **How it changes** | Only by a dated amendment appended at the end, which cannot change a status already read |

## Why

The reversion rule pays while the index's ten-minute moves reverse and loses
when they do not ([findings](../findings.md#cross-sectional-reversion-at-ten-minutes)).
Two ways of keeping it in profit were built on the recorded tracks:

- **a model choosing which of its trades to take** (meta-labelling): the rule
  picks the side, a classifier fitted on the history's candidates says whether
  this one will clear its cost, from six inputs that describe the state the
  rule depends on (the move's strength, the index's volatility and its
  ten-minute autocorrelation over the last six hours, its trend, the traded
  coin's own move against the index, the spread);
- **the kill switch**, which stops the rule when its own paper results say the
  edge is gone (`reversion_guarded`).

The forward fortnight is the one sample no one could have fitted to.

## The family

[`plugins/builtins/meta_reversion.py`](../../src/trading_research/plugins/builtins/meta_reversion.py)
and [`guarded.py`](../../src/trading_research/plugins/builtins/guarded.py), at their defaults:

| strategy | model choosing the trades | kill switch |
|---|---|---|
| `reversion_guarded` | none | yes |
| `reversion_meta` | XGBoost | no |
| `reversion_meta_guarded` | XGBoost | yes |
| `reversion_lightgbm` | LightGBM | no |
| `reversion_logistic` | logistic regression | no |
| `reversion_transformer` | causal transformer, window of 16 candidates | no |
| `reversion_lstm` | LSTM, 16 | no |
| `reversion_gru` | GRU, 16 | no |
| `reversion_tcn` | TCN, 16 | no |

Each takes the trades whose predicted probability of a profit is in the top
half of its probabilities on the history; that half is fixed, not chosen.
Fitted on the track's fitted days (9 October to 5 November 2026), scored on
6 to 19 November 2026, by the benchmark's runner, taker at Bybit's base fee,
as the board's strategies.

The sources are frozen in
[`experiments/reversion_meta_forward.py`](../../experiments/reversion_meta_forward.py)
by sha256 (`FROZEN`); the script refuses to read if any differs, before the
forward track's own read is committed, and a second time.

## M1

For each of the nine: **passes** when its benchmark verdict on the track is
*pays the base fee* and its Holm-adjusted one-sided *p* of net > 0, over the
nine, is below 0.05. Otherwise **killed**. Reported beside, with no status:
each one's net per trade against the plain rule's on the same fortnight (from
the track's own read), and its trades.

## What was seen before this was written

On the two recorded tracks (`experiments/results/reversion_meta/`), the plain
rule made +7.0 bp a trade in February and lost in April. With XGBoost choosing
its trades: +14.3 bp in February on 2,010 trades, −30.3 bp in April on 865.
LightGBM: +12.2 in February. The other variants' numbers on those tracks are
in the same folder, written before the forward read.

## Expected outcome, stated now

**All nine killed under Holm**, whatever the state. Fourteen days of one
market-wide bet give a clustered *t* of one or two at best, and nine tests
need about three. Beside the statuses: if the fortnight's index reverts
(P1's state at or below −0.05), the model-chosen variants above the plain
rule's net per trade; if it does not, everything negative, the guarded ones
least.

## Results of M1, read once 21 November 2026

[`experiments/results/reversion_meta_forward_2026_11.json`](../../experiments/results/reversion_meta_forward_2026_11.json),
from `experiments/reversion_meta_forward.py` after the forward track's own read
was committed. All five frozen sources matched their registered sha256.

**M1: all nine killed**, as expected. Every Holm-adjusted *p* is 1.000 and no
net per trade at the base fee is above zero. The index's ten-minute
autocorrelation on the fortnight was −0.0237 (P1 *not tested*), neither
the reverting state nor clearly the flat one.

| strategy | trades | gross bp | net bp | day-clustered *t* | placebo | break-even fee, bp a side |
|---|---:|---:|---:|---:|---:|---:|
| plain `reversion` (the track's read) | 816 | +3.13 | −12.19 | −1.38 | 63% | −0.60 |
| `reversion_logistic` | 554 | +15.13 | **−0.49** | +0.02 | 100% | +5.25 |
| `reversion_meta_guarded` | 226 | +12.31 | −3.18 | −0.70 | 95% | +3.91 |
| `reversion_gru` | 561 | +11.24 | −4.08 | −0.22 | 100% | +3.46 |
| `reversion_lightgbm` | 683 | +10.35 | −5.07 | −0.57 | 95% | +2.96 |
| `reversion_lstm` | 553 | +9.48 | −6.13 | −0.46 | 100% | +2.44 |
| `reversion_tcn` | 725 | +6.05 | −9.54 | −1.09 | 84% | +0.73 |
| `reversion_meta` (XGBoost) | 683 | +5.75 | −9.68 | −0.94 | 84% | +0.66 |
| `reversion_transformer` | 701 | +5.23 | −10.23 | −1.23 | 84% | +0.39 |
| `reversion_guarded` | 159 | −13.87 | −29.48 | −1.22 | 0% | −9.24 |

Against the expectation stated beside the statuses:

- **Everything negative: right.**
- **Every model-chosen variant above the plain rule's −12.19 bp: true for all
  eight.** Their gross rose from +3.1 to between +5.2 and +15.1 bp a trade:
  the models did pick the better half of the rule's trades on days they had
  never seen. Not by enough to pay a 14 bp round trip, and with 13 days no
  difference is far from noise (best *t* +0.02).
- **"The guarded ones least": wrong for the plain switch.** `reversion_guarded`
  was the worst of all, −29.48 bp on 159 trades: the switch stopped the rule
  and restarted it at the wrong moments. With XGBoost choosing the trades the
  switch was the second best, −3.18 bp, but that is one ordering on one
  fortnight.
- The simplest model did best: logistic regression came within half a basis
  point of break-even, the transformer was the weakest of the models.
