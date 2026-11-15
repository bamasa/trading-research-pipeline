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
