# The reversion rule with a model choosing its trades, on the recorded tracks

Written by `experiments/reversion_meta_board.py` from the bench cards in this folder.
Net per trade in bp after Bybit's base taker fee, slippage and the spread; *t* clustered
by day; the last column sums net bp over both tracks' trades. These are the tracks the
variants were built on: a measurement, not a test. The test is
[M1](../../../docs/preregistration/reversion_meta_forward_2026_11.md), on the forward fortnight.

| who chooses the trades | Feb trades | Feb net | Feb *t* | Apr trades | Apr net | Apr *t* | summed net, bp |
|---|--:|--:|--:|--:|--:|--:|--:|
| nobody: the plain rule | 3,351 | +7.0 | +1.21 | 1,233 | -47.0 | -1.70 | -34,436 |
| nobody, kill switch (`reversion_guarded`) | 2,347 | +6.5 | +1.04 | 52 | -140.3 | - | +8,056 |
| logistic regression (`reversion_logistic`) | 1,663 | +18.1 | +1.92 | 936 | -40.5 | -1.47 | -7,727 |
| XGBoost (`reversion_meta`) | 2,010 | +14.3 | +1.72 | 865 | -30.3 | -1.00 | +2,518 |
| LightGBM (`reversion_lightgbm`) | 2,013 | +12.2 | +1.86 | 897 | -35.2 | -1.33 | -7,119 |
| transformer (`reversion_transformer`) | 2,107 | +10.6 | +1.50 | 859 | -51.9 | -1.53 | -22,156 |
| LSTM (`reversion_lstm`) | 1,662 | +9.3 | +1.67 | 877 | -51.3 | -1.62 | -29,551 |
| GRU (`reversion_gru`) | 1,891 | +5.9 | +0.99 | 889 | -52.3 | -1.66 | -35,333 |
| TCN (`reversion_tcn`) | 2,230 | +7.2 | +0.73 | 918 | -54.2 | -1.60 | -33,617 |
| XGBoost + kill switch (`reversion_meta_guarded`) | 817 | +18.2 | +0.50 | 55 | -90.2 | -1.10 | +9,927 |

- **Simple models beat the networks here.** On six inputs and a month of candidates,
  logistic regression and the boosters raise February's net from +7.0 to +12 to +18 bp
  a trade; the transformer, LSTM, GRU and TCN add little and lose the most in April.
- **No model saves April.** Every variant without the kill switch loses there. The
  boosters make each trade less bad (-30 to -35 bp against the rule's -47) and take a
  quarter fewer, but nothing in the six inputs at the hour of the trade says the state
  has changed.
- **The kill switch does.** It stops trading within days of the break: 52 and 55
  April trades instead of about 900. XGBoost choosing the trades and the switch
  stopping them is the best sum over both tracks (+9,927 bp against the switch alone's
  +8,056), on a third of the February trades.
- No February number is told apart from chance (*t* at most +1.92).
