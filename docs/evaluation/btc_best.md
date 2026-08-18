# Scorecard: btc_best

## Result

| Measure | Value |
|---|---:|
| trades | 703 |
| trades per day | 20.09 |
| gross per trade bp | 4.265 |
| net per trade bp | -7.028 |
| net bp | -4,941 |
| net bp per day | -141.2 |
| hit rate | 0.3514 |
| profit factor | 0.5012 |
| max drawdown bp | 4,942 |
| drawdown trades | 702 |
| return over drawdown | -0.9996 |
| days | 35 |

Net is **-7.03 bp per trade** against a round trip of 11.02 bp. Gross figures elsewhere in this document are before that cost and are not profits.

## Dispersion

- periods: 5
- positive: 0 of 5
- best -358.0 bp, worst -1,422.5 bp
- standard deviation across periods: 405.1 bp
- per-trade dispersion: 33.28 bp

## How it was produced

| | |
|---|---|
| cost bp per trade | 11.02 |
| source | experiments/results/best_trades_BTCUSDT.csv |
| configurations tried | 6 |
| trades for two sigma | 90 |
| trades available | 703 |

## Not established by this scorecard

- Whether the features leak — this is a property of the code, not of the trade log.
- Whether the reported configuration was chosen before or after seeing this period.
- Whether fills are achievable at size: no queue position, partial fills or impact are modelled.
- Whether the result holds outside the period and instrument measured here.
