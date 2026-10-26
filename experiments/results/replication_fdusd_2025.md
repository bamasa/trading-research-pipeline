# Replication: the 2025 BTC/FDUSD result at two fees

**Exploratory, not a blind test.** This is a descriptive replication, on public
data, of the author's 2025 research on Binance spot BTC/FDUSD. That research
covered these dates, so no day here is held out, and nothing
here is a registered result: no registration is changed, and the spot study's
blocks, statuses and verdicts are untouched. Its one question is where the 2025
profit on this pair came from, asked by running the same decisions at two fees.

## What was run

- **Data**: Binance spot BTCFDUSD daily aggTrades from data.binance.vision,
  179 days (2024-11-01 to 2025-06-30 without the days between the two spans), each
  verified against its published `.CHECKSUM` (`replication_fdusd_2025_fetch.csv`)
  and parsed by the spot study's reader.
- **Recipe**: the spot study's S25-PF code path, reused by import: S25's nine
  print members on the 100 ms print grid, the print label at a 0 bp label fee,
  `gbm` at the repository's defaults (seed 42) on every tenth row, hold 50 s
  closed by time, the hold-first rule at 240 trades a day with θ set on the
  window's threshold days, the spot folds' purges (the hold plus 61 minutes,
  and a training label that reads past the threshold days dropped), one clip
  of 0.010 BTC, taker in and taker out by the side-correct print rule, no
  entry before 00:01 or after 23:58 UTC, flat at 23:59.
- **Windows**: A is the author's 2025 split; B is the same recipe on May and
  June 2025.
- **Fees**, from the spot study's registered fee table, whose BTC/FDUSD interval
  from 2024-04-25 06:00 to 2026-01-29 covers every day here: (a) zero maker and
  zero taker, as the author's 2025 backtest charged on this pair; (b) the
  Regular account, 0 maker and 10 bp taker; (c) VIP 2 to 9, 0 maker and 0 taker
  in 2025, which is the same as (a), so (a) and (c) are one table. VIP 2 then
  required at least 5M USD of 30-day spot volume and 100 BNB held.
- **Long-only**: every short decision dropped before the one-position rule.

Net in quote is in FDUSD, a dollar stablecoin, per 0.010 BTC clip. "Pooled" is
the window's net over its entry notional (every trade weighted by its
notional); "mean of days" and "day t" are over the days with a trade and are
descriptive only.

## The windows and their fits

| Window | Training days | Thresholds on | Test days | Training rows | Long / hold / short labels | θ long / θ short |
|---|---|---|---|---|---|---|
| A | 2024-11-05 to 2025-01-14 | 2025-01-15 to 2025-01-21 | 2025-02-04 to 2025-02-26 | 2425362 | 1176819 / 69331 / 1179212 | 0.5205 / 0.4139 |
| B | 2025-05-01 to 2025-05-31 | 2025-06-01 to 2025-06-07 | 2025-06-08 to 2025-06-30 | 842676 | 414948 / 23322 / 404406 | 0.4432 / 0.4416 |

Test days excluded by the spot loader's rules: A: none; B: none.

## Results at zero fees: 0 maker, 0 taker (the 2025 backtest's charge; VIP 2-9 in 2025)

| Window | Variant | Test days | Trades a day | Net bp a trade (pooled) | Net FDUSD a trade (0.010 BTC) | Net FDUSD a day | Days positive | Mean of days, bp | Day t | Gross bp of turnover (break-even fee a side) |
|---|---|---|---|---|---|---|---|---|---|---|
| A | two-sided | 23 | 9.4 | -1.11 | -0.10 | -1.0 | 48% | -0.90 | -1.17 | -0.55 |
| A | long-only | 23 | 2.9 | -3.02 | -0.28 | -0.8 | 53% | -0.31 | -0.15 | -1.51 |
| B | two-sided | 23 | 74.2 | -0.08 | -0.01 | -0.7 | 52% | -0.06 | -0.32 | -0.04 |
| B | long-only | 23 | 53.4 | -0.29 | -0.03 | -1.6 | 48% | -0.39 | -2.08 | -0.14 |
| A+B | two-sided | 46 | 41.8 | -0.19 | -0.02 | -0.8 | 50% | -0.46 | -1.21 | -0.09 |
| A+B | long-only | 46 | 28.2 | -0.41 | -0.04 | -1.2 | 50% | -0.36 | -0.45 | -0.21 |

## Results at the Regular account: 0 maker, 10 bp taker

| Window | Variant | Test days | Trades a day | Net bp a trade (pooled) | Net FDUSD a trade (0.010 BTC) | Net FDUSD a day | Days positive | Mean of days, bp | Day t | Gross bp of turnover (break-even fee a side) |
|---|---|---|---|---|---|---|---|---|---|---|
| A | two-sided | 23 | 9.4 | -21.11 | -1.97 | -18.6 | 0% | -20.90 | -27.30 | -0.55 |
| A | long-only | 23 | 2.9 | -23.02 | -2.14 | -6.2 | 0% | -20.31 | -9.84 | -1.51 |
| B | two-sided | 23 | 74.2 | -20.08 | -2.13 | -158.0 | 0% | -20.06 | -104.97 | -0.04 |
| B | long-only | 23 | 53.4 | -20.29 | -2.16 | -115.2 | 0% | -20.39 | -108.18 | -0.14 |
| A+B | two-sided | 46 | 41.8 | -20.19 | -2.11 | -88.3 | 0% | -20.46 | -53.92 | -0.09 |
| A+B | long-only | 46 | 28.2 | -20.41 | -2.16 | -60.7 | 0% | -20.36 | -25.26 | -0.21 |

## Rate and signal (two-sided, the same at both fees)

| Window | Test days | Decisions a day | Trades a day | Long trades | Mean rank IC | Gross bp a trade | BTC's mean move a day, bp |
|---|---|---|---|---|---|---|---|
| A | 23 | 66.4 | 9.4 | 29% | 0.0053 | -1.11 | -79 |
| B | 23 | 786.7 | 74.2 | 62% | 0.0042 | -0.08 | 8 |
| A+B | 46 | 426.5 | 41.8 | 58% | 0.0048 | -0.19 | -35 |

## What it shows

Over both windows the result is positive at neither fee. At zero fees it is -1.11 bp a trade on window A and -0.08 bp on B (-0.19 bp over both, -0.020 FDUSD a 0.010 BTC trade, day t -1.17 and -0.32); at the Regular account's 10 bp taker fee it is -21.11 and -20.08 bp, with 0% of days positive. The trades are the same in both tables; only the fee differs. A taker round trip pays the fee on both fills, so it is positive only when its gross edge per side exceeds the fee per side: that edge is the break-even fee, here -0.55 bp of turnover on A and -0.04 bp on B (-0.09 bp over both). It is not above zero, so no fee, zero included, makes this recipe positive on these days, and the 10 bp taker fee adds about 20 bp of entry notional a round trip, which is the whole of the loss at the Regular account. The model has little to act on: its mean rank IC on the test days is 0.0053 on A and 0.0042 on B, and its thresholds, set for 240 decisions a day on the threshold days, give 66 and 787 a day on the test days, of which the one-position rule keeps 9.4 and 74.2 trades. The long-only variant nets -3.02 and -0.29 bp a trade at zero fees, while BTC moved -79 and 8 bp a test day on average. So on these windows the print-only recipe does not reproduce a profit, even at the zero fee the 2025 backtest charged. What the two tables do show is the size of the fee: at the Regular account's fee a taker recipe needs a gross edge above 20 bp a round trip, several times the gross edges measured on this pair here and on the spot study's 2026 days (2.65 bp a trade for S25-PF at zero fees on H), so a profit of this kind on BTC/FDUSD needed the zero fee. S25 itself reads 31 book features, which the public archive of this pair does not hold, so this is S25-PF's recipe, not S25's.

## Window A, day by day (two-sided)

| Day | Trades | Net bp a trade, zero fees | Net FDUSD a trade, zero fees | Net bp a trade, 10 bp taker | Net FDUSD a trade, 10 bp taker | BTC's move, bp |
|---|---|---|---|---|---|---|
| 2025-02-04 | 22 | 0.90 | 0.089 | -19.10 | -1.889 | -352 |
| 2025-02-05 | 5 | -0.52 | -0.051 | -20.52 | -2.008 | -118 |
| 2025-02-06 | 2 | -0.13 | -0.013 | -20.13 | -1.950 | -7 |
| 2025-02-07 | 16 | 1.96 | 0.193 | -18.03 | -1.770 | -2 |
| 2025-02-08 | 0 | n/a | n/a | n/a | n/a | -7 |
| 2025-02-09 | 7 | 2.83 | 0.273 | -17.17 | -1.658 | 2 |
| 2025-02-10 | 5 | 0.80 | 0.078 | -19.20 | -1.868 | 99 |
| 2025-02-11 | 3 | -0.67 | -0.066 | -20.68 | -2.022 | -167 |
| 2025-02-12 | 16 | -8.08 | -0.773 | -28.07 | -2.686 | 212 |
| 2025-02-13 | 9 | 3.42 | 0.328 | -16.58 | -1.590 | -124 |
| 2025-02-14 | 4 | 1.39 | 0.135 | -18.62 | -1.816 | 90 |
| 2025-02-15 | 2 | -6.41 | -0.626 | -26.42 | -2.580 | 9 |
| 2025-02-16 | 0 | n/a | n/a | n/a | n/a | -146 |
| 2025-02-17 | 2 | 1.27 | 0.122 | -18.73 | -1.800 | -33 |
| 2025-02-18 | 3 | -7.35 | -0.698 | -27.35 | -2.599 | -16 |
| 2025-02-19 | 5 | -4.78 | -0.462 | -24.79 | -2.393 | 103 |
| 2025-02-20 | 2 | -1.68 | -0.165 | -21.69 | -2.118 | 173 |
| 2025-02-21 | 16 | -1.76 | -0.172 | -21.76 | -2.130 | -208 |
| 2025-02-22 | 4 | 0.23 | 0.022 | -19.77 | -1.913 | 30 |
| 2025-02-23 | 7 | 0.27 | 0.026 | -19.73 | -1.896 | -28 |
| 2025-02-24 | 12 | 4.80 | 0.452 | -15.20 | -1.430 | -490 |
| 2025-02-25 | 36 | -0.78 | -0.069 | -20.78 | -1.834 | -326 |
| 2025-02-26 | 39 | -4.57 | -0.388 | -24.58 | -2.084 | -501 |

## Window B, day by day (two-sided)

| Day | Trades | Net bp a trade, zero fees | Net FDUSD a trade, zero fees | Net bp a trade, 10 bp taker | Net FDUSD a trade, 10 bp taker | BTC's move, bp |
|---|---|---|---|---|---|---|
| 2025-06-08 | 17 | 2.19 | 0.231 | -17.81 | -1.884 | 14 |
| 2025-06-09 | 98 | 0.07 | 0.007 | -19.93 | -2.158 | 427 |
| 2025-06-10 | 90 | -0.69 | -0.076 | -20.69 | -2.268 | 0 |
| 2025-06-11 | 199 | 0.14 | 0.015 | -19.86 | -2.174 | -145 |
| 2025-06-12 | 157 | -0.04 | -0.004 | -20.04 | -2.159 | -270 |
| 2025-06-13 | 109 | 1.04 | 0.109 | -18.96 | -1.982 | 38 |
| 2025-06-14 | 35 | -0.73 | -0.077 | -20.73 | -2.180 | -61 |
| 2025-06-15 | 47 | 0.43 | 0.045 | -19.57 | -2.066 | 14 |
| 2025-06-16 | 102 | 1.52 | 0.164 | -18.48 | -1.985 | 115 |
| 2025-06-17 | 148 | 0.25 | 0.026 | -19.75 | -2.088 | -207 |
| 2025-06-18 | 96 | -1.07 | -0.112 | -21.07 | -2.207 | 32 |
| 2025-06-19 | 25 | -0.54 | -0.057 | -20.54 | -2.154 | -23 |
| 2025-06-20 | 53 | -1.92 | -0.201 | -21.92 | -2.294 | -128 |
| 2025-06-21 | 44 | -0.52 | -0.054 | -20.53 | -2.107 | -111 |
| 2025-06-22 | 138 | -0.35 | -0.036 | -20.35 | -2.053 | -117 |
| 2025-06-23 | 142 | -1.14 | -0.117 | -21.14 | -2.171 | 424 |
| 2025-06-24 | 40 | -0.72 | -0.076 | -20.72 | -2.186 | 76 |
| 2025-06-25 | 49 | 0.26 | 0.028 | -19.74 | -2.119 | 116 |
| 2025-06-26 | 34 | 0.26 | 0.028 | -19.74 | -2.125 | -32 |
| 2025-06-27 | 25 | 0.83 | 0.088 | -19.17 | -2.055 | 11 |
| 2025-06-28 | 9 | 0.17 | 0.019 | -19.83 | -2.133 | 22 |
| 2025-06-29 | 23 | 0.12 | 0.013 | -19.88 | -2.150 | 97 |
| 2025-06-30 | 26 | -0.94 | -0.101 | -20.94 | -2.257 | -112 |

Every run's per-day row, both variants at both fees, with its counters, is in `replication_fdusd_2025_days.csv`.
