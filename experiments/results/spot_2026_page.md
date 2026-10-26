# S25 on spot, 2026: results

Written by `experiments/spot_2026.py` from the results files of the reads; every number beside the verdict tier is a measurement.

## H (first read)

### Headline sentences

- S25's coordinates on nine print inputs, on Binance BTC/FDUSD, August to October 2026: gross edge 1.33 bp of turnover against the 10 bp taker fee a Regular account paid; break-even 1.33 bp per side; killed.
- S25 as frozen on Bybit's perpetual, unchanged on Bybit's spot book: 15.0 trades a day against 240 registered; killed.
- On Bybit's spot book, the print rule priced buys 0.34 bp and sells 0.39 bp better than the book walk.
- S25-PF on H, Binance spot BTC/FDUSD: killed; first kill K1 at -17.35 bp against 0.00 bp.
- S25-BB on H, Bybit spot BTCUSDT: killed; first kill K1 at -19.43 bp against 0.00 bp.
- S25-T on H, Bybit spot BTCUSDT: killed; first kill K1 at -19.28 bp against 0.00 bp.

### Slots and measured rows

| Row | Status | First kill | Mean | t | NW t | Median day | Trades | MDE | Placebo pct | K-nbhd median | Break-even bp/side | Every tier | Brackets | Calibration |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S25-PF | killed | K1 at -17.35 against 0.00 | -17.35 | -109.86 | -130.99 | -17.03 | 4427 | 0.49 | 98.5 | -17.44 | 1.33 | regular -17.35, spot_liquidity_provider n/a, vip1 n/a, vip2_to_vip9 n/a, vip3 -9.35, vip9 -1.95, zero_fees 2.65 | optimistic -17.26, pessimistic -23.17, stale -22.95, worse_of_two -28.02 | thinned a_buy -2.91, a_sell -3.17 |
| S25-BB | killed | K1 at -19.43 against 0.00 | -19.43 | -336.17 | -400.14 | -19.43 | 31935 | 0.52 | 100.0 | -19.54 | 0.29 | non_vip -19.43, vip1 -15.43, vip2 -14.93, vip3 -14.43, vip4 -11.43, vip5 -9.43, zero_fees 0.57 | n/a | rule buy 0.28, sell 0.20; worse_of_two buy -1.45, sell -1.08 |
| S25-T | killed | K1 at -19.28 against 0.00 | -19.28 | -117.50 | -152.26 | -19.36 | 1124 | 0.74 | 100.0 | -19.51 | 0.36 | non_vip -19.28, vip1 -15.28, vip2 -14.78, vip3 -14.28, vip4 -11.28, vip5 -9.28, zero_fees 0.72 | n/a | n/a |
| S25-PB | measured beside the family; no status | K1 at -19.08 against 0.00 | -19.08 | -188.69 | -167.48 | -19.04 | 13952 | n/a | 79.0 | n/a | 0.46 | non_vip -19.08, vip1 -15.08, vip2 -14.58, vip3 -14.08, vip4 -11.08, vip5 -9.08, zero_fees 0.92 | optimistic -19.08, pessimistic -22.27, stale -19.78, worse_of_two -23.03 | rule buy 0.34, sell 0.39; worse_of_two buy -1.82, sell -1.58 |
| S25-PFT | measured beside the family; no status | K1 at -17.71 against 0.00 | -17.71 | -137.28 | -168.83 | -17.85 | 5581 | n/a | 75.0 | n/a | 1.14 | regular -17.71, spot_liquidity_provider n/a, vip1 n/a, vip2_to_vip9 n/a, vip3 -9.71, vip9 -2.31, zero_fees 2.29 | optimistic -17.63, pessimistic -23.75, stale -22.09, worse_of_two -27.65 | thinned a_buy -2.91, a_sell -3.17 |

Thinned calibration: a_buy -2.91 bp, a_sell -3.17 bp; capacity median 0.0044.

FILL-S S25-BB, net_bp_trade: book walk -19.43, print rule -18.88.
FILL-S S25-PB, net_bp_trade: book walk -19.76, print rule -19.08.
FILL-S S25-BB, mean advantage over the book walk (bp of mid): optimistic buy 0.28, sell 0.20; pessimistic buy -1.07, sell -0.89; rule buy 0.28, sell 0.20; stale buy -0.17, sell 0.02; worse_of_two buy -1.45, sell -1.08.
FILL-S S25-PB, mean advantage over the book walk (bp of mid): optimistic buy 0.34, sell 0.39; pessimistic buy -1.45, sell -1.17; rule buy 0.34, sell 0.39; stale buy -0.04, sell -0.03; worse_of_two buy -1.82, sell -1.58.

### Edge ladder

| Step | Zero fees | Verdict tier | Cheapest dated tier |
|---|---|---|---|
| S25-T on spot books (no refit) | 0.72 | -19.28 | vip5 -9.28 |
| S25-BB (refit on spot books) | 0.57 | -19.43 | vip5 -9.43 |
| S25-PB by the book walk (prints-only inputs, same venue) | 0.24 | -19.76 | vip5 -9.76 |
| S25-PB by the print rule | 0.92 | -19.08 | vip5 -9.08 |
| S25-PF (the other venue) | 2.65 | -17.35 | vip9 -1.95 |

## After P

- S25-PF: killed
- S25-BB: killed
- S25-T: killed
