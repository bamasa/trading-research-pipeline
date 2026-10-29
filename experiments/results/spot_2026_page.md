# S25 on spot, 2026: results

Written by `experiments/spot_2026.py` from the results files of the reads; every number beside the verdict tier is a measurement.

No number at a tier other than the verdict tier, at zero fees, under a bracket or the long-only variant, with the gap-excluded days put back, of a measured row, or in the inventory's P&L may be described as the strategy working. Such numbers are measurements. In particular, no account could trade either pair at zero taker fees on any day of W, D or H as far as the sources show, so a positive gross edge is a statement about a venue that did not exist for these pairs on those days.

## H (first read)

### Headline sentences

- S25's coordinates on nine print inputs, on Binance BTC/FDUSD, August to October 2026: gross edge 1.33 bp of turnover against the 10 bp taker fee a Regular account paid; break-even 1.33 bp per side; killed.
- S25 as frozen on Bybit's perpetual, unchanged on Bybit's spot book: 15.0 trades a day against 240 registered; killed.
- On Bybit's spot book, the print rule priced buys 0.34 bp and sells 0.39 bp better than the book walk.
- S25-PF on H, Binance spot BTC/FDUSD: killed; first kill K1 at -17.35 bp against 0.00 bp.
- S25-BB on H, Bybit spot BTCUSDT: killed; first kill K1 at -19.43 bp against 0.00 bp.
- S25-T on H, Bybit spot BTCUSDT: killed; first kill K1 at -19.28 bp against 0.00 bp.

### Slots and measured rows

| Row | Status | First kill | Mean | t | NW t | Median day | Trades | MDE | Placebo pct | K-nbhd median | Break-even bp/side | Every tier, labelled | 30-day traded volume | Brackets | Calibration |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S25-PF | killed | K1 at -17.35 against 0.00 | -17.35 | -109.86 | -130.99 | -17.03 | 4427 | 0.49 | 98.5 | -17.44 | 1.33 | regular -17.35 (verdict tier); vip3 -9.35 (requires Binance VIP 3 status); vip9 -1.95 (requires Binance VIP 9 status); zero fees 2.65 (conditional on a zero-fee venue) | 2.79M FDUSD | optimistic -17.26, pessimistic -23.17, stale -22.95, worse_of_two -28.02 | thinned a_buy -2.91, a_sell -3.17 |
| S25-BB | killed | K1 at -19.43 against 0.00 | -19.43 | -336.17 | -400.14 | -19.43 | 31935 | 0.52 | 100.0 | -19.54 | 0.29 | non_vip -19.43 (verdict tier); vip1 -15.43 (requires Bybit VIP 1 status); vip2 -14.93 (requires Bybit VIP 2 status); vip3 -14.43 (requires Bybit VIP 3 status); vip4 -11.43 (requires Bybit VIP 4 status); vip5 -9.43 (requires Bybit VIP 5 status); zero fees 0.57 (conditional on a zero-fee venue) | 21.11M USDT | n/a | rule buy 0.28, sell 0.20; worse_of_two buy -1.45, sell -1.08 |
| S25-T | killed | K1 at -19.28 against 0.00 | -19.28 | -117.50 | -152.26 | -19.36 | 1124 | 0.74 | 100.0 | -19.51 | 0.36 | non_vip -19.28 (verdict tier); vip1 -15.28 (requires Bybit VIP 1 status); vip2 -14.78 (requires Bybit VIP 2 status); vip3 -14.28 (requires Bybit VIP 3 status); vip4 -11.28 (requires Bybit VIP 4 status); vip5 -9.28 (requires Bybit VIP 5 status); zero fees 0.72 (conditional on a zero-fee venue) | 0.67M USDT | n/a | n/a |
| S25-PB | measured beside the family; no status | K1 at -19.08 against 0.00 | -19.08 | -188.69 | -167.48 | -19.04 | 13952 | n/a | 79.0 | n/a | 0.46 | non_vip -19.08 (verdict tier); vip1 -15.08 (requires Bybit VIP 1 status); vip2 -14.58 (requires Bybit VIP 2 status); vip3 -14.08 (requires Bybit VIP 3 status); vip4 -11.08 (requires Bybit VIP 4 status); vip5 -9.08 (requires Bybit VIP 5 status); zero fees 0.92 (conditional on a zero-fee venue) | 9.05M USDT | optimistic -19.08, pessimistic -22.27, stale -19.78, worse_of_two -23.03 | rule buy 0.34, sell 0.39; worse_of_two buy -1.82, sell -1.58 |
| S25-PFT | measured beside the family; no status | K1 at -17.71 against 0.00 | -17.71 | -137.28 | -168.83 | -17.85 | 5581 | n/a | 75.0 | n/a | 1.14 | regular -17.71 (verdict tier); vip3 -9.71 (requires Binance VIP 3 status); vip9 -2.31 (requires Binance VIP 9 status); zero fees 2.29 (conditional on a zero-fee venue) | 3.52M FDUSD | optimistic -17.63, pessimistic -23.75, stale -22.09, worse_of_two -27.65 | thinned a_buy -2.91, a_sell -3.17 |

Thinned calibration: a_buy -2.91 bp, a_sell -3.17 bp; capacity median 0.0044.

FILL-S S25-BB, net_bp_trade: book walk -19.43, print rule -18.88.
FILL-S S25-PB, net_bp_trade: book walk -19.76, print rule -19.08.
FILL-S S25-BB, mean advantage over the book walk (bp of mid): optimistic buy 0.28, sell 0.20; pessimistic buy -1.07, sell -0.89; rule buy 0.28, sell 0.20; stale buy -0.17, sell 0.02; worse_of_two buy -1.45, sell -1.08.
FILL-S S25-PB, mean advantage over the book walk (bp of mid): optimistic buy 0.34, sell 0.39; pessimistic buy -1.45, sell -1.17; rule buy 0.34, sell 0.39; stale buy -0.04, sell -0.03; worse_of_two buy -1.82, sell -1.58.

### Tiers and the 30-day volume

Every tier but the verdict tier (Binance's Regular, Bybit's non-VIP, the tier an account holds without 30-day volume, coin holdings or programme membership) is a measurement, labelled with the status it requires. A VIP status depends on 30-day volume and coin holdings at thresholds the registration does not transcribe, so no tier other than the verdict tier is said to be within a slot's reach; a zero fee at a tier would read "conditional on <tier> eligibility". The result at zero fees is conditional on a zero-fee venue.

Not dated for H: Binance BTC/FDUSD's spot_liquidity_provider, vip1 and vip2_to_vip9; no sourced interval lists them on any day of H, so no value is given at them.

Each row's 30-day traded volume is 30 times its mean daily turnover on H, in the pair's quote currency: a measurement, which never moves the verdict tier.

### Edge ladder

| Step | Zero fees (conditional on a zero-fee venue) | Verdict tier | Cheapest dated tier, labelled |
|---|---|---|---|
| S25-T on spot books (no refit) | 0.72 | -19.28 | vip5 -9.28 (requires Bybit VIP 5 status) |
| S25-BB (refit on spot books) | 0.57 | -19.43 | vip5 -9.43 (requires Bybit VIP 5 status) |
| S25-PB by the book walk (prints-only inputs, same venue) | 0.24 | -19.76 | vip5 -9.76 (requires Bybit VIP 5 status) |
| S25-PB by the print rule | 0.92 | -19.08 | vip5 -9.08 (requires Bybit VIP 5 status) |
| S25-PF (the other venue) | 2.65 | -17.35 | vip9 -1.95 (requires Binance VIP 9 status) |

## After P

- S25-PF: killed
- S25-BB: killed
- S25-T: killed
