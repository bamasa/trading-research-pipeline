# S25-PF as a maker on Binance BTC/FDUSD, 2026: results

Written by `experiments/spot_maker_2026.py` from the results files of the reads; registered in `docs/preregistration/spot_maker_2026.md`. The verdict of every slot is `net_bp_fill`, the day's net over its turnover in bp, averaged over the days with a fill, at the fees a Regular account was charged on each day: 0 bp on a resting order's fill and 10 bp on the 23:59 taker flatten. Resting orders fill only when a print trades through their price (the queue ahead taken as exhausted). Every other number is a measurement; the zero-fee figures are references beside the fee-inclusive ones, never a result.

## Summary

Added by hand after the read; every number below is copied from
`experiments/results/spot_maker_2026_H.json` and the tables under it.

- **H, 2026-08-08 to 2026-10-21, 75 of 75 days usable, the first and only
  read.** Every slot is **killed by K1** at the fees a Regular account was
  charged (maker 0 bp, the 23:59 taker flatten 10 bp). No slot is a candidate,
  so P is not read and these statuses are final.
- **Every verdict carries the label "uncalibrated fill rule".** On Bybit spot
  BTCUSDT's book on D's days the three fill rules ranked as they should on
  every counted day, but the prints rule's making part fell inside the book's
  trade-through to touch bracket on 22% (MK-S), 64% (MK-Q) and 58% (MK-SQ) of
  the days, under the registered 80%.

| Slot | Net bp per fill (verdict) | Net bp per round trip | Gross bp per fill (zero-fee reference) | Making part, bp per fill | Fills a day (maker / taker) | Days positive | First kill |
|---|---|---|---|---|---|---|---|
| MK-S | −2.35 | −4.70 (mean of days) | −2.35 | −1.62 | 426.6 (426.6 / 0.03) | 3 of 75 | K1 at −2.35 against 0 |
| MK-Q | −2.88 | no round trips | −2.87 | −1.21 | 4,360.6 (4,359.8 / 0.8) | 0 of 75 | K1 at −2.88 against 0 |
| MK-SQ | −2.79 | no round trips | −2.77 | −1.31 | 6,148.3 (6,147.4 / 1.0) | 0 of 75 | K1 at −2.79 against 0 |

- **The loss is in the fill price, not the fee.** The maker fee is 0, and the
  flattens were few (2, 60 and 72 over the block), so the net and the gross
  differ by at most 0.02 bp per fill. The 5 s markout of the maker fills is
  negative for every slot (K-dir fires on all three), and it worsens with the
  horizon (1 s, 5 s, 30 s: MK-S −0.48, −1.62, −2.68; MK-Q −0.69, −1.21, −2.37;
  MK-SQ −0.72, −1.31, −2.41 bp per fill): a resting order filled only when
  the market trades through it is filled against the move.
- **The predictions registered in advance**: MK-S was predicted killed by K1
  with a net between −1 and 0 bp per fill; it was killed, at −2.35, below the
  predicted range. MK-Q was predicted killed by K1 with a negative making part
  (K-dir firing too): as predicted. MK-SQ was predicted between MK-Q and MK-S:
  −2.79 lies between −2.88 and −2.35.
- **What else fired**: K-nbhd and K-days on all three slots, K-cap on MK-Q
  (0.17 of printed volume) and MK-SQ (0.31), K-drift on MK-Q, K-placebo on
  MK-SQ. MK-S beat its placebos (percentile 100 of its 200 random-direction
  runs) and its drift runs, whose references were negative too (−2.49 and
  −2.56 bp per fill). No void condition fired in any run.
- Under the touch rule (the optimistic bracket) MK-S's net was −2.21 bp per
  fill and the quoters' −2.82 and −2.73: still negative.

## H (first read)

### Headline sentences

- MK-S on H, Binance spot BTC/FDUSD at the fees charged: killed; first kill K1 at -2.35 bp against 0.00 bp.
- MK-Q on H, Binance spot BTC/FDUSD at the fees charged: killed; first kill K1 at -2.88 bp against 0.00 bp.
- MK-SQ on H, Binance spot BTC/FDUSD at the fees charged: killed; first kill K1 at -2.79 bp against 0.00 bp.
- S25-PF's signal as a maker on Binance BTC/FDUSD, August to October 2026: -2.35 bp per fill at a 0 bp maker fee, 426.6 fills a day, making part -1.62 bp per fill; killed.
- Two-sided quoting at the touch proxies on Binance BTC/FDUSD, August to October 2026: -2.88 bp per fill, making part -1.21 bp per fill; killed.
- Quoting with S25-PF's side doubled on Binance BTC/FDUSD, August to October 2026: -2.79 bp per fill, making part -1.31 bp per fill; killed.
- Under the touch rule MK-S's net was -2.21 bp per fill against -2.35 bp under trade-through.

### The verdict, net of the fees charged

| Slot | Status | First kill | Net bp per fill (verdict) | t | NW t | Median day | Days with a fill | Days positive (share) | Fills a day (maker / taker) | Net bp per round trip | Net bp of a clip a day | MDE | Gross bp per fill, zero-fee reference |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MK-S | killed | K1 at -2.35 against 0.00 | -2.35 (uncalibrated fill rule) | -13.03 | -14.29 | -2.18 | 75 of 75 | 3 (0.04) | 426.6 (426.6 / 0.0) | -4.70 | -165.82 | 0.55 | -2.35 |
| MK-Q | killed | K1 at -2.88 against 0.00 | -2.88 (uncalibrated fill rule) | -50.49 | -45.40 | -2.91 | 75 of 75 | 0 (0.00) | 4360.6 (4359.8 / 0.8) | no round trips | -1816.42 | 0.17 | -2.87 |
| MK-SQ | killed | K1 at -2.79 against 0.00 | -2.79 (uncalibrated fill rule) | -45.54 | -41.57 | -2.83 | 75 of 75 | 0 (0.00) | 6148.3 (6147.4 / 1.0) | no round trips | -2740.52 | 0.16 | -2.77 |

### The fill rule's calibration against a book

Bybit spot BTCUSDT on D's days, with its book: the share of days on which the maker volume filled orders trade-through <= queue <= touch in the market-making simulator, and on which the prints rule's making part (the prints thinned to BTC/FDUSD's count that day) lies between the book's trade-through and touch making parts; it does not pass; every verdict above carries the label "uncalibrated fill rule".

| Slot | Days | Ordered | Inside the bracket | Passes |
|---|---|---|---|---|
| MK-S | 89 | 1.00 | 0.22 | no |
| MK-Q | 92 | 1.00 | 0.64 | no |
| MK-SQ | 92 | 1.00 | 0.58 | no |


### Every kill, in each slot's registered order (value against reference)

| Slot | Kill | Value | Reference | Fired |
|---|---|---|---|---|
| MK-S | K1 | -2.3475 | 0.0000 | fired |
| MK-S | K-dir | -1.6158 | 0.0000 | fired |
| MK-S | K-placebo | -2.3475 | -2.4905 | no |
| MK-S | K-drift | -2.3475 | -2.5593 | no |
| MK-S | K-nbhd | -2.3475 | 0.0000 | fired |
| MK-S | K-days | 0.0400 | 0.5000 | fired |
| MK-S | K-latency | -2.4240 | 0.0000 | no |
| MK-S | K-fill | -2.5318 | 0.0000 | no |
| MK-S | K-cap | 0.0296 | 0.1000 | no |
| MK-Q | K1 | -2.8795 | 0.0000 | fired |
| MK-Q | K-dir | -1.2104 | 0.0000 | fired |
| MK-Q | K-drift | -2.8795 | 11.0115 | fired |
| MK-Q | K-nbhd | -2.8733 | 0.0000 | fired |
| MK-Q | K-days | 0.0000 | 0.5000 | fired |
| MK-Q | K-latency | -2.8893 | 0.0000 | no |
| MK-Q | K-fill | -3.1270 | 0.0000 | no |
| MK-Q | K-cap | 0.1689 | 0.1000 | fired |
| MK-SQ | K1 | -2.7893 | 0.0000 | fired |
| MK-SQ | K-dir | -1.3119 | 0.0000 | fired |
| MK-SQ | K-placebo | -2.7893 | -2.7853 | fired |
| MK-SQ | K-drift | -2.7893 | -2.7909 | no |
| MK-SQ | K-nbhd | -2.7893 | 0.0000 | fired |
| MK-SQ | K-days | 0.0000 | 0.5000 | fired |
| MK-SQ | K-latency | -2.8263 | 0.0000 | no |
| MK-SQ | K-fill | -3.0322 | 0.0000 | no |
| MK-SQ | K-cap | 0.3052 | 0.1000 | fired |

### The making part and the markouts of the maker fills (bp per fill)

| Slot | Making part (5 s) | 1 s | 5 s | 30 s | Spread part | Adverse part |
|---|---|---|---|---|---|---|
| MK-S | -1.62 | -0.48 | -1.62 | -2.68 | 1.27 | -2.88 |
| MK-Q | -1.21 | -0.69 | -1.21 | -2.37 | 0.15 | -1.36 |
| MK-SQ | -1.31 | -0.72 | -1.31 | -2.41 | 0.29 | -1.60 |

### The runs beside the verdict (block means, bp per fill; measurements)

| Slot | Touch | Delayed | 110 ms | All long | All short | Sign-flipped | Placebo pct | K-nbhd median | The nine cells |
|---|---|---|---|---|---|---|---|---|---|
| MK-S | -2.21 | -2.53 | -2.42 | -2.56 | -2.85 | -2.80 | 100.0 | -2.35 | -2.30 / -2.40 / -2.51 / -2.28 / -2.35 / -2.45 / -2.29 / -2.29 / -2.43 |
| MK-Q | -2.82 | -3.13 | -2.89 | 11.01 | -23.89 | n/a | n/a | -2.87 | -2.79 / -2.87 / -2.92 / -2.81 / -2.88 / -2.93 / -2.80 / -2.87 / -2.94 |
| MK-SQ | -2.73 | -3.03 | -2.83 | -2.80 | -2.79 | -2.80 | 76.5 | -2.79 | -2.69 / -2.77 / -2.85 / -2.70 / -2.79 / -2.87 / -2.73 / -2.82 / -2.90 |

### Tiers and flattens (bp per fill; only the flattens' fees change)

| Slot | Regular (verdict) | VIP 3 | VIP 9 | Zero fees (reference) | Pooled net bp per fill | Fees paid (FDUSD) | Flattens | Flattened notional (FDUSD) |
|---|---|---|---|---|---|---|---|---|
| MK-S | -2.35 | -2.35 | -2.35 | -2.35 | -2.71 | 1.35 | 2 | 1346.07 |
| MK-Q | -2.88 | -2.87 | -2.87 | -2.87 | -2.96 | 35.86 | 60 | 35864.61 |
| MK-SQ | -2.79 | -2.78 | -2.78 | -2.77 | -2.90 | 63.73 | 72 | 63731.63 |

### Activity

| Slot | Fill ratio | Capacity (share of printed volume) | Max drawdown (FDUSD) | Proxies' width, median of days (bp) | Net (FDUSD) | Turnover (FDUSD) |
|---|---|---|---|---|---|---|
| MK-S | 0.5176 | 0.02963 | 969.06 | n/a | -969.46 | 3578672 |
| MK-Q | 0.3151 | 0.16895 | 10728.19 | 0.9254 | -10744.95 | 36358221 |
| MK-SQ | 0.2195 | 0.30524 | 16145.69 | 1.0747 | -16168.04 | 55782505 |

MK-S over the block: 35929 decisions; 4651 entries posted, 2435 filled in any part (242 in part), 2216 cancelled unfilled; skipped: 31209 busy, 69 outside the hours, 0 without a proxy; 2 exits flattened; realised hold from the entry's first fill to the exit's last: mean 128.3 s, median of the daily medians 92.9 s.

Days: 75 usable of 75; excluded none. The venue printed 56028 aggregated prints a day; the held unit's own P&L, never in a verdict, was 36.57 bp a day on average.
