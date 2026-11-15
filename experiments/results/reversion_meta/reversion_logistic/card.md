# reversion_logistic: benchmark card

A strategy (built-in), parameters as registered, source sha256 `ef04d8107e9a1f20`. Benchmark version 1, code 0.1.0 da1b1d4. Written 2026-11-14 22:17 UTC by `trp bench run`.

Marks, one per track in the order below: **✗✗** (✓ as a good strategy should, ✗ not, · a planted effect it was not built for, - not run).

| track | verdict | trades | gross bp | net bp | t (net) | placebo | break-even taker fee | data |
|---|---|--:|--:|--:|--:|--:|--:|---|
| bybit-2024-02 | positive after the fee, not distinguishable from chance | 1,663 | +32.55 | +18.12 | +1.92 | 100% | +14.56 bp | matches |
| bybit-2024-04 | nothing beyond chance | 936 | -25.58 | -40.46 | -1.47 | 5% | -14.73 bp | matches |

## Net per trade at each fee, bp

| track | base (5.5 bp) | vip1 (4 bp) | pro6_g1 (1.5 bp) | none (hypothetical) (0 bp) |
|---|--:|--:|--:|--:|
| bybit-2024-02 | +18.12 | +21.12 | +26.12 | +29.12 |
| bybit-2024-04 | -40.46 | -37.46 | -32.46 | -29.46 |

## What a pass would still need

Every number here is a backtest on a fixed span, at a stated fee, with a simple taker fill on a five-second grid (one second for the book tracks). The recorded spans were chosen after this repository had read them. A pass on a track is not a forecast. To trade a strategy that passes, it still needs: a pre-registered test on data no one has looked at (a held-out block, then a forward one); an execution simulated at the depth and latency it would really meet, since the top of the book on a grid is optimistic; the fee tier the account actually has; and a paper run on live data before money. This repository's own studies, which did all of that, found nothing that pays Bybit's published taker fees.
