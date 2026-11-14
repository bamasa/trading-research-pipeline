# reversion_lightgbm: benchmark card

A strategy (built-in), parameters as registered, source sha256 `9b6e96be4be3a1f9`. Benchmark version 1, code 0.1.0 7eea149. Written 2026-11-14 22:08 UTC by `trp bench run`.

Marks, one per track in the order below: **✗✗** (✓ as a good strategy should, ✗ not, · a planted effect it was not built for, - not run).

| track | verdict | trades | gross bp | net bp | t (net) | placebo | break-even taker fee | data |
|---|---|--:|--:|--:|--:|--:|--:|---|
| bybit-2024-02 | positive after the fee, not distinguishable from chance | 2,013 | +26.49 | +12.16 | +1.86 | 100% | +11.58 bp | matches |
| bybit-2024-04 | nothing beyond chance | 897 | -20.62 | -35.23 | -1.33 | 5% | -12.12 bp | matches |

## Net per trade at each fee, bp

| track | base (5.5 bp) | vip1 (4 bp) | pro6_g1 (1.5 bp) | none (hypothetical) (0 bp) |
|---|--:|--:|--:|--:|
| bybit-2024-02 | +12.16 | +15.16 | +20.16 | +23.16 |
| bybit-2024-04 | -35.23 | -32.23 | -27.23 | -24.23 |

## What a pass would still need

Every number here is a backtest on a fixed span, at a stated fee, with a simple taker fill on a five-second grid (one second for the book tracks). The recorded spans were chosen after this repository had read them. A pass on a track is not a forecast. To trade a strategy that passes, it still needs: a pre-registered test on data no one has looked at (a held-out block, then a forward one); an execution simulated at the depth and latency it would really meet, since the top of the book on a grid is optimistic; the fee tier the account actually has; and a paper run on live data before money. This repository's own studies, which did all of that, found nothing that pays Bybit's published taker fees.
