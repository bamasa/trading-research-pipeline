# The author's 2025 ETH result, replicated on public data and tested frozen

Part 1 is complete; part 2 (the frozen test on 2026-08-08 .. 21) is running.
The summary is written when it lands.

## What was replicated

The author's 2025 research trained a TCN on ten days of ETH/USDT spot books
(2025-01-28 .. 2025-02-06): seven days to train, one to validate, the last two
to test. Its gate turned the three-class probabilities into signals, and a
backtest held 0.01 ETH on them. Of about thirty gate configurations run on the
two test days, one ended positive after the backtester's fee: +3.1 USDT before
it and +0.9 after, 358 fills, turnover 9,865 USDT, so the fee was about 2.2 bp
of turnover. That configuration (its notebook's cell 48) held positions from
one signal to the next opposite one (`baseline_strategy`), not for a fixed time:
the close-by-time lines are commented out in that cell. Its close-by-time runs
are in the sweep below.

## Venue

The original's books (Binance spot ETH/USDT, 100 ms snapshots) are not public.
Both parts here use Bybit's public **ETHUSDT linear perpetual** archives: the
order book replayed from `quote-saver.bycsi.com` (`ob500` for 2025 days, `ob200`
for 2026 days) and the prints from `public.bybit.com/trading/ETHUSDT/`. A
perpetual on another venue is a different market: the same model trained here
sees other participants, another fee schedule and a funding rate, and its
numbers cannot be read as the original's. Every number below is Bybit's.

## The recipe as re-typed

The code is in `src/trading_research/replication_eth/`, the run in
`experiments/replication_eth_2025.py`.

- **Grid.** Each day replayed by `data.bybit.reconstruct` at 20 levels a side;
  each 100 ms boundary takes the latest replayed book at or before it and the
  prints since the previous boundary. Every 2025 and 2026 day replayed without
  a sequence gap, at 863,980 of 863,999 boundaries.
- **Inputs.** The original's 16 auxiliary inputs and 40 book inputs (56), its
  volatility filter (rows below 2e-5 carry no label) and its gate's three
  inputs, re-typed from its preprocessing. Checked against the original's own
  functions on a synthetic day: the 16 auxiliary inputs agree within 6e-7 (the
  original rounds per-row sums to nine decimals), the 40 book inputs and the
  labels exactly, the gate's decisions exactly.
- **Labels.** 5 s ahead against 5 s behind, at +-5e-4. The original's code holds
  two thresholds (2e-4 printed when it ran on one day, 5e-4 in its source); on
  the seven training days 5e-4 gives 7.7 / 84.6 / 7.7 % (down / flat / up, on
  filtered rows) against the original's printed 9.1 / 81.9 / 9.1 %, and 2e-4
  gives 24.8 / 51.0 / 24.2 % (`replication_eth_2025_label_shares.csv`).
- **Network and training.** As the original: five residual blocks of 128
  channels, kernel 3, dilations 1 to 16, channel dropout 0.5, head on the last
  step and the window mean; windows of 50 rows; class-weighted cross-entropy
  with label smoothing 0.05, AdamW 3e-5, five epochs, the epoch with the best
  validation accuracy kept. Seed 0, on Apple MPS.
- **Gate.** The original's `futures_taker` scenario, including its reading of
  three z-scored columns as basis points (the audit's units bug), kept as it
  computed them.
- **Execution.** `backtest.taker.execute_day`: every fill crosses the spread
  0.01 ETH at the next 100 ms snapshot after a 10 ms latency, entries 00:01 to
  23:58 UTC, flatten at 23:59, no funding. Fees are taker fees per side on each
  fill's notional: 0, 2.2 bp (the original's effective fee on turnover) and
  5.5 bp (Bybit's base taker fee). The original's backtester and its order
  types are not public, so its fills cannot be matched one to one.

## Where the replication departs from the original, and why

1. **Venue and instrument.** Bybit's ETHUSDT perpetual, not Binance's ETH/USDT
   spot (see above).
2. **Book depth for the depth totals.** Three of the 16 inputs sum the sizes
   on every level a row carries; the original's rows are not printed with
   their length. Twenty levels are used, the depth of the venue's 100 ms
   partial-book stream, which matches rows stamped on every 100 ms boundary.
3. **Split at day boundaries.** The original cut its ten days 0.7 / 0.1 / 0.2
   by row count, so its training split took 119 rows of the eighth day and its
   test began 105 rows (about ten seconds) into the ninth; here the splits are
   whole days.
4. **Event order.** The original grouped rows into events after pandas'
   default sort on whole seconds, which is unstable, and then kept the row at
   each event's *sorted* position in the *unsorted* labels; a signal survived
   only where the sort left it in place. The rule as written keeps time order,
   which is what part 2 registered. The original's code is re-enacted as one
   more part-1 run (`ref48_unstable_sort`), with this machine's sort, whose
   permutation need not match the original machine's.
5. **Holding and fills.** The original's backtester, its order types and its
   `Slippage 100` setting are not public. `signal_to_signal` follows the
   documented behaviour of its `baseline_strategy` with MaxPos = OrderSize =
   0.01 ETH (a signal from flat opens, the next opposite signal closes to flat,
   a repeat is ignored); every fill here is a taker fill at the next snapshot.
   The original's tables count fills of which some were sent and not executed;
   here every decision that opens is filled.
6. **The unit of `close_time`.** Not in the original's code. Read as seconds:
   its close-by-time runs show about one round trip per 100 s plus the wait for
   a signal (e.g. 1,407 fills from 1,811 signals over two days). The 10 s
   reading (100 rows) is run too.
7. **Hours and funding.** The executor takes no entry before 00:01 or after
   23:58 UTC and closes at 23:59; the original ran whole days. Funding is left
   out (the original traded spot; a settlement on 0.01 ETH is about 0.003
   USDT).
8. **Training.** One fit, seed 0, on Apple MPS (the original trained on a CPU,
   seed not recorded); MPS kernels are not bit-for-bit deterministic, so a
   re-run gives a nearby model, not the same one.
9. **Not replayed.** The original's dollar-unit gate (its v0.2 run, 3,909
   fills, +10 before its fee and -14 after) and its REF cell 50 (the best gate
   with MaxPos 0.03 ETH) are a different gate and a different position limit
   and are not part of the sweep here.

## Part 1: the reproduction on 2025-02-05 .. 06 (exploratory)

Trained on 2025-01-28 .. 02-03, validated on 02-04, tested on 02-05 .. 06.
The fit (`replication_eth_2025_part1_training.csv`): validation accuracy on
the targeted rows 0.607, 0.623, 0.636, 0.634, 0.666 over the five epochs, the
fifth kept (the original: 0.657 to 0.671, its third kept). On the test days,
1,352,079 of 1,727,160 windows pass the filter; argmax accuracy on them is
0.770 against 0.878 for always calling flat (the original: 0.743 against
0.868), so, as there, the classes' accuracy says nothing about money.

Every configuration tried, in USDT over the two days, 0.01 ETH, all twenty-nine
of them, none chosen. "Gross bp of turnover" is the gross PnL over the fills'
notional, which is also the break-even taker fee per side. The original's
columns are its printed tables (Binance spot, its backtester, its fills).

| run | holding | signals | fills/day | gross | net 2.2 bp | net 5.5 bp | gross bp of turnover | orig. fills | orig. PnL | orig. after fee |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ref37 | signal_to_signal | 1,261 | 409 | 0.07 | -4.89 | -12.34 | 0.03 | 185 | 1.0 | 0.0 |
| c121 | signal_to_signal | 1,840 | 608 | -2.43 | -9.81 | -20.89 | -0.72 | 298 | -1.0 | -2.0 |
| c122 | signal_to_signal | 1,853 | 613 | -2.15 | -9.60 | -20.77 | -0.64 | 300 | 0.0 | -2.0 |
| c124 | signal_to_signal | 1,840 | 608 | -2.43 | -9.81 | -20.89 | -0.72 | 298 | -1.0 | -2.0 |
| c125 | signal_to_signal | 2,877 | 938 | -0.19 | -11.58 | -28.66 | -0.04 | 496 | 1.0 | -2.0 |
| c126 | signal_to_signal | 4,803 | 1,525 | 2.06 | -16.43 | -44.18 | 0.25 | 798 | 1.0 | -4.0 |
| c133 | signal_to_signal | 2,605 | 780 | 1.35 | -8.10 | -22.28 | 0.31 | 350 | 2.0 | 0.0 |
| c138 | signal_to_signal | 3,627 | 1,005 | 1.49 | -10.69 | -28.96 | 0.27 | 592 | 3.0 | -1.0 |
| c149 | signal_to_signal | 2,605 | 780 | 1.35 | -8.10 | -22.28 | 0.31 | 350 | 2.0 | 0.0 |
| c150 | signal_to_signal | 4,603 | 1,072 | 2.44 | -10.55 | -30.03 | 0.41 | 930 | 5.0 | -1.0 |
| c155 | signal_to_signal | 4,801 | 1,526 | 2.05 | -16.46 | -44.22 | 0.24 | 796 | 1.0 | -4.0 |
| c156 | signal_to_signal | 9,721 | 2,416 | 2.46 | -26.82 | -70.75 | 0.18 | 2,424 | 3.0 | -12.0 |
| c164 | signal_to_signal | 7,189 | 1,767 | 1.19 | -20.22 | -52.33 | 0.12 | 1,591 | 4.0 | -6.0 |
| c150 | close_by_time_100s | 4,603 | 935 | 0.70 | -10.65 | -27.68 | 0.14 | 1,407 | 3.0 | -6.0 |
| c176 | signal_to_signal | 4,601 | 1,070 | 2.25 | -10.71 | -30.16 | 0.38 | 932 | 5.0 | -1.0 |
| c183 | signal_to_signal | 1,808 | 421 | 2.27 | -2.83 | -10.48 | 0.98 | 316 | 2.0 | 0.0 |
| c189 | close_by_time_100s | 6,325 | 936 | 0.69 | -10.68 | -27.72 | 0.13 | 1,936 | 6.0 | -6.0 |
| c189 | signal_to_signal | 6,325 | 1,153 | 3.01 | -10.96 | -31.92 | 0.47 | 1,458 | 7.0 | -2.0 |
| c195 | close_by_time_100s | 6,606 | 946 | 1.29 | -10.19 | -27.42 | 0.25 | 1,982 | 6.0 | -7.0 |
| c195 | signal_to_signal | 6,606 | 1,199 | 2.45 | -12.08 | -33.87 | 0.37 | 1,502 | 7.0 | -3.0 |
| ref48 | signal_to_signal | 2,674 | 808 | 0.76 | -9.03 | -23.72 | 0.17 | 358 | 3.1 | 0.9 |
| c205 | signal_to_signal | 3,425 | 1,051 | 0.69 | -12.04 | -31.15 | 0.12 | 504 | 1.0 | -2.0 |
| c206 | signal_to_signal | 2,713 | 821 | 0.49 | -9.46 | -24.39 | 0.11 | 352 | 3.0 | 1.0 |
| ref48 | close_by_time_500s | 2,674 | 287 | -0.66 | -4.15 | -9.38 | -0.42 | 454 | 2.0 | -1.0 |
| c218 | signal_to_signal | 2,696 | 814 | 0.60 | -9.26 | -24.06 | 0.13 | 348 | 3.0 | 1.0 |
| c223 | signal_to_signal | 4,756 | 1,114 | 1.78 | -11.72 | -31.96 | 0.29 | 960 | 5.0 | -1.0 |
| ref48 | close_by_time_100s | 2,674 | 938 | 0.10 | -11.29 | -28.37 | 0.02 |  |  |  |
| ref48 | close_by_time_10s | 2,674 | 2,621 | 2.29 | -29.48 | -77.15 | 0.16 |  |  |  |
| ref48_unstable_sort | signal_to_signal | 510 | 166 | -1.09 | -3.10 | -6.12 | -1.19 |  |  |  |

No configuration is positive at 2.2 bp or at 5.5 bp, on either day. Gross
PnL is within a few USDT of zero everywhere (the largest, 3.0 USDT, is 0.47 bp
of turnover); the best gross edge, 0.98 bp of turnover (`c183`), is below the
2.2 bp fee. The replication's gate passes four to five times as many signals
as the original's (2,674 against 583 for its best configuration): re-enacting
the original's unstable sort brings the count to 510, close to its 583, and
that run loses before any fee (-1.09 USDT). Per day and per fee:
`replication_eth_2025_part1_days.csv`.

## Part 2: the frozen test on 2026-08-08 .. 2026-08-21

Running: see `docs/replication_eth_plan.md` for what it does.

## Files

- `replication_eth_2025_part1_days.csv`, `replication_eth_2025_part2_days.csv`:
  one row per run, day and fee.
- `replication_eth_2025_part1_runs.csv`, `replication_eth_2025_part2_runs.csv`:
  one row per run, pooled over its days, with the original's printed numbers
  beside part 1's.
- `replication_eth_2025_part1_summary.csv`, `replication_eth_2025_part2_summary.csv`:
  pooled per run and fee.
- `replication_eth_2025_part1_training.csv`, `replication_eth_2025_part2_training.csv`:
  the fits' epochs.
- `replication_eth_2025_label_shares.csv`: the threshold's choice.
- `docs/replication_eth_plan.md`: part 2's plan, committed before any 2026 day
  was fetched.
