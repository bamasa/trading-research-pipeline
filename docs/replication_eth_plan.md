# The ETH replication: plan for the frozen test

Written and committed before any 2026 ETHUSDT day was fetched or read. Part 1
(the reproduction on 2025-01-28 .. 2025-02-06) is exploratory; this document
fixes everything part 2 does, so that its numbers are a test rather than a
search.

## What is tested

The author's 2025 research reported one crypto result that was positive after a
non-zero fee: a TCN on ten days of ETH/USDT spot books, gated in basis points,
backtested on 2025-02-05 .. 06 with 0.01 ETH, +3.1 USDT before the fee and +0.9
after an effective fee of about 2.2 bp of turnover. It was the best of about
thirty gate configurations run on those same two days. Part 2 asks whether the
same recipe, frozen, makes money on fourteen days nobody has looked at.

## Venue

The original's books are not public. Both parts use Bybit's public ETHUSDT
linear perpetual archives: the order book
(`quote-saver.bycsi.com/orderbook/linear/ETHUSDT/<day>_ETHUSDT_ob200.data.zip`
for 2026 days) and the prints (`public.bybit.com/trading/ETHUSDT/`). This is a
different venue and a different instrument (a perpetual, not spot) from the
original's; every number is this venue's.

## The recipe, fixed

Everything below is as part 1 ran it; the code is in
`src/trading_research/replication_eth/` and `experiments/replication_eth_2025.py`.

- **Data.** Each day replayed by `data.bybit.reconstruct` at 20 levels a side
  and put on 100 ms boundaries (the latest replayed book at or before each
  boundary; the prints in the interval ending at it). Rows 200 to N-200 of each
  day are kept.
- **Inputs.** The original's 16 auxiliary inputs (rolling 100-row z-scores
  except `book_imbalance`) and its 40 normalised book inputs (ten levels a
  side), 56 in all, in the original's order.
- **Labels.** Mean mid over the next 50 rows (5 s) against the mean over the
  last 50, relative, against plus or minus 5e-4 (chosen in part 1 from the
  original's printed class mix on its training days, before any test day was
  scored: `experiments/results/replication_eth_2025_label_shares.csv`); a row whose
  volatility input is below 2e-5 has no label.
- **Network.** Five residual blocks of 128 channels, kernel 3, dilations 1 to
  16, weight-normalised causal convolutions, channel dropout 0.5, head on the
  last step and the window mean; windows of 50 rows.
- **Training.** Seven days to train, the next day to validate, then the test
  days: 2026-07-31 .. 2026-08-06 and 2026-08-07. Cross-entropy with
  inverse-frequency class weights and label smoothing 0.05, AdamW 3e-5 with
  weight decay 1e-5, batches of 128, gradient norm 1, the learning rate halved
  after two epochs without a lower validation loss, five epochs, the weights of
  the epoch with the best validation accuracy kept. Seed 0, Apple MPS. One fit;
  it is not repeated or re-seeded.
- **Threshold rule.** The original's final gate, unchanged (its REF cell 48):
  `tau_move 0.9, min_margin 0.45, buffer_bp 4, t_min 1.6, event_max_gap 15 s,
  cooldown 10 s`, with the strategy inputs computed as the original computed
  them (from z-scored columns; see `gate.py`). Nothing is tuned on 2026 data.
- **Holding.** Primary: `signal_to_signal`, what the original's +0.9 run used
  (a signal from flat opens 0.01 ETH, the next opposite signal closes it, a
  repeated signal is ignored, 23:59 closes what is open). Secondary:
  `close_by_time` 100 s (the strategy named in the brief). Both are reported;
  neither is chosen over the other after the fact.
- **Execution.** `backtest.taker.execute_day`: 10 ms latency, the next 100 ms
  snapshot walked for 0.01 ETH, entries 00:01 to 23:58 UTC, flatten at 23:59,
  no funding.

## Code

Part 1 runs first, on 2025 data only. If it finds a defect in the code, the fix
is made before any 2026 day is fetched and listed in the results. The commit
that part 2 runs is the last one before its first fetch; its hash is recorded
in the results, and nothing under `src/trading_research/replication_eth/` or in
`experiments/replication_eth_2025.py` changes after it until part 2's numbers
are written.

## What is reported

For both holdings, on each of the fourteen test days and pooled: signals,
round trips and fills a day, gross PnL, PnL at a taker fee of 2.2 bp per side
(the original's effective fee on turnover) and at 5.5 bp (the venue's base
taker fee), turnover, the share of days with a positive net at each fee, and
the gross edge in bp of turnover, which is the break-even fee per side.

The result is called positive at a fee if its pooled net over the fourteen days
is above zero at that fee. Fourteen days is a small sample; the per-day rows
are published so the spread can be read directly.

## What happens if something is missing

A test day the archive lacks, or whose replay has a sequence gap, is reported
as such and not replaced. If a training or validation day is missing, the span
moves to the nearest earlier days so that it stays seven plus one, and that is
reported. No other change is made after a 2026 day is read; anything that has
to change is recorded in the results as a deviation, with its reason.
