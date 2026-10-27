# What a streaming path must reproduce

A survey of the batch pipeline, written before the paper-trading package
([`src/trading_research/live/`](../src/trading_research/live/)) and kept as its
specification. Each line names the batch code that defines the behaviour; the
streaming path calls that code or is tested against it.

## Data contracts

- **Book rows** ([`data/schema.py`](../src/trading_research/data/schema.py),
  [`docs/data_contract.md`](data_contract.md)): `timestamp` (UTC, the moment
  the state became observable), `symbol`, `sequence_id`, `source`, then
  `bid_price_i`, `bid_size_i`, `ask_price_i`, `ask_size_i` for levels 0..9;
  prices in quote, sizes in base. Depth is a parameter; ten is what the
  archive replay keeps.
- **Bybit's book** ([`data/bybit.py`](../src/trading_research/data/bybit.py)):
  a snapshot then deltas (`[price, size]`, size 0 deletes); a delta whose
  update id `u` is not the previous one plus one is a sequence gap, counted
  and carried, never repaired. `reconstruct` emits at most one row per 100 ms
  interval, at the interval's **first** update, stamped with that update's
  venue `ts`, only when both sides hold ten levels; no row for an interval
  without an update. The archive is the public websocket stream written to a
  file, so the live parser must be this loop, one message at a time.
- **Prints** ([`data/bybit_trades.py`](../src/trading_research/data/bybit_trades.py)):
  `timestamp`, `price`, `size`, `aggressor` (+1 a buyer crossed); Bybit's
  `side` is the taker's. Binance's trade plane
  ([`data/binance.py`](../src/trading_research/data/binance.py)) keeps
  `is_buyer_maker` (True: the seller crossed) and the aggregate id, which runs
  on by one. Binance publishes no spot book history; the live partial depth
  (ten levels every 100 ms) is the only multi-level Binance book here, and
  spot pushes carry no venue time.

## Features and their look-backs

- **Row registry** ([`features/registry.py`](../src/trading_research/features/registry.py),
  [`book.py`](../src/trading_research/features/book.py),
  [`microstructure.py`](../src/trading_research/features/microstructure.py)):
  each feature declares `lookback` rows; the value at row *t* reads rows
  *t-lookback..t*. Longest: 70 (`ofi_20_norm`).
- **Timed** ([`features/timed.py`](../src/trading_research/features/timed.py),
  [`micro2025.py`](../src/trading_research/features/micro2025.py)): `lookback_s`
  on row stamps plus `lookback_rows`; prints follow the backward rule of
  [`features/prints.py`](../src/trading_research/features/prints.py) (a print
  belongs to the first row stamped at or after it). The live buffer is
  `validation/leakage_timed.streaming_window`.
- **Flow** ([`features/flow.py`](../src/trading_research/features/flow.py)):
  windows in rows, at most 120, on `bucket_prints`.
- **Prepared** (the `prepare` stage, [`pipeline/stages.py`](../src/trading_research/pipeline/stages.py)):
  [`features/generated.py`](../src/trading_research/features/generated.py)
  (rolling windows to 1,000 rows, lags to 50, and EWMs of span to 1,000, which
  remember every row since the frame began), the trade features of
  [`features/align.py`](../src/trading_research/features/align.py) (trailing
  time windows to 300 s, a backward as-of join with 60 s tolerance, an impact
  term reading 300 rows back) and the registry features the generated set
  lacks. `prepare` starts each day's frame 1,200 rows before midnight, reads
  only that day's prints, and keeps every `subsample`-th row. Models from
  `train` are fitted on these rows, so a live decision is made where
  `prepare` would keep a row.

## Models

`train` pickles `{"model", "features"}` ([`models/base.py`](../src/trading_research/models/base.py)):
`predict_proba` gives (sell, hold, buy) in `CLASSES` order; `predict` scores
only rows with every feature present. `backtest/evaluate.decide` trades when
the stronger side's probability clears `min_confidence` and beats the other.
[`models/export.py`](../src/trading_research/models/export.py) checks an ONNX
copy against the original; the paper loop reads the pickle and records its
sha256.

## Execution policies

- **Taker on prints** ([`spot/fills.py`](../src/trading_research/spot/fills.py),
  [`spot/print_taker.py`](../src/trading_research/spot/print_taker.py)): the
  order arrives at decision + latency (10 ms) and fills at the **first print of
  its side** at or after arrival (a buy at a buyer-initiated print), sweeping
  that stamp's same-side prints at their volume-weighted price, the part beyond
  the sweep at the sweep's worst price; RPI prints never fill. Exit at the
  first row stamped at or after the entry fill's print + hold, by the same
  rule; no print after an exit: the last same-side print at or before arrival.
  The book walk of [`backtest/taker.py`](../src/trading_research/backtest/taker.py)
  is the other taker rule; the paper loop records its price beside the print
  rule's.
- **Passive** ([`market_making/queue.py`](../src/trading_research/market_making/queue.py),
  `FillRule.THROUGH`; [`market_making/signal_entry.py`](../src/trading_research/market_making/signal_entry.py)):
  one clip posted at the touch on the decision's side, kept at its first
  price, filled **only by prints strictly through it**, up to their size;
  cancelled `hold` after the post (it can fill until the cancel lands);
  exit a taker order `hold` after the first fill.
- **Maker** ([`market_making/quoters.py`](../src/trading_research/market_making/quoters.py),
  [`simulator.py`](../src/trading_research/market_making/simulator.py)): a
  quoter sees a `MarketView` (prices in ticks, `vol_bp_1m` from a 60 s
  half-life EWMA of per-second squared log returns, the trailing-hour median
  touch size, the clip, the soft limit) and returns a quote per side; orders
  live after the order latency, cancels land after the cancel latency, the
  hard limit is the soft limit plus one clip.

## Fees and accounting

[`backtest/costs.py`](../src/trading_research/backtest/costs.py) holds the
Bybit linear tiers (transcribed 2026-10-02) and Binance USD-M's standard
taker; [`configs/spot_2026.yaml`](../configs/spot_2026.yaml) (`fees`, read by
[`spot/costs.py`](../src/trading_research/spot/costs.py)) holds dated
intervals for Bybit spot BTCUSDT and Binance spot BTC/FDUSD: a fill pays the
fee of the interval holding its instant; an instant no interval holds is "fee
not sourced". [`market_making/accounting.py`](../src/trading_research/market_making/accounting.py)
keeps position, average cost, cash, fees, and checks
`realised + unrealised - fees + funding == cash + position x mark` after every change.

## What the streaming path adds

Wall-clock latency (receipt to decision), late prints (stamped before a row
already evaluated), reconnects, and a kill switch on loss, position, stale
data and sequence gaps. None of these exists in a batch run, so each is
reported on the run's page rather than folded into a number.
