# Paper trading

The deployment step of the pipeline: the backtest's features, model, execution
rules and fee table, run on a public market-data stream or on a replayed day,
producing decisions and a marked-to-market paper account. The code is
[`src/trading_research/live/`](../src/trading_research/live/); the survey it
was built from is [`live_survey.md`](live_survey.md).

## What it is and is not

**It is** the research model running where a deployed one would run: on the
venue's public stream, deciding at the rows `prepare` would keep, with every
order filled against the prints that actually happened, at the fee the dated
table gives for that instant, with limits and a kill switch, and a page per
run that says what happened and how fast the loop was.

**It is not trading.** No order is sent anywhere. There is no key, no
signature, no private endpoint and no order path in the code: the only URLs
are the public market-data streams of Bybit and Binance
([`stream.py`](../src/trading_research/live/stream.py)), checked against a
fixed list before a run starts ([`guard.py`](../src/trading_research/live/guard.py)).
A run also refuses to start if any value of its configuration names something
that looks like an exchange credential (an environment variable such as
`BYBIT_API_KEY`, a `.env` or `credentials.json`, a file holding an
`api_key = ...` line). There is no such configuration; the refusal is a guard
that stays.

**A paper fill is a model of a fill**, the same model the backtests use, with
the same optimism: a taker order is assumed to get the next print of its side,
a resting order to be filled by a print through its price, and the paper
order's own size never moves the market. A paper result is a measurement of
the strategy under those rules on that stretch of market, never evidence that
it would have made money.

## Commands

```bash
# record a public stream to day files (raw log, plus the archive and staged layouts)
uv run trading-research paper record --venue bybit --market spot --symbol BTCUSDT --minutes 60

# replay a stored day: a recording (file or date), a Bybit archive, a dataset, a staged day
uv run trading-research paper replay --day data/demo --model artifacts/model \
    --execution taker --fee-tier base --fee-asof 2026-09-15
uv run trading-research paper replay --day 2026-10-27 --venue bybit --market spot \
    --symbol BTCUSDT --model artifacts/model --execution passive

# paper-trade the live public stream for N minutes (no key; refuses one)
uv run trading-research paper run --venue binance --market spot --symbol BTCUSDT \
    --model artifacts/model --execution taker --minutes 120

# where a run stands, while it runs or after
uv run trading-research paper status runs/paper/<run>

# streaming against batch features on a stored day
uv run trading-research paper parity --day data/demo --output parity.csv
```

`--model` takes the directory `train` writes (its `model.pkl`), a `.pkl`
file, or `always_hold`. A pickle runs code when it is read: load only a model
you trained. `--execution` is `taker`, `passive` or `maker` (with `--quoter
touch|skew`). The defaults are the backtests': a 10 ms order and cancel
latency, a 120 s hold, a decision every 50 rows (`prepare`'s `--subsample`),
`--min-confidence 0.5`. Limits: `--max-position-clips`, `--max-loss` (quote
currency), `--stale-after-s`, `--kill-on-gap/--no-kill-on-gap`.

Recordings go under `data/paper/<venue>-<market>/<SYMBOL>/` and runs under
`runs/paper/`, both ignored by git.

## The loop

One loop serves a live stream and a replayed day
([`session.py`](../src/trading_research/live/session.py)):

- **Rows and prints.** Bybit's book is replayed exactly as
  `data.bybit.reconstruct` replays the venue's archive (the archive is the
  same stream written to a file): at most one row per 100 ms interval, at its
  first update, stamped with that update's venue time, ten levels a side.
  Binance's partial depth gives ten levels every 100 ms; USD-M pushes carry the
  venue time and the previous update id, spot pushes neither, so a Binance
  spot row is stamped at receipt and a print stamped before it in venue time
  can arrive after it (counted as a late print on the page).
- **Features** ([`features.py`](../src/trading_research/live/features.py)) are
  computed by the batch functions on a window sized by each set's declared
  look-back. A model from `train` reads the `prepare` stage's columns; the loop
  computes them by the calls `prepare` makes.
- **Decisions** ([`decide.py`](../src/trading_research/live/decide.py)):
  `predict_proba`, then the backtest's `decide` rule with the confidence
  threshold; a boundary with a missing feature is not scored.
- **Execution**, never sent anywhere:
  - `taker`: arrival at decision plus latency; fill at the first print of the
    order's side at or after arrival, sweeping that stamp's same-side prints at
    their volume-weighted price, the part beyond the sweep at the sweep's worst
    price (`spot/fills.py`'s rule); RPI prints never fill; the book-walk price
    of the first row at or after arrival is recorded beside it. Exit at the
    first row stamped at or after the entry fill plus the hold.
  - `passive`: one clip at the touch, kept at its first price, filled only by
    prints strictly through it, up to their size (`queue.on_print_rule`,
    `FillRule.THROUGH`); cancelled a hold after the post; the position closed
    by a taker order a hold after its first fill.
  - `maker`: a quoter of `market_making/quoters.py` asked at every row, every
    quote logged, orders replaced when the quote moves, the same trade-through
    fills, no quote on a side whose fill could pass the hard limit (soft limit
    plus one clip).

  In a live run an order's arrival also carries the loop's own time since the
  row arrived and the feed's measured delay (the median receipt-minus-venue
  time of recent prints), so a live decision is never credited with arriving
  before it could have. A replay carries neither, as a backtest does not.
- **The account** ([`ledger.py`](../src/trading_research/live/ledger.py)) is
  `market_making.accounting.Account`, whose identity
  `realised + unrealised - fees == cash + position x mark` is checked after
  every fill and every mark. It is marked at the mid at every row.

## Fees

Each fill pays the maker or taker fee of the chosen tier in force at the
fill's instant ([`fees.py`](../src/trading_research/live/fees.py)), from the
tables the backtests use: Bybit spot BTCUSDT and Binance spot BTC/FDUSD from
`configs/spot_2026.yaml` (dated intervals, read by the spot study's own
reader), Bybit linear from `backtest/costs.py` (the page dated 2026-09-02,
transcribed 2026-10-02), Binance USD-M's standard taker from the same file
(undated). One row was added for the smoke test's venue: Binance's spot
schedule as read on 2026-10-27 (UTC) from the public commission endpoint the
spot registration lists (regular 10 bp maker and taker, BTCUSDT not on the
zero-fee list), held for the UTC day it was read.

Every fill records what its fee rests on: `sourced` (inside a dated
interval), `carried` (after the last interval, carried forward, which a live
run's instants always are), `undated`, `given` (`--fee-bp`, not sourced), or
`... as of <date>` (`--fee-asof`, for a synthetic or older day the table does
not reach). An instrument no table covers is refused unless `--fee-bp` is
given.

## The parity guarantee

On a replayed day, the streaming features at every boundary equal the batch
features of that day ([`parity.py`](../src/trading_research/live/parity.py),
`paper parity`, `tests/test_paper_parity.py`). The batch side is
`features.build` for the registry, `features.timed.build_timed` for the 2025
timed set, and **the `prepare` stage itself**, run on the day written in the
staged layout, for the prepared columns a `train` model reads. The streaming
side holds only each set's declared window:

| set | window | declared by |
|---|---|---|
| registry (20 book features) | the row and the longest `lookback` rows before it (70) | `features/registry.py` |
| timed (14 features) | rows since `s_t - lookback_s`, `lookback_rows` before them, the prints they own | `validation/leakage_timed.streaming_window` |
| prepared (211 columns) | 13,816 rows (the EWM of span 1,000 to 1e-12), prints of the last 360 s, `prepare`'s day start (1,200 rows before midnight, that day's prints) | this package, from `features/generated.py` and `features/align.py` |

The tolerance is the timed-leakage checks' (relative 1e-7, absolute 1e-10).
On the default synthetic demo day (20,000 rows, 8,901 prints; generated by
`generate-demo-data`, seed 20260101):

| set | boundaries | features | largest absolute deviation | largest relative deviation |
|---|---:|---:|---:|---:|
| registry | 20,000 (every row) | 20 | 1.3e-13 (`ofi_20`) | 5.0e-11 |
| timed | 20,000 (every row) | 14 | 3.0e-14 (`volatility_1s_bp`) | 2.0e-12 |
| prepared, the 177 columns that are not rolling dispersions | 400 (every row `prepare` keeps at subsample 50) | 177 | 1.6e-9 | 1.2e-8 (`vwap_dev_bp_5s`) |

Twenty-seven of the 34 registry and timed features, and 106 of the 211
prepared columns, are bit-for-bit equal; the EWMs cut at the declared
look-back differ by at most 1.1e-8 relative (`microprice_dev_bp_ewm1000`).
Seven prepared columns lie outside the tolerance: the rolling standard deviations of
the spread at 5 and 50 rows (absolute 5.3e-10 and 6.8e-10 bp, on dispersions
of about 5e-7 bp), the z-scores divided by them (`spread_bp_z5`, `_z20`,
`_z50`, `_z200`, up to 1.3e-3), and `trade_disp_bp_5s` (2.4e-7). These are
windows where the spread barely moves (it sits at one tick and its basis
points move only with the mid), so the dispersion is of the order of the
input's rounding, and pandas' online rolling variance carries rounding from
every row that has left its window: against an exact two-pass computation the
batch value and the streaming value are both off by about 1e-9 bp, neither
closer. The report lists such columns as rounding-sensitive; on a day shorter
than the declared look-back, where the two frames coincide, every column is
held to the tolerance (the fast test), and on the long day every other column
is (the slow test).

A live session is reproduced exactly by replaying its own raw log: in both
smoke tests below, every boundary's probabilities were identical between the
live run and the replay of its recording.

## The kill switch

Latched by the first of: net equity at a mark at or below `-max_loss`; a
position beyond the limit after a fill (an order that could cause one is
refused before it is sent); no book row for `stale_after_s` (on the wall clock
in a live run, between row stamps in a replay); a sequence gap or a reconnect
(with `--kill-on-gap`, the default). Once latched nothing new is sent, resting
orders are cancelled, the position is closed by a taker order, and the run
stops once flat, or `flatten_wait_s` later if no print came, in which case the
close is priced at the last print of its side and flagged `no_print`.

## The page

Each run directory holds `page.md` (money, decisions, fills, the loop, gaps),
`summary.json`, `manifest.json` (the model file's sha256 and features, the fee
table with its sources and the statuses the fills paid, the code commit and
whether the tree differed from it, the command, the seeds, and the sha256 of
every table), `decisions.csv`, `orders.csv`, `fills.csv`, `gaps.csv`,
`marks.parquet` (a mark at every row) and, for the maker, `quotes.parquet`.
`status.json` is rewritten every five seconds while a live run is going.

## The two-minute live smoke test

Run once on 2026-10-27 between 21:46 and 21:50 UTC, on each venue's spot
BTCUSDT, with the five-feature logistic model `train` fitted on the synthetic
demo day (`select --max-features 5`; every feature a window of at most 20
rows), `--execution taker --every 10 --hold-s 30 --clip 0.001`. The model was
fitted on synthetic data: its decisions on a real market mean nothing, and the
numbers below test the plumbing, not a strategy.

| | Bybit spot BTCUSDT | Binance spot BTCUSDT |
|---|---:|---:|
| messages a second | 12.1 | 23.2 |
| rows / prints | 1,166 / 1,311 | 888 / 1,571 |
| gaps, reconnects | 0, 0 | 0, 0 |
| late prints | 0 | 151 (rows stamped at receipt) |
| feed delay, rows, p50 / p99 ms | 39.5 / 497.6 | n/a (no venue time) |
| feed delay, prints, p50 / p99 ms | 44.4 / 100.3 | 87.8 / 142.1 |
| loop: row to decision, p50 / p99 / max ms | 39.0 / 66.9 / 72.9 | 39.1 / 68.2 / 68.6 |
| features per boundary, p50 ms | 38.2 | 38.2 |
| boundaries / decisions (buy, sell) / entered | 117 / 12, 24 / 3 | 89 / 7, 47 / 4 |
| fills (all taker, fill rate 1.0) | 6 | 8 |
| PnL gross, USDT | +0.0254 | +0.0359 |
| fees, USDT (10 bp taker) | 0.4862 | 0.6485 |
| PnL net, USDT | -0.4608 | -0.6126 |
| fee status | carried (Bybit's interval ends 2026-10-22) | sourced (read 2026-10-27) |
| kill switch | not tripped | not tripped |

Two shorter checks of the other policies ran the same evening for thirty
seconds each: the S1 quoter on Bybit linear BTCUSDT posted 56 orders and none
was traded through (a one-tick book); the passive policy on Binance USD-M
BTCUSDT posted two entries, one was cancelled unfilled after its hold, the
other was filled by a print through its price, and the run ended before its
hold, so the close was priced at the last print of its side and flagged
`no_print`.

## Limits of this step

- The `prepare` columns are recomputed in full at each boundary (38 ms a
  boundary in the smoke tests, about 100 ms in the demo replay): fine for a
  decision every second or few, too slow for one every 100 ms row.
- Binance spot publishes no venue time for its book, so its rows and prints
  run on two clocks; the page counts the prints that arrive after a row they
  belong to.
- A paper fill cannot move the market or lose queue priority it never had; a
  live order would. The fill rules are the backtests' and carry their biases
  (the print rule's in `spot/fills.py`, the trade-through rule's in
  `market_making/fill_rules.py`).
- The day boundaries of the studies (no entry before 00:01 or after 23:58, a
  flatten at 23:59) are not applied: a paper session runs through midnight.
