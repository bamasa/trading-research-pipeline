# Spot study: the format probe of 25 October 2026

The spot study's readers were written against the column layout of each
archive. To learn it, the first 4 KB of one file per plane were read, and
nothing else: one file, one request, one byte range, on the first of the
registered reader-test days (2026-03-03). No other day and no other byte of
any spot file was read. The registration cites this note (its "What was
known", item 7), and stage 0 measures on the three reader-test days what it
saw in these bytes.

## What was read

Each request carried `Range: bytes=0-4095`, read at most 4,096 bytes of the
body and closed the connection. Every server answered `206 Partial Content`
with exactly 4,096 bytes; the totals in `Content-Range` equal the sizes the
head check of 2026-10-22 recorded for the same files.

| Plane | URL | Byte range | Requested (UTC) | Status | `Content-Range` |
|---|---|---|---|---|---|
| Bybit spot `ob200` book | `https://quote-saver.bycsi.com/orderbook/spot/BTCUSDT/2026-03-03_BTCUSDT_ob200.data.zip` | 0-4095 | 2026-10-25T11:26:20.598Z | 206 | `bytes 0-4095/100382621` |
| Bybit spot prints | `https://public.bybit.com/spot/BTCUSDT/BTCUSDT_2026-03-03.csv.gz` | 0-4095 | 2026-10-25T11:26:20.869Z | 206 | `bytes 0-4095/10690155` |
| Binance BTCFDUSD `aggTrades` | `https://data.binance.vision/data/spot/daily/aggTrades/BTCFDUSD/BTCFDUSD-aggTrades-2026-03-03.zip` | 0-4095 | 2026-10-25T11:26:21.452Z | 206 | `bytes 0-4095/3316422` |

The bytes were inflated as far as they go (a raw deflate stream from the zip's
first local header, a gzip stream for the prints) in a scratch directory
outside the repository and deleted after this note was written. No byte of
them is committed, and no price, size or count from them is quoted below:
only the layout.

## What the layout is

**Bybit spot `ob200`.** A zip with one member, `2026-03-03_BTCUSDT_ob200.data`,
of JSON lines. Each line is
`{"topic": "orderbook.200.BTCUSDT", "ts": <ms>, "type": "snapshot" | "delta", "data": {"s", "b", "a", "u", "seq"}, "cts": <ms>}`,
the shape of Bybit's linear archive: `b` and `a` are lists of
`[price, size]` strings, a size of `"0"` deletes the level, `u` is the update
id and rises by one a message, `seq` is a cross-sequence. The first line is a
snapshot stamped in the day's first second; the fourteen messages the bytes
held were 200 ms apart, with consecutive `u`. Prices carry at most one
decimal, sizes up to six. `data/bybit.reconstruct` reads this shape
unchanged.

**Bybit spot prints.** A gzipped CSV with a header,
`id,timestamp,price,volume,side,rpi`: `id` an integer from 1 rising by one a
row, `timestamp` integer milliseconds (13 digits), `price` and `volume` decimal
strings (the size in BTC), `side` the aggressor's side in lower case (`buy`,
`sell`), `rpi` 0 or 1. Every row held had `rpi` 0. This is not the linear
print archive's layout (float seconds, `size`, `Buy`/`Sell`), so the spot
parser is new code, as registered.

**Binance BTCFDUSD `aggTrades`.** A zip with one member,
`BTCFDUSD-aggTrades-2026-03-03.csv`, with no header row and eight fields:
aggregate trade id, price, quantity, first trade id, last trade id, transact
time, `is_buyer_maker` (`True` / `False`) and `is_best_match` (`True` in every
row held). The transact time has 16 digits: **microseconds**, as Binance
documents for spot from 2025-01-01. `data/binance.parse_agg_trades` reads the
layout as it stands (headerless eight-field spot rows, the epoch unit taken by
magnitude).

## What it sets

Nothing in the registration. It fixes the readers' contracts:

- `spot/bybit_book.py` replays the `ob200` archive through
  `data/bybit.reconstruct` with the source tag `bybit-spot-ob200`;
- `spot/bybit_prints.py` parses `id,timestamp,price,volume,side,rpi`, with
  `rpi` 1 marking a print no order can reach;
- `spot/binance.py` adapts `parse_agg_trades`' output, with the side flag
  mapped as the venue documents it (`is_buyer_maker` true: a seller-initiated
  print).

Whether the whole of each day has these shapes (every line a snapshot or a
delta, `u` without gaps, a constant header, the share of `rpi` prints) is
stage 0's to record on the three reader-test days.
