# Data contract

Everything downstream reads one of three frames. Pinning their meaning in one
place is what keeps the rest of the pipeline honest: a feature cannot quietly
depend on a column that exists for only one source, and a new source cannot
silently change what a column means.

Defined in [`src/trading_research/data/schema.py`](../src/trading_research/data/schema.py), checked by
[`src/trading_research/data/validate.py`](../src/trading_research/data/validate.py). Print either
contract with:

```bash
uv run trading-research describe-schema book
```

## Conventions

**Time.** `timestamp` is timezone-aware UTC. It denotes the moment information
became *observable*, not when it was written or received. Every look-ahead
guarantee in the project is stated against this column, so a naive timestamp is
a hard error rather than something to coerce.

**Units.** Prices are in quote currency, sizes in base currency. No implicit
scaling and no ticks-as-integers: a column means what it says.

**Versioning.** `SCHEMA_VERSION` is written into every dataset manifest. Any
change to the columns or to the meaning of a column bumps it, so stored data
cannot be silently misread by later code.

## The three planes

They are separate because their availability differs, and pretending otherwise
would hide a real limitation.

| Plane | One row is | Real source |
|---|---|---|
| `trades` | a trade or aggregated trade | public exchange archives |
| `book` | an order-book snapshot, *N* levels per side | a dedicated collector |
| `bars` | a closed bar at a stated interval | public exchange archives |

A full order book **cannot** be reconstructed from public trade archives. The
resting size that was never hit leaves no trace in the trade tape, so depth,
imbalance and queue position are unrecoverable. This is why the book plane is
exercised against synthetic data first: the features and their tests exist
before the collector does, rather than being written in a hurry afterwards.

## Trades

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `timestamp` | `datetime64[ns, UTC]` | UTC | exchange match time |
| `symbol` | `string` | | instrument identifier |
| `price` | `float64` | quote | execution price |
| `quantity` | `float64` | base | executed size, always positive |
| `is_buyer_maker` | `bool` | | `True` when the **buy** side was resting |
| `trade_id` | `int64` | | monotone per-symbol identifier |
| `source` | `string` | | provenance, per row |

### The sign convention

`is_buyer_maker` is kept in the exchange's own form rather than pre-signed,
because the conversion is easy to invert by accident and an inverted sign flips
the meaning of every order-flow feature without making anything fail.

`is_buyer_maker = True` means the buyer was resting, so the **seller** was the
aggressor and signed flow is **negative**. Use `schema.signed_quantity(df)`
rather than rederiving it.

### Bybit prints and funding

Bybit's prints (`data/bybit_trades.py`) are stored a file per instrument-day
with `timestamp`, `price`, `size` and `aggressor` (+1 a buyer crossed, -1 a
seller), and, for archives downloaded after the market-making simulator was
added, the venue's `match_id`. Rows are sorted stably by time, so prints sharing
a timestamp keep the archive's order; files written earlier may not have, and the
market-making loader imposes its own sweep order within a timestamp rather
than relying on either. Funding settlements (`data/bybit_funding.py`) are a
file per instrument-day with `timestamp` and `rate` (a fraction per settlement;
positive means longs pay). Both are read one day at a time by `load_day`.

## Book

Per snapshot: `timestamp`, `symbol`, `sequence_id`, `source`, and for each level
*i* from 0 (the touch) outward: `bid_price_i`, `bid_size_i`, `ask_price_i`,
`ask_size_i`.

Depth is a parameter, not a constant. A stream capped at five levels and a
full-depth reconstruction are both legitimate inputs, and a feature needing ten
levels should fail loudly on a five-level frame rather than quietly return
`NaN`.

### Derived quantities

`mid_price`, `spread`, `best_bid` and `best_ask` live with the schema rather
than in the feature registry, because labelling and mark-to-market must use the
same definition of price. If those two ever drift apart, a backtest reports
profit on a price nobody could have traded.

## Bars

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `timestamp` | `datetime64[ns, UTC]` | UTC | **close** of the bar |
| `open_time` | `datetime64[ns, UTC]` | UTC | open of the bar |
| `symbol` | `string` | | instrument identifier |
| `open`, `high`, `low`, `close` | `float64` | quote | the bar's prices |
| `volume` | `float64` | base | base volume traded in the bar |
| `source` | `string` | | provenance, with the interval: `binance-futures-um-klines-1d` |

One row is a closed bar. `timestamp` is the close, so a value stamped at
`timestamp` has seen the whole bar and nothing after it — stamping a bar at its
open would let a feature read a close that was still a day away. The interval
is part of `source`, so bars at two frequencies cannot be concatenated by
accident.

This is the coarse plane, and the only thing that reads it is regime
monitoring (`validation/structural_breaks.py`), where a year of market in a few
hundred rows is the point. Nothing at the trade or book level should be built
from it: a feature at bar frequency has already thrown away the order the
trades happened in.

## Quality checks

Findings are graded rather than fatal, and returned as a report. A handful of
crossed snapshots in a million is usually worth dropping; non-monotone
timestamps make a dataset not worth modelling. Only the caller can decide, so
the decision is explicit:

```python
report = validate_book(df)
report.raise_if_failed()
```

**Errors** — an invariant the pipeline relies on is broken:

- schema or dtype mismatch; timezone-naive or null timestamps
- timestamps decreasing within a symbol
- non-positive prices, negative sizes
- crossed book (best bid above best ask)
- level prices out of order away from the touch
- duplicate or non-increasing `sequence_id`; duplicate `trade_id`

**Warnings** — real, survivable, and always shown:

- locked book (zero spread), where costs and imbalance are degenerate
- sequence gaps: every later snapshot is unreliable until resynchronisation
- unusually long pauses between observations

**Gaps are reported, never repaired.** Interpolating across a gap invents
observations, and an invented observation is indistinguishable from a real one
by the time it reaches a model.

The pause threshold is relative to the dataset's own median spacing rather than
an absolute duration, so it adapts to the instrument instead of assuming a tick
rate. Gaps matter because rolling windows silently span them: a sixty-observation
window covering a two-hour outage measures something quite different from the
same window during normal trading.

## Storage

A dataset is a directory, not a file: one Parquet file per plane plus a
`manifest.json` recording provenance — source, schema version, row count, time
span, and for synthetic data the generator version and seed. A directory copied
elsewhere still knows what it is, which is the difference between a reproducible
demo and a plausible-looking one.

Reading normalises dtypes back to the contract and then validates. Parquet
round-trips most things faithfully but not all: string columns can return as
`object`, and a timezone can be dropped by an intermediate tool. A frame is
therefore either contract-compliant or raises, never silently degraded.

Datasets are never committed. See [`disclosure_policy.md`](disclosure_policy.md).
