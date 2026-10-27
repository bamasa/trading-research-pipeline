# Pre-registration: market making on Bybit's BTCUSDT perpetual at the 2021 fees, from prints only

This document registers a market-making study before any of its data is read.
It is committed together with
[`configs/mm_bybit_2021.yaml`](../../configs/mm_bybit_2021.yaml), which holds
the same values in the form the code will read, and the two must agree.

| | |
|---|---|
| **Committed** | 27 October 2026, on the local branch `mm-bitmex/prereg`, started from `spot/prereg` at `8270e6a`, before any Bybit print or funding file of 2020 or 2021 is fetched for it |
| **What it registers** | Whether a passive quoter at the touch earned money on Bybit's BTCUSDT linear perpetual in the first half of 2021, **at the fees every account paid then** (maker −2.5 bp, a rebate; taker +7.5 bp); the data model built from prints alone; a grid of 40 quoter cells; the choice on D and its nulls; the family {F, S0} with its kill conditions, placebos and status rules; the measurements; the data budget; the code to come; access and the amendment protocol |
| **Searched** | 40 cells on D, one chosen (F) by the neighbourhood median, the whole choice re-run under two nulls |
| **Computed so far** | **Nothing.** No Bybit print, funding or book file of 2020 or 2021 has been opened by this repository. What was known is listed [below](#what-was-known-before-this-was-written) |
| **How it changes** | Only under the [amendment protocol](#amendment-protocol): dated amendments appended at the end. One is planned, after D and before H is read |

The 2025 study's conventions hold unless this document says otherwise; its
sections are cited as "the 2025 registration, §name", and the spot study's as
"the spot registration, §name". Names that do not exist yet refer to [the code
to come](#the-code-to-come).

## Why this study

Two pre-registered rounds of market making in this repository (Bybit linear
perpetuals, February to May 2024) killed every hypothesis. They were priced at
a maker fee of +2.0 bp, and their measured break-even was a maker rebate of 1.1
to 3.1 bp. From late 2015 to early 2022 several venues paid **every** account a
maker rebate of 2.5 bp. The owner's question, in his words: find whether market
making paid on a venue that paid every account a maker rebate, **at the fees of
the time**, with everything pre-registered before the held-out block is read.

The first venue chosen, BitMEX (XBTUSD and ETHUSD, January to June 2020), could
not be studied: its public archive answered every request, the bucket listing
included, with HTTP 403 AccessDenied on 2026-10-27, and its REST API with a
CloudFront 403. Nothing was bought. This study moves to **Bybit**, which paid
the same −2.5 / +7.5 bp to every account on its inverse and USDT perpetuals
until March 2022, and whose print archive is open. Bybit published no order
book before 2023-01-18, so this is a **prints-only** market-making study: the
touch is inferred from the aggressor prints, and a resting order fills only on
evidence from prints.

**The prior is negative on the making part and open on the net**
([below](#expected-outcome-stated-in-advance)). The study is built so that a
negative answer says how far the quoter was from paying, with the rebate and
without it.

---

## What was known before this was written

Everything below was available when the design was chosen. None of it is a
2020 or 2021 Bybit print, funding rate (except item 5) or book computed on by
this repository.

1. **The plan** of 2026-10-08, a read-only survey of fees and public data by
   venue and year (sources keyed F1–F17 and D1–D7 there). It sourced Bybit's
   −2.5 / +7.5 bp schedule for every account ([Fees](#fees)) and the dates of
   Bybit's print archives (BTCUSDT from 2020-03-25), and found no free Bybit
   book before 2023-01-18. Its research streamed three BitMEX days outside
   every block of any study (2016-06-01, 2017-06-01, 2019-06-01) for format
   and row counts; nothing of Bybit 2020–2021.
2. **This repository's earlier studies** ([`docs/findings.md`](../findings.md)):
   the two market-making rounds on Bybit's 2024 books, whose back-of-queue
   quoter lost 1.1 to 5.0 bp of turnover before a rebate; the 2025 short-horizon
   study; the spot study. They are the reason for the prior. None read a 2020
   or 2021 day.
3. **Archive metadata, read for this registration on 2026-10-27**, by HTTP
   `HEAD` requests only: every daily print file of BTCUSDT and BTCUSD on Bybit's
   archive from 2020-12-15 to 2021-06-30, 396 requests, all 200. The sizes
   were seen; per block, BTCUSDT's gzip files total 0.047 GB on W, 0.487 GB on
   D, 0.024 GB on E and 0.635 GB on H (median 5.2 MB a day on D), BTCUSD's
   0.42, 3.64, 0.19 and 2.95 GB. The per-day results are committed as
   [`experiments/results/mm_bybit_2021_head_check.csv`](../../experiments/results/mm_bybit_2021_head_check.csv).
   No value here was set from them except the data budget.
4. **The BitMEX attempt** of the same day: the 403s above, and an accidental
   read. While checking a third-party vendor's free BitMEX files, the first
   300 bytes of the gzip stream of its XBTUSD quotes for 2020-01-01 were
   printed to a terminal, compressed and undecoded: no row was seen. It
   concerns another venue and sets nothing here.
5. **Endpoint probes of 2026-10-27**: Bybit's public funding history answered
   for BTCUSDT (linear) and BTCUSD (inverse) in late December 2020, and five
   BTCUSDT and three BTCUSD settlements of 2020-12-30 and 31 (days of W) were
   printed, showing settlements at 00:00, 08:00 and 16:00 UTC; Bybit's
   instrument endpoint answered for BTCUSDT with today's specification (tick
   0.10, quantity step 0.001, launch 2020-03-15). The 2021 tick and lot are
   derived from the prints ([Common settings](#common-settings)).
6. **General market knowledge.** BTC rose to a record in mid-April 2021 and fell
   by about half by late June; 2021-05-19 was a widely reported crash and
   liquidation day. H is, publicly, a falling market. A quoter carrying long
   inventory through H would lose without making markets badly, one carrying
   short inventory would gain: what K-dir and K-drift exist to catch.
7. **Nothing else.** No Bybit print, book or funding file of 2020 or 2021 has
   been fetched or opened by this repository, and no quoter has run on one.

### The information set

This repository has read none of the prints, books or funding of Bybit's
BTCUSDT perpetual in 2020 or 2021; the only exceptions are the W settlements
of item 5 and the file sizes of item 3. The owner's statement that he traded
nothing covers BitMEX; for Bybit in 2020–2021 no statement has been made, and
one may be added by a dated amendment before stage 0, changing nothing else.
**H is blind** to this repository: no print of it has been opened.

---

## Expected outcome, stated in advance

The arithmetic. BTCUSDT's tick in 2021 was a small fraction of a basis point
of price, so the touch earns almost no spread; a passive fill earns the 2.5 bp
rebate and loses the move that follows it. Under the registered fill rule a
resting order fills only when a print trades **through** its price, so every
fill comes with the market already one tick or more beyond it: the making part
(spread plus the five-second markout, before fees) is expected to be negative.
Whether the rebate covers it, and the 7.5 bp taker flatten at 23:59, is the
open question.

The plan's priors, written on 2026-10-08 before Bybit was chosen, are quoted
unchanged: a chance of 25–35% that the BitMEX XBTUSD cell ends paid with the
rebate, and 15% or less that any cell ends paid ex-rebate; it gave no number
for a Bybit cell, whose free data it rated "screen only". The task that
started this study restated the first as "XBTUSD 2020 25–35% chance of a real
positive ex-rebate; higher with the rebate"; that does not match the plan, and
the plan's wording is the one registered.

Predictions for this study, fixed now:

- **The making part is negative** for S0 and for F on H: K-dir fires for both
  (probability about 90%), so neither ends `candidate` (about 5%).
- **"Paid with the rebate"** (net > 0 at −2.5 / +7.5 bp): about 35–50% for S0
  on H; the rebate is larger than the expected loss per fill of the cheapest
  fills, and smaller than that of the fills during fast moves.
- **"Paid ex-rebate"** (net > 0 at a 0 bp maker fee with a positive making
  part): 5% or less.
- **The negative control** (every passive fill charged the 7.5 bp taker fee)
  is negative on every block; a positive value is a fault of the pipeline.
- Not predicted: which cell D chooses, and the sign of the inventory part.

---

## Data

| Plane | Source | Code | Blocks |
|---|---|---|---|
| BTCUSDT prints with the aggressor's side | `https://public.bybit.com/trading/BTCUSDT/BTCUSDT<day>.csv.gz` (one gzip CSV per UTC day) | `mm_bybit/archive.py` (new) | reader-test days, W, D, H |
| BTCUSDT funding settlements | `https://api.bybit.com/v5/market/funding/history`, `category=linear`, public, no key | `data/bybit_funding.fetch` (reused) and `mm_bybit/funding.py` (new) | W, D, H |
| A book | none: Bybit's book archive starts on 2023-01-18 | — | — |

The print file's columns, as documented by the plan's survey and the
repository's Bybit parser: `timestamp` (float seconds), `symbol`, `side` (the
aggressor: `Buy` lifted an offer, `Sell` hit a bid), `size` (BTC), `price`
(USDT), `tickDirection`, `trdMatchID`, `grossValue`, `homeNotional`,
`foreignNotional`. The survey found 2019 inverse files written **newest
first**; the reader therefore detects the file's direction (non-increasing
stamps) and reverses a newest-first file before a stable sort by stamp, so
prints sharing a stamp keep the venue's order. The reader's proof on the two
reader-test days records the direction, the stamp resolution, the share of
duplicate `trdMatchID`s and the rows stamped outside the day.

Prints are kept in the spot study's print contract (`spot/prints.py`:
`timestamp`, `price`, `size`, `aggressor`, `trade_id`, `rpi`, `source`), with
`trade_id` the row's position in the venue's order, `rpi` always false (Bybit
had no RPI orders in 2021) and `source` `bybit-linear-prints`. A day is the
prints stamped from its midnight to the next (UTC); rows of a file stamped
outside it are dropped and counted.

**BTCUSD (inverse) is not computed.** Its P&L in BTC is sourced (Bybit's help
article "P&L Calculations (Inverse Perpetual and Expiry Contracts)", read
through search snippets on 2026-10-27; the page itself timed out), but the
inverse fee and funding terms were not read on a page in full, and the
condition for measuring it beside was that its whole accounting be sourced.
It is neither fetched nor measured.

---

## Fees

| Interval (UTC) | Product | Maker (bp) | Taker (bp) | Who | Sources |
|---|---|---:|---:|---|---|
| 2020-03-22 to 2022-03-10 inclusive | inverse and USDT perpetuals, BTCUSDT included | **−2.5** | **+7.5** | every account | Bybit help article 360039261154, "Trading Fee Structure", Wayback captures 20200410051922 (article updated 2020-03-22), 20201221, 20210827, 20220111 (updated 2021-12-23) and 20220311223741 (announcing the change: some pairs from 2022-03-11 06:30 UTC, every pair from 2022-03-18 07:00 UTC) |

The captures were read on 2026-10-08 for the plan (its source F10); a re-read
on 2026-10-27 failed (the Wayback Machine answered "temporarily offline" from
this machine) and is retried before the code freeze; a capture that then
contradicts the table stops the study for an amendment. The interval ends on
2022-03-10, the day before the earliest announced change, since BTCUSDT's own
change date is not separately sourced. Every day of W, D and H lies inside it.
A day outside every sourced interval would be excluded as `fee not sourced`.

The fee comes from the table only: the run settings are built from it, and a
run given any other fee is refused. The maker fee is charged on every passive
fill's notional (a rebate is credited), the taker fee on the 23:59 flatten. The
quoters never read the fee, so a re-pricing of the same fills at another maker
fee is exact: **ex-rebate** (maker 0 bp), the **negative control** (maker
+7.5 bp, the taker fee, on every passive fill) and the **positive control**
(maker −10 bp). Each is a measurement.

---

## Blocks

| Block | Dates (UTC, inclusive) | Days | Use | Read |
|---|---|---:|---|---|
| Reader-test days | 2020-12-15 and 16 (the first two days of W) | 2 | the reader's proof and stage 0's timing; nothing scored | freely |
| **W** warm-up | 2020-12-15 to 12-31 | 17 | the volatility reference and the volatility gate's thresholds; never scored | freely, through the development access |
| **D** development | 2021-01-01 to 03-31 | 90 | the grid on every cell, the choice of F, the nulls, σ_D and the minimum detectable effects | freely, through the development access |
| **E** embargo | 2021-04-01 to 04-07 | 7 | a gap, so that H tests transfer rather than continuation | never fetched |
| **H** held out | 2021-04-08 to 06-30 | 84 | the statuses of F and S0, every measurement | once, after the D amendment is committed |

There is no P: the fee regime ended in March 2022 and the venue has no later
rebate open to every account. 2021-05-19, in H, is a **registered stress day**:
it is in every series of H like any other day, and its values are reported
beside every verdict, with H's mean without it as a measurement.

Days are independent: each starts flat, holds nothing across midnight, and is
the cluster of every statistic. Rolling state (the volatility) starts cold each
day. **A block is unreadable** when more than a quarter of its days are
excluded; on H, F and S0 are then `not assessed (H unreadable)`.

---

## Common settings

- **Instrument**: `bybit-linear:BTCUSDT`, P&L, fees and funding in USDT, sizes
  in BTC. The **tick** is derived each day from the day's prints, as the
  greatest common divisor of their distinct prices (`market_making/events.derive_tick`),
  and the **lot** from their sizes (`derive_lot`); a print off the day's tick
  grid cannot occur by construction, and a day whose tick differs from the
  value most usable days of W and D derive is noted, not excluded. The values
  are recorded in the amendment.
- **Clip**: 0.010 BTC, fixed (the repository's clip since 2025). **Soft limit**:
  6 clips. A side is quoted only if its fill would leave |position| at or under
  the soft limit, counting every order of that side not yet done; so the
  position never exceeds the soft limit, and no soft-limit taker order exists.
  **K-limit** voids a run whose |position| exceeds the soft limit plus one clip
  at any event (it can fire only through a fault).
- **Latency**: every order and every cancel takes effect **50 ms** after its
  decision (primary; Bybit's matching engine and API of 2021 were not
  colocated for an account without a market-maker agreement, and this is the
  conservative choice); 150 ms is K-latency's run, 10 ms a measurement.
- **The day**: no quoting before 00:10 (the volatility's warm-up) or from
  23:58, when every order is cancelled; at 23:59 the position is closed by a
  reduce-only taker order. No print for 5 s pulls every order at that moment;
  quoting resumes at the next decision.
- **Funding** settles at 00:00, 08:00 and 16:00 UTC from Bybit's history:
  `payment = −position × mark × rate`, the mark being the mid proxy at the
  settlement (prints stamped before it). Days start flat, so 00:00 never pays.
  A day with a settlement time missing from the history is excluded
  (`no_funding`); a settlement off the 8-hour grid stops the study for an
  amendment.
- **Accounting**: `market_making/accounting.Account`, unchanged: average cost,
  cash kept independently, the identity `realised + unrealised − fees + funding
  == cash + position × mark` checked after every fill and settlement and at the
  day's end, turnover equal to the sum of fill notionals (K-acct).

## The data model: prints only

There is no book. Everything a quoter sees, and every fill, comes from the
prints and their aggressor side.

**The touch proxies** are the spot study's (`spot/grid.py`, reused): the **ask
proxy** at a moment is the price of the last buyer-initiated print stamped at
or before it, the **bid proxy** the last seller-initiated print's, the **mid
proxy** their mean. A proxy is undefined until its side has printed that day.

**Decisions** are taken at the rows of the spot study's **print grid**
(`spot/grid.grid_stamps_ns`, 100 ms): one row per 100 ms interval holding a
print, stamped at its first print and carrying the state after every print
with that stamp. A quoter sees, at a row: the two proxies (in ticks), the mid
proxy, the one-minute volatility, its own position, the clip and the soft
limit. It sees no fill, markout or anything later.

**The one-minute volatility** is the simulator's (`market_making/simulator._per_second`
form) on the mid proxy: at every whole second, the mid proxy of the last print
at or before it; one-second log returns; an exponentially weighted mean of
their squares with a 60 s half-life, scaled to one minute, in bp; a decision at
`t` reads the value of the last whole second at or before `t`. Each day starts
cold.

**Orders.** One working order per side. A quote at an unchanged price keeps its
order; a new price is a cancel and a new order. An order is live from its
decision + latency and cancelled at its cancel's decision + latency. At an
equal stamp the market goes first: a print stamped at an order's arrival does
not fill it, a print stamped at its cancel's landing does. **Post-only**: on
arrival, a bid at or above the ask proxy, or an ask at or below the bid proxy,
is rejected and counted; a quoter's bid at or above its own ask is pulled apart
around their middle to one tick each side before anything is sent (the
simulator's R4).

**The fill rule (primary): trade-through.** The queue ahead of a resting order
is taken to be infinite. A live bid at price `p` fills only on a
**seller-initiated print strictly below `p`**, a live ask only on a
buyer-initiated print strictly above it: the level was exhausted, so the
order, resting there, was filled. Each such print fills at most its size, at
the order's price `p`, floored to the lot; what remains of the order waits for
the next one; several prints of one sweep fill in their stamp and archive
order. A print at exactly `p` never fills. It is the simulator's R7 alone
(`FillRule.TRADE_THROUGH` in the book simulator), the most pessimistic of its
fill rules, and here also the queue model: it replaces round two's K-queue.

**The brackets**, each run on every finalist and on S0:

- **touch** (optimistic, a measurement): a print of the hitting side at or
  through `p` fills the whole remaining order at `p`, at the first such print,
  whatever was queued ahead: the simulator's `FillRule.TOUCH`;
- **one further print** (K-delay): as the rule, but an order may fill only at
  prints stamped **strictly after** the stamp of the first print that traded
  through its price after its arrival: the fill comes one print stamp later,
  when the market has moved further.

**The 23:59 flatten** is priced by the spot study's side-correct print rule
(`spot/fills.fill_orders`, reused unchanged, `fallback=True`): a buy fills at
the first buyer-initiated print stamped at or after its arrival, sweeping the
prints of that stamp at their volume-weighted price, any part beyond the sweep
at its worst price; with no later print that day, at the last print of its side
before the arrival. It pays the 7.5 bp taker fee. Its fill time is its print's
stamp (or its arrival, for the fallback).

**Markouts and the decomposition**, per passive fill at price `p`, size `q`,
side `s` (+1 bought), stamp `t`: `mid_ref` is the mid proxy from prints stamped
strictly before `t`; `mid(t + h)` from prints stamped at or before `t + h`.

```
spread   = s × (mid_ref − p) × q
adverse  = s × (mid(t + 5 s) − mid_ref) × q
making   = spread + adverse = s × (mid(t + 5 s) − p) × q     (USDT, before fees)
rebate   = 2.5e-4 × p × q
inventory = net − making − rebate + taker fees − funding     (the residual)
```

`making` is what K-dir reads: the spread plus the five-second markout per
passive fill, before any fee. Markouts at 1, 5 and 30 s are reported in bp of
the fill's notional.

**Day exclusions** (reasons): `no_prints_file` (HTTP 404 only), `unreadable`
(a file that does not parse), `no_print` (no print on the day), `print_gap`
(two consecutive prints stamped between 00:10 and 23:59 more than 120 s
apart: a gap in the archive, not a quiet market, at this instrument's rate),
`no_funding`, `fee not sourced`. A day excluded is excluded from every series
of the block. Reported per block and reason.

---

## The quoters and the grid

Every cell quotes **at the touch proxies or not at all**: its bid at the bid
proxy, its ask at the ask proxy, one clip each. It never improves a proxy and
never quotes behind one. Each cell is one inventory rule and at most one gate.
No cell reads the fee.

**Inventory rule, λ.** With `q` the position, `Q` the soft limit, `σ` the
one-minute volatility and `σ_ref` W's median of it (the amendment records it),
the side that would add to |q| (the bid when long, the ask when short) is not
quoted while

```
λ × (σ / σ_ref) × |q| / Q  ≥  1/2
```

(`σ / σ_ref` taken as 1 while `σ` is undefined). λ = 0 never withdraws a side:
it is **S0**, the touch quoter of the earlier rounds. λ ∈ {0.5, 1, 2} is S1,
inventory-skewed: at the median volatility it stops adding at the soft limit, at
half of it, at a quarter of it; in faster markets sooner. The side that reduces
|q| is always quoted.

**Gates.** A gate closed cancels the side that would add to |q| (both sides when
flat); the reducing side stays at its price, so inventory can still unwind
passively (round two's rule).

- **Wide spread** `W(m)`: open while the proxy spread, ask proxy − bid proxy,
  is at least `m` ticks; m ∈ {2, 3, 5}.
- **Calm** `V≤(k)`: open while `σ` is at most W's `k`-quantile of `σ`;
  k ∈ {0.33, 0.67, 0.9}.
- **Busy** `V≥(k)`: open while `σ` is at least W's `k`-quantile;
  k ∈ {0.33, 0.67, 0.9}.

W's quantiles are of `σ` at every whole second from 00:10 to 23:58 of W's
usable days, computed once at stage 1 and recorded in the amendment, as is
`σ_ref` (the median of the same sample).

**The grid: 40 cells**, λ ∈ {0, 0.5, 1, 2} × gate ∈ {none, W(2), W(3), W(5),
V≤(0.33), V≤(0.67), V≤(0.9), V≥(0.33), V≥(0.67), V≥(0.9)}. A cell's key is
`l<λ>|<gate>`, for example `l0|none` (S0) or `l1|vle0.67`. **The
neighbourhood** of a cell is every cell within one step on λ (ordered 0, 0.5,
1, 2) and, for a gated cell, within one step on its gate's parameter at the same
gate kind, diagonals included, the cell itself included; an ungated cell's
neighbourhood lies along λ only. Not registered, and so not available later:
other parameters, a gate on any other quantity, inside quotes, layered quotes,
a per-side gate.

---

## The development-period procedure (D only)

**Stage 0** (the reader-test days, 2020-12-15 and 16): fetch, parse and prove
the reader (direction, stamp resolution, sides, duplicates, rows outside the
day, tick and lot), load the funding of those days, and time S0 and one gated
cell on each day. Nothing is scored. Then **the code freeze**: a dated note,
`docs/mm_bybit_2021_code_freeze.md`, records the commit of all study code
before any other W or D file is fetched; any later change to that code before H
is read is a correction under the [amendment protocol](#amendment-protocol).

**Stage 1**: fetch W and D (prints, then funding), each raw file deleted once
converted; compute `σ_ref` and the quantiles from W; then **every cell on every
usable D day** at the primary settings. Each day's prints are loaded once and
every cell runs on them. The run log (`docs/mm_bybit_2021_run_log.md`) and a
manifest of the D results (`experiments/results/mm_bybit_2021_D.json`, every
cell's daily series and summary) are committed.

**The choice** (the 2025 registration, §Selection, on this grid):

- each cell's daily series `y` ([Metrics](#metrics)) over D's usable days; a
  cell's **S** is the median, over its neighbourhood, of each cell's mean `y`;
- **admissible**: a cell averaging at least 20 passive fills a day over D;
- **F** is the admissible cell with the highest S; ties go to its own mean,
  then to the key's byte order. The outright peak (highest S whatever the
  fills) is reported beside it. If no cell is admissible, F is `not run (no
  admissible cell on D)`.

**The nulls: the whole choice re-run on placebo targets.** The statistic is
**A**, the median over the chosen cell's neighbourhood of each cell's
day-level t of `y` (0 for a cell with fewer than two days or no dispersion).
Each replication re-runs the admissibility filter (fills do not change), the
neighbourhood medians, the choice and A on its own target:

- **Null A, hour-block sign flips, 999 draws.** Each cell's P&L is split into
  the UTC hours of D's usable days (the change in its marked equity over the
  hour, fees and funding included; a day's hours sum to its net). Draw `b`
  (b = 0 to 998) gives each hour, in time order, the sign
  `validation/placebo_search.null_a_signs(b, hours)`, shared by every cell; a
  replication's `y` is the day's signed hours summed and normalised as `y`.
- **Null B, day re-pairing over every shift.** One balanced pattern of signs
  over D's `n` usable days, `+1` where `numpy.random.default_rng(0).permutation(n)`
  is even and `−1` where it is odd; under shift `s`, 7 ≤ s ≤ n − 7, day `i`
  (circular calendar order of the usable days) takes the pattern's sign at
  `(i + s) mod n`, shared by every cell, and its `y` is multiplied by it.

The 2025 study's Null B executed one day's entries on another day's prices; a
market maker's quotes react to each moment's prints and cannot be carried to
another day, and any re-ordering of days leaves each cell's mean unchanged.
Null B therefore re-pairs whole days with signs (keeping each day's internal
dependence, which Null A's hour flips break), and Null A keeps the dependence
between cells within each hour. Both are centred at zero for every cell, so
they measure how large the chosen cell's A looks when no cell's daily result
has a positive centre, after the choice.

`p_A = (1 + #{b : A_b ≥ A}) / (1 + 999)`, `p_B = (1 + #{s : A_s ≥ A}) / (1 +
number of shifts)`, counted directly (`placebo_search.placebo_p`).

**Admission of F**: (i) F's S > 0 and (iv) max(p_A, p_B) ≤ 0.05. A finalist
that fails is `not admitted on D (<conditions>)`: it is still run on H as a
labelled measurement and enters Holm with p = 1. S0 needs no admission: it is
fixed, not chosen. If F is `l0|none`, F is `same as S0` and enters Holm with
p = 1.

**Reported, gating nothing**, labelled "development, selected, biased upward":
every cell's mean `y`, S, fills a day and making part on D; F's and S0's D
decomposition; the positive control (S0 re-priced at a −10 bp maker fee, which
should be positive); the negative control; σ_D and the minimum detectable effect
of F and S0: `3.84 × σ_D / √84`.

---

## Metrics

- **`y`, the primary daily series**: the day's net in USDT per 100 USDT of
  clip, `y = 100 × net / (0.010 × m)`, `m` the mid proxy at 00:10 (the first
  quoting moment), fees and funding included. A usable day without a fill
  scores 0. It is round two's "net per 100 USDT of clip a day".
- Beside it, per finalist and block: **net bp per fill**, the block's net over
  its passive turnover × 1e4 (turnover-weighted); the same at a 0 bp maker fee
  (**net ex-rebate**); the **making part** per day (as `y`) and in bp per fill;
  the decomposition (spread, adverse, rebate, taker fees, funding, inventory);
  **fills a day** and passive turnover a day; the **share of days positive**;
  the median day; markouts at 1, 5 and 30 s; the maximum |position| and the
  share of time at the soft limit; the flatten's cost; the share of quoting
  time each gate is open.
- **The daily series decides.** The mean, the day-level t
  (`market_making/verdicts.day_t`), its one-sided p, the Newey–West t (Bartlett,
  lag 5; `evaluation/significance.newey_west_t`), the median day, K1, K-days,
  the placebos, K-nbhd, σ_D and the MDE are computed on the daily series,
  never on fills; Holm is `verdicts.holm`.
- **Capacity**, per day: the larger over the two sides of the cell's passive
  fill volume against the volume of the prints of the side that fills it (bids
  against seller-initiated prints, asks against buyer-initiated), in BTC.

---

## Status rules

**The family** is {F, S0}, **m = 2**, on H. F is the cell chosen on D; S0 is
fixed and enters whatever D shows.

**The bar.** A primary passes when no kill fires and all of these hold on its
daily series over H's usable days: its one-sided day-level p passes Holm at
5% with m = 2; the day-level t is at least 3; the Newey–West t is at least 2;
the median day is positive; at least 20 usable days. At m = 2 Holm's first
step needs a t of about 2.2 at 83 degrees of freedom, so the t ≥ 3 bar binds.

**Precedence**, one status per primary:

1. **void** — K-acct or K-limit fires in any run of its read; counted as
   killed.
2. **inconclusive (too few fills)** — K-trades: fewer than 100 passive fills
   over H. Nothing below is evaluated.
3. **killed** — any kill condition fires.
4. **inconclusive (capacity)** — the bar is met but the median day's capacity
   is above 10% (K-cap). Applied after Holm; changes no p.
5. **candidate** — the bar is met and the cap does not apply. With no further
   block, candidate is the highest status this study can give.
6. **inconclusive** — otherwise; "below the minimum detectable effect" when
   |mean| is under the MDE.

A primary enters Holm with p = 1 when it is void, killed, inconclusive with
too few fills, not admitted on D, the same as S0, or not assessed.

### Kill conditions

In the order a first kill is named:

- **K1** — the mean of `y` over H is ≤ 0 (at −2.5 / +7.5 bp).
- **K-dir** — the mean daily making part (spread plus the five-second markout
  of every passive fill, before fees, per 100 USDT of clip) is ≤ 0: the rebate
  alone may not carry a candidate.
- **K-delay** — a positive result whose mean `y` under the one-further-print
  bracket is ≤ 0.
- **K-latency** — a positive result whose mean `y` at 150 ms is ≤ 0.
- **K-fund** — a positive result whose mean `y` less its funding is ≤ 0.
- **K-drift** — a positive result whose mean inventory part is more than half
  its mean `y`: a result carried by holding positions.
- **K-days** — half or fewer of H's usable days positive.
- **K-nbhd** — the median, over F's D-neighbourhood, of each cell's mean `y`
  on H is ≤ 0 (for S0, its neighbourhood in the grid: λ ∈ {0, 0.5}, ungated).
- **K-placebo** — the mean `y` does not exceed the 95th percentile of 200
  hour-block sign-flip draws of its own H hours (seeds 0–199, `null_a_signs`);
  for a gated cell, also not the mean of its **shifted-state** run, in which
  each day's gate reads the state of the previous usable day of H (the first
  day the last day's) at the same second of the day, on a one-second grid.

Void conditions: **K-acct** (the identity or the turnover check fails on any
day of any run), **K-limit** (|position| beyond the soft limit plus one clip
at any event). Not kills: **K-trades** and **K-cap**, above.

### Labels beside, gating nothing

For each primary on H, with its t:

- **paid with the rebate**: mean `y` > 0 at −2.5 / +7.5 bp;
- **paid ex-rebate**: the mean of `y` re-priced at a 0 bp maker fee > 0, and
  the making part > 0;
- **paid at the touch only**: neither under the rule, but the touch bracket's
  mean `y` > 0 (an upper bound, not a result).

No number re-priced, bracketed, at another latency, of a non-admitted finalist
or of a measurement **may be described as the strategy working**. Only a
status says that.

---

## Measurements without a hypothesis

On H, for F and S0, each labelled with what changed:

- the **touch** bracket (optimistic) and the **10 ms** run;
- **net ex-rebate** (maker 0 bp), the **negative control** (maker +7.5 bp on
  every passive fill; expected negative, a positive value is a fault) and the
  **positive control** (maker −10 bp);
- **the stress day** 2021-05-19: its `y`, making part, fills and inventory
  part, and H's mean without it;
- `y` by calendar month of H;
- markouts at 1, 5 and 30 s against the **market-wide passive markout** of
  every print scored from its resting side, at the same horizons, on the mid
  proxy;
- on D, the same for every cell (development, selected).

---

## Data budget and compute

| Block | Prints, gzip (HEAD) | Kept as Parquet (estimate) |
|---|---:|---:|
| W | 0.047 GB | under 0.1 GB |
| D | 0.487 GB | under 1 GB |
| H | 0.635 GB | under 1.2 GB |
| E | 0.024 GB | never fetched |

At most 20 GB of the study's files on disk at any time; in practice under
3 GB. Raw files are deleted once converted, and only aggregates are committed.
Funding is a few kilobytes. Compute is measured at stage 0 (seconds per cell
and day) and recorded; the D run is 40 cells × 90 days.

---

## Who may read what, and the code that will enforce it

### The code to come

All of it new files: `src/trading_research/mm_bybit/` (the reader and the
archive's URLs and checks; funding; the day's tape of prints, proxies, rows
and volatility; the quoters; the engine; the day's accounting and metrics;
the choice and the nulls; the statuses; the ledger; the stages; a synthetic
venue for the tests and the dry run), `experiments/mm_bybit_2021.py` (the
single entry point, stages 0 to H's read), `tests/test_mm_bybit_*.py`
(synthetic fixtures only, no network). Reused by import and unchanged:
`market_making/accounting.Account`, `market_making/events.derive_tick`,
`derive_lot` and `day_start_ns`, `market_making/verdicts` (`day_t`, `holm`),
`market_making/prereg` (`placeholder_lines`, `unfrozen_text`, `sha256_bytes`,
`render_value`), `evaluation/significance.newey_west_t`,
`validation/placebo_search` (`null_a_signs`, `placebo_p`), `spot/grid`
(`grid_stamps_ns`, `proxies_at`), `spot/prints` (the contract), `spot/fills`
(`PrintTape`, `fill_orders`, `PrintFillSettings`) and `data/bybit_funding`
(`fetch`, `parse`). Nothing existing is edited; nothing is written under
`data/{book,trades,bars,funding}/`; this study's data root is
`data/mm_bybit_2021/`.

### Access

**The development access** allows the reader-test days, W and D, and refuses
every other day of 2020 to 2022. **E is never fetched.** **H** opens only
through the ledger, `experiments/results/mm_bybit_2021_ledger.json`, which
refuses unless: the amendment records the configuration's sha256 and the file
on disk has it, fully frozen, differing from the registered file only in its
placeholders; the document and the configuration are committed and the tree
is clean but for the ledger; the study code (`src/trading_research/mm_bybit/`
and `experiments/mm_bybit_2021.py`) is unchanged since the commit the
amendment records under `compute.code_freeze_commit`, or since a later
correction the amendments name; and no file of H is on disk. The open entry
(block, commit, configuration sha256, UTC time) is committed before H's fetch
begins. The read closes by writing `experiments/results/mm_bybit_2021_H.json`
with every status, kill and measurement, and its sha256 into the ledger. A
second read raises unless forced, and what it produces is stamped `second
read` and changes no status. Nothing is pushed.

---

## Amendment protocol

**The D amendment**, after D and before H is read, is one dated entry under
[Amendments](#amendments), written by the entry point and reviewed before it
is committed. It records: the days excluded on W and D with their reasons;
the reader-test results; the tick and lot; `σ_ref` and the quantiles; F, its
neighbourhood, the outright peak, S, A, p_A, p_B and the admission; σ_D and
the MDEs; the compute record (the code-freeze commit, the D runs, the run
log's sha256, the seconds); and the configuration's sha256 after these values
are written into its placeholders, on a line of the form
``Frozen configuration sha256: `<hash>` ``. Nothing else may change.

**A correction before H** (a fault in the reader, the engine, the accounting
or the statistics) is made only by a further dated amendment that states it
and, if it could change a D number, re-runs D in full before anything is
frozen; D runs are numbered. No amendment may cite a D number as its reason
except a fault it names. An amendment that changes the configuration outside
its placeholders records the new null-placeholder sha256 on a line of the form
``Null-placeholder sha256: `<hash>` ``, before the code freeze.

**After H is read**, nothing is changed and nothing is re-run under another
rule. A defect found after the read can only lower a status: it is stated in
`docs/mm_bybit_2021_defects.md`, and every status it could affect becomes
`void (defect found after the H read)`.

---

## Artefacts and headline sentences

**Files**, written by the entry point: the ledger; `mm_bybit_2021_D.json` and
`mm_bybit_2021_H.json` under `experiments/results/`; the run log; and a table
with one row per primary (status, first kill with its value and reference,
mean `y`, t, Newey–West t, median day, net bp per fill, net ex-rebate, making
part, fills a day, share of days positive, placebo percentile, shifted-state
mean, K-nbhd median, the brackets, the latencies, capacity, MDE, the labels).

**Headline sentences, fixed now.** The bracketed parts are filled by the entry
point and never edited.

| Outcome | Sentence |
|---|---|
| Each primary | "[F or S0] ([cell]) on Bybit BTCUSDT, 8 April to 30 June 2021, at −2.5 / +7.5 bp: [status]; first kill [K-x] at [value] against [reference]." |
| The question | "At the fees every account paid in 2021, a touch quoter filled only on trade-throughs [made / lost] [x] bp per fill ([y] USDT per 100 USDT of clip a day); without the rebate it [made / lost] [z] bp per fill, of which the making part was [w] bp." |
| The labels | "[F or S0]: [paid with the rebate / not paid with the rebate]; [paid ex-rebate / not paid ex-rebate]." |

The first kill is the first that fires in the [order above](#kill-conditions);
a primary with no kill reads `none` at `n/a` against `n/a`. Values have two
decimals and their unit. "The question" is written for S0 and for F, from
their H runs under the rule. D numbers are never quoted as results.

---

## What this study cannot establish

- **What a queue position would have earned.** There is no book: the
  trade-through rule assumes an infinite queue ahead, so it is a lower bound on
  fills and an upper bound on their toxicity, and the touch bracket is an
  upper bound on fills. Neither is a measurement of a real resting order.
- **The touch itself.** The proxies are the last prints of each side, not the
  quotes; a stale proxy puts an order away from the touch.
- **Today.** No venue pays every account a 2.5 bp rebate in 2026; a positive
  status is a statement about 2021, not a strategy.
- **Other instruments, venues or years**, and BitMEX, whose archive closed.
- **Latency below 50 ms.** The 10 ms run is a measurement.

---

## Choices and their reasons

- **BTCUSDT, not BTCUSD**: the linear accounting is the repository's, already
  tested; the inverse's fee and funding terms were not read in full.
- **A fixed clip of 0.010 BTC**: no book means no touch size to scale by; the
  clip is the repository's, small against every print volume a day, and the
  capacity cap checks it.
- **Trade-through as the rule, not the touch**: without a queue, only a print
  through the price proves a resting order filled; the touch is reported as the
  optimistic bracket.
- **The proxies at decision**: the only touch prints reveal; the spot study's
  code computes them.
- **50 ms**: conservative for an account without colocation in 2021; 150 ms is
  the kill; 10 ms is shown.
- **A gate on the proxy spread and one on volatility**: the two state variables
  prints give without a book; the volatility's thresholds come from W so that
  nothing in a cell is set on D.
- **No soft-limit taker order**: the limits are enforced before quoting, so the
  only taker order is the 23:59 flatten.
- **Null B as day re-pairing of signs**: [above](#the-development-period-procedure-d-only).
- **The family of two**: the chosen cell and the fixed touch quoter; S0 needs no
  admission, and keeps a clean answer if the choice adds nothing.
- **The 120 s print gap**: at a median of 5 MB of prints a day, two minutes
  without a print in BTCUSDT's 2021 market is an archive gap, not a quiet
  market; the count of such days is reported.

---

## Amendments

### 2026-10-27: the registration

Null-placeholder sha256: `0ece0d7d66099b86f8eac46d7e73abdfe2b1219585a2da590bf821b54da36558`
