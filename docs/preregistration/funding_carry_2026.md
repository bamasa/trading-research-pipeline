# Pre-registration: funding-rate carry on Binance BTCUSDT and ETHUSDT, 2026 and 2027

This document registers a study before any of its data is read. It is
committed together with
[`configs/funding_carry_2026.yaml`](../../configs/funding_carry_2026.yaml),
which holds the same values in the form the code will read; the two must
agree.

| | |
|---|---|
| **Written** | 2 November 2026, before any funding rate, kline, mark price or basis of either coin is fetched or opened for this study |
| **What it registers** | Whether a delta-neutral carry on Binance (long spot, short the USD-margined perpetual in equal quantity, one clip of 10,000 USDT a leg) earns more funding than it pays in fees and slippage and loses in basis, at the fees of an account without volume. Four slots under Holm with m = 4: always on and gated, on BTCUSDT and ETHUSDT. Registered here: the blocks, the dated and sourced fee table, the accounting, the gate and its grid, the kills, the placebos, the status rules, the data budget, the code to come and the amendment protocol |
| **Searched** | θ of the two gated slots only, each from a grid of nine values on D by a fixed rule. Nothing else is chosen on data |
| **Computed so far** | **Nothing** for this study ([what was known](#what-was-known-before-this-was-written)) |
| **How it changes** | Only under the [amendment protocol](#amendment-protocol): dated amendments appended at the end. Before any fetch, the owner's information-set amendment may come; after D and before H, the D amendment; before P, the owner's P statement and, if needed, the fee amendment |

Names that do not exist yet refer to [the code to come](#the-code-to-come).

## Why this study

The roadmap's funding-rate carry item. A perpetual future has no expiry; its
price is held near spot by a payment every settlement between longs and
shorts, the funding rate. When leveraged longs dominate, the rate is positive
and the short side is paid. Long spot and short the perpetual in the same
quantity of the coin, and the price moves cancel: what is left is the funding
received, the fees and slippage of opening and closing both legs, and the
change in the basis (perpetual minus spot) between entry and exit.

The question, in four forms fixed now: **on Binance, at the fees an account
without volume pays, does the hedged position earn a positive daily net on
months whose funding this repository has not read, and does a gate on
trailing funding improve on holding it always?**

## What was known before this was written

None of it is a funding rate, perpetual price or basis read by this study.

1. **Binance BTCUSDT daily futures bars, 2024-01-01 to 2026-09-30**, are on
   disk and drawn in the README's breaks figure; the scale breaks 2026-02-06
   up and 2026-08-14 down are named there. BTC's 2026 price path is therefore
   known, including H's months.
2. **The spot studies** (`spot_2026`, `spot_maker_2026`) read Binance BTC/FDUSD
   prints and Bybit spot BTCUSDT books on 2026-03-03 to 05 and on their W and
   D, 2026-04-01 to 07-31, which hold all of this study's H, and `spot_2026`'s
   H, 2026-08-08 to 10-21.
3. **The ETH replication** read Bybit ETHUSDT perpetual books and prints from
   late July to 2026-08-07, inside this study's E2.
4. **Bybit funding read earlier**: BTCUSDT from December 2024 to June 2025 (the
   2025 study), the market-making studies' 2024 instruments, and two
   reader-test days of December 2020 (the Bybit 2021 study).
5. **The author's 2025 research** worked on BTC/FDUSD and ETH/USDT spot in 2025.
6. **The drafting sessions' general knowledge** runs to about mid-2026, which
   overlaps H; no 2026 funding value was recalled or used, and none is
   written here.
7. **Binance's fee pages**, read on 2026-11-02 ([Fees](#fees)).
8. **The plan** `docs/agents_demo_funding_carry.md`, which this registration
   departs from where [the choices](#choices-and-their-reasons) say.

### The information set

The owner did not state, on 2026-11-02, when they last looked at funding rates
of either coin. **Registered now:**

- **The owner's information-set amendment.** One dated amendment, committed
  before any file of this study is fetched, may state the last date the owner
  looked at BTC or ETH funding and the span of dates that view showed. It sets
  `information_set.owner_last_look` and nothing else, and records the new
  null-placeholder sha256.
- **Inside H** means a last look on or after 2026-04-01, or a view whose span
  holds any day of 2026-04-01 to 06-30. **No statement is treated as inside
  H.** Item 6 above places H inside the drafting sessions' general knowledge
  in any case, without any funding value.
- **As registered, H is inside the information set.** H's statuses are
  computed and reported, and its kills stand; a slot that meets H's bar is
  recorded **candidate (H in the information set)**; every slot neither killed
  nor void on H is taken to P (m = 4 on P); and **P's statuses are the
  verdict**. Only P can confirm.
- **If the amendment states a last look before 2026-04-01 with a span that
  holds no day of H**, H is blind: a slot that meets H's bar is a candidate,
  only candidates are taken to P, and P confirms them. A statement that puts
  the look inside H changes nothing from the paragraph above.
- **P.** P is blind only if the owner states, in a dated amendment to this
  document committed before P opens, that no funding rate of either coin for
  any day of P was viewed. The statement changes the document only, never the
  configuration, so it can come after the code freeze. Without it, every status
  read on P carries "(P information set not stated)". The owner undertakes not
  to view a funding rate of either coin for a day of P before P's read.

## Expected outcome, stated in advance

The arithmetic (fees from [the table](#fees), slippage as registered):

- A round trip opens and closes both legs as a taker: 2 × (10 + 5) bp of fees
  and 4 × 2 bp of slippage, **38 bp of the clip**.
- Binance's funding formula carries an interest component of 0.01% a
  settlement; three settlements a day give 3 bp a day, 273 bp over H's 91
  days. C1 then nets about (273 − 38) / 91 ≈ **2.6 bp a day** before basis.
- C1 is killed by K1 only if the mean rate is below 38 / (3 × 91) ≈
  **0.14 bp a settlement**, before basis. The basis term telescopes over a
  held position (its block sum is the basis at exit less the basis at entry),
  so its daily noise sets σ more than its sum sets the mean.
- A gate re-times payments the always-on position collects anyway and pays
  38 bp each time it reopens, about 13 days of funding at 3 bp a day.

**Predictions, one per slot, fixed now:**

- **C1**: candidate (H in the information set) if most settlements sit at or
  above the interest component and the long-run σ of the daily series is under
  about 8 bp; inconclusive if σ is larger.
- **C2**: killed by K-placebo on H (a low θ keeps the gate nearly always on, so
  random gates cover the same days); with a high θ, inconclusive with fewer
  than 60 held days.
- **C3**: killed by K-placebo on H, by C2's arithmetic.
- **C4**: candidate (H in the information set) under C1's conditions;
  inconclusive the next most likely, ETH's basis being noisier.
- Not predicted: the sign of the basis term on H, and P's statuses.

## Data

Binance's public archive, `https://data.binance.vision/data/<market>/monthly/<kind>/<symbol>/...`,
monthly files only, each checked against its published `.CHECKSUM`:

| Plane | Market and kind | Use |
|---|---|---|
| spot 1h klines, last price | `spot/klines/<symbol>/1h` | spot fills (open), daily marks (close) |
| perpetual 1h klines, last price | `futures/um/klines/<symbol>/1h` | perpetual fills (open), daily marks (close) |
| perpetual 1h mark-price klines | `futures/um/markPriceKlines/<symbol>/1h` | funding notional (open at the settlement hour), K-liq (high) |
| funding | `futures/um/fundingRate/<symbol>` | `calc_time`, `funding_interval_hours`, `last_funding_rate` |

Symbols BTCUSDT and ETHUSDT on both markets, keyed by venue and market
(`binance-spot:BTCUSDT`, `binance-um:BTCUSDT`, and the same for ETHUSDT).
`data/binance.py` parses klines and funding; a mark-price kline parser is new
code.

- **Settlements.** A settlement's time is its `calc_time` floored to the hour;
  it belongs to the UTC day that holds it. Settlements per day are counted
  from the archive, never assumed, and K-units checks the count against
  `funding_interval_hours`. Rates are parsed from the text as integers in
  units of 1e-8.
- **The daily mark** of each leg is the close of the 1h kline ending at 00:00
  UTC of the next day (the 23:00 kline), last price (the owner, 2026-11-02).
  No position is open at a block's first midnight, so no mark from before a
  block is ever read.
- **A missing row** (one of a day's 24 klines of either leg, a mark-price
  kline at a settlement hour or in an hour a position is open) excludes the
  day on its pair, reason `missing_row`; the day's funding, fees and price
  changes are carried to the next usable day, so the block's sum is unchanged.
  Other reasons: `unreadable`, `unfetchable`, `fee not sourced`. **A pair's
  block is unreadable** when more than a quarter of its days are excluded; its
  slots are then `not assessed (<block> unreadable on <pair>)`.
- **D's first month** per pair is the first month at or after 2020-01 that
  every plane of the pair lists; it is recorded in the D amendment
  (`data.D_available_from`) and set by the archive's listing, not by any value.

## Fees

**The rule.** Every fill pays the verdict tier's taker fee of its instrument's
sourced interval in force at the fill's time, in bp of the fill's notional.
Funding carries no fee. An interval runs from its start, included, to its end,
excluded (UTC). A day is covered when every instant of it lies in an interval
for both instruments of its pair; a day not covered is excluded on that pair,
reason `fee not sourced`. The code reads the table from the configuration when
it runs, so a fee amendment changes no code. K-units compares every fill's fee
with the table's at a relative tolerance of 1e-12.

**The verdict tier** is Binance's **Regular** (VIP 0): no 30-day volume and no
holdings. **No BNB discount** on either leg (the owner, 2026-11-02). Zero fees
and maker fees on both legs are measurements, labelled "conditional on a
zero-fee venue" and "conditional on maker fills".

| Instrument | Interval (UTC) | Regular, maker / taker bp | Sources |
|---|---|---|---|
| Binance spot BTCUSDT | 2026-04-01 00:00 to 2027-04-01 00:00 | 10.0 / 10.0 | F1, F2, F3 |
| Binance spot ETHUSDT | 2026-04-01 00:00 to 2027-04-01 00:00 | 10.0 / 10.0 | F1, F2, F3 |
| Binance USD-M BTCUSDT perpetual | 2026-04-01 00:00 to 2027-04-01 00:00 | 2.0 / 5.0 | F4, F5 |
| Binance USD-M ETHUSDT perpetual | 2026-04-01 00:00 to 2027-04-01 00:00 | 2.0 / 5.0 | F4, F5 |

**Sources**, each read on 2026-11-02:

- F1 `https://www.binance.com/en/fee/trading`: Regular spot 0.100% / 0.100%
  without the BNB discount (0.075% / 0.075% with it).
- F2 `https://www.binance.com/bapi/composite/v1/public/commission/trade-level/get`:
  level 0 `makerCommission` 0.001 and `takerCommission` 0.001.
- F3 `https://www.binance.com/bapi/accounts/v1/public/commission/symbol-activity-list?currentPage=1&pageSize=500`:
  210 special-fee rows; neither BTCUSDT nor ETHUSDT is listed.
- F4 `https://www.binance.com/en/support/faq/binance-futures-trading-fees-360033544231`,
  read through `https://www.binance.com/bapi/composite/v1/public/cms/article/detail/query?articleCode=360033544231`
  (published 2019-09-08, updated 2026-05-01, UTC, from the page's metadata;
  read again on 2026-11-02 with the same text): "a Regular User's maker fee
  is 0.02% and a Regular User's taker fee is 0.05%" for USD-M; BNB discount
  10% (not applied).
- F5 `https://www.binance.com/en/fee/futureFee`, the USD-M fee table: it
  rendered no rows ("No records found") when read. The USD-M row rests on F4
  alone, a statement in the venue's own FAQ dated inside H.

**Coverage.** The table was read on one day. The owner registered it, on
2026-11-02, as covering H and P. For development only, D is computed at the
same table; D numbers are never results. **If Binance changes the Regular
table before P opens**, a dated fee amendment ends the registered interval at
the change's instant and adds the new interval from it, before P's read and
by P's deadline, and adds nothing else. Before P opens, the fee amendment's
procedure is run whether or not a change is expected: F1 to F5 read live, and
Binance's announcements searched for a Regular-tier spot or USD-M fee change
for these symbols, each read between 2027-04-01 and the amendment's commit;
where two sources disagree, the higher verdict-tier fee on each side is taken.
If nothing changed, the amendment records the reads and adds no interval.

## Blocks

Whole months throughout, because Binance publishes `fundingRate` by the month
only and a month archive holding a protected day counts as that day.

| Block | Dates (UTC, inclusive) | Use | Read | Why its edges |
|---|---|---|---|---|
| **D** | 2020-01-01 to 2020-11-30; 2021-07-01 to 2024-01-31; 2024-07-01 to 2025-06-30 | θ of C2 and C3, σ_D, the MDE, every slot's development series | freely, through the development access | December 2020 to June 2021 hold the Bybit 2021 study's reader-test days, E and H; February to June 2024 the market-making studies' H, buffer, F, P and Q; July 2025 on the 2025 study's E |
| **E** | 2025-07-01 to 2026-03-31 | never fetched | — | the 2025 study's E, H, F and P; the 2025 study's reader-test days 2026-01-05 to 07 (in the calendar as `spot_2026`'s days in no block); the spot studies' reader-test days 2026-03-03 to 05; February 2026 stays with them as one gap, since a one-month D range would barely pass its gate's 21 settlements and would end a month before H |
| **H** | 2026-04-01 to 2026-06-30 (91 days) | the four statuses | once, after the D amendment is committed | the last whole months before 2026-07, clear of every protected range |
| **E2** | 2026-07-01 to 2026-12-31 | never fetched | — | the ETH replication's days, the spot studies' E, H, P embargo and P |
| **P** | 2027-01-01 to 2027-03-31 (90 days) | the confirmation | once, after March 2027 is published on every plane, by 2027-04-30 | the first whole months after the last protected day, outside every information set |

Each D range is a block of its own for positions and gates: a position opens
at its first hour and closes at its last, and a gate counts 21 settlements
from its start. A position is held across days, so the days are not
independent: every status reads the Newey-West t of the daily series.

## Common settings

- **The clip**: 10,000 USDT of notional a leg at entry (the owner,
  2026-11-02); no capacity beyond one clip is claimed. The quantity is q =
  10,000 / the spot entry fill price (after slippage), and the perpetual leg
  shorts the same q. Quantities are not rounded to a lot step.
- **Fills**: taker on both legs, at the open of the 1h kline (last price) of
  each leg, the hour after the decision. Slippage **2 bp a side** on each leg,
  against the account (a buy higher, a sell lower); 0 and 1 bp are
  measurements.
- **Entry and exit.** C1 and C4 enter at the open of the 00:00 kline of the
  block's first day. **Every slot exits at the open of the 22:00 kline of the
  block's last day**, so that K-delay's exit, an hour later, stays inside the
  block. No fill is made after 22:00 of the last day (23:00 under K-delay); a
  later fill is made at that hour.
- **Funding.** A settlement is received when the position was filled strictly
  before the settlement time and is open at it: funding = q × the open of the
  1h mark-price kline starting at the settlement hour × the rate (the owner,
  2026-11-02, "charged on the mark price"). A positive rate pays the short.
- **Margin**: the perpetual leg at isolated margin of one times its notional,
  q × the perpetual entry fill price.
- **Capital**: both legs' notional, 20,000 USDT; the annual return on it is a
  measurement. **The denominator of every daily value is the hedged notional,
  10,000 USDT.**

## The slots

| Slot | Pair | Rule |
|---|---|---|
| **C1** | BTCUSDT | always on: open at the block's first hour, closed at its last fill hour |
| **C2** | BTCUSDT | gated, θ_BTC |
| **C3** | ETHUSDT | gated, θ_ETH |
| **C4** | ETHUSDT | always on, as C1 |

**The gate.** At each settlement of the block, from its 21st on, the position
is wanted open when the sum of the last 21 settled rates of the block, in
integer units of 1e-8 and the settlement's own rate included, **strictly
exceeds 21 × θ** (θ in bp a settlement, so 21 × θ × 1e4 units). A change of
state fills at the open of the next hour. No opening comes before the block's
21st settlement. Each opening is a new clip, with its own q.

## The development-period procedure (D only)

On each D range, each slot is run as on a held-out block. For C2 and C3, every
θ of the grid **{0, 0.25, 0.5, 0.75, 1.25, 1.5, 2, 3, 5} bp a settlement** is
run, and θ is the one with **the largest mean daily net over D's usable days**
(all three ranges pooled, flat days 0), at H's fee table and 2 bp slippage;
ties go to the larger θ. Nothing else is chosen. The D amendment records θ,
σ_D, the MDE, D's first months and excluded days, and the compute record; every
θ of the grid on D is reported as development, never as a result.

**The run log.** Every computation on D is logged in
`docs/funding_carry_2026_run_log.md` and counted in the trial ledger; the
nine θ for each of C2 and C3 count as nine trials each.

## Metrics

- **The daily net**, in bp of the hedged notional: (funding received + q ×
  the spot leg's price change − q × the perpetual leg's price change − fees −
  slippage) / 10,000 × 1e4, every usable day of the block. A day's price change
  runs from the previous day's mark to the day's mark; a fill replaces the mark
  on its side of the day. A day without a position scores 0.
- **The basis term**: q × (spot change − perpetual change), that is −q × the
  day's change of (perpetual − spot). **The parts** (funding, fees, slippage,
  basis) are reported apart, so that a result carried by basis is not read as
  carry.
- **The statistic**: the Newey-West t (Bartlett, lag 7) of the daily series,
  `evaluation/significance.newey_west_t`; its one-sided p is the Student t
  upper tail at n − 1 degrees of freedom, `market_making/verdicts.t_sf`; Holm is
  `verdicts.holm`.
- **A held day** is a day on which a position is open at any instant.
- **σ_D**: the Newey-West long-run standard deviation (lag 7) of the slot's D
  series at its frozen θ, pooled over D's ranges. **The MDE**: (3 + 0.84) ×
  σ_D / √(91 × the share of D's usable days held); on P, 90 in place of 91.
- **The fee break-even per side**: (gross − slippage) / turnover × 1e4,
  turnover the sum of every fill's notional on both legs.

## Status rules

**The family** is **{C1, C2, C3, C4}, with m = 4** on H and on P.

**The bar.** A slot passes a block when no kill fires and: its one-sided p
passes Holm at 5% with m = 4; its Newey-West t is at least 3; the median of
its daily series over every usable day (flat days 0) is positive; and it held
a position on at least **60 days of H (30 of P)**. A gated slot held on fewer
than half the block's days has a median of 0 and cannot pass; that is
registered, not an accident. A slot enters Holm with p = 1 when it is void,
killed, inconclusive with too few held days, not assessed, or not taken to
the block.

**Precedence**, one status per result:

1. **void**: K-acct, K-units, K-cap or K-leak fires; counted as killed.
2. **inconclusive (too few held days)**: K-held. Nothing below is evaluated.
3. **killed**: a kill fires; the first in the slot's order is reported.
4. **candidate (H in the information set)** on H (**candidate** if the owner's
   amendment makes H blind); **passes** on P: the bar is met.
5. **inconclusive (below the minimum detectable effect)**: no pass, no kill,
   and |mean| below the MDE.
6. **inconclusive**: otherwise.

**After P**, for each slot taken to P:

| On P | Status |
|---|---|
| passes | **confirmed** |
| no kill, does not pass | **not confirmed** |
| a kill fires | **killed on P** |
| a void condition fires | **void** |
| P unreadable on the pair | **not confirmed (P unreadable)** |
| P's read failed | **not confirmed (P read failed)** |
| not read by 2027-04-30 | **not confirmed (P not read by its deadline)** |

A slot not taken to P keeps its H status. Every P status carries "(P
information set not stated)" unless the owner's P statement was committed
before P opened. Until P is read, a slot taken to it reads "<H status> (P not
yet read)".

**Unreadable and failed reads.** A read that fails is resumed at the same
commit; if the same stage fails again with the same exception, or the read
has not finished after five resumptions, the session closes: on H every slot
is `not assessed (H read failed)`, on P as the table says. The family is
assessed only when every slot has values or one of these statuses.

**No number** at zero fees, at maker fees, at 0 or 1 bp of slippage, without
the basis term, or on D **may be described as the strategy working.**

### Kill conditions per slot

Void conditions, read first, for every slot:

- **K-acct**: the sum of the daily values × 10,000 / 1e4 differs from the
  ledger's net in USDT by more than 1e-9 × 10,000; or the reversed position
  (short spot, long perpetual) does not net minus the gross less the same fees
  and slippage.
- **K-units**: a rate or price reaches a result through a normalised column;
  a day's settlements differ from 24 / `funding_interval_hours`; a fill's fee
  differs from the table's.
- **K-cap**: more than one clip open, or the legs' quantities differ, at any
  instant.
- **K-leak**: the gate reads a rate settled after its decision.

Then **K-held**, which sets `inconclusive (too few held days)`. Then the kills,
in the order of the table:

| Slot | Order |
|---|---|
| C1, C4 | K1, K-fund, K-basis, K-liq, K-timing, K-delay, K-days, K-fee |
| C2, C3 | K1, K-fund, K-basis, K-liq, K-placebo, K-nbhd, K-delay, K-days, K-fee |

- **K1**: the mean daily net on the block ≤ 0.
- **K-fund**: funding received over the block ≤ fees and slippage paid.
- **K-basis**: the mean of the daily series without its basis term ≤ 0. As
  written it fires on the same blocks as K-fund (the series without basis sums
  to funding less costs); it is kept as proposed, and the basis share of the
  net is reported beside every slot ([choices](#choices-and-their-reasons)).
- **K-liq**: at any hour a position is open, the hour's high of the mark price,
  H_t, satisfies q × F0 − funding paid out of the margin − q × (H_t − F0) ≤
  mmr × q × H_t, with F0 the perpetual entry fill and **mmr = 0.05**, a
  conservative maintenance rate with no source, so K-liq fires no later than
  Binance would liquidate (the owner, 2026-11-02: the mark price's high at 1x
  isolated margin).
- **K-timing** (C1, C4): the median, over 200 runs (seeds 0 to 199), of the
  mean daily net with entry at a random hour open of the block's first 7 days
  and exit at a random hour open of its last 7 days (to 22:00 of the last day)
  ≤ 0.
- **K-placebo** (C2, C3): the true mean ≤ the 95th percentile of 200 random
  gates (seeds 0 to 199), or ≤ the gate shifted 21 settlements later.
- **K-nbhd** (C2, C3): the median, over the cells {θ's grid neighbour below,
  θ, its neighbour above} × windows {9, 21, 42} settlements (nine cells, six
  at an edge of the grid), of each cell's mean ≤ 0; each window opens no
  earlier than its own count of settlements.
- **K-delay**: a positive mean ≤ 0 with every entry and exit fill one hour
  later.
- **K-days**: half or fewer of the held days positive.
- **K-fee**: the break-even per side ≤ the notional-weighted mean Regular
  taker fee of the slot's fills (7.5 bp when both legs trade equal notional).
  Its use is the report of how far each slot was from paying.

### Placebos

Each runs through the same fills, fees, funding and day rules as the true slot
on the same block, drawn with `numpy.random.default_rng(seed)`.

| Placebo | Slots | What changes | Seeds |
|---|---|---|---|
| random timing | C1, C4 | entry uniform over the 168 hour opens of the first 7 days; exit uniform over the hour opens of the last 7 days up to 22:00 of the last day | 0 to 199 |
| random gate | C2, C3 | the true run's spells (lengths in settlements) in random order; the flat settlements after the 21st split into k + 1 gaps by a uniformly random composition | 0 to 199 |
| shifted gate | C2, C3 | the true gate's state 21 settlements later; a spell running past the block closes at the last fill hour | — |
| reversed position | all | short spot, long perpetual; an accounting check inside K-acct | — |

## Measurements without a hypothesis

Reported beside every slot, with no status: funding, fees, slippage and basis
apart, and the basis share of the net; the result at 0 and 1 bp of slippage,
at zero fees and at maker fees; the annual return on 20,000 USDT; the share of
settlements at or above 1 bp; the largest hourly mark-price rise against the
liquidation distance; and, on D only and labelled development, every θ of the
grid.

## Data budget

Monthly archives only: D's 54 months, H's 3 and P's 3, each month four planes
for each of two pairs. Under 100 MB in all, an estimate (nothing has been
listed or fetched). E and E2 are never fetched; no daily archive, no Bybit file
and no book or print file is fetched. Fetch rules: every listed month is
fetched, a failure other than HTTP 404 is retried with backoff, every archive
is checked against its `.CHECKSUM` before it is read, and files live under
`data/funding_carry_2026/`.

## Who may read what, and the code that will enforce it

### The code to come

No role writes study code. The owner writes, in new files of about 120 lines
with a test each, under `src/trading_research/carry/`: `archives.py` (monthly
URLs, checksums, the access every fetcher asks), `mark.py` (the mark-price
kline parser), `fees.py` (the table and its exclusion), `position.py` (the
hedged position, its fills, funding and identity), `series.py` (the daily
series and its parts), `gate.py`, `liquidation.py`, `placebos.py`,
`statuses.py`, `registration.py` (the loader and its `REGISTERED_SHA256`),
`ledger.py`, `state.py`, `runlog.py`, `pages.py` and `synthetic.py` (a local
venue for the tests and the dry run); the entry point
`experiments/funding_carry_2026.py`; and `tests/test_carry_*.py`. Reused by
import, unchanged: `data/binance.py`'s kline and funding parsers,
`evaluation/significance`, `market_making/verdicts` and the helpers of
`market_making/prereg`. The skeptic reviews the code in its three lenses;
every choice the code makes is written into this document before any day of a
block is fetched; the code-freeze note `docs/funding_carry_2026_code_freeze.md`
names the commit. No dependency is added.

### Access

**The development access** allows D's months and refuses every other month of
the four instruments. **The ledger**, `experiments/results/funding_carry_2026_ledger.json`,
guards H and P on the spot study's pattern: a block opens only if the D
amendment's frozen sha256 matches the committed configuration, the tree is
clean but for the ledger, the study code is the frozen commit's, and no path
under `data/funding_carry_2026/` names a month of the block; the open entry is
committed before the block's fetch; the read closes by writing
`experiments/results/funding_carry_2026_<block>.json` and its sha256 into the
ledger. A block opens only if, in addition, the configuration changed since the
code-freeze commit only in its placeholders, and the document only by
appending, with one ``Frozen configuration sha256:`` line and no
``Null-placeholder sha256:`` line after that commit. **P** opens once, only
after H's results are committed, March 2027 is published on every plane, the
fee amendment is committed, and by 2027-04-30; for every slot neither killed
nor void on H (only candidates if H is blind). Since H's read, only
`README.md`, `experiments/results/`, `docs/` outside the registrations, this
document's appended amendments and the fee intervals may have changed.
A second read is stamped `second read` and cannot change a status. H and P
are spent; a successor uses days after 2027-03-31.

## Amendment protocol

- **The information-set amendment**, before any fetch, sets
  `information_set.owner_last_look` and nothing else, and records the new
  hash on a line ``Null-placeholder sha256: `<hash>` `` under Amendments.
- **The D amendment**, after D and before H is read, written by the entry
  point and reviewed before it is committed, fills the six placeholders
  (`data.D_available_from`, `data.excluded_days_D`, `gate.theta_bp`,
  `metrics.sigma_d`, `metrics.mde_bp`, `compute.record`) and records the
  configuration's sha256 on a line ``Frozen configuration sha256: `<hash>` ``.
  Nothing else changes.
- **A correction before H** is a dated amendment that names the fault and, if
  it could change a D number, re-runs D in full. No amendment cites a number
  seen on the data as its reason, except a fault it names. An amendment that
  changes the configuration outside its placeholders records the new hash on a
  line ``Null-placeholder sha256: `<hash>` ``, before the code freeze.
- **The P statement and the fee amendment**, before P opens, by 2027-04-30;
  the P statement changes this document only, and the fee amendment adds
  intervals only.
- **After a read**, nothing changes and nothing is re-run under another rule. A
  defect found after a read is written to `docs/funding_carry_2026_defects.md`
  with the mechanism by which it could have raised a result, and every status
  it touches becomes `void (defect found after the <block> read)`.

## Artefacts and headline sentences

**Files**, written by the entry point: the ledger; `experiments/results/funding_carry_2026_H.json`
and `…_P.json`; and the page `experiments/results/funding_carry_2026_page.md`,
with one row per slot (status, first kill against
its reference, mean, Newey-West t, median day, held days, MDE, the parts,
placebo percentile, K-nbhd median, break-even per side).

**Headline sentences, fixed now.** Brackets are filled by the entry point and
never edited.

| Outcome | Sentence |
|---|---|
| Each slot on H | "[slot] on H, Binance [pair]: [status]; first kill [K-x] at [value] against [reference]." |
| C1 and C4 on H | "[slot] on H: funding [x] bp a day against fees and slippage [y] and basis [z]; [status]." |
| Each slot on P | "[slot] on P, Binance [pair]: [status]; first kill [K-x] at [value] against [reference]." |

How the brackets are filled: [pair] reads `BTCUSDT spot and perpetual` or
`ETHUSDT spot and perpetual`; [status] is the status as the precedence gives
it, with its marks. A slot with no kill reads `none` at `n/a` against `n/a`.
Values have two decimals and their unit: K-days' `(share of held days
positive)`, K-fee's `bp per side`, K-liq's `(mark high over liquidation
price)`, every other `bp`. [x], [y] and [z] are the block's means a day over
usable days, in bp of the hedged notional, [y] written as a positive cost. D
numbers are never quoted as results.

## What this study cannot establish

- **That H was blind.** As registered, H is inside the information set and only
  P confirms.
- **The fee any account paid on H's days.** The table was read on 2026-11-02
  and registered by the owner as covering H; the USD-M row rests on an FAQ
  statement, the table page having rendered no rows.
- **A real fill.** Fills are at hour opens with a fixed slippage; the clip's
  impact, queue and latency are not modelled.
- **Liquidation as Binance computes it.** mmr = 0.05 is conservative and has no
  source; funding received is not credited to the isolated margin.
- **Capacity** beyond one clip of 10,000 USDT a leg.
- **Counterparty, stablecoin and venue risk**: USDT is never converted, and a
  venue failure is not modelled.
- **Anything beyond two pairs on one venue and these blocks.**
- **When the ledger's entries were made, to anyone but the owner**: nothing is
  pushed, and the order rests on the local history.

## Choices and their reasons

Departures from `docs/agents_demo_funding_carry.md`, and choices of this
registration:

- **Four slots, m = 4**, where the plan had three: the proposal adds C4, ETH
  always on, so that each pair has an ungated slot to set the gate against.
- **The gate is a strict sum of 21 rates in integer units against 21 × θ**, θ
  from an explicit grid that holds 0: the plan's "mean exceeds θ" in floating
  point could tie at the clamp.
- **K-nbhd uses θ's grid neighbours**, not θ × {0.5, 0.75, 1.25, 1.5}, which
  collapse when θ = 0.
- **K-timing** is added for the always-on slots, which have no gate to
  randomise.
- **Funding on the mark price, K-liq on the mark price's high, the daily mark
  at the 23:00 kline's close, the clip, no BNB discount, P's deadline**: the
  owner's answers of 2026-11-02, where the plan had the perpetual's open and
  the hour's high.
- **Every exit at 22:00 of the last day** (the plan: the last hour), so that
  K-delay's later exit never reads the hour after the block, which lies in a
  protected block (E2 after H, an unpublished month after P).
- **H's reason.** The plan called H the last whole months before the spot
  studies' days; those studies' W and D cover April to July 2026, so BTC's spot
  path in H was read by this repository. H is kept because it is the last
  stretch clear of every protected range, and because, as registered, H is
  inside the information set anyway.
- **Every slot not killed on H goes to P**, where the plan took only a
  candidate: with H inside the information set, P is the only test.
- **The fee table** is one reading of 2026-11-02, registered by the owner as
  covering H and P, where the plan expected dated intervals across H.
- **Bybit funding as a measurement is dropped**: it would add a venue and a
  plane to the read and answers no hypothesis.
- **K-basis kept as proposed**, though it coincides with K-fund: changing a
  kill is the researcher's step, not the registrar's; the basis share is
  reported beside every slot.
- **mmr = 0.05**, unsourced and conservative: a kill can only fire earlier.
  The fee host this run lists was read for fees only, so no maintenance-margin
  bracket was read.
- **The owner's P statement is a line of this document**, not a value of the
  configuration, since it comes after the code freeze, when the configuration
  may change only in its placeholders and fee intervals.
- **No owner's date for the last look at funding** (the owner, 2026-11-02):
  as registered, H is inside the information set. A later statement can only
  narrow this, by the information-set amendment before any fetch; a statement
  placing the look inside H leaves every rule above as it stands.

---

The configuration as registered on 2 November 2026, with every placeholder
`null`, has sha256
`5ce30410bd3e6928ea92ce11db0ed68c206a52dd14912501813678ea63930500`, computed
with `python -m agents.workflows.research_loop null-hash configs/funding_carry_2026.yaml`
(the sha256 of `market_making.prereg.unfrozen_text` applied to the file). A
later null hash stands only on a ``Null-placeholder sha256:`` line under
Amendments.

## Amendments
