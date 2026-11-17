# Pre-registration: S25-PF as a maker on Binance BTC/FDUSD, at the fees charged, on 2026 data

This document registers a study before any of its data is read. It is
committed together with
[`configs/spot_maker_2026.yaml`](../../configs/spot_maker_2026.yaml), which
holds the same values in the form the code will read, and the two must agree.

| | |
|---|---|
| **Committed** | 27 October 2026, on the local branch `spot-maker/prereg`, cut from `spot/prereg` at `8270e6a` (after the spot study's held-out results were committed), before this study copied, opened or computed on any print of any year |
| **What it registers** | Whether a maker version of S25-PF, the spot study's prints-only variant of S25, pays on Binance spot BTC/FDUSD at the fees a Regular account is charged: 0 on a resting order's fill, 10 bp on a taker's. Three slots, MK-S (the signal as a maker), MK-Q (two-sided quoting without a signal, the control) and MK-SQ (quoting with the signal's side doubled), their fill rule on prints, kill conditions, placebos and measurements, the status rules under Holm with m = 3, the blocks, the data and the amendment protocol |
| **Searched** | **Nothing.** The signal is the spot study's frozen fit and thresholds; every quoting coordinate is fixed here. Block D estimates the spread of daily results and chooses nothing |
| **Computed so far** | **Nothing.** This study has copied, opened or computed on no print. What was known is listed [below](#what-was-known-before-this-was-written) |
| **How it changes** | Only under the [amendment protocol](#amendment-protocol): dated amendments appended at the end; one is planned, after block D and before block H is read |

The spot study's conventions ([`spot_2026.md`](spot_2026.md), cited as "the
spot registration, §name") hold here unless this document says otherwise.
Names that do not exist yet refer to [the code to come](#the-code-to-come).

## Why this study

The spot study tested S25 on spot as a taker and found every slot killed by
K1 at the verdict tier on H. Its primary slot, S25-PF on Binance BTC/FDUSD,
paid a taker fee of 10 bp on each side, while Binance charges **nothing** on a
resting order's fill on this pair, to every account, from 2024-04-25 06:00
UTC to the end of the sourced fee table (the spot registration, §Fees, B3 to
B8). This study asks one question, in three forms fixed in advance: **does a
maker version of S25-PF pay on BTC/FDUSD at the fees actually charged, on
days whose prints the author never saw?**

- **MK-S** posts S25-PF's decision as a resting order at the touch proxy of
  its own side and exits with a resting order at the other side's proxy.
- **MK-Q** quotes both sides at the touch proxies with no signal: the
  control, which says what quoting alone earns on this pair.
- **MK-SQ** quotes both sides with the signal's side doubled.

Binance publishes no spot book, so a resting order's fill is inferred from
prints by a rule fixed [below](#fills) whose pessimistic form is the verdict.
Every verdict is taken at the fees a Regular account paid on each day.

**The prior is negative** ([below](#expected-outcome-stated-in-advance)).

---

## What was known before this was written

Everything below was available when the design was chosen. None of it is a
number this study computed.

1. **The spot study**: its registration, configuration, code and frozen
   values on `spot/prereg`, and everything its own registration lists as
   known (its items 1 to 8). In particular: its configuration at its frozen
   sha256 `d1625173c0b7a15c8785a957d77596210f46893e792c834449152e39cf5476b7`,
   with the D amendment's development numbers (for S25-PF on D: 77.5 trades a
   day, a gross edge of 2.41 bp a round trip at zero fees, an IC of 0.150),
   its stage 0 records on the reader-test days (the median sweep 0.30 to 0.39
   of a clip on BTC/FDUSD), the tick 0.01 and lot 1e-05 of BTC/FDUSD, the
   frozen thresholds of S25-PF and the sha256 of every frozen file.
2. **The spot study's confirmed information set**: the author's information
   on BTC/FDUSD and the knowledge behind S25's coordinates end on 2025-12-31
   (the spot registration's amendment of 2026-10-25). No print of D, H or P
   was seen by the author.
3. **The spot study's H results page**,
   [`experiments/results/spot_2026_page.md`](../../experiments/results/spot_2026_page.md),
   read in full on 2026-10-27. It holds taker results only. For S25-PF on H:
   killed by K1 at −17.35 bp a round trip at Regular; **gross +2.65 bp a round
   trip at zero fees**, 1.33 bp of turnover per side; 4,427 trades over 75
   days; placebo percentile 98.5; K-nbhd median −17.44; brackets optimistic
   −17.26, pessimistic −23.17, stale −22.95, worse of two −28.02; VIP 3 −9.35
   and VIP 9 −1.95. For S25-PFT: −17.71, zero fees +2.29. The thinned
   calibration: a_buy −2.91 and a_sell −3.17 bp, capacity median 0.0044; and
   the Bybit rows (S25-BB, S25-T, S25-PB, FILL-S). The results file
   `experiments/results/spot_2026_H.json`, its per-day records, the spot
   ledger and the spot run log were **not opened** for this registration.
4. **A file listing**, on 2026-10-27, of the spot study's converted
   BTCFDUSD day files in its checkout: 200 files (the three reader-test days,
   W, D and H's 75 days; no day of E), their sizes per month (under
   [data](#data)) and H's five smallest files (70.6 to 133.1 KB). They set the
   copy budget and nothing else.
5. **The 2025 registration's maker slot S-M**: its making part (the 5 s
   markout over passive fills, in bp per fill) and K-dir, reused here as
   definitions. That study's held-out reads have not happened and none of its
   results is known.
6. **Not known**: the width of BTC/FDUSD's touch in 2026, any markout of a
   passive fill, any fill rate of a resting order and any statistic of the
   touch proxies on any day. No such number was computed by this repository.

### The information set

- **The signal is frozen.** S25-PF's fit (hold 50 s, label fee 0) and its
  hold-first thresholds at 240 a day were made by the spot study on D: fitted
  on 2026-05-01 to 07-24, thresholds on 07-25 to 31. Nothing is refitted, on H
  or anywhere.
- **H was never seen by the author** (item 2).
- **The spot study's reads of H are spent for its slots only.** This study's
  slots are new: no maker number, no resting-order fill, no passive markout
  and no touch-proxy statistic on H was computed by anyone in this repository
  (item 6). The spot H read showed taker results only.
- **What H's taker results could inform here, listed in full**: (a) the
  positive gross edge of S25-PF at zero fees on H is why this study exists:
  it selected the question; (b) the thinned calibration (the print rule worse
  than a book walk by about 3 bp a side on thinned prints) says taker prints
  on this pair are a poor proxy of a touch, which motivated a pessimistic
  maker rule, fixed here without any maker number; (c) the trade rate (59 a
  day) and the capacity median (0.0044) describe the taker slot. No
  coordinate below was set from a number of H: the requote interval, the
  skew, the limits, the hold and the clip are the owner's specification and
  S25's.
- **So MK-S and MK-SQ are not blind to the sign of their signal's gross
  edge on H's days at taker fills.** A positive MK-S on H would say that
  resting orders kept an edge already seen to exist at taker fills on the
  same days; it would not confirm the signal out of sample. MK-Q uses no
  signal.
- **The spot registration's successor rule.** It spends H and P for its
  slots and sends "a successor registration, a corrected re-run included" to
  days after 2026-12-31, never its reader-test days. By the owner's decision
  of 2026-10-27, maker slots are new slots, not successors or corrections of
  the taker slots, and this study reads H once under its own ledger and uses
  the reader-test days for its readers' proof and timing only. That is a
  departure from the spot registration's text, made knowingly and stated
  here.
- **Consequence, registered now**: H can make a slot no more than a
  **candidate**; P (2026-11-01 to 12-31, after this registration, outside
  every information set) is the only confirmatory block, as in the spot study.

---

## Expected outcome, stated in advance

**No slot is expected to pay at the verdict fees.** The arithmetic:

- S25-PF earned +2.65 bp a round trip before fees on H at taker fills, +1.33
  bp per side of turnover (item 3). A taker crosses the touch on both legs;
  a maker pays no fee and, if it were filled at the same instants, would
  also earn the touch's width.
- But a resting order is filled only when the market trades through it (the
  registered rule): a long signal's buy at the bid fills when sellers push the
  price below the bid, mostly on the decisions where the signal was wrong;
  when it was right the price leaves and the entry is cancelled unfilled
  after 50 s. The filled decisions are selected against the signal.
- BTC/FDUSD's touch is likely one tick, 0.01 FDUSD on a price near 110,000,
  about 0.001 bp: no spread is earned at the touch, and a fill that the market
  traded through starts with a negative markout.
- An exit still resting at 23:59 pays 10 bp of its notional as a taker; a
  few such exits a day cost more than the whole taker gross edge.
- Plain quoting on a thin pair with a one-tick touch earns no spread and is
  filled by the flow that moves the price: its making part is negative.

**Predictions, one per slot, fixed now** (the verdict unit is bp per fill, half
a round trip's notional; per round trip in brackets):

- **MK-S**: killed by K1 on H, with a net between −1 and 0 bp per fill (−2 to
  0 bp a round trip), inside the owner's range of −2 to +2 bp a round trip:
  the negative half is predicted.
- **MK-Q**: killed by K1 on H, its making part negative (K-dir also fires).
- **MK-SQ**: killed by K1 on H, its net between MK-Q's and MK-S's.
- Not predicted: the fill rates, the sign of the touch bracket's result and
  the size of the making part.

---

## Data

| Plane | Source | Blocks |
|---|---|---|
| Binance spot BTCFDUSD aggregated prints with the aggressor's side, in the spot print contract | the spot study's converted day files, `data/spot/binance/prints/BTCFDUSD/<day>.parquet` in its checkout (ignored by git), which it fetched, checked against Binance's `.CHECKSUM` and converted by its strict parser | reader-test days, W, D, H (and P) |

**Copies, not fetches.** Every day this study may read is already on disk in
the spot study's checkout. This study copies each file, read-only at the
source and only inside an access that permits its day, to its own
`data/spot/binance/prints/BTCFDUSD/<day>.parquet`; the copy's sha256 equals the
source's, recorded per file in `data/spot_maker_2026/copies.json` (never
committed), whose sha256 enters the amendment and the ledger. Nothing is
fetched from the network and nothing is written under the spot study's
checkout. A day without a file at the source is excluded, `no_prints_file`.
E's days are absent at the source and are never copied. P's days are not on
disk; if P opens, they are fetched by the spot study's fetch code and terms
into this checkout.

**Loading.** `spot/loader.load_prints_day`, unchanged, inside this study's
access: the rows of the UTC day, the day's own tick, the exclusions
`no_prints_file`, `no_print` and `off_grid_price`, and `fee not sourced` from
the fee table. A block with more than a quarter of its days excluded is
unreadable.

**Sizes**, from the file listing (item 4), in MB of parquet:

| Month | Days | MB | MB a day |
|---|---:|---:|---:|
| 2026-03 (reader-test days) | 3 | 8.04 | 2.68 |
| 2026-04 (W) | 30 | 35.09 | 1.17 |
| 2026-05 | 31 | 23.04 | 0.74 |
| 2026-06 | 30 | 34.11 | 1.14 |
| 2026-07 | 31 | 19.16 | 0.62 |
| 2026-08 (H from 08-08) | 24 | 18.38 | 0.77 |
| 2026-09 | 30 | 20.73 | 0.69 |
| 2026-10 (to 10-21) | 21 | 12.22 | 0.58 |

---

## Fees

The spot registration's rule and its Binance rows, copied unchanged with their
sources into this study's configuration (`fees`): every fill pays the fee of
the sourced interval in force at its instant, in bp of its notional, booked
in quote at the fill price; a day not covered at the verdict tier is excluded,
`fee not sourced`. A resting order's fill pays the maker fee, a flatten the
taker fee.

| Interval (UTC) | Regular, maker / taker bp | Other tiers (measurements) | Sources |
|---|---|---|---|
| 2024-04-25 06:00 to 2026-01-29 00:00 | 0 / 10.0 | VIP 1 0 / 10.0; VIP 2–9 and Spot Liquidity Providers 0 / 0 | B3, B4, B6, B8 |
| 2026-01-29 00:00 to 2026-10-22 00:00 | 0 / 10.0 | VIP 3 0 / 6.0; VIP 9 0 / 2.3 | B4, B5, B6, B7, B8 |

**The maker fee is 0 for every account on every day of every block** to
2026-10-21. The verdict tier is **Regular**, the one an account holds without
volume or holdings; no discount is applied. P's days are not covered: the fee
amendment covers them by the spot study's P procedure before P opens. The
sources are the spot registration's B3 to B8, listed in the configuration.

---

## Blocks

The spot study's calendar, unchanged:

| Block | Dates (UTC, inclusive) | Days | Use | Read |
|---|---|---:|---|---|
| Reader-test days | 2026-03-03, 04 and 05 | 3 | the readers' proof, stage 0's timing and the fill rule's checks; numbers discarded | freely |
| **W** warm-up | 2026-04-01 to 04-30 | 30 | the signal's warm-up only (04-30 as D's first day's history); never scored | freely |
| **D** development | 2026-05-01 to 07-31 | 92 | σ_D and the minimum detectable effects; development summaries; the rehearsal on July. **In-sample for the signal**, whose fit was trained on D | freely |
| **E** embargo | 2026-08-01 to 08-07 | 7 | a gap | never |
| **H** held out | 2026-08-08 to 10-21 | 75 | the three slots' statuses and every measurement | once, after the D amendment is committed, through the ledger |
| **P** prospective | 2026-11-01 to 12-31 | 61 | the confirmation of a candidate on H | once, only for a candidate, by [its deadline](#status-rules) |

Days are independent: each starts flat with no resting order; anything open at
23:59 is cancelled or flattened. The signal's features read the previous
calendar day's prints when it is usable and in the same block, or W before D;
any other day starts cold and its first 61 minutes are not scored (H's first
day and P's start cold), as in the spot registration, §Blocks. The touch
proxies carry across midnight on a warm day; on a cold day a side's proxy is
undefined until that side prints.

---

## Common settings

- **Instrument**: Binance spot BTCFDUSD, `binance-spot:BTCFDUSD`, tick 0.01
  and lot 1e-05 (the spot study's D amendment).
- **Clip**: 0.010 BTC, S25's. Every order is one clip, or two for MK-SQ's
  signal side.
- **Latency**: every post, requote and cancel takes effect 10 ms after its
  decision.
- **The day**: no entry, quote or requote is decided before 00:01 or after
  23:58 UTC; an exit may be decided and requoted until 23:59. At **23:59**
  every resting order is cancelled and any position is closed by a taker
  order ([fills](#fills)).
- **Touch proxies**, `spot/grid.py`'s: the ask proxy is the price of the last
  buyer-initiated print stamped at or before the instant, the bid proxy the
  last seller-initiated one's, the mid proxy their mean; every print counts.
- **Post-only**: a buy is posted at min(target, ask proxy − tick) and a sell
  at max(target, bid proxy + tick) when the other proxy is defined, so no
  order is posted to cross; a buy's price is rounded down to the tick and a
  sell's up.
- **Skew**: a position q (BTC, positive long) moves both quote prices to
  price × (1 − κ × 1e-4 × q / clip) before the post-only rule, κ in bp per
  clip: a long lowers both quotes.
- **Decisions** are made at print-grid row stamps (the signal) and at clock
  instants (the requotes); every quantity in time is measured on stamps.

### The short side and the account

As the spot registration, §The short side: shorts come from held inventory.
Each day starts holding one **unit** of BTC and quote worth three units at
the day's first print, the unit being the slot's position limit (one clip for
MK-S and MK-Q, two for MK-SQ). `spot/account.py` replays every run's fills:
K-acct (both bookings), K-inv (either coin below zero) and the position
beyond the unit (here **K-pos**; K-cap names [capacity](#kill-conditions-per-slot)
in this study). The held unit's own P&L, marked from the day's first print to
its last, is a separate line, never in a verdict.

---

## The signal

**S25-PF, frozen.** The spot study's files, in its checkout (ignored by git):
`models/spot_2026/slots/S25-PF.json` (sha256
`d4048fa0902a5490c0b2238b52d7f0f4ed9c7bb558dc60345068598c8697a5d8`) and
`models/spot_2026/weights/S25-PF__hold_s=50__label_fee_bp=0.pkl` (sha256
`4cf14dd130d361f9f8159da9ca272b5d99d6d30dfee7eacd9e0cb5c6d13ec9ba`), both bound
by the spot study's frozen configuration (sha256 `d1625173…`). They are copied
to `models/spot_maker_2026/signal/` (ignored by git), each sha256 checked at the
copy and at every gate. The cell is S25's: hold 50 s, 240 trades a day, label
fee 0, θ_long 0.46583154797554016 and θ_short 0.4700717031955719.

- **Inputs**: the nine print members on the print grid,
  `spot/features_prints.py`, as the spot study computes them.
- **Probabilities**: `spot/days.predict`, on every row at or after the day's
  first scored stamp with every input present, through the store's float16
  cast; any other row holds.
- **Decisions**: the hold-first rule with the frozen θ on every scored row:
  long, short or hold. A decision's instant is its row's stamp.
- **Nothing is refitted**, and no other fit of the spot study is used.

---

## Fills

**Resting orders on prints, new code.** Binance publishes no spot book, so no
queue is seen. An order of side s (+1 buy, −1 sell), limit price p and size q
is live from its decision + 10 ms until its cancel's decision + 10 ms
(excluded).

- **Trade-through, the primary rule.** The order fills on reachable prints of
  the other aggressor side (a buy on seller-initiated prints, a sell on
  buyer-initiated ones) stamped while it is live at a price **strictly
  beyond** p (below p for a buy, above for a sell), in archive order, each
  print filling min(the remaining size, the print's size) at p. The queue at
  p is assumed exhausted before the order fills: the pessimistic rule.
- **Touch, the optimistic bracket**, reported beside every verdict: the same
  with prints at p or beyond.
- **Delayed, K-fill's run**: the trade-through rule with the first qualifying
  print after the order goes live filling nothing, so every fill comes one
  further print later.
- A fill's time, and its fee's instant, is its print's stamp; it pays the
  table's maker fee (0 at Regular). An order's unfilled part at its cancel
  never fills. A remainder of at most 1e-12 BTC is none.
- **The flatten.** At 23:59 every resting order is cancelled and a position
  is closed by one taker order of its size, priced by the spot study's
  side-correct print rule, `spot/fills.fill_orders`, unchanged (its clip the
  position, arrival 23:59 + 10 ms, the day's end as its bound, and its
  fallback when no print of its side follows), paying the table's taker fee at
  its instant: 10 bp at Regular. Flattens are counted, with their notional.

The rules order three fills of the same order; none is a bound on a real
fill. A real order at the touch can fill on a print at its own price before
the market trades through it (better than trade-through), or never, behind a
queue (worse than touch).

---

## The slots

| Coordinate | **MK-S** | **MK-Q** (control) | **MK-SQ** |
|---|---|---|---|
| Signal | S25-PF, frozen | none | S25-PF, frozen |
| Orders | one clip at its own proxy on the signal's side; the exit at the other side's proxy | a buy at the bid proxy and a sell at the ask proxy, one clip each | as MK-Q; the signal side's order two clips while the signal is live |
| Requote | the exit, every 60 s after its decision | every 60 s, the multiples of 60 s from midnight, 00:01 to 23:58 | every 60 s as MK-Q, and at each signal decision and its expiry (decision + 50 s) |
| Skew κ, bp per clip | 0 (the exit only) | 1 | 1 |
| Position limit (the unit) | 1 clip | 1 clip | 2 clips |
| Hold | 50 s (S25's) | — | 50 s, the signal's life |
| Neighbourhood (K-nbhd), 9 cells | requote {30, 60, 120} s × κ {−1, 0, 1} | requote {30, 60, 120} s × κ {0, 1, 2} | as MK-Q |
| Verdict | `net_bp_fill` at Regular, 0 / 10 bp | the same | the same |

**MK-S, step by step.**

1. **Entry.** At a decision row t of side s, flat and with no resting order,
   inside the posting hours: one clip on side s at its own proxy at t (a buy
   at the bid proxy, a sell at the ask proxy), post-only. Its unfilled part is
   cancelled at the earliest of t + 50 s, the first later decision of the
   other side, and 23:59. An entry whose own proxy is undefined is not posted
   and is counted.
2. **No fill.** The next decision may be taken at a row stamped at or after
   the cancel's decision; an opposite decision that cancelled the entry is
   itself a new entry.
3. **Exit.** After a fill, the exit is decided at the earliest of the first
   print-grid row stamped at or after the entry's first fill + 50 s, the first
   decision of the other side after t, and 23:59: a resting order of the other
   side for the position, at its own proxy (a long's sell at the ask proxy)
   shifted by the skew, post-only, sized at its arrival (any entry fill before
   the entry's cancel takes effect counts). It is requoted every 60 s after
   its decision at the proxy then. A remainder at 23:59 is flattened as a
   taker and counted.
4. **One position.** After an exit, the next decision may be taken at a row
   stamped at or after the position's last fill. Decisions arriving while an
   entry rests or a position is open are skipped and counted, except an
   opposite decision, which cancels a resting entry or starts the exit.

**MK-Q.** At every requote instant both orders are cancelled and reposted at
the proxies then, shifted by the skew, post-only, each one clip, the buy cut
to limit − position and the sell to limit + position at their arrival. A side
whose own proxy is undefined is not quoted. Orders rest until the next
instant's cancel or 23:59; there is no replenishment between instants.

**MK-SQ.** As MK-Q, with requote instants also at every S25-PF decision row
inside the posting hours and at each such decision + 50 s inside them. At an
instant the signal's side is that of the latest such decision stamped at or
before it and less than 50 s before it, or none; the signal side's order is
two clips, the other side's one clip, both cut by the limit of two clips.

What each asks:

- **MK-S**: does S25-PF's signal pay when both legs rest at the touch and pay
  no fee?
- **MK-Q**: what does quoting alone earn on this pair, at the same fill rule?
  It is the control: a positive MK-S beside a positive MK-Q is quoting, not
  the signal.
- **MK-SQ**: does leaning the quotes on the signal add to quoting?

---

## Metrics

- **`net_bp_fill`**, primary for every slot: the day's net in quote over every
  fill, maker and taker, fees included (the day ends flat), over the day's
  turnover, the sum of its fill notionals, × 1e4. For MK-S a round trip is two
  fills, so its `net_bp_fill` is about half its net per round trip. The daily
  series has one value per day with at least one fill.
- **`net_bp_clip_day`**, the net per day: the day's net in quote over the
  notional of one clip at the day's first print, × 1e4; a day without a fill
  counts 0.
- **`gross_bp_fill`**, the same before fees, labelled "conditional on a
  zero-fee venue": here it differs from the net only by the flattens' fees.
- **The making part**, the quantity K-dir reads, as the 2025 registration's
  S-M defines it: the mean, over every maker fill of the block, of its 5 s
  markout, side × (mid proxy at the fill's stamp + 5 s − the fill price) / the
  fill price × 1e4, in bp per fill. A fill whose mid proxy is undefined then
  has no markout. Markouts at 1, 5 and 30 s are reported; the **spread part**
  is side × (the mid proxy just before the fill's print − the fill price) /
  the fill price × 1e4, and the **adverse part** the 5 s markout less it.
- **The daily series decides.** The mean, the day-level t, the Newey–West t
  (Bartlett, lag 5), the median day, K1, K-days and every run's block mean are
  computed on daily series, one value per day, through
  `market_making/verdicts.day_t` and `holm` and
  `evaluation/significance.newey_west_t`; a series of fewer than two days
  enters Holm with p = 1.
- **A run's block mean** (placebos, drift runs, cells, the 110 ms and delayed
  runs) is the mean of its daily `net_bp_fill` over the block's days with a
  fill of that run; a run without a fill on the block scores 0.
- **Minimum detectable effect**, recorded in the amendment per slot:
  `(3 + 0.84) × σ_D / √n_eff`, σ_D the standard deviation (ddof 1) of the
  slot's daily series over D's days with a fill and `n_eff` = 75 × the share
  of D's usable days with a fill; on P, × √(75 / P's usable days). D is
  in-sample for the signal, so σ_D is a description of dispersion only.

---

## The development-period procedure (D only)

0. **Stage 0, on the reader-test days.** The three days are copied and
   loaded through the loader; the print contract and the tick are checked;
   the signal is computed and every slot run once per day; the fill rule is
   checked on every order of those runs (trade-through fills within touch
   fills, delayed fills within trade-through fills, every fill inside its
   order's life and size, the account's identity on every run); the copy,
   features, signal and runs are timed. Every number but the checks and the
   timings is discarded and never reported. A failure is fixed by a dated
   amendment before stage 1.
1. **The code freeze**: a dated note, `docs/spot_maker_2026_code_freeze.md`,
   names the commit on a line ``Code-freeze commit: `<sha>` ``. Stage 1, D
   and the amendment start only if the study code (`src/`,
   `experiments/spot_maker_2026.py`, the configuration outside its
   placeholders, `pyproject.toml`, `uv.lock`) is unchanged since it.
2. **Stage 1**: W and D copied; the signal computed on D's days.
3. **D**: every run of every slot on D's 92 days; σ_D and the MDEs; each
   slot's development summary; **the rehearsal**, H's status rules applied to
   July with the values about to be frozen, labelled rehearsal. Nothing is
   chosen.
4. **The amendment** ([amendment protocol](#amendment-protocol)).

### Nothing is searched

No value of this registration is chosen on an outcome. The signal is frozen
by the spot study; the requote interval, the skew, the limits, the hold, the
clip and the fill rules are fixed here; every slot goes to H whatever D shows.
There is no catalogue, no choice and no placebo search, and D's numbers,
in-sample for the signal, describe and never decide. **The run log**: every
computation on a D day appends a row (its date, the run, the stage and the
days it read) to `data/spot_maker_2026/run_log.json`, never committed; its
sha256 and row count are frozen in `compute.record.trials` and checked at H's
and P's gates.

---

## Status rules

**The family** is **{MK-S, MK-Q, MK-SQ}, with m = 3** on H and on P.

**The bar.** A slot passes on a block when no kill condition fires and all of
these hold on its daily series: its one-sided day-level p-value passes Holm's
procedure at 5% with m = 3; the day-level t is at least 3; the Newey–West t
is at least 2; the median day is positive; and the series has at least 20
days with a fill on H, 10 on P. A slot enters Holm with p = 1 when it is
void, killed, inconclusive with too few fills, not assessed, or not taken to
the block.

**Precedence**, one status per result:

1. **void** — K-units, K-acct, K-inv or K-pos fires in any run of the read;
   counted as killed.
2. **inconclusive (too few fills)** — K-trades: fewer than 100 fills, maker
   and taker, over the block. Nothing below is evaluated.
3. **killed** — any kill condition fires.
4. **candidate** (on H) or **passes** (on P) — the bar is met.
5. **inconclusive** — otherwise; "inconclusive, below the minimum detectable
   effect" when the mean is smaller in size than the MDE.

**P** is read once, in one session, only for H's candidates, after H's results
are committed, 2026-12-31 is published on Binance's archive and the fee
amendment covers P's days. **Its deadline** is the later of 2027-01-31 and 30
days after the committer date (UTC) of the commit that first adds
`experiments/results/spot_maker_2026_H.json`; no P read opens after it.

| On H | On P | Status |
|---|---|---|
| candidate | passes | **confirmed** |
| candidate | does not pass, no kill | **candidate, not confirmed** |
| candidate | a kill fires | **killed on P** |
| candidate | a void condition fires | **void** (counted as killed) |
| candidate | P unreadable | **candidate, not confirmed (P unreadable)** |
| candidate | P's read failed | **candidate, not confirmed (P read failed)** |
| candidate | not read by the deadline | **candidate, not confirmed (P not read by its deadline)** |
| any other status | not read | that status |

**Unreadable and failed reads**, as the spot registration: a block with more
than a quarter of its days excluded is unreadable, every slot `not assessed (H
unreadable on Binance)`; a read failing twice at the same stage with the same
exception, or not finished after five resumptions, closes, every slot `not
assessed (H read failed)`.

**No number** at another tier, at zero fees, under the touch bracket, in a
cell, under the 110 ms or delayed rule, or in the held unit's P&L may be
described as a slot working: they are measurements.

### Kill conditions per slot

| Slot | Primary series | Kills, in the order the first kill is read |
|---|---|---|
| MK-S | `net_bp_fill` at Regular | K1, K-dir, K-placebo, K-drift, K-nbhd, K-days, K-latency, K-fill, K-cap |
| MK-Q | `net_bp_fill` at Regular | K1, K-dir, K-drift, K-nbhd, K-days, K-latency, K-fill, K-cap (no signal, no K-placebo) |
| MK-SQ | `net_bp_fill` at Regular | K1, K-dir, K-placebo, K-drift, K-nbhd, K-days, K-latency, K-fill, K-cap |

Void conditions; the slot is recorded as `void`:

- **K-units** — any fill's fee, in bp of its notional, differs from the
  table's for its kind (maker or taker), tier and instant, at a relative
  tolerance of 1e-12, in any run.
- **K-acct** — the day's net differs from the spot account's change (both
  bookings), or turnover from the sum of fill notionals, in any run.
- **K-inv** — the account would hold less than zero of either coin at any
  fill, in any run.
- **K-pos** — the position exceeds the slot's limit either way at any fill,
  in any run.

Kill conditions:

- **K1** — the mean of the slot's daily `net_bp_fill` on the block is ≤ 0 at
  the registered fees.
- **K-dir** — the making part, the mean 5 s markout over the block's maker
  fills, is ≤ 0: a maker's claim is what it earns at the fill; a positive net
  without it is inventory carried through a move.
- **K-placebo** (MK-S, MK-SQ) — the true mean does not exceed both the 95th
  percentile of the 200 random-direction runs' means and the sign-flipped
  run's mean.
- **K-drift** — the true mean does not exceed the larger of the all-long and
  the all-short runs' means: for MK-S and MK-SQ every signal direction
  replaced by long (by short); for MK-Q, quoting the buy side only (the sell
  side only).
- **K-nbhd** — the median, over the slot's nine cells (its own included), of
  each cell's block mean is ≤ 0; a cell without a fill scores 0.
- **K-days** — half or fewer of the series' days are positive.
- **K-latency** — a positive result whose run with every post, requote and
  cancel taking effect 110 ms after its decision has a block mean ≤ 0.
- **K-fill** — a positive result whose run under the delayed rule has a
  block mean ≤ 0.
- **K-cap**, capacity — the slot's maker fills exceed 10% of the venue's
  printed volume at the fills' prices: the sum of the block's maker fill sizes
  over the sum, over each day and each distinct price the slot filled at that
  day, of the volume of every print at that price that day, in BTC.

Not kills, but they set the status: **K-trades** (fewer than 100 fills) and
**K-MDE** (a result that does not pass, with no kill, smaller in size than
the MDE). K-latency and K-fill fire only on a positive result.

### Placebos

Each runs through the same executor, fee, account and day rules as the true
run, at the true run's decision instants, re-applying its own position and
order rules (a maker's fills depend on the side, so the busy periods change).

| Placebo | What changes | Slots | Seeds |
|---|---|---|---|
| sign-flipped | every signal direction reversed | MK-S, MK-SQ | — |
| random direction | each direction drawn +1 or −1 | MK-S, MK-SQ | 200 draws, seeds 0–199, `numpy.random.default_rng([seed, day ordinal])` |
| all-long, all-short | K-drift's two runs | all three | — |

---

## Measurements without a hypothesis

Beside every slot, with no status: `net_bp_clip_day`; `gross_bp_fill`; the
making part, the spread part and the adverse part; markouts at 1, 5 and 30
s; fills a day, maker and taker; the fill ratio (filled size over posted
size); the touch bracket; the delayed and the 110 ms runs; the nine cells;
the drift runs; `net_bp_fill` at VIP 3 (0 / 6.0) and VIP 9 (0 / 2.3),
conditional on that tier's eligibility (only the flattens' fees change), and
at zero fees; the flattens, counted, with their notional; the held unit's own
P&L; the maximum drawdown of the cumulative daily net; the share of days
positive and the median day; the capacity ratio; the touch proxies' width
(ask proxy − bid proxy, in bp of the mid proxy) at the requote instants, its
median per day; and the venue's prints a day. For MK-S also: entries posted,
entries filled in any part, entries cancelled unfilled, decisions skipped
(busy, outside the hours, no proxy), exits flattened and the realised hold
from the entry's first fill to the exit's last.

---

## Data budget

Nothing is fetched. The copies, from the file listing: reader-test days 8.0
MB, W 35.1 MB, D 76.3 MB, H 51.3 MB, about **171 MB** in all; features,
records and models under 1 GB. Each stage starts only if free disk exceeds
its 1 GB budget by 10 GB; each measurement is recorded. On 2026-10-27, 31 GB
were free. **Compute**: features about 1 s a day; MK-S and MK-Q under 3 s a
day for every run; MK-SQ about 30 s a day with its 200 random draws; D and H
each under an hour on at most four worker processes, days in parallel.
Stage 0's timings replace these estimates in the amendment.

---

## Who may read what, and the code that will enforce it

### The code to come

All of it is new files, in steps of at most about 120 lines with a test each,
under `src/trading_research/spot_maker/`, with
`experiments/spot_maker_2026.py` as the single entry point and
`tests/test_spot_maker_*.py`: the access and blocks; the registration loader;
the copies (the day files and the signal's files, each sha256 checked); the
signal (features, probabilities and decisions by the spot study's code); the
touch proxies, post-only and skew; the resting-order fill rule and its
variants; the day's money, markouts, capacity and account checks; the MK-S
executor; the quoter (MK-Q and MK-SQ); one day's every run; the statuses;
the ledger; the stages and the reads; the page; and a dry run on the spot
study's synthetic venue. **Reused unchanged, by import**: `spot/prints`,
`spot/loader`, `spot/grid`, `spot/features_prints`, `spot/days` (`predict`,
`layout`), `spot/runs` (`decisions`, `random_sides`), `spot/fills`
(`PrintTape`, `fill_orders`), `spot/account`, `spot/costs`, `spot/state`,
`spot/synthetic`, `spot/binance`, `market_making/verdicts`,
`market_making/prereg` (`placeholder_lines`, `unfrozen_text`, `render_value`,
`sha256_bytes`), `evaluation/significance` and `strategies/hold_first`. No
existing module, `__init__.py`, `cli.py`, `pyproject.toml`, `uv.lock` or any
other study's configuration, registration or test is edited; nothing is
registered in a global feature registry.

### Access

**The development access** allows the reader-test days, W and D, and refuses
every other day; the reader-test days alone are also granted for one of the
purposes `readers' proof`, `stage 0 timing` and `dry run`, stamped `reader
test (<purpose>)`. E, every day of 2025 and the 2025 study's reader-test days
are never granted.

**The ledger**, `experiments/results/spot_maker_2026_ledger.json`, on the spot
study's pattern, is the one gate to H and P:

- **H opens only if** the amendment records the configuration's sha256 and the
  file on disk has it and is committed; the study code is the code the
  amendment froze (the code-freeze commit `compute.record` names, HEAD
  descending from it, nothing of `src/`, `experiments/spot_maker_2026.py`,
  `pyproject.toml` or `uv.lock` changed since); the configuration changed
  since that commit only in its placeholders and the document only by
  appending one ``Frozen configuration sha256:`` line and no
  ``Null-placeholder sha256:`` line; the tree is clean but for the ledger; the
  signal's copies are at the registered sha256s; the run log is at its frozen
  sha256 and row count; and no file of H is on disk (no path under
  `data/spot/` names a day of H or a month holding one).
- **The open entry** (block, the session's commit, the configuration's
  sha256, the environment fingerprint and the UTC time) is committed before
  the block is copied. Every later process refuses unless it is in HEAD's
  ledger, the tree and `git diff --name-only <session's commit>..HEAD` list
  nothing but the ledger, and the fingerprint equals the open entry's.
- **The read session**: its first process copies the block and records each
  copy in the ledger; they are committed; its second process computes, writes
  `experiments/results/spot_maker_2026_<block>.json` and the page, and closes
  the session with the results file's sha256.
- **P opens only** for a candidate on H, by its deadline, once H's results
  file is committed unchanged and, since H's read, only `README.md`, paths
  under `experiments/results/` and paths under `docs/` outside
  `docs/preregistration/` changed, but for the fee amendment's intervals.
- A second read raises unless forced; what it produces is stamped `second
  read` and changes no status. **H and P are spent** for these slots.

---

## Amendment protocol

**The D amendment**, after D and before H is read, is one dated entry under
[Amendments](#amendments), written by the entry point and reviewed before it
is committed. It records: the reader-test days' results; the days excluded on
W and D, with reasons; σ_D and the MDE of each slot; each slot's development
summary, labelled "development"; the compute record (stage 0's timings and
checks, free disk at each stage, the copy log's sha256 and counts, the
code-freeze commit stage 1 ran under, the run log's sha256 and row count);
its date; and the configuration's sha256 after these values are written into
its placeholders, on a line ``Frozen configuration sha256: `<hash>` ``. Each
value has one placeholder in the configuration, commented "frozen by the D
amendment". Nothing else may change.

**A correction before H** (a bug in the copies, the loader's use, the signal's
use, the fill rule, an executor or the accounting) is made only by a further
dated amendment that states it and, if it could change a D number, re-runs D
in full before anything is frozen. Every change to the study code after the
code-freeze note is such a correction. No amendment may cite a D number or
anything seen on spot data as its reason, except a fault it names. An
amendment that changes the configuration outside its placeholders records a
new null-placeholder hash on a line ``Null-placeholder sha256: `<hash>` ``
before the code freeze; the latest such line is accepted, a superseded hash
never is.

**The fee amendment** is the only amendment after H: P's intervals, one line
each under `fees.intervals`, from 2026-10-22, by P's deadline and before P
opens, transcribed by the spot study's P procedure.

**After H is read**, nothing is changed and nothing is re-run under another
rule. A defect found after a read only lowers a status: it is stated in
`docs/spot_maker_2026_defects.md`, and every status it could affect becomes
`void (defect found after the <block> read)`. A corrected run is a successor
registration, on days after 2026-12-31. Only reads through the ledger count.

---

## Artefacts and headline sentences

**Files**, written by the entry point:
`experiments/results/spot_maker_2026_ledger.json`;
`experiments/results/spot_maker_2026_H.json` (and `…_P.json`), each block's
statuses, kills and measurements; `experiments/results/spot_maker_2026_trials.json`,
the run log's manifest; and a page, `experiments/results/spot_maker_2026_page.md`,
with one row per slot (status, first kill against its reference, mean, t,
Newey–West t, median day, days, fills a day, net per day, the making part,
markouts, MDE, placebo percentile, K-nbhd median, the touch and delayed runs,
capacity).

**Headline sentences, fixed now.** The bracketed parts are filled by the entry
point and never edited.

| Outcome | Sentence |
|---|---|
| Each slot on H | "[slot] on H, Binance spot BTC/FDUSD at the fees charged: [status]; first kill [K-x] at [value] against [reference]." |
| MK-S | "S25-PF's signal as a maker on Binance BTC/FDUSD, August to October 2026: [net] bp per fill at a 0 bp maker fee, [fills] fills a day, making part [making] bp per fill; [status]." |
| MK-Q | "Two-sided quoting at the touch proxies on Binance BTC/FDUSD, August to October 2026: [net] bp per fill, making part [making] bp per fill; [status]." |
| MK-SQ | "Quoting with S25-PF's side doubled on Binance BTC/FDUSD, August to October 2026: [net] bp per fill, making part [making] bp per fill; [status]." |
| Fill rule | "Under the touch rule MK-S's net was [touch] bp per fill against [net] under trade-through." |
| P | "[slot] on P: [status]." |

Values have two decimals and their unit: K-days' `(share of days positive)`,
K-cap's `(share of printed volume)`, every other `bp`; [fills] has one
decimal. A slot with no kill reads `none` at `n/a` against `n/a`. The first
kill is the first that fires in the order of the slot's row of the kill
table. D numbers are never quoted as results.

---

## What this study cannot establish

- **A real fill on BTC/FDUSD.** There is no book: no queue position, no
  hidden or iceberg liquidity, no cancellation by others and no own impact is
  seen. Trade-through and touch are two registered rules, not bounds. A real
  order can fill earlier than trade-through, at its own price, or later than
  touch, behind a queue.
- **That the signal is blind to H.** S25-PF's gross edge at taker fills on
  H's days is known to be positive (item 3); a positive MK-S on H says resting
  orders kept it, not that the signal holds out of sample. Only P confirms.
- **The touch.** The proxies are last prints by side, which can cross and
  lag; post-only is applied against them, not against a book, so an order
  posted here could have been rejected or could have crossed.
- **The venue's mechanics**: post-only rejections, rate limits, self-trade
  prevention, latency beyond the registered 10 and 110 ms.
- **Capacity beyond K-cap.** The clip is 0.010 BTC; own impact is not
  modelled.
- **The fee a given account paid.** Regular is the verdict tier; a maker fee
  of 0 for every account is sourced to 2026-10-21 only.
- **Out-of-sample numbers on D.** D is in-sample for the signal.
- **Shorting through margin** (shorts come from held inventory), **quote
  currency risk** (results are in FDUSD, never converted), and anything beyond
  one pair, one venue and the 2026 blocks.
- **When the ledger's entries were made, to anyone but the owner.** Nothing is
  pushed; the order of the reads rests on the local history alone.

---

## Choices and their reasons

One line each; every value is in the configuration.

- **Copies, not fetches**: the files are on disk and the owner asked that
  nothing on disk be fetched again; a copy keeps this study's data root apart
  from the spot study's and lets the no-H-on-disk gate hold.
- **The signal frozen at S25's cell** (hold 50 s, 240 a day, label fee 0): the
  owner's specification; nothing is refitted on H or anywhere.
- **Trade-through primary, touch beside it, partial fills by print size**: the
  owner's specification; the pessimistic rule decides.
- **"Delayed by one further print" read as the first qualifying print filling
  nothing**: each fill moves to the next print that would have filled it.
- **MK-S's exit requoted every 60 s**: the specification leaves the exit's
  life open; an exit fixed in price waits for a reversal, and one clock for all
  three slots makes K-nbhd's requote axis meaningful for MK-S.
- **MK-S's skew 0 and its axis {−1, 0, 1} bp per clip**: "exit at the ask
  proxy" is a skew of 0; the neighbourhood steps one either way.
- **The quoters' skew 1 bp per clip, axis {0, 1, 2}**: a skew without a value
  was specified; 1 bp per clip moves a full position's quotes by 1 bp, set
  without knowledge of the pair's touch width; a skew in ticks would be about
  0.001 bp.
- **Limits of 1, 1 and 2 clips**: one position for MK-S; MK-Q's limit equals
  its quote; MK-SQ's doubled side needs room for two clips.
- **The account's unit is the limit**: K-inv then fires only through a fault,
  as in the spot study.
- **Requote instants on the clock from midnight**, and for MK-SQ also at each
  decision and its expiry: the doubled size lives exactly the signal's hold.
- **Orders sized at their arrival**: no fill during a latency can carry the
  position past its limit.
- **Post-only against the proxies, rounded away from the touch**: no order is
  posted to cross, and a skewed price lands on the tick.
- **`net_bp_fill` as every slot's verdict**: the quoters have no round trips,
  and the owner asked for net bp per fill.
- **K-cap is capacity and a kill; the position limit is K-pos and void**: the
  owner lists capacity among the kills; a position beyond the limit is a
  fault, as the spot study's K-cap was.
- **Capacity pooled over each day's fill prices**: the owner's "printed volume
  at that price", the day as the window.
- **K-dir read after K1**: it fires on any non-positive making part, so a
  candidate needs a positive one, as specified.
- **K-placebo on MK-S and MK-SQ, not MK-Q**: MK-Q has no signal to replace.
- **K-drift for MK-Q as one-sided quoting**: the no-signal analogue of all-long
  and all-short.
- **K-trades counts fills**: the owner's "fewer than 100 fills".
- **σ_D over all of D, no test months**: nothing is fitted here; D is
  in-sample for the signal, so its dispersion is all it can describe.
- **No null on D**: nothing is chosen, so no search is described.
- **The reader-test days for the readers' proof and timing only**: the owner's
  decision (the information set).
- **E never copied; P fetched only if it opens**: as the spot study.

---

The configuration as registered, with every placeholder `null`, has sha256
`8af7842a9b56d8108956548197985ee910993a7e87fc7253bf8de130e862ea6d`,
computed as the spot study computes its own: the sha256 of
`market_making.prereg.unfrozen_text` applied to the file, which puts `null`
back into every line commented "frozen by the D amendment". This study's
loader (`spot_maker/registration.py`, `REGISTERED_SHA256`) checks that the
file, with its placeholders reset, gives this hash, or the one the latest
``Null-placeholder sha256:`` line under [Amendments](#amendments) records,
and refuses every superseded one.

---

## Amendments

Dated entries, appended in order; nothing above this heading changes after the code freeze.

### 2026-10-27: the code's choices where the text left one open

Committed before the code freeze and before this study copied, opened or
computed on any print. The configuration is unchanged: with its placeholders
reset it still has the registered sha256 `8af7842a…`, so no new
null-placeholder hash is recorded. No reason below is drawn from data.

- **This section's first line** replaces "None yet.": entries are appended in
  order, and nothing above the heading changes after the code freeze.
- **The skew reads the position known at the decision**: a quote's price is
  skewed by the position of the fills stamped at or before the instant it is
  decided; a fill stamped inside the 10 ms latency after that instant is not
  yet known there. Its size is cut at its arrival, when every earlier fill is
  known ("Orders sized at their arrival"). For MK-S's exit the same holds: an
  entry fill inside the cancel's latency counts in the exit's size, not in its
  first price.
- **MK-S's exit at a requote whose own proxy is undefined** keeps its last
  price; an exit that never had a price is not posted, and its position is
  flattened at 23:59.
- **Stage 0's fill-rule checks** run on every resting order of every slot's
  run under each of the three rules: each order is filled again from its side,
  price, size and live window, and must have its fills inside its life and
  size, the fills its run recorded exactly those its own rule gives, and its
  delayed fills within its trade-through fills within its touch fills. They
  also run on probe orders, one a side at its proxy at every 60 s instant. The
  account's identity is every run's void conditions. Stage 0 also checks the
  print contract (`spot/prints.check`), both aggressor sides present and the
  tick equal to 0.01, and times the copy, the load, the features, the signal,
  the runs and the checks apart.
- **Stage 1 computes the signal on D's usable days**, its decisions counted
  under the work directory; nothing of it is reported.
- **Free disk is checked and recorded** at stages 0, 1 and D and at each
  process of a read, against the 1 GB budget and its 10 GB margin.
- **Copies, not links**: the copies' 171 MB lie far inside the budget, so every
  day file is copied as registered and none is hard-linked.
- **Measurements beside the verdict**, each named under "Measurements without
  a hypothesis" or computed from them, never a status: the block's net over its
  turnover, pooled; for MK-S the net per round trip, the day's net over its
  entries' notional (a round trip is an entry and its exit of the same size;
  the quoters have none); fills a day, maker and taker; the days positive,
  counted; the fees paid; the flattens and their notional; MK-S's counters and
  its realised hold (the mean, and the median of the daily medians); the
  proxies' width as the median of the daily medians.
- **P's not-assessed statuses in the P table**: "not assessed (P unreadable on
  Binance)" reads "candidate, not confirmed (P unreadable)", "not assessed (P
  read failed)" reads "candidate, not confirmed (P read failed)".

### 2026-10-27: the fill rule's calibration against a book

Committed before the code freeze and before this study copied, opened or
computed on any print of BTC/FDUSD; no result of the calibration had been
computed. Its reason is a review of the design, not data: every verdict here
rests on a prints-only resting-order rule (the touch taken from the last
print of each side, a fill only on a print through the order's price), which
is uncalibrated unless the same quoters are also run on days where a book
exists, under a book's queue model and its trade-through and touch rules,
and the prints-only rule is seen to rank with them. A positive result at a
zero maker fee could otherwise be an artefact of the proxy.

- **Added as a registered measurement**, run in stage D and written into a new
  placeholder, `calibration.result`, frozen by the D amendment; the
  configuration's new section `calibration` holds every value below.
- **The days**: Bybit linear BTCUSDT, 2024-02-01 to 25, the earlier
  market-making rounds' development block D, whose book (ten levels every
  100 ms), prints and funding were checked on 2026-10-27 to be on disk in the
  main checkout's data, all 25 days; read in place, never copied, moved or
  fetched. They belong to no block of this study and are read by the
  calibration alone. Nothing is added to the disk budget.
- **The runs**, for MK-Q and MK-SQ, each day: (a) with the book, in the
  repository's market-making simulator, under its trade-through, queue and
  touch rules, the slot's quoting rule as a quoter (its requote instants,
  skew and limit; the book's best bid and ask as the touch; post-only against
  the other side; no replenishment between instants; an order the simulator
  keeps when its price is unchanged keeps its place); (b) with this study's
  prints-only quoter on the same day's prints thinned to the median count of
  BTC/FDUSD's aggregated prints a day over D's usable days, seeds 0 to 9, the
  proxies cold. MK-SQ's decisions are S25-PF's, frozen, on the day's seed-0
  thinned prints, cold. MK-S is not run apart: its orders rest at the same
  proxies under the same rule, and the label below covers its verdict too.
- **Recorded per day**: per slot and rule, the maker fills, their volume and
  the making part (the mean 5 s markout of the maker fills, bp per fill).
- **The registered test**: on at least 80% of the days the simulator does not
  exclude (with a maker fill under every rule), the maker volume filled
  orders trade-through <= queue <= touch (volume, since the book's touch rule
  fills an order whole at one print), and the prints rule's making part lies
  between the book's trade-through and touch making parts, inclusive; for
  both slots. If it fails, or cannot run, every verdict of this study carries
  the label "uncalibrated fill rule" on the page beside its number, and the
  calibration is recorded "not run (no book day on disk)" when no book day is
  found. Nothing else changes: no status, kill or threshold depends on it.
- **The run log** gets a row for it in stage D.
- The configuration changed outside its placeholders, so its null-placeholder
  hash is recomputed as registered (the sha256 of
  `market_making.prereg.unfrozen_text` of the file); the registered
  `8af7842a…` is superseded.

Null-placeholder sha256: `aa367b6434a7e8dc7553a2a3ce20eed2814d33f7844ff4d2e919bfe003be0db2`

### 2026-10-28: the calibration's days, venue and slots corrected, after the code freeze and before stage 1

A correction under this protocol: it changes the study code after the
code-freeze note of 2026-10-28 (commit `4b5d7d3`), so the note names a new
code-freeze commit. It is made before stage 1, before any day of W, D, H or P
is copied, and before any number of the calibration was computed (one day of
it was run for its timing and memory only). Its reason is the design, not
data: the calibration of 2026-10-27 ran on another instrument (Bybit's linear
perpetual, 2024) and left MK-S out; the closest book to this study's pair and
period is Bybit spot BTCUSDT on D's days, which the spot study holds.

- **The days and the venue**: Bybit spot BTCUSDT on D's days, 2026-05-01 to
  07-31, each with a usable BTC/FDUSD day; its book (ten levels) and prints in
  the spot study's checkout, read in place inside the spot study's own
  development access (D only), never copied or moved. Development days only:
  no day of E, H or P of either venue is read. Nothing is added to the disk
  budget.
- **The slots**: MK-S, MK-Q and MK-SQ. MK-S's rule on the book is a quoter of
  the simulator (its entry at its own side's best price, cancelled at the
  hold, the opposite decision or 23:59; its exit at the other side's best
  price from the first fill seen + hold, requoted every 60 s), the fill seen
  at the snapshot after it.
- **The book run**: the repository's market-making simulator and its queue
  model (the spot study has a taker's print rule, no resting order's queue),
  under its trade-through, queue and touch rules; its tape the day's reachable
  prints on the book's tick 0.1 (Bybit spot prints at a finer price are
  counted and left out of the book run only); no funding.
- **The prints run**: this study's executors on the day's prints thinned to
  BTC/FDUSD's count of aggregated prints that same day, seeds 0 to 9, the tick
  the book's.
- **Unchanged**: what is recorded per day, the registered test (80% of the
  days, for every slot, the maker volume ordered trade-through <= queue <=
  touch and the prints rule's making part inside the book's bracket), the
  label "uncalibrated fill rule" and "nothing else changes".
- **Stage 0** ran at `4b5d7d3` on 2026-10-28 and passed; this correction
  touches only the calibration (its module, the simulator's quoters, its
  call in stage D and its line on the page), none of which stage 0 runs, so
  stage 0 is not run again.
- The configuration changed outside its placeholders, so its null-placeholder
  hash is recomputed; `aa367b64…` is superseded.

Null-placeholder sha256: `147aea710dbef9c390b5efff27816ab72d93d39fa636ab44c2b22160b33298e7`

### 2026-10-27: the D amendment

Written by the entry point (`experiments/spot_maker_2026.py amend`) and reviewed before it is committed. Every frozen value is in the configuration's placeholders.

Frozen configuration sha256: `8dbd3181a474ae378ca6a9211bf17f487d963585de5c3056eb3198d33597a53b`

- Complete D run D1; the run log's manifest `experiments/results/spot_maker_2026_trials.json`: {'sha256': '2f8ef9d2d4d1225910d691bcf68d1841ccccf7b16dcc551175b21c609fdf9523', 'rows': 3}.
- The code-freeze commit stage 1, D and this amendment ran under: `bf044ea3c2eb76710b33fe6e9528cb4faac2eccc`.
- Days excluded on W and D: none.
- The reader-test days pass: True.
- Stage 0's timings a day: load 0.09672 s, features 0.9939 s, signal 0.04868 s, runs 21.84 s, checks 1.843 s; free disk 0 27.13 GB, 1 28.65 GB, D 28.55 GB.
- sigma_D and the MDE (development, bp per fill): MK-S 1.237 and 0.5484; MK-Q 0.3808 and 0.1689; MK-SQ 0.3586 and 0.159.
- D's means (development, in-sample for the signal; never a result): MK-S -2.288; MK-Q -2.839; MK-SQ -2.776.
- The fill rule's calibration on the book days: MK-S: fills ordered on 1 and the prints rule's making part inside the book's bracket on 0.2247 of 89 days; MK-Q: fills ordered on 1 and the prints rule's making part inside the book's bracket on 0.6413 of 92 days; MK-SQ: fills ordered on 1 and the prints rule's making part inside the book's bracket on 0.5761 of 92 days; it does not pass; every verdict labelled "uncalibrated fill rule".
- The rehearsal on July (H's rules with these values; discarded): MK-S killed; MK-Q killed; MK-SQ killed.
- Reviewed before it was committed, on 2026-10-28 (Moscow time). Dates here
  are UTC; the correction above is dated by Moscow time and was made on
  2026-10-27 at about 21:15 UTC, before stage 1.
