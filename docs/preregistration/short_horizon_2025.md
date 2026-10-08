# Pre-registration: the 2025 short-horizon strategy search, on public data

This document registers a strategy search before any of its data is read. It
is committed together with
[`configs/short_horizon_2025.yaml`](../../configs/short_horizon_2025.yaml),
which holds the same values in the form the code will read, and the two must
agree.

| | |
|---|---|
| **Committed** | 8 October 2026, against `main` at `af148b5`, before any 2025 book, print, hourly bar or funding file is fetched for it. Revised the same day, before it was merged and before any data was read: after three reviews (peeking, multiplicity, feasibility), a consistency check and a last review for decisions left until after a read, it was recast from a test of fixed hypotheses into a registered search; the choices the reviews and the recast left open are under [Choices and their reasons](#choices-and-their-reasons) |
| **What it registers** | A search for short-horizon strategies that pay at published fees, on Bybit's public BTCUSDT, ETHUSDT and XRPUSDT archives, over features, labels, models, entry gates, execution paths (taker, passive entry, market maker), a regime-break policy, holding horizons and instruments: the blocks, the data and its coverage, the [search space](#the-search-space) and its hashed catalogue, how the development block [chooses and admits](#selection-on-d-admission-and-the-finalists) finalists, the [placebo search](#the-placebo-search) a finalist must beat, the confirmation of five finalists on held-out blocks (the named configuration [S25](#s25-the-named-2025-configuration) and one slot per stratum and execution path) with their kill conditions and placebos, the measurements A0, FILL and the regime map, the break test R, the break detector, the status rules, the fetch plan, the compute budget and the amendment protocol |
| **Computed so far** | **Nothing.** No 2025 order book, print, hourly bar or funding rate has been fetched, opened or computed on. The only 2025 data on disk is the daily bars of an earlier, unrelated break-detector run (item 5 below). What was known is listed under [What was known before this was written](#what-was-known-before-this-was-written) |
| **How it changes** | Only under the [amendment protocol](#amendment-protocol): dated amendments appended at the end. One amendment is planned, after the development-period runs and before the held-out block is read |

The market-making rounds are
[`market_making.md`](market_making.md) and
[`market_making_round2.md`](market_making_round2.md); their conventions hold here
unless this document says otherwise. Names that do not exist on `main` at
`af148b5` refer to code added by the pull requests that follow this one; they
are listed under [the code to come](#the-code-to-come).

## Why this study

The repository's earlier directional and market-making studies were all
killed. This one asks whether a **registered search** over the parts
of a short-horizon strategy finds one that pays at published fees and survives
held-out data: the instrument, the features, the label and its fee, the model,
an entry gate, the holding horizon, the decision threshold, a policy for
regime breaks, and the execution path (taker, passive entry, market maker).
The search is fixed in advance. Its catalogue of configurations is enumerated
and hashed; every configuration is scored on the same 150 development days and
recorded in a trial ledger; the choice of finalists is a registered rule; a
searched finalist must beat the same search re-run on placebo targets before
it is admitted; and five finalists, a number fixed now, are tested on
held-out blocks under Holm's procedure with m = 5.

One configuration is named in advance: **S25**, a gradient-boosted classifier
on BTCUSDT with a hold-first rule and a 50 s close-by-time exit (its
coordinates are under [S25](#s25-the-named-2025-configuration)). It is a point
of the catalogue that the search can reach, and it is also tested by name,
whatever the search chooses, so the study always reports how one fixed
configuration fared, and the passive and market-making paths always have a
fixed signal to run on.

The study also asks whether a strategy paid for a while in 2025 and then
stopped. That question is the easiest one to answer by looking for the period
that flatters it. It is asked only in forms fixed in advance: as a searched
[regime-break policy](#the-regime-break-policy-axis) (a guard after each alarm
of a frozen break detector, causal on the development block), as a causal
stopping policy ([R](#r-the-break-conditional): stop at the first alarm),
tested only for a finalist that has first passed on the held-out block, and as
a [regime map](#the-regime-map) that cuts every finalist's daily net at the
detector's alarms and shows every segment, none selected, as a measurement.

**The prior is negative**, for reasons written down [below](#expected-outcome-stated-in-advance).
The study is built so that a negative answer is informative: every verdict
carries its fee break-even, so a reader can see how far each configuration
was from paying.

---

## What was known before this was written

Everything below was available when the design was chosen, and some of it
shaped the design. None of it is a 2025 number computed by this repository.

1. **General market knowledge**: crypto markets had a widely reported
   liquidation cascade on 10–11 October 2025, and BTC's broad 2025 path (a
   rising year, the reversal from October) is public.
2. **This repository's own results**: directional taker trading from one
   instrument's book is killed (§25, §26), passive entry of a directional signal
   is killed (the gross edge inverts when resting: +3 to +8 bp crossing, −0.5 to
   −4.2 bp resting), and both market-making rounds are killed, the one apparent
   profit being inventory carried through a rising market
   ([`docs/findings.md`](../findings.md)). These are the reason for the prior.
3. **Bybit's fee schedule**, transcribed in 2026 into `backtest/costs.py`
   (`BYBIT_LINEAR_TIERS`), with its caveat (`BYBIT_FEE_CAVEAT`): it is the
   schedule published in 2026, applied here to 2025 data, when it may have
   differed. The constant's text names the earlier rounds' period, February
   to April 2024; the follow-up pull requests parametrise the caveat by
   period, and every caveated number here carries it with 2025's dates.
4. **Archive metadata, read for this registration.** HTTP `HEAD` requests for
   Bybit's BTCUSDT, ETHUSDT and XRPUSDT order-book and trade archives on every
   day from 2024-12-01 to 2025-12-31 returned each file's existence and size and
   nothing else; no archive was downloaded. The result, which moved two block
   boundaries, is in [Data](#data). The per-day sizes were seen; they reflect
   the archives' depth and a day's activity, not prices, spreads or fills, and
   a file's size is a rough proxy for that day's activity in H, F and P. The
   per-day tables are committed as
   [`experiments/results/short_horizon_2025_head_check.csv`](../../experiments/results/short_horizon_2025_head_check.csv)
   (BTCUSDT, sha256
   `86a5719fd92845d111bf846c15fded09409ff084f42679a556c23e113cf4d8e0`) and
   [`experiments/results/short_horizon_2025_head_check_eth_xrp.csv`](../../experiments/results/short_horizon_2025_head_check_eth_xrp.csv)
   (ETHUSDT and XRPUSDT, sha256 in [Data](#data)).
   **No value in this registration was set from what they showed, except the
   H/F boundary the BTCUSDT check moved**; ETHUSDT and XRPUSDT show the same
   archive change on the same day. The revisions made after the reviews used
   only that archive change, and the fetch plan's sizes and disk budget use
   the per-block sums and the largest and median archive of W and D, which
   set no searched value. Three further `HEAD` requests on the same day found
   the `ob200` book archive, and no `ob500` one, for BTCUSDT on 2026-01-05, 06
   and 07 (225, 205 and 197 MB); these are the reader-test days of
   [Data](#data), outside every block.
5. **A break-detector run that already covers 2025.** The repository holds a
   run of `validation/structural_breaks.py` at its daily setting on Binance
   BTCUSDT daily bars from 2024-01-01 to 2026-09-30
   (`experiments/results/structural_breaks_BTCUSDT*.csv` and the README's
   figure), committed by an earlier run in this repository for another
   purpose; the daily bars it read are on disk. Its 2025 alarms are public,
   and they were known when this was written: the README, in the paragraph
   under that figure, written on 2 October 2026, names the run's one break as
   **2025-09-13, scale, downward** (odds 9.36 against 9.09, seventy-five days
   into a window opened on 2025-06-30) and a near miss on **2025-06-29**
   (8.96). Both were in this repository's README when F's span and the hourly
   detector's setting were set; the 2025-09-13 alarm falls inside F. Because
   it is public, that run cannot place the break blind. It is reported beside
   R and never used to define the break date; the detector that does is a
   different one, on hourly bars, never run on 2025
   ([The break detector](#the-break-detector)). The same figure, and the daily
   bars on disk, show BTC's 2025 daily price path.
6. **Nothing else.** The repository's Bybit books end on 2024-05-10, and the
   daily bars above are its only 2025 data. No 2025 book, print, hourly bar or
   funding rate has been fetched or opened, and no 2025 number has been
   computed for this study.

**The information set of S25.** S25's coordinates are not claimed to have been
chosen without knowledge of market conditions after 2025-08-01. Of the scored
2025 days, only D and H's days before that date (2025-07-08 to 07-31) lie
outside that information set. Every slot whose signal is S25's (S25, S25-Z,
and S-P and S-M when they run on S25's signal) therefore takes its status on
H from those 24 days only, its best status is **candidate (H before
2025-08-01)**, and its numbers on the rest of H, on F and on P are
measurements labelled "inside S25's information set"
([status rules](#status-rules)).

---

## Expected outcome, stated in advance

**No finalist is expected to pay at the base fee.** The arithmetic:

- A taker strategy pays its fee on the notional of each side, and turnover
  counts both sides, so the fee at which it breaks even, per side, equals its
  gross profit in basis points of turnover.
- Bybit's taker fee is 5.5 bp at the base tier and **1.5 bp at the cheapest
  published tier** (`pro6_g1`, from 5,000 million USDT of 30-day volume). A
  taker round trip costs 11 bp at the base tier and 3 bp at the cheapest, so a
  taker configuration pays at the base tier only with a gross edge above
  5.5 bp of turnover.
- A passive entry with a taker exit costs 2.0 + 5.5 = 7.5 bp a round trip at
  the base tier, a break-even of 3.75 bp of turnover; with a 0.0 bp maker
  entry (published for BTCUSDT only from Supreme VIP or Pro 5) and the
  cheapest taker exit it costs 1.5 bp, a break-even of 0.75 bp.
- In 2024 this repository found the same directional signal at the same
  moments worth +3 to +8 bp crossing and −0.5 to −4.2 bp resting, against
  costs of 14–16 bp crossing and 5.9–7.1 bp resting; every taker winner of the
  grand search was negative on both blocks, and both market-making rounds were
  killed ([`docs/findings.md`](../findings.md)).

Predictions, one per slot, fixed now:

- **S25** killed by K1 on H before 2025-08-01; **S25-Z** killed by its K1
  there.
- **S-C and S-R**: whether a searched signal is admitted for its information
  is not predicted (a crossing gross edge of a few basis points can beat both
  nulls); its payoff is: the chosen cell's S ≤ 0 at the base fee in both
  strata, so both are `not admitted on D (i)`.
- **S-P and S-M** `not admitted on D (i)` on whichever signal they run; if
  either reaches H, killed by K1 or, for S-M, K-dir.
- **The regime axis** adds nothing beyond random pauses.
- **R** not tested, because nothing reaches it; **FILL** as stated under its
  heading.

---

## Data

| Plane | Source | Code | Blocks |
|---|---|---|---|
| Ten-level book on a 100 ms grid | Bybit order-book archive, BTCUSDT, ETHUSDT and XRPUSDT linear perpetuals, 500 levels (`ob500`) | `data/bybit.py`, `ARCHIVE_URL` | W, D, H |
| the same | the same archive at 200 levels (`ob200`), same path with `ob200` in the name | `data/bybit.py`, a second name to be added | F, P |
| Prints with aggressor side | Bybit trade archive, `public.bybit.com/trading/<symbol>/` | `data/bybit_trades.py`, `ARCHIVE` | all but E |
| Funding | Bybit public REST endpoint, no credentials | `data/bybit_funding.py` | all but E |
| Hourly bars for the break detector (BTCUSDT only, for every instrument) | Binance USDT-M futures BTCUSDT 1h klines, `data.binance.vision`; daily archives only for every day after 2025-06-30 | `data/binance.py`, with a daily-only option to be added | from 2024-12-01, E included (bars only) |

**Coverage, checked by HTTP `HEAD` on 8 October 2026** (existence and size
only; nothing downloaded). Sizes in MB. The `ob500` and print columns use the
URL patterns in `data/bybit.py` and `data/bybit_trades.py` as they stand; the
`ob200` column is the same book URL with `ob500` replaced by `ob200`.

| Day | Book `ob500` | MB | Book `ob200` | MB | Prints | MB |
|---|---|---:|---|---:|---|---:|
| 2024-12-01 | 200 | 211 | 404 | — | 200 | 35 |
| 2025-01-01 | 200 | 218 | 404 | — | 200 | 34 |
| 2025-01-15 | 200 | 368 | 404 | — | 200 | 75 |
| 2025-02-01 | 200 | 231 | 404 | — | 200 | 34 |
| 2025-02-15 | 200 | 194 | 404 | — | 200 | 19 |
| 2025-03-01 | 200 | 351 | 404 | — | 200 | 51 |
| 2025-03-15 | 200 | 274 | 404 | — | 200 | 23 |
| 2025-04-01 | 200 | 376 | 404 | — | 200 | 66 |
| 2025-04-15 | 200 | 359 | 404 | — | 200 | 65 |
| 2025-05-01 | 200 | 283 | 404 | — | 200 | 59 |
| 2025-05-15 | 200 | 347 | 404 | — | 200 | 62 |
| 2025-06-01 | 200 | 210 | 404 | — | 200 | 32 |
| 2025-06-15 | 200 | 218 | 404 | — | 200 | 28 |
| 2025-07-01 | 200 | 224 | 404 | — | 200 | 40 |
| 2025-07-15 | 200 | 362 | 404 | — | 200 | 94 |
| 2025-08-01 | 200 | 374 | 404 | — | 200 | 84 |
| 2025-08-15 | 200 | 304 | 404 | — | 200 | 50 |
| 2025-09-01 | **404** | — | 200 | 206 | 200 | 58 |
| 2025-09-15 | **404** | — | 200 | 149 | 200 | 47 |
| 2025-10-01 | **404** | — | 200 | 169 | 200 | 62 |
| 2025-10-15 | **404** | — | 200 | 237 | 200 | 89 |
| 2025-11-01 | **404** | — | 200 | 106 | 200 | 25 |
| 2025-11-15 | **404** | — | 200 | 196 | 200 | 53 |
| 2025-12-01 | **404** | — | 200 | 289 | 200 | 129 |
| 2025-12-15 | **404** | — | 200 | 243 | 200 | 96 |

The same check on **every day** from 2024-12-01 to 2025-12-31 (396 days), for
BTCUSDT, and identically for ETHUSDT and XRPUSDT:

| Archive | Present | Absent |
|---|---|---|
| Book `ob500` | 2024-12-01 to 2025-08-20 (263 days) | 2025-08-21 to 2025-12-31 (133 days) |
| Book `ob200` | 2025-08-21 to 2025-12-31 (133 days) | 2024-12-01 to 2025-08-20 (263 days) |
| Prints | all 396 days | none |

**The book archive changed on 2025-08-21.** Under the URL pattern the code
uses, every book day from 2025-08-21 is missing; the venue publishes those
days at 200 levels instead of 500, and no day has both. Ten levels are inside
either, but the two archives may differ in ways the existence check cannot
show (message cadence, snapshot frequency), and no overlap day exists to
compare them on. Prints are unaffected.

**ETHUSDT and XRPUSDT** were checked the same way on the same day, every day
from 2024-12-01 to 2025-12-31
([`short_horizon_2025_head_check_eth_xrp.csv`](../../experiments/results/short_horizon_2025_head_check_eth_xrp.csv),
sha256 `6e294fbed3d259a2eafa96472380cbc52fa64abdc45e686ee3157453e3aa3c9e`).
Their coverage is BTCUSDT's: `ob500` on 263 days to 2025-08-20, `ob200` on 133
days from 2025-08-21, no day with both, and prints on all 396 days. The
archive change therefore falls on the same H/F boundary for every instrument.
Their archive sizes per block, in GB of book and prints:

| Instrument | W and D | H | F | P |
|---|---:|---:|---:|---:|
| ETHUSDT | 70.2 + 14.1 | 12.7 + 3.8 | 18.5 + 7.4 | 9.4 + 3.6 |
| XRPUSDT | 49.6 + 12.3 | 11.1 + 2.3 | 17.3 + 3.3 | 7.0 + 1.3 |

**What the archive change moved.** The plan before the check ended H on
2025-09-21 and began F on 2025-09-22. H would then have mixed the two archives,
and every held-out verdict would have rested partly on a book source that D
never saw. Before anything else in this registration was written, the blocks
were adjusted: **H now ends on 2025-08-20**, the last `ob500` day, and **F
begins on 2025-08-21**, the first `ob200` day. D and H, where every value is
chosen and every held-out verdict is taken, share one archive; F and P share
the other. The archive change falls on the H/F boundary, where it is visible,
instead of inside a block. Its consequences:

- the data pull request adds the `ob200` name to `data/bybit.py` for days from
  2025-08-21, with a test on a synthetic `ob200` fixture and on the three
  **reader-test days, 2026-01-05, 06 and 07** (BTCUSDT; the reader is the
  same code for every instrument), which lie outside every block
  and are used for nothing else but the entry point's dry run (no successor
  study may put them in a block). No `ob200` day from 2025-08-21 to 2025-12-31 is fetched before
  stage 3. The `ob200` replay is proven before H is opened: each reader-test
  day must pass as the study's loader passes a block day, with no
  `positive_prices` error in `validate_book`'s report, no price off the tick
  grid, and no more sequence gaps than D's frozen `max_sequence_gaps`. Their
  message counts, median interval between messages and sequence-gap counts,
  and every other finding of their reports by check and severity beside the
  same counts for D, go in the amendment. A day that fails is a reader fault,
  corrected under the amendment protocol before H is opened; the replay
  itself is unchanged;
- F and P use D's frozen `max_sequence_gaps` (below), never a value set on
  `ob200` days. If more than a quarter of a held-out block's days are
  excluded, that block is reported as **unreadable**, nothing is retuned, and
  no status changes on it (the
  [status rules](#status-rules) say what
  follows for H, F and P);
- the downloader records each F and P day's message count, median interval
  and sequence gaps only inside that block's opened access, as it records
  row counts;
- every comparison across the H/F boundary (a strategy's persistence from H
  into F) is labelled as crossing an archive change;
- R compares days before and after the break **within F**, on one archive,
  and the break date comes from Binance bars, which the change cannot move;
- the order-book CUSUM reported beside R (below) labels any alarm within a day
  of 2025-08-21 as "at the archive change".

---

## Blocks

| Block | Dates (UTC, inclusive) | Days | Book archive | Use | Read |
|---|---|---:|---|---|---|
| **W** warm-up | 2024-12-01 to 2024-12-31 | 31 | `ob500` | normalisers, size floors, rolling statistics; never scored | freely |
| **D** development | 2025-01-01 to 2025-06-30 | 181 | `ob500` | features, models, the whole search and its placebo search, the walk-forward results, σ of daily results, the minimum detectable effects, the detector's alarm rate; A0 and FILL on O | freely, through the development access |
| O (inside D) | 2025-02-04 to 2025-02-26 | 23 | `ob500` | A0 and FILL, descriptive only | as D |
| **E** embargo | 2025-07-01 to 2025-07-07 | 7 | `ob500` | a gap, so that H tests transfer rather than continuation | book, prints and funding never fetched; its Binance hourly bars only, for the detector, opened with H |
| **H** held out | 2025-07-08 to 2025-08-20 | 44 | `ob500` | the five finalists and S25-Z | once, after the D amendment is committed |
| **F** boundary | 2025-08-21 to 2025-11-16 | 88 | `ob200` | the break detector's alarms; every frozen finalist's numbers; persistence of anything that passed on H; R | once, after H's results are committed |
| **P** after | 2025-11-17 to 2025-12-31 | 45 | `ob200` | the last check of anything that passed (below) | once, after F's results are committed, only if the condition holds |

Why these blocks.

- **W** exists because the size floors and rolling normalisers must be fitted
  on data before the first scored day; December 2024 is never scored.
- **D** covers the first half of 2025, and with it O, a 23-day window on
  which A0 and FILL are measured. O is development data: the measurements on
  it are descriptive, and nothing on O is held out.
- **E** is a week no strategy reads, so that a model frozen on 30 June is
  first scored a week later. Only its hourly bars are read, by the detector,
  so that its stream has no hole.
- **H**, six weeks of July and August, is before the autumn of 2025, and
  ends at the archive change.
- **F** contains the autumn, the 10–11 October cascade, and the archive change
  on its first day. It is where the frozen detector looks for the break.
- **P**, the last six weeks of 2025, is read only for a finalist that passed
  before it.

Days are independent: each starts flat, no position is held across midnight,
and the day is the cluster for every statistic. Rolling statistics may read
the previous days of the same block, and of the block before it where the two
are adjacent and share an archive (W before D, F before P); a reader never
reads a later block than its access permits. **Two days start cold**:
2025-07-08, because E's book is never fetched, so H's rolling statistics read
nothing of D (not even 2025-06-30), and 2025-08-21, because the archive
changes on it, so F's read nothing of H. On each, rolling state is rebuilt
from the block's own rows, and the first `N` minutes from its midnight are
not scored, `N` being the longest feature lookback (recorded in the
amendment), whatever days they span; a day stays usable with fewer decision
rows. Fitted values (the size floors fitted on W, the frozen models and
thresholds) are not rolling state and are used on every block, never
refitted at a read. Every fitted value that is not a weight (W's size floors
and normalisers, the tensor's floors, each instrument's clip, any other
fitted constant) is written
to one file whose sha256 the amendment records, and the H, F and P gates
check it beside the weights.

Exclusions are per instrument-day, and a block's unreadability is judged per
instrument, on the days of the instrument a slot trades. Days excluded from
every verdict, counted and reported per block and instrument: more book
sequence gaps than `max_sequence_gaps`; a non-positive price, found as a
`positive_prices` error in `validate_book`'s report (the function returns its
findings and does not raise, so the study's loader excludes the day on that
finding); a price off the tick grid, which `validate_book` does not check
(the simulator's day loader, `market_making/events.load_day`, refuses such a
day under R16, and the study's loader applies the same check to every day,
taker strategies included); a missing plane (an HTTP 404 only; a failed
download is retried, and excluded only under the [fetch
plan](#fetch-plan)'s two terminal rules);
a missing funding history; or a day flagged `flatten_stale` (R16: a taker
order, an exit or the day-end close, that no snapshot followed before the
book ended walked a snapshot more than 5 s older than the order); a day
`market_making/events.load_day` refuses (`DayUnavailable`, whatever the
reason: a funding interval other than 8 h, print sizes or aggressor flags
outside its contract, no print on the day), for every configuration, taker
included; or an unreadable archive or unfetchable day ([fetch
plan](#fetch-plan)). Each is counted by reason. **A day flagged
`flatten_stale` in any run on the block** (on a held-out block, any run of
its read session: the true strategy, a placebo draw, a neighbour cell, the
K-latency re-run, a bracket) is excluded from every series of every
slot on that block, and is counted once toward unreadability.
**`max_sequence_gaps` is the 95th percentile of D's per-day gap
counts, rounded up**, per instrument, computed by that formula and recorded in
the amendment; the same value is used on every block for that instrument. Each verdict is also reported with the
gap-excluded days included, as a measurement, so that the exclusion cannot
quietly favour a result.

---

## Common settings

- **Instruments**: Bybit BTCUSDT, ETHUSDT and XRPUSDT linear perpetuals, a
  categorical axis of the search. Each is fetched, replayed, validated and
  excluded day by day under the same rules; the loader is per instrument, and
  its tick and lot steps come from the venue's instrument specification,
  recorded in the amendment.
- **Rows and timing**: the 100 ms rows are `data/bybit.reconstruct`'s, at
  `grid_ms=100`: at most one row per 100 ms interval, emitted at the
  interval's first update, stamped with that update's venue time and carrying
  the book as it stood right after it. Later updates in the interval appear
  only in the next row, and an interval with no update has no row; nothing is
  forward-filled. A row's value is therefore known at its stamp, and decisions
  are made at row stamps. Orders and cancels are live 10 ms after the decision
  (simulator rule R3). Every quantity in seconds (hold, label horizon, entry
  wait, lookbacks) is measured on the stamps: "`t + h`" is the first row
  stamped at or after `t + h`. `data/grid.to_grid` works in whole seconds and
  is used only for the 1 s grid of the book CUSUM, with `label="right"`.
- **Clip**: 0.010 BTC a trade on BTCUSDT, for every configuration and every
  execution path, with no Kelly sizing. On ETHUSDT and XRPUSDT the clip in coin
  units is fixed per instrument at 0.010 × (BTC's mean mid on W) / (the coin's
  mean mid on W), rounded down to the lot step, and written to the
  fitted-values file ([Blocks](#blocks)), so that a clip has about the same
  notional on every instrument. Results are also given per 100 USDT of clip a
  day, with the clip's notional taken at entry. One position at a time, never more than one clip (K-cap).
  In the simulator the clip is the fixed `clip_btc` setting, which replaces
  `clip_notional` and the touch-share cap, so K-cap compares with a constant.
- **Taker fills** come from an event-time taker executor (new code; the
  existing `pipeline/execution.run_taker` prices mid to mid and is not used).
  An entry walks the first snapshot at or after decision + 10 ms across the
  ten visible levels, at their volume-weighted price for the clip, with no
  slippage added on top of the walk. An entry whose clip exceeds that
  snapshot's visible depth is skipped and counted; **this skip applies only at
  entry**, and no entry is skipped for its snapshot's age (a decision is made
  at a row stamp, so the book it reads is fresh, and the snapshot walked is
  never older than the order). An exit walks the first snapshot at or after
  its decision + 10 ms whatever its age; depth beyond the tenth level is
  charged at the tenth level's price and counted, as the simulator's
  `flatten_beyond_visible_book`. **An order whose walked snapshot is stamped
  more than 1 s after the order arrives** (a feed gap) is counted as
  `entry_late_snapshot` or `exit_late_snapshot` and reported, for every taker
  order, the passive slots' taker exits included; it is a measurement
  and excludes no day. An exit that no snapshot follows before the book ends
  walks the last one, and if that snapshot is more than 5 s
  (`suspend_after_pause_s`) older than the order the day is flagged
  `flatten_stale` and excluded, as in the simulator (R16). Funding (R14) and
  the accounting identity (R15), with turnover equal to the sum of fill
  notionals, hold for it as for the simulator. The label's walks and the
  executor's call the same function.
- **Passive fills** (S-P, S-M, S25-Z, A0 and FILL) come from round one's event-time simulator
  with these settings: a queue per order joining the tail of the visible size
  (R5), advance and fills only from prints at the price (R6) or through it
  (R7), proportional cancellation attribution, pro-rata arrival growth,
  trade-tape crossed policy, post-only, order and cancel latency 10 ms, feed
  latency 0, `soft_limit_clips` 1, `max_sequence_gaps` the frozen value of
  [Blocks](#blocks) (the simulator's default of 0 is not used), `warmup_s` 60,
  quoting stops 23:58 and the
  flatten at 23:59, markouts at 1, 5, 10 and 30 s, decomposition at 5 s,
  `suspend_after_pause_s` 5 (no snapshot for 5 s pulls every resting order
  and nothing is quoted until the next one; the same 5 s is R16's
  `flatten_stale` threshold), and `vol_half_life_s` 60 and `touch_window_s`
  3600, the simulator's defaults, which the signal-entry quoter does not use.
  The entry is a new signal-entry quoter (round one's `SignalExecutor`, which
  re-pegs and exits passively, is not used): one clip posted at the touch on
  the signal's side at decision + 10 ms, **kept at its first price**, never
  re-pegged, and cancelled if unfilled after `h`. **An entry pulled by a pause
  is not re-posted**: its unfilled part counts as unfilled, and a part filled
  before the pause is exited as any fill. Once filled, the exit is a
  reduce-only cross `h` after the fill, priced by the taker executor's walk
  (`flatten_slippage_bp` 0), with the late-snapshot counters of the taker
  executor. **The decision reaches the quoter as a tape row stamped at the
  decision row's stamp `t`.** Under the simulator's strictly-before rule that
  row is first seen by the event after `t`, and the order is live no earlier
  than `t` + 10 ms; a unit test checks that no event at or before `t` sees the
  signal and that every post is live at or after `t` + 10 ms. A regime guard
  ([the regime-break policy axis](#the-regime-break-policy-axis)) cancels a
  resting entry at the first event after the flag. The brackets (`none`,
  `all_ahead`; `pessimistic`) are run for K-queue.
- **The market maker** (S-M) is round one's S1 with the signal lean, on the
  same simulator settings: S1 at round one's frozen values `skew_bp` 2, `k`
  0.5 and `min_edge_bp` 4 (`configs/mm_prereg.yaml`), `soft_limit_clips` 1,
  `clip_btc` the instrument's clip, the lean λ ∈ {0.5, 1, 2} on its signal's
  tape (`pipeline/market_making.signal_tapes`: the signal's accepted entry
  decisions, each with a signed strength in bp, μ for a regressor and, for a
  classifier, the side times the median absolute forward move over the hold
  on the fold's training rows), `lean_window_s` equal to the signal's hold,
  and the regime guard of
  `pipeline/market_making` (`guard_pull`, or `guard_widen` as a measurement)
  with `guard_window_min` equal to 60 × the window in hours.
- **Fees for verdicts**: the base tier, 2.0 bp maker and 5.5 bp taker
  (`BYBIT_BASE`), for every instrument. Other published tiers
  (`BYBIT_LINEAR_TIERS`, group `g1`, which holds all three instruments, down
  to 0.0 maker and 1.5 taker) and the programme's −1.0 bp maker rebate are
  measurements only, each with `BYBIT_FEE_CAVEAT`. A 0/0 schedule is used for
  A0 and S25-Z only, and labelled; a label fee of 0 bp is a searched value of
  the classifier's label, not a fee any verdict is taken at.
- **Funding** is charged at each settlement on the position held then
  (R14); a day without its funding history is excluded.
- **The day**: no entry before 00:01 or after 23:58 UTC; anything open at
  23:59 is closed by a costed taker order. The accounting identity (R15) is
  checked at every day's end.

---

## Labels, features and models

Every label, feature and model below is built per instrument; nothing is
pooled across instruments.

**Labels.**

- **The walked-book taker label** (new code), stratum C. At row `t` with hold
  `h`, for the clip: long P&L = (VWAP of selling the clip into the bids at the
  exit snapshot − VWAP of buying it from the asks at the entry snapshot) / the
  entry snapshot's mid × 1e4 − 2 × fee, the entry snapshot being the first at
  or after `t` + 10 ms and the exit snapshot the first at or after that + `h`,
  walked by the taker executor's function; short symmetric. Class `long` if
  the long P&L is positive, `short` if the short P&L is, `hold` otherwise. The
  **label fee** is a searched value, 0 or 5.5 bp, charged on both sides; S25's
  and A0's label is at 0 bp. The label clip is the traded clip and the label
  horizon is the hold, so the label declares `h` and the purge is derived
  from it.
- **The forward move**, stratum R: the mid at the exit snapshot over the mid
  at the entry snapshot, − 1, in bp (a simple return), both snapshots as for
  the walked label; forward-only, unsmoothed. It is new code:
  `labels/targets.py`'s `forward_smoothed_move_bp` at `smoothing=1` gives the
  log of the mid's ratio from the decision row with its horizon in rows, and
  is not used.
- **The control label**: a time-based form (new code) of
  `labels/targets.py`'s `smoothed_move_bp` at `smoothing=20` rows: the log of
  the mean mid over the 20 rows ending at the first row stamped at or after
  `t + h`, over the mean mid over the 20 rows ending at `t`, × 1e4. Its
  horizon is in time, as every horizon here; only its smoothing is in rows.
  It is partly known at decision time and is used only to train the control
  models of K-label: the network on the control label, at the frozen fits
  only, on R*'s instrument and the holds of R*'s neighbourhood, when R*'s model
  is the network ([selection](#selection-on-d-admission-and-the-finalists)).

**Features.**

- **The microstructure set**: the registry's book, depth and flow features
  over ten levels, plus the 2025 set the features pull request adds where the
  registry lacks it (one-second realised volatility, return-sign
  autocorrelation, turn count, run length, inter-trade time, lagged level-0
  logs). **Winsorising** clips each such feature at the 0.1% and 99.9%
  quantiles of its own trailing 3,600 s on row stamps, never at whole-day
  quantiles.
- **The ten-level tensor**: 40 columns. Each level's price distance from the
  touch is divided by the larger of σ × mid and a spread floor, σ being the
  standard deviation of the mid's row-to-row log changes over the trailing
  60 s on row stamps, and the floor the instrument's median quoted spread on
  W. Each level's log(1 + size) is z-scored against its mean and standard
  deviation on W. W's values are written to the fitted-values file.
- **Prints are aligned backward**: a print belongs to the first row stamped
  at or after it, so row `t` holds the prints in (s_{t−1}, s_t], s being the
  row stamps (`searchsorted(..., side="left")`, the rule of
  `features/align.py`). `features/flow.align_to_grid` puts a print into the
  last row stamped at or before it, which on `reconstruct`'s event-time rows
  gives row `t` the prints after its stamp; it is not used as it stands. The
  existing flow features and the 2025 trade-plane features (inter-trade time
  included) use the backward rule.
- Every feature passes `validation/leakage.py` before any D run (K-leak). The
  depth and flow builders, which are frame builders outside the registry,
  are wrapped as registered features with declared lookbacks, so that the
  checker, the purge and `N` cover them. For trade-plane features the
  checker cuts the prints at a time strictly inside a row interval, not at a
  row, and the streaming-parity test compares values at each decision stamp.
  Every gate input is in basis points computed from prices, never a
  normalised column (K-units).
- **Lookbacks and `N`**: every rolling window is declared in seconds on row
  stamps, except the network's 64-row window and the control label's 20-row
  smoothing. `N` is the longest lookback declared in seconds, rounded up to
  whole minutes; on a cold-start day a row is also unscored until the
  row-declared windows have their rows since that day's midnight.
- **Selection**: `features/selection.py`, once per instrument, on that
  instrument's January, from the microstructure set, on every tenth row
  (31 × 86,400 = 2.68 million rows at most), against one target: the
  walked-book label at 0 bp with `h` = 20 s as `y`, and the forward move at
  `h` = 20 s as `forward`. `FeatureSelector` runs at its defaults
  (`score_target="ic"`, at most 40 features, correlation 0.95, variance 1e-6),
  so it ranks by absolute correlation with `forward`, and `y` only marks the
  rows that have a label. The selected list ("micro-selected") is each
  instrument's input set for stratum C's micro inputs, for the bound regressor
  and as the network's auxiliary input; it is recorded in the amendment.

**Models.** No hyperparameter of any model is searched; every model uses
this repository's defaults, with one registered seed per fit.

- **Gradient boosting** (`models/gbm.py`, `gbm` below), three classes. With
  `balance_classes` (its default), a fit (a walk-forward fold or a frozen
  model) whose training rows lack the `long` or the `short` class gives that
  side a probability of 0 (`models/linear._align_columns` fills a missing
  class with zeros), so that side never trades with that model. This is the
  code's intended behaviour, not a fault; the fits and sides it affects are
  reported. The same holds for the logistic model.
- **Logistic regression** (`models/linear.LogisticBaseline`, `logistic`),
  three classes, the cheap baseline of stratum C.
- **The Gaussian-head network** (`tcn`): `models/tcn.py`'s causal dilated
  network (window 64 rows, 32 channels, 4 levels, kernel 3, dropout 0.1, at
  most 20 epochs) with a head for μ and log σ², trained by Gaussian negative
  log-likelihood plus 0.01 × the mean of (log σ² − log v)², v being the
  variance of the fold's training targets, with log σ² clamped to
  log v ± 6. Early stopping (patience 3 epochs) and the checkpoint both read
  the rank correlation of μ with the realised forward move on the inner
  validation; `TCNBaseline`'s internal 15% split is not used. Fits run on the
  MPS device, one at a time.
- **The bound regressor** (`bound_ridge`, `pipeline/stages.UNCERTAIN`): two
  ridge models (`models/uncertainty.HeteroscedasticRegressor`, at its
  defaults), μ from the first and σ from the second, the cheap baseline of
  stratum R.
- **Windows**: each of the network's windows is the 64 consecutive 100 ms
  rows ending at the decision row, in training and in inference alike.
  Training targets are taken on every tenth row (about one a second);
  decisions are made on every row. Windows are built per batch by a lazy
  window dataset (new code) that streams them per day from the every-row
  feature store and never materialises them for a whole fit or day:
  `models/tcn._windowise` copies, and at 4 million rows of about 70 inputs a
  materialised fit would need about 72 GB.
- **Training rows**: every fit reads at most 4 million training rows, taken
  evenly in time from every tenth row.

---

## The search space

The search is a catalogue of configurations. A configuration's **cell** is
its full set of values; its **group** is its categorical values (instrument,
stratum, model, inputs, gate); the ordered values (hold, decision, label fee,
regime window) span the group's grid. Every cell is scored on every one of
D's 150 walk-forward test days ([selection](#selection-on-d-admission-and-the-finalists)).

| Axis | Values | Kind | Code |
|---|---|---|---|
| Instrument | BTCUSDT, ETHUSDT, XRPUSDT | categorical | `data/bybit.py`, `data/bybit_trades.py`; a per-instrument loader and clip (new code) |
| Stratum | C: a three-class classifier with the hold-first rule; R: a Gaussian regressor with the EV gate | one slot each | the hold-first rule and the EV gate (new code) |
| Model | C: `gbm`, `logistic`; R: `tcn`, `bound_ridge` | categorical | `models/gbm.py`, `models/linear.py`, `models/tcn.py` with the Gaussian head (new code), `models/uncertainty.py` |
| Label | C: the walked-book taker label; R: the forward move | fixed by stratum | new code |
| Label fee (C) | 0, 5.5 bp | ordered | — |
| Inputs | C: micro-selected, or micro-selected + tensor; R: the network on tensor + micro-selected, the bound regressor on micro-selected | categorical in C, fixed by the model in R | `features/selection.py`; the tensor and the 2025 set (new code) |
| Hold `h`, equal to the label horizon | 5, 20, 50, 120 s on row stamps | ordered | time-based horizons (new code) |
| Decision (C) | 25, 50, 100, 240 trades a day, the hold-first thresholds' target rate | ordered | per-side θ (new code), on the pattern of `labels/directional.py` |
| Decision (R) | k ∈ {0, 0.5, 1} and β ∈ {0, 1, 2} bp | ordered | the EV gate (new code) |
| Market gate | `open`; `quiet_out` (realised volatility at or above the 30th percentile of the fold's training rows); `tight_only` (spread at or below their 70th percentile) | categorical | `backtest/gating.MarketGate`, as `experiments/grand_search.gate_for`, on the mid's realised volatility over the trailing 60 s and the quoted spread, both in bp from book prices |
| Regime policy, taker and passive entry | none, pull 4 h, pull 24 h, pull 120 h | ordered by window (none is window 0) | `pipeline/execution.guarded_rows`; fold-wise flags (new code) |
| Regime policy, market maker | pull at its signal's window; widen and the other windows measured | slot value | `pipeline/market_making` (`guard_pull`, `guard_widen`) |
| Execution | taker (S25, S-C, S-R); signal-entry quoter (S-P, S25-Z); S1 with the lean (S-M) | slot, never chosen | the taker executor and the quoter (new code); `pipeline/market_making` |
| Exit | close by time at `h`; a passive exit measured only | fixed | — |
| Measured, never chosen | the fee tier; the fill rule (touch, trade-through, queue) and the brackets `none`, `all_ahead`, `pessimistic`; latency 110 ms | robustness | the fill-rule comparison (new code) |

**Fixed, not searched**, with the reason:

- hyperparameters and seeds: the repository's defaults, one registered seed
  per fit;
- cooldown 0 beyond the one-position rule, and the monthly walk-forward
  cadence;
- the normaliser window: each value would need a new feature store, about
  45 GB per instrument;
- feature selection: once per instrument on its January, at most 40
  features;
- sizing: one clip;
- holds of 300 s and longer, already negative on the repository's 5 s grid;
- exits other than the clock: a take-profit or stop needs a walk along each
  holding path, about 25 times the cost of the clock exit;
- the flag source: BTC's hourly monitor drives every instrument; the book
  CUSUM stays a reported second detector.

**Counts.**

| | Categorical groups | Ordered grid per group | Taker cells | Signal configurations (fits per fold) |
|---|---:|---|---:|---:|
| C, per instrument | 12 (2 models × 2 inputs × 3 gates) | hold 4 × rate 4 × label fee 2 × regime 4 = 128 | 1,536 | 32 |
| R, per instrument | 6 (2 models × 3 gates) | hold 4 × k 3 × β 3 × regime 4 = 144 | 864 | 8 |
| **All three instruments** | **54** | | **7,200** | **120** (600 walk-forward fits) |

A gate, a regime window and a decision threshold need no fit of their own,
so 120 signal configurations (C: model × inputs × hold × label fee; R: model
× hold, per instrument) cover the 7,200 taker cells. The simulator runs on D
are fixed by rule ([selection](#selection-on-d-admission-and-the-finalists)):
S25-Z's 12 cells, always; S-P's at most 27 cells and 3 regime measurements
(its cell at each regime window other than its signal's own, of none, 4, 24
and 120 h); S-M's 3 cells and 6 regime measurements (`guard_pull` and
`guard_widen` at 4, 24 and 120 h with λ = 1); A0 and FILL's 4 runs on O. The
passive-exit variant of S-P and S25-Z runs on the held-out blocks only, never
on D. The **trial bound** is N_max = 7,200 + 12 + 30 + 9 + 4 = **7,255**; the
trial ledger records the actual N, and every row counts in N_trials even
beyond N_max.

**The catalogue** is the 7,200 taker cell keys, written
`<instrument>|C|<model>|<inputs>|<gate>|hold_s=<h>|trades_per_day=<n>|label_fee_bp=<f>|regime=<r>`
and `<instrument>|R|<model>|<inputs>|<gate>|hold_s=<h>|k=<k>|beta_bp=<β>|regime=<r>`,
with inputs `micro`, `micro+tensor` or `tensor+micro`, regime `none`,
`pull_4h`, `pull_24h` or `pull_120h`, and numbers written as in the
configuration (`0`, `0.5`, `5.5`). Sorted in byte order and joined with a
newline after each key, they have sha256
`45a32e8ada0401a4f4d225ba5c96197d5833927409b1db0a92e9fc8142aac4b5`. The
single entry point checks this hash before stage 1 and refuses any key
outside the catalogue. S25 is the key
`BTCUSDT|C|gbm|micro|open|hold_s=50|trades_per_day=240|label_fee_bp=0|regime=none`.

---

## Strata, decision rules and execution paths

**The hold-first rule** (stratum C). The model gives `p_long`, `p_hold`,
`p_short`. Go long when `p_long ≥ θ_long` and `p_long > p_short`; short
symmetrically; otherwise hold. `θ_long` and `θ_short` are set on the inner
validation, without the market gate or the regime guard, so that each side
would enter half of the target rate of trades a day before the one-position
rule. The gate and the guard only remove decision rows, so a gated or guarded
cell trades less than its target rate.

**The EV gate** (stratum R). With μ and σ the predicted forward move over `h`
in bp, `c` = 2 × 5.5 bp, `w_t` the cost in bp of walking the current book for
the clip and back (from the snapshot, in bp of mid), and `β` the buffer in bp
(not the break date `b`): go long when `μ − kσ > c + w_t + β`, short when
`−μ − kσ > c + w_t + β`. One entry per event (a run of rows on which the gate
is open); a new entry needs the gate to close and reopen. Every trade is one
full clip: no Kelly sizing is applied (`strategies/sizing.py`'s
`FractionalKelly` cannot size this gate in either unit; see
[Choices and their reasons](#choices-and-their-reasons)).

**Every configuration** holds one position at a time: the next entry is
possible only when the previous position is closed and no entry order is
resting. For a taker that is the hold after each entry; for the quoter it is
the wait of at most `h` for a fill, and after a fill the time to the exit.
Profit and loss comes only from realised fills in the execution path, never
from a sum of predicted values.

**The execution paths.** The taker enters and exits through the event-time
taker executor, the exit `h` after the entry. The signal-entry quoter posts
one clip at the touch on the signal's side, kept at its first price, waits at
most `h`, and once filled exits by a taker order `h` after the fill. The
market maker quotes both sides with round one's S1, its reservation price
leaning on the signal ([Common settings](#common-settings)). **The passive
variant with a passive exit** (S-P and S25-Z, measurement only): the exit is
posted at the touch at `h` after the fill and crossed if still open after a
further `h`.

---

## The regime-break policy axis

**The flags** come from the frozen break detector ([The break
detector](#the-break-detector)): hourly Binance BTCUSDT log returns over one
stream from 2024-12-01, each divided by the root mean square of its UTC hour,
under `MonitorSpec(history_len=365, online_len=90, min_history=121)` with the
statistics scale and dependence. A flag is an alarm, effective at the close of
the hour whose return produced it. BTC's flags guard every instrument, as a
market-wide break.

**Causal on D, fold by fold.** For test month m of the walk-forward, the
24-value hour profile is fitted on the hours from 2024-12-01 to the last hour
before m's first day; one `detect_breaks` call runs over the stream from
2024-12-01 to m's last hour; and the guard uses that run's alarms in
[m's start − 120 h, m's end]. On H, F and P, and in the dry run, the flags are
the frozen detector's, its profile fitted on all of D. A **flag-causality
test**, on the pattern of `validation/leakage.py`, checks in every fold that
mutating every bar after hour t leaves every flag at or before t unchanged; if
it fails, K-flag-leak voids every slot with a window above 0.

**The policy acts before execution**, and only on pairs that
`pipeline/execution.VALID_PAIRS` already allows:

- **none**: no guard;
- **pull w** (taker and passive entry): no new entry decision in
  (flag, flag + w], which is what `pipeline/execution.guarded_rows` computes.
  An open position runs to its exit at `h`. A resting passive entry is
  cancelled at the first event after the flag, since its fill would be a new
  entry;
- **the market maker**: pull is `guard_pull` (no quotes) and widen is
  `guard_widen` (half-spread doubled), with `guard_window_min` equal to
  60 × w.

The windows are w ∈ {4, 24, 120} h. 120 h is the monitor's re-arm time
(`min_history` 121 hours), so the longest guard covers the span after an alarm
in which the monitor cannot alarm again.

**Regime attribution**, reported for every slot with a window above 0: the
slot's mean minus that of its window-0 twin, against the 95th percentile of
the same difference under 200 random flag sets, each with as many flags as the
block has at hours drawn uniformly over its usable hours, seeds 0–199 (20 sets
for a simulator slot). It is a measurement, not a kill: a strategy that pays is
not a false positive because its guard adds nothing.

**Relation to R.** [R](#r-the-break-conditional) stays a separate,
conditional hypothesis about the policy "stop at the first alarm in F", with
the same frozen detector; the axis is a short guard after any alarm. A finalist
with a window above 0 keeps its guard on H, F and P, so F− and F+ are its own
guarded series, and `b`'s alarm triggers its guard too.

---

## The development-period procedure (D only)

The search runs in registered stages; the [compute budget](#compute-budget)
gives each one's time and disk.

0. **Stage 0, before any 2025 fetch: the hard budget.** The entry point
   checks the catalogue's sha256. The code to come is timed on 2024 days
   already on disk (BTCUSDT 2024-02-11, a quiet day, and 2024-03-05, the
   busiest; one XRPUSDT day of 2024), and D's wall time is projected from
   those times. **The ceiling is 6 days of wall time for D.** Over it, cuts
   are made in this order until the projection fits: (1) the network on
   ETHUSDT and XRPUSDT; (2) XRPUSDT entirely; (3) the 120 s hold; (4) stratum
   C's micro-selected + tensor inputs; (5) the `quiet_out` gate. The cuts, the
   new catalogue hash, the new N_max, the configuration's new
   null-placeholder sha256 and S25's and S25-Z's neighbourhoods as cut go
   into a dated amendment committed before stage 1. Nothing is cut once a 2025 file is on disk; a run slower
   than projected only takes longer. **The walk-table parity test** of [the
   placebo search](#the-placebo-search) runs here, on the same 2024 days,
   before any 2025 fetch: a failure blocks stage B until it is fixed by a
   dated amendment.
1. **Stage 1: fetch, replay and features**, W and D, one instrument at a time
   ([fetch plan](#fetch-plan)); no H, F or P file is on disk before the
   amendment. Validate every day; exclusions per the [blocks](#blocks).
   Features go to two float16 stores: every tenth row, for every instrument;
   and every row, holding the network's inputs, for one instrument at a time.
2. **Leakage**: every feature through `validation/leakage.py`, the fold-level
   leakage test and the flag-causality test; any failure stops the study
   until fixed by a dated amendment (K-leak, K-flag-leak).
3. **Selection**, per instrument, on its January.
4. **Walk-forward over D** by calendar month: for each test month from
   February to June (150 test days), train on D's days before it except the
   last seven, which are the inner validation; purge the hold plus the
   longest feature lookback before the inner validation and before the test
   month (`experiments/grand_search.py`'s purge is not used). The inner
   validation sets the per-side θ from the target rate, which reads no
   outcome, and the network's checkpoint by inner rank IC, which reads only
   days before the purge. **The fold-level leakage test**: mutating every
   label and price after fold m's training cutoff leaves fold m's weights, θ,
   gate levels and checkpoint unchanged (by sha256).
5. **Stage A: fits.** The 600 walk-forward fits (120 signal configurations ×
   5 folds), the predictions on the test days, and for each signal
   configuration and test day a **candidate-row tape**: the rows at which the
   loosest cell could enter (θ at 240 trades a day; k = 0 and β = 0; gate
   open; no guard), each with its scores, gate states and guard state. Every
   stricter cell is a subset of these rows. The daily IC is computed on all
   decision rows and stored as one value per day.
6. **Stage B: taker scoring, every cell on every test day.** Per
   instrument-day and hold, a **walk table** holds the long and the short
   walked P&L at every 100 ms row, from the taker executor's walk (10 ms
   latency, funding included), built in memory one day at a time and never
   stored. Each cell's entries are thinned from its tape (one position at a
   time, an event-jump loop); its trades' long and short walked P&L,
   `pnl_long_bp` and `pnl_short_bp`, are looked up; net is read on the cell's
   side, and **G = net − (`pnl_long_bp` + `pnl_short_bp`)/2**, which equals
   side × (`pnl_long_bp` − `pnl_short_bp`)/2: the cell's edge with the costs
   taken out. Drift is not taken out (a tape with more longs than shorts keeps
   the difference of the two shares times the drift); Null B and K-drift on
   the held-out blocks control it. Every cell-day writes one row to the trial
   ledger.
7. **Stage D: the choice, admission and the placebo search**
   ([selection](#selection-on-d-admission-and-the-finalists), [the placebo
   search](#the-placebo-search)), with the refit replication.
8. **Stage C: the simulator**: S25-Z, S-P and S-M on their signals, and A0
   and FILL on O.
9. **Stage E: the frozen fits.** For every hold and label fee in the
   neighbourhood of each stratum's chosen cell, admitted or not, and of
   S25's: at most 16 boosted or logistic fits and at most 4 network fits with
   4 control fits, trained on D from 2025-01-01 to 06-23 with the purge
   before 06-24; their θ and checkpoints set on 06-24 to 06-30. These cover
   every cell of every finalist's neighbourhood, so K-nbhd on a held-out block
   runs only frozen files. The weights stay in the ignored `models/`
   directory; every weight file's sha256, every threshold and every checkpoint
   epoch go in the amendment.
10. **σ of the daily series and the minimum detectable effects** (below), per
    slot, and the detector's alarm rate on D.
11. **A dry run of the entry point**, end to end, before the amendment: on
    D's last 44 days (2025-05-18 to 06-30) standing in for H, and on the three
    `ob200` reader-test days standing in for F and P. Its outputs are
    discarded and never reported; it is the reader-test days' one use besides
    the reader test. It repeats stage 0's walk-table parity test as a check; a
    failure there is a bug under the [amendment
    protocol](#amendment-protocol), and D is re-run in full.
12. **The amendment**, committed before H is read, with the study's single
    entry point (`experiments/short_horizon_2025.py`), which computes every
    slot, placebo, bracket, neighbourhood cell, tier, status and regime-map
    row for a block in one read session.

---

## Selection on D, admission and the finalists

**Frozen before scoring.** The configuration's `search` block fixes, before
stage 1: the catalogue and its hash, the choice and admission rules, the nulls
with their seeds, shifts and replications, the ceiling and the reduction
order. **The study code is frozen before the stage 1 fetch**: a dated note,
`docs/short_horizon_2025_code_freeze.md`, records the commit of all study code
(the entry point, features, labels, models, executor, quoter, nulls and
statistics) before any W or D file is fetched, and any change to that code
before H is read is a correction under the [amendment
protocol](#amendment-protocol). Every score on a test day of February to June
comes from the single entry point. It writes the **trial ledger**,
`experiments/results/short_horizon_2025_trials.parquet` (zstd), one row per
D run, configuration and test day, with the columns `run`, `key`, `day`,
`net_bp`, `G_bp`, `pnl_long_bp`, `pnl_short_bp`, `trades` and `skips`; the
catalogue's sha256 and each run's commit go in a sidecar,
`…_trials.json`, with the file's sha256 and row count, and the file is split
into parts under 100 MB if it needs to be. The ledger is committed at the end
of each stage. **N_trials** is the number of distinct (run, key) pairs: every
configuration scored in every complete D run, counted even beyond N_max. It
is the N quoted on every page and given to N_eff and the deflated Sharpe
ratio; the rows of a superseded run stay in the ledger and in N_trials.

**Hand runs.** No configuration is added to the catalogue after stage 0. A
computation on a February to June test day outside the entry point (a
configuration run by hand, or any statistic of a label, a return, a
prediction or a P&L on those days, a feature's IC included) is a **hand
run**. Replay, validation, the leakage tests and feature building are not.
A hand run is logged in the ledger's sidecar with its date, what it computed
and the days it read, and it sets every searched slot (S-C, S-R, and S-P or
S-M when their signal is C\* or R\*) to `not admitted on D (hand run)`, with
p = 1 on every block. The rule can only lower statuses, and it leaves the
catalogue and its hash unchanged. W and each instrument's January are the
sandbox, where nothing is counted.

**The choice: one cell per stratum.**

- In each group, a cell's **S** is the median, over its neighbourhood, of
  each cell's mean daily `net_bp_trade` at the base tier over the 150 test
  days. **The neighbourhood** is `validation/search.neighbourhood_scores`'
  block over the ordered axes (hold; rate, or k and β; label fee; regime
  window): every cell within one step on every ordered axis, diagonals
  included, at the cell's categorical values. Every cell is counted whatever
  its trade rate; a cell without a trade on D scores 0, which the entry point
  writes (`neighbourhood_scores` drops a missing value). The entry point uses
  a vectorised version of `neighbourhood_scores`, tested equal to it on small
  grids.
- Across every group of the stratum on all three instruments, the chosen cell
  (**C\*** or **R\***) is the one with the highest S among cells averaging at
  least 5 trades a day; ties go to the cell's own mean. The 5-trade filter
  only limits which cell may be chosen. The outright peak is reported beside
  it. If no cell qualifies, the stratum's slot is `not run (no admissible cell
  on D)`; the filter is not relaxed.

**Signal admission (information)**, for C\* and R\*; both conditions must
hold:

- **(iii) carry-forward**: for each test month m in March to June, the choice
  is re-run on the test months before m, and the newly chosen cell's G is
  scored on m's days after a 5-day embargo. The mean over the four months must
  be > 0. This mean, computed for net too, is the D estimate of the effect
  that the power on H is computed from;
- **(iv) the placebo search**: max(p_A, p_B) ≤ 0.05 ([below](#the-placebo-search)).

**Slot admission (payoff).** A slot that fails is `not admitted on D
(<conditions>)`, naming every condition that fails (for example `not admitted
on D (i, iv)`), and enters Holm with p = 1 on every block. A prediction of
`not admitted on D (i)` is scored on condition (i) alone.

- **S-C ← C\*, S-R ← R\***: the signal is admitted, and (i) the chosen cell's
  S > 0; and (ii) the neighbourhood median of the monthly mean net is > 0 in at
  least 4 of the 5 test months.
- **S-P**: its signal is the admitted C\* or R\*; if both are admitted, the one
  with the larger carry-forward G, C\* on a tie; **if neither is admitted,
  S25's signal**. It enters by the signal-entry quoter at that cell. Its
  neighbourhood is the signal's hold × rate × label fee (or hold × k × β) at
  the signal's categorical and regime values, every cell simulated on the 150
  test days. It is admitted on (i) and (ii) computed on
  `net_per_100_clip_day`.
- **S-M**: S1 with the lean on S-P's signal, the cell λ = 1 and its
  neighbourhood λ ∈ {0.5, 1, 2}, the lean window the signal's hold, the guard
  `guard_pull` at the signal's regime window (none when it has none). It is
  admitted on (i) and (ii) on `net_per_100_clip_day`, and on its making part
  being positive on D: the mean of `spread + adverse(5 s)` over its passive
  fills, in bp per fill, the quantity K-dir reads.
- S-P and S-M make no choice on a passive outcome, and S25's signal is fixed
  before D, so neither needs a search-level null; their conditions only remove
  slots.
- A searched slot whose chosen cell is S25's is `same as S25` and enters Holm
  with p = 1.

**Reported, gating nothing.** Every D number of a finalist is labelled
"selected, biased upward". Beside it: N_trials and
N_eff = `effective_count(N_trials, mean_pairwise_correlation(daily G panel))`
(`evaluation/significance.py`); the deflated Sharpe ratio of each finalist's
daily G and net; the selection premium (the median of the null maximum of A,
and A minus it); the probability of backtest overfitting by combinatorially
symmetric cross-validation per stratum (10 blocks of 15 days, 252 splits, the
identical rule); and the minimum detectable effect and the power at t ≥ 3 for
the carry-forward estimate.

**The finalists are fixed now: k = 5**, four searched slots (S-C, S-R, S-P,
S-M) and the named S25. Every slot exists on every block, whatever D shows.
The chosen cell of a non-admitted C\* or R\*, and a non-admitted S-P or S-M,
are still run once on H and F as labelled measurements, with no placebo and
no status, and on P only when P is opened for a qualifying slot, in that read
session. A sixth slot would cost power for no information: at 10 days on F
or P (9 degrees of freedom) the one-sided p at t = 3 is 0.0075, and Holm's
first step α/m stays above it only for m ≤ 6, so with more slots Holm rather
than the t ≥ 3 bar would decide on short F and P series. One slot per stratum
and execution path is the most informative use of five.

---

## The placebo search

**Purpose.** To measure how good the best configuration of each stratum looks
when no configuration has information, under exactly D's selection. A searched
finalist's p-value against the maximum over strata of that null must be at
most 0.05: single-step max-T (Romano–Wolf), at most a 5% chance that any
searched slot is admitted by noise. **p_search** is max(p_A, p_B) of the
stratum with the larger A, times R_D and capped at 1 (Bonferroni over D runs):
whether the search found anything beyond selection noise whatever the fee.
It is the one D number quoted as a result, labelled "on development data,
selected".

**What is re-run, and what is not.** Re-run: every step that reads a test
day's outcome, which is the per-cell daily series, the 5-trade filter, the
neighbourhood medians, the choice inside each group and across groups and
instruments, and the admission statistic. Not re-run: fits, θ, checkpoints,
feature selection, gate levels and guard flags; none of them reads a test
day's outcome, which the fold-level leakage test and the flag-causality test
enforce. Since no step culls a configuration on an outcome, re-running these
steps is the whole search re-run on placebo targets.

**The statistic.** A = the median, over the chosen cell's neighbourhood, of
each cell's day-level t of G, a cell without trades scoring 0. G is centred at
zero for every fee and execution path; the t keeps the scale when a null
pairing has another volatility; the neighbourhood median lowers the null
maximum.

- **Null A: hour-block sign flips, 999 draws.** Each UTC hour of the usable
  test days draws a sign from `numpy.random.default_rng(b)`, b = 0 to 998.
  The sign is shared by every configuration and instrument in that hour, so
  their dependence is kept. A flipped trade's P&L, funding included, is the
  stored walk of the opposite side.
- **Null B: day re-pairing over every shift** s with 7 ≤ s ≤ 143, on the
  circular calendar order of D's 150 test days (n = 150, 137 shifts), shared
  by every instrument, so that one shift pairs the same calendar days on
  every instrument. The source day's accepted entries (time of day and side,
  after its own guard) are executed on the target day's walk tables; an entry
  that falls while the mapped previous position is still open is skipped and
  counted. A pair whose source or target day is excluded for an instrument
  contributes nothing for that instrument, and is counted. **The parity
  test** runs at stage 0, on the 2024 days already on disk (BTCUSDT
  2024-02-11 and 2024-03-05), before any 2025 fetch: at s = 0 the walk-table
  route equals the executor entry by entry, and ordinary entries scored from
  walk tables equal the executor's. A failure blocks stage B until it is fixed
  by a dated amendment; Null B has no fallback, and the dry run repeats the
  test as a check.
- **The bar.** p_A = (1 + #{b : max over strata of A^b ≥ A_j}) / (1 + 999),
  and p_B = (1 + #{s : max over strata of A^s ≥ A_j}) / (1 + 137). A searched
  slot passes (iv) when **max(p_A, p_B) ≤ 0.05 / R_D**, R_D being the number
  of complete D runs (1 unless a correction re-ran D;
  [selection](#selection-on-d-admission-and-the-finalists)). The rule is the
  count itself: the entry point computes the counts directly, and a unit test
  checks the boundary at 137 shifts (at R_D = 1, at most 5 shifts at or above
  A_j pass and 6 fail). `verdicts.exceeds_placebos`, whose interpolated
  percentile admits more, is not used for (iv).
- **The refit replication, mandatory, once.** Every fold's training labels are
  circularly shifted by 28 days within its training span, every boosted and
  logistic fit is redone, and stratum C's whole selection is re-run, giving
  the refit A of stratum C's chosen cell, A_refit. It adds a condition for C\*
  only: C\* passes (iv) when the bar above holds **and** A_C\* > A_refit,
  strictly. It compares stratum C's A with stratum C's own refit, because only
  stratum C is refitted; the maximum over strata already enters through p_A
  and p_B. The network and the bound regressor are not refitted, for cost.
- Also: a global sign flip, reported as a diagnostic only (it is an
  anti-signal when information exists).

Outside the null: S-P and S-M, which make no choice on a passive outcome, and
S25, which is fixed. S25's D values are shown against the same two nulls
applied to its fixed cell, with no selection; every slot's own placebos run on
the held-out blocks. The amendment records every replication's A per stratum,
the seeds, the shifts, the quantiles and the p-values.

**What the breadth costs.** A synthetic null with no market data, under an
assumed correlation along each axis, puts the 95th percentile of A at about
2.2 for a 126-cell catalogue on BTCUSDT, 2.6 for this catalogue on BTCUSDT
alone and 2.85 for this catalogue on the three instruments; the real nulls
decide. A true edge therefore needs a neighbourhood-median t of G of about
2.85 over 150 days to be admitted. Breadth costs admission power on D; it
never costs the validity of H, which rests only on H being untouched and m
being fixed.

---

## Metrics

- **`net_bp_trade`**: for one round trip, (exit value − entry value) over
  the entry notional, × 1e4, less fees in bp of the entry notional, plus
  funding. **Primary for S25, S-C and S-R**, and the measure the choice on D
  ranks by. A day's value is the day's net in USDT
  summed over its round trips, over the sum of their entry notionals, × 1e4:
  basis points of traded notional (every trade is one clip, so the weights
  differ only with the price at entry). The daily series has one value per day with at least
  one trade.
- **`net_per_100_clip_day`**: **primary for S-P and S-M.** Its daily series has one
  row for every usable calendar day of the block: the sum of that day's net
  in USDT × 100 / the clip's notional at each entry, and 0 for a day without
  a fill. The zero rows are written into the table given to `assess`, which
  averages only the rows it is given (`evaluation/significance.py`).
- **`gross_bp_trade`**: `net_bp_trade` before fees and funding. **Primary for
  S25-Z**, with the day's value formed as for S25 (gross USDT over entry
  notional), beside its net at 0/0.
- **G** (on D only): for each trade, net on its side minus the mean of the
  long and the short walked P&L over the same entry and hold; a day's value
  is the mean over its trades. It is the edge with the costs taken out, and
  it is centred at zero under Null A of [the placebo
  search](#the-placebo-search), whose sign flips make the side independent of
  the move. It keeps the drift times the imbalance of longs and shorts, so it
  is not centred under Null B, which applies a source day's sides to a target
  day's drift; drift is controlled by Null B and by K-drift on held-out
  blocks.
- **The daily series decides.** The mean, the day-level t, the Newey–West t,
  the median day, K1, K-days, the placebos, K-nbhd, σ_D and the minimum
  detectable effect are all computed on the slot's daily series as
  defined above, never on trade rows. `assess` is given one row per day
  (`cluster="day"`, `series=None`); a series with fewer than two days is not
  given to it (`assess` raises on one cluster), fails the bar's day count and
  enters Holm with p = 1. The degrees of freedom are the number of
  days in the series minus one, per slot and per block.
- **`gross_bp_turnover`**: gross over turnover (both sides' notional), × 1e4:
  the fee break-even per side. Reported for every slot against the base
  fee and the cheapest published fee for the side used.
- **IC**: per day, the rank correlation between the signal's score
  (`p_long − p_short`, or μ) and the realised forward mid move over `h`,
  measured from the mid at the entry snapshot (the first at or after the
  decision + 10 ms) to the mid at the first snapshot at or after that + `h`,
  over all decision rows; its mean over days.
- For passive fills: fill ratio, markouts at 1, 5, 10 and 30 s by fill path
  (queue or trade-through), and the decomposition `spread + adverse(5 s)` (the
  part of the result earned at the fill, the "making" part of round one) and
  the rest (the move held after it).
- Beside every verdict: trades a day, share of days positive, median day,
  maximum drawdown, net in USDT a day at the clip used, funding, the result
  at every published tier, the result split by side, the counts of skipped
  entries, `entry_late_snapshot`, `exit_late_snapshot` and
  `flatten_beyond_visible_book`, and, for a slot with a regime window above 0,
  its [regime attribution](#the-regime-break-policy-axis).
- **Statistics**: `evaluation/significance.assess` on the daily series; the
  day level decides. Score every calendar day of the block, with no row
  filtered or reweighted by label, outcome or model state. The one-sided
  day-level p-value is `market_making/verdicts.day_t(daily series).p_one_sided`
  (Student t at days − 1, through `verdicts.t_sf`), Holm is `verdicts.holm`
  and the placebo comparison `verdicts.exceeds_placebos`. Beside every
  verdict, the **Newey–West t** of the daily mean (Bartlett weights, lag 5
  days; new code in `evaluation/significance.py`), because consecutive days in
  one regime are not independent.
- **Minimum detectable effect**, recorded in the amendment for each slot
  and for S25-Z: `(3 + 0.84) × σ_D / √n_eff`, the mean that the t ≥ 3 bar would
  catch with 80% power, with `σ_D` the standard deviation of the daily series
  over D's 150 walk-forward test days at the chosen cell and `n_eff` = 44 × the
  share of those 150 days with a trade (44 for S-P and S-M, whose series has
  every usable day; 24 in place of 44 for a slot on S25's signal, whose status
  window on H is 24 days). For a searched slot, σ_D is also given at the
  carry-forward estimate, with the power of the t ≥ 3 bar against it. On F (F− or all of F) and on P, the "below the minimum
  detectable effect" label uses the same formula, with σ_D and the share as
  frozen and the usable days of the series the status is assessed on in place
  of 44.

---

## Status rules

**The family** is the five finalists, **{S25, S-C, S-R, S-P, S-M}, with m = 5
on every block**: H, F (F− or all of F), P, and each of R's split days.

**The bar.** A slot passes on a block when no kill condition fires and all
of these hold on its daily series ([Metrics](#metrics)):

- its one-sided day-level p-value passes Holm's procedure at 5% with
  **m = 5**. A slot enters with p = 1 when it is killed, void or inconclusive
  under K-trades; `not run`, `not admitted on D` or `same as S25`; not taken
  to the block; or scored on P only by its mean (R supported). So m never
  depends on an outcome, and a result on too few trades never moves another
  slot to a laxer Holm step. At m = 5, Holm's first step needs t ≥ 2.42 at 43
  degrees of freedom, 2.54 at 19 and 2.82 at 9, so the t ≥ 3 bar below binds
  on every block, and the breadth of the search costs nothing on H;
- the day-level t is at least 3, its degrees of freedom being the days in the
  series minus one;
- the Newey–West t is at least 2;
- the median day is positive;
- the series has at least 20 days with a trade on H, and at least 10 on F
  (or F−) and on P (S-P and S-M: usable days).

**No deflation on H for D's trials.** H is untouched by the search, and E
separates selection from confirmation. The finalists, their neighbourhoods,
θ, checkpoints, weights, the fitted-values file and the regime windows are
frozen by sha256 before H's gate opens. D's trial count acts only on D,
through the placebo search, the deflated Sharpe ratio and the probability of
backtest overfitting.

**Precedence**, the same on every block, so that each result has exactly one
status:

1. **void** — K-leak, K-flag-leak (for a slot with a regime window above 0),
   K-units, K-acct or K-cap fires; counted as killed.
2. **inconclusive (too few trades)** — K-trades: fewer than 100 trades over
   the block (round trips for the taker slots; entries with any fill, that is
   round trips, for S-P and S25-Z; returns to a flat position for S-M). Nothing below
   is evaluated.
3. **killed** — any kill condition fires. Rejection needs no significance.
4. **candidate** (on H) or **passes** (on F and P) — the bar is met.
5. **inconclusive** — otherwise; labelled "below the minimum detectable
   effect" when the mean is smaller in size than the MDE (only when no kill
   fired). Never reported as a positive.

**The statuses set on D**, each entering Holm with p = 1 on every block:
`not admitted on D (<condition (i) to (iv)>)`, `not admitted on D (hand
run)`, `not run (no admissible cell on D)` and `same as S25`.

**Slots on S25's signal**: S25, S25-Z, and S-P and S-M when their signal is
S25's ([the information set of
S25](#what-was-known-before-this-was-written)). On H their status, kills,
placebos, bar and K-trades are computed on H's usable days from 2025-07-08 to
07-31 only (24 days; the bar's 20 days with a trade, or usable days for S-P
and S-M, still apply), and `candidate` is written **candidate (H before
2025-08-01)**. That is their best and final status: they are not taken to F, R
is not tested for them, and on F and P they enter Holm with p = 1. Their
numbers on H from 2025-08-01 to 08-20 and on all of H, on F (one whole block)
and on P (only when P is opened for another slot) are computed in the same
read sessions, kills and placebos included, and reported as measurements
labelled "inside S25's information set".

**F.** With `b` the first alarm of the frozen detector in F, F− is F's usable
days before the alarm's day and F+ its usable days after it; the alarm's own
day is in neither. **Statuses on F, kills included, are computed on F−** when
there is an alarm in F, F− has at least 20 usable days, R was not declared
untestable on D, and R is not void (below). Otherwise they are computed on
all of F, labelled "assessed on all of F", and F is not split. F+ enters only
R. Taken to F, except the slots on S25's signal: every
candidate, and every slot inconclusive on H at precedence step 5 (with
or without the label "below the minimum detectable effect") with a positive
mean (for S25-Z, both claims' means positive). **A result inconclusive under
K-trades (step 2) is not taken to F, whatever its sign**: its mean is not
evaluated, so it cannot route it.

| On H | On F (F− or all of F) | R (below) | On P | Status |
|---|---|---|---|---|
| candidate | passes | not supported | passes | **confirmed** |
| candidate | passes | not supported | does not pass, no kill | **candidate, not confirmed** |
| candidate | passes | supported | mean ≤ 0 | **confirmed until the break** |
| candidate | passes | supported | mean > 0 | **candidate; the break did not last** |
| candidate | does not pass, no kill | reported, not used | not read for it | **candidate, not confirmed** |
| inconclusive (step 5), positive | passes | not tested | passes | **confirmed on P, first significant on F** |
| inconclusive (step 5), positive | passes | not tested | does not pass, no kill | **exploratory: significant on F, not confirmed** |
| inconclusive (step 5), positive | does not pass, no kill | not tested | not read for it | **inconclusive** |
| either | a kill fires | — | not read for it | **killed on F** |
| either | passes | not supported or not tested | a kill fires | **killed on P** |

"Passes" on F and P is the bar above, with m = 5. "R untestable" and "R not
tested" are read as "not supported". For a slot with R supported, P is
scored only as mean ≤ 0 (the break persisted) or > 0 (it did not last):
neither K-trades nor any kill condition is applied to it on P, but the void
conditions are (a void result on P is `void`), and a P series without a day
with a trade has mean 0, as a neighbour cell without a trade scores 0 in
K-nbhd, so it reads as the break persisted. A slot that is `candidate`
after H keeps that status whatever F and P show unless a kill fires there; the
later reads add to it. **A slot first significant on F gets no second
route to candidacy**: it is "exploratory: significant on F" until P, which is
its only confirmatory test; R is not tested for it, and the README never calls
it a candidate.

P is read once, after F's results are committed, only for slots that
passed on F. F is read once, after H's results are committed, whatever H's
outcome: its detector alarms and every frozen slot's and measured row's
numbers on F are reported as measurements even when nothing is taken there.
**For a slot not taken to F, F is reported as one whole block, never split at `b`.** Every
comparison from H into F crosses the archive change and is labelled so.

**An unreadable block** (more than a quarter of its days excluded,
[Blocks](#blocks), judged on the slot's instrument) changes no status. A slot
whose instrument's H is unreadable is recorded as `not assessed (H
unreadable)` and is not taken to F; F is still read and reported as
measurements, and H, F and P are spent all the same. If F is unreadable for a
slot, its status after H stands, marked "F unreadable", and P is not read for
it. If P is unreadable, each slot read on P keeps its status after F, marked "P unreadable": a candidate that
passed on F is **candidate, not confirmed (P unreadable)**, whatever R showed,
and one first significant on F is **exploratory: significant on F, not
confirmed (P unreadable)**.

**R void.** If a later run of the detector does not reproduce an earlier
run's alarms ([The break detector](#the-break-detector)), R is `void` and read
as not supported. Found when H is read (against D's alarms in the
amendment), R is void, and F is assessed on all of F, labelled. Found when F
is read, `b` does not split F and statuses on
F are computed on all of F, labelled. Found when P is read, F's statuses stand
as committed, and P is scored with the full bar, R read as not supported.

**No number at a fee tier other than the base, under a non-default fill rule
or exit, on F for a slot not taken to F, for a measured row, for a
regime-map segment, on H before 2025-08-01 (except as the status window of a
slot on S25's signal), with the gap-excluded days included, or inside S25's
information set may be described as the strategy working.** Such numbers are
measurements.

**S25-Z** is outside the family. It is tested alone at α/6 = 0.83%
(α/(m + 1)), one-sided,
at the larger of its two claims' p-values (round two's rule for a hypothesis
with two claims), with the t, Newey–West t, median-day and day-count
conditions holding for both claims; it follows the same precedence as a slot
on S25's signal, on H before 2025-08-01, and is not taken to F. Its best
possible status is **candidate (H before 2025-08-01), conditional on a
zero-fee venue**: the condition (0 bp on both sides) is stated as a number,
and the base-tier result, measured on the same trades, is reported beside it.

### Kill conditions per slot

| Slot | Execution | Primary series | Kills | Reported beside |
|---|---|---|---|---|
| S25, S-C, S-R | taker | `net_bp_trade` | K1, K-label, K-placebo, K-drift, K-nbhd (the full ordered neighbourhood, regime window included), K-days, K-fee, K-latency | the control model's IC (S-R on the network); regime attribution |
| S-P | signal-entry quoter | `net_per_100_clip_day` | the taker slots' list, the stale signal in K-placebo, and K-queue | the fill decomposition; regime attribution |
| S-M | S1 with the lean | `net_per_100_clip_day` | S-P's list, and **K-dir as a kill** | the λ neighbourhood; pull against widen |
| S25-Z | signal-entry quoter at 0/0 | `gross_bp_trade`, and net at 0/0 | K1 and K2, K-label, K-placebo with the stale signal, K-drift, K-nbhd, K-days, K-queue, K-latency; no K-fee | net at the base tier on the same trades |

Void conditions; the slot is recorded as `void`:

- **K-leak** — any ported feature fails the leakage checker
  (`validation/leakage.py`, data mutated after a cutoff changes a value before
  it), or the fold-level leakage test fails. Every result of the study is
  void.
- **K-flag-leak** — the flag-causality test fails
  ([the regime-break policy axis](#the-regime-break-policy-axis)). Every slot
  with a regime window above 0 is void; found before D, the fix is a dated
  amendment, as for K-leak.
- **K-units** — an input to the EV gate, the market gate or a cost (a fee, the
  walk of the book, funding) is not in basis points computed from prices (a
  normalised column reached it). The result of every slot whose gate or cost
  input it reaches is void, S-P's and S-M's also when it reaches their
  signal's gate. S25-Z is not exempt: its verdict is at 0/0, but its gross
  comes from the same walks and its net at 0/0 from the same funding.
- **K-acct** — the accounting identity (R15) fails on any day, or turnover
  does not equal the sum of fill notionals.
- **K-cap** — |position| exceeds one clip at any event.

Kill conditions:

- **K1** — the mean of the slot's daily series on the block is ≤ 0 (for
  S25-Z, K1 on gross and K2 on net at 0/0).
- **K-label** — the signal's IC (from the entry snapshot's mid, as defined
  under [Metrics](#metrics)) is ≤ 0 on the block (the model has no
  information about the move it trades).
- **K-placebo** — the true mean does not exceed: the 95th percentile of the
  200 random-direction draws; the sign-flipped signal's mean; the one-day
  shifted signal's mean; and, for the passive slots and S25-Z, the stale
  signal's mean (below).
- **K-drift** — the true mean does not exceed the larger of the all-long and
  the all-short versions' means at the same entry times (the result is the
  block's drift, not the signal).
- **K-nbhd** — the mean of the daily series on the block, recomputed with
  each cell of the chosen cell's neighbourhood in place of the chosen cell,
  has a median ≤ 0 over the cells. The neighbourhood is the one of
  [selection](#selection-on-d-admission-and-the-finalists), every cell
  counted whatever its trade rate on D; a neighbour cell with no trade on the
  block scores 0. **S25's** covers 24 cells: hold {20, 50, 120} s × rate {100,
  240} × label fee {0, 5.5} × regime {none, pull 4 h}. For S-P it is its
  signal's hold × rate × label fee (or hold × k × β), each cell entered
  passively; for S-M, λ ∈ {0.5, 1, 2}; for S25-Z, S25's hold × rate × label
  fee (12 cells). It runs only the frozen models, thresholds and checkpoints
  whose sha256 the amendment records.
- **K-days** — half or fewer of the days in the daily series are positive
  (days with a trade for the taker slots and S25-Z; every usable day for S-P
  and S-M).
- **K-latency** — a positive result that is ≤ 0 when every order and cancel is
  live 110 ms after the decision instead of 10 ms (one more 100 ms row). A unit
  test checks that every fill's snapshot is stamped at or after its decision
  + 10 ms, and that the label's walks and the executor's use the same
  function.
- **K-fee** — the fee break-even per side (`gross_bp_turnover` for a taker
  slot; for S-P and S-M the maker fee at which net is zero with the taker exit
  at the base fee) is at or below the base fee for that side. For a positive
  net at the base tier this can only fire through an accounting fault, so it
  is also a check; its main use is the report of how far each slot was from
  paying.
- **K-queue** (S-P, S-M, S25-Z) — a positive result that is not positive under
  the joint pessimistic queue bracket (pessimistic cancellation attribution
  and `all_ahead` arrival growth), or under either `none` or `all_ahead`
  alone.
- **K-dir** (S-M only) — the making part, `spread + adverse(5 s)` over the
  passive fills, is not positive while the net is (`verdicts.k_dir`): a
  quoter's claim is the part earned at the fill, and a positive net without it
  is inventory carried through a move.

Not kills, but they set the status:

- **K-trades** — fewer than 100 round trips over the block (entries with any
  fill for the quoter, not the simulator's fill rows, which count partial
  fills and taker exits; returns to a flat position for the market maker)
  makes the result inconclusive whatever its sign; it
  is applied before any kill (the precedence above).
- **K-MDE** — a result that does not pass, with no kill, and is smaller in
  size than the MDE is "inconclusive, below the minimum detectable effect".

Reported, not a kill: **the fill decomposition** for S-P and S25-Z,
`spread + adverse(5 s)` over the passive fills and the rest. A directional
signal entered passively is meant to earn the move held after the fill, so a
non-positive part at the fill with a positive result is the mechanism under
test, not a fault.

Each condition is applied to every claim of a slot. A condition that needs a
positive result (K-queue, K-latency, K-dir) fires only when the result is
positive under the default rules. On F the conditions are applied to the
series the status is assessed on (F− or all of F).

### Placebos

Each is re-run through the same execution, fee, guard and accounting as the
true slot, on the same block. For S-M each placebo replaces its signal tape.

| Placebo | What changes | Used by | Seeds |
|---|---|---|---|
| sign-flipped | the signal's direction reversed at the same entry times | every slot, S25-Z | — |
| random direction | the same entry times, each direction drawn at random | every slot, S25-Z | 200 draws, 0–199 |
| one-day shift | the signal tape shifted by one day at the same time of day, circularly within the block | every slot, S25-Z | — |
| stale | the signal delayed by one hold | S-P, S-M, S25-Z | — |
| all-long, all-short | the same entry times, one direction (K-drift) | every slot, S25-Z | — |
| control model | the network trained on the control label; its IC against its own label and against the realised move, reported beside K-label | S-R on the network | the training seed |
| other break dates | `b` replaced by every admissible split day of F other than `b` (at least 20 usable days before it, at least 10 after), enumerated in full; the reference set keeps those at which the slot also passes on the days before them | R | none; enumerated |

The taker slots' placebos come from the walk tables, and their K-latency from
walk tables at 110 ms. **Placebo positions.** Each placebo runs on the true
slot's entry decision times with its own one-position state. A decision time
at which the placebo is still in a position, or has an entry resting, is
skipped and counted; the skip counts are reported beside K-placebo and
K-drift. For a taker slot nothing is skipped, since a taker exit's time does
not depend on the direction; for the quoter a placebo order fills at other
times than the true one, or not at all, so it may still be busy at the next
one. **The one-day shift** gives each usable day the signal tape of the
previous usable day, circularly over the block's usable days, so it steps
over excluded days.

---

## S25, the named 2025 configuration

**S25 is one point of the catalogue**, fixed now:

| Coordinate | Value |
|---|---|
| Instrument | BTCUSDT |
| Stratum | C |
| Model | `gbm`, the repository's defaults |
| Inputs | micro-selected |
| Label | the walked-book taker label at a 0 bp label fee |
| Hold | 50 s, closed by time |
| Decision | the hold-first rule at 240 trades a day |
| Gate | `open` |
| Regime policy | none |
| Execution | taker in and taker out, through the event-time taker executor |
| Clip | 0.010 BTC |
| Verdict | `net_bp_trade` at the base tier (5.5 bp taker), funding included |

**How it is treated.**

- It is reachable: it is a cell of the catalogue, and the leaderboard shows it
  with a badge and its rank "r of N".
- It is fixed before D: no admission and no search-level null. It goes to H
  whatever D shows; its D values are shown against its own Null A and Null B
  distributions.
- Its status is taken on H's days before 2025-08-01 only and stops at
  **candidate (H before 2025-08-01)**; the rest of H, F and P are inside its
  [information set](#what-was-known-before-this-was-written) and give
  measurements ([status rules](#status-rules)).
- It is a member of the family and carries the taker slots' kill list. Its
  K-nbhd covers 24 cells (hold {20, 50, 120} s × rate {100, 240} × label fee
  {0, 5.5} × regime {none, pull 4 h}), with one frozen fit per hold and label
  fee (6 fits).
- It is the fallback signal of S-P and S-M, so passive entry and the market
  maker are always measured on at least one fixed signal.

**Statement.** On H's days before 2025-08-01, at the base tier, S25 earns a
positive `net_bp_trade`.
**Prediction:** killed by K1.

### S25-Z, the gross edge before fees (outside the family)

**Statement.** On H's days before 2025-08-01, under the queue fill rule,
S25's signal entered by the
signal-entry quoter (one clip posted at the touch, a taker exit `h` after the
fill) has a positive `gross_bp_trade`, and its net with fees at zero on both
sides (gross plus funding) is positive too. It asks whether the signal pays
before fees, which is what a venue charging nothing would earn. Its
neighbourhood is S25's hold × rate × label fee, 12 cells. Tested alone at α/6
([status rules](#status-rules)); its best status is "candidate (H before
2025-08-01), conditional on a zero-fee venue", and it is never reported without the same trades' net at the base
tier and at the cheapest published tier beside it, nor as a result a Bybit
account could trade. **Prediction:** killed by K1: resting entries are
adversely selected, and the repository's maker finding is that this inverts a
crossing signal's gross edge.

### Leaderboard regions

Two regions of the catalogue are reported as measurements, with no slot: **T1**
= C × `gbm` × micro-selected × label fee 5.5 × gate `open` × regime none, and
**T2** = R × `tcn` × gate `open` × regime none. Each region's best
neighbourhood cell is reported with its S, A and rank. Stratum R has no named
point.

### The edge ladder

One table, each step labelled with the one thing that changed:

1. A0 on O at 0/0, crossing and under the three fill rules;
2. S25's gross and net on D's walk-forward;
3. S25 on H before 2025-08-01, then on all of H and on F, labelled inside its
   information set;
4. S25's fee break-even per side against every published tier;
5. S25-Z's net at 0/0, with FILL's fill ratios beside it.

---

## The searched finalists

- **S-C** — on H at the base tier, the cell C\*, chosen and admitted on D
  ([selection](#selection-on-d-admission-and-the-finalists)), earns a positive
  `net_bp_trade`, taker in and out. **Prediction:** `not admitted on D (i)`.
- **S-R** — the same for R\*: the network or the bound regressor with the EV
  gate `μ − kσ > 11 bp + walk cost + β`, one entry per event, one clip a trade.
  **Prediction:** `not admitted on D (i)`. Reported beside, when R\* is the
  network: the control model's IC against its own label and against the
  realised move, which shows on public data how much of a smoothed label is
  known at decision time.
- **S-P** — on H at the base tier, its signal (the admitted C\* or R\*, else
  S25's), one clip posted at the touch on the signal's side and kept at its
  first price, filled from prints after the queue ahead under rules R5–R7, a
  taker exit `h` after the fill, earns a positive `net_per_100_clip_day`, on
  the zero-filled series of every usable day. Reported beside: the fill
  decomposition, the fill ratio, markouts by fill path, the passive-exit
  variant, and the result at maker fees of 0 and −1 bp (the cheapest
  published maker tier and the programme's rebate), each with the fee caveat.
  **Prediction:** `not admitted on D (i)`; if it reaches H, killed by K1. On
  S25's signal its status is taken on H before 2025-08-01 only, as for S25.
- **S-M** — on H at the base tier, S1 with the lean on S-P's signal earns a
  positive `net_per_100_clip_day` with a positive making part. Reported
  beside: the λ neighbourhood, and `guard_pull` against `guard_widen` at 4, 24
  and 120 h with λ = 1. **Prediction:** `not admitted on D (i)`; if it reaches
  H, killed by K1 or K-dir.

## R, the break (conditional)

**Tested for** every slot that is a candidate on H, whether or not it passes
on F−, except the slots on S25's signal, which are never taken to F
([status rules](#status-rules)); its statistic is computed and reported in
either case, with F−'s result beside it. Not tested for any other
slot, including one first significant on F; recorded as `not tested`,
with the reason, and read as not supported.

**Statement.** With `b` the first alarm of the frozen detector in F, and
Δ(`d`) the mean of the daily series on F's usable days before split day `d`
minus its mean on F's usable days after `d`: R is **supported** if the
slot passes on F−, its mean on F+ is ≤ 0, and p_R ≤ 0.05, with
p_R = (1 + #{`d` ≠ `b` in the **reference set** : Δ(`d`) ≥ Δ(`b`)}) / (1 +
the number of days of the reference set other than `b`); **not supported**
otherwise. `b` is never counted in its own reference set, so p_R ≤ 0.05
needs at least 19 other split days. S25-Z, like every slot on S25's signal,
is never tested for R ([status rules](#status-rules)), and nothing about it
is evaluated at the split days.

**The reference set** is every admissible split day `d` ≠ `b` of F (at least
20 usable days before it and at least 10 after; about 58 days), enumerated in
full, at which the slot also passes the bar on F's days before `d`.
**The whole bar is evaluated at each `d`**, kills and Holm included, as on F−
with F's usable days before `d` in its place, with m = 5: the other slots
taken to F are evaluated there the same way and enter Holm with p = 1 where they are
killed, void or inconclusive under K-trades on those days, as do those not
taken to F, and K-trades counts the trades before `d`. Nothing is re-run for
it: every series the bar reads (the slot's daily values and IC, its 200
random-direction draws and other placebos, its neighbour cells, its 110 ms and
bracket re-runs) is computed once per day over all of F in F's read session,
the one-day shift circular within all of F as on F−, and each split day only
slices those per-day values.
Passing on F− selects lucky days before the true split, so the placebo splits
go through the same selection; a comparison with unselected splits would make
R come out supported too often. When the slot does not pass on F−,
Δ(`b`) is reported with its percentile among all admissible days, as a
measurement.

**Untestable**, declared and not retuned, and read as not supported: no
alarm in F; fewer than 20 usable days before `b` or fewer than 10 after it;
the detector's alarm rate on D above one per 14 days (declared in the
amendment, before H is read; `b` then does not split F); or fewer than 19
days in the reference set besides `b`.

**Reported beside**: every alarm in H and F with the channel that moved
(scale or dependence, and the mean channel's odds), the second detector's
alarms, and the earlier
daily-bar run's alarms (item 5 of [what was known](#what-was-known-before-this-was-written)),
labelled as already public. The blocks are not moved by any alarm, including
alarms inside H.

**The regime axis does not replace R.** A slot with a regime window above 0
enters R with its own guarded series; the axis is a short guard after any
alarm, R a stop at the first alarm in F.

**What R can establish** is that a policy fixed in advance, "stop at the first
alarm", separates days on which a strategy paid from days on which it did not;
not why it stopped.

---

## Measurements without a hypothesis

None enters a verdict, and none can be quoted as a result.

**A0, S25 under a short-window protocol, on O.** S25's signal (gradient
boosting on BTCUSDT on the walked-book label at 0 bp, micro-selected inputs),
trained on 2025-01-01 to 01-27, thresholds for 240 trades a day set on 01-28
to 02-03, a 50 s hold, run on O (2025-02-04 to 02-26): crossing at 0/0, and
posted at the touch under each of the three fill rules of FILL. Reported:
`gross_bp_turnover`, net at the base tier and at every published tier, trades
a day and IC. A0 differs from S25 only in protocol (thresholds from O's
preceding week, 0/0 fees, scored on O); it is the first step of S25's
[edge ladder](#s25-the-named-2025-configuration). It carries no comparison
figure and no prediction.

**FILL, the fill rule, with a prediction.** A0's passive order stream on O,
the same orders under three rules:

| Rule | A resting order fills | Bias |
|---|---|---|
| (a) on touch | in full, at the first print at or through its price from the side that hits it, whatever is queued ahead | optimistic bound, labelled so |
| (b) trade-through | only on prints strictly through its price, up to their size (R7 alone) | pessimistic |
| (c) queue | after the visible queue ahead is consumed by prints at its price (R5, R6), or on prints through it (R7) | the simulator's default |

Reported for each: fill ratio, markouts at 1, 5, 10 and 30 s by path, gross and
net. **The fill ratio** is the share of posted orders that receive any fill
(the convention of `fill_rate` in `backtest/maker.py`); filled volume over
posted volume is reported beside it. The prediction below is on the share of
orders.
**Prediction, fixed now: the fill ratio is at least 0.75 under (a) and at most
0.50 under (c).** The same three rules are reported for S25-Z's and S-P's
streams on H.

**The fee break-even**, per slot and block: `gross_bp_turnover` for the
taker slots; for S-P, S-M and S25-Z, the maker fee at which net is zero with
the taker exit at the base fee and, separately, at 1.5 bp. Compared with
every published tier, with the fee caveat.

**Measured rows**: the chosen cell of a non-admitted C\* or R\*, and a
non-admitted S-P or S-M, run once on H and F with no placebo and no status,
and on P only when P is opened for a qualifying slot, in that read session
([selection](#selection-on-d-admission-and-the-finalists)).

**The D leaderboard, marginals and nulls**: every configuration with its D
numbers, ranks and region tags, the per-axis marginals, and every replication
of the placebo search ([artefacts](#artefacts-and-headline-sentences)). Every
D number is labelled "selected, biased upward" and is never quoted as a
result, except p_search ([the placebo search](#the-placebo-search)).

**H before 2025-08-01**, a labelled sensitivity: every slot not on S25's
signal at its frozen cell on H's usable days from 2025-07-08 to 07-31 (24
days), the part of H that the [information set of
S25](#what-was-known-before-this-was-written) does not cover. Same daily
series and statistics, no status. For a slot on S25's signal these 24 days
are its status window, and its numbers on all of H are the measurement.

**Verdicts with the gap-excluded days included**: each verdict's mean, t and
median day recomputed with the days excluded for sequence gaps put back.

### The regime map

A measurement for every slot and every measured row, never a status.

- On D, each slot's walk-forward daily net is cut at each test month's
  fold-wise alarms ([the regime-break policy
  axis](#the-regime-break-policy-axis)); on H, F and P it is cut at the
  frozen detector's alarms (on P only when P is opened for a qualifying slot,
  in that read session; otherwise the map ends at F's last day).
- Each segment is reported with its dates, usable days, mean, day-level t and
  cumulative net; a segment under 10 usable days is drawn but carries no t.
- One chart per execution path (taker, passive entry, market maker) shows the
  cumulative daily net through 2025, with the alarms marked and the blocks
  shaded.
- The detector cuts the segments before any result is computed for them.
  Every segment is shown and none is selected; no segment enters any rule. A
  positive stretch for a taker or a maker appears here as what it is: a
  description between two alarms, read against the statuses beside it.

---

## The break detector

Fixed now, run as the ledger opens each block.

- **Input**: hourly log returns, close to close, of Binance USDT-M futures
  BTCUSDT 1h klines, each divided by the root mean square of D's hourly
  returns at the same UTC hour (a 24-value profile fitted on D only), so that
  the daily cycle of volatility is not read as a series of breaks. Inside the
  search on D the profile is fitted fold by fold, on the hours before each
  test month only ([the regime-break policy
  axis](#the-regime-break-policy-axis)); the frozen profile fitted on all of D
  gives the flags on H, F and P and in the dry run. Both flag lists are
  recorded, and their difference on D is reported, not corrected.
- **One continuous stream from 2024-12-01**, so the monitor is warm by
  2025-01-01. Bars are fetched from that day, so the stream's first return is
  its second hour (`log_returns`, close to close); no 2024-11 bar is needed. E's hourly bars are in it: they are fetched and read when the
  ledger opens H (bars only; E's book, prints and funding are never fetched),
  so no return spans the embargo. **A missing hour** has its close carried
  forward and a zero return, and is counted and reported; the walk continues
  without a reset, and the move over the gap enters the next return.
- **Bars for E, H, F and P come from Binance's daily archives only**, one day
  at a time, inside the opened access of their block. No monthly kline
  archive for a month after 2025-06 is fetched or cached before P's read (the
  existing code prefers, and always caches, monthly kline archives, which
  would put a later block's bars on disk early). The study's loader refuses a
  monthly kline file whose month reaches past the highest opened block.
- **Monitor**: `validation/structural_breaks.py` on the `structural-break`
  library at commit `0636813fdd4ccc9f25490d97f62d6fccecb01881` (tag `v0.1.0`,
  as locked in `uv.lock`; a moved tag does not move it),
  `MonitorSpec(history_len=365, online_len=90, min_history=121,
  statistics=("scale", "dependence"), minimum_gap=0)`, in hours, with the
  recorded thresholds 9.09 and 7.93 (`RECORDED_THRESHOLDS[(365, 90)]`). One
  `detect_breaks` call runs over the whole stream from 2024-12-01 to the last
  hour of the highest opened block, its windows anchored at the stream's
  start. The walk is causal, so each later run must reproduce every alarm of
  the earlier runs, D's as the amendment records them (timestamp and
  channel, from the run with the frozen profile over 2024-12-01 to
  2025-06-30; the fold-wise runs use other profiles, are recorded separately
  and never enter this check) and each block's as its results file records
  them; if it does not, R is void and read as not supported (see
  "R void" under the status rules).
- **The mean channel** is read from the `odds_mean` column of the same run,
  against 8.93, and reported. It is not in `statistics`, so it never
  re-anchors the walk and never defines `b`.
- **`b`**: the first alarm effective in F, effective at the close of the hour
  whose return produced it. F− is F's usable days before the alarm's day, F+
  its usable days after it; the alarm's own day is in neither.
- **On D**, the monitor's alarm rate over 2025-01-01 to 06-30, from the
  frozen-profile run, is recorded in the amendment. Above one alarm per 14 days, R is declared untestable then,
  before H is read, `b` does not split F, and the monitor is not retuned.
  This rate is optimistic: the hour-of-day profile is fitted on D itself, and
  the recorded thresholds were calibrated on Gaussian null paths
  (`structural_breaks.py`), not on heavy-tailed hourly returns. On F, where
  the profile is out of sample, the first alarm is likely to come sooner, and
  after a re-anchor the monitor can alarm again once `min_history` (121 hours)
  has passed. An early first alarm leaves F− short or R untestable: it lowers
  R's power, and the placebo splits keep it from making R easier to support.
- **Reading**: bars up to 2025-06-30 are development data; E's and H's bars
  open with H, F's with F and P's with P, through the ledger.
- **Reported beside, never defining `b`**: `validation/changepoint.detect` on
  the 1 s grid of BTCUSDT's mid and spread from the Bybit book (`to_grid`,
  `label="right"`), at its defaults (window 2000, threshold 20, reference 40,
  minimum gap 20,000, its four statistics). It runs continuously over D,
  **restarts on H's first day**, because E's book is never fetched, runs
  continuously over H, **restarts on F's first day**, 2025-08-21, so that no
  `ob500` state is carried into `ob200` data and F's read needs nothing of H,
  and runs continuously from F through P; any alarm within a day of
  2025-08-21 is labelled "at the archive change". Also the earlier daily-bar run,
  labelled as public before this registration.

---

## Fetch plan

Nothing below has been fetched. Sizes are the archives' own, summed from the
`HEAD` requests. Every archive goes through this study's fetch (new code, on
the pattern of `data/ensure.py`: only missing days are fetched, and no
archive is kept after its replay).

**Replay workers.** `data/ensure.ensure_book` replays one day at a time
(`workers=1`), which would take W and D about 7–14 hours. Running
`data/bybit.download_range` with more workers runs out of memory instead:
`reconstruct` holds a day's rows as Python dictionaries until the day ends, at
most 864,000 rows (one per 100 ms) of about 5.5 KB each, so 5–6 GB per worker
at peak. The fetch therefore replays books with **at most 4 worker
processes**, fewer if four times the peak resident memory of one replayed day
exceeds 24 GB. That peak is measured before stage 1 on a 2024 day the
repository has already replayed (a day's row count, not its archive's size,
sets it) and recorded in the amendment. The replay itself is unchanged.

**A failed download is not an exclusion.** `data/ensure._fetch_days` reports
any exception as an unavailable day, and `data/bybit._one_day` turns any
download error, a 404 or a dropped connection alike, into a skipped day. A
failed download of a large archive, which belongs to a busy day, would then
be excluded as a missing plane and tilt D and H toward calm days. In this
study's fetch, **every day the `HEAD` check lists as present must be
fetched**: any failure other than an HTTP 404 (a timeout, a dropped or reset
connection, another HTTP status, an archive that does not decompress) deletes
the partial file and is retried with backoff until the day is on disk, and a
stage is not complete, and nothing is computed on its block, until every such
day is on disk or excluded by one of two terminal rules, on every block, W
and D included. An **unreadable archive**: a day whose archive was downloaded
completely at least twice with the same sha256 and still does not
decompress, or whose replay raises on a complete archive (`reconstruct`'s
"produced no usable book rows", a line the decoder cannot parse); likewise
for prints. An **unfetchable day**: one whose download still fails after 10
attempts spread over at least 48 hours. Each is an exclusion, counted per
block by reason, and counts toward the more-than-a-quarter unreadable rule.
Only an HTTP 404 makes a missing plane; a 404 on a day the `HEAD`
check found present is reported as such. Funding and hourly bars follow the
same rule: a failed request is retried, and only the venue's answer that it
has no data (an HTTP 404, or an empty funding history) is a missing plane.
The print downloader's 120 s timeout (`data/bybit_trades.download_day`)
bounds each socket read, not the whole file, so a large file does not fail by
its size; a stalled connection does, and is retried.

| Stage | When | What | Book archives (GB): BTC, ETH, XRP | Print archives (GB): BTC, ETH, XRP |
|---|---|---|---:|---:|
| 1 | after this registration and the code it names are merged, and after stage 0's amendment if it cuts anything | W and D (212 days, `ob500`) for each instrument, one instrument at a time; funding for W and D only; hourly bars 2024-12-01 to 2025-06-30; the three BTCUSDT `ob200` reader-test days of 2026 | 76.0, 70.2, 49.6 | 15.3, 14.1, 12.3 |
| 2 | inside H's opened access, after the amendment is committed and the ledger has recorded the open | H (44 days, `ob500`), H's funding, E's and H's hourly bars from daily archives | 11.4, 12.7, 11.1 | 2.1, 3.8, 2.3 |
| 3 | inside F's opened access, after H's results are committed | F (88 days, `ob200`), F's funding and hourly bars from daily archives | 15.8, 18.5, 17.3 | 5.9, 7.4, 3.3 |
| 4 | inside P's opened access, after F's results are committed, only if the condition holds | P (45 days, `ob200`), P's funding and hourly bars from daily archives | 9.9, 9.4, 7.0 | 3.5, 3.6, 1.3 |

BTCUSDT's held-out archives are always fetched, for S25. ETHUSDT's and
XRPUSDT's for H, F and P are fetched only when a slot or a measured row is on
that instrument, each through the ledger as BTCUSDT's are.

**The ledger opens before the fetch.** `open` writes its entry first and
commits and pushes it, and the block's fetch then runs under that access; the entry records each fetched
file's sha256 and UTC download time. `open` refuses if any book, print,
funding or hourly-bar file of that block is already on disk, and H's `open`
also if any hourly bar of E is.

E's book, prints and funding are never fetched; its hourly bars are fetched
with H's. At most one archive per worker is on disk at once; the largest book
archive of W and D is 0.92 GB for BTCUSDT and ETHUSDT and 0.95 GB for XRPUSDT
(BTCUSDT's median 0.34 GB), so under 4 GB with four workers. Archives pass
through and are deleted after replay. Replayed at ten levels, a BTCUSDT day of
2024 took 40 MB on average and 60 MB at most, an XRPUSDT day 25 MB, so W and D
take 25–38 GB for the three instruments. Print parquet is budgeted at 0.13 to
0.25 of its archive, 7–10 GB for W and D.

**Disk.** 121 GB were free on the machine when this was registered; with the
10 GB margin about 111 GB are usable. Features go to **two float16 stores, one
zstd-compressed parquet file per instrument-day**, holding model inputs only;
labels, costs, the gate's inputs, the walk tables and every fill are computed
from the book's float64 prices, never from a store, and the parity tests run
at float16. The every-tenth-row store holds every instrument; the every-row
store, which holds the network's inputs, holds one instrument at a time, and
two every-row stores never coexist. Peak on D:

| Item | GB |
|---|---:|
| Replayed books, 3 × 212 days at 40–60 MB | 25–38 |
| Print parquet | 7–10 |
| Every-tenth-row store, float16 | 8 |
| Every-row store, one instrument | 18 |
| Archives in flight | 3.8 |
| Candidate tapes and the nulls' hourly sums | ≤ 10 |
| Models and results | about 7 |
| **Peak** | **79–95** |

Walk tables are never stored. **W's and D's features are deleted after H's
results are committed and before stage 3**; no later read uses them (F and P
run only frozen models), and they can be rebuilt from the books. Before each
stage, the fetch checks that free disk exceeds the stage's budget (its
archives in flight, books, prints and features) by 10 GB, and refuses to start
otherwise. The amendment records the measured sizes.

---

## Compute budget

On the machine the market-making rounds ran on (10 cores, 32 GB, an MPS
device). Worker counts are runtime arguments the ledger records, not hashed
values: at most 4 for the book replay; at most 6 for the simulator, while 6 ×
the peak resident memory of one day stays within 24 GB (2.9 GB measured on
the busiest BTCUSDT day of 2024); 8 for everything else. Fits run one at a
time on all cores (`models/gbm.py`, `n_jobs=-1`); network fits queue on the
single MPS device, which is part of the environment fingerprint. Each heavy
run starts under `nohup caffeinate` with its log in `logs/`.

**Unit costs**, measured on this machine on synthetic arrays and 2024 days
already on disk: a boosted fit on 1 million rows × 40 (× 100) columns 15 s
(32 s), its prediction on a day 0.6 s; a logistic fit on 1 million × 100
2.2 s; a network training batch 8.9 ms on MPS (557 ms on the CPU), so a full
network fit 22–44 minutes; the event-jump entry loop 6 ms per cell-day; the
simulator 5.6–10.3 s per BTCUSDT cell-day in one process, 0.85–2.5 s of wall
time on 6 workers; the normaliser 49 s per day for 28 columns.

| Stage | Size | Estimate |
|---|---|---|
| 1: fetch, replay, features, selection | 237 GB of archives; 636 instrument-days replayed at 2–3 minutes on 4 workers; two feature stores | 10–15 hours |
| A: fits | 600 walk-forward fits (boosted 240 × 1–2.2 minutes, logistic 240, bound regressor 60, network 60 × 22–44 minutes on MPS), predictions, candidate tapes | 30–55 hours, the network dominating |
| B: taker scoring | 7,200 cells × 150 days = 1.08 million cell-days from walk tables | 1–2 hours on 8 workers |
| D: the placebo search | Null A (999 draws on hourly sums), Null B (137 shifts), the refit replication, carry-forward, PBO and DSR | 7–15 hours |
| C: simulator | at most about 7,750 cell-days (S25-Z 1,800; S-P at most 4,500; S-M 1,350; A0 and FILL 92) | 1.8–5.4 hours (19–58 hours at 9–27 s a cell-day, the bound stage 0 tests) |
| E: freeze | frozen fits, A0 and FILL on O, σ_D and the MDEs, the dry run | 17–33 hours |
| **D in all** | | **68–126 hours, 2.8–5.3 days**, under the 6-day ceiling |
| H's read | the taker slots from walk tables; about 237 runs × 44 days for each simulator slot (206 placebo runs, at most 27 neighbours, 3 brackets, 1 latency); measured rows and the regime map | at most about 31,300 cell-days, 7–22 hours, one read session of at most 1.5 days |
| F's read | every frozen slot once as a measurement; each slot taken to F with its full kills about twice its H cost (88 days) | under a day, plus about 2 × H per slot taken |
| P's read | each slot read on P | about 1 × H per slot read |
| R's reference set, on F | about 58 split days per candidate, slices of per-day values computed on F | minutes, inside F's read session |

Every fit reads at most 4 million training rows, taken evenly in time, so a
fit's cost does not grow with D. The amendment records the measured times and
the largest worker's peak memory; a run slower than projected takes longer,
and nothing registered is dropped to save time once a 2025 file is on disk
(the reduction order acts only at stage 0).

---

## Who may read what, and the code that will enforce it

### The code to come

The pull requests that follow this one add, before stage 1:

- **the search's single entry point** over stages 0 to E, with the catalogue
  hash check and the trial ledger; candidate-row tapes, walk tables at every
  100 ms row and the event-jump thinning loop; the vectorised neighbourhood
  scores, tested equal to `validation/search.neighbourhood_scores`;
- **the signal pieces**: the per-side θ of the hold-first rule, the EV gate,
  the Gaussian head with its lazy window dataset, the ten-level tensor, the
  2025 microstructure set, the walked-book label, and the time-based forward
  move and control label, each under the leakage checker and with
  streaming-versus-batch parity tests; the depth and flow builders wrapped as
  registered features with declared lookbacks, and prints aligned backward
  ([Labels, features and models](#labels-features-and-models));
- **the per-instrument loader and clip**; the `ob200` archive name in
  `data/bybit.py`, proven on a synthetic fixture before stage 1, and on the
  three reader-test days, against D's frozen `max_sequence_gaps`, before H is
  opened ([Data](#data)); and the daily-only fetch of hourly bars in
  `data/binance.py`;
- **the fold-wise detector flags**, the flag-causality test and the
  fold-level leakage test;
- **the placebo search**: Null A, Null B with its walk-table route and its
  parity test against the executor (run at stage 0), the counts of admission
  (iv) computed directly with a unit test at the 137-shift boundary, the
  refit replication, carry-forward,
  CSCV PBO, and the deflated Sharpe ratio with N_eff;
- **execution**: the event-time taker executor; the simulator's signal-entry
  quoter and S1 with the lean on a fixed `clip_btc`, wired to `guard_pull`
  and `guard_widen`; the simulator's taker orders priced by the executor's
  walk function in place of `_walk` with `flatten_slippage_bp` 0.5, with a
  parity test of the two walks and the late-snapshot counters in the
  simulator; the fill-rule comparison;
- **statistics**: the one-sided day-level p-value, Holm and the placebo
  comparison through `market_making/verdicts.day_t`, `holm`,
  `exceeds_placebos` and `t_sf`, and the Newey–West t in
  `evaluation/significance.py`;
- **this study's fetch and feature stores** ([fetch plan](#fetch-plan)): the
  book replay on at most 4 workers with its measured peak memory, the retry
  of every failed download until success (only an HTTP 404 is a missing
  plane), the two float16 stores, the disk check before each stage, and the
  deletion of W's and D's features before stage 3;
- **the presentation**: regime attribution and the regime map; the
  leaderboard, marginals, nulls and regime-map files, the figures, and the
  filler of the headline sentences ([artefacts](#artefacts-and-headline-sentences));
- **this study's loader and ledger**, on the pattern of
  `market_making/round2.py`, with block constants of its own (round one's
  `block_of` is not used): the YAML validated against this registration,
  placeholders hashed as in the earlier rounds, and an access that permits
  days per block, plane and instrument. The entry point,
  `experiments/short_horizon_2025.py`, is committed with all study code
  before the stage 1 fetch (the code-freeze note of
  [selection](#selection-on-d-admission-and-the-finalists)) and frozen with
  the amendment, and computes everything a block's read produces, statuses
  and kills included, in one read session.

### Access

**The development access** permits W and D (book, prints, funding) for the
three instruments, the hourly bars up to 2025-06-30 and the reader-test days.
It refuses every 2025 day after 2025-06-30 in every plane, including the daily
bars already on disk (`data/bars/BTCUSDT/klines-1d`); a read of those after
that date falls under the ledger rule below.

**The ledger**, `experiments/results/short_horizon_2025_ledger.json`, guards
E's bars, H, F and P.

- **H** opens only if the amendment records the YAML's sha256, the file on
  disk has it and is committed, the tree is clean, every frozen weight file's
  sha256, every frozen finalist file's and the fitted-values file's sha256
  equal the amendment's, the trial ledger and its sidecar are committed with
  the sha256 and the row count the amendment records, the catalogue's sha256 equals the
  registered one (or stage 0's amendment's), and no file of H and no hourly
  bar of E is already on disk.
- **Every open (H, F and P), and the start of every process of a read
  session**, also require: `git status --porcelain --untracked-files=normal`
  lists nothing but the ledger file; `git diff --name-only <session's
  commit>..HEAD` lists nothing but the ledger file (for an open, HEAD is the
  session's commit); and the **environment fingerprint**, the sha256 of the
  installed distributions with their versions (`structural-break` at its
  locked commit included), equals the one H's open entry records.
- **The open entry is committed and pushed before the block's fetch
  begins.** It records block, session commit, config sha256, environment
  fingerprint and UTC time; the fetch's per-file records are committed when
  the fetch ends, before anything is computed. Every later process of the
  session, and every later open, refuses unless that entry is in HEAD's
  history, so a crashed read cannot be erased and opened again.
- **A read session** is bound to (block, commit, config sha256, environment
  fingerprint). Any number of processes and resumptions at that commit count
  as the first read; outputs are not inspected until the session writes the
  block's results file. A resumption at another commit is a second read. The
  **worker count** is a runtime argument recorded in the ledger, not part of
  the configuration's hash; a test shows the entry point's outputs identical
  at 1 and at N workers, and a resumption may use fewer workers.
- **A read that cannot finish.** A failed process of a first read is resumed
  at the same commit. If the same stage fails again with the same exception,
  or the read has not finished after five resumptions, the session closes by
  writing the block's results file with every slot `not assessed
  (<block> read failed)`, the stage reached and the exception. The block is
  then handled as an unreadable block of the same letter: after H, nothing is
  taken to F, F is read as measurements at the same commit and P is not
  read; after F, H's statuses stand, marked "F read failed", and P is not
  read; after P, as P unreadable. No code changes; a corrected run is a
  successor registration.
- **The session closes** by writing the block's statuses and every kill,
  computed by the entry point, to `experiments/results/short_horizon_2025_H.json`
  (`_F.json`, `_P.json` for the later blocks), and that file's sha256 into
  the block's ledger entry.
- **F** opens only if H's results file is committed unchanged (its sha256
  equals the ledger's), `git diff --name-only <H read's commit>..HEAD` lists
  only `README.md` and paths under `experiments/results/` and under `docs/`
  outside `docs/preregistration/` (so the code, the configuration, this
  registration, the tests, the entry point and the lock file are those of the
  H read), the frozen weights' and the fitted-values file's sha256 still
  match, the conditions above on every open hold, and no file of F is on
  disk. **P** opens on the same conditions against F's read, and
  only when a slot qualifies; its session then computes the qualifying
  slots' statuses and, as measurements, every other frozen slot, measured row
  and regime-map segment on P.
- Each first read writes block, commit, config sha256 and UTC time before the
  access is returned; a second read raises unless forced, and everything it
  produces is stamped `second read` and cannot change a status.

**H, F and P are spent** for every configuration of this catalogue on the
three instruments. A successor
registration, a corrected re-run after a bug included, uses blocks after
2025-12-31 and never the reader-test days.

Round one's `prereg.Access` and the round-two access are not used to read any
day of this study. `pipeline/execution`'s market-maker path builds its access
with round one's `access_for`, so it is not used either: the passive
strategies call `market_making/simulator` directly, on days loaded through
this study's loader.

---

## Amendment protocol

Stage 0's cuts, if the projection needs any, are a dated amendment of their
own, committed before stage 1 with the new catalogue hash, N_max, the
configuration's new null-placeholder sha256 and S25's and S25-Z's
neighbourhoods as cut; the code-freeze note follows it, also before stage 1. After the
development runs, and before H is read, one dated amendment is appended and
committed: per instrument, the days excluded and the sequence-gap threshold
(by its formula), the tick and lot steps, the clip and the selected features;
the longest feature lookback `N`; the reader-test days' message counts,
intervals and gaps; the trial ledger's sha256 and row count, N_trials, N_eff,
R_D and the hand-run log; for each stratum the chosen cell with its neighbourhood, its S and the
outright peak; every replication of the placebo search (A per stratum under
Null A and Null B and at the refit, the seeds, the shifts, the quantiles and
the p-values); each admission decision with its reason, the carry-forward
estimates, PBO, the deflated Sharpe ratios and the selection premium; S-P's
signal with the values that chose it; σ of each slot's daily series on D and
the minimum detectable effects; the A0 and FILL tables on O; the detector's
alarm rate on D, with every alarm's timestamp and channel, and, if it applies,
the declaration that R is untestable; the fold-wise flag lists and the
detector's 24-value hour-of-day profile fitted on D; the sha256 of the
fitted-values file ([Blocks](#blocks)); the sha256 of every frozen weight
file and of each frozen finalist file, every threshold and every checkpoint
epoch; the entry point, committed before the stage 1 fetch and frozen with
it; the
compute record, with the replay's measured peak memory, the worker counts,
the device and the stores' sizes; the fits that lacked a class, with the
side; and the sha256 of `configs/short_horizon_2025.yaml` after the frozen
values are written into its placeholders, on a line of the form
``Frozen configuration sha256: `<hash>` ``. Nothing else may change. Each
frozen value has one placeholder: weight sha256s under
`models.weights_sha256`, checkpoint epochs under `models.checkpoint_epoch`,
hold-first thresholds under `slots.<slot>.thresholds`, the search's records
under `search.result_trials`, `search.result_choice`, `search.result_admission`
and `search.result_nulls`, the fold-wise flags under
`detector.flags_D_foldwise` and the frozen finalist files' sha256 under
`artefacts.frozen_finalist_files_sha256`; a neighbourhood lists its cells, their D values
and the outright peak, and refers to those by cell without repeating them.
**A frozen finalist file** holds a slot's configuration, weights, thresholds,
gate levels, regime window and fitted values; a later trial on the venue
loads exactly that set, and this study places no order.

A correction found after this registration and before H is read (a bug in a
ported feature, the label, the simulator or a reader) is made only by a
further dated amendment that states it and, if it could change a development
number, re-runs D in full before anything is frozen. Every change to the
frozen study code before H is read is such a correction. **Complete D runs
are numbered**, and R_D is their number: the rows of every superseded run stay
in the trial ledger and in N_trials, and admission (iv) and p_search are taken
at α / R_D (Bonferroni over runs), so that fixing and re-running until
something is admitted is paid for. After H is read, nothing
is changed and nothing is re-run under a different rule; the ledger's
conditions for opening F and P enforce this for the code, the configuration,
this document, the environment, the frozen weights and the fitted values.
After any block is read, no finalist is re-ranked, no cell re-chosen, no m
changed, no regime window retuned and no detector refitted, and the D
leaderboard is never re-run on held-out data.

**A defect found after a read can only lower a status.** It is stated in a
dated note, `docs/short_horizon_2025_defects.md` (under `docs/` and outside
`docs/preregistration/`, so the F and P gates still pass), which names the
mechanism by which the defect could raise a result (a look-ahead in the
executor, a funding sign, a wrongly chosen cell). Every status it could
affect becomes `void (defect found after the <block> read)`, which counts as
killed for routing to F and P and enters Holm with p = 1 on later blocks.
Holm on a block already read is recomputed from its committed results file
with that p set to 1, and a status may only fall. Later reads still run on
the unchanged code; their numbers for a voided slot are reported as
measurements. Nothing can raise a status; a corrected run is a successor
registration.

**The ledger rule.** Only reads through the ledger count. A number computed on
H, F or P outside it — in a notebook, a script run by hand, or a second read —
is not a result, cannot change a status, and is reported as such if it is
reported at all. A negative verdict on H ends the regime question for that
slot: F is still read and reported, but nothing is taken to P.

---

## Artefacts and headline sentences

**Files**, written by the entry point:

- `experiments/results/short_horizon_2025_trials.parquet` and its sidecar
  `…_trials.json`, the trial ledger; N_trials, its distinct (run, key)
  pairs, is the N quoted on every page, and its sha256 and row count go in
  the amendment;
- `…_leaderboard.csv`: every configuration with gross, net, G, trades a day,
  day-level t, IC, S, A, the admissible flag, its rank within its stratum and
  overall, the region tags (T1, T2) and S25's badge; the header gives N
  planned, scored and skipped;
- `…_marginals.csv`: for each axis value, the median gross, net and G over
  cells, the form of `docs/findings.md`'s grand-search marginals; the regime
  row is the median per window;
- `…_nulls.csv`: every replication's A per stratum under Null A and Null B,
  the refit point, the quantiles and the p-values;
- `…_regime_map.csv`: per slot and block, every segment between alarms;
- the frozen finalist files, by sha256 in the amendment.

**Pages and figures**, in the README, in both themes through `<picture>`: the
placebo search (the real A against both null distributions and the refit
point, with p_search); the finalists on H, then F and P, one row per slot and
S25-Z, with the status, the first kill and its value against its reference
(`verdicts.KillCheck`), the mean, t, Newey–West t, median day, trades, MDE,
placebo percentile, K-nbhd median, the fee break-even per side against every
tier, H before 2025-08-01 and the regime attribution; the S25 page (the edge
ladder, the T1 and T2 region rows, the rank r of N); the regime map, one chart
per execution path; the break-detector view for 2024-12-01 to 2025-12-31
when P is opened, otherwise to F's last day, 2025-11-16
(blocks and the slots' guard windows shaded, every alarm with its channel,
`b`, F− and F+, the 1 s CUSUM's restarts, and the public 2025-09-13 daily-bar
alarm labelled); the fill-rule comparison (A0 on O; S-P and S25-Z on H); and
the rows of `docs/findings.md` and the README summary.

**Headline sentences, fixed now.** The bracketed parts are filled by the entry
point and never edited.

| Outcome | Sentence |
|---|---|
| Search on D | "Searched [N] configurations ([axes]) on 150 development days across [instruments]; on development data, selected, the best direction edge [was / was not] distinguishable from search noise (p_search = [x]; Null A p = [y], Null B p = [z]; [R_D] D runs)." |
| Payoff on D | "[No configuration paid the 5.5 bp taker fee on development data / S-[x] was admitted on development data, selected]." |
| Each slot on H, F, P | "[slot] on [block]: [status]; first kill [K-x] at [value] against [reference]." |
| S25 | "S25's gross edge was [x] bp of turnover against the 5.5 bp taker fee; break-even [y] bp per side; on H before 2025-08-01: [status]." |
| Passive entry and market maker | "Passive entry and the market maker on [signal]: [D status / H status]; the market maker's making part on [block] was [v] bp per fill." |
| Regime axis, one per slot with a window above 0 | "[slot] on H: its [w] h guard after detector alarms [did / did not] add more than random pauses ([percentile] of the random flag sets)." |
| Regime map | "Between detector alarms, [slot] was positive in [a] of [b] segments; the detector cut the segments, not the results." |
| Break | "R: [supported / not supported / untestable (reason) / not tested]." |

The first kill is the first that fires in the order of the slot's row of
[the kill table](#kill-conditions-per-slot). Values are in bp of traded
notional for S25, S-C, S-R and S25-Z, and per 100 USDT of clip a day for S-P
and S-M. In the regime-axis sentence "did" means the slot's difference from
its window-0 twin is above the 95th percentile of the random flag sets. The
making part's [block] is H when S-M has a status on H, else D.

A negative result therefore reads as the pipeline measuring how far each
configuration was from paying; it is not a missing result. D numbers are never
quoted as results, except p_search, labelled "on development data, selected";
otherwise only H, F and P statuses are.

**Later horizons.** The configuration's `search` block is generic: the same
block, entry point, ledger, nulls, regime axis, verdict helpers and status
rules carry over to a later registration with only values changing. Funding
carry would use a clock of 8 h settlements, a week as the cluster, a universe
from `pipeline/discovery.py`, the positioning features, taker execution at
settlement, the same detector with windows in settlements, Null A flipping
signs per settlement-week and Null B re-pairing whole weeks; daily trend, 1 d
klines with a month as the cluster. Each needs its own blocks, the Newey–West
lag in cluster units, and a replacement for the rule that a day starts flat
and is the cluster.

---

## What this study cannot establish

- **A fee schedule of 2025.** Bybit's tiers are the ones published in 2026,
  applied to 2025.
- **Queue dynamics inside 100 ms.** The book is a 100 ms photograph; queue
  position is a rule, bracketed, and K-queue requires a verdict to survive the
  pessimistic side of the bracket, not the truth.
- **Anything about capacity.** The clip is 0.01 BTC, or the same notional on
  the other instruments, and own impact is not modelled.
- **Anything beyond three instruments, one venue and one year.**
- **A strategy outside the catalogue.** The catalogue is finite and fixed in
  advance; what it does not contain is not looked for.
- **A passive-only edge without direction information.** The passive-entry and
  market-making slots inherit their signal from the taker search or from S25,
  so an edge that only a quoter could earn is not searched for.
- **That H and F are comparable.** The book archive changes on F's first day;
  a difference between H and F may be the archive.
- **That S25 was chosen blind to H's days from 2025-08-01, to F or to P.**
  Its [information set](#what-was-known-before-this-was-written) is stated;
  the slots on its signal take their status from H's days before 2025-08-01
  only and can be no more than candidates.
- **A null calibrated by refitting every model.** The refit replication of the
  placebo search covers the gradient-boosted and logistic models only; the
  network and the bound regressor rest on the fold-level leakage test.
- **A market-wide break measured per instrument.** BTC's hourly flags guard
  every instrument.
- **What a regime-map segment means.** Segments between alarms are
  descriptive; none is a status.
- **Why a strategy stopped.** R tests a causal stopping policy on a detector
  of BTC's returns; it does not identify the mechanism.

---

## Choices and their reasons

The registration was reviewed three times before it was merged and before any
data was read, for peeking, for multiplicity and for feasibility, and the
revision was checked once more for consistency; a last review read it for
consistency, for decisions left until after a read, and against the code it
names. Every
blocking and should-fix point was taken in, and the notes where they
pointed to a gap. It was then recast, still before any data was read, from
a test of hand-picked hypotheses into a registered search, from three
designs (one built for rigour, one for feasibility, one for presentation)
judged against each other; the design built for rigour is the base. Where
the reviews or the designs proposed different things, a fix left a value
open, or a proposal was not taken, the choice and its reason:

**The recast into a search.**

- **No configuration is culled on a test day's outcome**, rather than
  successive halving of families or fits: every fit is made and every taker
  cell is scored on all 150 days, so the placebo search re-runs the whole
  search exactly, and halving would raise its null maximum.
- **The choice ranks by net, admission reads G**: net is what a slot claims;
  G has the costs taken out, so its Null A is centred whatever the fee; the
  drift it keeps is controlled by Null B and K-drift.
- **Two admission levels**, information then payoff: a signal can be
  admitted for its information and still not pay, which is when passive entry
  and the market maker matter, so they are admitted on their own payoff.
- **S-P and S-M always run, on S25's signal when nothing is admitted**, rather
  than only inheriting an admitted signal: otherwise they would usually never
  run. They choose nothing on a passive outcome, so they need no search null.
- **The market maker is one S1 slot with the lean**, at round one's frozen
  values, rather than a searched family of quoters: both rounds of 2024
  killed pure quoting.
- **k = m = 5**, one slot per stratum and execution path plus S25: at m = 5
  the t ≥ 3 bar binds on every block, so the breadth of the search costs
  nothing on H; a seventh slot would let Holm decide on short F and P series.
- **S25-Z at α/6**: the earlier α/(m + 1) rule carried to m = 5.
- **The slots on S25's signal take their status from H before 2025-08-01 and
  stop at candidate** (multiplicity and protocol reviews), rather than a
  narrower statement of S25's information set: the registration does not
  claim S25 blind to market conditions after 2025-08-01, so only D and H's
  days before that date can test it, and a status on later days could be
  earned on blocks its coordinates may reflect.
- **S25 is one fixed point**, rather than named points chosen on D: a point
  chosen on D is a region, so T1 and T2 survive as leaderboard regions with
  no slot.
- **The regime policy is a searched axis with pull only for the taker and
  passive entry**, the pairs `VALID_PAIRS` allows; a new meaning of "widen"
  (a stricter entry) was rejected, and a synthetic null showed no saving from
  it either way. **The hour profiles are fitted fold by fold on D**, because
  one profile fitted on all of D would leak later months into earlier folds'
  guards. **BTC's flags guard every instrument**, as a market-wide break.
- **Three instruments, a market gate, a 120 s hold and 25 trades a day** were
  added to the rigour design's grid; a second boosted hyperparameter set, a
  4 × 4 grid of k and β, take-profit, stop and cooldown exits, and the
  network on the tensor alone were not, each for cost without a prior reason.
- **A hard budget fixed before any fetch**: the catalogue hash, a 6-day
  ceiling and a reduction order, applied only at stage 0, so that nothing is
  cut once a 2025 file is on disk.
- **Two float16 feature stores**, rather than one float32 store: three
  instruments do not fit the disk at float32, and costs, labels and fills
  come from float64 book prices, never from a store.
- **The regime map is a measurement**, every segment shown and none selected,
  rather than a status: it is the way to show a period in which a taker or a
  maker paid without choosing that period after seeing it.
- **No earlier figure enters the expected outcome or the edge ladder**; the
  prior rests on this repository's own results and the fee arithmetic.

**Earlier choices, kept.**

- **Holm's m is fixed on every block**, not the number taken to F or P: the
  stricter of the two proposals, and fixed before any read.
- **A missing hour carries its close with a zero return**, rather than being
  dropped: one row per hour keeps the windows in hours and the hour-of-day
  profile aligned, and the move over the gap enters the next return either way.
- **R is tested for every candidate on H, with a reference set of split days
  at which the slot also passes**: testing on H's candidacy removes the
  gate on F−, and the conditioned reference set removes the selection that
  "confirmed until the break" still makes. The split days are enumerated
  (about 58), `b` excluded, so the floor is 19 qualifying days besides `b`
  (the smallest p_R is then 1/20), not 50 of 200 draws.
- **A result first significant on F is "exploratory" and P is its only
  confirmatory test**, rather than α being split between H and F: H stays the
  only gate to candidacy.
- **S25-Z is tested alone**, not as a member of the family, so that a
  claim conditional on a zero-fee venue cannot change the finalists' bar.
- **The passive entry is kept at its first price**, not re-pegged: no
  re-peg cadence is left to choose, and the order keeps its queue place.
- **The clip is fixed** (`clip_btc`; 0.010 BTC, or the same notional at W's
  prices on the other instruments), rather than K-cap being measured against
  the clip in force at entry: it gives K-cap one meaning.
- **Three reader-test days in January 2026**, not one: more cadence and gap
  data for the `ob200` replay, all outside every block.
- **H's and F's first days start cold**, rather than carrying rolling state
  across E or across the archive change: no E day is read, and no `ob500`
  state is carried into `ob200` data.
- **K-latency applies to every slot and to S25-Z**, the passive paths
  included.
- **K-trades is applied before the kills**: a result on fewer than 100 trades
  is inconclusive whatever its sign, so a handful of losing trades neither
  kills nor passes a slot.
- **The mean channel is read from `odds_mean`**, the column `detect_breaks`
  writes; its internal name is `sr_mean`.
- **The EV gate trades one full clip, and Kelly sizing is dropped**
  (feasibility review). `FractionalKelly.scale(edge, volatility)` returns fraction × edge /
  volatility², clipped to [0, 2]. At its default fraction of 0.25, with μ and
  σ in bp, a gated trade with μ = 15 and σ = 10 gets 0.0375 of a clip, below
  the 0.001 BTC lot, so the slot would be inconclusive by construction; in
  decimal units the same trade gets 375, clipped to the cap, so every trade
  would be one clip and Kelly would do nothing. A scaling between the two
  would be a value set without a basis. Without it, stratum R's trade count is
  its gate's, and K-trades counts round trips as for stratum C.
- **Feature selection, per instrument, ranks by correlation with the
  forward move at `h` = 20 s** (`FeatureSelector`'s default `score_target="ic"`), with the
  walked-book label at 0 bp as `y`. The feasibility review proposed the label
  at 5.5 bp; the 0 bp label, fixed after the multiplicity review, is kept.
  Under this ranking the label only marks which rows are usable, and it
  exists on the same rows at either fee.
- **The other values the feasibility review found open on D were already
  pinned** after the other reviews, and are kept: the control label's
  smoothing at k = 20 rows (one constant for the three holds, rather than k
  equal to the horizon in rows); the 5-trade filter, which limits only the
  chosen cell while every neighbour counts; `not run` with p = 1 for a
  slot with no admissible cell; and `max_sequence_gaps` as the 95th
  percentile of D's per-day gap counts (rather than the smallest value that
  keeps 90% of D's days).
- **The values the code to come needed are fixed here**, each as a value or a
  rule that reads only W, January or a fold's training rows: the Gaussian
  head's variance-prior weight and clamp, the tensor's σ window and spread
  floor, the winsorising quantiles and window, the network's early stopping
  on the inner validation's rank IC, and how `N` is derived from lookbacks
  declared in rows. Left open, each would be an uncounted axis.
- **The minimum detectable effect on F and P** uses the frozen σ_D and share
  with that series' usable days, rather than H's 44, so that the label on a
  later block describes the days it has. Its formula on H was already
  `(3 + 0.84) × σ_D / √n_eff`, as the feasibility review asked.
- **At most 4 replay workers**, rather than a replay that writes rows into
  preallocated arrays (feasibility review): the replay stays the code that
  wrote the 2024 books, and the cost is a few hours of wall time.
- **A per-day feature store**, rather than features computed afresh on each
  run: every walk-forward fold and every cell reads the same D days again
  (float16 since the recast, above).
- **The fill ratio counts orders with any fill**, rather than filled volume
  over posted volume: it is the repository's existing `fill_rate`. A partial fill
  counts as filled, which makes FILL's prediction for the queue rule harder to
  meet, not easier; filled volume is reported beside it.
- **Only `validate_book`'s `positive_prices` error excludes a day**, as
  registered, rather than any error in its report: the registration named a
  non-positive price, and adding the function's other checks now would add
  exclusions that D has not been looked at for. The off-grid check moves to
  the simulator's day loader, where the code has it, and `flatten_stale`
  joins the exclusions as R16 has it.
- **K-dir is a kill for S-M only, and reported for S-P and S25-Z**
  (multiplicity review, notes). It guards a market maker, whose claim is the
  part earned at the fill, as in round one; a directional signal entered
  passively is meant to earn the move after it, so the kill would remove the
  mechanism under test. For S-P and S25-Z it could only cause false
  negatives; the false-positive routes stay guarded by K-placebo (with the
  stale signal), K-drift, K-queue and K-latency.
- **The detector's alarm rate on D stays in-sample, and is reported as
  optimistic** (peeking and multiplicity reviews, notes), rather than
  measured with a profile fitted elsewhere: the leniency can only end R early
  or make it untestable, never make it supported.
- **No 2026 confirmation block is registered here.** P is the last block of
  2025; a confirmation on 2026 data needs its own registration, which the
  spent-blocks rule confines to dates after 2025-12-31 and which is committed
  before any 2026 book or print is fetched (the reader-test days excepted).
- **No entry is skipped for a late snapshot**, and late snapshots are counted
  at entry and exit instead (final consistency check). The registered "more
  than 1 s old" entry skip could not apply: decisions are made at row stamps,
  and the snapshot walked is never older than the order. Reading it as "the
  next snapshot comes more than 1 s late" would make a trade depend on the
  feed after the decision, at the feed gaps where a skip would select most.
- **P is read for every slot that passed on F**, without the earlier
  "at least 20 usable days" condition: counted in calendar days it always
  held, and counted in days with a trade it left a slot that passed on
  F with 10 to 19 such days without a status; the bar on F already asks for
  10.
- **A slot inconclusive under K-trades enters Holm with p = 1**: its
  p-value is not evaluated, and a small one would otherwise move another
  slot to a laxer step.
- **With R supported, an empty P series has mean 0**, as a neighbour cell
  without a trade scores 0 in K-nbhd: a strategy that does not trade after
  the break has not paid after it.
- **A result inconclusive under K-trades is not taken to F** (consistency
  review, which left the reading open): its mean is not evaluated at
  precedence step 2, so its sign cannot route it. Taking it would give a
  slot with too few trades a route to "confirmed on P, first significant on
  F" through a sign the precedence does not read; F still reports its numbers
  as measurements.
- **Both terminal fetch rules, not one** (review of decisions left until
  after a read, which offered either): the identical-hash rule ends a corrupt
  archive, and only the attempt cap ends a download that keeps failing; both
  count toward the unreadable rule, so neither can quietly thin a block.
- **The open entry is pushed as well as committed**: a local commit can be
  reset without trace, a pushed one cannot.

---

The configuration as registered, with every placeholder `null`, has sha256
`035c4f44d85c3b3926e08016ad94a4c9bb20a51c7ac775514a256ef0af324f71`.
(Earlier versions of this file on its unmerged branch had other hashes, all
superseded drafts that do not count:
`e01ba377857e895b2d81d6a625434fb183210825ea906f2801e922eef7c93308`
before the reviews,
`c2da1566d9297499cd01e1d084e5bf6a0116b19a88d4dbccdf2b62ffae495262` before the
feasibility review's later points,
`6d35940bc3e4845b30056daaf60887d01dbf2c6e6ddbb361ff8ab3ecd1fec1f5` before the
final consistency check,
`729f042bc55cae70fbdf786d8f38581b2539fc1a422a978d8bf9c674cee58fd8` before the
last review, and
`511626b7c0464cd91bafc469ea151af0769121b13946a2066c46a6ca22265fe7` before the
recast into a search, and
`3db898a0100496b164785a8cabf9af5540e56fd8d8227915101e97ef884c969f` before the
market maker's round trips were defined.)
This study's loader will check that putting `null` back into every placeholder
of the frozen file gives this hash, or the hash recorded by the latest
amendment committed before stage 1 (stage 0's, if it cuts anything), as the
earlier rounds' loaders do.

---

## Amendments

None yet.
