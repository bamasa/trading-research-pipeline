# Pre-registration: the 2025 short-horizon study, re-run on public data

This document registers a study before any of its data is read. It is committed
together with [`configs/short_horizon_2025.yaml`](../../configs/short_horizon_2025.yaml),
which holds the same values in the form the code will read, and the two must
agree.

| | |
|---|---|
| **Committed** | 8 October 2026, against `main` at `af148b5`, before any 2025 book, print, hourly bar or funding file is fetched for it. Revised the same day, before it was merged and before any data was read, after three reviews (peeking, multiplicity, feasibility); the choices the reviews left open are under [Choices and their reasons](#choices-and-their-reasons) |
| **What it registers** | A re-run, on Bybit's public BTCUSDT archives, of the short-horizon order-book research the author did in 2025: the blocks, the data and its coverage, the labels, models and strategies, what is searched on the development block, hypotheses T1, T2, M1, Z and R with their metrics, kill conditions and placebos, the measurements A0 and FILL, the break detector, the status rules, the fetch plan, the compute budget and the amendment protocol |
| **Computed so far** | **Nothing.** No 2025 order book, print, hourly bar or funding rate has been fetched, opened or computed on. The only 2025 data on disk is the daily bars of an earlier, unrelated break-detector run (item 7 below). What was known is listed under [What was known before this was written](#what-was-known-before-this-was-written) |
| **How it changes** | Only under the [amendment protocol](#amendment-protocol): dated amendments appended at the end. One amendment is planned, after the development-period runs and before the held-out block is read |

The market-making rounds are
[`market_making.md`](market_making.md) and
[`market_making_round2.md`](market_making_round2.md); their conventions hold here
unless this document says otherwise. Names that do not exist on `main` at
`af148b5` — the ten-level book tensor, the walked-book taker label, the
time-based forward move and control label, the Gaussian-head network and its lazy window dataset, the hold-first rule, the EV
gate, the event-time taker executor, the signal-entry quoter and the fixed
`clip_btc` setting of the simulator, the fill-rule comparison, the Newey–West
day-level t, the daily-only fetch of hourly bars, this study's fetch, feature
store, loader, ledger and single entry point — refer to code added by the pull
requests that follow this one. The code they re-type from the author's 2025
research comes in under [`docs/disclosure_policy.md`](../disclosure_policy.md);
none of that work's data, model weights, fee terms, internal paths or tuned
values enters the repository, and none of its results is evidence here.

## Why this study

In 2025 the author did short-horizon order-book research on BTC and other
crypto pairs while employed at a trading firm. Its recipes — a gradient-boosted
classifier on microstructure features with a hold-first decision rule, a
convolutional network over a normalised ten-level book with an
expected-value gate, a close-by-time exit — were never measured honestly: the
headline backtest ran at zero fees, its thresholds were chosen on the window it
was scored on, its fills came from a closed engine whose fill rule is not
known, and its documents contain a units error in the costs. This study is the
first honest measurement of those recipes, on data anyone can download, with
the repository's leakage checks, purged splits, event-time fills and
pre-registered verdicts.

It also asks a question the author raised: whether the recipes worked for a
while in 2025 and then stopped. That question is the easiest one in the study
to answer by looking for the period that flatters it. It is asked here only as
a causal policy fixed in advance (stop at the first alarm of a frozen
detector), and only for a strategy that has first passed on the held-out block.

**The prior is negative**, for reasons written down [below](#expected-outcome-stated-in-advance).
The study is built so that a negative answer is informative: every verdict
carries its fee break-even, so a reader can see how far each recipe was from
paying.

---

## What was known before this was written

Everything below was available when the design was chosen, and some of it
shaped the design. None of it is a 2025 number computed by this repository.

1. **The author's 2025 research**, read for the plan that precedes this
   registration: its code and its own summaries. What shaped this design, and
   is cited only to set predictions, never as evidence:
   - its headline backtest ran on Binance spot BTC/FDUSD, at zero fees, over 23
     days from 4 to 26 February 2025, with entries passed to a closed backtest
     engine whose fill rule is not known; that engine reported most orders
     filled (about four in five);
   - its own summary of that backtest's profit corresponds to about
     36 parts per million of turnover, **0.36 bp of turnover**, gross, at zero
     fees;
   - its decision thresholds and backtest window were chosen on the window
     that was scored, and its configurations were ranked on test days;
   - a costs field was built as 1e4 times a rolling z-score and read as basis
     points;
   - a smoothed label used for one model is partly known at decision time;
   - its label described a much larger clip over a much shorter horizon than
     the trades it was used for;
   - its splits by row had no purge, and some statistics (winsorising bounds,
     size floors) were fitted on whole days, after the decision time.

   **The research's development data overlaps this study's held-out blocks,
   by date.** The network recipe that T2 re-types was developed on XRP data
   from 2025-08-01 to 2025-11-16, its packaged example ran on 2025-11-09, and
   the boosted-classifier track ran until 2025-08-25. Recipe-level choices
   (architecture, feature families, the gate) were therefore made while the
   author could see the last 20 days of H and all of F, on other instruments
   and venues. Only these dates are disclosed; none of that data is used. H and
   F are therefore not fresh for the recipes, only for every value this study
   sets on D, and T1 and T2 are also reported on H's days before 2025-08-01, as
   a labelled sensitivity ([measurements](#measurements-without-a-hypothesis)).
   P begins after every dated entry in the research.

   Each of these is corrected, or tested, by a rule below. The research's
   tuned values (thresholds, sizing caps, smoothing constants, volatility
   floors, cost scenarios) are not used; every value a recipe needs is
   re-derived on block D or fixed here without reference to them.
2. **The author's recollection**, given for this registration and recorded
   verbatim as a recollection: *"it worked until an anomaly in autumn 2025"*.
   The research documents, searched for this registration, record no date for
   any such change; their dated entries run to November 2025 and none of them
   marks one. The blocks below were fixed with this recollection in mind (F
   contains the autumn), and the date of any break is left to the frozen
   detector.
3. **General market knowledge, not from the research documents**: crypto
   markets had a widely reported liquidation cascade on 10–11 October 2025. Its
   relevance to the recollection is not known.
4. **This repository's own results**: directional taker trading from one
   instrument's book is killed (§25, §26), passive entry of a directional signal
   is killed (the gross edge inverts when resting: +3 to +8 bp crossing, −0.5 to
   −4.2 bp resting), and both market-making rounds are killed
   ([`docs/findings.md`](../findings.md)). These are the reason for the prior.
5. **Bybit's fee schedule**, transcribed in 2026 into `backtest/costs.py`
   (`BYBIT_LINEAR_TIERS`), with its caveat (`BYBIT_FEE_CAVEAT`): it is the
   schedule published in 2026, applied here to 2025 data, when it may have
   differed.
6. **Archive metadata, read for this registration.** HTTP `HEAD` requests for
   Bybit's BTCUSDT order-book and trade archives on every day from 2024-12-01
   to 2025-12-31 returned each file's existence and size and nothing else; no
   archive was downloaded. The result, which moved two block boundaries, is in
   [Data](#data). The per-day sizes were seen; they reflect the archives'
   depth and a day's activity, not prices, spreads or fills, and a file's size
   is a rough proxy for that day's activity in H, F and P. The full per-day
   table is committed as
   [`experiments/results/short_horizon_2025_head_check.csv`](../../experiments/results/short_horizon_2025_head_check.csv)
   (sha256 `86a5719fd92845d111bf846c15fded09409ff084f42679a556c23e113cf4d8e0`).
   **No value in this registration was set from what it showed, except the
   H/F boundary it moved**; the revisions made after the reviews used only the
   archive change it found, and the fetch plan's sizes and disk budget use its
   per-block sums and the largest and median archive of W and D, which set no
   research value. Three further `HEAD` requests on the same day found
   the `ob200` book archive, and no `ob500` one, for 2026-01-05, 06 and 07
   (225, 205 and 197 MB); these are the reader-test days of
   [Data](#data), outside every block.
7. **A break-detector run that already covers 2025.** The repository holds a
   run of `validation/structural_breaks.py` at its daily setting on Binance
   BTCUSDT daily bars from 2024-01-01 to 2026-09-30
   (`experiments/results/structural_breaks_BTCUSDT*.csv` and the README's
   figure), committed by earlier work for another purpose; the daily bars it
   read are on disk. Its 2025 alarms are therefore public. They were not looked
   at for this registration; the only thing seen, while listing which files
   carry 2025 dates, was a count: one of the three breaks that file lists falls
   in 2025. Because it is public, that run cannot place the break blind. It is
   reported beside R and never used to define the break date; the detector
   that does is a different one, on hourly bars, never run on 2025
   ([The break detector](#the-break-detector)). The same figure, and the daily
   bars on disk, show BTC's 2025 daily price path, and its broad shape (a
   rising year, the reversal from October) is general knowledge; the author
   knew it in that sense when the blocks were set.
8. **Nothing else.** The repository's Bybit books end on 2024-05-10, and the
   daily bars above are its only 2025 data. No 2025 book, print, hourly bar or
   funding rate has been fetched or opened, and no 2025 number has been
   computed for this study.

---

## Expected outcome, stated in advance

**Every primary is expected to be killed at the base fee, and Z is expected to
be killed too.** The arithmetic:

- A taker strategy pays its fee on the notional of each side, and turnover
  counts both sides, so the fee at which it breaks even, per side, equals its
  gross profit in basis points of turnover.
- The original's own summary corresponds to a gross profit of about
  **0.36 bp of turnover**, at zero fees. That is its break-even fee per side.
- Bybit's BTCUSDT taker fee is 5.5 bp at the base tier, and **1.5 bp at the
  cheapest published tier** (`pro6_g1`, from 5,000 million USDT of 30-day
  volume). 0.36 bp is about a quarter of the cheapest taker fee and about a
  fifteenth of the base one.
- At the original's gross edge, a taker round trip on the clip would net
  2 × (0.36 − 5.5) = **−10.3 bp** at the base tier and
  2 × (0.36 − 1.5) = **−2.3 bp** at the cheapest one.
- Entering passively does not close the gap on paper: with a 0.0 bp maker
  entry (published for BTCUSDT only from Supreme VIP or Pro 5) and the cheapest
  taker exit, a round trip pays 1.5 bp, so the break-even is 0.75 bp of
  turnover, twice the original's edge. At the base tier, 2.0 + 5.5 = 7.5 bp a
  round trip, 3.75 bp of turnover.
- The 0.36 bp is itself an upper bound: thresholds chosen on the scored
  window, a closed fill engine, and a zero-fee spot venue all bias it upwards.
  This repository found that a directional signal's gross edge inverts when it
  is entered passively (+3 to +8 bp crossing, −0.5 to −4.2 bp resting), which
  is why Z, the gross edge under an honest queue, is also expected to be
  non-positive.

Predictions, one per hypothesis, fixed now: T1, T2 and M1 killed by their K1;
Z killed by its K1; R not tested, because nothing reaches it; FILL as stated
under its heading; A0's gross edge on public data below 0.36 bp of turnover
under every fill rule except fill-on-touch.

---

## Data

| Plane | Source | Code | Blocks |
|---|---|---|---|
| Ten-level book on a 100 ms grid | Bybit order-book archive, BTCUSDT linear perpetual, 500 levels (`ob500`) | `data/bybit.py`, `ARCHIVE_URL` | W, D, H |
| the same | the same archive at 200 levels (`ob200`), same path with `ob200` in the name | `data/bybit.py`, a second name to be added | F, P |
| Prints with aggressor side | Bybit trade archive, `public.bybit.com/trading/BTCUSDT/` | `data/bybit_trades.py`, `ARCHIVE` | all but E |
| Funding | Bybit public REST endpoint, no credentials | `data/bybit_funding.py` | all but E |
| Hourly bars for the break detector | Binance USDT-M futures BTCUSDT 1h klines, `data.binance.vision`; daily archives only for every day after 2025-06-30 | `data/binance.py`, with a daily-only option to be added | from 2024-12-01, E included (bars only) |

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

The same check on **every day** from 2024-12-01 to 2025-12-31 (396 days):

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

**What the archive change moved.** The plan before the check ended H on
2025-09-21 and began F on 2025-09-22. H would then have mixed the two archives,
and every primary verdict would have rested partly on a book source that D
never saw. Before anything else in this registration was written, the blocks
were adjusted: **H now ends on 2025-08-20**, the last `ob500` day, and **F
begins on 2025-08-21**, the first `ob200` day. D and H, where every value is
chosen and every primary verdict is taken, share one archive; F and P share
the other. The archive change falls on the H/F boundary, where it is visible,
instead of inside a block. Its consequences:

- the data pull request adds the `ob200` name to `data/bybit.py` for days from
  2025-08-21, with a test on a synthetic `ob200` fixture and on the three
  **reader-test days, 2026-01-05, 06 and 07**, which lie outside every block
  and are never used for anything else (no successor study may put them in a
  block). No `ob200` day from 2025-08-21 to 2025-12-31 is fetched before
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
  no status changes on it (the [status rules](#status-rules-family-of-three-primaries-t1-t2-m1)
  say what follows for H, F and P);
- the downloader records each F and P day's message count, median interval
  and sequence gaps only inside that block's opened access, as it records
  row counts;
- every comparison across the H/F boundary (a strategy's persistence from H
  into F) is labelled as crossing an archive change;
- R compares days before and after the break **within F**, on one archive,
  and the break date comes from Binance bars, which the change cannot move;
- the order-book CUSUM reported beside R (below) labels any alarm within a day
  of 2025-08-21 as "at the archive change".

**Differences from the original setup**, declared now: Bybit, not Binance; a
linear perpetual, not spot; USDT, not FDUSD; published fees, not zero; a 100 ms
grid rebuilt from the venue's archive, not the firm's sampler (their
equivalence is not known); funding charged; fills from this repository's
event-time simulator, not a closed engine.

---

## Blocks

| Block | Dates (UTC, inclusive) | Days | Book archive | Use | Read |
|---|---|---:|---|---|---|
| **W** warm-up | 2024-12-01 to 2024-12-31 | 31 | `ob500` | normalisers, size floors, rolling statistics; never scored | freely |
| **D** development | 2025-01-01 to 2025-06-30 | 181 | `ob500` | features, models, every searched value, the walk-forward results, σ of daily results, the minimum detectable effects, the detector's alarm rate; A0 and FILL on O | freely, through the development access |
| O (inside D) | 2025-02-04 to 2025-02-26 | 23 | `ob500` | the original backtest's window: A0 and FILL, descriptive only | as D |
| **E** embargo | 2025-07-01 to 2025-07-07 | 7 | `ob500` | a gap, so that H tests transfer rather than continuation | book, prints and funding never fetched; its Binance hourly bars only, for the detector, opened with H |
| **H** held out | 2025-07-08 to 2025-08-20 | 44 | `ob500` | T1, T2, M1, Z | once, after the D amendment is committed |
| **F** boundary | 2025-08-21 to 2025-11-16 | 88 | `ob200` | the break detector's alarms; every frozen strategy's numbers; persistence of anything that passed on H; R | once, after H's results are committed |
| **P** after | 2025-11-17 to 2025-12-31 | 45 | `ob200` | the last check of anything that passed (below) | once, after F's results are committed, only if the condition holds |

Why these blocks.

- **W** exists because the size floors and rolling normalisers must be fitted
  on data before the first scored day; December 2024 is never scored.
- **D** covers the first half of 2025, and with it O, the 23 days the original
  backtest was scored on. O is development data here: the replication on it
  is descriptive, and nothing on O is held out.
- **E** is a week no strategy reads, so that a model frozen on 30 June is
  first scored a week later. Only its hourly bars are read, by the detector,
  so that its stream has no hole.
- **H**, six weeks of July and August, is before the autumn the author
  remembers, so the primaries are tested where the recollection says the
  recipes still worked. It ends at the archive change.
- **F** contains the autumn, the 10–11 October cascade, and the archive change
  on its first day. It is where the frozen detector looks for the break.
- **P**, the last six weeks of 2025, is read only for a hypothesis that passed
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
amendment), whatever days they span; a day stays usable with fewer decision rows. Fitted
values (the size floors fitted on W, the frozen models and thresholds) are not
rolling state and are used on every block.

Days excluded from every verdict, counted and reported per block: more book
sequence gaps than `max_sequence_gaps`; a non-positive price, found as a
`positive_prices` error in `validate_book`'s report (the function returns its
findings and does not raise, so the study's loader excludes the day on that
finding); a price off the tick grid, which `validate_book` does not check
(the simulator's day loader, `market_making/events.load_day`, refuses such a
day under R16, and the study's loader applies the same check to every day,
taker strategies included); a missing plane (an HTTP 404 only; a failed
download is retried, never excluded, as the [fetch plan](#fetch-plan) says);
a missing funding history; or a day flagged `flatten_stale` (R16: a taker
order, an exit or the day-end close, that no snapshot followed before the
book ended walked a snapshot more than 5 s older than the order). **`max_sequence_gaps` is the 95th percentile of D's per-day gap
counts, rounded up**, computed by that formula and recorded in the amendment;
the same value is used on every block. Each verdict is also reported with the
gap-excluded days included, as a measurement, so that the exclusion cannot
quietly favour a result.

---

## Common settings

- **Instrument**: Bybit BTCUSDT linear perpetual (tick 0.1 USDT). ETHUSDT and
  XRPUSDT, the original's other instruments, are not part of this
  registration; using them needs a separate one, committed before they are
  fetched.
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
- **Clip**: 0.010 BTC a trade, the original's traded size, fixed for every
  strategy and both execution paths (T2 included: it is not Kelly-sized);
  results are also given per 100 USDT of clip a day, with the clip's notional
  taken at entry. One position at a time, never more than one clip (K-cap).
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
  order, the passive strategies' taker exits included; it is a measurement
  and excludes no day. An exit that no snapshot follows before the
  book ends walks the last one, and if that snapshot is more than 5 s
  (`suspend_after_pause_s`) older than the order the day is flagged
  `flatten_stale` and excluded, as in the simulator (R16). Funding (R14) and the accounting identity
  (R15), with turnover equal to the sum of fill notionals, hold for it as for
  the simulator. The label's walks and the executor's call the same function.
- **Passive fills** (M1, Z, FILL) come from round one's event-time simulator
  with these settings: a queue per order joining the tail of the visible size
  (R5), advance and fills only from prints at the price (R6) or through it
  (R7), proportional cancellation attribution, pro-rata arrival growth,
  trade-tape crossed policy, post-only, order and cancel latency 10 ms, feed
  latency 0, `soft_limit_clips` 1, `warmup_s` 60, quoting stops 23:58 and the
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
  (`flatten_slippage_bp` 0). The brackets (`none`, `all_ahead`;
  `pessimistic`) are run for K-queue.
- **Fees for verdicts**: the base tier, 2.0 bp maker and 5.5 bp taker
  (`BYBIT_BASE`). Other published BTCUSDT tiers (`BYBIT_LINEAR_TIERS`, group
  `g1`, down to 0.0 maker and 1.5 taker) and the programme's −1.0 bp maker
  rebate are measurements only, each with `BYBIT_FEE_CAVEAT`. A 0/0 schedule is
  used for A0 and Z only, and labelled.
- **Funding** is charged at each settlement on the position held then
  (R14); a day without its funding history is excluded.
- **The day**: no entry before 00:01 or after 23:58 UTC; anything open at
  23:59 is closed by a costed taker order. The accounting identity (R15) is
  checked at every day's end.

---

## Labels, features and models

**Labels.**

- **The walked-book taker label** (new code). At row `t` with hold `h`, for
  the clip: long P&L = (VWAP of selling the clip into the bids at the exit
  snapshot − VWAP of buying it from the asks at the entry snapshot) / the
  entry snapshot's mid × 1e4 − 2 × fee, the entry snapshot being the first at
  or after `t` + 10 ms and the exit snapshot the first at or after that + `h`,
  walked by the taker executor's function; short symmetric. Class `long` if
  the long P&L is positive, `short` if the short P&L is, `hold` otherwise. The
  fee is the base taker fee (5.5 bp) for T1, and 0 for A0 and Z. The label
  clip is the traded clip and the label horizon is the hold, so the label
  declares `h` and the purge is derived from it.
- **The forward move** for T2: the mid at the exit snapshot over the mid at
  the entry snapshot, − 1, in bp (a simple return), both snapshots as for the
  walked label; forward-only, unsmoothed. It is new code:
  `labels/targets.py`'s `forward_smoothed_move_bp` at `smoothing=1` gives the
  log of the mid's ratio from the decision row with its horizon in rows, and
  is not used.
- **The control label**: a time-based form (new code) of
  `labels/targets.py`'s `smoothed_move_bp` at `smoothing=20` rows: the log of
  the mean mid over the 20 rows ending at the first row stamped at or after
  `t + h`, over the mean mid over the 20 rows ending at `t`, × 1e4. Its
  horizon is in time, as every horizon here; only its smoothing is in rows.
  It is partly known at decision time and is used only to train the control
  models of K-label, one per hold in {5, 20, 50} s; their IC is reported at
  T2's chosen hold.

**Features.**

- **The microstructure set**: the registry's book, depth and flow features
  over ten levels, plus the 2025 set the features pull request adds where the
  registry lacks it (one-second realised volatility, return-sign
  autocorrelation, turn count, run length, inter-trade time, lagged level-0
  logs). Winsorising bounds come from trailing quantiles only.
- **The ten-level tensor**: 40 columns, each level's price distance from the
  touch over a rolling σ with a spread floor, and each level's size z-scored
  against a floor fitted on W only.
- Every feature passes `validation/leakage.py` before any D run (K-leak).
  Every gate input is in basis points computed from prices, never a
  normalised column (K-units).
- **Selection**: `features/selection.py`, once, on D's January, from the
  microstructure set, against one target: the walked-book label at 0 bp with
  `h` = 20 s as `y`, and the forward move at `h` = 20 s (as T2's, from the
  entry snapshot) as `forward`. `FeatureSelector` runs at its defaults
  (`score_target="ic"`, at most 40 features, correlation 0.95, variance 1e-6),
  so it ranks by absolute correlation with `forward`, and `y` only marks the
  rows that have a label. The selected list is used by T1, A0, Z and as T2's auxiliary
  input; it is recorded in the amendment.

**Models.** No hyperparameter of either model is searched; both use this
repository's defaults, not the 2025 research's.

- **Gradient boosting** (`models/gbm.py`), three classes, for T1, A0 and Z.
  With `balance_classes` (its default), a fit (a walk-forward fold or a
  frozen model) whose training rows lack the `long` or the `short` class gives
  that side a probability of 0 (`models/linear._align_columns` fills a missing
  class with zeros), so that side never trades with that model. This is the
  code's intended behaviour, not a fault; the fits and sides it affects are
  reported.
- **The Gaussian-head network** for T2: `models/tcn.py`'s causal dilated
  network (window 64 rows, 32 channels, 4 levels, kernel 3, dropout 0.1, at most
  20 epochs) with a head for μ and log σ², trained by Gaussian negative
  log-likelihood with a variance prior and a clamped variance; the checkpoint
  is chosen by the rank correlation of μ with the realised forward move on the
  inner validation.
- **The control model**: the same network trained on the control label.
- **Windows**: each of the network's windows is the 64 consecutive 100 ms
  rows ending at the decision row, in training and in inference alike.
  Training targets are taken on every tenth row (about one a second);
  decisions are made on every row. Windows are built per batch by a lazy
  window dataset (new code) and never materialised for a whole fit or day:
  `models/tcn._windowise` copies, and at 4 million rows of about 70 inputs a
  materialised fit would need about 72 GB.

---

## Strategies

| Strategy | Signal | Entry and exit | Fee | Role |
|---|---|---|---|---|
| **T1** | gradient boosting on the walked-book label at 5.5 bp; hold-first rule | taker in, taker out after `h` | base | primary T1 |
| **T2** | Gaussian-head network on the tensor and the selected set; EV gate | taker in, taker out after `h` | base | primary T2 |
| **M1** | the signal of T1 or T2 (chosen on D) | one full clip posted at the touch on the signal's side, kept at its first price, waits at most `h`; once filled, taker out `h` after the fill | base | primary M1 |
| **Z** | gradient boosting on the walked-book label at 0 bp; hold-first rule | as M1 | 0 / 0 | secondary Z |
| **A0** | as Z, at the original protocol's settings | taker, and as M1 under each fill rule | 0 / 0 | measurement A0, FILL |

**The hold-first rule** (T1, Z, A0). The model gives `p_long`, `p_hold`,
`p_short`. Go long when `p_long ≥ θ_long` and `p_long > p_short`; short
symmetrically; otherwise hold. `θ_long` and `θ_short` are set on the inner
validation so that each side would enter half of the target rate of trades a
day before the one-position rule.

**The EV gate** (T2). With μ and σ the predicted forward move over `h` in bp,
`c` = 2 × 5.5 bp, `w_t` the cost in bp of walking the current book for the
clip and back (from the snapshot, in bp of mid), and `β` the buffer in bp (not
the break date `b`): go long when `μ − kσ > c + w_t + β`, short when
`−μ − kσ > c + w_t + β`. One entry per event
(a run of rows on which the gate is open); a new entry needs the gate to close
and reopen. Every T2 trade is one full clip: no Kelly sizing is applied
(`strategies/sizing.py`'s `FractionalKelly` cannot size this gate in either
unit; see [Choices and their reasons](#choices-and-their-reasons)).

**Every strategy** holds one position at a time: the next entry is possible
only when the previous position is closed and no entry order is resting. For
T1 and T2 that is the hold after each entry; for M1 and Z it is the wait of
at most `h` for a fill, and after a fill the time to the exit.
Profit and loss comes only from realised fills in the execution mode, never
from a sum of predicted values.

**The passive variant with a passive exit** (M1 and Z, measurement only): the
exit is posted at the touch at `h` after the fill and crossed if still open
after a further `h`.

### Fixed now, and searched on D

| Value | Strategy | Setting | How set |
|---|---|---|---|
| hold `h` | T1, T2, Z | {5, 20, 50} s | searched on D |
| target trades a day | T1, Z | {50, 100, 240} | searched on D |
| `k` (σ multiple) | T2 | {0, 0.5, 1} | searched on D |
| `β` (buffer) | T2 | {0, 1, 2} bp | searched on D |
| M1's signal | M1 | T1 or T2, whichever has the higher mean of its daily `net_bp_trade` series over D's test days at its chosen cell (T1 on a tie); the one with a chosen cell, if only one has | chosen on D |
| A0's settings | A0 | 240 trades a day, `h` = 50 s, trained 2025-01-01 to 01-27, thresholds on 01-28 to 02-03, run on O | fixed: the original protocol's trade rate and its close-by-time hold (about 50 s, inferred from the research code) |
| clip, latency, fees, entry wait | all | as above | fixed |

The axes are 9 cells for T1, 27 for T2 and 9 for Z. Nothing else is searched,
and nothing else may be chosen after the amendment.

### The development-period procedure (D only)

1. **Fetch** W and D only ([fetch plan](#fetch-plan)); no H, F or P file is
   on disk before the amendment. Validate every day; exclusions per the
   [blocks](#blocks).
2. **Leakage**: every feature through `validation/leakage.py`; any failure
   stops the study until fixed by a dated amendment (K-leak).
3. **Selection** on D's January.
4. **Walk-forward over D** by calendar month: for each test month from
   February to June, train on D's days before it except the last seven, which
   are the inner validation for thresholds and checkpoints; purge the hold plus
   the longest feature lookback before the inner validation and before the
   test month. Every cell of every search runs in this walk-forward, which
   gives 150 test days per cell.
5. **The choice**: per strategy, the objective is the mean of the daily
   series (see [Metrics](#metrics)) over the 150 test days (net at the base
   tier for T1 and T2; gross under the queue rule for Z). The cell chosen is
   the one with the highest neighbourhood median among cells averaging at
   least 5 trades a day, ties broken by its own value. **The neighbourhood** is
   `validation/search.neighbourhood_scores`' block: every cell within one step
   of the cell on every axis, diagonals included (up to 9 cells for T1 and Z,
   27 for T2), every cell counted whatever its trade rate; the 5-trade filter
   only limits which cell may be chosen. The outright peak is reported beside
   it. **If no cell qualifies**, the hypothesis is recorded as `not run (no
   admissible cell on D)` and enters Holm with p = 1; the filter is not
   relaxed. If only one of T1 and T2 has a chosen cell, M1 uses its signal;
  M1 is not run if neither has.
6. **M1's signal** from the T1 and T2 choices; M1 run on D's test days for its
   σ.
7. **The frozen models**: for T1, Z, T2 and the control, one model per hold
   in {5, 20, 50} s, trained on D from 2025-01-01 to 06-23 with the purge (the
   hold plus the longest feature lookback) before 06-24; T2's and the
   control's checkpoint epoch per hold, and the hold-first thresholds
   `θ_long`, `θ_short` for every (hold, trades-a-day) cell of T1 and Z, set on
   06-24 to 06-30. These cover every cell of every neighbourhood, so K-nbhd on
   a held-out block runs only frozen files. The weights stay in the ignored
   `models/` directory; every weight file's sha256, every threshold and every
   checkpoint epoch go in the amendment.
8. **A0 and FILL on O**, the detector's alarm rate on D, σ of the daily
   results and the minimum detectable effects (below).
9. **The amendment**, committed before H is read, with the study's single
   entry point (`experiments/short_horizon_2025.py`), which computes every
   hypothesis, placebo, bracket, neighbourhood cell, tier and status for a
   block in one read session.

---

## Metrics

- **`net_bp_trade`**: for one round trip, (exit value − entry value) over
  the entry notional, × 1e4, less fees in bp of the entry notional, plus
  funding. **Primary for T1 and T2.** A day's value is the day's net in USDT
  summed over its round trips, over the sum of their entry notionals, × 1e4:
  basis points of traded notional (every trade is one clip, so the weights
  differ only with the price at entry). The daily series has one value per day with at least
  one trade.
- **`net_per_100_clip_day`**: **primary for M1.** Its daily series has one
  row for every usable calendar day of the block: the sum of that day's net
  in USDT × 100 / the clip's notional at each entry, and 0 for a day without
  a fill. The zero rows are written into the table given to `assess`, which
  averages only the rows it is given (`evaluation/significance.py`).
- **`gross_bp_trade`**: `net_bp_trade` before fees and funding. **Primary for
  Z**, with the day's value formed as for T1 (gross USDT over entry notional).
- **The daily series decides.** The mean, the day-level t, the Newey–West t,
  the median day, K1, K-days, the placebos, K-nbhd, σ_D and the minimum
  detectable effect are all computed on the hypothesis's daily series as
  defined above, never on trade rows. `assess` is given one row per day
  (`cluster="day"`, `series=None`). The degrees of freedom are the number of
  days in the series minus one, per hypothesis and per block.
- **`gross_bp_turnover`**: gross over turnover (both sides' notional), × 1e4:
  the fee break-even per side. Reported for every strategy against the base
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
  at every published BTCUSDT tier, the result split by side, and the counts
  of skipped entries, `entry_late_snapshot`, `exit_late_snapshot` and
  `flatten_beyond_visible_book`.
- **Statistics**: `evaluation/significance.assess` on the daily series; the
  day level decides. Score every calendar day of the block, with no row
  filtered or reweighted by label, outcome or model state. Beside every
  verdict, the **Newey–West t** of the daily mean (Bartlett weights, lag 5
  days), because consecutive days in one regime are not independent.
- **Minimum detectable effect**, recorded in the amendment for each primary
  and for Z: `(3 + 0.84) × σ_D / √n_eff`, the mean that the t ≥ 3 bar would
  catch with 80% power, with `σ_D` the standard deviation of the daily series
  over D's 150 walk-forward test days at the chosen cell and `n_eff` = 44 × the
  share of those 150 days with a trade (44 for M1, whose series has every
  usable day). On F (F− or all of F) and on P, the "below the minimum
  detectable effect" label uses the same formula, with σ_D and the share as
  frozen and the usable days of the series the status is assessed on in place
  of 44.

---

## Status rules (family of three primaries: T1, T2, M1)

**The bar.** A primary passes on a block when no kill condition fires and all
of these hold on its daily series ([Metrics](#metrics)):

- its one-sided day-level p-value passes Holm's procedure at 5% with
  **m = 3 on every block** (T1, T2, M1). A primary that is killed, void,
  inconclusive under K-trades, not run or not taken to the block enters with
  p = 1, so m never shrinks after a result is seen and a result on too few
  trades never lowers another primary's step;
- the day-level t is at least 3, its degrees of freedom being the days in the
  series minus one;
- the Newey–West t is at least 2;
- the median day is positive;
- the series has at least 20 days with a trade on H, and at least 10 on F
  (or F−) and on P (M1: usable days).

**Precedence**, the same on every block, so that each result has exactly one
status:

1. **void** — K-leak, K-units, K-acct or K-cap fires; counted as killed.
2. **inconclusive (too few trades)** — K-trades: fewer than 100 trades over
   the block (round trips for T1 and T2; fills for M1 and Z). Nothing below
   is evaluated.
3. **killed** — any kill condition fires. Rejection needs no significance.
4. **candidate** (on H) or **passes** (on F and P) — the bar is met.
5. **inconclusive** — otherwise; labelled "below the minimum detectable
   effect" when the mean is smaller in size than the MDE (only when no kill
   fired). Never reported as a positive.

A primary with no admissible cell on D is `not run`.

**F.** With `b` the first alarm of the frozen detector in F, F− is F's usable
days before the alarm's day and F+ its usable days after it; the alarm's own
day is in neither. **Statuses on F, kills included, are computed on F−** when
there is an alarm in F, F− has at least 20 usable days, R was not declared
untestable on D, and R is not void (below). Otherwise they are computed on all of F, labelled "assessed
on all of F", and F is not split. F+ enters only R. Taken to F: every
candidate, and every hypothesis inconclusive on H with a positive mean (for Z,
both claims positive).

| On H | On F (F− or all of F) | R (below) | On P | Status |
|---|---|---|---|---|
| candidate | passes | not supported | passes | **confirmed** |
| candidate | passes | not supported | does not pass, no kill | **candidate, not confirmed** |
| candidate | passes | supported | mean ≤ 0 | **confirmed until the break** |
| candidate | passes | supported | mean > 0 | **candidate; the break did not last** |
| candidate | does not pass, no kill | reported, not used | not read for it | **candidate, not confirmed** |
| inconclusive, positive | passes | not tested | passes | **confirmed on P, first significant on F** |
| inconclusive, positive | passes | not tested | does not pass, no kill | **exploratory: significant on F, not confirmed** |
| inconclusive, positive | does not pass, no kill | not tested | not read for it | **inconclusive** |
| either | a kill fires | — | not read for it | **killed on F** |
| either | passes | not supported or not tested | a kill fires | **killed on P** |

"Passes" on F and P is the bar above, with m = 3. "R untestable" and "R not
tested" are read as "not supported". For a hypothesis with R supported, P is
scored only as mean ≤ 0 (the break persisted) or > 0 (it did not last):
neither K-trades nor any kill condition is applied to it on P, but the void
conditions are (a void result on P is `void`), and a P series without a day
with a trade has mean 0, as a neighbour cell without a trade scores 0 in
K-nbhd, so it reads as the break persisted. A hypothesis that is `candidate`
after H keeps that status whatever F and P show unless a kill fires there; the
later reads add to it. **A hypothesis first significant on F gets no second
route to candidacy**: it is "exploratory: significant on F" until P, which is
its only confirmatory test; R is not tested for it, and the README never calls
it a candidate.

P is read once, after F's results are committed, only for hypotheses that
passed on F. F is read once, after H's results are committed, whatever H's outcome: its
detector alarms and every frozen strategy's numbers on F are reported as
measurements even when nothing is taken there. **For a hypothesis not taken
to F, F is reported as one whole block, never split at `b`.** Every
comparison from H into F crosses the archive change and is labelled so.

**An unreadable block** (more than a quarter of its days excluded,
[Blocks](#blocks)) changes no status. If H is unreadable, no hypothesis is
assessed: each is recorded as `not assessed (H unreadable)`, nothing is taken
to F, F is still read and reported as measurements, P is not read, and H, F
and P are spent all the same. If F is unreadable, statuses after H stand,
marked "F unreadable", and P is not read. If P is unreadable, each hypothesis
read on P keeps its status after F, marked "P unreadable": a candidate that
passed on F is **candidate, not confirmed (P unreadable)**, whatever R showed,
and one first significant on F is **exploratory: significant on F, not
confirmed (P unreadable)**.

**R void.** If a later run of the detector does not reproduce an earlier
run's alarms ([The break detector](#the-break-detector)), R is `void` and read
as not supported. Found when F is read, `b` does not split F and statuses on
F are computed on all of F, labelled. Found when P is read, F's statuses stand
as committed, and P is scored with the full bar, R read as not supported.

**No number at a fee tier other than the base, under a non-default fill rule
or exit, or on F for a hypothesis not taken to F, may be described as the
strategy working.** Such numbers are measurements.

**Z** is outside the family. It is tested alone at α/4 = 1.25%, one-sided,
at the larger of its two claims' p-values (round two's rule for a hypothesis
with two claims), with the t, Newey–West t, median-day and day-count
conditions holding for both claims; it follows the same precedence and table,
but its best possible status is **conditional on a zero-fee venue**: the
condition (0 bp on both sides) is stated as a number, and the base-tier
result, measured on the same trades, is reported beside it.

### Common kill conditions (every primary and Z, on every block it is read on)

Void conditions; the hypothesis is recorded as `void`:

- **K-leak** — any ported feature fails the leakage checker
  (`validation/leakage.py`, data mutated after a cutoff changes a value before
  it). Every result of the study is void.
- **K-units** — an input to the EV gate or to a cost is not in basis points
  computed from prices (a normalised column reached it). T2's result is void,
  and M1's if it uses T2's signal.
- **K-acct** — the accounting identity (R15) fails on any day, or turnover
  does not equal the sum of fill notionals.
- **K-cap** — |position| exceeds one clip at any event.

Kill conditions:

- **K1** — the mean of the hypothesis's daily series on the block is ≤ 0.
- **K-label** — the signal's IC (from the entry snapshot's mid, as defined
  under [Metrics](#metrics)) is ≤ 0 on the block (the model has no
  information about the move it trades).
- **K-placebo** — the true mean does not exceed: the 95th percentile of the
  200 random-direction draws; the sign-flipped signal's mean; the one-day
  shifted signal's mean; and, for M1 and Z, the stale signal's mean (below).
- **K-drift** — the true mean does not exceed the larger of the all-long and
  the all-short versions' means at the same entry times (the result is the block's
  drift, not the signal).
- **K-nbhd** — the mean of the daily series on the block, recomputed with
  each cell of the chosen cell's neighbourhood in place of the chosen cell,
  has a median ≤ 0 over the cells. The neighbourhood is
  `neighbourhood_scores`' block, diagonals included, every cell counted
  whatever its trade rate on D; a neighbour cell with no trade on the block
  scores 0. It runs only the frozen models, thresholds and checkpoints whose
  sha256 the amendment records. For M1, the neighbourhood is that of its
  signal's cell, each cell entered passively.
- **K-days** — half or fewer of the days in the daily series are positive
  (days with a trade for T1, T2 and Z; every usable day for M1).
- **K-latency** (T1, T2, M1, Z) — a positive result that is ≤ 0 when every
  order and cancel is live 110 ms after the decision instead of 10 ms (one
  more 100 ms row). A unit test checks that every fill's snapshot is stamped
  at or after its decision + 10 ms, and that the label's walks and the
  executor's use the same function.
- **K-fee** — the fee break-even per side (`gross_bp_turnover` for a taker
  strategy; for M1 the maker fee at which net is zero with the taker exit at
  the base fee) is at or below the base fee for that side. For a positive net
  at the base tier this can only fire through an accounting fault, so it is
  also a check; its main use is the report of how far each strategy was from
  paying.
- **K-queue** (M1, Z) — a positive result that is not positive under the
  joint pessimistic queue bracket (pessimistic cancellation attribution and
  `all_ahead` arrival growth), or under either `none` or `all_ahead` alone.

Not kills, but they set the status:

- **K-trades** — fewer than 100 trades over the block (round trips for T1
  and T2; fills for M1 and Z) makes the result inconclusive whatever its sign; it
  is applied before any kill (the precedence above).
- **K-MDE** — a result that does not pass, with no kill, and is smaller in
  size than the MDE is "inconclusive, below the minimum detectable effect".

Reported, not a kill:

- **The fill decomposition** (M1, Z; called K-dir in round one) —
  `spread + adverse(5 s)` over the passive fills, and the rest. A directional
  signal entered passively is meant to earn the move held after the fill, so
  a non-positive part at the fill with a positive result is the mechanism
  under test, not a fault; it is reported beside every M1 and Z verdict.

Each condition is applied to every claim of a hypothesis. A condition that
needs a positive result (K-queue, K-latency) fires only when the
result is positive under the default rules. On F the conditions are applied
to the series the status is assessed on (F− or all of F).

### Placebos

Each is re-run through the same execution, fee and accounting as the true
strategy, on the same block.

| Placebo | What changes | Used by | Seeds |
|---|---|---|---|
| sign-flipped | the signal's direction reversed at the same entry times | T1, T2, M1, Z | — |
| random direction | the same entry times, each direction drawn at random | T1, T2, M1, Z | 200 draws, 0–199 |
| one-day shift | the signal tape shifted by one day at the same time of day, circularly within the block | T1, T2, M1, Z | — |
| stale | the signal delayed by one hold | M1, Z | — |
| all-long, all-short | the same entry times, one direction (K-drift) | T1, T2, M1, Z | — |
| control model | the network trained on the control label; its IC against its own label and against the realised move, reported beside K-label | T2 | the training seed |
| other break dates | `b` replaced by every admissible split day of F (at least 20 usable days before it, at least 10 after), enumerated in full; the reference set keeps those at which the hypothesis also passes on the days before them | R | none; enumerated |

---

## T1 — the boosted classifier, crossing, at the real fee

**Statement.** On H at the base tier, T1 (gradient boosting on the walked-book
label at 5.5 bp, the hold-first rule at the chosen trade rate, a taker exit
after the chosen hold) earns a positive `net_bp_trade`.

**Kill conditions.** K1 (mean of the daily `net_bp_trade` series over H
≤ 0), K-label, K-placebo, K-drift, K-nbhd, K-days, K-fee, K-latency; void
under K-leak, K-units, K-acct, K-cap; inconclusive under K-trades.

**Prediction.** Killed by K1.

## T2 — the Gaussian network with an expected-value gate

**Statement.** On H at the base tier, T2 (the network on the tensor and the
selected set, the EV gate `μ − kσ > 11 bp + walk cost + β`, one entry per
event, one clip a trade, a taker exit after the chosen hold) earns a
positive `net_bp_trade`.

**Kill conditions.** As T1. Reported beside: the control model's IC against
its own label and against the realised move, which shows on public data how
much of a smoothed label is known at decision time.

**Prediction.** Killed by K1, or inconclusive under K-trades: an 11 bp round
trip on BTC moves of a few basis points over a minute may leave the gate
almost always shut.

## M1 — the same signal, entered passively

**Statement.** On H at the base tier, M1 (the chosen signal, one clip posted
at the touch on its side and kept at its first price, filled from prints after
the queue ahead under rules R5–R7, a taker exit `h` after the fill) earns a
positive `net_per_100_clip_day`, on the zero-filled series of every usable
day.

**Kill conditions.** As T1, plus K-queue. Reported beside: the fill
decomposition, the fill ratio, markouts by fill path, the passive-exit variant, and the result at
maker fees of 0 and −1 bp (the cheapest published BTCUSDT maker tier and the
programme's rebate), each with the fee caveat.

**Prediction.** Killed by K1.

## Z — the gross edge, at a zero-fee venue (secondary, conditional)

**Statement.** On H, under the queue fill rule, Z (gradient boosting on the
walked-book label at 0 bp, the hold-first rule at its chosen rate and hold,
posted at the touch, a taker exit `h` after the fill) has a positive
`gross_bp_trade`, and its net with fees at zero on both sides (gross plus
funding) is positive too: it would pay at a venue that charged nothing. Binance
offered zero-fee trading on the spot BTC/FDUSD pair in 2025, as a public
promotion; that is the context for this hypothesis, not a test of it (its
terms are not verified here, and the archives this repository reads hold no
order book for that pair).

**Kill conditions.** K1 (mean of the daily `gross_bp_trade` series over H
≤ 0); K2 (the mean of the daily net at 0/0, gross plus funding, ≤ 0);
K-label, K-placebo (with the stale signal), K-drift, K-nbhd, K-days,
K-queue, K-latency; void under K-leak, K-acct, K-cap; inconclusive under
K-trades. K-fee does not apply: the
hypothesis is about a fee the base tier does not offer.

**Label, fixed now.** Z is conditional on a fee schedule of zero on both
sides. No published Bybit tier offers it: BTCUSDT's lowest published taker fee
is 1.5 bp. A Z result is never reported without the same trades' net at the
base tier and at the cheapest published tier beside it, and never as a result
a Bybit account could trade.

**Prediction.** Killed by K1: resting entries are adversely selected, and the
repository's maker finding is that this inverts a crossing signal's gross
edge.

## R — the break (conditional)

**Tested for** every hypothesis, Z included, that is a candidate on H,
whether or not it passes on F−; its statistic is computed and reported in
either case, with F−'s result beside it. Not tested for any other
hypothesis, including one first significant on F; recorded as `not tested`,
with the reason, and read as not supported.

**Statement.** With `b` the first alarm of the frozen detector in F, and
Δ(`d`) the mean of the daily series on F's usable days before split day `d`
minus its mean on F's usable days after `d`: R is **supported** if the
hypothesis passes on F−, its mean on F+ is ≤ 0, and Δ(`b`) exceeds the 95th
percentile of Δ(`d`) over the **reference set**; **not supported**
otherwise.

**The reference set** is every admissible split day `d` of F (at least 20
usable days before it and at least 10 after; about 58 days), enumerated in
full, at which the hypothesis also passes the bar on F's days before `d`.
**The whole bar is evaluated at each `d`**, kills and Holm included, as on F−
with F's usable days before `d` in its place: the other primaries taken to F
are evaluated there the same way and enter Holm with p = 1 where they are
killed, void or inconclusive under K-trades on those days, as do those not
taken to F, and K-trades counts the trades before `d`. Nothing is re-run for
it: every series the bar reads (the strategy's daily values and IC, its 200
random-direction draws and other placebos, its neighbour cells, its 110 ms and
bracket re-runs) is computed once per day over all of F in F's read session,
the one-day shift circular within all of F as on F−, and each split day only
slices those per-day values.
Passing on F− selects lucky days before the true split, so the placebo splits
go through the same selection; a comparison with unselected splits would make
R come out supported too often. When the hypothesis does not pass on F−,
Δ(`b`) is reported with its percentile among all admissible days, as a
measurement.

**Untestable**, declared and not retuned, and read as not supported: no
alarm in F; fewer than 20 usable days before `b` or fewer than 10 after it;
the detector's alarm rate on D above one per 14 days (declared in the
amendment, before H is read; `b` then does not split F); or fewer than 20
days in the reference set.

**Reported beside**: every alarm in H and F with the channel that moved
(scale or dependence, and the mean channel's odds), the second detector's
alarms, and the earlier
daily-bar run's alarms (item 7 of [what was known](#what-was-known-before-this-was-written)),
labelled as already public. The blocks are not moved by any alarm, including
alarms inside H.

**What R can establish** is that a policy fixed in advance, "stop at the first
alarm", separates days on which a strategy paid from days on which it did not;
not why it stopped.

---

## Measurements without a hypothesis

None enters a verdict, and none can be quoted as a result.

**A0, the original recipe on public data, on O.** Gradient boosting on the
walked-book label at 0 bp, trained on 2025-01-01 to 01-27, thresholds for 240
trades a day on 01-28 to 02-03, a 50 s hold, run on O (2025-02-04 to 02-26):
crossing at 0/0, and posted at the touch under each of the three fill rules of
FILL. Reported: `gross_bp_turnover` against the original's 0.36 bp, net at the
base tier and at every published tier, trades a day, IC. The table is
labelled **not comparable: different venue, product, quote currency and
fee**. Prediction: below 0.36 bp under every rule except fill-on-touch.

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
0.50 under (c).** The original's closed engine reported about four in five
orders filled; if (c) also fills at least 0.75, the suspicion that the
original's fill rule was optimistic is withdrawn in writing in the results.
The same three rules are reported for Z's stream on H.

**The fee break-even**, per strategy and block: `gross_bp_turnover` for the
taker strategies; for M1 and Z, the maker fee at which net is zero with the
taker exit at the base fee and, separately, at 1.5 bp. Compared with every
published BTCUSDT tier, with the fee caveat.

**H before the research's later development data**, a labelled sensitivity:
T1 and T2 at their frozen cells on H's usable days from 2025-07-08 to 07-31
(24 days), the part of H that precedes every dated entry of the 2025
research's later development ([what was known](#what-was-known-before-this-was-written),
item 1). Same daily series and statistics, no status.

**Verdicts with the gap-excluded days included**: each verdict's mean, t and
median day recomputed with the days excluded for sequence gaps put back.

---

## The break detector

Fixed now, run as the ledger opens each block.

- **Input**: hourly log returns, close to close, of Binance USDT-M futures
  BTCUSDT 1h klines, each divided by the root mean square of D's hourly
  returns at the same UTC hour (a 24-value profile fitted on D only), so that
  the daily cycle of volatility is not read as a series of breaks.
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
  the earlier runs; if it does not, R is void and read as not supported (see
  "R void" under the status rules).
- **The mean channel** is read from the `odds_mean` column of the same run,
  against 8.93, and reported. It is not in `statistics`, so it never
  re-anchors the walk and never defines `b`.
- **`b`**: the first alarm effective in F, effective at the close of the hour
  whose return produced it. F− is F's usable days before the alarm's day, F+
  its usable days after it; the alarm's own day is in neither.
- **On D**, the monitor's alarm rate over 2025-01-01 to 06-30 is recorded in
  the amendment. Above one alarm per 14 days, R is declared untestable then,
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
  **restarts on H's first day**, because E's book is never fetched, and runs
  continuously from H through F and P, with any alarm within a day of
  2025-08-21 labelled "at the archive change". Also the earlier daily-bar run,
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
day is. Only an HTTP 404 makes a missing plane; a 404 on a day the `HEAD`
check found present is reported as such. Funding and hourly bars follow the
same rule: a failed request is retried, and only the venue's answer that it
has no data (an HTTP 404, or an empty funding history) is a missing plane.
The print downloader's 120 s timeout (`data/bybit_trades.download_day`)
bounds each socket read, not the whole file, so a large file does not fail by
its size; a stalled connection does, and is retried.

| Stage | When | What | Book archives (GB) | Print archives (GB) |
|---|---|---|---:|---:|
| 1 | after this registration and the code it names are merged | W and D (212 days, `ob500`), funding for W and D only, hourly bars 2024-12-01 to 2025-06-30; the three `ob200` reader-test days of 2026 | 76.0 | 15.3 |
| 2 | inside H's opened access, after the amendment is committed and the ledger has recorded the open | H (44 days, `ob500`), H's funding, E's and H's hourly bars from daily archives | 11.4 | 2.1 |
| 3 | inside F's opened access, after H's results are committed | F (88 days, `ob200`), F's funding and hourly bars from daily archives | 15.8 | 5.9 |
| 4 | inside P's opened access, after F's results are committed, only if the condition holds | P (45 days, `ob200`), P's funding and hourly bars from daily archives | 9.9 | 3.5 |

**The ledger opens before the fetch.** `open` writes its entry first, and the
block's fetch then runs under that access; the entry records each fetched
file's sha256 and UTC download time. `open` refuses if any book, print,
funding or hourly-bar file of that block is already on disk, and H's `open`
also if any hourly bar of E is.

E's book, prints and funding are never fetched; its hourly bars are fetched
with H's. At most one archive per worker is on disk at once; the largest book
archive of W and D is 0.92 GB (the median 0.34 GB), so under 4 GB with four
workers. Replayed at ten levels, a BTCUSDT day took about 35 MB in the 2024
books, so W and D take about 7 GB and the whole year about 14 GB. The print
archives come to 26.8 GB for the year, and the parquet written from them is
budgeted at the same size.

**Disk.** 121 GB were free on the machine when this was registered. The
feature store holds model inputs only, **as float32, one zstd-compressed
parquet file per day**; labels, the gate's walk cost and every fill are
computed from the book's float64 prices, never from the store. At about
864,000 rows and about 100 columns a day is about 345 MB before compression,
budgeted at 250 MB: about 53 GB for W and D, 11 GB for H and 33 GB for F and
P. With the books, the prints and the archives in flight, keeping everything
would need about 140 GB, so **W's and D's features are deleted after H's
results are committed and before stage 3**; no later read uses them (F and P
run only frozen models), and they can be rebuilt from the books. Before each
stage, the fetch checks that free disk exceeds the stage's budget (its
archives in flight, books, prints and features) by 10 GB, and refuses to start
otherwise. The amendment records the measured sizes.

---

## Compute budget

On the machine the market-making rounds ran on (10 cores, 32 GB), with
8 worker processes (at most 4 for the book replay), each heavy run started under `nohup caffeinate` with its
log in `logs/`.

| Run | Size | Estimate |
|---|---|---|
| Download, stage 1 | 91.3 GB of archives (76.0 book, 15.3 print) | 0.5–2.5 hours at 10–50 MB/s, overlapping the replay |
| Replay W and D | 212 days at 2–3 minutes of replay and up to a minute of download a day (the 2025 archives are larger than the 90 s measured on 2024's), on at most 4 workers | about 2–3.5 hours |
| Download and replay, stages 2–4 | H 13.5 GB (44 days), F 21.7 GB (88 days), P 13.4 GB (45 days) | about 1, 2 and 1 hours |
| Features W and D | 212 days | 2–4 hours |
| Boosting (T1, Z, A0) | 3 holds × 5 folds + 3 frozen fits, each for T1 and Z, plus A0 | 4–10 hours |
| Network (T2, control) | 3 holds × 5 folds + 3 frozen fits, plus the control | 8–24 hours |
| Simulator on D (Z's 9 cells, M1, FILL on O) | about 1,600 configuration-days | 4–12 hours |
| R's reference set, on F | about 58 split days per candidate, the bar on slices of per-day values already computed on F | minutes, inside F's read session |
| First held-out read (H): every strategy, placebos (200 random draws each for M1 and Z through the simulator), brackets, K-latency re-runs, neighbourhoods, tiers | about 25,000 configuration-days | 1–3 days, as one read session |

Every fit reads at most 4 million training rows, taken evenly in time, so a
fit's cost does not grow with D. The network's windows are built per batch,
never materialised for a fit or a day. The amendment records the measured times and
the largest worker's peak memory; a run that is slower than estimated takes
longer, and nothing registered is dropped to save time.

---

## Who may read what, and the code that will enforce it

The pull requests that follow this one add, before stage 1:

- the ported pieces (the tensor, the microstructure set, the walked-book
  label, the Gaussian head, the hold-first rule, the EV gate, the fill-rule
  comparison), each under the leakage checker and with streaming-versus-batch
  parity tests;
- the `ob200` archive name in `data/bybit.py`, proven on a synthetic fixture
  and the three reader-test days ([Data](#data)), and the daily-only fetch of
  hourly bars in `data/binance.py`;
- the event-time taker executor, the simulator's signal-entry quoter and
  fixed `clip_btc` setting, the lazy window dataset, and the Newey–West t in
  `evaluation/significance.py`;
- **this study's fetch and feature store** ([fetch plan](#fetch-plan)): the
  book replay on at most 4 workers with its measured peak memory, the retry
  of every failed download until success (only an HTTP 404 is a missing
  plane), the float32
  per-day feature store, the disk check before each stage, and the deletion of
  W's and D's features before stage 3;
- **this study's loader, ledger and single entry point**, on the pattern of
  `market_making/round2.py`, with block constants of its own (round one's
  `block_of` is not used): the YAML validated against this registration,
  placeholders hashed as in the earlier rounds, and an access that permits
  days per block and plane. The entry point, `experiments/short_horizon_2025.py`,
  is committed with the amendment and computes everything a block's read
  produces, statuses and kills included, in one read session.

**The development access** permits W and D (book, prints, funding), the
hourly bars up to 2025-06-30 and the reader-test days. It refuses every 2025
day after 2025-06-30 in every plane, including the daily bars already on disk
(`data/bars/BTCUSDT/klines-1d`); a read of those after that date falls under
the ledger rule below.

**The ledger**, `experiments/results/short_horizon_2025_ledger.json`, guards
E's bars, H, F and P.

- **H** opens only if the amendment records the YAML's sha256, the file on
  disk has it and is committed, the tree is clean, every frozen weight file's
  sha256 equals the amendment's, and no file of H and no hourly bar of E is
  already on disk.
- **A read session** is bound to (block, commit, config sha256). Any number of
  processes and resumptions at that commit count as the first read; outputs
  are not inspected until the session writes the block's results file. A
  resumption at another commit is a second read.
- **The session closes** by writing the block's statuses and every kill,
  computed by the entry point, to `experiments/results/short_horizon_2025_H.json`
  (`_F.json`, `_P.json` for the later blocks), and that file's sha256 into
  the block's ledger entry.
- **F** opens only if H's results file is committed unchanged (its sha256
  equals the ledger's), `git diff --name-only <H read's commit>..HEAD` lists
  only `README.md` and paths under `experiments/results/` and under `docs/`
  outside `docs/preregistration/` (so the code, the configuration, this
  registration, the tests, the entry point and the lock file are those of the
  H read), the frozen weights' sha256 still match, and no file
  of F is on disk. **P** opens on the same conditions against F's read, and
  only for the hypotheses that qualify.
- Each first read writes block, commit, config sha256 and UTC time before the
  access is returned; a second read raises unless forced, and everything it
  produces is stamped `second read` and cannot change a status.

**H, F and P are spent** for these recipe families on BTCUSDT. A successor
registration, a corrected re-run after a bug included, uses blocks after
2025-12-31 and never the reader-test days.

Round one's `prereg.Access` and the round-two access are not used to read any
day of this study. `pipeline/execution`'s market-maker path builds its access
with round one's `access_for`, so it is not used either: the passive
strategies call `market_making/simulator` directly, on days loaded through
this study's loader.

---

## Amendment protocol

After the development runs, and before H is read, one dated amendment is
appended and committed: the days excluded and the sequence-gap threshold
(by its formula); the longest feature lookback `N`; the reader-test days'
message counts, intervals and gaps; the selected features; for T1, T2 and Z
the chosen cell with its neighbourhood and the outright peak; M1's signal with
the D values that chose it; σ of each daily series on D and the minimum
detectable effects; the A0 and FILL tables on O; the detector's alarm rate on
D and, if it applies, the declaration that R is untestable; the sha256 of
every frozen weight file (one per hold for T1, Z, T2 and the control), every
threshold and every checkpoint epoch; the entry point, committed with it; the
compute record, with the replay's measured peak memory and worker count and
the feature store's size; the fits that lacked a class, with the side; and
the sha256 of `configs/short_horizon_2025.yaml` after
the frozen values are written into its placeholders, on a line of the form
``Frozen configuration sha256: `<hash>` ``. Nothing else may change. Each
frozen value has one placeholder: weight sha256s under
`models.weights_sha256`, checkpoint epochs under `models.checkpoint_epoch`,
hold-first thresholds under `strategies.<X>.thresholds`; a neighbourhood
lists its cells, their D values and the outright peak, and refers to those
by cell without repeating them.

A correction found after this registration and before H is read (a bug in a
ported feature, the label, the simulator or a reader) is made only by a
further dated amendment that states it and, if it could change a development
number, re-runs D in full before anything is frozen. After H is read, nothing
is changed and nothing is re-run under a different rule; the ledger's
conditions for opening F and P enforce this for the code, the configuration,
this document and the frozen weights.

**The ledger rule.** Only reads through the ledger count. A number computed on
H, F or P outside it — in a notebook, a script run by hand, or a second read —
is not a result, cannot change a status, and is reported as such if it is
reported at all. A negative verdict on H ends the regime question for that
hypothesis: F is still read and reported, but nothing is taken to P.

---

## What this study cannot establish

- **Whether the original result was real.** Venue, product, quote currency,
  fee, sampler and fill engine all differ (listed under [Data](#data)). A
  negative here does not refute the original on its own venue, and a positive
  would not validate it.
- **A fee schedule of 2025.** Bybit's tiers are the ones published in 2026,
  applied to 2025.
- **Queue dynamics inside 100 ms.** The book is a 100 ms photograph; queue
  position is a rule, bracketed, and K-queue requires a verdict to survive the
  pessimistic side of the bracket, not the truth.
- **Anything about capacity.** The clip is 0.01 BTC and own impact is not
  modelled.
- **Anything beyond one instrument and one year.** ETHUSDT and XRPUSDT are
  outside this registration.
- **That H and F are comparable.** The book archive changes on F's first day;
  a difference between H and F may be the archive.
- **That H and F are fresh for the recipes.** The 2025 research developed
  its recipes on dates that overlap H's last 20 days and all of F (item 1 of
  [what was known](#what-was-known-before-this-was-written)); they are fresh
  only for the values this study sets on D. P is the one held-out block after
  every dated entry of that research.
- **Why a strategy stopped.** R tests a causal stopping policy on a detector
  of BTC's returns; it does not identify the mechanism, and it does not test
  the author's recollection, which stays a recollection.

---

## Choices and their reasons

The registration was reviewed three times before it was merged and before any
data was read, for peeking, for multiplicity and for feasibility. Every
blocking and should-fix point was taken in, and the notes where they
pointed to a gap. Where the reviews proposed different fixes, a fix left a
value open, or a proposal was not taken, the choice and its reason:

- **Holm's m is 3 on every block**, not the number taken to F or P: the
  stricter of the two proposals, and fixed before any read.
- **A missing hour carries its close with a zero return**, rather than being
  dropped: one row per hour keeps the windows in hours and the hour-of-day
  profile aligned, and the move over the gap enters the next return either way.
- **R is tested for every candidate on H, with a reference set of split days
  at which the hypothesis also passes**: testing on H's candidacy removes the
  gate on F−, and the conditioned reference set removes the selection that
  "confirmed until the break" still makes. The split days are enumerated
  (about 58), so the floor is 20 qualifying days, not 50 of 200 draws.
- **A result first significant on F is "exploratory" and P is its only
  confirmatory test**, rather than α being split between H and F: H stays the
  only gate to candidacy.
- **Z is tested alone at α/4**, not as a fourth member of the family, so that
  a secondary cannot change the primaries' bar.
- **The passive entry is kept at its first price**, not re-pegged: no
  re-peg cadence is left to choose, and the order keeps its queue place.
- **The clip is fixed at 0.010 BTC** (`clip_btc`), rather than K-cap being
  measured against the clip in force at entry: it keeps the original's traded
  size and gives K-cap one meaning.
- **Three reader-test days in January 2026**, not one: more cadence and gap
  data for the `ob200` replay, all outside every block.
- **H's and F's first days start cold**, rather than carrying rolling state
  across E or across the archive change: no E day is read, and no `ob500`
  state is carried into `ob200` data.
- **K-latency also applies to Z**, which shares M1's execution path.
- **K-trades is applied before the kills**: a result on fewer than 100 trades
  is inconclusive whatever its sign, so a handful of losing trades neither
  kills nor passes a hypothesis.
- **The mean channel is read from `odds_mean`**, the column `detect_breaks`
  writes; its internal name is `sr_mean`.
- **T2 trades one full clip, and Kelly sizing is dropped** (feasibility
  review). `FractionalKelly.scale(edge, volatility)` returns fraction × edge /
  volatility², clipped to [0, 2]. With μ and σ in bp, a gated trade with
  μ = 15 and σ = 10 gets 0.075 of a clip, below the 0.001 BTC lot, so T2 would
  be inconclusive by construction; in decimal units the same trade gets 750,
  clipped to the cap, so every trade would be one clip and Kelly would do
  nothing. A scaling between the two would be a value set without a basis,
  and the research's sizing values are not used. Without it, T2's trade count
  is its gate's, and K-trades counts round trips as for T1.
- **Feature selection ranks by correlation with the forward move at
  `h` = 20 s** (`FeatureSelector`'s default `score_target="ic"`), with the
  walked-book label at 0 bp as `y`. The feasibility review proposed the label
  at 5.5 bp; the 0 bp label, fixed after the multiplicity review, is kept.
  Under this ranking the label only marks which rows are usable, and it
  exists on the same rows at either fee.
- **The other values the feasibility review found open on D were already
  pinned** after the other reviews, and are kept: the control label's
  smoothing at k = 20 rows (one constant for the three holds, rather than k
  equal to the horizon in rows); the 5-trade filter, which limits only the
  chosen cell while every neighbour counts; `not run` with p = 1 for a
  strategy with no admissible cell; and `max_sequence_gaps` as the 95th
  percentile of D's per-day gap counts (rather than the smallest value that
  keeps 90% of D's days). Added: M1 uses whichever of T1 and T2 has a chosen
  cell when only one has.
- **The minimum detectable effect on F and P** uses the frozen σ_D and share
  with that series' usable days, rather than H's 44, so that the label on a
  later block describes the days it has. Its formula on H was already
  `(3 + 0.84) × σ_D / √n_eff`, as the feasibility review asked.
- **At most 4 replay workers**, rather than a replay that writes rows into
  preallocated arrays (feasibility review): the replay stays the code that
  wrote the 2024 books, and the cost is a few hours of wall time.
- **A float32 per-day feature store**, rather than features computed afresh
  on each run: every walk-forward fold and every cell reads the same D days
  again, and the store's 250 MB a day fits the disk once W's and D's features
  are deleted after H's results; float64 would not fit.
- **The fill ratio counts orders with any fill**, rather than filled volume
  over posted volume: it is the repository's existing `fill_rate`, and it is
  the measure closer to the original engine's "orders filled". A partial fill
  counts as filled, which makes FILL's prediction for the queue rule harder to
  meet, not easier; filled volume is reported beside it.
- **Only `validate_book`'s `positive_prices` error excludes a day**, as
  registered, rather than any error in its report: the registration named a
  non-positive price, and adding the function's other checks now would add
  exclusions that D has not been looked at for. The off-grid check moves to
  the simulator's day loader, where the code has it, and `flatten_stale`
  joins the exclusions as R16 has it.
- **K-dir is reported, not a kill, for M1 and Z** (multiplicity review,
  notes). In round one it guarded a market maker whose result should be
  earned at the fill; a directional signal entered passively is meant to earn
  the move after it, so the kill would remove the mechanism under test. It
  could only cause false negatives; the false-positive routes stay guarded by
  K-placebo (with the stale signal), K-drift, K-queue and K-latency.
- **The detector's alarm rate on D stays in-sample, and is disclosed as
  optimistic** (peeking and multiplicity reviews, notes), rather than
  measured with a profile fitted elsewhere: the leniency can only end R early
  or make it untestable, never make it supported.
- **No 2026 confirmation block is registered here.** P begins after every
  dated entry of the 2025 research, so it is fresh for the recipes; a
  confirmation on 2026 data needs its own registration, which the
  spent-blocks rule confines to dates after 2025-12-31 and which is committed
  before any 2026 book or print is fetched (the reader-test days excepted).
- **No entry is skipped for a late snapshot**, and late snapshots are counted
  at entry and exit instead (final consistency check). The registered "more
  than 1 s old" entry skip could not apply: decisions are made at row stamps,
  and the snapshot walked is never older than the order. Reading it as "the
  next snapshot comes more than 1 s late" would make a trade depend on the
  feed after the decision, at the feed gaps where a skip would select most.
- **P is read for every hypothesis that passed on F**, without the earlier
  "at least 20 usable days" condition: counted in calendar days it always
  held, and counted in days with a trade it left a hypothesis that passed on
  F with 10 to 19 such days without a status; the bar on F already asks for
  10.
- **A primary inconclusive under K-trades enters Holm with p = 1**: its
  p-value is not evaluated, and a small one would otherwise move another
  primary to a laxer step.
- **With R supported, an empty P series has mean 0**, as a neighbour cell
  without a trade scores 0 in K-nbhd: a strategy that does not trade after
  the break has not paid after it.

---

The configuration as registered, with every placeholder `null`, has sha256
`729f042bc55cae70fbdf786d8f38581b2539fc1a422a978d8bf9c674cee58fd8`.
(Earlier versions of this file on its unmerged branch had other hashes,
among them `e01ba377857e895b2d81d6a625434fb183210825ea906f2801e922eef7c93308`
before the reviews and
`c2da1566d9297499cd01e1d084e5bf6a0116b19a88d4dbccdf2b62ffae495262` before the
feasibility review's later points; they do not count.)
This study's loader will check that putting `null` back into every placeholder
of the frozen file gives this hash, as the earlier rounds' loaders do.

---

## Amendments

None yet.
