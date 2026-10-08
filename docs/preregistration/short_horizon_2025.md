# Pre-registration: the 2025 short-horizon study, re-run on public data

This document registers a study before any of its data is read. It is committed
together with [`configs/short_horizon_2025.yaml`](../../configs/short_horizon_2025.yaml),
which holds the same values in the form the code will read, and the two must
agree.

| | |
|---|---|
| **Committed** | 8 October 2026, against `main` at `af148b5`, before any 2025 book, print, hourly bar or funding file is fetched for it |
| **What it registers** | A re-run, on Bybit's public BTCUSDT archives, of the short-horizon order-book research the author did in 2025: the blocks, the data and its coverage, the labels, models and strategies, what is searched on the development block, hypotheses T1, T2, M1, Z and R with their metrics, kill conditions and placebos, the measurements A0 and FILL, the break detector, the status rules, the fetch plan, the compute budget and the amendment protocol |
| **Computed so far** | **Nothing.** No 2025 order book, print, hourly bar or funding rate has been fetched, opened or computed on. The only 2025 data on disk is the daily bars of an earlier, unrelated break-detector run (item 7 below). What was known is listed under [What was known before this was written](#what-was-known-before-this-was-written) |
| **How it changes** | Only under the [amendment protocol](#amendment-protocol): dated amendments appended at the end. One amendment is planned, after the development-period runs and before the held-out block is read |

The market-making rounds are
[`market_making.md`](market_making.md) and
[`market_making_round2.md`](market_making_round2.md); their conventions hold here
unless this document says otherwise. Names that do not exist on `main` at
`af148b5` — the ten-level book tensor, the walked-book taker label, the
Gaussian-head network, the hold-first rule, the EV gate, the fill-rule
comparison, this study's loader and ledger — refer to code added by the pull
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
   [Data](#data). The per-block sizes were seen; they reflect the archives'
   depth and activity, not prices, spreads or fills.
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
   ([The break detector](#the-break-detector)).
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
| Prints with aggressor side | Bybit trade archive, `public.bybit.com/trading/BTCUSDT/` | `data/bybit_trades.py`, `ARCHIVE` | all |
| Funding | Bybit public REST endpoint, no credentials | `data/bybit_funding.py` | all |
| Hourly bars for the break detector | Binance USDT-M futures BTCUSDT 1h klines, `data.binance.vision` | `data/binance.py` | from 2024-11-01 |

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
  2025-08-21, with a test that a replayed `ob200` day passes `validate_book`;
  the replay itself is unchanged;
- as each F and P day is replayed, the downloader records its message count
  and median interval between messages, as it records row and sequence-gap
  counts; nothing else of F or P is looked at before the ledger opens them;
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
| **D** development | 2025-01-01 to 2025-06-30 | 181 | `ob500` | features, models, every searched value, the walk-forward results, σ of daily results, the minimum detectable effects, the detector's alarm rate; A0 and FILL on R | freely, through the development access |
| R (inside D) | 2025-02-04 to 2025-02-26 | 23 | `ob500` | the original backtest's window: A0 and FILL, descriptive only | as D |
| **E** embargo | 2025-07-01 to 2025-07-07 | 7 | `ob500` | a gap, so that H tests transfer rather than continuation | never; not fetched |
| **H** held out | 2025-07-08 to 2025-08-20 | 44 | `ob500` | T1, T2, M1, Z | once, after the D amendment is committed |
| **F** boundary | 2025-08-21 to 2025-11-16 | 88 | `ob200` | the break detector's alarms; every frozen strategy's numbers; persistence of anything that passed on H; R | once, after H's results are committed |
| **P** after | 2025-11-17 to 2025-12-31 | 45 | `ob200` | the last check of anything that passed (below) | once, after F's results are committed, only if the condition holds |

Why these blocks.

- **W** exists because the size floors and rolling normalisers must be fitted
  on data before the first scored day; December 2024 is never scored.
- **D** covers the first half of 2025, and with it R, the 23 days the original
  backtest was scored on. R is development data here: the replication on it
  is descriptive, and nothing on R is held out.
- **E** is a week the code never reads, so that a model frozen on 30 June is
  first scored a week later.
- **H**, six weeks of July and August, is before the autumn the author
  remembers, so the primaries are tested where the recollection says the
  recipes still worked. It ends at the archive change.
- **F** contains the autumn, the 10–11 October cascade, and the archive change
  on its first day. It is where the frozen detector looks for the break.
- **P**, the last six weeks of 2025, is read only for a hypothesis that passed
  before it.

Days are independent: each starts flat, no position is held across midnight,
and the day is the cluster for every statistic. Rolling statistics may read
the previous days of the same or an earlier block (W before D, and so on); a
reader never reads a later block than its access permits. Days excluded from
every verdict, counted and reported per block: a book sequence gap
(`max_sequence_gaps` 0 unless the amendment sets another value from D's
counts), a non-positive price or a price off the tick grid (`validate_book`
raises), a missing plane, or a missing funding history.

---

## Common settings

- **Instrument**: Bybit BTCUSDT linear perpetual (tick 0.1 USDT). ETHUSDT and
  XRPUSDT, the original's other instruments, are not part of this
  registration; using them needs a separate one, committed before they are
  fetched.
- **Grid and timing**: decisions on the 100 ms grid. A grid value enters
  event time at the end of its bin, as `data/grid.to_grid` labels a bin by its
  start and carries its last value. Orders and cancels are live 10 ms after
  the decision (simulator rule R3).
- **Clip**: 0.01 BTC a trade, the original's traded size; results are also
  given per 100 USDT of clip a day, with the clip's notional taken at entry.
  One position at a time, never more than one clip (K-cap).
- **Taker fills** walk the first snapshot at or after arrival across the ten
  visible levels, at their volume-weighted price for the clip. A trade whose
  clip exceeds the visible depth, or whose last snapshot is more than 1 s old,
  is skipped and counted.
- **Passive fills** (M1, Z, FILL) come from round one's event-time simulator,
  unchanged: a queue per order joining the tail of the visible size (R5),
  advance and fills only from prints at the price (R6) or through it (R7),
  proportional cancellation attribution and pro-rata arrival growth, post-only,
  10 ms latency. The brackets (`none`, `all_ahead`; `pessimistic`) are run for
  K-queue.
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

- **The walked-book taker label** (new code). At grid row `t` with hold `h`,
  for the clip: long P&L = (VWAP of selling the clip into the bids at `t + h`
  − VWAP of buying it from the asks at `t`) / mid(`t`) × 1e4 − 2 × fee, both
  walks at the decision plus 10 ms; short symmetric. Class `long` if the long
  P&L is positive, `short` if the short P&L is, `hold` otherwise. The fee is
  the base taker fee (5.5 bp) for T1, and 0 for A0 and Z. The label clip is the
  traded clip and the label horizon is the hold, so the label declares `h` and
  the purge is derived from it.
- **The forward move** for T2: mid(`t + h`) / mid(`t`) − 1 in bp, forward-only,
  unsmoothed (`labels/targets.py`, `forward_smoothed_move_bp` with no
  smoothing).
- **The control label**: the smoothed move (`smoothed_move_bp`), partly known
  at decision time, used only to train the control model of K-label.

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
  microstructure set. The selected list is used by T1, A0, Z and as T2's
  auxiliary input; it is recorded in the amendment.

**Models.** No hyperparameter of either model is searched; both use this
repository's defaults, not the 2025 research's.

- **Gradient boosting** (`models/gbm.py`), three classes, for T1, A0 and Z.
- **The Gaussian-head network** for T2: `models/tcn.py`'s causal dilated
  network (window 64 rows, 32 channels, 4 levels, kernel 3, dropout 0.1, at most
  20 epochs) with a head for μ and log σ², trained by Gaussian negative
  log-likelihood with a variance prior and a clamped variance; the checkpoint
  is chosen by the rank correlation of μ with the realised forward move on the
  inner validation.
- **The control model**: the same network trained on the control label.
- Training reads every tenth grid row (one per second); decisions are made on
  every row.

---

## Strategies

| Strategy | Signal | Entry and exit | Fee | Role |
|---|---|---|---|---|
| **T1** | gradient boosting on the walked-book label at 5.5 bp; hold-first rule | taker in, taker out after `h` | base | primary T1 |
| **T2** | Gaussian-head network on the tensor and the selected set; EV gate | taker in, taker out after `h` | base | primary T2 |
| **M1** | the signal of T1 or T2 (chosen on D) | posted at the touch on the signal's side, waits at most `h`; once filled, taker out `h` after the fill | base | primary M1 |
| **Z** | gradient boosting on the walked-book label at 0 bp; hold-first rule | as M1 | 0 / 0 | secondary Z |
| **A0** | as Z, at the original protocol's settings | taker, and as M1 under each fill rule | 0 / 0 | measurement A0, FILL |

**The hold-first rule** (T1, Z, A0). The model gives `p_long`, `p_hold`,
`p_short`. Go long when `p_long ≥ θ_long` and `p_long > p_short`; short
symmetrically; otherwise hold. `θ_long` and `θ_short` are set on the inner
validation so that each side would enter half of the target rate of trades a
day before the one-position rule.

**The EV gate** (T2). With μ and σ the predicted forward move over `h` in bp,
`c` = 2 × 5.5 bp, and `w_t` the cost in bp of walking the current book for the
clip and back (from the snapshot, in bp of mid): go long when
`μ − kσ > c + w_t + b`, short when `−μ − kσ > c + w_t + b`. One entry per event
(a run of rows on which the gate is open); a new entry needs the gate to close
and reopen. Size by `strategies/sizing.py`'s `FractionalKelly` at fraction 0.5
on μ and σ, capped at one clip; below one lot (0.001 BTC) the trade is
skipped.

**Every strategy** holds one position at a time: the next entry is possible
only after the previous position is closed (cooldown equal to the hold).
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
| `b` (buffer) | T2 | {0, 1, 2} bp | searched on D |
| M1's signal | M1 | T1 or T2, whichever has the higher D net per trade at its chosen cell (T1 on a tie) | chosen on D |
| A0's settings | A0 | 240 trades a day, `h` = 50 s, trained 2025-01-01 to 01-27, thresholds on 01-28 to 02-03, run on R | fixed: the original protocol's trade rate and its close-by-time hold (about 50 s, inferred from the research code) |
| clip, latency, fees, entry wait, Kelly fraction | all | as above | fixed |

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
5. **The choice**: per strategy, the objective is the mean daily metric over
   the 150 test days (net per trade at the base tier for T1 and T2; gross per
   trade under the queue rule for Z), among cells averaging at least 5 trades
   a day; the cell with the highest neighbourhood median
   (`validation/search.neighbourhood_scores`: the cell and every cell one step
   away on each ordered axis), ties broken by its own value. The outright peak
   is reported beside it.
6. **M1's signal** from the T1 and T2 choices; M1 run on D's test days for its
   σ.
7. **The frozen models**: each chosen cell retrained on D from 2025-01-01 to
   06-23, thresholds and checkpoint on 06-24 to 06-30. The weights stay in the
   ignored `models/` directory; their sha256 goes in the amendment.
8. **A0 and FILL on R**, the detector's alarm rate on D, σ of the daily
   results and the minimum detectable effects (below).
9. **The amendment**, committed before H is read.

---

## Metrics

- **`net_bp_trade`**: for one round trip of the clip, (exit value − entry
  value) over the entry notional, × 1e4, less fees in bp of the entry
  notional, plus funding. **Primary for T1 and T2**: each day's mean over its
  trades; the day-level statistic is over the days with at least one trade.
- **`net_per_100_clip_day`**: a day's net in USDT × 100 / the clip's
  notional, days without a fill counting as zero. **Primary for M1.**
- **`gross_bp_trade`**: `net_bp_trade` before fees and funding. **Primary for
  Z**, as a daily mean over trades.
- **`gross_bp_turnover`**: gross over turnover (both sides' notional), × 1e4:
  the fee break-even per side. Reported for every strategy against the base
  fee and the cheapest published fee for the side used.
- **IC**: per day, the rank correlation between the signal's score
  (`p_long − p_short`, or μ) and the realised forward mid move over `h`, over
  all decision rows; its mean over days.
- For passive fills: fill ratio, markouts at 1, 5 and 10 s by fill path
  (queue or trade-through), and the decomposition `spread + adverse(5 s)` (the
  part of the result earned at the fill, the "making" part of round one) and
  the rest (the move held after it).
- Beside every verdict: trades a day, share of days positive, median day,
  maximum drawdown, net in USDT a day at the clip used, funding, the result
  at every published BTCUSDT tier, and the result split by side.
- **Statistics**: `evaluation/significance.assess(cluster="day")`; the day
  level decides. Score every calendar day of the block, with no row filtered
  or reweighted by label, outcome or model state.
- **Minimum detectable effect**, recorded in the amendment for each primary
  and for Z: `2 × σ_D / √n`, with `σ_D` the standard deviation of the
  strategy's daily primary metric over D's 150 walk-forward test days at its
  chosen cell, and `n` = 44 (the H days).

---

## Status rules (family of three primaries: T1, T2, M1)

On H, the first held-out read:

- **killed** — any of the hypothesis's kill conditions fires. Rejection needs
  no significance. A hypothesis whose result is void (K-leak, K-units, K-acct,
  K-cap) is recorded as `void`, which counts as killed.
- **candidate** — no kill condition fires, and all of: the day-level one-sided
  t passes Holm's procedure at 5% across T1, T2 and M1 (43 degrees of freedom,
  fewer if days are excluded); that t is at least 3; and the median day is
  positive.
- **inconclusive** — no kill condition fires and the bar is not met; labelled
  "below the minimum detectable effect" when the mean is smaller in size than
  the MDE. Never reported as a positive.

After H, with `b` the break date from the frozen detector (the first alarm in
F; if there is none, all of F counts as before the break), F is split into F−
(F's days before the alarm's day) and F+ (F's days after it); the alarm's own
day is in neither. Taken to F: every candidate, and every
inconclusive hypothesis with a positive mean. Holm runs again across those
taken, on F−.

| On H | On F− | R (below) | On P | Status |
|---|---|---|---|---|
| candidate | no kill, Holm passes | not supported | no kill, Holm passes | **confirmed** |
| candidate | no kill, Holm passes | not supported | no kill, Holm fails | **candidate, not confirmed** |
| candidate | no kill, Holm passes | supported | mean ≤ 0 | **confirmed until the break** |
| candidate | no kill, Holm passes | supported | mean > 0 | **candidate; the break did not last** |
| candidate | no kill, Holm fails | — | not read for it | **candidate, not confirmed** |
| inconclusive, positive | no kill, Holm passes | as above | as the first four rows | the same statuses, marked **first significant on F−** |
| inconclusive, positive | no kill, Holm fails | — | not read for it | **inconclusive** |
| either | a kill fires | — | not read for it | **killed on F** |
| either | no kill, Holm passes | either | a kill fires | **killed on P** |

"Holm passes" on F− and P includes the t ≥ 3 and median-day conditions, as
on H. A hypothesis that is `candidate` after H keeps that status whatever F
and P show unless a kill fires there; the later reads add to it.

P is read only if some hypothesis, Z included, passed on F−, and only for
those. F is read once, after H's results are committed, whatever H's
outcome: its detector alarms and every frozen strategy's numbers on F are
reported as measurements even when nothing is taken there. Every comparison
from H into F crosses the archive change and is labelled so.

**Z** is outside the family. It is tested alone, one-sided at 5%, with the
same t ≥ 3 and median-day conditions, and follows the same table, but its
best possible status is **conditional on a zero-fee venue**: the condition
(0 bp on both sides) is stated as a number, and the base-tier result, measured
on the same trades, is reported beside it.

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

- **K1** — the hypothesis's primary metric, averaged over the block's days,
  is ≤ 0.
- **K-label** — the signal's IC against the realised forward move over `h`
  is ≤ 0 on the block (the model has no information about the move it
  trades).
- **K-placebo** — the true mean does not exceed: the 95th percentile of the
  200 random-direction draws; the sign-flipped signal's mean; the one-day
  shifted signal's mean; and, for M1 and Z, the stale signal's mean (below).
- **K-drift** — the true mean does not exceed the larger of the all-long and
  the all-short versions' means at the same entry times (the result is the block's
  drift, not the signal).
- **K-nbhd** — the primary metric on the block, recomputed with each cell of
  the chosen cell's D neighbourhood in place of the chosen cell, has a median
  ≤ 0. For M1, the neighbourhood is that of its signal's cell.
- **K-days** — half or fewer of the block's days with a trade are positive.
- **K-fee** — the fee break-even per side (`gross_bp_turnover` for a taker
  strategy; for M1 the maker fee at which net is zero with the taker exit at
  the base fee) is at or below the base fee for that side. For a positive net
  at the base tier this can only fire through an accounting fault, so it is
  also a check; its main use is the report of how far each strategy was from
  paying.
- **K-dir** (M1, Z) — `spread + adverse(5 s)` over the passive fills is ≤ 0
  while the primary metric is > 0: the result is the move held after a fill
  that was adversely selected at the fill.
- **K-queue** (M1, Z) — a positive result that is not positive under the
  joint pessimistic queue bracket (pessimistic cancellation attribution and
  `all_ahead` arrival growth), or under either `none` or `all_ahead` alone.

Not kills, but they set the status:

- **K-trades** — fewer than 100 trades (fills, for M1 and Z) over the block
  makes the result inconclusive whatever its sign.
- **K-MDE** — a result that does not pass and is smaller in size than the MDE
  is "inconclusive, below the minimum detectable effect".

Each condition is applied to every claim of a hypothesis. A condition that
needs a positive result (K-queue, K-dir) fires only when the result is
positive under the default rules.

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
| random break dates | `b` drawn uniformly over F's days that leave at least 10 usable days on each side | R | 200 draws, 0–199 |

---

## T1 — the boosted classifier, crossing, at the real fee

**Statement.** On H at the base tier, T1 (gradient boosting on the walked-book
label at 5.5 bp, the hold-first rule at the chosen trade rate, a taker exit
after the chosen hold) earns a positive `net_bp_trade`.

**Kill conditions.** K1 (mean daily `net_bp_trade` over H ≤ 0), K-label,
K-placebo, K-drift, K-nbhd, K-days, K-fee; void under K-leak, K-units, K-acct,
K-cap; inconclusive under K-trades.

**Prediction.** Killed by K1.

## T2 — the Gaussian network with an expected-value gate

**Statement.** On H at the base tier, T2 (the network on the tensor and the
selected set, the EV gate `μ − kσ > 11 bp + walk cost + b`, one entry per
event, Kelly-sized up to one clip, a taker exit after the chosen hold) earns a
positive `net_bp_trade`.

**Kill conditions.** As T1. Reported beside: the control model's IC against
its own label and against the realised move, which shows on public data how
much of a smoothed label is known at decision time.

**Prediction.** Killed by K1, or inconclusive under K-trades: an 11 bp round
trip on BTC moves of a few basis points over a minute may leave the gate
almost always shut.

## M1 — the same signal, entered passively

**Statement.** On H at the base tier, M1 (the chosen signal posted at the
touch on its side, filled from prints after the queue ahead under rules
R5–R7, a taker exit `h` after the fill) earns a positive
`net_per_100_clip_day`.

**Kill conditions.** As T1, plus K-dir and K-queue. Reported beside: the
fill ratio, markouts by fill path, the passive-exit variant, and the result at
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

**Kill conditions.** K1 (mean daily `gross_bp_trade` over H ≤ 0); K2 (the
mean daily net at 0/0, gross plus funding, ≤ 0); K-label, K-placebo (with the
stale signal), K-drift, K-nbhd, K-days, K-dir, K-queue; void under K-leak,
K-acct, K-cap; inconclusive under K-trades. K-fee does not apply: the
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

**Tested only for** a hypothesis, Z included, that passes on F− (see the
status table). Otherwise R is recorded as `not tested`, with the reason.

**Statement.** With `b` the first alarm of the frozen detector in F, the
hypothesis's daily primary metric has mean ≤ 0 on F+, and the difference of
daily means F− − F+ exceeds the 95th percentile of the same difference at 200
random break dates. R is **supported** if both hold, and **not supported**
otherwise.

**Untestable**, declared and not retuned: no alarm in F; fewer than 10 usable
days on either side of `b`; or the detector's alarm rate on D above one per
14 days (declared in the amendment, before H is read).

**Reported beside**: every alarm in H and F with the channel that moved
(scale, dependence or mean), the second detector's alarms, and the earlier
daily-bar run's alarms (item 7 of [what was known](#what-was-known-before-this-was-written)),
labelled as already public. The blocks are not moved by any alarm, including
alarms inside H.

**What R can establish** is that a policy fixed in advance, "stop at the first
alarm", separates days on which a strategy paid from days on which it did not;
not why it stopped.

---

## Measurements without a hypothesis

None enters a verdict, and none can be quoted as a result.

**A0, the original recipe on public data, on R.** Gradient boosting on the
walked-book label at 0 bp, trained on 2025-01-01 to 01-27, thresholds for 240
trades a day on 01-28 to 02-03, a 50 s hold, run on R (2025-02-04 to 02-26):
crossing at 0/0, and posted at the touch under each of the three fill rules of
FILL. Reported: `gross_bp_turnover` against the original's 0.36 bp, net at the
base tier and at every published tier, trades a day, IC. The table is
labelled **not comparable: different venue, product, quote currency and
fee**. Prediction: below 0.36 bp under every rule except fill-on-touch.

**FILL, the fill rule, with a prediction.** A0's passive order stream on R,
the same orders under three rules:

| Rule | A resting order fills | Bias |
|---|---|---|
| (a) on touch | in full, at the first print at or through its price from the side that hits it, whatever is queued ahead | optimistic bound, labelled so |
| (b) trade-through | only on prints strictly through its price, up to their size (R7 alone) | pessimistic |
| (c) queue | after the visible queue ahead is consumed by prints at its price (R5, R6), or on prints through it (R7) | the simulator's default |

Reported for each: fill ratio, markouts at 1 and 10 s by path, gross and net.
**Prediction, fixed now: the fill ratio is at least 0.75 under (a) and at most
0.50 under (c).** The original's closed engine reported about four in five
orders filled; if (c) also fills at least 0.75, the suspicion that the
original's fill rule was optimistic is withdrawn in writing in the results.
The same three rules are reported for Z's stream on H.

**The fee break-even**, per strategy and block: `gross_bp_turnover` for the
taker strategies; for M1 and Z, the maker fee at which net is zero with the
taker exit at the base fee and, separately, at 1.5 bp. Compared with every
published BTCUSDT tier, with the fee caveat.

---

## The break detector

Fixed now, run once over each block as the ledger opens it.

- **Input**: hourly log returns, close to close, of Binance USDT-M futures
  BTCUSDT 1h klines, each divided by the root mean square of D's hourly
  returns at the same UTC hour (a 24-value profile fitted on D only), so that
  the daily cycle of volatility is not read as a series of breaks. The stream
  starts on 2024-12-01, so the monitor is warm by 2025-01-01.
- **Monitor**: `validation/structural_breaks.py`,
  `MonitorSpec(history_len=365, online_len=90)` (in hours), families `scale`
  and `dependence` with the recorded thresholds 9.09 and 7.93
  (`RECORDED_THRESHOLDS[(365, 90)]`); the `mean` channel's odds against 8.93
  are reported, and do not define `b`.
- **`b`**: the first alarm effective in F, effective at the close of the hour
  whose return produced it. F− is F's days before the alarm's day, F+ its days
  after it; the alarm's own day is in neither.
- **On D**, the monitor runs over 2025-01-01 to 06-30 and its alarm rate is
  recorded in the amendment. Above one alarm per 14 days, R is declared
  untestable then, before H is read, and the monitor is not retuned.
- **Reading**: bars up to 2025-06-30 are development data; H's, F's and P's
  bars open with their blocks, through the ledger.
- **Reported beside, never defining `b`**: `validation/changepoint.detect` on
  the 1 s grid of BTCUSDT's mid and spread from the Bybit book, at its
  defaults (window 2000, threshold 20, reference 40, minimum gap 20,000, its
  four statistics), run continuously from 2025-01-01 through the same access,
  with any alarm within a day of 2025-08-21 labelled "at the archive change";
  and the earlier daily-bar run, labelled as public before this registration.

---

## Fetch plan

Nothing below has been fetched. Sizes are the archives' own, summed from the
`HEAD` requests. Every archive goes through `data/ensure.py`, which fetches
only missing days and keeps no archive after the replay.

| Stage | When | What | Book archives (GB) | Print archives (GB) |
|---|---|---|---:|---:|
| 1 | after this registration and the code it names are merged | W and D (212 days, `ob500`), funding, hourly bars 2024-11 to 2025-06 | 75.4 | 15.3 |
| 2 | after the amendment is committed, immediately before the read | H (44 days, `ob500`), its funding and bars | 11.4 | 2.1 |
| 3 | after H's results are committed | F (88 days, `ob200`), its funding and bars | 15.8 | 5.9 |
| 4 | after F's results are committed, only if the condition holds | P (45 days, `ob200`), its funding and bars | 9.9 | 3.5 |

E is never fetched. At most about 0.4 GB of archive is on disk at once per
worker. Replayed at ten levels, a BTCUSDT day took about 35 MB in the 2024
books, so W and D take about 7 GB and the whole year about 14 GB; prints add a
few GB.

---

## Compute budget

On the machine the market-making rounds ran on (10 cores, 32 GB), with
8 worker processes, each heavy run started under `nohup caffeinate` with its
log in `logs/`.

| Run | Size | Estimate |
|---|---|---|
| Replay W and D | 212 days at 2–3 minutes a day (the 2025 archives are larger than the 90 s measured on 2024's) | 1–2 hours |
| Features W and D | 212 days | 2–4 hours |
| Boosting (T1, Z, A0) | 3 holds × 5 folds + 3 frozen fits, each for T1 and Z, plus A0 | 4–10 hours |
| Network (T2, control) | 3 holds × 5 folds + 3 frozen fits, plus the control | 8–24 hours |
| Simulator on D (Z's 9 cells, M1, FILL on R) | about 1,600 configuration-days | 4–12 hours |
| First held-out read (H): every strategy, placebos (200 random draws each for M1 and Z through the simulator), brackets, neighbourhoods, tiers | about 20,000 configuration-days | 1–2 days |

Every fit reads at most 4 million training rows, taken evenly in time, so a
fit's cost does not grow with D. The amendment records the measured times and
the largest worker's peak memory; a run that is slower than estimated takes
longer, and nothing registered is dropped to save time.

---

## Who may read what, and the code that will enforce it

The pull requests that follow this one add, before stage 1:

- the ported pieces (the tensor, the microstructure set, the walked-book
  label, the Gaussian head, the hold-first rule, the EV gate, the fill-rule
  comparison), each under the leakage checker and with streaming-versus-batch
  parity tests;
- the `ob200` archive name in `data/bybit.py` and the hourly-bar input;
- **this study's loader and ledger**, on the pattern of
  `market_making/round2.py`: the YAML validated against this registration,
  placeholders hashed as in the earlier rounds, and an access that permits
  days per block. The development access permits W and D (book, prints,
  funding) and the hourly bars up to 2025-06-30. The ledger,
  `experiments/results/short_horizon_2025_ledger.json`, opens H only if the
  amendment records the YAML's sha256, the file on disk has it and is
  committed, and the tree is clean; F only after H's results are committed and
  the ledger records H; P only after F's results are committed, for the
  hypotheses that qualify. Each first read writes block, commit, config sha256
  and UTC time before the access is returned; a second read raises unless
  forced, and everything it produces is stamped `second read` and cannot
  change a status.

Round one's `prereg.Access` and the round-two access are not used to read any
day of this study.

---

## Amendment protocol

After the development runs, and before H is read, one dated amendment is
appended and committed: the days excluded and the sequence-gap threshold; the
selected features; for T1, T2 and Z the chosen cell with its neighbourhood and
the outright peak; M1's signal with the D values that chose it; σ of each
daily primary metric on D and the minimum detectable effects; the A0 and FILL
tables on R; the detector's alarm rate on D and, if it applies, the
declaration that R is untestable; the sha256 of every frozen model's weights;
the compute record; and the sha256 of `configs/short_horizon_2025.yaml` after
the frozen values are written into its placeholders, on a line of the form
``Frozen configuration sha256: `<hash>` ``. Nothing else may change.

A correction found after this registration and before H is read (a bug in a
ported feature, the label, the simulator or a reader) is made only by a
further dated amendment that states it and, if it could change a development
number, re-runs D in full before anything is frozen. After H is read, nothing
is changed and nothing is re-run under a different rule.

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
- **Why a strategy stopped.** R tests a causal stopping policy on a detector
  of BTC's returns; it does not identify the mechanism, and it does not test
  the author's recollection, which stays a recollection.

---

The configuration as registered, with every placeholder `null`, has sha256
`e01ba377857e895b2d81d6a625434fb183210825ea906f2801e922eef7c93308`.
This study's loader will check that putting `null` back into every placeholder
of the frozen file gives this hash, as the earlier rounds' loaders do.

---

## Amendments

None yet.
