# Pre-registration: market making, second round, on wide-spread instruments

This document registers the second round of the market-making study before any
of its data is fetched. It is committed together with
[`configs/mm_prereg_round2.yaml`](../../configs/mm_prereg_round2.yaml), which
holds the same values in the form the code will read, and the two must agree.

| | |
|---|---|
| **Committed** | 4 October 2026, against `main` at `3071de7`, after round one's held-out results and before any book, print or funding file of a basket instrument is fetched |
| **What it registers** | The basket and its admission rule, the blocks, the strategies and what is searched, a causal "room" gate, hypotheses B1–B4 with their metrics, kill conditions and placebos, the status rules, how a survivor is reported, the fetch plan, the compute budget and the amendment protocol |
| **Computed so far** | **No market-making number has been computed on any basket instrument**, and no basket instrument's book, prints or funding has been fetched. What was known about them is listed under "What was known before this was written" |
| **How it changes** | Only under the [amendment protocol](#amendment-protocol): dated amendments appended at the end. One amendment is planned, after the development-period runs and before the first held-out read |

Round one is
[`docs/preregistration/market_making.md`](market_making.md) with
[`configs/mm_prereg.yaml`](../../configs/mm_prereg.yaml); its held-out results
are in its [results section](market_making.md#results-on-block-h-read-once-4-october-2026)
and in [§30](../results.md#30-market-making-on-the-held-out-fortnight-read-once).
Every simulator rule, metric definition and statistical convention of round one
applies here unchanged unless this document says otherwise. Names under
`market_making/` that do not exist on `main` at `3071de7` (the gate, the
round-two loader and ledger) refer to code added by the pull requests that
follow this one.

## Why a second round

Round one killed all four of its primaries on BICOUSDT, the one instrument its
admission rule let through. The measurements beside the verdicts say why: the
touch quoter S0 earned +21.5 USDT a day of spread on BICOUSDT and gave back 47.3
to the move in the five seconds after its fills, −5.00 bp of turnover; it would
have broken even only at a maker fee of −3.01 bp, three times the largest rebate
Bybit advertises. The tighter instruments needed less (S0's break-even was
−1.89 bp on CRVUSDT, −1.49 on XRPUSDT, −1.14 on BTCUSDT), but their one-tick
spread of 1.9 bp could not pay the base maker fee at all.

The question this round asks is whether there is **any** condition, stated in
advance and tested on data no market-making code has read, under which this
market maker earns money: a wider spread (the basket), quoting only when the
recent market says there is room (the gate), or a professional fee tier. The
original registration anticipated it: a basket of wide-spread instruments under
"a second, separately committed pre-registration, before any fetch", and a
pristine block P.

**The prior is negative.** Round one's arithmetic says that a wider spread is
usually paid for by more adverse selection, and that the base fee is the larger
obstacle. This round is designed so that a negative answer is as informative as
a positive one: if nothing survives, the fee break-even per instrument says how
far each one was from paying.

---

## What was known before this was written

Everything below was available when the design was chosen, and some of it
shaped the design. None of it is a quote, fill, markout or profit of a
strategy on a basket instrument.

1. **Round one's results**, all of them: the verdicts, the markouts by fill
   path, the advantage ladder, the fee break-even and the robustness sweeps on
   BICOUSDT, CRVUSDT, XRPUSDT and BTCUSDT over D and H. The gate of B2 and the
   fee hypotheses B3 are direct responses to them. Round one's development
   values for BICOUSDT (Amendment 2) are reused by B4.
2. **The committed screen table**
   [`screen_bybit_ranked.csv`](../../experiments/results/screen_bybit_ranked.csv),
   computed by [`experiments/screen_bybit.py`](../../experiments/screen_bybit.py)
   from Bybit's touch on a one-second grid on 2024-02-05 to 02-07 (inside D) and
   2024-03-25 to 03-27 (inside the unused block F). Its February taker round
   trip less the 12 bp of taker fees and slippage is an implied mean spread:

   | Instrument | February round trip (bp) | Implied spread (bp) |
   |---|---:|---:|
   | ALICEUSDT | 21.14 | 9.1 |
   | ALGOUSDT | 18.23 | 6.2 |
   | JTOUSDT | 17.57 | 5.6 |
   | ZECUSDT | 16.82 | 4.8 |
   | GALAUSDT | 16.59 | 4.6 |
   | GMTUSDT | 16.17 | 4.2 |
   | CAKEUSDT | 16.13 | 4.1 |
   | IOTAUSDT | 16.09 | 4.1 |
   | BICOUSDT, for comparison | 17.97 | 6.0 |

   These are the eight candidates round one named. The table's next instruments
   (PEOPLEUSDT 3.9, DASHUSDT 3.7, VETUSDT 3.6, DYDXUSDT 3.6) fall below 4 bp and
   are not candidates; the screen's other 60 instruments are not in the
   committed table and are not screened here (below).
3. **Other reads of basket data by earlier code, none of it market-making code
   except item (b).**
   (a) The screen above read the touch of all eight on those six days.
   (b) GALAUSDT and ALGOUSDT are members of §27's 26-instrument universe. Their
   Bybit touch on a one-second grid over 2024-02-01 to 03-10 and 03-12 to 04-20
   was read by §27's taker study, and over D and H by round one's reversion tape,
   where their mids entered the equal-weighted index and nothing else. The
   panel that builds the tape also computes each member's touch spread; no
   spread of GALAUSDT or ALGOUSDT was reported, used or looked at. The second
   range covers the first 13 days of P.
   (c) Binance's best bid and offer (another venue) for ALICEUSDT, CAKEUSDT,
   GALAUSDT and ZECUSDT over 2024-02-01 to 03-09, and for ALICEUSDT and CAKEUSDT
   over 2024-03-26 to 03-29, read by the Binance screens of §19–§20.
   (d) BICOUSDT's touch over 2024-03-12 to 04-20, read by §27, covers the first
   13 days of its block P (B4).
4. **Archive metadata, read for this registration.** An HTTP `HEAD` request for
   every order-book and print archive of the eight candidates on every day of
   D, H and P, and of BICOUSDT on every day of P and Q, returned each file's
   existence and size and nothing else; no archive was downloaded. Every file
   exists. The per-instrument totals were seen; they are in the
   [fetch plan](#fetch-plan) and say how active an instrument was in a block,
   not what its spread, prices or fills were.
5. **The calendar.** H (26 February to 9 March 2024) is publicly known as a
   fortnight of fast, broad price rises in crypto. A quoter that carried long
   inventory through it would have profited without making markets, which is
   what K-dir and K-fund exist to catch, and why the inventory component is
   reported beside every verdict.
6. **Nothing else.** No basket instrument's ten-level book, prints or funding
   has been fetched, opened or computed on; no quoter has run on one.

---

## Blocks

| Block | Dates (UTC, inclusive) | Days | Instruments | Use | Read |
|---|---|---:|---|---|---|
| **D** development | 2024-02-01 to 2024-02-25 | 25 | the eight candidates; BICOUSDT for B4 (round one's D) | admission, every searched value, the hour masks, σ of daily results, the minimum detectable effects | freely, through the development access |
| **H** held out | 2024-02-26 to 2024-03-09 | 13 | the admitted basket | B1, B2, B3a, B3b | once, in the first held-out read |
| **P** pristine | 2024-04-08 to 2024-05-05 | 28 | BICOUSDT | B4 | once, in the first held-out read, together with H |
| **P** pristine | 2024-04-08 to 2024-05-05 | 28 | the admitted basket | the second held-out block for B1–B3b survivors | once, after H's results are committed; fetched only then |
| **Q** confirmation (optional) | 2024-05-06 to 2024-06-02 | 28 | as needed | a third block, only for a hypothesis that is `candidate` after P (below) | once, after P's results are committed; fetched only then |
| F | 2024-03-12 to 2024-04-07 | 27 | — | not used by this round; not fetched for the basket | never |

Why these blocks.

- **D and H for the basket.** No market-making code has read a basket
  instrument's book or prints on any day; round one's simulator, screen and
  searches read only BICOUSDT, CRVUSDT, XRPUSDT and BTCUSDT. The one exception
  is the reversion tape's use of GALAUSDT's and ALGOUSDT's touch (item 3b
  above): only their mids entered a number, their touch spreads were computed
  inside the panel and never reported or looked at, and their depth and prints
  were never read. Every pooled verdict is also reported without them. Reusing
  round one's calendar keeps the development and held-out lengths that round
  one's statistics were built for (25 and 13 days, 12 degrees of freedom).
- **P as a second held-out block.** Four weeks, beginning a month after H,
  outside every block either round has searched on. For BICOUSDT it is unread by
  any market-making code. Round one reserved BICOUSDT's P for a primary that was
  `candidate` after F; all four were killed on H, so that condition can no
  longer be met and this round takes BICOUSDT's P for B4. For GALAUSDT,
  ALGOUSDT and BICOUSDT the first 13 days of P were read at the touch by §27
  (item 3).
- **F is not used.** For the basket it is unread at depth, but the screen read
  three of its days, §27 read GALAUSDT's and ALGOUSDT's touch over all of it,
  and round one reads it next for BICOUSDT (H2.3). Leaving it out keeps this
  round's reads few and its blocks clean. Round one's F read cannot inform this
  round, which is committed before it.
- **Q** exists so that a hypothesis whose first positive is on P, rather than
  on H, still meets a block that played no part in it. It is fetched only if
  needed.

**Days are independent**, as in round one: each starts flat, is flattened at
23:59, and is the cluster for every statistic. No reader crosses midnight, so
the trailing gate below reads only the day's own data.

---

## Instruments and admission

### The basket

The eight candidates named by round one, from the committed screen table:
**ALICEUSDT, ALGOUSDT, JTOUSDT, ZECUSDT, GALAUSDT, GMTUSDT, CAKEUSDT, IOTAUSDT.**
No other instrument is added. The committed table holds the screen's 20
best-ranked instruments (by its edge-over-cost score) of the 80 it measured, and
the eight are those in it with an implied spread of at least 4 bp; its next four
are below. Instruments outside the table may have wider spreads, but finding
them would mean fetching ten-level books for all 60 on D, and would make the
basket the result of a much wider search, chosen on the block that also sets
every parameter.

### Admission (computed on D only, committed before H)

[`market_making/screen.py`](../../src/trading_research/market_making/screen.py)
computes, per instrument over all D days and exactly as in round one: tick (bp,
time-weighted), time-weighted spread (bp), time-weighted share at two ticks or
more, median touch over median print, prints per day, market-wide passive
markout at 1, 5 and 30 s, and coverage of D and H (whether a non-empty book and
print file exists for each day; for H from the file system alone, no file
opened). An instrument is admitted if it meets either criterion.

- **A1, round one's rule.** Spread ≥ 4.0 bp (twice the base maker fee) **and**
  share at two ticks or more ≥ 0.25 **and** prints per day ≥ 5,000 **and**
  coverage ≥ 90% of D days and of H days.
- **A2, tick-constrained.** Tick ≥ 4.0 bp **and** prints per day ≥ 5,000
  **and** coverage ≥ 90% of D and of H. (The spread is then at least 4 bp by
  construction.)

Why A2. Round one required a quarter of the time at two ticks or more because
H1 quoted one tick inside the spread, which needs room for a tick. No
hypothesis here quotes inside the spread. What a two-sided quoter needs at the
base fee is a spread of at least twice the maker fee, and a one-tick spread
gives that when the tick itself is 4 bp or more. Round one excluded CRVUSDT and
XRPUSDT because their single tick of 1.9 bp could not pay the fee; A2 admits
the instruments for which the same arithmetic runs the other way. Such books
are queue-dominated: a resting order joins a long queue at the touch, and the
fills that reach its tail are the most adversely selected ones (round one's
ladder: the front of the queue was worth about 1 bp of turnover on BICOUSDT).
The simulator's queue rules apply unchanged, and median touch over median print
is reported for each admitted instrument as the measure of queue length.

**If few or none are admitted.** The committed D table wins, whatever it
admits. If one instrument is admitted, B1–B3b run on it alone, and every pooled
statistic is that instrument's. **If none is admitted, the round ends at the
amendment**: the admission table is committed and reported, B1–B3b are recorded
as `not tested: no instrument admitted`, H is never read for the basket, and
only B4 (on BICOUSDT, admitted in round one) goes on.

Days excluded from every verdict, as in round one (its Amendment 1, items 7–10):
a day with a book sequence gap (`max_sequence_gaps` 0 unless the amendment sets
another value from D's counts), a flatten that found no fresh book, a missing
funding plane, or a price off the day's tick grid. Exclusions are counted and
reported per instrument and block.

---

## Strategies

All strategies run in round one's simulator with its settings unchanged
(round one's "Simulator settings a verdict depends on" and Amendment 1): fills
only from prints, a queue per order joining the tail of the visible size,
proportional cancellation attribution, pro-rata arrival growth, 10 ms order and
cancel latency, post-only, a clip of at most a tenth of the touch, soft and hard
inventory limits with a costed taker flatten, funding at each settlement, days
that start and end flat. The simulator is the code on `main` at `3071de7`; a
change to it before the held-out read is made only by a dated amendment that
says why, and the run cache is keyed by a hash of its source.

| Strategy | What it is | Role |
|---|---|---|
| **S0 touch** | Round one's S0: bid at best bid, ask at best ask, fixed clip, soft limit 6 clips | Reference only: fee break-even |
| **S1 skew** | Round one's S1 formula, searched on D per admitted instrument over round one's 144 cells | B1; the base of B2; B3 |
| **S1-touch** | S1 at `k` = 0 and `min_edge_bp` = 0, with the instrument's chosen `skew_bp` and `soft_limit_clips`: the edge gate is the maker fee alone, so it quotes at the touch whenever the half-spread clears the fee | A second base for the gate of B2 and B4 |
| **G(base)** | A base quoter, quoting only while the room gate below is open | B2, B4; B3 if chosen on D |

Per admitted instrument, fixed by rule on D and frozen by the amendment, as in
round one: the clip's notional cap (a tenth of D's median touch notional,
sampled every second, then the adaptive cap of a tenth of the trailing one-hour
median touch) and `sigma_ref` (the median over D's quoting hours of the
one-minute volatility the simulator computes). For BICOUSDT, round one's frozen
values (20.0434 USDT and 7.92472 bp) are reused.

### The room gate

The gate quotes only when the recent market says a passive fill is worth more
than the fee it pays. Every print in the tape is scored from its resting side,
as round one's market-wide passive benchmark already does
(`analysis.market_wide_markouts`):

```
markout_5s(print) = side × (mid(τ + 5 s) − p) / p × 1e4
                  = side × (mid_ref − p) / p × 1e4          half-spread captured
                  + side × (mid(τ + 5 s) − mid_ref) / p × 1e4   adverse move, usually negative
```

with `τ` the print's time, `p` its price, `side` +1 when the resting side
bought, `mid_ref` the last snapshot mid at or before the print, and
`mid(τ + 5 s)` the last snapshot mid at or before `τ + 5 s`. At a decision at
time `t`:

```
room_t = vwmean{ markout_5s(print) : t − W ≤ τ and τ + 5 s ≤ t } − maker_bp
```

the volume-weighted mean over the day's prints in the trailing window whose
five-second horizon has closed by `t` (the snapshot that closes it is stamped at
or before `t`), less the maker fee the quoter pays. It is the trailing
half-spread a passive fill captured, less the trailing adverse move that
followed it, less the fee: the "spread exceeds adverse selection plus the fee"
reading, measured as one quantity the code already computes. The room is
defined, and the gate can open, only once the whole window lies inside the day
(`t` at least `W` minutes after 00:00, since days are independent) and holds at
least 20 such prints; otherwise the gate is closed.

**Hours of the day.** Per instrument, on D only: for each UTC hour, the
volume-weighted mean of `markout_5s` over every D print in that hour, less the
maker fee. Hour `h` is in the instrument's mask at margin `μ` if that value is
at least `μ`. The mask is a fixed table, causal by construction.

**Kinds** (each at margin `μ`):

- `trailing`: open when `room_t ≥ μ`;
- `hours`: open when the current UTC hour is in the mask at `μ`;
- `both`: open when both hold.

**While the gate is closed** the base's quotes are cancelled except on the side
that reduces |position|, which stays at the base's price, so inventory can
unwind passively; when flat, nothing is quoted. Soft-limit flattens and the
23:59 flatten are unchanged. The gate is evaluated at every decision with the
10 ms decision latency, and reads the fee its quoter pays, so at a cheaper tier
the room is larger by the difference.

**Searched on D** (42 cells): base ∈ {S1, S1-touch} × (`trailing` and `both`
over `W` ∈ {15, 60, 240} min and `μ` ∈ {0, 1, 2} bp, plus `hours` over
`μ` ∈ {0, 1, 2} bp). Not registered, and so not available later: a gate per
side, other windows or horizons, other margins, or a gate on any other
quantity.

### Fee tiers

| Tier | Maker (bp) | Taker (bp) | Source | Used by |
|---|---:|---:|---|---|
| base | 2.0 | 5.5 | `backtest/costs.py`, `BYBIT_BASE` | B1, B2, B4, every search |
| `pro1_altcoin` | 0.0 | 2.8 | `BYBIT_LINEAR_TIERS`: the first published row with a 0 bp maker fee for perpetuals outside the major coins (from 100 million USDT of 30-day volume, more than a fifth through the API) | B3a |
| programme | −1.0 | 2.8 | `BYBIT_MM_PROGRAMME_MAKER_BP`: "up to" a 1 bp rebate, on application; the programme publishes no taker rate, so `pro1_altcoin`'s is assumed | B3b |

Every number at a tier other than the base carries the code's caveat
(`BYBIT_FEE_CAVEAT`): the schedule is the one published in 2026, applied to
2024 data, when it may have differed. The taker fee is paid only on flattens;
B3's verdicts are also reported with the taker fee at the base 5.5 bp.

---

## The development-period search (D only)

1. **Fetch and screen.** The admission table above, on D, with H's coverage
   from the file system.
2. **Per admitted instrument:** clip notional cap and `sigma_ref`.
3. **S1, per admitted instrument** (B1). Round one's procedure: a
   `successive_halving` run over the 144 cells with budgets of 5, 12 and 25 D
   days (a nested subset drawn with seed 0), keep fraction 0.34, objective the
   instrument's mean daily net in USDT, minimum 20 passive fills a day; the
   choice by `neighbourhood_scores` over the cells measured at the full budget
   (the median of a cell and every measured cell within one step on each
   ordered axis, ties broken by the cell's own value, cells below the fill
   minimum taking no part), the outright peak reported beside it. Each
   instrument gets its own cell: ticks, spreads and volatilities differ by large
   factors across the basket, and the parameters are in basis points.
4. **The gate** (B2). The 42 cells, each on all 25 D days and every admitted
   instrument, on top of each instrument's chosen S1 (or its S1-touch cell).
   Objective: the pooled daily metric below, averaged over D. A cell takes part
   only if the admitted instruments average at least 5 passive fills a day under
   it. Choice: the highest neighbourhood median, the neighbourhood being the
   cells of the same base and kind within one step on `W` and `μ` (on `μ` alone
   for `hours`), ties broken by the cell's own value. One gate cell for the whole
   basket: the room is already in basis points of the fee, so its parameters
   transfer, and one choice is less room for luck than eight.
5. **B3's strategy at each tier.** Each instrument's chosen S1 and the chosen
   G(base), every value frozen, are run on all D days at `pro1_altcoin` and at
   the programme tier, the edge gate and the room gate reading the fee they pay.
   At each tier the one with the higher pooled mean on D is B3's strategy there.
   Nothing is re-searched at a tier: B3 measures what the fee changes, everything
   else held.
6. **B4 on BICOUSDT's D.** The same 42 gate cells, on round one's frozen S1 for
   BICOUSDT (2 / 0.5 / 4 / 6) and its S1-touch cell (2 / 0 / 0 / 6), over all 25
   days of round one's D, chosen by the same rule on BICOUSDT alone. BICOUSDT's
   D was read by round one, which the development block permits; P was not.
7. **Spread of daily results and the minimum detectable effects** (below).
8. **The amendment**, committed before the first held-out read.

---

## Metrics

**Primary, pooled over the basket: net per 100 USDT of clip a day.** For
instrument `i` on day `t`, `y_it = net_usd_day_it × 100 / clip_i`, with `clip_i`
the instrument's frozen clip notional cap; the pooled daily value is the mean of
`y_it` over the admitted instruments with a usable day `t`. It is the daily
profit of quoting one clip on each instrument, in USDT per 100 USDT of clip.
Round one pooled in USDT, and its H2.1 verdict was weighted 170 to 1 by
XRPUSDT's clip; normalising by the clip weights each admitted instrument
equally. For B4, the same quantity on BICOUSDT alone. The pooled USDT sum is
reported beside it.

Every round-one metric is reported beside the primary, per instrument and
pooled: `net_usd_day`, `net_bp_turnover`, `making_usd_day` (spread plus the
5 s adverse move less fees), the decomposition (spread, adverse, inventory,
fees, funding), markouts at 1, 5 and 30 s by fill path against the market-wide
passive benchmark, fills a day, time quoted, RMS and maximum inventory,
flattens, maximum drawdown and clip over touch. For a gated strategy, also the
share of quoting time the gate is open, per instrument.

**Statistics** from `significance.assess(cluster="day", series="symbol")`; the
day level decides. A paired difference is taken per instrument-day where both
days are usable, then pooled over instruments each day as above.

**Minimum detectable effect**, recorded in the amendment for each primary's
metric: `2 × σ_D / √n`, with `σ_D` the standard deviation of the pooled daily
metric over D's 25 days and `n` = 13 for B1–B3b (H) and 28 for B4 (P).

---

## Status rules (family of five primaries: B1, B2, B3a, B3b, B4)

On the first held-out read (H for B1–B3b, P for B4):

- **killed** — any of the hypothesis's kill conditions fires. Rejection needs
  no significance.
- **candidate** — no kill condition fires and the day-level one-sided t passes
  Holm's procedure at 5% across the five primaries (12 degrees of freedom for
  B1–B3b, 27 for B4).
- **inconclusive** — no kill condition fires and the significance bar is not
  met. Never reported as a positive.

A hypothesis with two claims (B2 and B4: positive, and more than the base) is
tested at the larger of its two one-sided p-values, as in round one.

**The second held-out read, P, for B1–B3b.** Taken to P: every hypothesis that
is `candidate` on H, and every one that is `inconclusive` on H with every claim
positive. Every frozen value is reused, on every admitted instrument (never a
subset chosen after H), and every kill condition is evaluated again on P. Holm
at 5% runs across the hypotheses taken to P, one-sided, 27 degrees of freedom.

| On H | On P | Status |
|---|---|---|
| candidate | no kill, Holm passes | **confirmed** |
| candidate | no kill, Holm fails | **candidate, not confirmed** |
| inconclusive, positive | no kill, Holm passes | **candidate** (first significant on P), taken to Q |
| inconclusive, positive | no kill, Holm fails | **inconclusive** |
| either | a kill condition fires | **killed on P** |

**Q** is read only for a hypothesis that is `candidate` after P (B4 if
`candidate` on its P, or a B1–B3b hypothesis first significant on P), with the
same rules: no kill and Holm passing across the hypotheses taken to Q gives
**confirmed**; a kill gives **killed on Q**; otherwise it stays **candidate**.

**Conditional (B3a, B3b).** A B3 hypothesis that ends `confirmed` while B1, the
same quoter at the base fee, does not end `confirmed` is reported as
**conditional on the fee tier**: the condition (a professional tier) is stated
as a number, and the failure outside it (the base fee) was measured. If B1 is
also confirmed, B3 adds nothing and says so.

### Common kill conditions (every primary, on every block it is read on)

Carried over from round one:

- **K-pess** — a positive verdict under proportional cancellation attribution
  that is not positive under pessimistic attribution;
- **K-fund** — net minus funding ≤ 0 (the result is funding);
- **K-dir** — `making_usd_day` ≤ 0 while net > 0 (the result is inventory
  drift, not making).

New in this round:

- **K-queue** — a positive verdict that is not positive under the joint
  pessimistic queue bracket: pessimistic cancellation attribution **and**
  `all_ahead` arrival growth. Round one varied the brackets one at a time; a
  demonstrative result has to survive the pessimistic queue as a whole.
- **K-nbhd** — the hypothesis's primary metric on the held-out block, recomputed
  with each cell of the chosen cell's D neighbourhood in place of the chosen
  cell, has a median ≤ 0. For B1 and B3 (when S1 is chosen) the neighbourhood is
  each instrument's S1 neighbourhood, the medians pooled over instruments; for
  B2, B4 and B3 (when the gate is chosen) it is the gate cell's neighbourhood. A
  result that only the exact frozen cell shows is a parameter accident, which is
  what the neighbourhood rule was meant to avoid on D.

As in round one, each common condition is applied to every sign claim of a
hypothesis, and K-pess and K-queue fire only when every claim is positive under
the default rules and one is not under the bracket.

---

## B1 — a wider spread, at the base fee

**Statement.** On the admitted basket over H at the base fee, S1 with each
instrument's parameters chosen on D earns a positive pooled daily net.

**Metric.** The pooled net per 100 USDT of clip a day; day-level one-sided t.

**Kill conditions.** (K1) the pooled mean daily net over H ≤ 0; plus K-pess,
K-queue, K-fund, K-dir, K-nbhd. **Inconclusive** rather than tested if the
basket's passive fills over H number fewer than 100.

**Placebo.** An unconditional quoter has no signal to shift or flip, so B1 has
no placebo in round one's sense. Its falsifiers are the controls that catch the
ways a market maker shows a profit it did not make: the neighbourhood (a
parameter accident), the pessimistic queue (a simulator accident), K-dir (the
inventory carried through a rising fortnight) and K-fund (funding).

**Reported beside.** Every instrument's own net with its day t; the pooled net
with each instrument left out in turn; the pooled net without GALAUSDT and
ALGOUSDT (item 3b); S0 on every instrument.

## B2 — quoting only where there is room

**Statement.** On the admitted basket over H at the base fee, G(base) with the
gate cell chosen on D earns a positive pooled daily net, and more than its base
without the gate on the same days.

**Metric.** The pooled net per 100 USDT of clip a day of G(base), and the mean
paired daily difference G(base) − base, pooled; day-level one-sided t on each,
tested at the larger p.

**Kill conditions.** (K1) the pooled mean daily net of G(base) ≤ 0; (K2) the
mean paired daily difference G(base) − base ≤ 0; (K3) the true paired
difference does not exceed the 95th percentile of 50 **shifted-gate** placebos;
(K4) **the room does not persist**: over H, the volume-weighted market-wide
passive markout at 5 s less the maker fee, over the prints that arrived while
the gate was open, is not greater than over those that arrived while it was
closed, pooled over instruments and leaving out each day's first `W` minutes,
when the gate is closed by construction (computed from the tape, not the simulator: if
the trailing room does not forecast the next minutes' room, the gate has no
mechanism); plus K-pess, K-queue, K-fund, K-dir, K-nbhd. **Inconclusive** if
G(base)'s passive fills over H number fewer than 100, or the gate is open less
than 1% of quoting time pooled.

**Placebo.** The gate's open/closed series on H, per instrument, circularly
shifted around the block by an offset uniform in [6 h, 13 days − 6 h], the same
offset for every instrument in a draw, seeds 0–49; the base and everything else
unchanged, and the shifted strategy re-simulated. The share of time open is
preserved; only the timing moves. If the gain comes from quoting when the room
is there, the shifted gates must gain less.

**Reported beside.** G(base) − S1 (B1's quoter) when the base is S1-touch;
every instrument's gated net and share of time open; the realised room inside
and outside the open periods per instrument; the same leave-one-out and
without-GALA-and-ALGO pooled values as B1.

## B3 — what a professional fee tier changes

**B3a statement.** On the admitted basket over H at `pro1_altcoin` (maker 0.0,
taker 2.8 bp), the strategy chosen for that tier on D (each instrument's S1 or
the chosen G(base), every value frozen) earns a positive pooled daily net.

**B3b statement.** The same at the programme tier (maker −1.0, taker 2.8 bp),
with its own D choice.

**Metric.** The pooled net per 100 USDT of clip a day at the tier.

**Kill conditions** (each). (K1) the pooled mean daily net over H at the tier
≤ 0; plus K-pess, K-queue, K-fund, K-dir and K-nbhd, all at the tier.
**Inconclusive** if fewer than 100 passive fills over H.

**Placebo.** None: a fee is not a signal, and the claim is about the fee alone.
B3's comparison with B1 is the point and is reported, not tested: the same
quoter at the base fee is B1 (or, if G(base) is chosen, B2's G(base)).

**Framing, fixed now.** B3 is a conditional result about what a professional
fee tier changes, not a result a retail account can use: the 0 bp tier needs
100 million USDT of 30-day volume, the rebate needs an accepted application,
and both rest on a schedule published in 2026. A B3 survivor is never reported
without the base-fee result beside it.

## B4 — the gate on BICOUSDT, on a block no market maker has read

**Statement.** On BICOUSDT over P (28 days) at the base fee, G(base) with the
base and gate cell chosen on BICOUSDT's D earns a positive daily net, and more
than its base without the gate on the same days.

**Metric, kill conditions, placebo.** As B2, on BICOUSDT alone over P: (K1) mean
daily net ≤ 0; (K2) mean paired daily difference ≤ 0; (K3) not above the 95th
percentile of 50 shifted-gate placebos, offset uniform in [6 h, 28 days − 6 h],
seeds 0–49; (K4) the room does not persist on P; plus K-pess, K-queue, K-fund,
K-dir, K-nbhd; inconclusive under 100 fills or 1% of time open. Day-level
one-sided t with 27 degrees of freedom.

**What is known about it already.** Round one: on BICOUSDT over H, S1 lost 1.20
USDT a day and S0 −5.00 bp of turnover; the market-wide passive markout at 5 s
was −0.94 bp on D and −1.36 on H, so at the base fee the average room was about
−3 bp. B4 asks whether there are stretches of BICOUSDT's day, identified in
advance, where it is positive. Its prior is the weakest of the five. P's first
13 days of BICOUSDT's touch were read by §27 (item 3d), not its depth or prints.

---

## Measurements without a hypothesis (on H for the basket, on P for BICOUSDT)

None enters a verdict, and none can be quoted as a result.

- **The fee break-even**, per instrument, for S0 (analytically, from one run),
  B1's S1 and B2's G(base) (re-run over round one's maker-fee grid, −1.5 to
  +2.0 bp in 0.25 steps, the taker fee held at the base, each gate reading the
  fee it pays): the fee at which each would have broken even. This is the
  answer to "how far from paying" if nothing survives.
- **Robustness**, for B1's S1 and B2's G(base) on every admitted instrument,
  each axis alone: cancellation attribution ×3, arrival growth ×3, latency
  {1, 10, 50, 100, 250} ms, feed latency {0, 20, 50} ms. Only K-pess and
  K-queue enter a verdict.
- **By hour of the day**, the net of B1's S1 and the market-wide room, per
  instrument, so a reader can see where the hour mask was right and wrong.
- **The admission criterion**: every pooled number also split by A1 and A2.

---

## What counts as a demonstrative interval

The question of this round is whether there is any condition under which this
market maker earns money: at least a small, demonstrable one. The answer is written
in a form fixed now, so that the README can state it without choosing words
after the numbers are known.

**A demonstrative interval is a hypothesis that ends `confirmed` (or
`conditional`, for B3).** Nothing else is. It is reported with all of the
following, and the README states it in this sentence, with the blanks filled:

> On [every admitted instrument, named], [strategy, with its frozen values]
> at [fee tier] earned [x] USDT per 100 USDT of clip a day ([y] USDT a day at
> the clips used, [z] bp of turnover) over the held-out fortnight (day t [t],
> Holm-adjusted p [p]), and [x'] over the four pristine weeks (day t [t'],
> Holm-adjusted p [p']). It loses [where it fails].

"Where it fails" is required and lists, from the registered measurements: each
admitted instrument on which it lost on H or P; the base fee, if the survivor
is B3; the latency at which its net turns non-positive; the pessimistic
brackets, with their values; the fee break-even; the hours in which it lost;
and the share of the day the gate was open, for a gated survivor. The size is
stated in USDT at the clips used (a tenth of the touch), with the reminder that
own impact is not modelled, so the size says nothing about capacity.

**What cannot be one**, however it looks: a sub-period of H, P or Q; an
instrument or set of instruments chosen after a held-out read; a cell, window,
margin or hour mask other than the frozen ones; a fee tier other than the three
registered; any bracket other than the default; a measurement without a
hypothesis (a fee break-even below the published fees, an hour that paid). The
first five may be described, labelled exploratory, and are never called a
result. A hypothesis that ends `candidate`, `candidate, not confirmed` or
`inconclusive` is reported under that status in the findings register, with its
numbers, and is not the demonstrative interval.

For B4 the same sentence names BICOUSDT, with P in place of the held-out
fortnight and Q in place of the pristine weeks.

**If nothing survives**, the README says so in one sentence, and gives the fee
break-even per instrument as the measure of how far each was from paying.

---

## Fetch plan

Nothing below has been fetched. Sizes are the archives' own, from the `HEAD`
requests of item 4; every file exists. Books come from
`quote-saver.bycsi.com/orderbook/linear/{symbol}/{day}_{symbol}_ob500.data.zip`
(500 levels, replayed by `data/bybit.py` into ten levels on a 100 ms grid),
prints from `public.bybit.com/trading/{symbol}/{symbol}{day}.csv.gz`
(`data/bybit_trades.py`), funding from Bybit's public REST endpoint without
credentials (`data/bybit_funding.py`), all through `data/ensure.py`, which
fetches only missing days and keeps no archive after the replay.

**Stage 1, after this registration is merged: D and H for the eight
candidates, P for BICOUSDT.** H's files are written, not read: the admission
screen checks their presence from the file system, and nothing opens them
before the held-out ledger permits it.

| Instrument | Book archives, D + H (MB) | Print archives, D + H (MB) |
|---|---:|---:|
| ALICEUSDT | 385 | 48 |
| ALGOUSDT | 591 | 111 |
| JTOUSDT | 972 | 188 |
| ZECUSDT | 417 | 71 |
| GALAUSDT | 1,428 | 610 |
| GMTUSDT | 777 | 129 |
| CAKEUSDT | 582 | 55 |
| IOTAUSDT | 600 | 103 |
| **Basket, 8 × 38 days** | **5,753** | **1,316** |
| BICOUSDT, P (28 days) | 784 | 80 |

About 7.9 GB downloaded in all. Replayed at ten levels the books take, by round
one's files (9.6 to 82 MB a day from the lightest instrument to BTCUSDT),
roughly 4 to 9 GB on disk, and the prints about 1.5 GB, against 110 GB free.
At `data/bybit.py`'s measured 20 s to download and 90 s to replay a day, 332
instrument-days on 8 processes take about 80 minutes. Funding is a few REST
pages per instrument and block.

**Stage 2, only if a hypothesis is taken to P, after H's results are
committed:** P for every admitted basket instrument (all eight: 6,357 MB of
books and 823 MB of prints).

**Stage 3, only if a hypothesis is `candidate` after P:** Q for the
instruments it covers (BICOUSDT: 532 MB of books and 47 MB of prints, every day
present; the basket's Q archives were not checked and will be before any
fetch).

---

## Compute budget

On the machine round one ran on: 10 cores, 32 GB, 8 worker processes, each
instrument-day job in a fresh process, every heavy run started as
`nohup caffeinate -dimsu uv run python -m experiments.<script> ... > logs/<run>.log 2>&1 &`
so that it survives the terminal and the machine does not sleep. Round one's
rates: 0.22 s of wall time per configuration-day on D (BICOUSDT alone) and
0.64 s on H (all four instruments, BTCUSDT and XRPUSDT included).

| Run | Configuration-days | Estimated wall time |
|---|---:|---:|
| D: screen, clips, `sigma_ref` | 200 instrument-days | minutes |
| D: S1 searches, 8 × (5 × 144 + 12 × 48 + 25 × 16) | 13,568 | |
| D: gate, 42 cells × 25 days × 8 | 8,400 | |
| D: B3 choice, B4 gate on BICOUSDT, the spreads for the MDEs | about 2,300 | |
| **D total** | **about 24,000** | **2 to 5 hours** |
| First held-out read (H for the basket, P for BICOUSDT): defaults, brackets, neighbourhoods, 50 gate placebos, fee grid, robustness | about 21,000 | **2 to 4 hours** |
| P for the basket, only if needed | 28/13 of the survivors' part of the first read | at most about 8 hours |

**Memory.** Round one's largest simulation worker reached 1.67 GB on D and
2.32 GB on H (a BTCUSDT day). Each worker is budgeted at 2 GB. The D run
records the peak resident memory of the largest worker per instrument; if any
exceeds 2 GB, the held-out run uses 6 workers instead of 8. Worker count does
not change results: round one's development run, made twice, came out
byte-identical. The per-day fill cache (`artifacts/mm/cache`, gitignored) grew
to 1.7 GB in round one and is expected to grow by a few gigabytes.

---

## Who may read what, and the code that will enforce it

The pull requests that follow this one add, before any basket file is fetched
for reading:

- **the gate**: the trailing room series and the hour mask, and a quoter
  wrapper `G(base)`, each tested on synthetic markets for causality (the room
  at `t` changes only with prints whose five-second horizon has closed by `t`),
  for the reduce-only behaviour while closed, and for a shifted placebo that
  preserves the share of time open;
- **the round-two loader and ledger**: the YAML validated against this
  registration with its placeholders hashed exactly as round one's are, and an
  access that permits days **per instrument and block**. The development access
  permits the basket's D and BICOUSDT's D. The held-out ledger,
  `experiments/results/mm_round2_ledger.json`, is separate from round one's and
  opens H for the basket together with P for BICOUSDT as one first read,
  refusing unless the amendment records the YAML's sha256, the file on disk has
  it and is committed, and the tree is clean; P for the basket opens only after
  H's results are committed, and Q only after P's;
- **the experiment scripts**: the development runs and the freeze, the held-out
  reads, and the verdicts, on the pattern of round one's
  `experiments/market_making*.py`.

Round one's `prereg.Access` and ledger are not used to read any basket day.
Round one's own F read for BICOUSDT (H2.3) goes ahead independently; when this
registration was committed, F had not been read by any market-making code, and
nothing here depends on it.

---

## Amendment protocol

After the development runs, and before the first held-out read, one dated
amendment is appended and committed: the admission table with each admitted
instrument's criterion; the clip caps and `sigma_ref`; each instrument's chosen
S1 cell with its neighbourhood and the outright peak; the chosen gate cell
(base, kind, `W`, `μ`) with its neighbourhood, and every instrument's hour masks
at each `μ`; B3's strategy at each tier with the D values that chose it; B4's
base and gate cell with its neighbourhood and BICOUSDT's hour masks; σ of the
pooled daily metric on D and the minimum detectable effect for each primary;
the compute record; and the sha256 of `configs/mm_prereg_round2.yaml` after the
frozen values are written into its placeholders. Nothing else may change.

A correction found after this registration and before the first held-out read
(a bug in the simulator, the gate or a reader) is made only by a further dated
amendment that states it and, if it could change a development number, re-runs
D in full before anything is frozen. After the first held-out read, nothing is
changed and nothing is re-run under a different rule; a second read of a block
is stamped `second read` and cannot change a status.

---

## What this round cannot establish

- **Capacity.** The clip is a tenth of the touch, own impact is not modelled,
  and a survivor's size in USDT says nothing about what it could trade.
- **The book is a 100 ms photograph.** Queue dynamics inside it are a rule,
  bracketed; K-pess and K-queue require a verdict to survive the pessimistic
  side of the bracket, not the truth.
- **Two months of one year.** H and P are four weeks apart in early 2024; a
  confirmed result is a statement about those months on those instruments.
- **The fee schedule** is the one published in 2026, applied to 2024.
- **Selection of the basket.** The eight candidates were chosen by their
  February spreads on three days inside D; a survivor generalises at most to
  instruments selected the same way.

The configuration as registered, with every placeholder `null`, has sha256
`98dbec924e13e48ed92680515b36e2cceaa04c7dc98e47ba8c62cec629c07345`.
The round-two loader will check that putting `null` back into every
placeholder of the frozen file gives this hash, as round one's does.
