# Pre-registration: market making on public Bybit data

This document registers a study before it is run. It is committed together with
[`configs/mm_prereg.yaml`](../../configs/mm_prereg.yaml), which holds the same
values in the form the code will read, and the two must agree.

| | |
|---|---|
| **Committed** | 2 October 2026, against `main` at `4367aca`, before any market-making code exists in this repository |
| **What it registers** | A two-sided quoter with inventory limits, simulated in event time on Bybit's ten-level book and trade prints; the blocks; the instrument admission rule; the strategies and what is searched; hypotheses H1–H3 with their metrics, kill conditions and placebos; the advantage ladder and the fee break-even; the amendment protocol |
| **Computed so far** | **No market-making number has been computed on any block**: no quote, fill, markout or profit of any strategy. The only real-data figures below describe the market, were read while designing, and are listed under "What was seen before this was written" |
| **How it changes** | Only under the [amendment protocol](#amendment-protocol): dated amendments appended at the end, never edits to the text above them. One amendment is planned, after the development-period search and before the held-out block is read |

Names under `market_making/` (the simulator, quoters, screen and ledger) and
`validation.search.neighbourhood_scores` refer to code added by the pull
requests that follow this one; every other module, function and constant named
here exists on `main` at registration. Section
numbers (§26, §27) refer to [`docs/results.md`](../results.md). The simulator's
full rules, each with the direction of its bias, will be documented in
`docs/execution_assumptions.md` when the simulator is added; the settings a
verdict depends on are fixed [here](#simulator-settings-a-verdict-depends-on).

The mechanisms are standard in the literature: Avellaneda and Stoikov (2008) for
the reservation price, Guéant, Lehalle and Fernandez-Tapia (2013) for inventory
limits, Cont, Kukanov and Stoikov (2014) for order-flow imbalance, Huang, Lehalle
and Rosenbaum (2015) for queue-reactive cancellation, Moallemi and Yuan (2017)
for the value of queue position, and Glosten and Milgrom (1985) for informed
flow.

---

## What was seen before this was written

- **Three D days** (2024-02-05, 02-12, 02-19) of BICOUSDT, CRVUSDT, XRPUSDT and
  BTCUSDT: spread, time at two ticks or more, touch-to-print ratio, prints per
  day and the market-wide passive markout. Read to design the admission rule.
  The ranges over the three days:

  | Instrument | Tick (bp) | Spread, time-weighted (bp) | Time at spread ≥ 2 ticks | Touch / median print | Prints per day | Market-wide passive markout 5 s (bp) |
  |---|---:|---:|---:|---:|---:|---:|
  | BICOUSDT | 2.4 | 5.6–6.2 | 0.66–0.89 | 12–26 | 7k–21k | −1.4 to +1.2 |
  | CRVUSDT | 1.9 | 1.9–2.3 | 0.06–0.08 | 13–20 | 60k–82k | −0.6 to −0.3 |
  | XRPUSDT | 1.8 | 1.8–2.0 | 0.00 | 1,700–5,900 | 250k–280k | −2.3 to −0.9 |
  | BTCUSDT | 0.02 | 0.02 | 0.00 | 460–540 | 0.9M–1.9M | −0.5 to −0.2 |

  At Bybit's base tier a passive round trip pays 2 × 2.0 = 4 bp of maker fee.
  CRVUSDT and XRPUSDT quote one tick of 1.8–2.1 bp, so two-sided quoting at the
  touch cannot recover the fee even with no adverse selection, and BTCUSDT
  quotes 0.02 bp. That arithmetic is why the admission rule below expects one
  instrument.
- **One D day** (2024-02-15) for the alignment of prints and snapshots, the
  spread distribution in ticks, timestamp ties, and a timing benchmark of a bare
  event loop on BICOUSDT, XRPUSDT and BTCUSDT. Of all prints, the share landing
  exactly at the touch of the last snapshot at or before the print was 73%
  (BICO), 67% (CRV), 68% (XRP) and 50% (BTC); the share through that touch was
  19%, 28%, 25% and 48%; 5–8% were on the wrong side of it. Distinct timestamps
  were 35–61% of prints. BICOUSDT's spread was four ticks or more about 15% of
  the time.
- **Counts on every block, and five BICOUSDT days in H and F.** Book rows and
  prints per day for every block (counts only), and BICOUSDT's spread and tick
  on 2024-02-28, 03-05, 03-15, 03-28 and 04-05. BICOUSDT's prints per day were
  15.5k in D, 58.7k in H and 63.2k in F; its price about 0.41 in D, 0.39–0.50 in
  H and 0.58–0.65 in F, so the tick went from 2.4 bp to 2.0–2.6 bp and then
  1.5–1.7 bp; its time-weighted spread was 6.6–6.8 bp on the H days looked at and
  2.6–5.5 bp on the F days.
- **No quoter, no fill and no markout of any strategy has been computed on any
  block.**
- **Block H has been read before**, by earlier taker and directional-passive
  studies in this repository: §27's taker reversion (its held-out block,
  26 February to 11 March), the directional maker study
  [`experiments/maker_bybit.py`](../../experiments/maker_bybit.py) (its final 35%
  of BICOUSDT, about 26 February to 9 March, reported under
  ["Maker execution — built, and it does not pay"](../../README.md#maker-execution--built-and-it-does-not-pay)),
  and the BICOUSDT search of
  [§26](../results.md#26-everything-searched-again-on-three-instruments-with-the-answer-held-back)
  (its final 13 days). It has not been read by a market maker. For H2 the
  signal's held-out performance is already public, so only the execution
  increment is new information. There is no block on disk where the reversion
  state is present and unread.

---

## Blocks

| Block | Dates (UTC, inclusive) | Days | Instruments with book and prints | Use | Reversion state |
|---|---|---:|---|---|---|
| **D** development | 2024-02-01 to 2024-02-25 | 25 | BICO, CRV, XRP, BTC | admission, every parameter, `θ`, `β̂`, guard settings, σ of daily net, minimum detectable effect | present (inside [§27](../results.md#27-cross-sectional-reversion-and-the-market-state-it-requires)'s search block) |
| **H** held out | 2024-02-26 to 2024-03-09 | 13 | BICO, CRV, XRP, BTC | read once; the verdicts | present (inside §27's held-out block) |
| buffer | 2024-03-10 to 2024-03-11 | 2 | — | excluded: CRV/XRP books end 03-09, no universe on 03-11 | — |
| **F** boundary | 2024-03-12 to 2024-04-07 | 27 | BICO, BTC | read once, after H's verdict is committed: H2's boundary prediction; persistence of any H1/H3 survivor | absent (index autocorrelation −0.0003 over 03-12 to 04-20, §27) |
| **P** pristine (optional) | 2024-04-08 to 2024-05-05 | 28 | BICO, to be fetched | only if a primary hypothesis is `candidate` after F; H1 and H3 only (the universe ends 04-20) | not measured |
| **B** basket (optional) | D and H dates | 25 + 13 | instruments admitted by the same rule from [`experiments/results/screen_bybit_ranked.csv`](../../experiments/results/screen_bybit_ranked.csv) (February window), to be fetched | a second, separately committed pre-registration, before any fetch | as above |

Why the boundaries fall where they do: the CRVUSDT and XRPUSDT books end on
2024-03-09, and only BICOUSDT and BTCUSDT have a book after that; the reversion
index's universe has no 2024-03-11; so 03-10 and 03-11 are a buffer, and F is
BICOUSDT and BTCUSDT only. Bybit funding history is not on disk; it is fetched
from the venue's public endpoint, without credentials, before any block is
simulated.

Candidates for B among the 20 instruments in the committed screen table, by its
February round trip less the 12 bp of taker fees and slippage (implied spread at
least 4 bp): ALICE, ALGO, JTO, ZEC, GALA, GMT, CAKE, IOTA, alongside BICO. The
screen's other 60 instruments are not in the committed table and would be
re-screened on D first.

---

## Instrument admission (computed on D only, committed before H)

`market_making/screen.py` computes per instrument over all D days: tick (bp),
time-weighted spread (bp), time-weighted share at two ticks or more, median
touch over median print, prints per day, market-wide passive markout at 1/5/30 s,
and book-and-print coverage of D and H.

- **MM-admitted** (S0–S3 verdicts, H1, H3): spread ≥ 4.0 bp (twice the base
  maker fee) **and** share at two ticks or more ≥ 0.25 **and** prints per day
  ≥ 5,000 **and** coverage ≥ 90% of D and of H days. Expected from the design-time
  look: BICOUSDT only.
- **H2-admitted** (H2.1, H2.2): in §27's universe
  (`pipeline/discovery.UNIVERSE`), not in `ReversionConfig.exclude`, coverage
  ≥ 90% of D and H. Expected: BICO, CRV, XRP.
- **References** (S0, ladder, fee break-even, no verdict): all four.

If the committed D table admits a different set than expected, the table wins
and the hypotheses run on what it admits.

---

## Strategies

All strategies share the [simulator settings](#simulator-settings-a-verdict-depends-on)
below. Fair value is the mid. Economic quantities are in basis points; prices
are in integer ticks.

| Strategy | What it adds | Role |
|---|---|---|
| **S0 touch** | Bid at best bid, ask at best ask, fixed clip, soft limit 6 clips, nothing else | Baseline; also rung 0 of the ladder |
| **S1 skew** | Avellaneda-Stoikov reservation price and a volatility-scaled half-spread with an edge gate | The market maker every hypothesis is compared against |
| **S2 inside** (H1) | S1, plus one tick inside the touch when the spread is wide enough and the improved price still clears the gate | H1 |
| **S3 guard** (H3) | The better of S1/S2 on D, plus pull-or-widen after a regime flag | H3 |
| **S4 lean** (H2.1) | S1, plus the reservation price shifted against the index's ten-minute move | H2.1 |
| **X1 passive reversion** (H2.2) | One-sided passive execution of the frozen §27 signal | H2.2 |

**S1 formula.**

```
r      = mid × (1 − skew_bp × 1e-4 × clamp((σ/σ_ref)², 0.25, 4) × q / Q_soft)
δ_bp   = maker_bp + min_edge_bp + k × σ_bp_1m
bid    = min(floor_tick(r × (1 − δ_bp × 1e-4)), best_bid)     # S1 never improves the touch
ask    = max(ceil_tick (r × (1 + δ_bp × 1e-4)), best_ask)
```

`q` is the position and `Q_soft` the soft limit. A side is quoted only if its
price is within the visible ten levels and the side does not grow inventory past
the soft limit. With a one-tick spread of 1.8 bp and a 2 bp maker fee,
`δ_bp ≥ 2` keeps S1 off the touch of CRV and XRP by construction — the
arithmetic of the design-time table expressed as a rule.

**S2 rule.** When the spread is at least `m` ticks, the side permitted by the
gate quotes `best_bid + 1 tick` (or `best_ask − 1 tick`) if that price still
satisfies `r − bid ≥ δ` (or `ask − r ≥ δ`) and is post-only safe. When the spread
is exactly two ticks only one side can improve: the side that reduces |q|, and
the bid when flat. Arithmetic worth stating before the test: on BICO at a
2.4 bp tick and the base fee, an improved price clears the gate only when the
spread is at least about four ticks (flat inventory: `spread/2 − 1 tick ≥ 2 bp`),
which was about 15% of the time on the one day measured (2024-02-15). H1 is
therefore expected to be decided mostly by the fee tier, and the fee break-even
reports at which tier it would pay.

**S3 rule.** Flags from both detectors (H3 below) open a guard window of `G`
minutes; inside it the action is `pull` (no quotes) or `widen` (δ doubled).
Overlapping windows merge.

**S4 rule.** With `s_t` the index's ten-minute return excluding the instrument
(bp, right-labelled 5 s grid), a trigger fires when `|s_t| ≥ θ`; for the next ten
minutes `r ← r × (1 + λ × β̂ × s_t × 1e-4)` with `s_t` read live. If `one_sided`,
the side that would follow the move is not quoted inside the window.

**X1 rule (H2.2).** At each trigger (with the frozen cooldown), post the fading
side at the touch with clip notional; the entry waits at most 10 minutes; once
filled, the exit is posted at the opposite touch, and whatever is still open
10 minutes after the fill is crossed (taker). No entries in the last 25 minutes
of a day. The taker twin is §27's rule at the same trigger times (`thin` with
hold = cooldown = 120 rows, `TakerCosts(5.5, 0.5)`).

### Fixed in advance, and searched on D

| Parameter | Value | How set |
|---|---|---|
| clip notional | 10% of D's median touch notional, then the adaptive cap below | fixed rule |
| hard limit | soft + one clip | fixed |
| volatility estimator | EWMA of 1 s mid returns, half-life 60 s, scaled to 1 min; `σ_ref` = D median | fixed |
| skew clamp | [0.25, 4] | fixed |
| latency, feed, cancel and arrival rules, crossed policy, fee tier | 10 ms, 0, proportional, pro-rata time, trade tape, base | fixed; swept on H for robustness only |
| warm-up, stop, flatten | 10 min, 23:58, 23:59 | fixed |
| markout and decomposition horizons | 1/5/30 s; 5 s | fixed |
| S0 soft limit | 6 clips | fixed |
| S1 `skew_bp` | {2, 5, 10} | searched on D |
| S1 `k` (volatility multiple) | {0, 0.5, 1, 2} | searched on D |
| S1 `min_edge_bp` | {0, 1, 2, 4} | searched on D |
| S1 `soft_limit_clips` | {3, 6, 12} | searched on D |
| S2 `m` (min spread in ticks) | {2, 3, 4} | searched on D with S1's four axes (432 candidates) |
| S3 detectors and their settings | see H3 | fixed |
| S3 action × `G` | {pull, widen} × {15, 60, 240} min | searched on D (6 cells) |
| S4 `θ` | per instrument, `threshold_for_rate` at the frozen 60 signals/day on D | fitted on D |
| S4 `β̂` | OLS slope, no intercept, of the instrument's next 10-min return on `s_t` over D rows with `|s_t| ≥ θ` | fitted on D |
| S4 `λ` × `one_sided` | {0.5, 1, 2} × {False, True} | searched on D (6 cells) |
| X1 | everything frozen (§27 configuration, touch entry, 10 min timeout) except `θ` | no search |
| Ladder rungs | see the ladder below | fixed |

### The D search

Two `successive_halving` runs, S1 over its 144 cells and S2 over its 432, with
budgets of 5, 12 and 25 D days (a seeded, nested subset of days), objective the
mean daily net in USDT, minimum 20 fills a day; the winner of each is chosen by
`neighbourhood_scores` over the cells measured at the full budget (median of the
cell and its immediate neighbours on the ordered axes), with the outright peak
reported beside it as `pipeline/discovery.py` does. H1's paired twin is S2's
chosen parameters with the inside rule off, so the pair differs in that rule
alone. S3 sits on whichever of S1 and S2 scores higher on D. S4's six cells are
run on all 25 D days for BICO, CRV and XRP, on top of S1's parameters (BICO's
own; for CRV and XRP, BICO's choice expressed in bp). The seed of the day subsets
and the fraction kept at each rung are in the YAML.

---

## Simulator settings a verdict depends on

The book on disk is a 100 ms conflated view, not a message feed, so queue
dynamics within 100 ms are unobservable. The rules below follow from that.

- **Fills only from prints.** A resting order advances in its queue only on
  prints at exactly its price from the side that hits it, and fills on the
  excess beyond the size ahead of it; a print strictly through its price fills
  it at its own limit, up to the through-volume of that sweep. A snapshot alone
  never fills an order (`trade_tape`); `assume_filled` exists only as a
  diagnostic, reported once. Prints sharing a timestamp and aggressor are
  processed in sweep order.
- **Ordering and information.** One merged stream per instrument-day; at equal
  timestamps funding, then activations, then trades, then books. A quoter
  decides only at book events and sees nothing after them. Every grid-derived
  signal enters event time at the end of its bin, because `data/grid.to_grid`
  labels a bin by its start but carries its last value.
- **Queue.** An order joins the tail of the visible size at its price in the
  latest snapshot at or before it goes live; a price better than the touch starts
  at the front. Growth of the level during the arrival interval is split by time
  (`pro_rata_time`, bracketed by `none` and `all_ahead`); later growth is behind
  the order. Cancellations at the level are attributed in proportion to the size
  ahead (`proportional`, bracketed by `pessimistic` and `optimistic`). Quotes
  beyond the tenth visible level are not placed.
- **Latency.** A new order is live and a cancel effective 10 ms after the
  decision; until then the order can still fill. A price change is
  cancel-and-new and loses priority; an unchanged quote keeps it. Feed latency 0.
- **Post-only.** An order at or through the opposite touch on arrival is
  rejected and counted.
- **Size and limits.** Clip = min(clip notional from D, 10% of the trailing
  one-hour median touch), rounded down to the lot; below one lot the side is not
  quoted. Soft limit = `soft_limit_clips` × clip; the side that would grow
  inventory past it is not quoted. Hard limit = soft + one clip; a breach of the
  soft limit by a sweep triggers a taker flatten back to it, walking the last
  snapshot, with 0.5 bp of slippage on top. Own impact is not modelled; the
  clip-to-touch ratio is reported.
- **Fees.** Maker fee on passive fills, taker fee on flattens, in bp of notional;
  a rebate is a negative fee. Base tier 2.0 / 5.5 bp.
- **Funding.** Settled at the instrument's settlement times (8 h: 00:00, 08:00,
  16:00 UTC; the interval is read from the fetched history and asserted) on the
  position held then: payment = −position × mark × rate, mark = mid.
- **Days.** Each day starts flat, quotes nothing in its first 10 minutes or after
  23:58, and is flattened at 23:59 by a costed taker order, so the 00:00 funding
  settlement never applies. After a book gap or a pause over 5 s in snapshots the
  quoter is suspended until the next snapshot. A day missing either plane is
  reported, not simulated. Days are independent and are the cluster for every
  statistic.
- **Accounting.** Average cost; at day end
  `realised + unrealised − fees + funding == cash + position × mark` to 1e-9
  relative, or the run stops.
- **Oracle isolation.** A quoter that reads the future (the ladder's forecast
  rungs) can only run inside the ladder, and its results are stamped as upper
  bounds.
- **Determinism.** No randomness in the simulator. Placebos and the ladder's
  noise draw from `np.random.default_rng(seed)` with the seeds registered in the
  YAML.

---

## Metrics

- `net_usd_day`: spread + adverse + inventory − fees + funding over a day that
  starts and ends flat. Primary.
- `net_bp_turnover`: net over maker plus taker turnover, × 1e4.
- `making_usd_day`: spread + adverse(5 s) − fees, the part of net that is not
  inventory or funding.
- Markouts at 1, 5, 30 s (and 60, 300, 600 s for H2), volume-weighted, by fill
  path, against the market-wide passive benchmark at the same horizons. Per fill,
  `side × (mid(t + h) − p_fill) / p_fill × 1e4`, from the last snapshot at or
  before `t + h`. The benchmark scores every print in the tape from the resting
  side, independently of the simulator.
- Fills per day, fill ratio per order, time quoted per side, RMS and maximum
  |inventory|, flattens, maximum drawdown, clip over touch.
- Statistics from `significance.assess(cluster="day", series="symbol")`: trade
  (fill) level and day level; the day level is the one that decides.
- Minimum detectable effect, recorded in the D amendment: `2 × σ_D(daily net of
  the chosen strategy) / √13`.

The decomposition, per fill: `spread = side × (mid_ref − p) × size` with
`mid_ref` the last snapshot mid at or before the fill;
`adverse = side × (mid(t + 5 s) − mid_ref) × size`;
`inventory = gross trading − spread − adverse` (the residual mark-to-market of
held inventory); `fees`; `funding`. Only `spread + adverse` (the 5 s markout) is
free of the 100 ms staleness; the split between them is indicative, and every
report says so.

---

## Status rules (family of four primaries: H1, H2.1, H2.2, H3)

- **killed** — any of the hypothesis's kill conditions fires. Rejection needs
  no significance, as in `significance.py`.
- **candidate** — no kill condition fires and the day-level one-sided t passes
  Holm's procedure at 5% across the four primaries (12 degrees of freedom).
- **conditional** (H2 only) — candidate on H, and the pre-registered boundary
  prediction on F holds.
- **inconclusive** — no kill condition fires and the significance bar is not
  met. Reported as such, never as a positive.

Three conditions apply to every primary and count as kill conditions:

- **K-pess** — a positive verdict under proportional cancellation attribution
  that is not positive under pessimistic attribution;
- **K-fund** — net minus funding ≤ 0 (the result is funding);
- **K-dir** — `making_usd_day` ≤ 0 while net > 0 (the result is inventory
  drift, not making).

---

## H1 — inside the spread, first in the queue

**Statement.** On MM-admitted instruments over H at the base tier, S2 (S1 plus
one tick inside when the spread is at least `m` ticks and the improved price
clears the edge gate) earns a positive daily net, and more than its twin (S2's
parameters with the inside rule off) on the same days. In the kill conditions,
"S1" is that twin.

**Kill conditions.** (K1) sum of S2's daily net over H ≤ 0; (K2) mean paired
daily difference S2 − S1 ≤ 0; (K3) the 5 s markout plus spread captured per
inside fill is not greater than that of S1's touch fills; (K4) the stale-trigger
placebo gains at least as much over S1 as S2 does; plus K-pess, K-fund, K-dir.

**Placebo.** H1 has no signal to flip. Its falsifications are the paired S1
twin, the pessimistic bracket, and a **stale trigger**: the inside decision
reads the spread state of 60 s earlier (prices still post-only at arrival). If
the gain comes from reacting to a wide spread, the stale version must gain
less.

---

## H2 — the reversion state

[§27](../results.md#27-cross-sectional-reversion-and-the-market-state-it-requires)
measured that the state (the index's ten-minute autocorrelation, −0.10 where
reversion pays) is present throughout D and H and absent in F, and that a
trailing estimate of it does not forecast it a day ahead. Being filled against
the move is the mechanism: a resting order on the fading side fills exactly when
the move it fades is happening.

**H2.1 (lean, market-making framing).** On H2-admitted instruments over H,
S4 earns more than S1 on the same days. Metric: paired daily difference
S4 − S1 pooled over instruments, clustered by day.
Kill: (K1) mean paired difference ≤ 0; (K2) the **flipped** placebo (λ → −λ)
does at least as well; (K3) the true difference does not exceed the 95th
percentile of 50 **shuffled-state** placebos (the signal tape circularly shifted
by a whole number of days in [1, 12] plus an intraday offset drawn uniformly
between one and twenty-three hours, so never less than one hour, seeds 0–49);
plus K-pess, K-fund, K-dir. Prediction on F (BICO): mean paired difference ≤ 0.
If H passes and F behaves as predicted, the status is conditional; if F does not
behave as predicted, it is candidate with the condition unconfirmed and says so.

**H2.2 (passive execution of the frozen signal).** On H2-admitted instruments
over H, X1's net per signal attempt (misses count as zero) exceeds the taker
twin's net per trade on the same triggers, and exceeds zero.
Kill: (K1) mean paired daily difference X1 − taker ≤ 0; (K2) X1's net per
attempt ≤ 0; (K3) the flipped placebo (post the side that follows the move) does
at least as well; (K4) the shuffled-state placebo as in H2.1; (K5) fewer than
100 fills in H, which makes the result inconclusive rather than passed.
Prediction on F (BICO): X1's net per attempt ≤ 0.

**H2.3 (causal state gate, secondary, prior negative).** Days admitted by the
trailing one-day index autocorrelation at or below D's median have a larger
S4 − S1 difference than days it rejects, over H ∪ F on BICO. Kill: admitted mean
≤ rejected mean. Recorded because it is the literal "quote only in the state"
reading; §27 found the same gate failed for the taker in all four windows tried,
and it is not part of the Holm family.

The index autocorrelation on D, H and F is reported beside every H2 number.

---

## H3 — the regime guard

**Detectors, fixed.** (1) `changepoint.detect` on the instrument's 1-second grid
(mid, spread bp) with its defaults (window 2000, threshold 20, reference 40,
minimum gap 20,000, all four statistics), run continuously from 2024-02-01 on a
right-labelled grid; a break at `Break.index` (the exclusive end of the block
that fired it) is effective at the label of row `index − 1`, the moment that
block closed. (2) `structural_breaks.detect_breaks` on one-minute log mid returns
(right-labelled), `MonitorSpec(history_len=250, online_len=60,
statistics=("scale", "dependence"))`, recorded thresholds 7.77 and 7.65
(`RECORDED_THRESHOLDS[(250, 60)]`); a flag is effective at the close that
produced it. Both plus the 10 ms decision latency.

**Statement.** On MM-admitted instruments over H, S3 earns more than the
strategy it guards on the same days.

**Kill.** (K1) mean paired daily difference ≤ 0; (K2) the true difference does
not exceed the 95th percentile of 50 **shifted-flag** placebos (guard windows
circularly shifted by an offset uniform in [6 h, block length − 6 h], count and
spacing preserved, seeds 0–49); (K3) a flag rate on D outside 0.2–24 a day makes
H3 untestable, declared before H is read, with no retuning; (K4) fewer than 5
flags in H makes it inconclusive; plus K-pess, K-fund, K-dir. Reported beside:
the unguarded strategy's net per hour inside and outside the guard windows.

---

## Measurements without a hypothesis

**The advantage ladder**, on H for all four instruments, S0 at the touch, net bp
of turnover and fills per day at each rung, each rung adding to the previous:

| Rung | Change |
|---|---|
| 0 | tail of the queue, base fee, 10 ms |
| 1 | front of the queue (`QueuePriority.FRONT`) |
| 2 | fair value = mid + a forecast of the next 1 s mid change with R² = 0.1 (an oracle blend of the realised change with seeded noise; near the 0.07 implied by queue imbalance's 0.27 correlation with the next second), quoting a side only if it clears the fee |
| 3 | the same at R² = 0.3 |
| 4 | latency 1 ms |
| 5 | perfect foresight of the next 1 s (R² = 1) |
| 6 | the best published Bybit maker tier |

Rungs 2–6 use the future on purpose (rungs 4 and 6 through the forecast they
inherit from the rungs below) and are labelled upper bounds.

**The fee break-even**, per strategy and instrument on H: the maker fee at which
net is zero, by re-running gated strategies over a fee grid (−1.5 to +2.0 bp in
0.25 steps) and interpolating, and analytically for S0. Compared with Bybit's
published linear-perpetual tiers, transcribed into `backtest/costs.py` with the
source and retrieval date at implementation time (base 2.0 / 5.5 bp is already
in the code; the VIP and market-maker programme rows are to be transcribed, not
assumed).

**Robustness on H** (BICO, every strategy): cancellation attribution ×3, arrival
growth ×3, latency {1, 10, 50, 100, 250} ms, feed latency {0, 20, 50} ms, crossed
policy ×2, each axis varied alone with the others at their defaults. Only the
pessimistic cancellation rule enters a verdict (K-pess).

---

## Amendment protocol

After the D runs, and before H is read, one dated amendment is appended and
committed: the admission table, every searched parameter's chosen value with
its neighbourhood, `θ` and `β̂` per instrument, the guard settings, the flag rate
on D, σ of daily net and the minimum detectable effect, and the sha256 of
`configs/mm_prereg.yaml` after the frozen values are written into it. Nothing
else may change. `HeldOutLedger` refuses to run block H unless that file's hash
matches the amendment and the file is committed and clean; the first read writes
`experiments/results/mm_heldout_ledger.json` (commit, config hash, UTC time) and
a second read raises unless explicitly forced, in which case every output of it
is stamped `second read`. F has its own ledger entry, written after H's results
are committed.

## Amendments

### Amendment 1 (2026-10-02, before any result): simulator corrections from code review

**When this was made, no market-making number had been computed on any block**:
no quote, fill, markout or profit of any strategy, on D, H or F. The simulator
had been built and tested on synthetic markets only. Making these corrections
read one development day already disclosed above (BICOUSDT, 2024-02-15), for
three things and nothing else: the offset between the book's and the prints'
clocks (the book lags by about 3 ms; it does not lead), a count of prints
outside the visible book (19 of 15,506 beyond the tenth level, all on the side
they hit and at most three ticks out; none on the wrong side), and the event
loop's speed. The development-period amendment planned above becomes
Amendment 2.

Each correction in one line (the full rules are in
[`docs/market_making_simulator.md`](../market_making_simulator.md)):

1. **Prints before an order's first snapshot.** Print volume at the order's price
   beyond the size ahead goes first to joiners hidden by the conflation
   interval, in the arrival-growth bracket's share (none / time share / all), as a
   running total, never more than the print brought.
2. **A snapshot that lags its prints.** When a snapshot shows the level larger
   than the prints at the price allow, the part of the growth they could explain
   is carried into the next interval, so one print is not also read as a
   cancellation.
3. **Ordering.** Own arrivals and cancels fire only strictly before the next
   market event, so at an equal timestamp funding, prints and snapshots come
   first and an arriving order joins the queue of a snapshot stamped with its
   arrival time. This replaces "funding, then activations, then trades, then
   books" under "Simulator settings a verdict depends on"; the code already did
   this, and the text said otherwise.
4. **Flattens pay latency.** A flatten (past the soft limit, or at 23:59) is a
   reduce-only taker order that arrives after the order latency and walks the
   first snapshot at or after its arrival, plus 0.5 bp and the taker fee. This
   replaces "walking the last snapshot".
5. **The hard limit under a shrinking clip.** At every decision an order kept at
   its price is cancelled and re-placed if it is larger than the current clip
   or its side's exposure exceeds the current hard limit.
6. **Own quotes never meet.** A bid at or above the quoter's own ask is pulled
   apart around the middle, every cancel of a decision is sent before its new
   orders, and an order arriving at or through our own resting opposite order
   is rejected.
7. **Sequence gaps.** The stored book keeps a day's count of sequence gaps but
   not where they fell, so the quoter cannot be suspended from a gap onward. A
   day with more than `max_sequence_gaps` gaps is instead flagged and excluded
   from every verdict, and counted. The threshold is 0 (any gap) unless
   Amendment 2 sets another value from D's gap counts, before H is read. This
   replaces "after a book gap ... the quoter is suspended until the next
   snapshot" for gaps; a pause over 5 s still suspends it.
8. **A stale day-end book.** A flatten with no snapshot within 5 s of its
   arrival walks the first snapshot after the pause; if there is none, the day
   is flagged `flatten_stale` and excluded from every verdict.
9. **No funding plane.** A day simulated without its funding history is flagged
   and excluded from every verdict.
10. **Bad data skips a day, not a run.** A price off the tick grid or a funding
    interval other than 8 h makes that day unavailable, with its reason; the
    other days run.
11. **Off-book prints.** A print more than 25 bp outside the price range of the
    ten visible levels of the latest snapshot at or before it is never offered
    to an own order, and is counted. On the day checked it excluded nothing.
12. **The run cache** is keyed by the files each day read (path, size,
    modification time), the data and funding roots, and a hash of the
    simulator's source, as well as the quoter and the configuration.
13. **Own impact** stays unmodelled, now stated explicitly: an own order does
    not change the book others see, and a second own order reads the volume the
    first took as a cancellation at its level.

[`configs/mm_prereg.yaml`](../../configs/mm_prereg.yaml) is unchanged: none of
the fields it holds changes, and the new settings — `max_sequence_gaps` 0, the
25 bp off-book limit, flatten latency equal to the order latency — are
registered here. Its sha256 remains
`27471cd62ef4bbc67a29467bcff0ea5fda4ddd2c7d00413ca05a9541eac9b272`.
