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

### Amendment 2 (2026-10-03, before any held-out read): values frozen on the development period

**When this was made, no day from 2024-02-26 onward had been read by any
market-making code**: no book, print, funding rate or universe row of blocks H,
buffer, F or P was opened, loaded, simulated or counted. Every reader of the
study checks each day against `prereg.Access` before any file is opened, and the
development run held only `development_access()` (block D). Two things touched
later days without reading them, both required by the protocol: the admission
rule's coverage of H was computed from the file system alone (whether a
non-empty book and print file exists for each H day; no file opened), and the
Bybit funding history was downloaded for D, H and F into the gitignored data
directory, with only D's files opened. The runs are
[`experiments/market_making.py`](../../experiments/market_making.py); their
tables are `experiments/results/mm_*_D.csv`. The values below are written into
the placeholders of [`configs/mm_prereg.yaml`](../../configs/mm_prereg.yaml),
and nothing else in that file changed (with every placeholder put back to
`null` it hashes to the registered `27471cd6…b272`, which the loader checks).

#### Admission (computed on D, 2024-02-01 to 2024-02-25)

| Instrument | Tick (bp) | Spread, time-weighted (bp) | Time at ≥ 2 ticks | Touch / median print | Prints a day | Market-wide passive markout 1 / 5 / 30 s (bp) | Coverage D / H | MM | H2 |
|---|---:|---:|---:|---:|---:|---|---|:-:|:-:|
| BICOUSDT | 2.62 | 6.08 | 0.792 | 12.9 | 14,790 | −0.53 / −0.94 / −0.65 | 1.00 / 1.00 | yes | yes |
| CRVUSDT | 1.96 | 2.13 | 0.082 | 10.7 | 72,163 | −0.59 / −0.75 / −0.86 | 1.00 / 1.00 | no | yes |
| XRPUSDT | 1.88 | 1.88 | 0.000 | 2,462.7 | 263,038 | −0.98 / −0.98 / −0.67 | 1.00 / 1.00 | no | yes |
| BTCUSDT | 0.0209 | 0.0214 | 0.002 | 474.8 | 1,090,765 | −0.44 / −0.53 / −0.40 | 1.00 / 1.00 | no | no |

MM-admitted: **BICOUSDT**. H2-admitted: **BICOUSDT, CRVUSDT, XRPUSDT**.
References: all four. As expected at registration. No D day of any of the four
instruments has a book sequence gap, so `max_sequence_gaps` stays 0
(Amendment 1, item 7).

Per instrument, the clip's notional cap (a tenth of D's median touch notional,
sampled every second) and `sigma_ref` (the median over D's quoting hours of the
one-minute volatility the simulator computes):

| Instrument | Clip notional cap (USDT) | `sigma_ref` (bp, one minute) |
|---|---:|---:|
| BICOUSDT | 20.0434 | 7.92472 |
| CRVUSDT | 55.8147 | 8.04503 |
| XRPUSDT | 3,402.63 | 4.70073 |
| BTCUSDT | 25,256.8 | 3.42176 |

#### The development search (S1 and S2)

Successive halving with the registered budgets of 5, 12 and 25 D days, a nested
subset of D drawn with seed 0 (the 5 days: 2024-02-05 2024-02-11 2024-02-12 2024-02-20 2024-02-25; the 12: 2024-02-03 2024-02-04 2024-02-05 2024-02-07 2024-02-11 2024-02-12 2024-02-17 2024-02-20 2024-02-22 2024-02-23 2024-02-24 2024-02-25), keep
fraction 0.34, objective the mean daily net in USDT on the MM-admitted
instrument, minimum 20 fills a day. S1: 144 cells, 48 kept after 5 days, 16
measured on all 25. S2: 432, 146, 49. The choice is
`validation.search.neighbourhood_scores` over the cells measured at the full
budget: the median of a cell and every measured cell within one step of it on
each ordered axis, ties broken by the cell's own value; the outright peak is
reported beside it.

| | S1 chosen | S2 chosen |
|---|---:|---:|
| `skew_bp` | 2 | 2 |
| `k` | 0.5 | 0.5 |
| `min_edge_bp` | 4 | 4 |
| `soft_limit_clips` | 6 | 6 |
| `m_ticks` | — | 3 |
| Own mean daily net on D (USDT) | +0.0976 | +0.0868 |
| Neighbourhood median (USDT/day) | −0.1497 | −0.1236 |
| Cells in the neighbourhood | 8 | 25 |
| Outright peak at the full budget | 2/0.5/4/6 at +0.0976 | 2/0.5/4/6/2 at +0.0868 |

S1's neighbourhood (`skew_bp`, `k`, `min_edge_bp`, `soft_limit_clips`: mean daily
net in USDT):

| `skew_bp` | `k` | `min_edge_bp` | `soft_limit_clips` | Mean daily net (USDT) |
|---:|---:|---:|---:|---:|
| 2 | 0.5 | 4 | 6 | +0.0976 |
| 5 | 0.5 | 4 | 12 | +0.0469 |
| 2 | 0.5 | 4 | 12 | −0.0459 |
| 2 | 1 | 2 | 6 | −0.1237 |
| 5 | 1 | 2 | 6 | −0.1756 |
| 5 | 1 | 2 | 12 | −0.1827 |
| 2 | 1 | 2 | 12 | −0.1872 |
| 2 | 0.5 | 2 | 12 | −0.5941 |

S2's neighbourhood is in `d_search.neighbourhood.S2` of the YAML (25
cells). On D, `m_ticks` 2, 3 and 4 give identical days for S2's other chosen
values: with a gate near 10 bp, one tick inside clears it only when the spread
is far wider than four ticks, so the inside rule's decisions do not depend on
`m` there. Cells with `m` 2 and 3 tie exactly, on their neighbourhood median and
their own value; the tie went to `m = 3` by the order the search scored them in,
which is deterministic. H1's twin is S2's chosen parameters with the inside rule
off, which here is S1's chosen cell.

On D, S1 earned +0.0976 USDT/day and S2 +0.0868: **S3 guards S1.**

#### The regime guard (S3) and the flags

Flags from both detectors with the registered settings, run on BICOUSDT's book
continuously from 2024-02-01 over D: **222 flags in 25 days, 8.88
a day** (changepoint:autocorrelation 16; changepoint:intensity 31; changepoint:spread 22; changepoint:volatility 30; structural breaks:dependence 6; structural breaks:scale 117). That is inside K3's 0.2 to 24 a day, so **H3 is
testable**, declared here before H is read. On H the flags will be computed by
the same detectors run continuously from 2024-02-01 through H.

S3's six cells, each on all 25 D days, chosen by the mean daily net (the
pre-registration names no neighbourhood rule for six cells):

| Action | `G` (min) | Mean daily net (USDT) | Minus S1, paired (USDT/day) | Passive fills a day |
|---|---:|---:|---:|---:|
| pull | 15 | +0.0822 | −0.0154 | 86.2 |
| pull | 60 | −0.1721 | −0.2697 | 61.8 |
| pull | 240 | −0.1323 | −0.2300 | 14.4 |
| widen | 15 | +0.0988 | +0.0012 | 86.6 |
| widen | 60 | −0.1035 | −0.2011 | 65.4 |
| widen | 240 | −0.1754 | −0.2730 | 26.6 |

Chosen: **guards S1, action `widen`, `G` = 15 minutes.**

#### The reversion signal (S4, X1, H2.3)

The tape is §27's frozen signal on a right-labelled five-second grid built from
D's universe files only (26 instruments, 432,000 rows); the index
autocorrelation over D is −0.0327.

| Instrument | θ (bp) | β̂ | Labels with \|s\| ≥ θ a day | Triggers a day (thinned) | X1 attempts on D | Taker twin on D: trades, net bp a trade |
|---|---:|---:|---:|---:|---:|---:|
| BICOUSDT | 98.3829 | −0.254135 | 60.0 | 1.92 | 34 | 48, −0.87 |
| CRVUSDT | 97.7647 | −0.146584 | 60.0 | 1.96 | 43 | 49, +5.85 |
| XRPUSDT | 99.7363 | −0.134149 | 60.0 | 1.92 | 48 | 48, −6.01 |

The thinned triggers reproduce §27's search block: it took 48 to 51 trades per
instrument on its 25-day search block at the frozen configuration
([`market_reversion.csv`](../../experiments/results/market_reversion.csv)); the
taker twin takes 48, 49 and 48 here.

S4's six cells on all 25 D days for the three H2-admitted instruments, on S1's
chosen parameters, chosen by the mean paired daily difference S4 − S1 pooled
over the instruments:

| λ | `one_sided` | S4 − S1 pooled (USDT/day) | BICOUSDT | CRVUSDT | XRPUSDT |
|---:|---|---:|---:|---:|---:|
| 0.5 | false | +12.007 | +0.190 | +0.309 | +11.508 |
| 0.5 | true | +10.628 | +0.162 | +0.363 | +10.102 |
| 1 | false | +23.236 | +0.109 | +0.689 | +22.438 |
| 1 | true | +22.941 | +0.071 | +0.652 | +22.218 |
| 2 | false | −12.069 | +0.132 | +0.357 | −12.559 |
| 2 | true | −10.206 | +0.123 | +0.380 | −10.709 |

Chosen: **λ = 1, `one_sided` = false.** The pooled difference is in USDT and
is dominated by XRPUSDT, whose clip cap is about 170 times BICOUSDT's: the
registered clip rule sizes every instrument by its own touch, so the pooled
metric weights instruments by their touch notional. This is the registered
metric, stated here so the verdict on H is read with it.

H2.3's gate threshold, the median over D of the trailing one-day index
autocorrelation known at each day's open (24 days), is **−0.0541**.

#### Spread of daily results on D, and the minimum detectable effect

`2 × σ_D / √13`, from `experiments/results/mm_power_D.csv`. The YAML's single
value is that of the chosen strategy (S1, which S3 guards); the others are
recorded here for each primary.

| Quantity | Primary | Unit | Days | Mean on D | σ_D | MDE on H (2σ/√13) |
|---|---|---|---:|---:|---:|---:|
| S1 (the chosen strategy, guarded by S3) daily net | the registered MDE | USDT/day | 25 | +0.098 | 1.606 | 0.891 |
| S1 daily net | H2.1's comparator on the MM-admitted instruments | USDT/day | 25 | +0.098 | 1.606 | 0.891 |
| S2 daily net | H1 K1 | USDT/day | 25 | +0.087 | 1.602 | 0.889 |
| S2 − twin, paired daily | H1 K2 | USDT/day | 25 | −0.011 | 0.055 | 0.031 |
| S3 − guarded, paired daily | H3 K1 | USDT/day | 25 | +0.001 | 0.355 | 0.197 |
| S4 − S1, paired daily, pooled over H2 instruments | H2.1 K1 | USDT/day | 25 | +23.236 | 125.469 | 69.598 |
| X1 net per attempt (misses zero), pooled | H2.2 K2 | bp/attempt | 19 | −2.515 | 5.143 | 2.853 |
| X1 − taker twin, paired daily, pooled | H2.2 K1 | bp/attempt | 19 | −8.844 | 39.571 | 21.950 |

Two consequences, stated before H is read. H1's paired difference is tiny on D
because the inside rule rarely fires at the chosen gate (above), so H1 is
expected to turn on the fee tier, as the pre-registration anticipated. H2.2
needs 100 fills in H (K5); on D, X1 made 125 attempts over 25 days on
three instruments, every one filled: about 5.0 a day across the three. At
that rate H's 13 days would give about 65 attempts, and H2.2 would be
inconclusive by K5 unless H's trigger rate is higher than D's.

#### Corrections and choices made in the code since Amendment 1

None changes a registered value; each is disclosed because it is a choice the
registration did not spell out, or a correction to how data is read.

1. **A day reads only its own rows.** The book and universe files on disk carry
   up to five rows stamped just after the next midnight. Every reader now drops
   rows stamped outside the day (`events.within_day`), so a day's simulation
   never reads the next day and D's last day never reads H. Found while building
   this amendment's readers, before any run.
2. **The minimum of 20 fills a day** is applied to the mean passive fills a day
   over the days a cell was measured on (flattens excluded).
3. **The neighbourhood** is the block of measured cells within one step on every
   axis (up to 3^4 or 3^5 cells), the generalisation of `pipeline/discovery.py`'s
   two-axis block; cells below the fill minimum at the full budget take no part.
4. **S3 and S4** are chosen by their objective alone (six cells each).
5. **X1** acts on triggers known between 00:10 (the end of the simulator's
   warm-up) and 23:35 (no entries in the last 25 minutes); the taker twin is
   scored on exactly the same triggers. The exit clock starts at the first
   decision that sees the fill, within one snapshot of it. The exit cross is a
   reduce-only taker order priced as a flatten.
6. **H2.3's trailing autocorrelation** uses only pairs whose forward return ended
   before the day opened (`known_at_open=True`); §27's version reads the first
   ten minutes of the gated day.
7. **The simulation runs** each instrument-day job in a fresh process (results
   do not depend on it; it bounds memory).

#### The compute record

The development run was made twice, from commit `0dcb596` and from `2c1fa62`
(which only runs each job in a fresh process); every table and every frozen
value came out byte-identical, and the amendment's date was then set to the day
it is committed. The second run, whose outputs are committed, took
32 minutes of compute on 8 worker processes of a 10-core, 32 GB machine (stage
times on the monotonic clock, which stops while the machine sleeps; the run's
wall time was longer because the machine slept during it):

| Stage | Wall time (s) |
|---|---:|
| screen | 17 |
| reversion tape | 15 |
| flags | 34 |
| search S1 | 370 |
| search S2 | 1,111 |
| chosen strategies | 14 |
| S3 | 40 |
| S4 | 308 |
| X1 | 28 |
| total | 1,937 |

Peak resident memory: 1,671 MB in the largest simulation worker, 1,740 MB
in the largest worker of any kind (one of the screen's, which read every
instrument's days, BTCUSDT's included), 1,417 MB in the driver. The first run's largest worker reached 1,955 MB, before each job had a
fresh process; both are above the 1.5 GB a worker the run was budgeted for, and
eight such workers stay well inside the machine.

Frozen configuration sha256: `ed79d5ff520f5e93e9a8ea142058305117b530041534b1dc115b9e3507379c9d`

---

## Results on block H (read once, 4 October 2026)

**Block H was read once**, through `HeldOutLedger.open("H")`, from a clean tree
at commit `2a7daaa` on branch `mm/heldout`, with
`configs/mm_prereg.yaml` at the frozen hash recorded in Amendment 2. The ledger
entry ([`mm_heldout_ledger.json`](../../experiments/results/mm_heldout_ledger.json)):
first read, commit `2a7daaa918b2d9e93b8b630d3a0598e2f4773639`, configuration
`ed79d5ff…379c9d`, 2026-10-04T12:34:52Z. Every value used is the frozen one; no
parameter, threshold, seed or rule was changed, and no test was added after the
read. The scripts that made the read
([`market_making_heldout.py`](../../experiments/market_making_heldout.py)) and
turned its outputs into the verdicts
([`market_making_verdicts.py`](../../experiments/market_making_verdicts.py)) were
committed at the ledger's commit and ran unchanged. All 13 days of all four
instruments were simulated; none had a book sequence gap, a stale flatten or a
missing funding plane, so no day was excluded. Blocks F and P were not touched.

**All four primaries are killed.** None passes Holm's procedure either; the
kill conditions decide it without significance, as the status rule says.

| Hypothesis | Metric (unit) | Value on H | Day t (12 df) | p, one-sided | Holm | Kill conditions that fired | Status |
|---|---|---:|---:|---:|:-:|---|---|
| H1 | S2 − twin, paired daily (USDT/day); and S2's daily net | −0.0072; S2 −1.21 a day, −15.75 over H | −0.93; S2 −1.51 | 0.921 | fail | K1, K2, K4, K-fund | **killed** |
| H2.1 | S4 − S1, paired daily, pooled over BICO, CRV, XRP (USDT/day) | −91.81 | −0.55 | 0.705 | fail | K1, K3, K-fund | **killed** |
| H2.2 | X1 − taker twin, net per attempt, paired daily (bp); and X1's net per attempt | −9.96; X1 −4.92 | −1.78; X1 −4.53 | 1.000 | fail | K1, K2, K3, K4, K-fund | **killed** |
| H3 | S3 − S1, paired daily (USDT/day) | +0.107 | +0.62 | 0.272 | fail | K2 | **killed** |
| H2.3 (secondary) | S4 − S1 on BICO, admitted against rejected days | H only: +0.575 (10 days) against +0.488 (3 days) | — | — | — | — | **not decided**: registered over H ∪ F; F is read next |

Every kill condition, mechanically, with the value it read and what it was
compared with ([`mm_kill_conditions_H.csv`](../../experiments/results/mm_kill_conditions_H.csv)):

| | Condition | Value | Against | Fired |
|---|---|---:|---:|:-:|
| H1 | K1: sum of S2's daily net over H ≤ 0 | −15.75 | 0 | yes |
| | K2: mean paired daily S2 − twin ≤ 0 | −0.0072 | 0 | yes |
| | K3: 5 s markout per inside fill not above the twin's touch fills (bp) | −5.52 (6 fills) | −16.80 (11 fills) | no |
| | K4: the stale-trigger placebo gains at least as much over the twin | −0.0005 | −0.0072 | yes |
| | K-pess | S2 −16.18; S2 − twin +0.0003 | | no (the default verdict is not positive) |
| | K-fund: net minus funding ≤ 0 | S2 −15.61; S2 − twin −0.0067 | 0 | yes |
| | K-dir | making −10.86 with net −15.75 | | no |
| H2.1 | K1: mean paired difference ≤ 0 | −91.81 | 0 | yes |
| | K2: the flipped placebo (λ → −λ) does at least as well | −141.82 | −91.81 | no |
| | K3: not above the 95th percentile of 50 shuffled-state placebos | −91.81 | +8.84 | yes |
| | K-pess | −78.15 | | no |
| | K-fund | −93.01 | 0 | yes |
| | K-dir | making −45.84 with net −91.81 | | no |
| H2.2 | K1: mean paired daily X1 − taker ≤ 0 (bp) | −9.96 | 0 | yes |
| | K2: X1's net per attempt ≤ 0 (bp) | −4.92 | 0 | yes |
| | K3: the flipped placebo (following the move) does at least as well per attempt | −4.33 | −4.92 | yes |
| | K4: not above the 95th percentile of 50 shuffled-state placebos | −9.96 | +21.32 | yes |
| | K5: fewer than 100 fills in H (inconclusive) | 275 filled attempts | 100 | no |
| | K-pess | −10.24; −5.14 | | no |
| | K-fund (the attempt net holds no funding) | −9.96; −4.92 | 0 | yes |
| | K-dir, on X1's own totals (USDT over H) | making −105.4 with net −113.3 | | no |
| H3 | K1: mean paired daily difference ≤ 0 | +0.107 | 0 | no |
| | K2: not above the 95th percentile of 50 shifted-flag placebos | +0.107 | +0.562 | yes |
| | K3: flag rate on D outside 0.2–24 a day (declared before H) | 8.88 | | no: testable |
| | K4: fewer than 5 flags in H (inconclusive) | 173 | 5 | no |
| | K-pess | +0.092 | | no |
| | K-fund | +0.108 | 0 | no |
| | K-dir | making +0.059 with net +0.107 | | no |

#### What each verdict rests on

- **H1.** The inside rule almost never acts at the frozen gate: S2's orders
  placed inside the spread were filled 6 times in 13 days (the twin, none), so
  S2 and its twin differ by less than a cent a day, and both lose about
  1.2 USDT a day on BICOUSDT. The stale-trigger placebo loses no more than S2
  over the twin. On D the same arithmetic was stated in advance (Amendment 2):
  with a gate near 10 bp, one tick inside clears it only when the spread is far
  wider than four ticks.
- **H2.1.** S4 − S1 by instrument: BICOUSDT +0.55, CRVUSDT −2.70, XRPUSDT
  −89.66 USDT a day. The pooled metric is in USDT and, as Amendment 2 stated,
  weighted by XRPUSDT's clip; BICOUSDT alone is positive but small against its
  own spread of days, and the pooled difference is below most of its 50
  shuffled-state placebos (median −79.7, 95th percentile +8.8).
- **H2.2.** Passive execution of the frozen signal loses where the taker
  version wins. Over 376 eligible triggers (125 BICOUSDT, 129 CRVUSDT, 122
  XRPUSDT), X1 posted 275 attempts, every one of which filled, and skipped 101
  that arrived while an attempt was open: its net is −4.92 bp per attempt
  (−5.72, −4.86, −4.15 by instrument), against +6.38 bp per trade for the taker
  twin on the same triggers (+12.90, +3.23, +3.02). The fading side fills, as
  the mechanism says it would, but at a markout worse than the market-wide
  passive benchmark, and following the move instead (the flipped placebo) does
  slightly better. K5 did not fire: H's trigger rate (9.6 a day across the
  three, against about 5 on D) gave 275 fills.
- **H3.** The guard does what it was built to do — the unguarded S1 lost
  0.115 USDT an hour inside the 15-minute guard windows and 0.040 outside — and
  S3 earned +0.107 USDT a day more than S1. But 50 copies of the same flags
  moved around the block earn as much on average (+0.146) and more at the 95th
  percentile (+0.562): the gain is not specific to when the flags fired.
  173 flags in H, 13.3 a day, against 8.9 on D.
- **H2.3** reads H and F together and is decided after F. On H's 13 days, the
  gate (trailing one-day index autocorrelation at or below −0.0541) admitted
  10, with a mean S4 − S1 on BICOUSDT of +0.575 USDT, against +0.488 on the 3
  it rejected.

The index's ten-minute autocorrelation was −0.033 on D and −0.135 on H (the
state is present on both, as §27 measured); on F it is §27's −0.0003, not yet
read here.

#### The measurements without a hypothesis

**Every strategy on BICOUSDT, the MM-admitted instrument, over H**
([`mm_strategies_H.csv`](../../experiments/results/mm_strategies_H.csv); USDT a
day, the decomposition with fees as a cost):

| Strategy | Net | Net, bp of turnover | Passive fills a day | RMS position (BICO) | Spread | Adverse (5 s) | Inventory | Fees | Funding |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 touch | −43.41 | −5.00 | 12,277 | 103 | +21.49 | −47.26 | −0.30 | 17.37 | +0.03 |
| S1 skew | −1.20 | −7.50 | 192 | 56 | +2.45 | −2.96 | −0.35 | 0.33 | −0.01 |
| S2 inside | −1.21 | −7.54 | 192 | 56 | +2.45 | −2.96 | −0.37 | 0.33 | −0.01 |
| S3 guard | −1.10 | −7.93 | 164 | 58 | +2.02 | −2.51 | −0.30 | 0.28 | −0.01 |
| S4 lean | −0.65 | −2.64 | 316 | 72 | +2.43 | −3.15 | +0.58 | 0.50 | −0.01 |
| X1 passive reversion | −0.06 | −8.14 | 10 | 0.5 | +0.02 | −0.05 | −0.02 | 0.01 | 0.00 |

Only the 5 s markout (spread plus adverse) is free of the book's 100 ms
staleness; the split between the two is indicative. Markouts by fill path
([`mm_markouts_H.csv`](../../experiments/results/mm_markouts_H.csv)): on
BICOUSDT a fill from the queue was worth +4.3 bp at 5 s to S1 and −2.1 bp to
S0, a fill by a print trading through the order −4.6 and −3.8 bp; against a
market-wide passive markout of −1.36 bp. Every strategy's passive fills taken
together are worse than the market-wide benchmark at 5 s on every instrument
except CRVUSDT, where S1 (+0.07 bp) and S4 (+0.36 bp) are above its −0.48.

Outside the MM-admitted instrument, S1 ran only as H2.1's comparator. It made
+0.62 USDT a day on CRVUSDT and +70.79 on XRPUSDT, and on both the making part
was negative (−0.55 and −21.66 a day) and the profit was inventory (+1.21 and
+96.02 a day): thirteen days of carried position, not quoting. No hypothesis
was registered about it and none is drawn.

**The advantage ladder**, S0 on all four instruments, net in bp of turnover
([`mm_ladder_H.csv`](../../experiments/results/mm_ladder_H.csv)); rungs 2 to 6
read the future and are upper bounds, not strategies:

| Rung | BICOUSDT | CRVUSDT | XRPUSDT | BTCUSDT |
|---|---:|---:|---:|---:|
| 0 tail of the queue, base fee, 10 ms | −5.00 | −3.89 | −3.49 | −3.14 |
| 1 front of the queue | −4.00 | −3.35 | −2.78 | −2.63 |
| 2 forecast of the next second, R² 0.1 | −1.91 | −0.70 | +3.33 | +13.30 |
| 3 R² 0.3 | −0.20 | +0.91 | +2.00 | +3.83 |
| 4 latency 1 ms | −0.04 | +1.01 | +2.13 | +3.79 |
| 5 perfect foresight of the next second | +5.66 | +4.26 | +2.92 | +2.31 |
| 6 best published maker tier (0 bp) | +2.77 | +2.36 | +2.13 | +0.99 |

Being first in the queue, with no other change, loses more in USDT on every
instrument (it is filled about twice as often, and the extra fills are the
adverse ones) while losing less per unit traded. On BICOUSDT no rung short of
perfect foresight of the next second turns the touch quoter positive.

**The fee break-even**: the maker fee at which each strategy's net over H is
zero ([`mm_fee_breakeven_H.csv`](../../experiments/results/mm_fee_breakeven_H.csv),
grid in [`mm_fee_grid_H.csv`](../../experiments/results/mm_fee_grid_H.csv)).
S0 and X1 decide without reading the fee, so theirs is exact from one run; the
gated strategies were re-run at every fee of the grid, the gate reading the fee
it pays, the taker fee held at 5.5 bp. "Below the grid" means the strategy
loses at every fee down to a 1.5 bp rebate; "above" that it already pays at the
base fee.

| Strategy | BICOUSDT | CRVUSDT | XRPUSDT | BTCUSDT |
|---|---:|---:|---:|---:|
| S0 | −3.01 | −1.89 | −1.49 | −1.14 |
| S1 | below the grid | above (pays at the base) | above (pays at the base) | |
| S2 | −1.48 | | | |
| S3 | below the grid | | | |
| S4 | −0.52 | +0.62 | +1.70 | |
| X1 | −6.14 | −2.04 | −1.57 | |

Bybit's published linear-perpetual schedule, transcribed into
`backtest/costs.py`, has no maker rebate: its best maker fee is 0 bp (Supreme
VIP, from 500 million USDT of 30-day volume, or the Pro levels, from 100 million
with more than a fifth of it through the API), and the market-maker programme
advertises a rebate of up to 1 bp, on application.
No strategy on BICOUSDT, the instrument the study admitted for market making,
breaks even at any published fee, nor at the programme's rebate. Of the
strategies run on the other instruments, only S4 on CRVUSDT (at a 0 bp tier)
and on XRPUSDT (from VIP 2, 1.6 bp) would have broken even, and S1 there, whose
profit was inventory (above). The caveat the code carries with every tier: the
schedule is the one published in 2026 (page last updated 2026-09-02, transcribed
2026-10-02), applied to data from February and March 2024, when it may have
differed; and a better tier also lowers the taker fee, which the grid holds at
the base.

**Robustness on BICOUSDT**, each axis alone
([`mm_robustness_H.csv`](../../experiments/results/mm_robustness_H.csv)): under
every cancellation attribution, arrival-growth rule, latency from 1 to 250 ms,
feed latency up to 50 ms and the `assume_filled` diagnostic, every strategy's
net stays negative (S0 −39.8 to −57.7 USDT a day, S1 −1.03 to −1.51, S2 −1.03
to −1.52, S3 −0.99 to −1.22, S4 −0.56 to −1.03, X1 −0.06 to −0.17). Only the
pessimistic cancellation rule enters a verdict, through K-pess, and it never
turned a positive verdict negative, because no default verdict was positive.

#### How the registered statistics were read

Each choice below was committed in the scripts at the ledger's commit, before H
was opened, and none was changed after.

1. **Daily series.** A paired difference is taken per instrument-day where both
   days are usable, and pooled by summing over instruments each day, as on D.
2. **Two claims, one test.** H1 ("a positive daily net, and more than its
   twin") and H2.2 ("exceeds the taker twin, and exceeds zero") make two claims
   each, and each is tested at the larger of its two one-sided p-values.
3. **K-pess, K-fund and K-dir** are applied to every sign claim of a
   hypothesis. K-pess fires only when every claim is positive under
   proportional attribution and one is not under pessimistic attribution.
4. **H1's K3** compares the volume-weighted 5 s markout of S2's fills from
   orders placed inside the spread with that of the twin's fills from orders
   placed at the touch, each against its side's touch in the snapshot the
   order arrived on.
5. **H2.2's net per attempt** is the cash of an attempt's fills over the
   notional it posted, misses and triggers skipped while busy counting zero,
   over the eligible triggers (Amendment 2's definition on D). It holds no
   funding, so K-fund reads the claim itself; K-dir reads X1's own day totals,
   since the taker twin has no making part. The flipped placebo is compared on
   net per attempt.
6. **The shuffled-state placebo** rolls H's part of the index tape on its full
   five-second grid by the registered shift (seed 0 to 49): S4 reads the rolled
   tape and its trigger times; X1 the triggers re-derived from it, thinned from
   the start of the block as the real ones are; the taker twin is scored on
   those triggers; the statistic is the hypothesis's own.
7. **The shifted-flag placebo** moves the flags effective inside H around H's
   circle by an offset uniform in [6 h, 13 days − 6 h] (seeds 0 to 49).
8. **The ladder's noise** is drawn per instrument-day from the registered seed,
   so rungs 2 to 4 see the same draws; the forecast is `R² × (realised change +
   noise)`, whose R² against the realised change is the rung's.
9. **Robustness** runs S0 to S4 and X1; the twin is S1's cell and was not run
   twice.

#### What could not be done as registered

- **H2.3** is registered over H and F together, so its status waits for F; only
  the H half is reported above.
- **Memory.** The largest simulation workers reached 2,320 MB on a BTCUSDT day
  and 2,210 MB on an XRPUSDT day, above the 2 GB a worker was budgeted for;
  BICOUSDT and CRVUSDT stayed under 1.6 GB. Nothing was cut to fit.

#### The compute record

One run, 4,958 seconds of wall time on 8 worker processes of a 10-core, 32 GB
machine: the reversion tapes 27 s, the flags 53 s, and 1,872 simulation jobs
(7,566 configuration-days, every day of every cell) 4,877 s. Peak resident
memory 2,320 MB in the largest worker and 2,748 MB in the driver
([`mm_compute_H.csv`](../../experiments/results/mm_compute_H.csv)).
