# Synthetic data with planted answers, and the checks that read it

A backtest that finds an edge proves little until the same code has been shown
to find an edge of known size where one was put, to find nothing where none
was, and to lose it when the decisions are moved away from the market they were
made on. This page describes the generators that put the answers in
(`src/trading_research/synthetic/`), the checks that run the repository's own
strategies, detectors and simulator on them (`trading-research verify`), and
what the last run of every check found.

Everything here is generated. Every row is stamped `source="synthetic"`, every
dataset carries a `truth.json` with its generator, spec, seed and planted
parameters, and nothing computed on it is evidence about a market: it is a
statement about the code that read it.

## The generators

Each has a typed, frozen spec with defaults that generate in seconds, a seed
(the same seed and spec give the same bytes), a null, and a docstring stating
what its data can and cannot prove.

| Command | Module | Planes written | What is planted | The null |
|---|---|---|---|---|
| `trading-research synthetic book-days` | `book_days` | book (10 levels, 100 ms), trades | episodes in which queue imbalance (or signed flow, `--feature flow`) shows a direction and the expected log mid then moves `size_bp` that way over `horizon_s`, at a rate decaying with half-life `decay_s`, over random-walk noise | the same episodes and feature, the move's direction drawn independently of the feature and made in the single step after the onset |
| `trading-research synthetic break-days` | `break_days` | book, trades | the same stream over whole days; at `break_at_s` every part of the return becomes `vol_ratio` times larger, the spread widens, the edge flips sign | the same draws without the break |
| `trading-research synthetic cross-section` | `cross_section` | bars (5 s, a file per instrument-day) | a universe whose log prices carry market and idiosyncratic transients (AR(1), half-life `half_life_bars`): reversal, or momentum with `--variant momentum` | the transients replaced by random walks with the same step variance |
| `trading-research synthetic order-flow` | `order_flow` | book, trades, the order log | an order-level book (Poisson arrivals, cancels uniform or biased in the queue, market orders through the touch); after a print the book moves against its maker with probability `pi`: an adverse selection of `adverse_bp` | no adverse selection, the same number of moves as noise |
| `trading-research synthetic funding` | `funding` | funding (a file per day, the venue's layout) | an AR(1) rate per 8 h settlement with persistence `persistence` | the same mean and spread, persistence zero |
| `trading-research pairs synthetic` | `pairs` | bars (1 h, the klines layout `<dir>/<SYMBOL>/klines-1h/`) | two legs with `log a = alpha + beta log b + s`, `b` a random walk, `s` an AR(1) of half-life `half_life_bars`: a cointegrated pair | `--variant independent`: two unrelated random walks of the same volatility; `--variant break`: cointegrated until halfway, then `s` a random walk |

Book and trades planes are written in the day-partitioned layout
`trading-research validate-data` discovers, and every generator validates its
frames with `data.validate` before it writes them; the tests run
`validate-data` on the written directories. Bars and funding are planes the
command does not discover (it reads books and trades only): bars are checked
with `validate_bars`, funding with `bybit_funding.load_day` and
`settlement_interval_h`, in the tests.

Defaults, on this laptop, generated and written: book days 16 sessions of an
hour (52 MB) and break days two whole days at 100 ms (1.7 million snapshots,
131 MB) in a few seconds each; the cross-section, 20 instruments over 16 days,
in about twelve seconds (388 MB); order flow, four hours, in about five
seconds; funding, a year, in under a second. `--days N` scales each.

### Causality

The one property every generator must get right is that nothing observable at
a moment depends on anything after it. In `book_days` the feature shows at the
onset snapshot, the prints that show it are stamped in the interval that
snapshot closes (the backward rule of `features.prints`), and the drift moves
the mid only from the next snapshot on; a test pins all three. In
`order_flow` the move after a print comes after the print, and an own order's
truth (below) applies market events stamped at its arrival before it, as the
simulator does.

### The order-level truth

The market-making simulator cannot see the queue: it estimates the size in
front of an order from 100 ms snapshots and the prints (rules R5 to R10 of
[`market_making_simulator.md`](market_making_simulator.md)). `order_flow`
simulates every order, so it knows the queue. `order_flow.true_fills` replays
the order log with own orders that change nothing (the simulator's own
assumption, R11): an order joins the tail of the true queue at its arrival, a
cancel of an order that arrived before it shortens the queue in front by that
order's remaining size, a print at its price consumes the queue in front first
and fills it with the excess, and a print through its price fills it. That is
the answer the simulator's estimate is checked against.

## The checks

`trading-research verify edge|detector|simulator|reversion|pairs|all` writes
`artifacts/verify/<what>/page.md` and `result.json` (each criterion with the
planted value, the recovered value, the tolerance and pass or fail, and the
seeds) and exits 1 when a binding criterion fails. `verify all` adds
`artifacts/verify/all/page.md`, the table below.

The significance bar is the repository's own (`evaluation.significance`): an
effect is found when its day-clustered t is at least 3; nothing is found when
its absolute t is below 3; a t on fewer than 30 trades or 5 days is not read
either way. Every check compares, trade by trade, what was recovered with what
the planted effect owed that very trade (read from the generator's latent
state), and requires the difference, the residual, to have an absolute
day-clustered t below 3. For a causal strategy the residual is noise whatever it
trades; a strategy that reads the future picks trades whose noise is
favourable, and its residual is positive, so the residual is a leak detector as
well as a bias check (`test_a_strategy_that_reads_the_future_is_caught_by_the_residual`).

| Check | What runs | What it proves | What it does not prove |
|---|---|---|---|
| `edge` | the registry's book features and `features.flow` on the backward-bucketed prints; the three-class label at the round-trip cost; a rule or model from `pipeline.stages.build_model` fitted on the first 6 days; `backtest.evaluate.choose_confidence` on the next 2; `backtest.taker.execute_day` (10 ms latency, one position at a time, hold = the planted horizon, the stated fee) on the last 8; placebo: each test day's decisions at the same times of day on every other test day | that the pipeline finds a planted drift of known size, measures what its trades were owed without bias, finds nothing significant on the null, and loses the effect under the shifts | that any market carries such a drift, that real features show it this clearly, or that the stated fee, the latency and the generator's spread and depth are real |
| `detector` | the market-making study's registered detectors (`market_making.flags`): `changepoint.detect` with its defaults on the right-labelled one-second grid, and `structural_breaks.detect_breaks` (history 250, online 60, scale and dependence at 7.77 and 7.65) on one-minute returns; three planted paths and three nulls | how long each takes to flag a change of known size and time, and how often it flags one where none was planted | how they behave on a market, whose changes are gradual, several at once and of no known time |
| `simulator` | `market_making.simulator.simulate_day` with the touch quoter S0 (markouts) and `simulator_check.ProbeQuoter`, which posts one lot at each touch and holds it a fixed time whatever happens (fills), under each cancellation bracket; a planted flow, its null, and a queue-stress flow with almost no price moves | that the simulator's post-fill markouts recover a planted adverse selection and find none on the null, and that its fills, estimated from snapshots, match the order-level replay of the same orders, with its brackets on either side | that real queues and informed flow behave like these Poisson flows, or that a quoter would earn anything |
| `reversion` | `strategies.relative.backtest` (demeaned, basket 3 a side, lookback and hold one half-life) and the frozen `strategies.reversion` configuration (index of the others over 120 bars, threshold for 60 trades a day set on the first 3 days, `execution.thin`); three paths of reversal, momentum and null | that both rules recover a planted reversion, lose what a planted momentum takes, and find nothing on the null, each without bias | that real instruments revert, at what half-life, or that the effect survives costs |
| `pairs` | `pairs.screen.screen` on the first formation window of a planted and an independent pair together (six pairs, Holm at 5%); `pairs.backtest.backtest` walk-forward (formation 720 bars, trading 168, z bands 2 / 0.5 / 4, no gate, next-bar-open fills, taker fee and slippage on both legs) on the planted pair, the independent walks and a pair that breaks halfway; three paths; the backtest re-run with every price after the middle bar scrambled | that the screen ranks a planted cointegration first and does not pass independent walks after correction, that the hedge ratio and half-life come back, that the backtest earns the planted reversion before costs and nothing significant on the nulls, and that no decision reads a later bar | that any real pair cointegrates, at what half-life, or that a relation found on one window holds on the next |

Adverse selection is measured against the true mid at the fill: the half-spread
a fill captured against the true mid, less its five-second markout (the
simulator's `markout_5s`). The simulator's own split (R18) measures the
captured spread from the last snapshot's mid, which can be 100 ms stale, so its
adverse part also carries any move between that snapshot and the fill; the
check reports it beside, unbound, as the simulator's page says it should be
read.

The fill-rate criteria read the planted (proportional) setting against the
order-level replay within two percentage points; the bracket criterion asks
that the pessimistic attribution fill no more, and the optimistic no less, than
the truth, give or take two standard errors.

## Results of `verify all`

Run on 28 October 2026 with `trading-research verify all` (default sizes and
seed 20200106; about three minutes on the laptop). It exits 1: **simulator
and reversion pass; edge and detector fail**, each failure in an existing
component, reported here and not changed. Every number is in
`artifacts/verify/<what>/result.json` after a run, which is not committed.

**edge** (planted 16 bp over 30 s, taker fee stated at 5.5 bp a side; 8 test
days; "owed" is what the planted drift owed the same trades)

| Feature | Strategy | Recovered (t) | Owed per trade | Residual t | Null (t) | Placebo | Result |
|---|---|---|---|---|---|---|---|
| imbalance | `order_flow` (rule on queue imbalance) | 15.03 bp (84.7), 330 trades | 14.95 bp (93%) | 0.25 | -0.46 bp (-2.09) | -0.25 bp, max shift t 1.67 | pass |
| imbalance | `spread_capture` (rule) | -0.11 bp (-0.19), 272 trades | 0.83 bp (5%) | **-3.48** | -0.34 bp (-1.08) | -0.06 bp | **FAIL** |
| imbalance | `logistic` | 15.02 bp (65.8), 330 trades | 14.89 bp (93%) | 0.46 | -2.11 bp (-0.88) | -0.28 bp, max 1.65 | pass |
| imbalance | `lightgbm` | 13.12 bp (18.1), 212 trades | 12.79 bp (80%) | 0.63 | 0.57 bp (1.20) | -0.66 bp, max 0.12 | pass |
| flow | `order_flow:flow_imbalance_24` (rule) | 0.66 bp (2.11), 910 trades | 0.77 bp (5%) | -0.42 | 0.14 bp (0.32) | 0.04 bp | **FAIL** |
| flow | `logistic` | 13.67 bp (21.8), 53 trades | 13.79 bp (86%) | -0.22 | -0.19 bp (-0.12) | -0.15 bp, max 2.26 | pass |
| flow | `lightgbm` | 14.06 bp (45.1), 159 trades | 14.35 bp (90%) | -1.13 | -0.11 bp (-0.33) | -0.90 bp, max 0.31 | pass |

Net of the stated costs (a spread of about 2 bp and 11 bp of fees) `order_flow`
and `logistic` on imbalance kept 1.9 bp a trade and the others less than 1 bp:
the planted 16 bp is sized to be found against those costs, not to resemble a
market.

* `spread_capture` fails, and its residual is the finding. Its tight-spread
  test (`models.classical.SpreadCapture.signal`) is `spread <=
  nanquantile(spread, 0.3)` over the whole frame it is handed, so a decision
  depends on the spread, and here the price, of every later row of that frame:
  a look-ahead inside the rule. On a spread that is a whole number of ticks the
  spread in bp moves with the price, so the test selects price levels rather
  than tight books, the rule trades few episodes (5% of the planted size owed),
  and its trades realise 0.94 bp less than the planted drift owed them (t
  -3.48). On four other seeds the residual was -0.18 to -0.54 bp on planted
  data and about zero on their nulls.
* The flow rule fails because a share of signed volume cannot tell an episode's
  burst of prints from two prints on one side: its trades sit on episodes 5% of
  the time. The learned models, which also see the print rate, find it.
* Every strategy found nothing on the null, no residual reached |t| 3 except
  `spread_capture`'s, and no placebo kept a positive edge. The pooled placebo
  means are slightly negative (-0.15 to -0.90 bp), matched by what the planted
  drift owed the moved trades (-0.10 to -0.81 bp): the target days' own drifts,
  not a leak.

**detector** (three paths of two days; break at 30 h: return scale x1.6,
spread 2 to 4 ticks, edge flipped)

| Detector | Finds the break (within) | Lags | False alarms (bound 0.5 a day) | Result |
|---|---|---|---|---|
| `changepoint.detect`, one-second grid | 3 of 3 (4000 s) | 2000 s on every path (one block) | 7 in 9.8 days, **0.72 a day** | **FAIL** |
| `structural_breaks.detect_breaks`, one-minute returns | 3 of 3 (60 min) | 59, 35, 27 min | 0 in 9.8 days | pass |

The changepoint detector's false alarms are its `spread` statistic (and two
`intensity` flags): it reads the spread in bp of the mid, and a spread of a
fixed number of ticks moves in bp as the price moves. With the spread expressed
at a fixed price the `spread` flags on the three nulls disappear and the two
`intensity` flags remain. These streams are more volatile than a large-cap
perpetual (the price ranged 10 to 17% over each two-day null), so the rate is at
the high end of what such an instrument would show; the mechanism is not. After
detecting the break it also flags again every 20,000 s or so (28 further flags
over three paths), because its reference keeps the blocks before the break.

**simulator** (planted adverse selection 0.6 bp; four one-hour sessions)

| Criterion | Planted | Recovered | Result |
|---|---|---|---|
| S0 adverse selection, planted flow | 0.60 bp | 0.65 bp over 641 fills (tolerance 0.24 bp) | pass |
| S0 adverse selection, null | 0 bp | 0.05 bp over 955 fills (tolerance 0.20 bp) | pass |
| Fill rate of held touch orders, planted flow | 0.290 (replay) | 0.285, 8,998 orders, 99.5% agree order by order | pass |
| Fill rate, queue stress (proportional) | 0.907 | 0.895, 5,450 orders | pass |
| Brackets, queue stress | 0.907 | pessimistic 0.878, optimistic 0.922 | pass |
| The simulator's own adverse part (R18, unbound) | 0.60 / 0 bp | 1.02 / 0.29 bp | info |

The last row is the documented caveat measured: the split between spread and
adverse selection is indicative, because the reference mid can be 100 ms stale;
here it moves 0.3 to 0.4 bp from the spread to the adverse part. The fill
estimate is slightly pessimistic overall (proportional 1.2 points under the
replay on the queue-stress flow).

**reversion** (three paths of 20 instruments over 16 days; half-life 60 bars)

| Rule | Reversal: recovered (t) / owed | Momentum: recovered (t) / owed | Null (t) | Residual t, worst | Result |
|---|---|---|---|---|---|
| `relative.backtest` | 0.95 bp (15.1) / 0.95 bp | -1.03 bp (-15.8) / -0.97 bp | -0.07 bp (-0.97) | -0.97 | pass |
| frozen `reversion` index rule | 19.01 bp (7.4) / 20.25 bp | -19.91 bp (-7.5) / -21.80 bp | 3.90 bp (1.31) | 1.31 | pass |

Per basket for the relative rule (weights sum to one in absolute value), per
trade for the index rule.

## Adding a planted effect

1. Write a generator in `src/trading_research/synthetic/` with a frozen spec
   (defaults that run in seconds, a `seed`, a `null` flag), a `truth.json`
   written through `common.Truth`, frames stamped with `common.SYNTHETIC_SOURCE`
   in the repository's contracts, `common.check_frame` before writing, and a
   docstring with the three sections every generator here has: what is
   planted, the null, what it can and cannot prove.
2. Keep the latent state that made the data (a drift path, a transient, an
   order log), so a check can compute what any trade or order was owed by the
   planted effect alone.
3. Make the null keep the marginals and lose the predictability, and say how.
   A drift spread over time stays predictable from its own start even with a
   random direction: `book_days` learnt this when a model found it on the first
   null, and its null now makes the whole move in one step.
4. Write the check in `verify.py` as a function returning a `CheckResult`:
   one `add` per criterion, with the planted value, the recovered value and a
   tolerance stated before the run; register it in `CHECKS` and give it a
   command in `cli_synthetic.py`.
5. Test the generator (validity, determinism, causality, the null) and the
   check (it passes on planted data, finds nothing on the null, and catches a
   deliberately leaky or broken component) in `tests/test_synthetic_*.py`.

## For other code: the stable interface

```python
from trading_research import synthetic

data = synthetic.planted_edge(feature="flow", size_bp=12.0)  # truth in data.truth
report = synthetic.recover("lightgbm", data)  # or a model object, or a callable
report.summary["recovered_bp"], report.summary["planted_bp"], report.summary["residual_t"]
synthetic.recover("lightgbm", synthetic.null(feature="flow")).summary["recovered_t"]
```

`recover` accepts a registered model or rule name (`"order_flow"`, or
`"order_flow:flow_imbalance_24"` to point a rule at another column), an object
with `fit` and `predict_proba`, or a callable from the feature frame to
decisions in {-1, 0, +1}; a callable never sees the forward move.
