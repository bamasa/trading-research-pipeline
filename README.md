# trading-research-pipeline

**A leakage-aware research pipeline for systematic trading.**

Instruments, features, targets and models are compared under one honest cost
model, with look-ahead checked mechanically rather than promised. Demonstrated
end to end on crypto limit-order-book data; the pipeline itself is not tied to
that example.

It covers the whole path from raw events to a trading result: data contracts,
feature engineering, forward-looking labels, chronological validation with
purging and embargo, model comparison, and a backtest that charges realistic
transaction costs.

The emphasis is on the parts that decide whether a result means anything —
validation and cost accounting — rather than on the model. Most short-horizon
strategies that look profitable are not: they leak future information into
features, they tune on the data they report on, or they ignore what it costs to
cross a spread two hundred times a day. This project is built so that each of
those failure modes is a test that fails, not a caveat in a footnote.

> **Status: complete end to end, taker execution only.** Data contracts,
> synthetic market, downloader, feature registry with look-ahead checks, purged
> walk-forward splits, five models, calibration, cost-aware backtesting, a
> searched retraining schedule, rule-based strategies alongside the learned
> ones, ensembles, six exit rules, a market gate, successive-halving search, and
> the experiment scripts behind every published table. Maker execution is measured as a bound in
> [§11](docs/results.md) and deliberately not built on — see the
> [roadmap](#roadmap).

---

## Why this project

Three things here are deliberately stricter than the norm in public trading
repositories:

**1. No look-ahead, checked mechanically.** Every feature registers the number
of observations it looks back. The test suite then mutates the data *after* a
cutoff and asserts that no feature value *before* the cutoff moved. That catches
centred rolling windows, backward fills and normalisation statistics fitted over
the whole sample — the three ways look-ahead usually gets in, none of which
makes anything visibly fail.

**2. Purging and embargo, derived rather than guessed.** A label at time *t*
reads prices up to *t + H*, so the tail of a training block overlaps the head of
the block after it. The split derives the embargo from the label's declared
horizon instead of leaving it to be set by hand, because an embargo set by hand
is usually set too small.

**3. Costs applied consistently.** The same cost model is used when labelling,
when deciding to trade, and when computing profit and loss. Backtests routinely
overstate results by labelling against a mid-price move that would not have
survived the spread.

The synthetic market underpins all of this. It contains a known, deliberately
weak predictable component, so the test suite can assert both directions: that
the pipeline finds the signal when there is one, and — with the signal switched
off — that it finds nothing. A pipeline that reports an edge on data with no
edge has a leak, and that is a test rather than a hope.

## The findings

Most of this project is a negative result, arrived at carefully. One part is
not, and it is a candidate rather than a conclusion. Both are below, in that
order, because the second only makes sense given the first.

---

## What works: the market goes too far, and comes back

An equal-weighted index of twenty-six USDT perpetuals overshoots over roughly
ten minutes and comes back. Its recent return predicts the *next* move of each
constituent, negatively. There is no model in the rule: measure the index's
return over ten minutes, take the opposite side, close ten minutes later.

The holding period is the whole argument, and it comes from arithmetic rather
than search. Edge per trade is roughly the information coefficient times the
dispersion of the move; dispersion grows with the square root of the holding
period and the cost of a round trip does not grow at all. So a horizon exists
where prediction and cost cross.

![Net per trade by holding period](docs/images/reversion_horizon_profile.png)

Both curves have the same shape, and one of them was measured on days that chose
nothing. At the peak — ten-minute lookback, ten-minute hold — **twenty of
twenty-six instruments are positive on the held-out block**, median +6.9 bp per
trade over about 126 trades each. The ordering follows the identity rather than
the search: DOGEUSDT +26.7, APTUSDT +21.2, UNIUSDT +19.7, and BTCUSDT −5.1
because it moves a fifth as far for the same fee.

### A model doubles it, once it is asked the right question

The rule holds for a fixed ten minutes and gives back 51–86 bp from each trade's
peak, because the peak arrives around minute five. Asking a model to *name the
exit row* fails on every architecture tried — the peak moves too much between
trades. Asking it *how far this trade will run*, and leaving the decision to the
rule, does not.

![Net per trade by variant](docs/images/reversion_variants_ALL.png)

On 3,297 held-out trades across 26 instruments, models fitted on the search
block only, compared **paired** against the same entries:

| Variant | Net per trade | paired *t* |
|---|---:|---:|
| hold through the trigger when confident | **+18.8 bp** | 10.0 |
| per-trade take-profit level | +17.2 bp | 8.5 |
| cut the dead ones early | +16.6 bp | 7.1 |
| fixed take-profit, level chosen on search | +14.4 bp | 5.5 |
| **baseline: fixed ten-minute clock** | **+7.4 bp** | — |
| fixed trailing stop | +1.2 bp | −5.3 |

![Cumulative result](docs/images/reversion_equity_ALL.png)

### Why this is still called a candidate

Four things are wrong with it, recorded in
[`docs/findings.md`](docs/findings.md) alongside the conditions that would kill
it — written before the tests, not after:

- **Sixfold decay** between blocks: +17.2 bp median on the search block, +2.8 on
  the held-out one.
- **It is one bet, not twenty-six.** Against a market-neutral target the
  coefficient collapses from −0.056 to +0.004 and the instruments stop agreeing.
  Trading the panel is leverage, not diversification.
- **Thirteen held-out days.** +2.8 bp against a 14 bp round trip is close enough
  to zero to be a period rather than an effect.
- **The decisive test has not been run.** The frozen configuration on days
  neither block has seen is the only thing that would settle it, and the venue
  began rate-limiting before that span could be fetched.

Four earlier findings in this project looked at least this good and did not
survive that test. [`docs/findings.md`](docs/findings.md) lists them with what
killed each one.

---

## What does not work: direction from one instrument's own book

Run on 38 days of BTCUSDT and 59 of XRPUSDT futures from early 2024, then again
on three instruments over thirteen axes, the pipeline reaches a definite
negative conclusion, and — more usefully — explains it.

Short-horizon direction **is** predictable. Queue imbalance correlates with the
next second's mid return at about 0.27, decaying to roughly 0.02 by ten
minutes. That is real signal, and it is reproducible.

It is also not enough. A taker round trip on Binance USD-M costs about 11 basis
points: 5 bp of fee per side, plus the spread, plus slippage. Against that, the
best gross edge observed was around 5 bp per trade, at a two-to-five minute
horizon — short by a factor of about two. No fold, on either instrument, at any
horizon tested, was profitable after costs.

The mechanism is visible in one table. Signal strength falls with horizon at
almost exactly the rate volatility rises, so their product — the expected gross
edge per trade — barely moves, while the fee stays fixed. **The horizon where
prediction works and the horizon where trading pays do not overlap.**

Four obvious levers were tested and none of them helps. A longer horizon does
not, because the edge is horizon-invariant. A wider feature set — 194 generated
columns against 10 hand-picked — measurably made it worse. A better fee tier
falls short by about a factor of five: break-even needs roughly 0.3 bp per side
against about 1.7 at the top volume tiers, and even a zero fee leaves under a
basis point per trade once the spread is paid. And a sequence model does not:
across twenty-one model-configuration-fold combinations, none was profitable,
and the best gross edge belonged to plain logistic regression rather than to the
network.

And the learning is not what produces what edge there is. Trading the queue
imbalance directly — one feature, no model, no parameters — beats logistic
regression, gradient boosting and their ensemble on gross edge per trade, on
both instruments, through identical machinery. The models' extra apparatus
mostly buys selectivity, which the cost floor then eats.

![Rules against models](assets/strategies_btc.png)

Given five strategies and a daily choice between them — including the choice
not to trade — the selector stands aside on every single day it is offered
(§15). It is not a strategy that failed to find an edge; it is one that
identified in real time that there was none and acted accordingly. The
configurations that trade are the ones forbidden from declining, and they lose
around 2,600 bp of notional a day doing it.

One lever does move the number: **how often the model is retrained.** Fitting on
a fortnight and refitting every day — schedule chosen on validation, applied
once to test — raises XRP's gross edge to 12.31 bp against a 12.68 bp cost. That
is a shortfall of 0.37 bp rather than the ~10 bp gap everywhere else, and it is
still a loss: negative in sign, over 141 trades, with fewer than half its windows
positive. It is the closest this study gets, and close is not across.

![Gross against net](assets/bt_equity.png)

The whole study in one figure: the model's calls are right often enough for the
gross line to climb, and the fee takes it 1,200 basis points the other way.

![Where the model traded](assets/bt_trades.png)

### Why one works and the other does not

The audit in [§28](docs/results.md) measures each source of information before a
model is chosen, and the answer is not subtle. At a two-minute horizon on
BICOUSDT, the best single column of the touch plane — the plane every earlier
section used — reaches an information coefficient of 0.022 out of sample. The
best single *cross-sectional* column reaches **0.046**, twice as much. Break-even
needs about 0.08.

So the ceiling was never the model. Eight features off the top of one book do
not carry enough, and no arrangement of trees, ensembles, stacking or networks
changes that — measured, in [§28](docs/results.md), rather than assumed.

See [`docs/results.md`](docs/results.md) for the full picture,
[`docs/findings.md`](docs/findings.md) for the register of every candidate and
its status, and [`docs/limitations.md`](docs/limitations.md) for what none of it
establishes. No backtest here should be read as evidence that any strategy is or
was profitable.

---

## The pipeline, step by step

Twelve steps from "which market" to "running in production". Eleven are built;
the last is marked and is not.

```
 ┌── 0 ──────────┐   ┌── 1 ──────────┐   ┌── 2 ──────────┐   ┌── 3 ──────────┐
 │ screen        │   │ get the data  │   │ build         │   │ label what    │
 │ instruments   │──▶│ + contracts   │──▶│ features      │──▶│ counts as a   │
 │ by headroom   │   │ + validation  │   │ (declared     │   │ move worth    │
 │               │   │               │   │  lookback)    │   │ trading       │
 └───────────────┘   └───────────────┘   └───────────────┘   └───────────────┘
                                                                     │
 ┌── 7 ──────────┐   ┌── 6 ──────────┐   ┌── 5 ──────────┐   ┌── 4 ──▼───────┐
 │ decide how to │   │ decide when   │   │ fit + choose  │   │ split in time │
 │ leave         │◀──│ to enter      │◀──│ the model     │◀──│ purge+embargo │
 │               │   │ (threshold)   │   │               │   │               │
 └───────────────┘   └───────────────┘   └───────────────┘   └───────────────┘
         │
 ┌── 8 ──▼───────┐   ┌── 9 ──────────┐   ┌── 10 ─────────┐   ┌── 11 ─────────┐
 │ charge the    │   │ search the    │   │ review it     │   │ deploy + run  │
 │ costs         │──▶│ configuration │──▶│ against a     │──▶│   ⚠ not built │
 │               │   │ (smart, not   │   │ strict rubric │   │               │
 └───────────────┘   │  exhaustive)  │   └───────────────┘   └───────────────┘
                     └───────────────┘
```

**0. Choose the instrument.** `trading-research discover` enumerates what the
venue lists; `trading-research screen` ranks it on both halves of the edge
identity — how far the price moves against the cost, *and* whether the book
predicts where. Three days of data per instrument.

Both halves matter and the first version had only one. Ranked on movement alone
it put XMRUSDT third of forty-four; XMRUSDT moves nine times as far as BTCUSDT
and its book predicts nothing (correlation 0.0002 against 0.049), and the full
pipeline run on it did worse than on the instrument it was meant to replace.
Corrected, the ranking inverts — and gives the sharpest result in the project:
**the best of forty-four instruments has an expected edge one-sixteenth of its
cost to trade.**

*Choosing the model class is still a judgement made by hand.* The choice is not cosmetic: it fixes the cost
floor, and the cost floor decides everything downstream. BTCUSDT is one tick
wide, so almost the entire round trip is fee; XRPUSDT carries a 1.7 bp spread on
top. What should happen here is a screen across candidate instruments —
spread, tick size, volume, how often a move clears the cost — that ranks them
before any modelling. What happens today is that two were picked and the
consequences measured afterwards.

**1. Get the data, under a contract.** `trading-research download` pulls
Binance public archives, verifies checksums, and converts to a declared schema.
Two planes are modelled separately — trades and book — because a full order book
cannot be reconstructed from a trade tape. `validate-data` checks the contract
and reports gaps, crossed books and clock problems as findings rather than
crashes.

**2. Build features.** Every feature is a pure function that declares how far
back it looks. The test suite mutates data *after* a cutoff and asserts nothing
before the cutoff moved, which catches centred windows, backward fills and
whole-sample normalisation — the three ways look-ahead usually gets in.

**3. Label what counts as a move worth trading.** A forward return larger than
the round-trip cost is a BUY or a SELL; everything else is HOLD. The threshold
is the cost, so the model is learning to spot moves that could actually pay,
not moves that merely happen.

**4. Split in time.** Walk-forward over whole days, with the embargo derived
from the label's horizon rather than guessed. A label at *t* reads prices to
*t+H*, so the tail of every training block is dropped.

**5. Fit and choose the model.** Rule-based strategies (momentum, mean
reversion, order-flow, breakout) run through the identical machinery as the
learned ones (logistic, gradient boosting, a dilated causal network) and their
ensembles. A comparison where the baseline is scored differently is not a
comparison.

**6. Decide when to enter.** One threshold on the model's confidence, swept on
validation. Raising it trades less and more selectively; the sweep can optimise
either total profit or profit per trade, and the second is what "trade only the
best signals" means when the cut is made honestly.

**7. Decide how to leave.** Six rules: a fixed clock, take-profit, stop-loss,
trailing stop, and two that read the model's ongoing opinion — leave when its
confidence decays, or when its prediction flips. Plus a market gate that
declines to trade at all when volatility cannot support the cost.

**8. Charge the costs.** The same cost model at labelling, at the decision, and
in the profit and loss. Taker on both legs: fee, spread, slippage.

**9. Search the configuration.** Successive halving over sampled configurations
rather than a nested product — cheap candidates are killed after a few windows
and only survivors are measured on the full span. Everything is chosen on
validation and the test span is scored once.

**10. Review it.** `trading-research review` assembles the arithmetic half of
[`docs/evaluation/rubric.md`](docs/evaluation/rubric.md) — result, dispersion,
drawdown, and how many trades the edge would need to be established — into a
document a reviewer or an agent can read.
[`docs/evaluation/prompt.md`](docs/evaluation/prompt.md) is the reviewer's
instructions. The tool scores nothing: something that produced the evidence and
graded it would be marking its own work.

This project's own scorecard is in
[`docs/evaluation/btc_best.md`](docs/evaluation/btc_best.md), and it is not
flattering.

**11. Deploy and run.** ⚠ *Not built.* What is missing is not the model but
everything around it: a live data feed with gap recovery, the feature pipeline
running in streaming rather than batch, an order router, position and risk
limits, a kill switch, and monitoring that compares live fills against what the
backtest assumed. That last one is the point — a strategy that silently decays
looks identical to one that is working until it does not.

---

## Architecture

```
        data                features            labels           validation
   ┌──────────────┐   ┌──────────────────┐  ┌─────────────┐  ┌────────────────┐
   │ trade events │──▶│ registry of pure │─▶│ forward     │─▶│ chronological  │
   │ book snaps   │   │ functions, each  │  │ returns,    │  │ splits, purge  │
   │ + contracts  │   │ declaring its    │  │ declaring   │  │ + embargo from │
   │ + validation │   │ lookback         │  │ its horizon │  │ label horizon  │
   └──────────────┘   └──────────────────┘  └─────────────┘  └────────────────┘
          │                                                          │
          │                                                          ▼
          │                             models              ┌────────────────┐
          │                    ┌──────────────────────┐     │ walk-forward   │
          └───────────────────▶│ naive · logistic ·   │◀────│ evaluation     │
                               │ xgboost · sequence   │     └────────────────┘
                               └──────────────────────┘
                                          │
                                          ▼
                              backtest              reporting
                        ┌────────────────────┐  ┌──────────────────┐
                        │ costs in bp,       │─▶│ manifest, plots, │
                        │ spread crossing,   │  │ cost attribution │
                        │ slippage, fills    │  │ regime breakdown │
                        └────────────────────┘  └──────────────────┘
```

Two data planes are modelled separately, because they have different
availability:

| Plane | Contents | Real data |
|---|---|---|
| **trades** | one row per trade or aggregated trade | public exchange archives |
| **book** | one row per snapshot, *N* levels per side | requires a dedicated collector |

A full order book cannot be reconstructed from public trade archives — the
resting size that was never hit leaves no trace in the trade tape. The book
plane is therefore defined and exercised against synthetic data from the start,
so that book features and their tests exist before the collector does.

---

## Quickstart

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/bamasa/trading-research-pipeline
cd trading-research-pipeline
uv sync --all-extras
```

Generate a synthetic dataset and check it:

```bash
uv run trading-research generate-demo-data --output data/demo
uv run trading-research validate-data --input data/demo
uv run pytest
```

Inspect a data contract:

```bash
uv run trading-research describe-schema book
```

Nothing here downloads anything or needs credentials.

### The pipeline on real data

Five stages, each a command, each writing files and a manifest. Download once,
then run the rest as often as needed:

```bash
# once: public archives, checksum-verified, never committed
uv run trading-research download --symbol BTCUSDT --kind bookTicker \
    --start 2024-02-01 --end 2024-03-09 --grid 100ms
uv run trading-research download --symbol BTCUSDT --kind aggTrades \
    --start 2024-02-01 --end 2024-03-09

# features, one file per day
uv run trading-research prepare --symbol BTCUSDT --horizon 1200 --subsample 50

# everything below reads only the window it is given
uv run trading-research select --train-start 2024-02-01 --train-end 2024-02-14
uv run trading-research train  --train-start 2024-02-01 --train-end 2024-02-14 \
    --model logistic
uv run trading-research predict --start 2024-02-22 --end 2024-02-28
uv run trading-research backtest --min-confidence 0.62 --hold 24 --cooldown 120

# how often to retrain: window length, apply length and step, searched on the
# first half of the span and applied once to the second
uv run trading-research retrain-search --model logistic --start 2024-02-15 \
    --train-days 3,7,14,21 --apply-days 1,3,7

# how to leave a trade: clock, take-profit, stop, trail, or the model's own
# fading opinion — chosen on validation, applied once to test
uv run trading-research exit-search --model logistic --hold 24
```

The model is one flag — `--model logistic | xgboost | tcn` — and so are the
instrument, the horizon and the feature set, so comparing combinations is the
normal case rather than a special one.

Run each model in its own invocation rather than looping inside one process:
XGBoost and PyTorch each bundle an OpenMP runtime and deadlock together on
macOS. The staged layout avoids this by construction.

`make help` lists the common tasks.

---

## Data

**Synthetic (default).** A generated market with volatility regimes, a
liquidity state, a spread that widens under stress, geometric depth decay, and
Poisson trade arrivals that execute against the book. It exists for tests, CI
and the quickstart. It is *not* evidence about markets: every row is stamped
`source="synthetic"` and every artefact carries a warning, because a synthetic
backtest measures the agreement between a generator and a model and nothing
else.

**Public exchange data.** Two venues, for two different things.

*Binance* publishes best bid and ask, aggregated trades, open interest and
funding as daily archives. Used for the instrument screen, which needs breadth
rather than depth. Note that daily `bookTicker` archives stop after 30 March
2024, which is why the screen can be measured on two windows and not more.

*Bybit* publishes the **full order book** as an incremental stream — a snapshot
and its deltas — which reconstructs to ten levels a side, and publishes trade
prints carrying the aggressor's side. Everything from §24 onward runs on these:
the depth features need the levels, and the maker model needs the prints,
because a book says what was offered and only a print says what was taken.

**Nothing under `data/` is in version control.** A day of one instrument's book
is hundreds of megabytes.

**The code fetches what it needs.** Every experiment declares its instruments
and span, and [`data/ensure.py`](src/trading_research/data/ensure.py) compares
that against what is on disk and downloads only the gaps. A clean checkout
therefore runs — slowly the first time, instantly afterwards. Three properties
are deliberate:

- an interrupted run resumes rather than restarts, since a day already present
  is never fetched twice;
- a day the venue does not have is reported and skipped, not raised — an
  instrument listed mid-span has no archive before it existed — while a *total*
  absence does raise, so an experiment cannot quietly run on nothing;
- nothing downloads silently: each call prints what it will fetch first, and
  `dry_run=True` answers the question without acting.

```python
from datetime import date
from trading_research.data.ensure import ensure_book, ensure_universe

# ten levels a side, about two minutes of replay per day
ensure_book("DOGEUSDT", date(2024, 2, 1), date(2024, 3, 10))

# top of book only for a cross-section: eleven seconds per instrument-day
ensure_universe(["BTCUSDT", "ETHUSDT"], date(2024, 2, 1), date(2024, 3, 10))
```

The venue rate-limits sustained downloading. A stalled fetch is the venue, not
the code; wait and re-run, and it resumes from where it stopped.

**Synthetic data** remains the default for tests and the quickstart, so neither
needs the network.

---

## Methodology

Documented in [`docs/`](docs/) as it lands:

| Document | Contents |
|---|---|
| `data_contract.md` | Column semantics, units, time conventions, quality rules |
| `methodology.md` | Features, labels, splits, model comparison protocol |
| `execution_assumptions.md` | Cost model, fills, what the backtest does and does not simulate |
| `limitations.md` | What this does not establish |
| `disclosure_policy.md` | What is public here and why, and what is not |

---

## Limitations

Stated plainly, because the honest version is more useful than the flattering
one:

- **No profitability claim is made.** No result in this repository should be
  read as evidence that any strategy is or was profitable.
- **Synthetic results are circular.** They demonstrate that the pipeline works,
  not that markets behave this way.
- **The backtest is a simulation.** It does not model queue position, partial
  fills against a real book, latency, market impact, or the fact that a real
  order changes the book it is trading against.
- **Public data is coarser than production data.** Exchange archives are
  aggregated; a real system runs on a live feed with different timing.
- **Short horizons are the hardest regime for this.** Costs dominate, edges are
  thin, and results are unstable across periods. Where that shows up, it is
  reported rather than tuned away.

---

## Reproducibility

- One seed per run, recorded in the run manifest along with the full
  configuration and the schema version.
- Datasets are directories with a `manifest.json` describing provenance — for
  synthetic data, the generator version and seed that produced it.
- Every assumption that moves a number lives in the config file, not in a
  default buried in a function.
- CI runs lint, types, tests, the end-to-end demo, and a disclosure audit that
  scans for private paths, credential-shaped strings, notebook outputs and
  oversized binaries.

---

## Roadmap

Done:

- [x] Package skeleton, tooling, CI, disclosure audit
- [x] Data contracts for both planes, with quality validation
- [x] Synthetic market with regimes and a known injected signal
- [x] Binance public-data downloader with checksums and manifests
- [x] Feature registry with declared lookbacks and automated look-ahead tests
- [x] Directional labels and purged walk-forward splits
- [x] Naive, logistic and gradient-boosted baselines
- [x] Taker cost model and execution-aware evaluation
- [x] Wide generated feature set with train-only selection
- [x] Horizon diagnostics: predictability against tradeability
- [x] BTC → XRP transfer experiment
- [x] Staged pipeline: `prepare`, `select`, `train`, `predict`, `backtest`
- [x] Sequence model (dilated causal convolutions), compared against the rest
- [x] Trade thinning with cooldown, take-profit and stop-loss
- [x] Isotonic and Platt calibration
- [x] Retraining schedule as a searched parameter, not an assumption
- [x] Experiment scripts behind every published table
- [x] Six exit rules, with the policy searched on validation
- [x] Rule-based strategies — momentum, mean reversion, order flow, breakout —
      scored through the same machinery as the learned ones
- [x] Model ensembles, and a market gate that declines to trade in conditions
      that cannot support the cost
- [x] Successive halving over sampled configurations, in place of a grid
- [x] Rolling normalisation of features against a trailing window
- [x] Pooling instruments into one training set — **tried and rejected**: every
      instrument did worse on the pool than on itself (§22), and the scope is
      deliberately one instrument at a time from here
- [x] Instrument screening by headroom, before anything is fitted
- [x] A review rubric and an evidence pack for it, applied to this project's
      own result

Next, staying with taker execution:

- [ ] **Step 0: automate instrument selection.** A screen over candidate
      instruments — spread in bp, tick size, volume, share of moves clearing
      the cost — ranking them before any modelling. Today the instrument is a
      judgement made by hand, and it fixes the cost floor that decides
      everything downstream.
- [ ] Unified report: calibration, equity, drawdown, cost attribution, regimes
- [ ] Order-book collector with sequence-gap recovery, for depth beyond the touch
- [ ] Position sizing from calibrated probabilities rather than a fixed unit
- [ ] Holding period searched jointly with the entry threshold, since §12 shows
      the label horizon is not the right one to hold for

**Step 10: deployment.** Nothing here runs live, and the gap is not the model:

- [ ] Live feed with sequence-gap recovery and a reconnect that does not silently
      skip data
- [ ] Features computed in streaming rather than batch, producing bit-identical
      values to the research path — a discrepancy here is the classic way a
      backtest and a live system quietly diverge
- [ ] Order router, position limits, risk limits, kill switch
- [ ] Monitoring that compares live fills against what the backtest assumed:
      realised spread, slippage, fill rate. A strategy that is decaying looks
      exactly like one that is working until it does not

Maker execution — deferred, deliberately
----------------------------------------
[§11](docs/results.md) measures what posting would be worth and finds +2.3 to
+5.0 bp per trade after adverse selection, against the 0.37 bp the best taker
configuration was short by. That is a **bound, not a strategy**, and the work
below is what would be needed to turn it into one. Until it is done, everything
in this repository is taker-only and should be read that way.

- [ ] Measure adverse selection at the moments a model wants to trade, not at
      unconditional moments. Those are precisely the moments the market is about
      to move — the worst case for a resting order, and the figure that decides
      whether the margin above survives.
- [ ] Passive exits. A position held to a horizon currently closes by crossing.
      A fully passive strategy needs the same analysis on the way out, where
      failing to fill is a risk rather than a missed opportunity.
- [ ] A queue model that is not optimistic: size ahead that grows, cancellations
      ahead of the order, and the order's own size treated as non-negligible.
- [ ] Capacity. Fill rates in the teens mean a strategy that took 141 trades as
      a taker takes about twenty as a maker, which establishes nothing.

## Relationship to prior closed-source work

The author has worked on short-horizon prediction for order-book data in a
commercial setting. This repository shares none of that code, data,
configuration or results. It is an independent implementation, written from
scratch, using public and generated data only, and it deliberately diverges
where the public version can be stricter — the purging, embargo and automated
look-ahead checks described above are additions, not reproductions.

See [`docs/disclosure_policy.md`](docs/disclosure_policy.md).

---

## Disclaimer

This is research and engineering code published as a portfolio and a teaching
artefact. It is **not** investment advice, **not** a trading system, and
**not** a claim about future returns. Backtested results — including any shown
here — are computed with the benefit of hindsight under stated assumptions, and
do not establish that a strategy would have been profitable in live trading or
would be profitable in future.

## License

[MIT](LICENSE).
