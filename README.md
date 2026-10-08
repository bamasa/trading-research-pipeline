# trading-research-pipeline

**A leakage-aware research pipeline for systematic trading — and what it found
when pointed at crypto perpetuals.**

*Solo project · 2026 · Python, pandas, NumPy, LightGBM, PyTorch · market microstructure, limit order books, walk-forward validation, event-time market-making simulation, regime detection · MIT licence*

| | |
|---|---|
| **Found** | a conditional edge: short-horizon cross-sectional reversion pays on a held-out fortnight (+6.99 bp a trade net, 22 of 26 instruments) in a measurable market state, and loses everywhere once the state is gone |
| **Killed** | directional prediction from one instrument's book; passive execution of the signal; market making in two pre-registered rounds — nine hypotheses on data no code had read, every one refuted, including an apparent +2.2 a day that a pre-registered check showed was inventory carried through a rising market |
| **Built** | data contracts and downloaders for two venues, a feature registry with mechanical look-ahead checks, purged walk-forward splits, sixteen models, cost-aware backtests for taker, passive and market-making execution, an event-time simulator that fills only from trade prints, two regime-break detectors (one from [adia-structural-break](https://github.com/bamasa/adia-structural-break)), and a search over instrument × model × execution × regime policy |
| **How** | every candidate is registered with the condition that would refute it before the test runs; held-out blocks are opened once through a ledger that refuses them unless the frozen configuration is committed |

**The result.** Short-horizon cross-sectional reversion in crypto perpetuals
pays — under a market state that can be measured but, on the evidence here, not
forecast. On a held-out fortnight in the state that supports it, a
three-parameter rule earns **+6.99 bp per trade net of realistic costs on 22 of
26 instruments**, and a model that predicts how far each trade will run raises
that to **+18.8 bp over 3,297 trades**. On the six weeks that followed, where the
state was absent, the identical frozen configuration loses on all 26.

The state is one number: the index's ten-minute autocorrelation, −0.10 where the
strategy works and −0.0003 where it does not. The mechanism, the boundary, and
the three attempts to escape the boundary are in
[§27](docs/results.md#27-cross-sectional-reversion-and-the-market-state-it-requires).

Everything else here is negative and was established the same way. Directional
prediction from one instrument's own order book does not clear a taker round
trip: forty-four instruments screened, the three best searched over thirteen
axes, no winner profitable on a block that chose nothing.
Market making was tested the same way, in two pre-registered rounds, each read
once on a held-out fortnight: all nine hypotheses are killed. The spread a
public-data market maker captures is smaller than the move that follows its
fills, on BICO and on six wide-tick altcoins alike; the one result that looked
like profit (+2.2 USDT per 100 of clip a day at the market-maker rebate) was
inventory carried through a rising market, which a check written down before the
read caught ([below](#market-making-quoting-both-sides-on-public-bybit-data)).
[`docs/findings.md`](docs/findings.md) is the register — every candidate, its
status, and the condition written down before the test that would refute it.

Instruments, features, targets and models are compared under one honest cost
model, with look-ahead checked mechanically rather than promised.

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

> **Status: complete end to end: taker, passive entry and market making.** Data contracts, a synthetic
> market, downloaders for two venues with data that fetches itself when missing,
> a feature registry with look-ahead checks, purged walk-forward splits, sixteen
> models from parameter-free rules to dilated convolutions, bagging, stacking on
> forward-chained out-of-fold predictions, a bias-variance decomposition that
> says which ensemble is worth using, calibration, cost-aware backtesting, a
> queue-level maker model, an event-time market-making simulator that fills only
> from trade prints, two regime-break detectors, an information audit that
> measures each data source before a model is chosen, successive-halving search
> over thirteen axes with execution and the regime policy scored at its last
> rung, and the experiment script behind every published table. 1,130 tests, a
> disclosure audit in CI.

## The pipeline map

From raw events to a verdict. Every box is a module or a document, and the
execution step is a choice a run makes rather than an assumption built into it:
the same signal can cross the spread, rest at the touch, or be quoted on both
sides in event time, and each is judged by the same statistic.

```mermaid
flowchart TD
    data["<b>1 Data</b><br/>Binance archives · Bybit book, 10 levels / 100 ms<br/>prints with the aggressor · funding<br/>contracts and validation, fetched when missing"]
    regimes["<b>2 Regimes</b><br/>changepoint: CUSUM over book blocks<br/>whitened Shiryaev-Roberts on returns<br/>reversion state: index autocorrelation"]
    screen["<b>3 Instruments</b><br/>edge over cost, for a taker<br/>spread of two maker fees or more, for a maker"]
    signal["<b>4 Signal</b><br/>features with a declared look-back · labels<br/>purged walk-forward · rules and 16 models"]
    search["<b>5 Search</b><br/>successive halving · neighbourhood choice<br/>held-out block read once"]
    subgraph execution["6 Execution, a searched axis"]
        taker["taker"]
        passive["passive entry"]
        maker["market maker S0-S3<br/>+ regime guard"]
    end
    grid["<b>7 Grid</b><br/>thin + taker costs · queue behind the touch"]
    event["<b>7 Event time</b><br/>a queue per order · latency<br/>fills only from prints · fees · funding"]
    verdict["<b>8 Verdict</b><br/>clustered t · kill conditions written first<br/>placebos · ladder · fee break-even · register"]
    deploy["9 Deploy: not built"]

    data --> regimes --> screen --> signal --> search --> execution
    regimes -. flags and state .-> maker
    taker --> grid
    passive --> grid
    maker --> event
    grid --> verdict
    event --> verdict
    verdict -.-> deploy

    click data "https://github.com/bamasa/trading-research-pipeline/blob/main/docs/data_contract.md"
    click regimes "https://github.com/bamasa/trading-research-pipeline/blob/main/src/trading_research/validation/structural_breaks.py"
    click screen "https://github.com/bamasa/trading-research-pipeline/blob/main/src/trading_research/data/screen.py"
    click signal "https://github.com/bamasa/trading-research-pipeline/blob/main/src/trading_research/features/registry.py"
    click search "https://github.com/bamasa/trading-research-pipeline/blob/main/src/trading_research/validation/search.py"
    click taker "https://github.com/bamasa/trading-research-pipeline/blob/main/src/trading_research/pipeline/execution.py"
    click passive "https://github.com/bamasa/trading-research-pipeline/blob/main/src/trading_research/backtest/maker.py"
    click maker "https://github.com/bamasa/trading-research-pipeline/blob/main/src/trading_research/pipeline/market_making.py"
    click grid "https://github.com/bamasa/trading-research-pipeline/blob/main/src/trading_research/backtest/execution.py"
    click event "https://github.com/bamasa/trading-research-pipeline/blob/main/docs/market_making_simulator.md"
    click verdict "https://github.com/bamasa/trading-research-pipeline/blob/main/docs/findings.md"
```

| Step | Where it lives |
|---|---|
| 1 Data | [`data/`](src/trading_research/data/) · [`docs/data_contract.md`](docs/data_contract.md) |
| 2 Regimes | [`validation/changepoint.py`](src/trading_research/validation/changepoint.py) · [`validation/structural_breaks.py`](src/trading_research/validation/structural_breaks.py) · [`market_making/flags.py`](src/trading_research/market_making/flags.py) · [`strategies/reversion.py`](src/trading_research/strategies/reversion.py) |
| 3 Instruments | [`data/screen.py`](src/trading_research/data/screen.py) · [`market_making/screen.py`](src/trading_research/market_making/screen.py) |
| 4 Signal | [`features/`](src/trading_research/features/) · [`labels/`](src/trading_research/labels/) · [`validation/splits.py`](src/trading_research/validation/splits.py) · [`models/`](src/trading_research/models/) |
| 5 Search | [`validation/search.py`](src/trading_research/validation/search.py) · [`pipeline/discovery.py`](src/trading_research/pipeline/discovery.py) · [`experiments/grand_search.py`](experiments/grand_search.py) |
| 6 Execution | [`pipeline/execution.py`](src/trading_research/pipeline/execution.py) · [`pipeline/market_making.py`](src/trading_research/pipeline/market_making.py) |
| 7 Simulation and costs | [`backtest/`](src/trading_research/backtest/) · [`market_making/simulator.py`](src/trading_research/market_making/simulator.py) · [`docs/execution_assumptions.md`](docs/execution_assumptions.md) · [`docs/market_making_simulator.md`](docs/market_making_simulator.md) |
| 8 Verdict | [`evaluation/significance.py`](src/trading_research/evaluation/significance.py) · [`market_making/verdicts.py`](src/trading_research/market_making/verdicts.py) · [`docs/findings.md`](docs/findings.md) · [`docs/results.md`](docs/results.md) |

### Run it

```bash
uv sync --all-extras
uv run pytest                                          # every test, synthetic data, no network
uv run trading-research reproduce                      # §27 end to end, taker execution
uv run trading-research reproduce --execution both     # taker against passive entry, where there are prints
uv run trading-research mm-backtest --symbol BICOUSDT --start 2024-02-01 --end 2024-02-25 \
    --strategy s1 --frozen                             # the market maker on the development block
uv run python -m experiments.grand_search --symbol BTCUSDT --execution-axis
```

`reproduce` fetches what it needs. `mm-backtest` reads the book, prints and
funding the market-making study fetched (`data/book`, `data/trades`,
`data/funding`) and refuses the held-out fortnight unless `--allow-heldout` is
given, and then opens it only through the ledger.

---

## Why this project

Four things here are deliberately stricter than the norm in public trading
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

**4. Findings are recorded with what would kill them, before the test.**
[`docs/findings.md`](docs/findings.md) lists every candidate this project has
produced with its status — candidate, confirmed, killed, artefact — and, for the
open ones, the condition that would refute them, written down in advance. Four
findings are in the killed column, each with the test that killed it. Two are
artefacts: an effect that turned out to be a property of the measurement rather
than the market.

That register exists because the same mistake kept recurring in different
costumes — a promising number, a plausible mechanism, and no test that could
have refuted it. Two rounds of adversarial review against this repository's own
code found nineteen defects, including a market gate that silently let
everything through while three searches reported a gate axis, and a test whose
name asserted the opposite of the behaviour it pinned. Both are documented where
they happened rather than quietly fixed.

The synthetic market underpins all of this. It contains a known, deliberately
weak predictable component, so the test suite can assert both directions: that
the pipeline finds the signal when there is one, and — with the signal switched
off — that it finds nothing. A pipeline that reports an edge on data with no
edge has a leak, and that is a test rather than a hope.

## The findings

One positive result with its domain of validity stated, and a broad negative
that motivated looking for it. In that order, because the second is what makes
the first worth having.

---

## The finding: reversion, and the state it needs

An equal-weighted index of twenty-six USDT perpetuals is computed, excluding the
instrument being traded. When the index has moved far over the last ten minutes,
take the opposite side in that instrument and close ten minutes later. No model,
three parameters.

### Where it pays

On a held-out fortnight — parameters chosen on the 26 days before it, the block
read once:

| | |
|---|---|
| Instruments positive | **22 of 26** |
| Median net per trade | **+6.99 bp** |
| Gross per trade | +20.50 bp |
| Cost per trade | 12–16 bp |

A model on top, predicting *how far each trade will run* and leaving the exit to
the rule, raises the pooled figure to **+18.8 bp per trade over 3,297 trades**,
paired *t* of 10.0, with two independent boosters agreeing to within 1.6 bp.
Asking the model *when to exit* instead fails on every architecture tried — that
distinction is the useful part of the exercise.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/reversion_variants_ALL_dark.png">
  <img alt="Net per trade by variant" src="docs/images/reversion_variants_ALL_light.png">
</picture>

### The state it requires, measured

| | Where it works | Where it does not |
|---|---:|---:|
| **Index 10-minute autocorrelation** | **−0.1014** | **−0.0003** |
| Index move over the span | +49.8% | −33.1% |
| Dispersion of daily moves | 265 bp | 483 bp |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/reversion_horizon_profile_dark.png">
  <img alt="Net per trade by holding period" src="docs/images/reversion_horizon_profile_light.png">
</picture>

The holding-period profile is the mechanism made visible: two minutes loses, ten
is best, longer decays — the shape a reverting market produces and a trending one
does not.

### Why

A rising market's ten-minute surge is impatient buying that runs ahead of the
book, and it retraces; the strategy is paid for supplying the other side. A
falling market's ten-minute surge is forced flow — margin being closed — which
does not retrace, because the seller is not choosing and there is more behind.
The rule selects the *largest* ten-minute index moves, which in a cascade are
precisely the continuations. That predicts the sign of the failure, and gross
went from +20.50 to −5.07 rather than to zero.

### The boundary

Three attempts to escape the condition, all measured, all negative: daily
refitting of the threshold, daily re-estimation of the sign, and a gate that
trades only when trailing autocorrelation is low. The gate fails in the
informative direction — the days it admits are *worse* than the days it rejects,
in all four windows tried.

So: **the state is measurable, and on this evidence not forecastable a day
ahead.** The effect is real and conditional; deploying it needs a regime forecast
this project does not have. Anyone continuing should start there, not with the
trading rule, which is the easy half.

### And what fourteen days is worth

The +6.99 figure counts each trade as an observation, and 26 instruments carry
one market-wide bet — their per-day results correlate at +0.47. By day it is
+6.18 bp over 14 days, *t* = 1.18, 8 days positive. That establishes the effect
was present and how large; it does not establish persistence, which the six weeks
that followed then settled. Both statistics are computed for everything in this
repository by
[`evaluation/significance.py`](src/trading_research/evaluation/significance.py).

---

## What does not work: direction from one instrument's own book

Run first on Binance best-bid-ask for BTCUSDT and XRPUSDT in early 2024, and
then on Bybit's full order book for BTCUSDT (83 days) and BICOUSDT, CRVUSDT and
XRPUSDT (38 days each), the pipeline reaches a definite negative conclusion and
— more usefully — explains it.

Short-horizon direction **is** predictable. Queue imbalance correlates with the
next second's mid return at about 0.27, decaying to roughly 0.02 by ten
minutes. That is real signal, and it is reproducible.

It is also not enough. A taker round trip costs about 11 bp on Binance USD-M and
12–16 bp on Bybit, depending on the instrument's spread: a fee each side, plus
the spread, plus slippage. Against that, the best gross edge the corrected
searches produced is +3 to +9 bp per trade on the block that chose it — short by
a factor of two to four — and on the block that chose nothing, two of the three
winners have a gross edge of the wrong sign. No fold, on any instrument, at any
horizon tested, was profitable after costs.

The mechanism is visible in one table. Signal strength falls with horizon at
almost exactly the rate volatility rises, so their product — the expected gross
edge per trade — barely moves, while the fee stays fixed. **The horizon where
prediction works and the horizon where trading pays do not overlap.**

Every lever anyone reaches for was tested, and none of them helps. A longer
horizon does not, because within this range the edge is horizon-invariant. A
wider feature set — 194 generated columns against 10 hand-picked — measurably
made it worse. A better fee tier falls short by about a factor of five:
break-even needs roughly 0.3 bp per side against about 1.7 at the top volume
tiers, and even a zero fee leaves under a basis point per trade once the spread
is paid. A sequence model does not. Neither do bagging, random forests,
extremely randomised trees, a second booster, voting, or stacking — and the
[bias-variance decomposition](docs/results.md) says why: prediction variance is
negligible for all eight candidates, so there is nothing for averaging to
remove, and the models explain under 0.03% of the forward move.

Nor does *how* the strategy is run. Six exit rules, a market gate, four targets,
five rule-based strategies, a daily choice between them, retraining frequency,
refitting at detected regime breaks, and a full grid of training-window against
apply-window lengths — 672 cells of the last, of which three were positive, with
the two axes moving the median by 0.9 and 0.3 bp against a 13 bp cost.

And the learning is not what produces what edge there is. Trading the queue
imbalance directly — one feature, no model, no parameters — beats logistic
regression, gradient boosting and their ensemble on gross edge per trade, on
both instruments, through identical machinery. The models' extra apparatus
mostly buys selectivity, which the cost floor then eats.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/strategies_btc_dark.png">
  <img alt="Rules against models" src="docs/images/strategies_btc_light.png">
</picture>

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

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/bt_equity_dark.png">
  <img alt="Gross against net" src="docs/images/bt_equity_light.png">
</picture>

The whole study in one figure — the best BTCUSDT configuration's own trades,
rebuilt from the committed table by
[`scripts/render_readme_charts.py`](scripts/render_readme_charts.py): the
model's calls are right often enough for the gross line to climb three thousand
basis points, and the cost of crossing the spread takes the same trades eight
thousand the other way.

### Where the ceiling actually is

The audit in [§28](docs/results.md) measures each source of information before a
model is chosen, which is the order this project should have used from the
start. Fitting a model and looking at the money answers "did this arrangement
work" and never answers "was there anything here to find", so every negative
result before the audit was ambiguous.

At a two-minute horizon on BICOUSDT, the best single column of the touch
plane — the plane every earlier section used — reaches an information
coefficient of 0.022 out of sample. Break-even needs about **0.08 to 0.10**,
measured by blending predictions with the realised future at increasing weight
until profit crosses zero. The best of everything tried, over sixteen model
types, reaches 0.023.

So the ceiling was never the model, and no arrangement of trees, ensembles,
stacking or networks moves it. The bias-variance decomposition says why:
prediction variance is negligible, so averaging has nothing to remove, and the
models explain under 0.03% of the forward move.

The cross-sectional column that reached 0.046 — twice the best own-book
feature — is what sent this project after the reversion candidate above. That
0.046 was real on the span it was measured on, and the six weeks that followed
say it was a property of those weeks.

See [`docs/results.md`](docs/results.md) for the full picture,
[`docs/findings.md`](docs/findings.md) for the register of every candidate and
its status, and [`docs/limitations.md`](docs/limitations.md) for what none of it
establishes. No backtest here should be read as evidence that any strategy is or
was profitable.

---

## Market making: quoting both sides on public Bybit data

**The result: all four pre-registered hypotheses are killed** on the held-out
fortnight, read once. On the one instrument whose spread admits a market maker,
quoting loses to the move that follows its fills; stepping inside the spread,
leaning on the reversion state, resting on the reversion signal and pulling back
after regime flags each failed a condition written down before the test. No
strategy there breaks even at any maker fee Bybit publishes.

**What was pre-registered.** Before any market-making code existed,
[`docs/preregistration/market_making.md`](docs/preregistration/market_making.md)
and [`configs/mm_prereg.yaml`](configs/mm_prereg.yaml) (commit `cb4ae4c`) fixed
the blocks, the instrument admission rule, three hypotheses with their kill
conditions and placebos, and the rule that the held-out fortnight is read once.
Amendment 1 recorded simulator corrections found in review, before any result;
Amendment 2 froze every value searched on the 25-day development block, with the
configuration's hash, before the held-out block was opened. The hypotheses:

- **H1** — quoting one tick inside a wide spread, first in the queue, earns a
  positive net and more than the same quoter without the rule;
- **H2** — the reversion state of [§27](docs/results.md#27-cross-sectional-reversion-and-the-market-state-it-requires)
  pays a market maker that leans against the index's move (H2.1), and passive
  execution of the frozen reversion signal beats crossing for it (H2.2);
- **H3** — widening quotes after a regime-break flag earns more than quoting
  through it.

**The held-out fortnight was read once**, 26 February to 9 March 2024, through a
ledger that refuses to open it unless the frozen configuration is committed and
the tree clean, and records the read
([`mm_heldout_ledger.json`](experiments/results/mm_heldout_ledger.json): commit
`2a7daaa`, 2026-10-04 12:34:52 UTC). Every day of every instrument was simulated
and none was excluded.

**The simulator, in six rules.** Built first, tested on synthetic markets with
known answers, and documented rule by rule with the direction each one biases a
result in [`docs/market_making_simulator.md`](docs/market_making_simulator.md):

1. **Fills only from prints.** A resting order fills only when a print reaches
   its price from the side that hits it, never because a snapshot moved.
2. **A queue per order.** It joins the tail of the visible size, moves up only
   as prints consume what is ahead, and gains on cancellations by a rule that is
   bracketed — and a verdict must survive the pessimistic bracket.
3. **Latency.** Orders, cancels and flattens take 10 ms; a cancel in flight
   does not protect the order; a price change loses priority.
4. **Own size and limits.** A tenth of the touch at most; soft and hard
   inventory limits; past the soft limit a reduce-only taker order flattens
   back, walking the first book after it arrives.
5. **Fees and funding.** Maker fee on passive fills, taker fee on flattens,
   funding on the position held at each settlement; days start and end flat.
6. **What it does not model.** The book is a 100 ms photograph, so queue
   dynamics inside 100 ms are a stated rule rather than an observation; own
   impact on others is ignored; a snapshot that crosses an order fills nothing
   without a print, which is optimistic and is bracketed. A day whose stored
   book lost messages, or whose flatten found no fresh book, is flagged and
   kept out of every verdict.

### What each increment earns

BICOUSDT, the only instrument the admission rule (spread at least twice the base
maker fee, two ticks or more a quarter of the time) let through, over the
held-out fortnight, USDT a day
([`mm_strategies_H.csv`](experiments/results/mm_strategies_H.csv)):

| Strategy | Net | Net, bp of turnover | Fills a day | RMS position (BICO) | Spread | Adverse, 5 s | Inventory | Fees | Funding |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 at the touch | −43.41 | −5.00 | 12,277 | 103 | +21.49 | −47.26 | −0.30 | 17.37 | +0.03 |
| S1 skewed, gated | −1.20 | −7.50 | 192 | 56 | +2.45 | −2.96 | −0.35 | 0.33 | −0.01 |
| S2 + one tick inside | −1.21 | −7.54 | 192 | 56 | +2.45 | −2.96 | −0.37 | 0.33 | −0.01 |
| S3 + regime guard | −1.10 | −7.93 | 164 | 58 | +2.02 | −2.51 | −0.30 | 0.28 | −0.01 |
| S4 + reversion lean | −0.65 | −2.64 | 316 | 72 | +2.43 | −3.15 | +0.58 | 0.50 | −0.01 |
| X1 passive reversion | −0.06 | −8.14 | 10 | 0.5 | +0.02 | −0.05 | −0.02 | 0.01 | 0.00 |

The clip is a tenth of the touch, about 20 USDT, so the amounts are small; the
sign and the basis points are the point. Every row is negative, and stays
negative under every cancellation rule, arrival rule and latency from 1 to
250 ms ([`mm_robustness_H.csv`](experiments/results/mm_robustness_H.csv)).

### Where the money goes

The touch quoter earns its spread and gives back more than twice as much to the
move in the five seconds after it is filled. The fills that reach a resting
order through the queue are worth something; the ones a print trades straight
through are worth −3.8 to −4.6 bp at five seconds, and for the gated quoters
they are three fills in four. On BICOUSDT every strategy's passive fills, taken
together, are worth less than being the passive side of the average print (the
dashed line).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/mm_markouts_dark.png">
  <img alt="Markouts by fill path against the market-wide passive benchmark" src="docs/images/mm_markouts_light.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/mm_decomposition_dark.png">
  <img alt="The decomposition of each strategy's daily net" src="docs/images/mm_decomposition_light.png">
</picture>

### Inventory

S1, the quoter every hypothesis is compared against, held a root-mean-square
position of 56 BICO (about 25 USDT) against a soft limit of six clips, and was
flattened back to that limit twice in thirteen days.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/mm_inventory_dark.png">
  <img alt="S1's inventory over the held-out fortnight" src="docs/images/mm_inventory_light.png">
</picture>

### The three hypotheses

| Hypothesis | Metric | Value | Day t (12 df) | Kill conditions that fired | Status |
|---|---|---:|---:|---|---|
| H1 inside the spread | S2 − twin, paired daily, USDT | −0.007 | −0.93 | K1 net ≤ 0, K2 no gain over the twin, K4 the stale placebo gains as much, K-fund | **killed** |
| H2.1 lean on the state | S4 − S1, paired daily, pooled, USDT | −91.81 | −0.55 | K1 no gain, K3 below the shuffled-state placebos, K-fund | **killed** |
| H2.2 rest on the signal | X1 − taker, net per attempt, bp | −9.96 | −1.78 | K1, K2 X1 itself loses (−4.92 bp), K3 following the move does as well, K4, K-fund | **killed** |
| H3 regime guard | S3 − S1, paired daily, USDT | +0.107 | +0.62 | K2 no better than the same flags at random times | **killed** |

None passes Holm's procedure across the four, and none needed to fail it: each
is killed by a condition stated in advance. The secondary H2.3 reads the
boundary block as well and waits for it. Every condition with the value it read
is in [`mm_kill_conditions_H.csv`](experiments/results/mm_kill_conditions_H.csv)
and in the pre-registration's
[results section](docs/preregistration/market_making.md#results-on-block-h-read-once-4-october-2026).

H2.2 was the maker case for the reversion result above: a resting order on the
fading side fills exactly when the move it fades is happening. On the same 376
triggers the taker version earned +6.38 bp per trade, as §27 found on this
fortnight, and the passive version lost 4.92 bp per attempt.

### Placebos

The real difference (red) against 50 placebos each: the reversion tape shifted
by one to twelve days and an hour or more (H2.1, H2.2), and the regime flags
moved around the fortnight with their count and spacing kept (H3). A hypothesis
survives only above the dotted 95th percentile; none is.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/mm_placebos_dark.png">
  <img alt="Each true difference against its placebos" src="docs/images/mm_placebos_light.png">
</picture>

### What it would take

The advantage ladder adds one advantage at a time to the touch quoter, on all
four instruments; from rung 2 each reads the future and is an upper bound, not
a strategy. Being first in the queue fills about twice as often and loses
more. On BICOUSDT nothing short of perfect foresight of the next second makes
the touch quoter pay; on the tighter instruments a forecast with an R² of 0.1 to
0.3 would.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/mm_ladder_dark.png">
  <img alt="Net bp of turnover by rung of the advantage ladder" src="docs/images/mm_ladder_light.png">
</picture>

### The fee that would make it pay

The maker fee at which each strategy's net over the fortnight is zero, against
Bybit's published linear-perpetual schedule, transcribed in October 2026 and
applied to 2024 (the schedule may have differed then). The best published maker
fee is 0 bp and the market-maker programme advertises up to a 1 bp rebate; on
BICOUSDT, S0 would need −3.0 bp. Only S4 on CRVUSDT and XRPUSDT, and S1 there,
whose profit was carried inventory rather than quoting, cross the line at a
published tier.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/mm_fee_breakeven_dark.png">
  <img alt="Break-even maker fee per strategy and instrument" src="docs/images/mm_fee_breakeven_light.png">
</picture>

### Outside the state

The boundary block (12 March to 7 April, where §27's reversion state is absent)
is read once, after these results are committed, by the next pull request. It
decides H2.3 and records H2's boundary predictions; it cannot revive a
hypothesis killed here.

### What this does not establish

One admitted instrument and thirteen days, so a minimum detectable effect of
about 0.9 USDT a day against a quoter that loses about 1.2; a 100 ms conflated
book, with queue dynamics inside it a bracketed rule rather than an
observation; no own impact; a 2026 fee schedule applied to 2024 data; and a
fortnight whose reversion signal had been read before by takers, so only the
execution increment was new information.

**In one paragraph.** Built carefully and tested as registered, market making
on public Bybit data does not pay here at the base fee: the spread a quoter
captures is smaller than the move that follows its fills, and none of the three
ideas registered to change that — stepping inside the spread, leaning on or
resting on the reversion state, or stepping back after regime breaks — survived
its own kill conditions on a fortnight that played no part in choosing it.
Resting on the reversion signal, the one place the earlier negative maker result
might have reversed, loses where crossing for the same signal wins. What the
ladder and the fee break-even add is the size of the gap: a rebate three times
the largest advertised, or a forecast of the next second no public data here
provides.

### A second round, on wide-spread instruments: all five hypotheses killed

Is there any condition under which this market maker earns money? A second
round asked it on data no market-making code had read, registered before a
single file of it was fetched
([pre-registration](docs/preregistration/market_making_round2.md), with
[`configs/mm_prereg_round2.yaml`](configs/mm_prereg_round2.yaml)). Of eight
wide-spread candidates, six were admitted on the 25-day development block
(ALICE, ALGO, JTO, ZEC and GALA by a tick of at least 4 bp, CAKE by round
one's rule), every searched value was frozen
([amendment](docs/preregistration/market_making_round2.md#amendment-1-2026-10-04-before-any-held-out-read-values-frozen-on-the-development-period)),
and the held-out fortnight was read once, with BICOUSDT's four pristine weeks
from 8 April 2024 for B4
([results](docs/preregistration/market_making_round2.md#results-of-the-first-held-out-read-read-once-4-october-2026)).

**Nothing survives: on the held-out data neither the wider spreads, nor
quoting only while the recent market left room, nor a professional fee tier
made this market maker pay from making markets.**

| Hypothesis | Held-out net, USDT per 100 USDT of clip a day | Killed by |
|---|---:|---|
| B1: the quoter, re-searched per instrument, at the base fee | −0.067 | K1, K-nbhd |
| B2: quoting only while the trailing room clears the fee, against quoting throughout | +0.026 (+0.092 over it) | K3 (no better than a time-shifted gate), K-queue, K-dir |
| B3a: the same quoter at the 0 bp maker tier | +0.845 | K-dir: the making part −1.32, the rest inventory |
| B3b: at the programme's −1 bp rebate | +2.217 | K-dir: the making part −0.78 |
| B4: the room gate on BICOUSDT's pristine weeks | −0.347 (−1.83 under its base) | K1, K2, K3, K-fund, K-nbhd |

The positive nets at the professional tiers are inventory carried through a
fortnight of broad rises, which the registration's K-dir was written to catch:
the quoter's spread less the move after its fills and the fees is negative on
every instrument at the base fee, and on all but three instrument-tier pairs
at the professional tiers. No hypothesis is
`candidate` or positive-and-inconclusive, so the basket's pristine weeks are
not read. How far each was from paying, the measure the registration fixes for
this case: the touch quoter would have broken even at a maker fee of −1.7 to
−3.1 bp, a rebate larger than any published.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/mm_round2_fee_breakeven_dark.png">
  <img alt="Second round: break-even maker fee per strategy and instrument" src="docs/images/mm_round2_fee_breakeven_light.png">
</picture>

S0's break-even is exact; S1 and G(base) were re-run over maker fees from −1.5
to +2.0 bp, and an arrow marks one beyond that range. S1's arrows to the right
are the inventory gains above, not making.

---

## The pipeline, step by step

Twelve steps from "which market" to "running in production". Eleven are built;
the last is marked and is not. The [pipeline map](#the-pipeline-map) draws the
same path with execution as the branch it now is; the steps below keep the
numbering they were built in.

**0. Choose the instrument.** `trading-research discover` enumerates what the
venue lists; `trading-research screen` ranks it on both halves of the edge
identity — how far the price moves against the cost, *and* whether the book
predicts where. Three days of data per instrument.

Both halves matter and the first version had only one. Ranked on movement alone
it put XMRUSDT third of forty-four; XMRUSDT moves nine times as far as BTCUSDT
and its book predicts nothing (correlation 0.0002 against 0.049), and the full
pipeline run on it did worse than on the instrument it was meant to replace.
Corrected, the ranking inverts.

The screen has itself been corrected twice, and both corrections are in
[§20](docs/results.md). It measured the dispersion of the *absolute* move where
the identity needs the signed one, understating every edge by about forty per
cent; and "edge is bounded by IC × dispersion" was used as an impossibility
argument when it bounds the *average* trade rather than a selective one. With
both fixed: **the best of forty-four instruments has an average edge about a
tenth of its cost, and BTCUSDT — the instrument most of this project was built
on — has a thirtieth.** That three-fold gap is larger than anything the
modelling ever moved, which is why the instrument is chosen before the model
and not after.

The screen is also the one step here with a stability check. Re-run on Bybit
across eighty instruments over two windows seven weeks apart, the rank
correlation of edge-over-cost is 0.59 — so it measures a real property, not
noise. But half the top ten changes between windows, so it supports taking a
basket of five to ten rather than trusting a single best instrument.

*Choosing the model class is still a judgement made by hand.*

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

**Regime monitoring.** A model fitted on one stretch of market is expected to
apply until the market changes, and a fixed retraining cadence is a guess about
when that is. Two detectors say instead when the data changed. The first
([`validation/changepoint.py`](src/trading_research/validation/changepoint.py))
reads book statistics in non-overlapping blocks at tick frequency. The second
([`validation/structural_breaks.py`](src/trading_research/validation/structural_breaks.py))
comes from [adia-structural-break](https://github.com/bamasa/adia-structural-break),
the author's solution to the ADIA Lab Structural Break Challenge, Real-Time
Edition (CrunchDAO, 2026; 0.6299 TS-AUC at the close of submissions, about rank 160
of 1,716 registered participants on the public leaderboard),
installed as a dependency rather than copied: the return series is whitened by
its own history — an AR(p) fit by BIC, a conditional scale and the empirical
distribution of the innovations — and every test runs on the whitened stream,
with Shiryaev-Roberts odds of a change read per family (scale, dependence,
mean). Thresholds are calibrated against a null, not chosen by eye; the figure
below is BTCUSDT daily from January 2024, with the breaks the monitor flags, the
odds behind them, and the window resets that keep the odds from accumulating
forever. `uv sync --extra breaks`, then
`trading-research breaks --symbol BTCUSDT --interval 1d --start 2024-01-01 --end 2026-09-30 --plot`.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/structural_breaks_BTCUSDT_dark.png">
  <img alt="BTCUSDT daily: breaks flagged on the whitened stream, with the family log-odds" src="docs/images/structural_breaks_BTCUSDT_light.png">
</picture>

On 1,003 daily returns the walk opens six windows and flags one break: **13
September 2025, scale, downward** — the odds of a fall in variance cleared the
calibrated threshold (9.36 against 9.09) seventy-five days into the window that
opened on 30 June, after a near miss on 29 June (8.96). Monitoring then paused
until the re-anchored history reached a third of a year, which is the shaded
stretch. The same walk over forty shuffled copies of the returns — the same
marginal distribution, no regime — flags one copy, so the null reading is one
flag in forty against one on the real series; the shorter-memory setting
(history 250, online 60) flags two scale breaks, 6 February 2026 up and 14
August 2026 down, at six flags in forty on its null, and is reported beside
it in [`experiments/results/`](experiments/results/structural_breaks_BTCUSDT_breaks.csv)
rather than preferred. No dependence break is flagged on this series at either
setting.

What the figure does not claim: that refitting at these breaks would have
improved any trading result. That comparison belongs to the refit-policy axis of
`grand_search.py` and is listed under the roadmap.

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
in the profit and loss. Taker on both legs by default: fee, spread, slippage.
Execution is a mode a run chooses
([`pipeline/execution.py`](src/trading_research/pipeline/execution.py)): the same
decisions can instead rest at the touch and fill only through the queue, or be
quoted on both sides by the event-time market maker, and every mode returns the
same table of attempts, misses included, for the same statistic to judge.

**9. Search the configuration.** Successive halving over sampled configurations
rather than a nested product — cheap candidates are killed after a few windows
and only survivors are measured on the full span. Everything is chosen on
validation and the test span is scored once. Execution and the regime policy
are axes as well, scored only at the last rung (`grand_search.py
--execution-axis`): they are never drawn, so the configurations a search samples,
and every result it produced before, are unchanged.

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

### Reproducing the finding, one command

```bash
uv run trading-research reproduce
```

runs every stage in order on a clean checkout: fetches what is missing, audits
which family of signal carries anything, searches the configuration on the
search block — taking the best *neighbourhood* rather than the best cell, with
the peak it declined reported beside the choice — reads the held-out block once
with thresholds carried over, and prints the result with both statistics and the
market state it depends on.

With `--execution both` the run adds one stage after the held-out read: the
chosen configuration's decisions executed both ways, crossing and resting at the
touch, on the instruments that have prints on disk, the mode chosen on the search
block and both reported on the held-out block with the fill rate. `--execution
maker` takes passive entry instead of choosing it. The default, `taker`, is the
run described here.

Two honesty properties are worth knowing before reading its output. The held-out
block is read once; if the numbers disappoint, that is the answer, and the run
does not go back. And the printed verdict comes from the clustered statistic,
which counts days rather than trades — on the spans in this repository the two
differ by a factor of five.

Run on the spans the documents use, the pipeline's own end-to-end choice lands
one cell away from §27's configuration and loses on the held-out block — which
is reported here rather than smoothed over, because it is a fair measure of how
sharp the effect's edge is: one neighbouring cell, and it is gone.

### Reproducing the market-making study

```bash
uv run python -m experiments.market_making --workers 8          # block D: screen, search, freeze
uv run python -m experiments.market_making_heldout --dry-run    # every path, on the last days of D
uv run python -m experiments.market_making_heldout --block H    # the held-out read, through the ledger
uv run python -m experiments.market_making_verdicts --block H   # verdicts and tables, no market data
```

The held-out read refuses to start unless the frozen configuration is committed
and the tree is clean, and the ledger it writes makes a second read refuse too;
on this repository H has been read, so the third command reproduces it only
with `--force`, stamped `second read`. The fourth reads only the run's outputs,
so the verdicts can be recomputed without touching the block. The read took 83
minutes on eight workers.

One strategy on one span, from the command line:

```bash
uv run trading-research mm-backtest --symbol BICOUSDT --start 2024-02-01 --end 2024-02-25 \
    --strategy s2 --frozen --fee-tier base --latency-ms 10 --cancel-model pessimistic \
    --workers 8 --output artifacts/mm
```

It prints the day-by-day table and the decomposition (spread, adverse selection,
inventory, fees, funding) and writes both with a manifest. Parameters come from
the frozen registration with `--frozen`, or by hand (`--skew-bp`, `--k`,
`--min-edge-bp`, `--m-ticks`, `--clip-notional`, `--sigma-ref`). A day of the
held-out or boundary block is refused unless `--allow-heldout` is given; then it
is read only with the frozen values and only through the ledger, so on this
repository a second read of H also needs `--second-read` and is recorded as one.

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
funding as daily archives, and bars (`klines`) at every interval as monthly
ones. Used for the instrument screen, which needs breadth rather than depth,
and — the bars — for regime monitoring, which needs years rather than days.
Note that daily `bookTicker` archives stop after 30 March 2024, which is why the
screen can be measured on two windows and not more.

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
from trading_research.data.ensure import ensure_bars, ensure_book, ensure_universe

# ten levels a side, about two minutes of replay per day
ensure_book("DOGEUSDT", date(2024, 2, 1), date(2024, 3, 10))

# top of book only for a cross-section: eleven seconds per instrument-day
ensure_universe(["BTCUSDT", "ETHUSDT"], date(2024, 2, 1), date(2024, 3, 10))

# daily bars for regime monitoring: a few kilobytes a month
ensure_bars("BTCUSDT", "1d", date(2024, 1, 1), date(2026, 9, 30))
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
| [`execution_assumptions.md`](docs/execution_assumptions.md) | Cost model, fills, what each execution mode (taker, passive entry, market maker) does and does not simulate |
| [`market_making_simulator.md`](docs/market_making_simulator.md) | The event-time market-making simulator: every rule, the direction of its bias, and the test that pins it |
| `limitations.md` | What this does not establish |
| `disclosure_policy.md` | What is public here and why, and what is not |
| `preregistration/market_making.md` | The market-making study, registered before any code or number: blocks, hypotheses, kill conditions, placebos, amendment protocol |
| [`preregistration/market_making_round2.md`](docs/preregistration/market_making_round2.md) | The second market-making round, on eight wide-spread instruments, registered before any of their data is fetched |
| [`preregistration/short_horizon_2025.md`](docs/preregistration/short_horizon_2025.md) | The author's 2025 short-horizon recipes, re-run on Bybit's public BTCUSDT archives, registered before any 2025 data is fetched; predicted negative at published fees |

---

## Limitations

Stated plainly, because the honest version is more useful than the flattering
one:

- **No profitability claim is made.** No result in this repository should be
  read as evidence that any strategy is or was profitable.
- **Synthetic results are circular.** They demonstrate that the pipeline works,
  not that markets behave this way.
- **The backtest is a simulation.** The taker backtest does not model queue
  position, partial fills against a real book, latency, market impact, or the
  fact that a real order changes the book it is trading against. The
  market-making simulator models queue position, partial fills and latency, but
  from a 100 ms book rather than a message feed, with cancellations attributed
  by rule, own impact ignored, and a crossed snapshot filling nothing without a
  print — each stated with its direction in
  [`docs/market_making_simulator.md`](docs/market_making_simulator.md). Its
  verdicts rest on one admitted instrument and thirteen held-out days.
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
- [x] Instrument screening by headroom, before anything is fitted — **and the
      correction that followed**: the screen measured the dispersion of the
      *absolute* move where the identity needs the signed one, understating
      every edge by about forty per cent (§20)
- [x] A review rubric and an evidence pack for it, applied to this project's
      own result
- [x] **Step 0 automated.** A screen over eighty instruments on the venue that
      is actually traded, measured on two windows seven weeks apart. Rank
      correlation 0.59, so it measures a real property — but half the top ten
      changes between windows, so it supports a basket of five to ten rather
      than a single best instrument
- [x] Multi-level order books reconstructed from Bybit's incremental stream,
      ten levels a side, with sequence-gap detection
- [x] Trade prints with the aggressor's side — what a book cannot say, and what
      a maker model needs
- [x] **A queue-level maker model**: an order joins behind the size resting at
      its price and fills only once that much opposing volume has traded
      through. Measures adverse selection rather than assuming a figure for it
- [x] Regime-break detection: a CUSUM over four block statistics of the book at
      tick frequency, with its threshold calibrated against forty synthetic
      nulls rather than by eye
- [x] Regime monitoring on the whitened stream, from the ADIA structural-break
      solution, as a dependency: scale, dependence and mean breaks on any
      return series, thresholds calibrated against Gaussian nulls, the history
      re-anchored at each break
- [x] Randomised tree ensembles, a second booster, bagging over stretches of
      history, voting with a diversity figure, and stacking on forward-chained
      out-of-fold predictions
- [x] **A bias-variance decomposition** that says which ensemble is worth using,
      resampling contiguous history rather than random rows — on this data it
      says none of them are, because the variance to average away is not there
- [x] **An information audit**: each data source measured before a model is
      chosen, for what it reaches alone and what it adds on top of the sources
      already accepted. This is what found the cross-sectional signal
- [x] Cross-instrument features — leader and index returns, catch-up, beta,
      dispersion, rank — the omission that had no good excuse
- [x] Order-flow features from trade prints, alongside the queue features
- [x] Position sizing rules: volatility targeting, a confidence ramp, fractional
      Kelly — documented with the identity that sizing is multiplicative in the
      edge and cannot turn a losing one positive
- [x] Pairs trading: hedge ratio, Ornstein-Uhlenbeck residual, half-life
- [x] Data that fetches itself when it is missing, so a clean checkout runs
- [x] A findings register with each candidate's status and the conditions that
      would kill it, written before the test rather than after
- [x] **Execution as a searched axis**: taker, passive entry and the event-time
      market maker behind one interface
      ([`pipeline/execution.py`](src/trading_research/pipeline/execution.py)),
      in the grand search at its last rung, in `reproduce --execution`, and in
      `trading-research mm-backtest`

Next, and in this order, because the first one decides whether the rest matters:

- [ ] **The decisive test.** The frozen reversion configuration on days neither
      block has seen. Four earlier findings in this project looked at least as
      good and did not survive it. The span could not be fetched because the
      venue began rate-limiting sustained downloads; this is a matter of waiting,
      not of code.
- [ ] **A mechanism for the reversion.** Correlation without a cause is what
      falls over. Liquidation cascades, funding settlement and session
      boundaries are each testable, and an effect that concentrates in explicable
      windows is worth more than one spread evenly.
- [x] **The maker case for this signal specifically — tested, and killed.**
      Pre-registered as H2.2 and read once: resting on the fading side lost
      4.92 bp per attempt where crossing for the same triggers earned +6.38
      ([market making](#market-making-quoting-both-sides-on-public-bybit-data)).
- [ ] A volume-weighted index rather than an equal-weighted one, and the same
      question asked of order flow rather than price
- [ ] Refitting at the breaks the whitened monitor flags, against the fixed
      cadences of §10, on the refit-policy axis `grand_search.py` already has
- [ ] Unified report: calibration, equity, drawdown, cost attribution, regimes

**Longer horizons, in the same pipeline.** Everything above trades in seconds to
minutes, where costs and queue position decide the result. The same validation,
costs and search apply at hours and days, where a move is many times the cost of
a trade:

- [ ] Horizon as a searched axis, next to execution, with a first screen of the
      typical move over the horizon against the round-trip cost, per instrument
- [ ] **Funding-rate carry**: long spot and short the perpetual in equal size,
      so the price cancels and the position collects the funding that leveraged
      longs pay; on Binance and Bybit history from 2019, pre-registered, with
      held-out years
- [ ] Trend and cross-sectional momentum on daily and four-hour bars across a
      basket, with a portfolio backtest that rebalances on a schedule

**Market making, registered before it is built.** The maker case above is one
of three hypotheses in a market-making study: a two-sided quoter with inventory
limits, simulated in event time with a queue per order and fills only from
Bybit's trade prints. The blocks, the instrument admission rule, the hypotheses
with their kill conditions and placebos, the advantage ladder, the fee
break-even and the amendment protocol are in
[`docs/preregistration/market_making.md`](docs/preregistration/market_making.md),
with the values the code will read in
[`configs/mm_prereg.yaml`](configs/mm_prereg.yaml). Both were committed before
any simulator code existed and before any market-making number was computed on
any block.

- [x] **The event-time simulator**: fills only from prints, a queue per order
      with bracketed cancellation rules, latency, post-only, inventory limits,
      fees, funding and an accounting identity checked at every change, tested
      on known-answer markets ([rules](docs/market_making_simulator.md))
- [x] The quoters, the regime flags and the development-period search, with
      the amendment that froze every searched value
- [x] **The held-out fortnight, read once: all four hypotheses killed**, the
      advantage ladder and the fee break-even measured
      ([results](docs/preregistration/market_making.md#results-on-block-h-read-once-4-october-2026))
- [ ] The boundary block, read once: H2.3 and H2's boundary predictions
- [x] **A second round on wide-spread instruments, read once: all five
      hypotheses killed**; the touch quoter would have needed a 1.7 to 3.1 bp
      rebate ([results](docs/preregistration/market_making_round2.md#results-of-the-first-held-out-read-read-once-4-october-2026))
- [ ] Capacity: what a quoter could trade before its own size moves the book it
      is quoting into, which the simulator does not model

**The 2025 short-horizon study, registered before any of its data is read.**
The author's 2025 order-book research (a boosted classifier with a hold-first
rule, a Gaussian-head network with an expected-value gate, a close-by-time
exit) was never measured at real fees, on an audited fill rule, or on a
window it did not choose. It is re-run here on Bybit's public BTCUSDT
archives. The blocks, the hypotheses (T1, T2 and M1 at the base fee; Z, the
gross edge at a zero-fee venue, as a conditional secondary; R, the autumn
break, only for a survivor), their kill conditions and placebos, and the
expected outcome are in
[`docs/preregistration/short_horizon_2025.md`](docs/preregistration/short_horizon_2025.md),
with [`configs/short_horizon_2025.yaml`](configs/short_horizon_2025.yaml). The
prediction is negative: the original's own summary implies about 0.36 bp of
turnover, against a cheapest published Bybit taker fee of 1.5 bp.

- [x] Pre-registration, with an archive check that moved the held-out block's
      end to 2025-08-20, where Bybit's book archive changes from 500 to 200
      levels
- [x] Reviewed for peeking, multiplicity and feasibility before any data is
      read, and tightened: no file of a later block on disk before the ledger
      opens it, statuses computed by code and hashed, no code change between
      the held-out reads, metrics fixed at the day level, Holm's family fixed
      at three on every block, a break test whose placebo goes through the
      same selection, and fill and timing rules pinned in the configuration
- [ ] The ported pieces: ten-level tensor, walked-book label, Gaussian head
      with a lazy window dataset, EV gate, an event-time taker executor and a
      signal-entry quoter, fill-rule comparison, streaming parity, the
      200-level book reader proven on three 2026 days outside the study, and
      the study's ledger and single entry point
- [ ] Development period and the amendment that freezes it
- [ ] The held-out read, then the boundary block and the break

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

Maker execution — built, and it does not pay
--------------------------------------------
An early section estimated what posting would be worth by assuming a figure for
adverse selection. That estimate was replaced by a measurement, and the
measurement disagreed with it.

[`backtest/maker.py`](src/trading_research/backtest/maker.py) simulates the
mechanism: an order joins the queue behind the size resting at its price and
fills only once that much opposing volume has traded through, using Bybit's
trade prints for the flow. Costs fall exactly as the arithmetic promised, from
14–16 bp to 5.9–7.1 with a passive exit. **The gross edge inverts**: the same
signal at the same moments is worth +3 to +8 bp crossing and −0.5 to −4.2 bp
resting. Adverse selection is 5–11 bp — the same size as the fee saving. Zero of
48 configurations positive on net or on gross.

The mechanism shows in the detail: joining the touch is *worse* than posting
behind it, because filling faster means filling when the market is moving
against you, while the favourable moves leave the order unfilled.

Three optimisms remain, stated in the module rather than buried: the queue ahead
never grows, cancellations ahead are ignored, and the order's own size is
treated as negligible. All three flatter the passive strategy, and it loses
anyway.

What is left to do here:

- [x] The conditional version of the same measurement for the reversion signal,
      where being filled against the move might have been an advantage rather
      than the usual tax. Done in the market-making study, with a queue per
      order, fills only from prints and a taker exit: it loses too (H2.2,
      [market making](#market-making-quoting-both-sides-on-public-bybit-data)).
- [ ] Capacity. A fill rate in the teens turns a strategy that took 3,297 trades
      into one that takes a few hundred, which establishes much less.

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
