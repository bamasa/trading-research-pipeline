# Extending the pipeline: one file per strategy, model, feature set, label or execution

A fork adds a part by adding one file. The registry finds the file, the
conformance suite checks it the way the studies check their own code, and the
plugin commands list it, show it, check it and backtest it. Nothing else is
edited: no dict in `pipeline/stages.py`, no `if` in a command, no frozen file.

[`extending_survey.md`](extending_survey.md) records how the pipeline found its
parts before the registry existed; the registry adapts those parts without
changing them.

## The five kinds

| kind | decorator | file goes in | declares | does |
|---|---|---|---|---|
| strategy | `@register_strategy("name")` | `src/trading_research/strategies/contrib/` | `lookback`, `horizon`, `inputs` (optional `stateful`) | `fit(history)`, `decide(frame)` -> -1, 0, +1 per row |
| model | `@register_model("name")` | `src/trading_research/models/contrib/` | `inputs`, `lookback` (optional `target`) | `fit(x, y)`, `predict_proba(x)` -> `(n, 3)`: sell, hold, buy |
| features | `@register_features("name")` | `src/trading_research/features/contrib/` | `lookback`, `inputs` | `compute(book, trades)` -> columns on the book's index |
| label | `@register_label("name")` | `src/trading_research/labels/contrib/` | `horizon`, `kind` | `compute(frame)` -> one value per row, NaN where the outcome is unknown |
| execution | `@register_execution("name")` | `src/trading_research/backtest/contrib/` | `fill_rule`, `fees` | `execute(decisions, market, hold=)` -> `ExecutionResult` |

The protocols are in `src/trading_research/plugins/protocols.py`. A plugin
subclasses nothing; any object with the declarations and the methods is one.
Look-backs and horizons are counted in rows of the frame.

**The frame** a strategy reads is one instrument on a regular grid (one second
by default), each row labelled by the end of its bin so that it holds only what
had been observed at its label: the book's level columns (`bid_price_0`,
`bid_size_0`, ... from `data/schema.py`), `mid`, `log_mid`, `spread_bp`, the
prints each row owns under the backward rule (`buy_volume`, `sell_volume`,
`trade_count`, `signed_notional`, `traded_notional`, `largest_trade`,
`vwap_deviation_bp`) and, for cross-sectional rules, `peer_<k>_mid`
(`plugins/market.py`).

**Where plugins come from.** The registries are filled the first time one is
read, from three places, in this order:

1. the built-ins, `src/trading_research/plugins/builtins/`: adapters that
   register what the repository already has (below);
2. every module in each kind's `contrib/` folder (a module whose name starts
   with `_` is skipped);
3. installed packages that declare entry points, so a separate package can add
   plugins without touching this repository. In that package's `pyproject.toml`:

   ```toml
   [project.entry-points."trading_research.strategies"]
   my_rules = "my_package.rules"          # a module whose decorators register
   breakout_v2 = "my_package.rules:BreakoutV2"   # or one class, under this name
   ```

   The groups are `trading_research.strategies`, `.models`, `.features`,
   `.labels` and `.executions`.

A module that fails to import, or registers a name already taken, is reported
by `strategies list` and hides nothing else.

## Adding one, step by step

The commands below are `trading-research strategies ...`; the group lists every
kind, not only strategies.

1. **Write the file from the template.**

   ```
   trading-research strategies new my_rule --kind strategy
   ```

   writes `src/trading_research/strategies/contrib/my_rule.py`, a working
   plugin with its declarations explained, and `tests/test_plugins_contrib_my_rule.py`.
   Both pass as written (`--kind model`, `features`, `label` and `execution`
   work the same way). The name is the module name: lower case, digits and
   underscores.
2. **Replace the body, and keep the declarations true.** `lookback` is the rows
   of history a decision (a prediction, a feature) at row *t* reads, *t*
   excluded; `horizon` is the rows a position is held (a label reads forward).
   Anything chosen from data is chosen in `fit`, which is handed only the rows
   before trading starts. A strategy whose decisions carry state from row to
   row (a band it stays in, a position it remembers) says `stateful = True`.
3. **Run the test, then the conformance suite.**

   ```
   python -m pytest tests/test_plugins_contrib_my_rule.py
   trading-research strategies check my_rule
   ```

   `check` writes `artifacts/plugins/my_rule/page.md` and exits 1 if any check
   failed. What it checks is below.
4. **Backtest it.**

   ```
   trading-research strategies backtest my_rule --execution taker,passive
   ```

   fits it on the first 40% of the synthetic demo market and runs its decisions
   after that through each execution named, with a row of the fee table
   (`--fee-tier`, the base tier by default). `--input data/demo` runs it on a
   dataset in the store's layout instead (`generate-demo-data` writes one).
   A model is backtested the same way: it is fitted on the history, its
   confidence threshold is chosen on the history's last 30%
   (`backtest.evaluate.choose_confidence`), and it trades through
   `backtest.evaluate.decide`.

## What `check` checks

| | strategy | model | features | label | execution |
|---|---|---|---|---|---|
| **a. protocol** | declarations of the right type, inputs the frame carries, `decide` returns -1/0/+1 per row, the same twice, and writes nothing into the frame | the same for `fit` and `predict_proba` (rows of probabilities summing to one) | a numeric frame on the book's index, the same twice | one value per row, classes in -1/0/+1, the last row unlabelled | a stated fill rule, a fee-table row, the attempts' columns |
| **b. no look-ahead** | every input after a row changed (prices by `validation/fold_leakage.mutate_after_cutoff`, the rest reversed in time and scaled by 1.5): decisions up to the row, digested, unchanged, at 20 rows | the studies' fold-level test (`check_fold`, sha256 of the fit's predictions) on the fit, and its predictions up to a row unchanged (to 1e-12) when every later input row is reversed and tripled | truncation invariance at five cuts (`validation/leakage.py`'s rule) | rows beyond the horizon changed: labels up to the row unchanged | the market after a cut removed: settled attempts unchanged |
| **b. declared look-back** | the decision from only `lookback + 1` rows equals the batch decision (not applied to a stateful strategy) | the prediction from only `lookback + 1` input rows equals the batch one | each column from `lookback + 1` rows equals the batch value | | |
| **c. execution** | fitted on the first 40% of the demo market, its decisions after that through every execution plugin with the fee tier applied: one result row each | the same, as a model strategy | n/a | n/a | a fixed reference stream through it |
| **d. synthetic** | the planted-edge, null and placebo checks of `trading_research.synthetic` when that package offers `plugin_checks(kind=, name=, build=)`; otherwise "skipped: synthetic checks not on this branch" | as for a strategy | n/a | n/a | n/a |

What it cannot catch: a mutation test is black-box, so a strategy that peeks
one row ahead is caught only where the peeked value flips a decision (twenty
cutoffs make that likely, not certain); an execution is tested for reads beyond
its hold and entry timeout, not within them; and nothing here checks that the
data was timed correctly in the first place, which is the data contract's job.

## A worked example: book imbalance with a clock exit

`src/trading_research/strategies/contrib/book_imbalance.py` is the whole
strategy, added as a fork would add it: one file, one decorator, no other edit.

```python
@register_strategy("book_imbalance")
class BookImbalance:
    """Go with the heavier side of the touch once its imbalance is in the top share; exit on the clock."""

    inputs = ("bid_size_0", "ask_size_0")
    lookback = 0

    def __init__(self, *, hold: int = 30, share: float = 0.05) -> None:
        self.horizon = hold          # the clock exit, in rows
        self.share = share
        self.threshold = float("nan")

    def fit(self, history):          # the threshold from a trade rate, on history only
        values = np.abs(self.imbalance(history))
        self.threshold = float(np.nanquantile(values, 1.0 - self.share))
        return self

    def decide(self, frame):         # with the heavier side, where the imbalance is large
        value = self.imbalance(frame)
        out = np.zeros(len(frame), dtype=np.int8)
        strong = np.isfinite(value) & (np.abs(value) >= self.threshold)
        out[strong] = np.sign(value[strong]).astype(np.int8)
        return out
```

The synthetic generator plants exactly this effect, weakly: its latent signal
skews the touch sizes and drifts the next mid by 0.15 of the current
volatility. So the example tests that the pipeline finds a planted edge and
charges for it; it says nothing about a market.

**Listed.** `trading-research strategies list --kind strategy` shows it beside
the built-ins, with origin `contrib`; `strategies show book_imbalance` prints
`lookback 0`, `horizon 30`, its file and the file's sha256.

**Checked.** `trading-research strategies check book_imbalance` on the demo
market (72,000 snapshots of 200 ms, seed 20260101, gridded to 14,400 one-second
rows; fitted on the first 5,760):

| check | status | measured |
|---|---|---|
| a. protocol | pass | 214 long and 238 short decision rows in the trading span |
| b. no look-ahead | pass | 20 cutoffs, no decision moved |
| b. declared look-back | pass | 20 rows (10 with a decision) from a 1-row window, no mismatch |
| c. execution | pass | taker, passive, maker and event_taker, base fee tier |
| d. synthetic | skipped | synthetic checks not on this branch |

**Backtested in taker and passive modes.**

```
trading-research strategies backtest book_imbalance --execution taker,passive
```

printed, at the base tier (2 bp maker, 5.5 bp taker, 0.5 bp slippage on a
crossing leg), per attempt with misses included, in bp of notional:

| execution | attempts | fills | fill rate | gross | cost | net | net, total |
|---|---|---|---|---|---|---|---|
| taker | 168 | 168 | 1.000 | +1.956 | 12.082 | -10.126 | -1,701.2 |
| passive | 130 | 98 | 0.754 | -0.122 | 6.093 | -6.215 | -808.0 |

Crossing captures the planted move, about 2 bp a trade before costs, and pays
about 12 bp to do it. Posting halves the cost and loses the move: the orders
that fill are the ones the flow ran through, which is the adverse selection
`backtest/maker.py` describes. Neither clears its fees; on this generator's
default strength nothing at this horizon would. The check's own run puts the
event-time modes beside them: `event_taker` +1.715 gross, -9.285 net on 164
attempts; `maker` (the market-making simulator's queue) a fill rate of 0.671,
-0.232 gross, -5.265 net on 149.

## Putting a plugin into a study

A study names its parts; it does not import them. The registry API a study
runner calls:

```python
from trading_research.plugins import get, resolve
from trading_research.plugins.conformance import check, make_setup

spec = get("book_imbalance", "strategy")        # refuses an unknown or ambiguous name
assert check(spec, make_setup()).passed         # refuse a part that fails its checks
strategy = resolve("strategy", "book_imbalance", hold=30)
```

and what a registration records for each part, so that the frozen-code checks
cover a plugin the way they cover the rest of the study code:

```yaml
strategy:
  name: book_imbalance
  params: {hold: 30, share: 0.05}
  module: trading_research.strategies.contrib.book_imbalance
  sha256: <spec.fingerprint(): the sha256 of the defining file>
```

The study rules of this repository apply unchanged: the registration, its
hypothesis and its kill condition are committed before the data is read; a
parameter set on the development block is a `null` placeholder frozen by the D
amendment; once the code is frozen, the plugin's file is study code and a
change to it is a dated correction. The conformance page belongs with the run
record: it is the evidence that the part was checked, and the sha256 on it says
which file was.

## The built-ins

The adapters in `src/trading_research/plugins/builtins/` register, without
changing them:

- **strategies**: `reversion`, `relative`, `spread`, `hold_first`, `ev_gate`
  (`strategies/`; `clock_exit` and `sizing` are an exit schedule and a size
  multiplier, not decision rules, and are not registered);
- **models**: every name `stages.build_model` knows, the sixteen of `MODELS`
  and the thirteen named combinations, each built by `build_model` itself;
- **feature sets**: `book`, `depth`, `flow`, `generated`, `trade_stats`,
  `microstructure_2025` (`cross`, `positioning` and `lob_tensor` read more than
  one instrument's book and prints and are not registered);
- **labels**: `directional`, the seven targets of `labels/targets.py`,
  `walked_taker`, `taker_forward_move`, `control_move`;
- **executions**: `taker`, `passive`, `maker` (the market-making simulator's
  signal-entry quoter) and `event_taker`, the modes of `pipeline/execution.py`.

`trading-research strategies check --all` checks every one, each in its own
process with one OpenMP thread (the network and a booster in one process can
deadlock: `ensemble_all`'s fit hung past 90 s with the default threads and
finished in seconds with one), and writes `artifacts/plugins/index.md`.

On the demo market, at the base tier, 52 of the 56 registered plugins pass
every check. The four that do not are reported, not changed:

| plugin | failed check | what the check found |
|---|---|---|
| model `spread_capture` | b. no look-ahead (predict), b. declared look-back | its "tight spread" cut is `np.nanquantile` of the spread over the whole batch handed to `predict_proba`, so a prediction moves when later rows change (9 of 10 cutoffs) |
| model `random_forest`, model `extra_trees` | b. no look-ahead (fit) | the fit is not reproducible to the bit on the same rows (threaded trees, changes of 2e-16), so the fold-level test's sha256 differs; no prediction moved by more than 1e-12 when later inputs changed |
| features `generated` | b. declared look-back | its exponentially weighted means (`*_ewm200`, `*_ewm1000`) remember every earlier row, so no finite look-back is true; truncation invariance holds, so nothing reads forward |

Every other strategy, model, feature set, label and execution passes (a),
(b) and, where it applies, (c); (d) is skipped on this branch. The existing stage commands
(`train`, `backtest`, `reproduce`, `mm-backtest`) keep their own names and are
unchanged; the registry is how new code reaches the parts.
