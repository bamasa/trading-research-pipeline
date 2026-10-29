# How the pipeline finds its parts today

A survey of the seams, written before the plugin registry
(`src/trading_research/plugins/`) was added on top of them. Nothing below was
changed; the registry adapts these parts as they are.

## What is already a registry

| Part | Where | How a part is found |
|---|---|---|
| Book features | `features/registry.py` | `REGISTRY`, filled by `@feature(name=, plane=, lookback=)` when `trading_research.features` is imported (`book`, `microstructure`). Each feature declares a row look-back; `validation/leakage.py` tests the claim by truncation and by recomputing from the declared window. |
| Timed features | `features/timed.py`, `features/study_set.py`, `features/micro2025.py` | `TimedRegistry`, built by a function (`microstructure_registry()`), not at import. Declares `lookback_s` and `lookback_rows`; `validation/leakage_timed.py` tests truncation and streaming parity. |

## What is a hard-coded dict or tuple

| Part | Where | Keyed by |
|---|---|---|
| Models | `pipeline/stages.py`: `MODELS` (16 `"module:Class"` strings), `ENSEMBLES`, `UNCERTAIN`, `META`, `COMPOSITES`, `REGRESSION_MODELS` | `build_model(name, **params)`; used by `train`, `experiments/strategies.py`, `experiments/targets.py`, `experiments/grand_search.py`. A new model means editing the dict. |
| Targets | `labels/targets.py`: `TARGETS` (7 entries) | `build_target(name, frame, cost_bp=, horizon=)`; each `TargetSpec` declares `horizon` and `kind`. |
| Execution modes | `pipeline/execution.py`: `ExecutionMode` (taker, passive_entry, market_maker, event_taker, signal_entry), `VALID_PAIRS` | `execute(signals, mode, ...)` dispatches with `if` branches. `pipeline/discovery.py` has its own `EXECUTIONS = ("taker", "maker", "both")` for `reproduce`. |
| Market-making quoters | `pipeline/market_making.py`: `STRATEGIES = ("S0".."S3")`, `PARAMETERS`, `make_quoter` | `mm-backtest --strategy`; the plan refuses any other name. |
| Fee tiers | `backtest/costs.py`: `BYBIT_LINEAR_TIERS` | `fee_tier(name)`. |
| Short-horizon models | `pipeline/short_horizon/models.py`: `C_MODELS`, `R_MODELS` | the study's stages only. |

## What is neither: modules a caller imports by hand

`strategies/` has no package file and no registry. Each module is a set of
functions or frozen dataclasses that a script wires together:

- `reversion.py`: `index_level`, `signal`, `threshold_for_rate`, `decide` over a
  panel of log mids; `ReversionConfig` (look-back 120 rows, hold 120 rows).
- `relative.py`: `deviations`, `weights`, `backtest` over a panel (`CrossSection`).
- `spread.py`: `fit_hedge`, `fit_ou` on a training window, then `z_score` and a
  stateful band `signal` for a pair.
- `hold_first.py`: `fit_hold_first` (thresholds from a trade rate on the inner
  validation) and `HoldFirstRule.decide_proba` over a model's probabilities.
- `ev_gate.py`: `EVGate.sides` over a predicted mean and sigma in bp and the walk
  cost; `gate_events` and `thin_entries` for one entry per event.
- `clock_exit.py` and `sizing.py` are not decision rules: an exit schedule
  (used to thin entries) and position-size multipliers.

Each existing command picks its rule in code: `backtest` takes probabilities and
`decide` with a confidence threshold; `reproduce` runs the reversion rule only;
`mm-backtest` takes `S0`..`S3`; `experiments/strategies.py` takes a model name.

## The protocols the parts already follow, implicitly

- **Strategy**: decisions of -1, 0 or +1 per row of a frame, causal in the rows
  (`reversion.decide`, `HoldFirstRule.decide`, `EVGate.sides`), with a look-back
  and a hold in rows or seconds (`ReversionConfig.lookback`/`hold`,
  `ClockExit.hold_s`). A threshold or hedge is fitted on an earlier window.
- **Model**: `models/base.Model`: `fit(x, y)` on a training block and
  `predict_proba(x)` as an `(n, 3)` array in `CLASSES` order (sell, hold, buy).
  Inputs are implicit: rule models read one named `column`; learned models read
  whatever columns they were fitted on; the TCN reads a window of past rows.
- **Feature set**: a function from the book plane (and the trades plane) to a
  frame on the same index, causal, with a declared look-back in rows (registry)
  or seconds and rows (timed). Builders: `depth`, `flow` (on `bucket_prints`),
  `generated`, `align.build_trade_features`.
- **Label**: a function from the frame to a Series that reads forward exactly a
  declared horizon (`DirectionalLabel.horizon`, `TargetSpec.horizon`,
  `WalkedTakerLabel.purge_rows`), NaN where the outcome is not in the data.
- **Execution**: `SignalStream` (stamped directions and a hold) in, the
  `ATTEMPT_COLUMNS` table and a summary out, under a cost model
  (`TakerCosts`, `MakerCosts`, a `FeeTier` inside `TakerSettings` or
  `SimConfig`) and a fill rule (cross at the touch; the grid queue of
  `backtest/maker.py`; the simulator's `FillRule` queue, touch or trade-through).

## The look-ahead tests a plugin can reuse

- `validation/leakage.py`: truncation invariance and the declared look-back, per
  feature.
- `validation/fold_leakage.py`: `mutate_after_cutoff` (prices moved by a random
  walk, bid and ask pushed apart, labels redrawn) and `check_fold`, which fits
  twice and compares artefacts by sha256. The studies use it for every fold.
- `validation/leakage_timed.py`: the same for timed features, plus streaming
  parity.

## What this means for a fork

Adding a strategy today means a new module plus edits to a script or a command;
adding a model means editing `MODELS`, a frozen file. The registry adds one
place to register each kind, discovers files dropped under `contrib/` and
packages that declare entry points, adapts the parts above without editing
them, and runs the checks above on anything registered.
