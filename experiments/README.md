# Experiments

The runs behind the tables in [`../docs/results.md`](../docs/results.md). One
script per question, each printing a table and writing it to `results/`.

They exist so that a number in the document can be checked rather than taken on
trust. If a script and the document disagree, the script is right and the
document is stale — that has already happened once, and §6 was rewritten to the
numbers that reproduce rather than the other way round.

## Running them

Most read prepared feature files, so build those first:

```bash
uv run trading-research prepare --symbol BTCUSDT --horizon 1200 --subsample 50 \
    -o artifacts/prepared_BTCUSDT
```

A few read the raw book instead, because they need bid and ask sizes that
`prepare` does not carry forward — order-flow imbalance among them. Those need
`data/raw/<symbol>/` as the downloader leaves it.

| Script | Document | What it answers |
|---|---|---|
| `ceiling.py` | §1 | Share of moments whose move clears the cost, assuming perfect foresight |
| `signal_decay.py` | §2, §3 | Feature correlation by horizon, and why edge per trade is flat across horizons |
| `walk_forward.py` | §4, §5 | Model comparison over seven folds, at three cooldowns |
| `selection_leak.py` | §6 | The same strategy scored with the cutoff chosen on test and on validation |
| `fee_sensitivity.py` | §9 | What fee would be needed to break even. Reads every walk-forward run present |
| `maker_threshold.py` | §11 | Adverse selection of passive fills, measured against what posting saves |

Retraining schedules (§10) are not here: they run through the CLI, as
`trading-research retrain-search`.

## One model per invocation

`walk_forward.py` and `selection_leak.py` take a single `--model`. XGBoost and
PyTorch each bundle an OpenMP runtime and deadlock when both are used in one
process on macOS — the network fits 40k rows in twenty seconds alone and never
returns after a boosted tree has run in the same interpreter.

```bash
for m in logistic xgboost tcn; do
    uv run python -m experiments.walk_forward --model "$m" --symbols BTCUSDT
done
```

## What these are not

Not library code, and not imported by anything under `src/`. The dependency runs
one way. An experiment is a record of a question asked once; the package should
not carry weight for it.

Nor are they a backtesting harness. Each one answers its question and stops, and
none of them should be read as evidence that a strategy works — see
[`../docs/limitations.md`](../docs/limitations.md).
