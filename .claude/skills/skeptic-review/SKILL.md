---
name: skeptic-review
description: Review a study's code, run or result for leakage, fidelity and execution faults before its verdict is read. Use when asked to review a study, check a backtest for look-ahead or overfitting, find what would refute a result, or decide whether a finding blocks the verdict.
---

# The skeptic's review

The role is `agents/roles/skeptic.md`; the scoring is
`docs/evaluation/rubric.md`. A review is three lenses, each with the checks it
ran and the refutation it tried, and findings with `file:line`, severity and
lens. A finding blocks the verdict when, if true, it could change a status.

## The three lenses

- **Leakage.** Does any value used at a decision depend on data after it?
  Thresholds fitted on the block they are scored on; features computed over a
  window that crosses the decision; a held-out day reached by a route other
  than the ledger; a fold boundary that shares a day.
- **Fidelity.** Does the code do what the registration says? Fees, fill rule,
  units, the order of the kills, the family and its m, the headline sentences.
  Compare each registered value with the code that reads it.
- **Execution.** Could the trades have been made? Fills against the book or
  prints at the time, queue position, slippage, latency of one bar, capacity
  against the clip, liquidation at the hour's high.

## Running the checks

The study's tests and its dry run, which touches no held-out day:

```bash
PYTHONPATH=src python -m pytest tests/ -q -k <study>
```

A study's data on disk, and a backtest's trades against the trial ledger:

```bash
trading-research validate-data --input <dir> --strict
```

```bash
trading-research review --trades <trades> --configurations <n> --cost-bp <bp>
```

Each study's entry point has `status` and `--dry-run`, which touch no held-out
day:

```bash
python -m experiments.spot_maker_2026 --dry-run
```

## Handing back

The `handback` block of `agents/protocol.md` for the skeptic: `lenses`
(leakage, fidelity, execution, each with `checks` and `refutation`) and
`findings`, each with its `refutation`. A finding without the check that would
refute it is an opinion and is not handed back.

## Never

Edit study code, re-run a read block, or propose a new threshold after a read.
A defect found after a read can only lower a status.
