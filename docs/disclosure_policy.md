# Disclosure policy

## What this repository is

An independent implementation, written from scratch, of a research pipeline for
short-horizon prediction on order-book data. It runs on generated data and on
public exchange archives.

## What it is not

It is not a cleaned-up copy of anything. The author has worked on related
problems commercially; none of that code, data, configuration or results
appears here, and no file in this repository was derived from a private one by
editing.

Specifically excluded, and never to be added:

- source, history or artefacts of any closed-source project
- private libraries, internal binaries, and their configuration schemas
- absolute paths, hostnames, account names, or internal URLs
- credentials of any kind
- trained model weights
- datasets without a confirmed right to publish
- production configurations and tuned parameter values
- the full feature set, selection recipe, or thresholds of any private system
- personal contact details

## What is published, and why

The engineering and the method, not the recipe.

| Published | Reason |
|---|---|
| Pipeline architecture, data contracts, validation | The structure is the transferable part; none of it is proprietary |
| Synthetic generator | Entirely original, and required for tests to run |
| Public-data downloader | Reads documented public endpoints |
| Standard microstructure features | Textbook definitions from the open literature, cited where they come from a specific paper |
| Purged and embargoed splits, leakage probes | Standard method, applied properly |
| Backtest with an explicit cost model | The accounting, not a strategy |
| Baselines and their comparison protocol | Method, not parameters |
| Aggregate demo results, plainly labelled | Demonstrates the pipeline runs end to end |
| Limitations and negative results | These are the useful part of research |

Where a feature or method comes from a published paper, the paper is cited. The
implementations here are written from the papers, not adapted from any private
codebase.

## Parameters

Parameters in this repository are illustrative defaults, chosen to make the demo
run quickly and legibly. They are not tuned values from any system, and they
should not be read as recommendations. Where a parameter matters to a result,
the result is reported as a function of it — a sensitivity curve rather than a
single number — which is both more honest and more useful.

## Claims

No result here establishes that any strategy is profitable. Backtests are
computed with hindsight under stated assumptions. The correct reading of any
number in this repository is: *under these assumptions, on this fixed historical
or generated sample, this pipeline produced this figure.*

Synthetic results are circular by construction: the generator injects a signal
and the pipeline recovers it. That demonstrates the pipeline works. It
demonstrates nothing about markets, and every synthetic artefact is labelled so
it cannot be quoted as though it did.

## Enforcement

The policy is checked, not merely stated. [`scripts/audit_repo.py`](../scripts/audit_repo.py)
runs in CI on every push and as a pre-commit hook, and fails the build on:

- absolute paths identifying a machine, user or mount point
- credential-shaped strings and private-key blocks
- references to private libraries or internal tooling
- notebook outputs and execution counts
- files with binary or dataset extensions
- files above one megabyte
- missing ignore rules for generated directories

The audit is a floor, not a ceiling. It catches mechanical mistakes; it cannot
judge whether an idea should be published. That judgement is made per addition,
before the code is written, against the table above.

## Reporting a problem

If something here should not be public, open an issue without quoting the
material, and it will be removed.
