---
name: record-result
description: Record a study's result after a read or a kill, in the run log, the findings register and the README, using only the sentences the registration fixed. Use when asked to write up a result, add a findings row, update the run log, or say what a study found.
---

# Recording a result

The role is `agents/roles/scribe.md`. A result is recorded where the
repository keeps them, in its registered words, whatever its sign: a killed
hypothesis gets the same care as a candidate.

## Where

- `docs/<study>_run_log.md`: a dated entry appended at the end. What ran, on
  which block, the files written with their sha256, and anything that went
  wrong. Earlier entries are never edited.
- `docs/findings.md`: the row under its status (conditional, killed,
  artefact, pre-registered), with the kill that fired and its value, and a
  link to the page.
- `README.md`: the status and roadmap rows.

## How

Start from what the entry point says was read and written:

```bash
python -m experiments.spot_2026 status
```

- Copy each headline sentence verbatim from the page the entry point wrote.
  Do not paraphrase, round, or add an adjective.
- A number at any tier but the verdict tier is a measurement: say so. A
  development number is never quoted as a result.
- Name what the study cannot establish, from its registration.

## Never

Rewrite a results file, a ledger, a page or an earlier run log entry; describe
a measurement as the strategy working; round a number up.

## Then

The owner commits. The commit message names the study, the stage, the reading
and the verdict, with no trailer line (`CLAUDE.md`, rule 6). Run the
`check-before-commit` skill first.
