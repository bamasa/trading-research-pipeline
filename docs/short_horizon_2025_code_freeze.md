# Code freeze for the 2025 short-horizon study

Date: 2026-10-12.

Code-freeze commit: `9784ef7b9e1d7cdaa148a64a05b439ea87a77c7a`

Every piece of study code named by the registration — the entry point
`experiments/short_horizon_2025.py`, the package under
`src/trading_research/pipeline/short_horizon/`, the search engine, the
features, labels, models, the taker executor, the signal-entry quoter and the
simulator, the nulls and the statistics — is at that commit. No W or D file
has been fetched. Any change to that code before H is read is a correction
under the registration's amendment protocol.

Stage 0 ran at the same commit on the two 2024 timing days (2024-02-11 and
2024-03-05), with the record kept under the run's work directory. What it
established:

- the walk-table parity test passed at every hold (5, 20, 50 and 120 s) on
  both days, with no mismatch against the event-time executor;
- the development period projects at 75.7 hours against the 144-hour
  ceiling, so the catalogue is not cut (`cuts: []`); the largest terms are the
  fits (31.3 h), the fold-level leakage test's CPU fits (23.9 h) and the
  freeze (15.4 h);
- the replay of one day's 500-level archive peaks at 5.4 GB, which allows
  4 replay workers; the network's inference takes 13.4 microseconds a row on
  a full day;
- the catalogue hashes to the registered value and the configuration to the
  registered null-placeholder sha256.

## Corrections before H is read

The commit above is the original freeze, at which stage 0 ran and stages 1, A,
B, D, C and E computed every development number. Every later change to the
study code is a correction under the amendment protocol, dated in the
registration's amendments:

- 2026-10-19: the dry run copies the registration as registered
  (`7a80191`, `dry_run.py` and two tests).
- 2026-10-28: the corrections the two reviews of the code after the dry-run
  correction found (`7ca6b90` to `98e85b9`): the passive shift after a stale
  day, one FILL table, N on the pages, the replay's gap count and its day cut,
  the ledger's history and push checks, an access bound to its block, the
  statuses set on D from the frozen configuration, faults by day, the read
  caches, D's results files hashed, a slot preset on D read as a measurement,
  the fee caveat by period, the deflated Sharpe ratios' inputs, the passive MDEs
  on kept days, the pages' notes and the hand-run log. D is not re-run and no
  development number changes; the frozen configuration is unchanged
  (`e9cb5e03…`).

The study code is frozen again after them, before H is read; the open checks
the code at this commit through the session's commit and clean tree:

Code-freeze commit: `98e85b92fd0b1a159a8b907983b57f64ace13ae9`
