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
