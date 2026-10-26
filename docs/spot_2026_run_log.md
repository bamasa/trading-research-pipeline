# Run log of the 2026 spot study

One entry per day while the registered stages run (times UTC unless marked
MSK). The stage records live under the run's work directory and are summarised
in the D amendment; this log records what happened and when.

## 2026-10-25

- The owner's amendment of 25 October: the information-set dates (H blind) and
  the decision to run this study before the 2025 study's held-out reads.
- The study's code built in four branches, merged, reviewed adversarially in
  three lenses and corrected; every choice written into the registration
  before any 2026 file of W, D, H or P was read.

## 2026-10-26

- Stage 0 on the reader-test days 2026-03-03 to 05 in the study's own
  environment: every gate passed (readers' proof, parity, a median 200 ms
  between book messages, walk depth 10, instrument steps, print-rule records).
- Code freeze at `e8f2f5a`, committed in `a1317a5`.
- Stage 1, 01:51 to 03:41: 122 days of each plane fetched with no failure;
  max_sequence_gaps 0; the leakage checks passed on the three stores,
  touch proxies included; W's touch size 0.606 BTC; stores for D's 92 days.
- Stage D launched with the 2025 study's checkout as the source of its frozen
  files.
- Stage D, 03:45 to 05:37: the June, July and frozen folds fitted with the
  reads_until_ns purge; the full fold-level leakage test through the
  production fit path passed on every fold and fit; sigma_D, the MDEs, Null A
  and Null B and the thinned calibration written.
- The D amendment written by `amend` (frozen configuration `d1625173…`, 14
  frozen fits, the July rehearsal run and discarded) and committed after
  review.
- H opened at 06:10 (local commit `57458b1`); H's 75 days fetched on each
  plane and the fetch records committed (`bc5ed75`); the computing read ran
  and closed. Every slot killed by K1: S25-PF −17.35 bp, S25-BB −19.43 bp,
  S25-T −19.28 bp a trade at the Regular / non-VIP tier. P is not read: no
  slot was a candidate on H.
