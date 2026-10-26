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
