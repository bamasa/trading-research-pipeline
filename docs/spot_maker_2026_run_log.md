# Run log of the 2026 spot maker study

One entry per day while the registered stages run (times UTC unless marked
MSK). The stage records live under `artifacts/spot_maker_2026/` and are
summarised in the D amendment; this log records what happened and when.

## 2026-10-27

- The study's remaining code reviewed against the registration and finished:
  the skew made to read the position known at the decision; stage 0's checks of
  the fill rule on every resting order of every run; the measurements beside
  the verdict; stage 1's signal; free disk checked at every stage and read; the
  P gate and its fetch; the page. Two dated amendments before the code freeze:
  the code's choices where the text left one open, and the fill rule's
  calibration against a book (Bybit linear BTCUSDT, 2024-02-01 to 25, read in
  place), which made the configuration's null-placeholder sha256 `aa367b64…`.
- The fee table checked against the spot study's sourced table
  (`configs/spot_2026.yaml`, sources B3 to B8): the Binance BTC/FDUSD intervals
  and tiers are equal, and the maker fee is 0 at every tier on both intervals,
  2024-04-25 06:00 to 2026-10-22 00:00; the taker fee at Regular is 10 bp.
- The dry run on stand-ins passed end to end (stages 0, 1, D, the amendment,
  H's read through the ledger); its numbers are discarded.

## 2026-10-28

- Code freeze at `4b5d7d3` (note committed in `a81565b`).
- Stage 0 on the reader-test days 2026-03-03 to 05 (21:1x UTC on 10-27):
  every day usable, the print contract and the tick 0.01 checked, both sides
  present; 31,830, 38,701 and 29,101 resting orders of every slot's runs
  re-filled under the three rules with no violation, probe orders nested, no
  void condition. Timings a day: load 0.1 s, features 1.0 s, signal 0.05 s,
  every run of every slot 21.8 s, checks 1.8 s. Every other number discarded.
- The registration corrected before stage 1 (the amendment of 2026-10-28):
  the fill rule's calibration moved to Bybit spot BTCUSDT's book on D's days,
  with MK-S added; null-placeholder sha256 `147aea71…`. One calibration day
  was run for its timing (32 s, 1.6 GB) and its numbers were not read. Code
  freeze moved to `bf044ea` (note in `9d66d81`).
- Stage 1, 21:20 UTC: W and D copied from the spot study's checkout, 122
  files, 111.4 MB, none absent, every sha256 equal to its source's; no day of
  W or D excluded; the signal computed on D's 92 days; 28.6 GB free.
- Stage D, 21:23 to about 21:55 UTC, four workers: every run of every slot on
  D's 92 days (none excluded) and the fill rule's calibration on Bybit spot
  BTCUSDT's book on the same days. Development numbers, in-sample for the
  signal, never results: sigma_D 1.24 (MK-S), 0.38 (MK-Q), 0.36 (MK-SQ) bp
  per fill; MDEs 0.55, 0.17, 0.16 bp per fill; every D day had fills for every
  slot; the July rehearsal killed all three. The calibration: the maker
  volume ordered trade-through <= queue <= touch on every counted day for all
  three slots, but the prints rule's making part lay inside the book's
  bracket on 22% (MK-S, 89 days), 64% (MK-Q) and 58% (MK-SQ) of the days,
  under the registered 80%: it does not pass, and every verdict carries the
  label "uncalibrated fill rule".
- The D amendment written by `amend` and committed after review (`266babb`):
  frozen configuration `8dbd3181…`, the run log's manifest (3 rows).
