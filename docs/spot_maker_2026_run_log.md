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
