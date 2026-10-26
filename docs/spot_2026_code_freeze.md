# Code freeze for the 2026 spot study

Date: 2026-10-26.

Code-freeze commit: `e8f2f5a2d05955952301fb9e6b95aa9b63c5e0d2`

Every piece of study code the registration names — the entry point
`experiments/spot_2026.py` and the package under `src/trading_research/spot/`,
with the modules it reuses by import — is at that commit, together with
`configs/spot_2026.yaml`, `pyproject.toml` and `uv.lock`. No file of W, D, H or
P has been fetched. Any change to that code before H is read is a correction
under the registration's amendment protocol, and the ledger refuses to open a
block on code that differs from it.

Stage 0 ran at the same commit on the three reader-test days (2026-03-03 to
05), in the study's own environment, and passed every gate:

- the readers' proof: every day usable on both venues; no sequence gap in
  Bybit's spot book; the book's messages a median 200 ms apart (the format
  probe's finding confirmed), up to 5.6 s at most;
- the parity of the print walk tables with the print executor: no mismatch at
  any hold on any day, with decisions whose busy period depends on the side;
- the walk depth: a 0.010 BTC clip went beyond the ten visible levels on
  0.086% of the book's rows, under the 0.1% threshold, so the walk stays at
  ten levels;
- the instruments: Binance BTC/FDUSD tick 0.01 and lot 0.00001, Bybit spot
  BTCUSDT tick 0.1 and lot 0.000001;
- the print rule: on Binance BTC/FDUSD the median same-side sweep within the
  window covered 30 to 39% of the clip, so about three quarters of print fills
  have a part beyond the sweep and are charged at the sweep's worst price;
  on Bybit the median sweep covered about the whole clip;
- the disk: 49.7 GB free against the stage's 19 GB budget and 10 GB margin.
