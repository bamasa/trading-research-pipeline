# Code freeze for the Bybit 2021 market-making study

Date: 2026-10-27.

Code-freeze commit: `e1f996159b49d70e68652d621950bdf32643d0ff`

Every piece of study code the registration names — the entry point
`experiments/mm_bybit_2021.py` and the package under
`src/trading_research/mm_bybit/`, with the modules it reuses by import, unchanged
— is at that commit, together with `configs/mm_bybit_2021.yaml` (null-placeholder
sha256 `17fca5f80713ae1618f2acc598370833af2ccac275a58b4ad0d3d313f26289e3`, set by
the stage-0 amendment of the same day), `pyproject.toml` and `uv.lock`. No file
of W beyond the two reader-test days, of D or of H has been fetched; E is never
fetched. Any change to that code before H is read is a correction under the
registration's amendment protocol, and the ledger refuses to open H on code that
differs from it.

Before the freeze:

- the fee table was re-read on five Wayback captures of Bybit's help article
  360039261154 (2020-04-10 to 2022-03-11): BTC/USDT at −0.025% maker and
  0.075% taker for every account throughout, the change announced for
  2022-03-11; nothing contradicts the registered table;
- the tests (`tests/test_mm_bybit_*.py`, synthetic and hand-built days, no
  network) passed, the dry run among them: every stage, the amendment and H's
  read through the ledger on the synthetic venue, at this commit;
- stage 0 ran twice on the reader-test days 2020-12-15 and 16. The first run,
  at `78ef6c0`, passed the file checks and excluded both days as `print_gap` under
  the registered 120 s rule, which the gaps' smooth tail showed to be a quiet
  market, not a hole in the archive; the stage-0 amendment raised the
  threshold to 600 s and the code now reads it from the configuration. The
  second run, at this commit, passed every check:
  - both files newest first, reversed by the reader; stamps with four decimals,
    in order once reversed; no duplicate match id; no row outside its day;
    40,829 and 65,420 prints with both sides present;
  - tick 0.5 USDT and lot 0.001 BTC on both days;
  - three funding settlements a day, at 00:00, 08:00 and 16:00;
  - the longest gap between prints from 00:10 to 23:59: 158.7 s and 140.2 s;
  - S0 and `l1|vle0.67` took 0.08 to 0.16 s a cell-day; the accounting
    identity held and |position| never passed the soft limit.
- the disk had 28 GB free against the registered budget of at most 20 GB of
  the study's files and an expected 3 GB.
