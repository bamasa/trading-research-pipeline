# Run log of the Bybit 2021 market-making study

One entry per day while the registered stages run (times UTC). The study is
registered in [`preregistration/mm_bybit_2021.md`](preregistration/mm_bybit_2021.md)
with [`configs/mm_bybit_2021.yaml`](../configs/mm_bybit_2021.yaml); the stage
records live under `experiments/results/mm_bybit_2021_*.json` and the data under
the ignored `data/mm_bybit_2021/`. This log records what happened and when.
Nothing is pushed.

## 2026-10-27

- The registration committed at `27383df` (null-placeholder sha256
  `0ece0d7d66099b86f8eac46d7e73abdfe2b1219585a2da590bf821b54da36558`), with
  the HEAD check of every print file. Its text is unchanged since.
- The study code (`src/trading_research/mm_bybit/`, `experiments/mm_bybit_2021.py`)
  reviewed against the registration and completed before any file of the study
  was fetched. What the review changed in the first draft:
  - a day whose file answered 404, or did not parse, is recorded on disk with its
    reason, so it is excluded as `no_prints_file` or `unreadable` and a day never
    fetched stops the run instead of reading as a 404;
  - every run's settings are built from the fee table alone
    (`days.settings_for`), so no run can be given another fee;
  - the quotes are pulled apart (R4) only when both sides are quoted, as in the
    simulator; no decision is taken from 23:58;
  - the turnover check (K-acct) covers the taker fill too;
  - added the flatten's cost, the share of quoting time at the soft limit and
    each gate's open share, passive turnover a day, the median day;
  - the D record holds every cell's daily series, F's and S0's decomposition,
    the markouts and `y` by month; the touch bracket and the 10 ms run are
    measured on D for every cell, beside the primary rule that decides;
  - the instrument's tick and lot are taken over W and D, not D alone;
  - the amendment's compute record carries the run log's sha256, as registered;
  - Holm runs over the family {F, S0} with m = 2 even when F is not run; a
    finalist not admitted on D carries that label as its status, with the
    precedence applied beside it;
  - H unreadable still writes its record; a second read writes its own file;
    the registered table is written beside the H record; the headline
    sentences carry units;
  - the ledger also checks that the configuration and the document are tracked
    and that the committed configuration is the frozen one; H's access is
    issued by the ledger alone;
  - stage 0 loads the funding of the two reader-test days only.
- The fee table re-read, as the registration requires before the code freeze.
  The Wayback Machine answered today for all five captures of Bybit's help
  article 360039261154 (captures 20200410083714, reached from 20200410051922;
  20201221232354; 20210827182431; 20220111082854; 20220311223741). Each shows
  BTC/USDT at a maker rebate of −0.025% and a taker fee of 0.075%, with no tier
  or account condition; the last carries the note that the new fee structure
  applies to some derivatives pairs from 2022-03-11 06:30 UTC and to all from
  2022-03-18 07:00 UTC. Nothing contradicts the registered table; no amendment
  is needed.
- Code and tests committed (`39504d4`, `78ef6c0`); the dry run on the
  synthetic venue passed end to end (stages 0, 1, D, the amendment, H through
  the ledger).
- HEAD re-check of the two reader-test days, 16:44: both 200, 4.24 MB together,
  the sizes committed with the registration. Disk: 28 GB free.
- Stage 0, first run, 16:44 at `78ef6c0`: the file checks passed (files
  newest first and reversed, four-decimal stamps, no duplicate id, no row
  outside the day, both sides present), and both days were excluded as
  `print_gap`: the longest gaps from 00:10 to
  23:59 were 158.7 s and 140.2 s, with a smooth tail (none over 180 s) and the
  price unchanged or within 3.0 USDT across each, a quiet market. No timing was
  taken, since neither day was usable.
- The stage-0 amendment (`b7b1f21`): the print-gap threshold from 120 s to
  600 s, new null-placeholder sha256 `17fca5f8…`; the code reads the threshold
  from the configuration (`e1f9961`).
- Stage 0, second run, 16:47 at `e1f9961`: both days usable, every check
  passed; 0.08 to 0.16 s a cell-day for S0 and `l1|vle0.67`. The dry run
  passed again at the same commit.
- Code freeze at `e1f9961` ([note](mm_bybit_2021_code_freeze.md)).
