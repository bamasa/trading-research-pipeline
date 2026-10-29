# Run log of the 2025 short-horizon study

One entry per day while the registered stages run. Times are UTC unless
marked MSK. The stage records themselves live under the run's work directory
and are summarised in the amendment; this log records only what happened and
when, so that the gaps between stages can be read later without guessing.

## 2026-10-12

- Stage 0 ran at `9784ef7` on the two 2024 timing days: parity at every hold,
  75.7 h projected against the 144 h ceiling, no cut; replay peak 5.4 GB,
  4 replay workers.
- Code freeze committed (`4bbaba5`), then stage 1 launched at 19:53.

## 2026-10-13

- Stage 1: the 212 book days (W and D) were fetched and replayed by 09:30 MSK
  (about 3 h at 10 MB/s, 4 workers); prints, hourly bars and funding followed;
  the feature stores and targets were written by 17:00 MSK (about a minute a
  day, one process); selection on January chose 40 inputs.
- The fold-level leakage test began at about 18:00 MSK: every fold fitted
  twice on the production path, one fit at a time.

## 2026-10-14

- The fold-level test finished its five folds without a leak; the
  flag-causality test ran on 8 workers (672 to 744 hours per fold, no leak).
- Stage 1 complete at 15:45; stage A launched at 15:45 by the hand-off script.
  February's fold took about 1.5 h (flags, four network fits, 800 tapes).
- The machine entered low-power sleep at 23:25 MSK on battery.

## 2026-10-15

- Woke at 10:14 MSK on the user's return. The fits resumed; the first fit after
  the wake ran slowly while the feature store came back from disk.
- March and April completed; May began in the afternoon. The machine went to
  battery at about 16:30 MSK and dozed in maintenance sleep from about
  17:00 MSK, so June's second network fit made no progress overnight.

## 2026-10-16

- Mains power back at 11:00 MSK; June finished; stage A complete at 09:17
  (200 fits over five folds).
- Stage B complete at 09:28: 2,400 cells over 150 test days, 360,000 rows, no
  stale day; the trial ledger's manifest committed (`e3c8d59`).
- Stage D launched at 09:28; Null B done, the refit running at 13:48 MSK.
- Low-power sleep from 13:07 to 13:48 MSK (battery at 0%); stage D resumed on
  mains after the wake.
- Stage D's refit reached its third fold by 15:49 MSK; the machine went to
  battery again in the late afternoon and dozed from about 16:00 MSK, so the
  refit made no progress overnight.

## 2026-10-17

- 02:16 MSK: still on battery (66%), stage D paused on the refit's third fold,
  no error in any stage log; the chain to C and E is waiting on D.
- Mains power returned in the morning; the refit finished its five folds and
  stage D completed at 10:54 (13:54 MSK): p_search 0.007 against the placebo
  search, C* chosen at hold 120 s with a carry-forward G of +2.1 bp and a net
  of −8.9 bp per trade, no admissible R cell; the leaderboard, marginals and
  nulls files written. Stage C launched at 10:54.
- Low-power sleep from 14:09 MSK on 2026-10-17 to 16:43 MSK on 2026-10-18
  (battery), so stage C ran for about a quarter of an hour before the sleep.

## 2026-10-18

- Woke at 16:43 MSK on mains; stage C resumed (the passive slots on D).
- Stage C (the passive slots on D) completed at 15:02 (18:02 MSK): S-P and
  S-M not admitted on D (conditions i and ii); the market maker's making part
  was −0.43 bp per fill; 35 simulator trials added to the ledger
  (N_trials 2,435). Stage E launched at 15:02.

## 2026-10-19

- Stage E completed at 01:25 (04:25 MSK): the D amendment written into the
  registration with the frozen configuration (sha256 `e9cb5e03…`), which
  resets to the registered null-placeholder hash; the finalist file sets of
  S25, S25-Z, S-C, S-P and S-M hashed; the ob200 reader test passed on
  2026-01-05 to 07 with no sequence gap; the pages and figures written. The
  development period is complete.
- Low-power sleep 12:35–13:52 MSK, idle.
- Committed the amendment and the results (19:40 MSK). Two registration test
  modules assumed the unfrozen file; they now read the registered text in
  either state (tests only, no study code changed).
- The full suite after the commit found the dry run refusing to start: it
  copied the now-frozen configuration. Corrected in `dry_run.py` (it copies the
  registration as registered) and recorded as a dated amendment before H is
  read; no development number changes, D is not re-run. The dry-run tests pass
  again (18 of 18).

## 2026-10-20

- Nothing run. The development period is complete and committed; H is not
  opened yet. The registration asks that H's open entry be committed and
  pushed before the fetch, and the owner keeps intermediate work off the public
  repository, so the way the open entry is recorded is waiting on the owner's
  decision. No 2025 file of H, F or P is on disk.

## 2026-10-22 to 2026-10-24

- A private repository was created to hold H's open entry with a server-side
  timestamp; adding it as a remote and pushing is waiting on the owner, so H is
  not opened yet. No 2025 file of H, F or P is on disk.
- A spot study was registered as a draft on the separate local branch
  `spot/prereg` (Binance BTC/FDUSD prints and Bybit spot BTCUSDT book, blind
  blocks in 2026); it touches none of this study's frozen code and fetches
  nothing before this study's last read.

## 2026-10-25

- Nothing run; H still waits on the private remote for its open entry.

## 2026-10-26

- Nothing of this study was run. The owner chose to run the 2026 spot study
  first, on its own local branch and environment; it read only 2026 days and
  wrote nothing of this study. H still waits on the owner's decision on how
  its open entry is recorded. No 2025 file of H, F or P is on disk.

## 2026-10-27

- Nothing of this study was run. Two exploratory replications of the author's
  2025 results run on separate local branches (Binance BTC/FDUSD prints;
  Bybit ETHUSDT books); neither touches this study. H still waits on the
  owner's decision on how its open entry is recorded.

## 2026-10-28

- Nothing of this study was run on 2025 data; no file of H, F or P is on disk.
- Two reviews of the code after the dry-run correction found 21 defects; each
  is corrected and dated in the registration's amendments (2026-10-28), with a
  regression test where a scratch test had shown the fault. D is not re-run and
  no D number changes: the recomputation of every D number from the committed
  results files and stage records gives the same output before and after. The
  frozen configuration is unchanged (`e9cb5e03…`).
- The two exploratory replications of 2026-10-26 and 27 computed return
  statistics on days of D's test months on other instruments; read
  conservatively, they are hand runs, logged in the trial ledger's manifest,
  and S-C, S-R, S-P and S-M are `not admitted on D (hand run)` (each was at
  p = 1 already).
- The pages were written again: N 2,435 in the search sentence, the passive
  sentence with the hand-run status, notes beside the payoff and passive
  sentences, and the fee caveat beside the ladder's tier rows; no figure
  changed.
- The study's test modules pass (188), and the dry run passes (18 of 18) after
  two of its runs found that stage E's rehearsal and the gap days put back did
  not yet carry the statuses set on D; both corrected. The code is frozen again
  at `98e85b9` (the code-freeze note).
