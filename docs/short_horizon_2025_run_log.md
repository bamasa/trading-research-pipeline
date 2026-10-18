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
