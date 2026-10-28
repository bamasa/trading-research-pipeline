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
- Stage 1, 01:49:56 to 03:41:19: 122 days of each plane fetched with no
  failure; max_sequence_gaps 0; the leakage checks passed on the three
  stores, touch proxies included; W's touch size 0.606 BTC; stores for D's
  92 days.
- Stage D launched with the 2025 study's checkout as the source of its frozen
  files.
- Stage D, 03:42:26 to 05:37:16: the June, July and frozen folds fitted with
  the reads_until_ns purge; the full fold-level leakage test through the
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

## 2026-10-27

- No stage or read of this study was run (only the `pages` command below);
  H is read and P is not read, as recorded above. Two exploratory
  replications of the author's 2025 results ran on separate local branches
  (Binance BTC/FDUSD prints; Bybit ETHUSDT books, whose frozen test on
  2026-08-08 to 21 collapsed on its frozen label threshold and lost on every
  day at 2.2 bp). A maker study of this study's frozen signal
  S25-PF on Binance BTC/FDUSD at the fees charged, and a prints-only maker
  study on Bybit BTCUSDT at the 2021 rebate, are registered on separate local
  branches; neither writes into this study.
- A review after H recomputed every number of H's results file and page from
  the day records and found them exact; the kills fire as registered and every
  fill pays 10 bp a side. Its findings are acted on below; no number changed.
- The post-H presentation amendment (the registration, Amendments,
  2026-10-27): `spot/pages.py` now labels every tier but the verdict tier with
  the status it requires, the zero-fee results "conditional on a zero-fee
  venue", names without a value the Binance tiers no sourced interval lists on
  a day of H (Spot Liquidity Provider, VIP 1, VIP 2–9), adds each row's 30-day
  traded volume beside its tiers and puts the registered caveat on zero taker
  fees at the head of the page. The page was rewritten by `pages`;
  `experiments/results/spot_2026_H.json` is byte-identical before and after
  (sha256 `ed5301ed…`), the page's sha256 went from `0523cd14…` to
  `eafdcd75…`, and only the page changed among the results files. The code
  freeze no longer holds at HEAD, so the ledger would refuse to open P; P is
  not read, since no slot is a candidate.
- The spot tests' registration fixtures are built from the registered text
  (the configuration with its placeholders reset, the document cut before the
  D amendment), as the 2025 study's are: `-k spot -m "not slow"` went from 39
  failed and 344 passed to 384 passed, one test of the page's labels added.
- Stage 1's and D's times of 26 October are corrected above to the run's
  `state.json` (they read 01:51 and 03:45).
- Two limitations of the frozen code, found by the same review, are not fixed
  here (study code after H) and are carried into the next study's
  registration:
  - The calibration's two executor runs in `spot/heldout.py` (S25-PB by the
    book walk and S25-BB by the print rule) run outside the void machinery: a
    failure there would have closed H as not assessed instead of voiding one
    measurement, and K-cap and K-inv are not replayed on their fills. H ran
    with no failure there, so no number is affected.
  - The fold-level leakage test mutates prices and stored inputs
    multiplicatively but not book or print sizes, and on the frozen fold it
    is vacuous by construction: its cutoff, 2026-08-01, has nothing after it
    on disk, so there it proves deterministic refitting only. Its results on
    the June and July folds stand.
