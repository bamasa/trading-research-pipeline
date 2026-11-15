# Journal

This branch holds only records written before the period they are about, so
each commit's timestamp is the proof of when it was written. The code that
writes them, and their registrations, are on `main`.

- `carry/`: the carry search's choice on the future. At the start of each
  month, `trading-research search carry-live` writes the coin the rule
  `top1-lb3-floor_none` holds that month, from the previous three months'
  funding alone (`YYYY-MM.pick.json`); after the month it adds the score
  (`YYYY-MM.score.json`). The workflow `.github/workflows/carry-journal.yml`
  runs it on the first days of each month and commits here. Registered in
  `docs/preregistration/carry_live_2026_11.md`.

Entries are never rewritten.
