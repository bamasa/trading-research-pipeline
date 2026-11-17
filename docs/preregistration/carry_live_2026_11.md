# Registration: the carry search's choice on the future, a month at a time (L1, L2)

| | |
|---|---|
| **Committed** | 15 November 2026, with the journal's code, after November 2026's pick was committed to the `journal` branch (15 November 2026, 15:48 UTC) and before any month it is scored on had ended |
| **What it registers** | A journal that writes the carry search's chosen rule's coin down before each month and scores it after, and the two statuses read once from its first twelve full months |
| **Searched** | **Nothing.** The rule is the one `trp search carry` chose on January 2021 to June 2023 and read once on July 2023 to June 2025 ([registration](carry_search_2026_11.md)); no setting is chosen here |
| **How it changes** | Only by a dated amendment appended at the end, which cannot change a month already picked or a status already read |

## Why

The carry search's two blocks are both the past. The held-out block was never
chosen on, but it was there to be looked at, and the neighbouring walk-forward
replay had seen it. A month that has not happened yet cannot have been looked
at. Writing the coin down before the month, in a public commit, is the only
way this repository can show a record nobody could have fitted, without
trading money. The idea is borrowed from forward-test journals that commit
their forecasts before the outcome (for example `Navid-A-08/Forward-test` on
GitHub).

[`experiments/results/carry_overfitting/`](../../experiments/results/carry_overfitting/README.md)
measured, before this was written, that the carry survives the size of the
search but the choice among its sixteen rules does not (PBO 0.69 on D; on H
the chosen rule ranks 14 of 16 while all sixteen are positive). The journal
therefore asks the two questions apart.

## The journal

[`live/carry_journal.py`](../../src/trading_research/live/carry_journal.py),
run by `trading-research search carry-live` and by the workflow
[`carry-journal.yml`](../../.github/workflows/carry-journal.yml) on the first
five days of every month, three times a day; it commits to the repository's
`journal` branch, whose commits are the timestamps.

- **The rule.** `top1-lb3-floor_none`, unchanged: hold the one coin of the
  search's seventeen paying the most funding a day over the previous three
  calendar months, long spot and short the USD-M perpetual in equal quantity,
  the whole account split between the two legs; Binance Regular fees and 2 bp
  of slippage a leg and side. A coin kept pays no round trip.
- **When.** Binance publishes a month's funding a few hours into the next
  month; the pick is made by the first run after that, and the coin enters at
  the next full hour. Until then the previous month's coin is held. In the
  search the rule entered at the month's first hour; the lag is the one
  difference, and it is recorded (`made_at`, `enters_at`).
- **The score**, after the month's archives are published: the account's
  return after costs, and the carry of one unit of notional, gross, from the
  entry hour to the month's end, for the coin held and for each of the
  seventeen coins over the same hours.
- **Written once.** Neither a pick nor a score is rewritten; the code refuses.
  Each pick carries the sha256 of the journal's and the engine's sources.
- **November 2026** was entered mid-month (15 November, 16:00 UTC, on DOGEUSDT)
  when the journal began. It is reported and is **not** one of the twelve.

## L1 and L2

Read once, from the scores of **December 2026 to November 2027**, after
November 2027's score is committed.

- **L1, the carry pays on the future.** **Passes** when the twelve months'
  compounded return is positive and the probabilistic Sharpe ratio of the
  twelve monthly returns above zero (with their skewness and kurtosis,
  `deflated_sharpe_ratio` at one trial) is at least 0.95. Otherwise **killed**.
- **L2, choosing by funding adds something.** **Passes** when the coin held
  out-earns the average of the seventeen coins' carry over the same hours in
  at least 10 of the 12 months (one-sided sign test, *p* = 0.019). Otherwise
  **killed**.

Twelve months, because the held-out block's monthly returns, taken as normal,
need nine to be told from zero at 95% (the minimum track record length in
`carry_overfitting`), and with their skewness only three; twelve covers both.
After six months (May 2027's score) the journal's numbers are reported, with
no status.

## Expected outcome, stated now

**L1 passes; L2 is killed.** The carry has paid in every block this
repository has measured, but the funding the rule reads is lower now than in
either block (DOGEUSDT, the November pick, paid 1.78 bp a day over August to
October 2026; the coin the rule chose read a median of 2.9 bp a day on the
development months and 2.7 on the held-out ones), so the return will be
small: a few tenths of a percent a month. The choice among
coins was measured as a coin toss on the past, so the coin held should beat
the average about half the time, not ten months in twelve.
