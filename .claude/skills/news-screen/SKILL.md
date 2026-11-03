---
name: news-screen
description: Screen the instruments of a prospective block or a paper run against current news (exchange incidents, delistings, token unlocks, regulatory actions), dated and sourced, to propose exclusions before the block starts. Use when choosing instruments for a forward test or a paper run, or when asked whether news should change which instruments are traded.
---

# Screening instruments by news, forward only

A language model reads news well: it can tell a delisting notice from a rumour
and an unlock schedule from an opinion. In a backtest that skill is a leak.
Its training data reaches into the periods this repository holds out, and news
read about a past date is information from after the decision. So this skill
runs only **forward**: for a block that has not started (a prospective block,
P) or a paper run, and only on news published before the screen's own date.
`CLAUDE.md`, rules 3 and 4, bind it.

## When it may run

1. The block is prospective: its first day is after today. Check it against the
   calendar:

   ```bash
   PYTHONPATH=src python -m agents.workflows.research_loop calendar
   ```

2. The study's registration already names the screen as a rule, with the
   categories below and the date it runs. A screen not registered before the
   block starts is a choice made with knowledge, and is not run.

## What it reads, and what it never reads

- **Reads**: exchange announcements (listings, delistings, maintenance,
  contract changes), project announcements of token unlocks and migrations,
  regulators' published actions. Each with its URL and publication time.
- **Never**: prices, funding rates, volumes or returns of any date, from any
  page (`CLAUDE.md`, "Never"); commentary about where a price will go; anything
  published after the screen's date.

## The categories, decided in advance

| Category | Proposed action |
|---|---|
| delisting or contract change announced inside the block | exclude |
| exchange incident affecting deposits or withdrawals | exclude |
| token unlock above 2% of supply inside the block | exclude |
| regulatory action naming the asset or the venue | flag for the owner |

## Hand back

A table: instrument, category, source URL, publication time, the action, and
the time the screen ran. The owner commits it as a dated amendment before the
block's first day. The screen never adds an instrument, only removes or flags,
so it cannot select for a good outcome.
