---
name: pre-freeze-review
description: Hunt for defects in a study's code before its code freeze, against a checklist of the defects past reviews in this repository actually found. Use when a study's code is written and about to be frozen, when asked to review study code for bugs, look-ahead or accounting errors, or before a held-out read.
---

# Reviewing study code before the freeze

After the freeze a defect can only lower a status, and a corrected run is a
successor registration; before it, a defect costs a dated amendment. So the
review that matters is the one before. In the 2025 study, two reviews of the
code after its dry run found 21 defects, each corrected and dated in the
registration's amendments of 2026-10-28 (`docs/preregistration/short_horizon_2025.md`).
The checklist below is what they found. `agents/roles/skeptic.md` is the role;
`skeptic-review` holds the three lenses.

## Setup

Read the registration and its YAML, then the code it names under "The code to
come". Run the study's tests and its dry run first; a review of code whose tests
fail is a review of the wrong code:

```bash
PYTHONPATH=src python -m pytest tests/ -q -k <study>
```

## The checklist, from defects found here

- **Time.** A day's first second and the previous day's last; a one-day shift
  after a stale or missing day; a gap count that crosses midnight; UTC against
  local time; a window that includes the decision's own bar.
- **What a block may see.** An access object usable after its block closed or
  for another block; a cache written during one read and read by another; a
  value set on D recomputed on H instead of read from the frozen amendment.
- **The ledger.** An open entry that can be erased or rewritten; a push check
  that accepts a local branch as upstream; a results file not hashed into the
  ledger; a session that can start twice.
- **Accounting.** A fee from the wrong period or tier; a take-profit booked at
  the overshoot instead of its level; a rebate counted on the taker side;
  units through a normalised column; one table of fills used where two
  should be.
- **Statuses.** A preset slot treated as a hypothesis instead of a measurement;
  a fault outside the status window that still changes a status; the Holm
  family's m counted from slots that ran instead of slots registered.
- **Pages.** A headline sentence whose bracket is filled from a different
  number than the status used; N counted differently on two pages.

## Method

For each item, find the code that could commit it, write the smallest input
that would expose it, and run it as a scratch test. A finding is reported only
with that input and what the code returned; a suspicion without one is listed
separately as unverified. Rank by whether it could change a status.

## Hand back

Each finding with `file:line`, the input that exposes it, the effect on a
status, and the dated amendment it needs. The owner corrects, the tests are
extended with the scratch test, and only then is the code frozen.
