---
name: held-out-read
description: Open and read a study's held-out block once, through its ledger, after the owner's grant. Use when asked to run a study's H, F or P step, open a held-out block, read a verdict, or check whether a block may be read yet.
---

# Reading a held-out block

A held-out block is read once. The order is fixed by each study's entry point
(its stages refuse to run out of order) and by its ledger
(`experiments/results/<study>_ledger.json`). `CLAUDE.md`, rules 3 and 4, bind
every step here.

## Before

1. The owner granted this block, for this run, in so many words. Without it,
   stop and ask.
2. The registration is committed and its D amendment is committed: every
   `null` placeholder is filled and the frozen sha256 line is in the document.
3. The code freeze note `docs/<study>_code_freeze.md` is committed, and the
   tree under the study's code paths equals the frozen commit. A study whose
   code `main` has since changed is read from its own branch.
4. The environment is the one the block will be opened with: no package
   installed since. The ledger fingerprints it.
5. The study's status says the block is next:

   ```bash
   python -m experiments.spot_2026 status
   ```

   (each study's entry point has its own `status`; the run log names it).

## The read

1. `open` the block. It writes the open entry to the ledger; the owner
   commits that entry before anything is read.
2. `read` the block. It writes the results file, the page and the ledger's
   closing entry. It runs once; a second read is refused, and
   `--second-read` is never passed.
3. Record the path and sha256 of each file the entry point wrote.

## After

The verdict is the code's: the statuses in the results file, the headline
sentences from the page, verbatim. Then the record step (the
`record-result` skill). A defect found now can only lower a status; a
corrected run is a successor registration, never a re-read.

## Never

- Fetch, open or compute on a protected day by any other route, including a
  month archive that holds one.
- Choose among results, change a threshold, or re-run under another rule
  after a read.
