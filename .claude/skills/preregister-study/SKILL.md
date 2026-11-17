---
name: preregister-study
description: Write a new study's pre-registration (the document under docs/preregistration/ and its YAML under configs/) before any of its data is read. Use when asked to propose or register a hypothesis, design a study, choose its blocks, or compute a registration's null-placeholder sha256.
---

# Pre-registering a study

A study is a document `docs/preregistration/<study>.md` and a YAML
`configs/<study>.yaml` that agree, committed before any of its data is read.
The pattern to copy is `docs/preregistration/spot_2026.md` with
`configs/spot_2026.yaml`. The researcher's and the registrar's duties are in
`agents/roles/researcher.md` and `agents/roles/registrar.md`; follow them by
hand as the harness would.

## Order

1. **Read what is known.** `docs/findings.md` (what was killed and why),
   the README's roadmap, and the study's neighbours under
   `docs/preregistration/`. Write down the information set: what the owner and
   this repository have already seen of the data, with dates.
2. **Propose.** For each hypothesis: the claim, the metric (the daily series
   and its unit), the kill conditions in the order they are read, the
   placebos, the neighbourhood, and the expected outcome with the arithmetic
   behind it. A hypothesis without a kill condition is declined, not
   registered.
3. **Choose the blocks** against the protected calendar; no block may touch a
   protected day, and a month archive counts as its whole month:

   ```bash
   PYTHONPATH=src python -m agents.workflows.research_loop calendar
   ```

4. **Write the document and the YAML.** Every section of the pattern: data,
   fees with source and date read, blocks, settings, slots, metrics, status
   rules with the family and Holm's m, kills per slot, placebos, headline
   sentences with brackets, what the study cannot establish. A value set on
   the development block is a `null` placeholder commented "frozen by the D
   amendment".
5. **Compute the null-placeholder sha256** and write it into the document:

   ```bash
   PYTHONPATH=src python -m agents.workflows.research_loop null-hash configs/<study>.yaml
   ```

6. **Hand to the owner** to commit. Nothing of the study's data is fetched
   before that commit.

## Refuse when

- a hypothesis has no kill condition or no expected outcome;
- a block touches a protected day;
- a fee has no source and date;
- the family has no m, or a headline sentence is not fixed.

Say what is missing; do not fill it with a guess.

## Never

- Edit an existing registration or YAML: they are frozen (`CLAUDE.md`, rule 2).
  A correction is a dated amendment appended to the document.
- Look up the market data of any date on the web; a web page is read for fees
  and formats only.
