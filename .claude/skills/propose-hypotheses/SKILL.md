---
name: propose-hypotheses
description: Generate new trading hypotheses from what this repository has already found and killed, each with a mechanism, a kill condition and an expected outcome, ranked by how cheaply they can be refuted. Use when asked for new ideas, what to research next, which strategy or instrument to try, or how to build on a finding.
---

# Proposing hypotheses

A language model is good at reading a long record of what failed and why, and
at naming a mechanism that would explain it. It is bad at knowing whether an
idea works: that is the data's job, after a registration. This skill produces
candidates for `preregister-study`, never a result. `CLAUDE.md` binds it, and
`agents/roles/researcher.md` is the role it follows by hand.

## Read first

- `docs/findings.md`: every candidate so far and its status. A killed idea is
  not proposed again unless the proposal names what is different and why the
  kill would not fire again.
- `docs/results.md` for the numbers behind each finding, and the README's
  roadmap.
- The protected calendar, since a proposal names the blocks it would need:

  ```bash
  PYTHONPATH=src python -m agents.workflows.research_loop calendar
  ```

## Write, for each hypothesis

1. **Claim**, in one sentence: what earns money, where, at which fee tier.
2. **Mechanism**: who is on the other side and why they pay. No mechanism, no
   hypothesis.
3. **Closest finding** in `docs/findings.md`, and what this one changes.
4. **Metric**: the daily series and its unit.
5. **Kill conditions**, in the order they are read, with the placebo that would
   show the result is chance and the neighbourhood that would show it is a
   lucky setting.
6. **Expected outcome with the arithmetic**: the gross edge needed against the
   cost floor (12 to 16 bp a round trip taker, about 6 bp maker, from
   `docs/execution_assumptions.md`), and how many days a block needs to tell
   it from zero.
7. **Information set**: what of the data the owner or this repository has
   already seen.
8. **Cost to refute**: data to fetch, code to write, days of compute.

Rank by expected information per unit of cost: a cheap test that would kill a
popular idea ranks above an expensive one that could only confirm a weak one.

## What the model must not do

- Quote a number about a market from its own memory. Its training data reaches
  into periods this repository holds out; a number it recalls is a held-out
  read by another route. Every number in a proposal comes from a file of this
  repository, with its path.
- Look up prices, funding or volumes on the web.
- Propose a threshold, a window or an instrument because it "worked" in a
  result already read: that is choosing after a read.

## Hand back

A short list (three to five), each hypothesis with the eight items above, for
the owner to choose one from and register with `preregister-study`.
