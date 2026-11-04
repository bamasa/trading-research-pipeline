---
name: benchmark-strategy
description: Score a strategy or a model, the repository's own or a new plugin, on the benchmark's fixed tracks (planted edges and their nulls, two recorded Bybit spans) and read its card honestly, including at what fee it would pay. Use when asked to benchmark, score, compare or test a strategy, to check whether a new idea is any good, or to say at which fee a strategy would be profitable.
---

# Benchmarking a strategy

The benchmark is one rule applied to every strategy on the same six tracks, so
two strategies' cards can be compared line by line. `docs/benchmarks.md` is the
page; `CLAUDE.md` binds this skill. A card is a measurement, never a result
about a market: a strategy that passes still needs a registered study.

## The tracks

```bash
trading-research bench list
```

Four are synthetic, generated from a fixed seed with a known answer: an
order-book edge and a cross-sectional reversion, each with its null. Two are
recorded Bybit spans of 2024, chosen in hindsight: one where short reversion
paid, one after it stopped. They need the archives under `data/universe` and
`data/universe_fresh`; without them they are skipped, never faked.

## Scoring a strategy

A strategy of one's own starts as a plugin, written from the template and
passed through the conformance checks first (`docs/extending.md`):

```bash
trading-research strategies new my_rule
trading-research strategies check my_rule
```

Then the card, on every track (the option --tracks names fewer, --param sets
a constructor argument):

```bash
trading-research bench run my_rule
```

The board of every registered strategy (--kind model for the models):

```bash
trading-research bench board
```

## Reading a card

- **Marks** run in track order: ✓ did what a good strategy should, ✗ did not,
  · missed a planted effect it was not built to find, - not run.
- **A null that is not quiet** (gross with a *t* of 3 or more where nothing
  was planted) means the strategy reads the future or fits noise. Find the
  defect before anything else; `pre-freeze-review` has the checklist.
- **The placebo** is the same decisions moved in time. A gross that does not
  beat 95% of them is drift, not timing.
- **The fee ladder** gives net at Bybit's base taker fee, VIP 1, the lowest
  published taker fee, and none at all. The last is hypothetical: say so
  whenever it is quoted, and say which account the break-even fee would need.

## What the model must not do

- Tune a strategy's parameters on a recorded track and quote the card: the
  tracks are fixed and public, so a strategy fitted to them proves nothing.
- Call a ✓ on a recorded track a result. It is a measurement on a span chosen
  after it was read. The step after a good card is `propose-hypotheses`, then
  `preregister-study`.

## Hand back

The card's path, its marks, and for each recorded track the net at the base
fee and the break-even fee, in one line each, with the disclaimer's point that
a pass is not a forecast.
