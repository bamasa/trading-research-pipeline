# Findings register

Every candidate this project has produced, what was done to it, and where it
stands. A finding is not a result until it has survived data that played no part
in producing it, so the status column is the point of this table rather than the
numbers.

The register exists because the same mistake kept recurring in a different
costume: a promising number, a plausible mechanism, and no test that could have
refuted it. Writing each candidate down with its kill condition — stated before
the test — is what stopped that.

## Status meanings

**Tally: five killed, two artefacts, none open.** Every candidate this project
produced has now met the test it named in advance, and none survived it.

| Status | Meaning |
|---|---|
| **candidate** | Positive on a held-out block; has not yet met data neither block saw |
| **confirmed** | Survived a run on days chosen after the configuration was frozen |
| **killed** | Failed that run, or failed a mechanism check |
| **artefact** | The effect was an accounting property of the measurement, not the market |

---

## Killed

### Market reversion at ten minutes

**Status: killed**, on the test it named in advance. Measured in
[§27](results.md#27-the-market-goes-too-far-and-comes-back); the test itself is
[`experiments/reversion_fresh.py`](../experiments/reversion_fresh.py).

The claim was that an equal-weighted index of 26 USDT perpetuals overshoots over
roughly ten minutes and comes back, so its recent return predicts each
constituent's next move with a negative sign. On 1 February to 10 March 2024 it
looked good: negative information coefficient on 26 instruments of 26 on both
halves, a gap test separating it from a measurement artefact, a holding-period
profile that repeated across blocks, and 20 of 26 instruments positive on a
held-out block at +6.9 bp per trade.

The frozen configuration was then run on 12 March to 20 April — thresholds
carried over from the original span, nothing re-tuned — and both stated kill
conditions fired:

| Hold | Gross | Net | Instruments positive |
|---|---:|---:|---:|
| 2 min | +2.15 | −11.84 | 0 / 26 |
| 5 min | +1.27 | −13.21 | 0 / 26 |
| **10 min (frozen)** | **−5.31** | **−19.94** | **0 / 26** |
| 20 min | −15.47 | −29.62 | 0 / 26 |
| 40 min | −16.41 | −30.92 | 0 / 26 |

3,637 trades, −69,283 bp in total, and the best instrument of twenty-six is
DOGEUSDT at −9.9 bp per trade.

Condition 1 — a median at or below zero — fired at −19.94 with nothing positive.
Condition 2 — the profile failing to repeat — fired harder: the best holding
period is now the shortest tested rather than ten minutes, and at the frozen
horizon the *gross* edge has changed sign, from +16.6 bp to −5.31. This is not
an effect that weakened. It is one that was not there on a period which chose
nothing.

The sixfold decay between the original two blocks, recorded as the main worry
while it was still a candidate, was the warning.

### A model improving that rule

**Status: killed with the rule it improved.** The improvement was real and
large on the original span — +18.8 bp against the rule's +7.4 on 3,297 held-out
trades, paired *t* of 10.0, both boosters agreeing — and it is an improvement to
a strategy that does not work. A model that decides how far a losing trade will
run makes it lose less.

What it leaves behind is worth keeping, because it is about method rather than
about this signal: asking a model *how far a trade will run* and letting a rule
act on it beat asking the model *when to exit* on every architecture tried, and
the two accounting errors found on the way — booking a take-profit at the
overshoot rather than at its level, and reading 113 trades as though they could
settle a 12 bp difference — are the kind that reverse a conclusion.

### Trading rarely to escape the cost floor

**Status: killed** ([§25](results.md)). One and two trades a day on BTCUSDT
scored +27 and +20 bp per trade. The identical procedure on sixty-two later days
returned −6.4 and −8.5, and every rate was negative there while every rate had
been positive before. The sweep had selected a period, not a rate.

### Best-of-240 configuration search

**Status: killed** ([§26](results.md)). Re-run on corrected code across the
three best-screened instruments, every winner is negative on both blocks:
−8.98 and −17.32 on BTCUSDT, −5.49 and −45.43 on CRVUSDT, −11.68 and −14.70 on
BICOUSDT. Gross edges are positive on the block that chose them and two of three
flip sign on the block that chose nothing. No axis of thirteen moves the median
candidate by more than about 0.7 bp against a 12–16 bp cost.

### Maker execution

**Status: killed.** Costs fall as promised, from 14–16 bp to 5.9–7.1. The gross
edge inverts: the same signal at the same moments is worth +3 to +8 bp crossing
and −0.5 to −4.2 bp resting. Adverse selection is 5–11 bp, the same size as the
fee saving. Zero of 48 configurations positive.

### Ensembles over the book features

**Status: killed.** Variance is negligible for all eight candidates, so there is
nothing for averaging to remove; the models explain under 0.03% of the forward
move. Bagging improved gross from −11.7 to −4.0 bp and nothing else helped.
Trees actively destroyed the signal, and raw queue imbalance — one column, no
fitting — still had the best gross of anything tried.

### Instrument selection by screen

**Status: partially confirmed.** Rank correlation of 0.59 between two windows
seven weeks apart across 80 instruments, so the screen measures a real property.
But half the top ten changes between windows, so it supports a basket of five to
ten rather than a single best instrument.

---

## Artefacts

### FI-2010 smoothed labels

**Status: artefact** ([§21](results.md)). The standard two-sided smoothing
correlates +0.586 with a quantity known before the decision, because the average
it compares against includes the past. Published accuracies built on it are not
comparable to a clean forward label.

### The volatility gate

**Status: artefact.** `MarketGate.thresholds()` returned an empty dict when its
column was missing, and an empty dict masks nothing — so three instrument-wide
searches ran ungated while reporting a gate axis. A test asserted the empty dict
as correct behaviour. Both fixed.
