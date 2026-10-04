# Findings register

Every candidate this project has produced, what was done to it, and where it
stands. A finding is not a result until it has survived data that played no part
in producing it, so the status column is the point of this table rather than the
numbers.

The register exists because the same mistake kept recurring in a different
costume: a promising number, a plausible mechanism, and no test that could have
refuted it. Writing each candidate down with its kill condition — stated before
the test — is what stopped that.

**Conditional** deserves a note, since it is the status one candidate ended at
and the one most easily abused. It means the effect and its condition were both
measured, the condition was stated as a number rather than a story, and the
failure outside it was measured too. It does not mean the strategy can be run:
that additionally requires forecasting the condition, which is a separate claim
and, in the one case here, a tested and negative one. A conditional result is a
statement about how a market behaves. It is not a position.

## Status meanings

**Tally: one conditional result, eight killed (four of them the market-making
study's pre-registered primaries, on its held-out fortnight), two artefacts; one
secondary hypothesis waiting for its boundary block, and five hypotheses of a
second market-making round registered before their data is fetched.** Every
tested candidate met the test it named in advance. One survived as a statement
about a market state rather than as a strategy; the rest did not survive at all.

| Status | Meaning |
|---|---|
| **pre-registered** | Hypothesis, metric and kill conditions committed before any code or number; not yet tested |
| **candidate** | Positive on a held-out block; has not yet met data neither block saw |
| **conditional** | Holds under a stated, measured condition — and fails outside it, which was also measured |
| **confirmed** | Survived a run on days chosen after the configuration was frozen, unconditionally |
| **killed** | Failed that run, or failed a mechanism check |
| **artefact** | The effect was an accounting property of the measurement, not the market |

---

## Conditional results

### Cross-sectional reversion at ten minutes

**Status: conditional result.** The effect is real and measured; the condition it
needs is real and measured; the condition is not forecastable by what was tried.
Frozen at
[`strategies/reversion.py`](../src/trading_research/strategies/reversion.py),
reported in [§27](results.md#27-cross-sectional-reversion-and-the-market-state-it-requires).

**Where it holds.** On a held-out fortnight, 22 of 26 instruments positive, a
median of +6.99 bp per trade net of 12–16 bp of cost, +18.8 bp with a model
deciding how far each trade runs.

**The condition.** The index's ten-minute autocorrelation is −0.1014 on the span
where it works and −0.0003 on the span where it does not. That is the quantity
the rule trades, measured directly.

**The mechanism.** A rising market's ten-minute surge is impatient buying that
retraces; a falling market's is forced liquidation that does not. The rule
selects the largest surges, which in a cascade are the continuations — so it
should lose money rather than merely stop making it when the state changes, and
gross went from +20.50 to −5.07.

**The boundary, which is the part that stops this being a strategy.** Three ways
of escaping the condition were tried and all failed: daily refitting of the
threshold (−24 to −59 bp, gross negative in every arm), daily re-estimation of
the sign (nothing to find — the relationship vanished rather than inverted), and
a regime gate trading only on days whose trailing autocorrelation is low. The
gate fails informatively: days it admits are *worse* than days it rejects, in all
four windows. The state is measurable and, by this evidence, not forecastable one
day ahead.

**And the sample.** The +6.99 figure counts trades; the instruments carry one
market-wide bet, correlating at +0.47 per day. By day it is +6.18 bp over 14
days, *t* = 1.18. That establishes presence and size, not persistence.

### A model improving that rule

**Status: conditional, with the rule it improves.** On the span where the rule
works, a model raises it from +7.4 to +18.8 bp per trade over 3,297 trades,
paired *t* of 10.0, both boosters agreeing. On the span where the rule does not
work, a model that decides how far a losing trade will run makes it lose less.

What generalises beyond this signal: asking a model *how far a trade will run*
and letting a rule act on it beat asking it *when to exit* on every architecture
tried. And two accounting errors found on the way — booking a take-profit at the
overshoot rather than at its level, and reading 113 trades as though they could
settle a 12 bp difference — are the kind that reverse a conclusion rather than
shade it.

---

## Killed

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
fee saving. Zero of 48 configurations positive. The conditional version for the
reversion signal, where being filled against the move might have helped, was
pre-registered as H2.2 and is killed too (above).

### Market making on public Bybit data

**Status: killed, all four primaries** (H1, H2.1, H2.2, H3), on the held-out
fortnight read once
([results](preregistration/market_making.md#results-on-block-h-read-once-4-october-2026),
[§30](results.md#30-market-making-on-the-held-out-fortnight-read-once)). Each
hypothesis, its metric and its kill conditions were committed before any
market-making code existed, and every value was frozen on the development block
before the held-out block was opened.

- **H1, inside the spread: killed** (K1, K2, K4, K-fund). At the frozen gate the
  inside rule was filled six times in thirteen days; S2 lost 1.21 USDT a day on
  BICOUSDT and its twin 1.20.
- **H2.1, leaning against the index: killed** (K1, K3, K-fund). S4 − S1 pooled
  over three instruments was −91.8 USDT a day, below the 95th percentile of its
  50 shuffled-state placebos.
- **H2.2, passive execution of the reversion signal: killed** (K1–K4, K-fund).
  −4.92 bp per attempt resting against +6.38 bp per trade crossing, on the same
  376 triggers; following the move instead did slightly better. This was the
  maker case for the conditional result above, and it fails.
- **H3, the regime guard: killed** (K2). The guard earned +0.107 USDT a day over
  the quoter it guards, but the same flags moved to random times earned as much.

None of the four passed Holm's procedure either. The measurements beside them:
on the one instrument admitted for market making, no rung of the advantage
ladder short of perfect foresight of the next second makes the touch quoter
pay, and no strategy breaks even at any maker fee Bybit publishes.

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

---

## Pre-registered

### Market making: the causal state gate (H2.3)

**Status: pre-registered, half read.** The secondary hypothesis of the
market-making study, outside its Holm family and with a negative prior: days
whose trailing one-day index autocorrelation, known at the open, is at or below
D's median (−0.0541) give S4 a larger advantage over S1 on BICOUSDT than days the
gate rejects, over the held-out fortnight and the boundary block together. On the
held-out fortnight alone the gate admitted 10 days at +0.575 USDT a day against
+0.488 on the 3 it rejected; the verdict waits for the boundary block, which is
read once, after these results are committed. The four primaries of the same
study are under Killed above, and none of them can become conditional on that
block, since conditional requires a candidate on the held-out fortnight.

### Market making, second round: wide-spread instruments (B1–B4)

**Status: pre-registered, nothing fetched**
([pre-registration](preregistration/market_making_round2.md),
[config](../configs/mm_prereg_round2.yaml)). Whether any stated condition makes
the market maker pay, on data no market-making code has read: eight wide-spread
instruments admitted on the 25-day development block, their held-out fortnight
read once, four pristine weeks from 8 April 2024 for any survivor. Holm across
five primaries, each with round one's K-pess, K-fund and K-dir, a joint
pessimistic-queue condition and a neighbourhood condition:

- **B1**: the quoter, re-searched per instrument, pays at the base fee. Killed
  if the pooled net is not positive.
- **B2**: quoting only while the trailing room (passive markout less the fee)
  clears a margin pays, and beats quoting throughout. Killed if either is not
  positive, if 50 time-shifted gates do as well, or if the room does not persist.
- **B3a, B3b**: the B1 or B2 quoter at the 0 bp professional tier, and at the
  1 bp programme rebate. A conditional result about fees, not a retail one.
- **B4**: the gate on BICOUSDT over the pristine block, which no market maker
  has read.

A survivor counts only if it is confirmed on a further block (the pristine
weeks; for B4, the four weeks after them); the sentence that will report it,
with where it fails, is fixed in the registration.
