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

| Status | Meaning |
|---|---|
| **candidate** | Positive on a held-out block; has not yet met data neither block saw |
| **confirmed** | Survived a run on days chosen after the configuration was frozen |
| **killed** | Failed that run, or failed a mechanism check |
| **artefact** | The effect was an accounting property of the measurement, not the market |

---

## Open candidates

### Market reversion at ten minutes

**Status: candidate.** Frozen at
[`strategies/reversion.py`](../src/trading_research/strategies/reversion.py);
measured in [§27](results.md) and by
[`experiments/market_reversion.py`](../experiments/market_reversion.py).

An equal-weighted index of 26 USDT perpetuals overshoots over roughly ten
minutes and comes back, so the index's recent return predicts each
constituent's next move with a negative sign.

| | Search block | Held out |
|---|---|---|
| Median net per trade | +17.2 bp | +2.8 bp |
| Cells positive | 76% | 55% |
| Best cell, instruments positive | — | 20 of 26 |
| Median net at that cell | — | +6.9 bp |

What it has survived:

- Negative information coefficient on 26 of 26 instruments, both halves.
- A gap test. Shared price noise between the end of a lookback window and the
  start of a forward window manufactures negative correlation from nothing, and
  dies the moment a gap is inserted. This strengthens slightly out to thirty
  seconds and decays over ten minutes.
- The holding-period profile repeats across blocks: two minutes loses, ten
  minutes is best, longer decays. Configuration ranks correlate at +0.35.
- The instrument ordering follows the edge identity — volatile alts beat
  BTCUSDT and ETHUSDT at similar cost, which is what IC times dispersion
  predicts.

What would kill it, stated before the test:

1. The frozen configuration run on days after 11 March 2024 returning a median
   at or below zero across instruments.
2. The holding-period profile failing to repeat on that period.

Known weaknesses, recorded now rather than discovered later:

- **Sixfold decay** between blocks, +17.2 to +2.8. The direction of that trend
  is the main reason this is not called a result.
- **One bet, not twenty-six.** Against a market-neutral target the coefficient
  collapses from −0.056 to +0.004. The instruments carry the same position, so
  agreement across them is much weaker evidence than it looks, and a portfolio
  buys leverage rather than diversification.
- +2.8 bp against a 14 bp round trip is close enough to zero to be a period.

---

## Killed

### Trading rarely to escape the cost floor

**Status: killed** ([§25](results.md)). One and two trades a day on BTCUSDT
scored +27 and +20 bp per trade. The identical procedure on sixty-two later days
returned −6.4 and −8.5, and every rate was negative there while every rate had
been positive before. The sweep had selected a period, not a rate.

### Best-of-240 configuration search

**Status: killed** ([§26](results.md)). The winner of a ten-axis search cleared
2.23 bp of a 12.02 bp round trip on the block that chose it and −0.21 bp on the
block that chose nothing. No axis moved the median candidate by more than 0.7 bp
against a 12 bp cost.

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
