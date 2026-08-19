# Parity with the closed-source work

This project was built from scratch as a public counterpart to a closed research
codebase. No code, path, data or configuration was carried across — see
[`disclosure_policy.md`](disclosure_policy.md) — but the *techniques* are public
knowledge and it is worth recording which of them are present here, which are
absent, and why.

## Present

| Technique | Where it lives here |
|---|---|
| Rolling normalisation of features against a trailing window | `features/normalise.py` |
| Robust statistics (median / IQR) rather than mean and standard deviation | `RollingNormaliser(method="robust")` |
| A floor on the scale, so a pinned price does not divide by nothing | `RollingNormaliser.transform(floor=...)` |
| An exclusion list for features already bounded and centred | `RollingNormaliser(exclude=...)` |
| Smoothed target: mean of the next *k* against the mean of the last *k* | `labels/targets.py`, `smoothed_move_bp` |
| Target expressed as a relative change rather than an absolute price | every target; §21 |
| Queue imbalance, microprice deviation, order-flow imbalance | `features/book.py` |
| Spread, total depth, depth ratio | `features/book.py` |
| Trade-flow imbalance, trade counts, traded volume, buy share | `features/generated.py` |
| Price-impact coefficient per unit of signed flow | `features/generated.py`, `impact_*` |
| VWAP deviation, trade dispersion | `features/generated.py` |
| Realised volatility over several windows | `features/book.py`, `features/microstructure.py` |
| Dilated causal convolutional network | `models/tcn.py` |
| ONNX export for serving outside Python | `models/export.py` |
| A staged pipeline: preprocess, features, target, train, apply | `pipeline/`, `cli.py` |

## Absent, and why

**Multi-level book features.** The prior work reads ten levels per side and
normalises each level's price and size against the touch. Everything here uses
`bookTicker`, which publishes only the best bid and ask, because that is what
any exchange gives away free. Level imbalance, book slope and concentration are
defined in the feature registry's docstrings and left unimplemented rather than
faked on data that cannot support them — a collector recording the full book
forward in time is on the roadmap.

This is the single largest difference, and it is a data difference rather than a
methodological one.

**Calibration-period floors for volume scaling.** The prior work computes a
floor for each size column from the distribution of its rolling standard
deviations over a calibration window. Here the floor is either the spread or a
fallback measure of dispersion. The calibrated version is better and needs a
calibration period the public archives make awkward to define; the simpler floor
solves the failure it was written for.

## Different on purpose

**Everything is tested against a leakage checker.** The prior work computes
rolling statistics with prefix sums for speed; this uses pandas and is slower,
and every feature declares a lookback that a test verifies by mutating data
after a cutoff. The trade is deliberate: this repository exists to demonstrate
that the numbers can be trusted, and speed matters less than a mechanical
guarantee.

**Nothing is fitted on the whole sample.** Normalisation, feature selection,
thresholds and schedules are all computed inside a training block and carried
forward. That is stricter than a research notebook usually needs and it is the
point of the exercise.

**The result is reported rather than pursued.** The closed work aims at a model
to deploy. This aims at whether the thing is possible at all, which is why the
central document is a list of what did not work and the arithmetic that explains
it.
