# Parity with, and provenance from, the author's 2025 research

In 2025 the author did short-horizon order-book research while employed at a
trading firm. The code written for that research is the author's own, and the
terms of that employment permit reusing it. Everything in this repository that
the provenance table below does not list was written for it without carrying
that code across: the techniques in the first table were re-implemented from
public knowledge.
From the 2025 short-horizon study on, parts of the research code are re-typed
into the pipeline under the rules of [`disclosure_policy.md`](disclosure_policy.md):
outputs stripped, the firm's paths and tuned values removed, each piece tested
like the rest. None of the firm's data, model weights, infrastructure, fee terms
or configuration comes with them, and no result of the original research is
cited as evidence.

This document records which of the research's techniques are present here,
which are absent and why, and, in the provenance table, where each re-typed
piece came from and what was changed on the way.

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

**A normalised ten-level book tensor.** The 2025 research reads ten levels per
side and normalises each level's price and size against the touch into one
tensor. Much of this repository runs on Binance `bookTicker`, which publishes
only the best bid and ask. Bybit's archives publish the full book, and
`features/depth.py` computes slope, concentration, weighted imbalance and
book-walking cost from up to ten levels; §24 of the results found they add
nothing measurable to the touch. The per-level tensor itself is not here yet; it
is the first piece planned for porting.

**Calibration-period floors for volume scaling.** The 2025 research computes a
floor for each size column from the distribution of its rolling standard
deviations over a calibration window. Here the floor is either the spread or a
fallback measure of dispersion. The calibrated version is better and needs a
calibration period the public archives make awkward to define; the simpler floor
solves the failure it was written for. It is planned for porting with the
tensor, with the calibration window fixed before any scored data is read.

## Different on purpose

**Everything is tested against a leakage checker.** The 2025 code computes
rolling statistics with prefix sums for speed; this uses pandas and is slower,
and every feature declares a lookback that a test verifies by mutating data
after a cutoff. The trade is deliberate: this repository exists to demonstrate
that the numbers can be trusted, and speed matters less than a mechanical
guarantee.

**Nothing is fitted on the whole sample.** Normalisation, feature selection,
thresholds and schedules are all computed inside a training block and carried
forward. That is stricter than a research notebook usually needs and it is the
point of the exercise.

**The result is reported rather than pursued.** The 2025 research aimed at a
model to deploy. This aims at whether the thing is possible at all, which is why
the central document is a list of what did not work and the arithmetic that
explains it.

## Provenance of re-typed code

Each pull request that brings in a piece of the author's 2025 research code adds
a row here. The source is named by version only; no path, file name or tool of
the firm appears, and no value tuned on the firm's data is carried over.

| Pipeline module | Source version | What was changed |
|---|---|---|
| — | — | Nothing has been ported yet |

Planned, each in its own pull request and under the leakage checker: the
normalised ten-level book tensor with calibration-window floors; a taker label
that walks the book; a TCN with a Gaussian head; the hold-first decision rule;
an expected-value gate with tradability checks and event aggregation; a
close-by-time exit; a side-by-side comparison of fill rules; and parity tests of
streaming against batch features and of ONNX against the trained model.
