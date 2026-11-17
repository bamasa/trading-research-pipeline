# Pre-registration: the benchmark's forward track, 6 to 19 November 2026

This document registers a test on days that had not happened when it was
written. It is committed together with the code that reads them
([`bench/tracks.py`](../../src/trading_research/bench/tracks.py), track
`bybit-2026-11`) and the frozen list of what is read
([`bench/manifest.py`](../../src/trading_research/bench/manifest.py),
`FORWARD_STRATEGIES`), and the three must agree.

| | |
|---|---|
| **Committed** | 5 November 2026, before 00:00 UTC on 6 November, the first scored day. The commit's timestamp on the public repository is the proof |
| **What it registers** | Whether any of the version-1 board's twelve strategies pays Bybit's base taker fee on a fortnight no one could have fitted to, and a prediction, stated in advance, of what the cross-sectional reversion will do given the market state that fortnight turns out to have |
| **Searched** | **Nothing.** The twelve run at their default parameters and at the plugin sources frozen in `FORWARD_STRATEGIES`; the track, the fill, the fees and the verdict are version 1's |
| **Computed so far** | **Nothing on the track's span.** No archive of 9 October to 19 November 2026 was downloaded before this was committed; the venue was asked only whether the 26 instruments had an archive for 3 and 4 November (all did, POLUSDT standing for MATICUSDT) |
| **How it changes** | Not at all before the read. After it, only by a dated amendment appended at the end, which cannot change a status already read |

## Why

Every other result on the benchmark is a measurement: the recorded spans were
chosen after the repository had read them, and the synthetic tracks have an
answer planted by the same people who wrote the strategies. A forward test is
the one check that survives all of that (G-Research froze its competitors'
models and scored them on months that had not happened;
[`benchmarks_sota.md`](../benchmarks_sota.md)). The reversion's own finding
([`findings.md`](../findings.md#cross-sectional-reversion-at-ten-minutes)) is
conditional: it pays when the index's ten-minute autocorrelation is near
−0.10 and loses when it is zero, and the state was not forecastable a day
ahead. A fortnight that has not happened is where that can be checked without
the hindsight that chose the two recorded spans.

## The track

| | |
|---|---|
| Name | `bybit-2026-11`, kind `forward` |
| Instruments | the 26 USDT perpetuals of the recorded tracks, with POLUSDT for MATICUSDT (the venue retired MATICUSDT when the token became POL) |
| Data | Bybit's public order-book archives (`ob200` since 21 August 2025), top of book on the 5 s grid labelled at the end of each bin, as the recorded tracks |
| Fitted on | 9 October to 5 November 2026 (28 days), the rows before the first scored one |
| Scored | 6 to 19 November 2026 (14 days) |
| Opens | 21 November 2026: until then `bench` refuses to build it |
| Instruments missing | one the venue publishes no archive for on some day of the span is dropped and named; the track is read with at least 20, and not at all with fewer |

## What is read

The twelve of the version-1 board, at their default parameters:
`book_imbalance`, `ev_gate`, `hold_first`, `relative`, `reversion`, `spread`
(strategies) and `breakout`, `logistic`, `mean_reversion`, `momentum`,
`order_flow`, `xgboost` (models). A test fails if any of their plugin sources
changes before the read
(`test_the_strategies_read_on_the_forward_track_are_the_ones_registered`).

Each is scored by version 1's rule unchanged: a taker at the touch, Bybit's
base fee of 5.5 bp a side, 0.5 bp of slippage, one position at a time with the
strategy's cooldown, the day-clustered *t*, nineteen time-shifted placebos and
the fee ladder ([`benchmarks.md`](../benchmarks.md)).

## The test

**F1, for each of the twelve.** The strategy **passes** when its verdict on
the track is *pays the base fee* (net per trade above zero, day-clustered *t*
of at least 2, gross above 95% of the placebos) **and** its Holm-adjusted *p*
for net above zero, over the twelve (m = 12), is below 0.05. Otherwise it is
**killed** on the forward track.

The prior is that none passes. Neither recorded span had one, and the
conditional result needs a state that was not forecastable.

**P1, the reversion's state prediction.** Outside the Holm family, with no
status of its own. The index's ten-minute autocorrelation on the scored days
is measured by `trading-research bench state` (`bench/state.py`; −0.1323 on
the 2024-02 track's scored span and +0.0219 on the 2024-04 track's, measured
before this was committed). The prediction:

- if it is **−0.05 or below**, `reversion`'s net per trade at the base fee is
  **above zero**;
- if it is **above −0.02**, `reversion`'s net per trade at the base fee is
  **zero or below**;
- between the two, no prediction.

P1 is **right**, **wrong**, or **not tested** (the state fell between). A
right P1 with a killed F1 says the condition holds and fourteen days are too
few to show it; it is not a pass.

## The read, once

On or after 21 November 2026, and once:

```bash
trading-research bench fetch
trading-research bench state
trading-research bench board
```

with the board restricted to the twelve and to the forward track, and both
kinds. The output is committed as it comes, with the track's data
fingerprint, to `experiments/results/bench_forward_2026_11.json`, and the
fingerprint is then added to version 1's manifest. Nothing is re-run with a
different parameter, instrument or fee; a defect found in the code after the
read is fixed in an amendment that reports both numbers, and the status stays
the first read's.

## What was known before this was written

- The two recorded spans' boards and states, above and in
  [`benchmarks.md`](../benchmarks.md).
- Nothing about the market from 9 October 2026 on: no archive of the span was
  downloaded. The model that drafted this has general knowledge only to June
  2026, before the span.

## Note, 8 November 2026: the fitted days, fetched before the read

Not an amendment: nothing registered above changes.

- The 728 fitted instrument-days, 9 October to 5 November 2026, were
  downloaded with `trading-research bench fetch`, which fetches a sealed
  track's fitted days only. One (OPUSDT, 14 October) failed on the first
  attempt and was fetched on the second; the fetch now retries a failed day
  once and names any that fail twice, where it used to leave the gap silent.
- No scored day was downloaded: no file of 6 November or later exists under
  the track's folder. A day's archive ends with the next day's first
  snapshot; the fitted window is cut at 6 November 00:00 UTC, so that
  snapshot is never read.
- All 26 instruments have every fitted day, 483,840 rows each on the 5 s
  grid, with no crossed or missing touch. Their fingerprint,
  `6af6b9ab865a92a5…`, is in the manifest (`FORWARD_FIT_FINGERPRINT`);
  `trading-research bench verify` checks a checkout's fitted days against it.
- For context only, and with no bearing on P1, which is about the scored
  days: the index's ten-minute autocorrelation over the fitted days is
  +0.012, close to the 2024-04 track's +0.022, where the reversion lost. The
  repository found the state not forecastable a day ahead
  ([`findings.md`](../findings.md#cross-sectional-reversion-at-ten-minutes)),
  so this is not a forecast of the scored fortnight.

## Note, 9 November 2026: the read as code, before the read

Not an amendment: nothing registered above changes. The three commands under
*The read, once* are now one, `trading-research bench forward`
([`bench/forward.py`](../../src/trading_research/bench/forward.py)), committed
before the track opens, so how the read is decided is fixed before any scored
number exists.

- It refuses before 21 November; it refuses when
  `experiments/results/bench_forward_2026_11.json` exists, so it runs once;
  and it refuses when any of the twelve plugin sources differs from
  `FORWARD_STRATEGIES`.
- It scores the twelve on the track, measures the state as `bench state`
  does, checks the fitted days against `FORWARD_FIT_FINGERPRINT` and writes
  the board rows, F1 and P1 to that file, with the track's data fingerprint.
- Two cases the text above leaves implicit, stated now as the code decides
  them. F1's Holm adjustment counts all twelve (m = 12) even if one is not
  applicable on the track or makes no trade; such a strategy is killed. P1
  is **not tested** if the reversion makes no trade on the scored days,
  since it then has no net per trade to compare.
