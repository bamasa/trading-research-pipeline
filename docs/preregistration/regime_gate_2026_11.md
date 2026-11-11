# Pre-registration: pausing the reversion rule when the market stops reverting

| | |
|---|---|
| **Committed** | 11 November 2026, before any structural-break alarm was computed on the regime suite's ten episodes and before the benchmark's forward track opens (21 November 2026) |
| **What it registers** | One gate, two tests of it (G1 on the regime suite's ten episodes, G2 on the forward fortnight), their metric, placebos, kill conditions and the expected outcome |
| **Searched** | **Nothing.** The gate, the detector's setting and the strategy are fixed below; no value is chosen on any result |
| **How it changes** | Only by a dated amendment appended at the end, which cannot change a status already read |

## Why

The reversion rule (`strategies/reversion.py`) trades against the equal-weighted
index's large ten-minute moves. It pays while those moves partly reverse and
loses when they stop reversing ([`docs/findings.md`](../findings.md#cross-sectional-reversion-at-ten-minutes)).
Section 27 found the state measurable but not forecastable a day ahead: a gate
on its trailing value admitted worse days than it rejected.

The pipeline's structural-break detector
([`validation/structural_breaks.py`](../../src/trading_research/validation/structural_breaks.py))
reads the same quantity a different way: its `dependence` family accumulates,
on a stream whitened by its own history, the odds that the lag-one dependence
has changed. In [`posts/btc_ath_2024`](../../posts/btc_ath_2024/README.md) it
flagged weaker reversals on the index on 6 March 2024, eight days before the
rule's portfolio peaked, and again the morning after. One episode cannot say
whether that is a usable signal. This registration asks it on ten.

## The gate

- **Stream.** The equal-weighted index of the episode's instruments (the mean
  of their log mid prices), its non-overlapping ten-minute log returns, each
  labelled at its end.
- **Detector.** `detect_breaks` with `MonitorSpec(history_len=365,
  online_len=90, statistics=("dependence",))`: the detector's default,
  null-calibrated setting, the dependence family only.
- **State.** The rule trades from the first scored row. A `+1` dependence alarm
  (dependence rose: moves reverse less) **pauses** it; a `-1` alarm (moves
  reverse more) **resumes** it. Alarms raised during the fitted days set the
  state at the first scored row. A pause forbids new entries on every row
  labelled after the alarm's ten-minute return; positions already open run to
  their exits.
- **Strategy.** The `reversion` plugin at its default parameters, fitted on
  the episode's fitted days and frozen, traded by the benchmark's taker at
  Bybit's base fee.

## G1: the regime suite's ten episodes

The ten episodes of [`bench/regimes.py`](../../src/trading_research/bench/regimes.py),
their panels, bars, fitted and scored days as the suite defines them.

- **Metric.** `D` = the gated rule's net result minus the ungated rule's, in %
  of capital split equally over the twelve instruments, summed over the scored
  days of all ten episodes (each trade counted by the row it opened).
- **Placebos.** Nineteen copies of each episode's pause mask, circularly
  shifted over that episode's scored rows by 1/20 to 19/20 of their number;
  the same total time paused, at the wrong times. `D` is computed for each.
- **G1 passes** when `D > 0` **and** `D` exceeds all nineteen placebo values.
  Otherwise it is **killed**, with the first condition that failed.
- **Reported beside it, with no status:** `D` per episode, the number of
  episodes the gate helps, each episode's alarms, and the share of scored time
  paused.

## G2: the forward fortnight

The benchmark's forward track `bybit-2026-11`
([registration](bench_forward_2026_11.md)): its 26 instruments, fitted on 9
October to 5 November 2026 and scored on 6 to 19 November 2026, top of book on
the five-second grid. G2 is read once, after that track's own read (F1, P1) is
committed, and changes nothing in it.

- **Metric.** `D` as in G1, over the fortnight's scored days, in % of capital
  split over the track's instruments.
- **G2 passes** when `D > 0` and `D` exceeds all nineteen placebo values;
  otherwise it is killed. With no alarm in the fortnight or the fitted days,
  the gate never pauses, `D = 0`, and G2 is **not tested**.

## Expected outcome, stated now

**Both killed.** The trailing-state gate of section 27 failed in all four of its
windows, and in the one episode looked at the detector raised its first
warning eight days before the rule stopped paying, with good days in between.
A gate that pauses early gives up the good days; one that pauses late saves
little. The prior is that the detector's alarms are informative about the
state but too early or too frequent to time the rule.

## What was known before this was written

- The regime suite's results for `reversion` on the ten episodes (net per trade
  before and after each break, the kill switch's stops;
  [`experiments/results/bench_regimes`](../../experiments/results/bench_regimes/regimes.md)).
- The detector's dependence alarms on the 2024-02 and 2024-04 recorded tracks'
  index (26 Bybit instruments, top of book), for `posts/btc_ath_2024`: `-1` on
  6 February and 2 March, `+1` on 6 March, 15 March and 12 April 2024. The
  suite's `ath-2024` episode reads twelve instruments' Binance bars instead,
  over different days, so its alarms are not those.
- No dependence alarm on any other episode, and nothing of the forward
  fortnight's scored days.

## Results of G1, read once 11 November 2026

[`experiments/results/regime_gate_G1.md`](../../experiments/results/regime_gate_G1.md),
from the code committed before the read (`experiments/regime_gate.py`).

**G1: killed** — a placebo gained as much. The gate gained **+13.07%** of
capital over the ten episodes and beat **none** of the nineteen placebos, whose
gains ran from +27.1% to +60.5% (median +43.4%). It helped in 3 of the 10.

- Almost all of the gate's gain came from two episodes, the 2021 top and LUNA,
  where a `+1` alarm in the fitted days paused the rule for the whole scored
  span: +10.5% and +15.3% for not trading a losing strategy at all. A pause
  that covers everything is the same under every shift, so the placebos have
  it too.
- Where the alarms fell inside the scored span the gate timed its pauses worse
  than chance: FTX −11.3%, the ETF launch −7.0%, Celsius −3.0% against
  placebo medians of +1.1%, −7.3% and +3.1%. It paused the ETF fortnight, which
  paid, for 81% of its time.
- As the registration expected: the detector's dependence alarms carry
  information about the state, and as an on-off switch for the rule they are
  worse than switching at random for as long.

G2 is unchanged and is read after the forward track's own read.
