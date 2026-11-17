# Same strategy, different market

A post: bitcoin reaches its all-time high on 14 March 2024, and a strategy
fitted on February and then left alone gives back everything it had earned
within two days. Every number below is in [`numbers.json`](numbers.json),
written by [`make_data.py`](make_data.py) from the benchmark's recorded Bybit
archives; [`tests/test_posts.py`](../../tests/test_posts.py) fails if this page,
the animation and that file disagree.

This is a backtest on recorded data, a measurement and not a result: nothing
here was registered before the data was read, and the February span is the
benchmark's favourable track, chosen after the repository had read it.

## What was run

| | |
|---|---|
| Strategy | the repository's `reversion` plugin (ten-minute cross-sectional reversion), its source unchanged |
| Instruments | the benchmark's 26 Bybit USDT perpetuals |
| Data | top of book on the 5 s grid, `data/universe` and `data/universe_fresh` (the `bybit-2024-02` and `bybit-2024-04` tracks) |
| Fitted | 1 to 25 February 2024: the strategy's one fitted value, its entry threshold |
| Traded | 26 February to 10 March and 12 March to 19 April 2024, with that threshold never refitted |
| Fill and cost | the benchmark's taker: Bybit's base fee of 5.5 bp a side, 0.5 bp of slippage, the spread on both legs, one position at a time per instrument |

## The numbers

| | |
|---|---|
| 26 February to 10 March | 3,351 trades, +20.9 bp a trade gross, **+7.0 bp** net |
| 12 March to 19 April | 12,719 trades, +1.4 bp a trade gross, **−12.5 bp** net |
| Portfolio at its peak | **+15.3%**, in the hour to 15:00 UTC on 14 March |
| Portfolio below zero again | in the hour to 20:00 UTC on 16 March, 53 hours after the peak |
| Portfolio on 19 April | **−52.0%** |
| BTCUSDT's highest hourly mid | **$73,662**, 14 March 06:00 UTC |
| Index ten-minute autocorrelation, mean of the daily values | −0.113 (26 Feb to 10 Mar), −0.053 (12 Mar to 5 Apr), −0.015 (6 to 19 Apr) |

The February half is the benchmark's own scored span, and `make_data.py`
stops if its trades and net per trade do not match the `reversion` card in
[`bench_board.json`](../../experiments/results/bench_board.json) exactly.

## How to read it

- **The portfolio.** Capital is split equally over the 26 instruments; each
  trades one position at a time with its whole slice, no leverage, and
  profits are not reinvested. The portfolio's return is the sum of every
  trade's net bp divided by 26. The −52.0% is what a strategy paying about
  14 bp of cost a trade does over 12,719 trades with +1.4 bp a trade of gross
  edge left, down from +20.9 in the fortnight before.
- **Why it differs from the benchmark's April number.** The benchmark refits
  the threshold on 12 March to 5 April before scoring 6 to 19 April and gets
  −47.0 bp a trade there. This post never refits, which is the question it
  asks: what does a strategy left alone do when the market changes.
- **The market state.** The bars under the chart are each day's correlation of
  the equal-weighted index's ten-minute return with its next ten minutes,
  green at −0.05 or below (moves pull back, which the rule trades), red above
  −0.02. The finding in [`docs/findings.md`](../../docs/findings.md) is that
  this state is measurable and was not forecastable a day ahead.
- **The high.** $73,662 is the highest hourly mid of the perpetual in these
  archives; the intraday high printed on spot venues was slightly higher.
- **A gap.** The two tracks leave out 11 March, so nothing trades that day.

## The pipeline catches the break

Forecasting a regime change is a problem of its own; the repository's one
attempt to forecast this one a day ahead failed (a gate on the trailing state
admitted worse days than it rejected, [`docs/findings.md`](../../docs/findings.md)).
This post asks the other question: does the pipeline notice, from data it has
at each moment, that the strategy has broken? Two of its tools answer it, and
neither was tuned on this episode.

**The kill switch on the strategy's own results.**
[`validation/kill_switch.py`](../../src/trading_research/validation/kill_switch.py)
runs Page's CUSUM, the textbook sequential test for a drop in a mean, on the
portfolio's hourly net return ([`monitor.py`](monitor.py)). Each trade counts
in the hour it closed, when its result was known, and each hour is labelled by
its end; a decision at an hour's end applies from the next hour. A stop
forbids new entries and lets the positions already open run to their exits,
which count. Its values are the textbook ones: k = 0.5 and h = 5 in units of
the hourly return's standard deviation, estimated on the hours before the one
tested; the first 72 hours only estimate it.

| Policy after an alarm | Alarms | Portfolio on 19 April |
|---|---|---|
| No monitor | | **−52.0%** |
| Stop and stay stopped | 15 March 04:00 UTC, 13 hours after the peak | **+9.4%** |
| Stop, trade on paper, restart when the same CUSUM pointed upwards sees the paper result recover | stop 15 March, restart 14 April, stop 14 April | **+2.6%** |

- At its textbook values the monitor raised no alarm in the fortnight that
  paid. Its neighbours, k from 0.25 to 1 and h from 4 to 8, are less tidy:
  eight of the twelve stop 13 to 19 hours after the peak and keep +4.5% to
  +9.4%; three (k = 0.25 with h = 4 or 5, and k = 0.5 with h = 4) stop on 5 or
  6 March, inside the good fortnight, and keep +5.3% to +7.9%; one (k = 1,
  h = 8) stops only on 12 April, at −18.5%. None ends near −52%. A more
  sensitive switch costs good days; a duller one costs bad hours.
- Restarting is the hard part. The automatic restart came on a bounce on 14
  April and gave back most of what the stop had kept.
- This is one episode, known before the monitor was written. The values are
  the textbook's and their neighbours are shown, but it is not an
  out-of-sample test of the monitor.
**The structural-break detector on the market itself.**
[`validation/structural_breaks.py`](../../src/trading_research/validation/structural_breaks.py),
from the author's solution to the ADIA Lab Structural Break Challenge, whitens
a stream by its own history and accumulates the odds of a change family by
family. Its `dependence` family is the strategy's mechanism: a change in how
much a ten-minute move depends on the previous one. [`breaks.py`](breaks.py)
runs it on the index's ten-minute returns with the detector's default,
null-calibrated setting (history 365, online 90), the dependence family only:

| Alarm | Direction | Reading |
|---|---|---|
| 6 February 13:00 | dependence fell | moves reverse more: the strategy's state strengthening |
| 2 March 00:00 | dependence fell | again |
| 6 March 05:30 | dependence rose | an early warning; the strategy still earned for a week |
| 15 March 09:10 | dependence rose | again, the morning after the portfolio peaked and about 5 hours after the kill switch |
| 12 April 20:30 | dependence rose | again |

The kill switch says when to stop; the detector says why. Run on the
strategy's hourly results instead, with its mean family, the detector is
sensitive to its setting: at its calibrated 250/60 setting its first
falling-mean alarm is on 10 March, inside the good fortnight (a stop there
keeps +6.9%), and with a 120-hour history it is on 7 March (+5.9%).
Those runs were tried and are reported here; the post uses the CUSUM as the
switch and the detector on the index as the diagnosis.

## Rebuild

```bash
PYTHONPATH=src python posts/btc_ath_2024/make_data.py --data data
PYTHONPATH=src python posts/btc_ath_2024/monitor.py
PYTHONPATH=src python posts/btc_ath_2024/breaks.py
node posts/btc_ath_2024/render.mjs
```

The first needs the two tracks' archives (`trading-research bench fetch`) and
rewrites `data.json` and `numbers.json`; the next two write `monitor.json` and
`breaks.json` from `data.json` (the second needs the `breaks` extra); the last
needs Chrome and ffmpeg and
writes `artifacts/posts/btc_ath_2024.mp4` (1080 x 1080, 20 s) and a
self-contained page beside it. The video is drawn frame by frame by the page's
own `renderAt`, so it is the page, not a screen recording.
