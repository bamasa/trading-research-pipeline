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
| Portfolio at its peak | **+15.3%**, 14 March 14:00 UTC |
| Portfolio below zero again | 16 March 19:00 UTC, 53 hours after the peak |
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

## A kill switch, and what to do after it fires

The regime change could not be forecast a day ahead (the repository's own
attempt is in [`docs/findings.md`](../../docs/findings.md)). Detecting it from
the strategy's own results, using only the past, is a different question, and
[`monitor.py`](monitor.py) answers it with the textbook tool: Page's CUSUM for
a drop in the mean of the portfolio's hourly net return. Its values are the
textbook ones, fixed before it was run, not searched: reference value
k = 0.5 and threshold h = 5 in units of the hourly return's standard
deviation, which is estimated on the hours before the one tested; the first 72
hours only estimate it.

| Policy after an alarm | Alarms | Portfolio on 19 April |
|---|---|---|
| No monitor | | **−52.0%** |
| Stop and stay stopped | 15 March 02:00 UTC, 12 hours after the peak | **+10.7%** |
| Stop, trade on paper, restart when the same CUSUM pointed upwards sees the paper result recover | stop 15 March, restart 19 March, stop 20 March, restart 14 April, stop 14 April | **+1.8%** |

- The monitor raised no alarm in the fortnight that paid. With neighbouring
  values (k from 0.5 to 1, h from 4 to 8) it fires 6 to 19 hours after the
  peak and stops between +4.5% and +12.2%; at k = 0.25 with h from 4 to 6 it
  fires on 6 March,
  inside the good fortnight, and stops at +5% to +7%. A more sensitive switch
  costs good days; a duller one costs bad hours.
- Restarting is the hard part. Both automatic restarts came on a bounce, and
  together they gave back most of what the stop had kept.
- This is one episode, known before the monitor was written. The values are
  the textbook's and their neighbours are shown, but it is not an
  out-of-sample test of the monitor.
- The structural-break detector in `validation/structural_breaks.py`, run on
  the same hourly series with its mean family only and a 120-hour history,
  raised a falling-mean
  alarm on 14 March 10:00, four hours before the portfolio's peak; it also
  raised one on 11 March, when the series is zero because the day is missing,
  and from 26 March one every two to three days. As a second opinion it is useful; as a
  switch it fires too often.

## Rebuild

```bash
PYTHONPATH=src python posts/btc_ath_2024/make_data.py --data data
PYTHONPATH=src python posts/btc_ath_2024/monitor.py
node posts/btc_ath_2024/render.mjs
```

The first needs the two tracks' archives (`trading-research bench fetch`) and
rewrites `data.json` and `numbers.json`; the second writes `monitor.json` from
`data.json`; the third needs Chrome and ffmpeg and
writes `artifacts/posts/btc_ath_2024.mp4` (1080 x 1080, 20 s) and a
self-contained page beside it. The video is drawn frame by frame by the page's
own `renderAt`, so it is the page, not a screen recording.
