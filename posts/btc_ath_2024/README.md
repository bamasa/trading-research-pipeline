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

## Rebuild

```bash
PYTHONPATH=src python posts/btc_ath_2024/make_data.py --data data
node posts/btc_ath_2024/render.mjs
```

The first needs the two tracks' archives (`trading-research bench fetch`) and
rewrites `data.json` and `numbers.json`; the second needs Chrome and ffmpeg and
writes `artifacts/posts/btc_ath_2024.mp4` (1080 x 1080, 20 s) and a
self-contained page beside it. The video is drawn frame by frame by the page's
own `renderAt`, so it is the page, not a screen recording.
