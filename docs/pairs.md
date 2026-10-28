# Pairs: a cointegration screen and a walk-forward backtest on bars

Statistical arbitrage on two instruments: when the log price of one is a stable
linear function of the other's plus a residual that keeps coming back, a large
residual is a bet that it will shrink. `trp pairs` finds candidate pairs on one
window, then trades one pair walk-forward on the bars after, both legs charged
for every fill. The code is `src/trading_research/pairs/`; the synthetic pair
and its check are `synthetic/pairs.py` and `synthetic/pairs_check.py`.

```bash
trp pairs synthetic --out data/synthetic/pairs              # a planted pair and its truth.json
trp pairs screen   --root data/synthetic/pairs -s SYNTHAUSDT -s SYNTHBUSDT --start 2020-01-06 --end 2020-02-04
trp pairs backtest --root data/synthetic/pairs -s SYNTHAUSDT -s SYNTHBUSDT
trp verify pairs                                            # screen and backtest against the planted answer
```

Each command writes into `--out` (`artifacts/pairs/screen`,
`artifacts/pairs/backtest` by default): `page.md`, the tables as csv
(`screen.csv`; `trades.csv`, `days.csv`, `windows.csv`) and `summary.json`.

## What it computes

**Hedge ratio.** Least squares of `log a` on `log b` with an intercept, on the
formation window only: `log a = alpha + beta * log b + e`. The pair is ordered:
`a` is the first `--symbol`, regressed on the second.

**Cointegration.** Engle-Granger: an augmented Dickey-Fuller regression on the
residual `e`, without a constant (the intercept is in the hedge), one lag of
the residual's change by default (`--lags`). The residual was fitted to look
stationary, so its t does not follow the Dickey-Fuller table. The p-value is the
share of a null simulated by the same code at the same length (two independent
Gaussian random walks, the same regression, 4,000 draws, a fixed seed; its
resolution is 1/4,001). The tests check that null against MacKinnon's (2010)
response-surface critical values for two series, which the page also prints
(`critical_5`). No statistics package is needed.

**Half-life.** The AR(1) of the residual, `e(t+1) = c + phi * e(t)`, gives
`log(2) / -log(phi)` bars; infinite when `phi` is not in (0, 1).

**Correlation** of the two legs' bar log returns, reported beside: two legs can
correlate closely and still drift apart, which is why the screen tests the
residual rather than the correlation.

**Screen.** Every pair among the symbols given, on the formation window
(`--start`..`--end`) and nothing else: correlation, beta, Engle-Granger t and
p-value, half-life, the residual's standard deviation in bp, and the round trip
of both legs (four taker fills) in bp. Because many pairs are tested, the
p-values are corrected across all of them with Holm's step-down procedure at a
family-wise 5% (`--alpha`); a pair passes only on its adjusted p-value. Pairs
are ranked passing first, then by adjusted p-value, then by the t. A residual
whose standard deviation is below the round trip (`sd_over_cost` < 1) cannot pay
for itself however reliably it reverts.

**Backtest.** Rolling walk-forward: fit on `--formation` bars (default 720, 30
days of hours), trade the next `--trading` bars (168, a week), step by
`--trading` and fit again, so each bar is traded by exactly one fit made on bars
that closed before it. The hedge, the residual's mean and its standard
deviation are estimated on formation only. At each close the z-score of the
residual is compared with the bands: enter at `|z| >= --entry` (2) against the
deviation, leave at `|z| <= --exit` (0.5), stop at `|z| >= --stop` (4) against
the position, leave after `--max-hold` bars (72), and leave at the end of the
trading window. After a stop, a time exit or a window end the rule waits for
`|z|` to come back inside the entry band before it enters again. `--gate-p`
trades a window only when its formation p-value is at most the given level;
without it every window is traded, which is how the rule itself is tested.

A test perturbs every price from a given bar on and checks that every decision
and every fit before that bar is unchanged (`tests/test_pairs_backtest.py`);
`verify pairs` repeats it on each path.

## Assumptions

- **Bars, next-bar fills.** A decision is taken on the close of bar `t`
  (`timestamp` is the bar close in the data contract) and filled at the **open
  of bar `t + 1`**, both legs at once. The close that triggered a decision is
  never the price paid.
- **Accounting.** In bp of the first leg's notional; the second leg's notional
  is `|beta|` of it, on the opposite side when beta is positive. A trade's gross
  is `side * (r_a - beta * r_b)`, each `r` the leg's simple return from entry
  fill to exit fill. A trade's day is the UTC day of its exit fill.
- **Costs.** Four taker fills a trade, each charged the taker fee and a
  slippage on its own leg's notional, as `backtest.costs.TakerCosts` charges one
  side: `2c` on leg `a` and `2c|beta|` on leg `b`, `c` the cost of a fill. The
  fee comes from the repository's tables: Binance USD-M's standard taker fee
  (`BINANCE_UM_TAKER_BP`, 5 bp) by default, since the public bars are Binance's;
  `--tier vip1` reads a row of Bybit's linear schedule through `fees.tier`
  (`trp fees` lists them); `--fee-bp` states one. Bars carry no quoted spread,
  so `--slippage-bp` (0.5 by default) stands for everything between the open and
  the price paid, the half-spread included: on a thin pair it should be larger.
- **Statistics.** Per trade and per day: gross, cost and net; the
  day-clustered t (`evaluation.significance.assess`, the mean per trade within
  each day, then the t of those day means); the share of days with a positive
  net. The headroom restates the gross as `walkthrough.verdict.headroom` does
  for the search: the cost per trade at which the net is zero, and the cost per
  fill that gives it. The fee does not change any decision, so that break-even
  is exact for this rule.
- Trades within one trading window share a fit, so they are not independent
  even across days; the windows table shows each fit, and a result carried by
  one window should be read as one observation.

## What it does not model

- **Borrow** on a short spot leg: shorting spot needs a margin loan, whose rate
  and availability are not charged. On spot data the short leg is a fiction
  unless that is added.
- **Funding** on perpetuals: each leg of a pair of perpetuals pays or receives
  funding every settlement while held. The repository can fetch funding
  (`trp data download --kind fundingRate` for Binance, `data/bybit_funding.py`
  for Bybit) but the pairs backtest does not include it; for holds of hours it
  is small against the fees, for days it is not.
- Queue, partial fills, latency between the legs, margin, liquidation, the
  venue's minimum sizes and the market impact of either leg.
- Selection across pairs: the backtest trades the pair it is given. A pair
  chosen because it passed a screen on the same bars it is then backtested on
  is chosen on its answer; screen on one window and backtest on bars after it.

## On public data

Hourly bars of USD-M perpetuals, fetched a month at a time:

```bash
trp data download --market futures-um --kind klines --interval 1h --symbol ETHUSDT --start 2024-01-01 --end 2024-06-30
trp data download --market futures-um --kind klines --interval 1h --symbol BTCUSDT --start 2024-01-01 --end 2024-06-30
trp pairs screen --root data/raw -s ETHUSDT -s BTCUSDT --start 2024-01-01 --end 2024-01-31
trp pairs backtest --root data/raw -s ETHUSDT -s BTCUSDT --start 2024-02-01 --end 2024-06-30
```

`trp data download` writes `data/raw/<SYMBOL>/klines-1h/<day>.parquet`, which
is what `--root data/raw` reads. Symbols with different listings or a missing
hour are refused with the first bar that differs: the legs are never stitched
across a gap. Read the study rules in `CLAUDE.md` before running on a range a
registered study holds out.

## Relation to the `spread` plugin

`strategies/spread.py` and its plugin `spread` (`trp strategies show spread`)
trade one pair inside the plugin framework: the traded instrument against its
first peer on the book's mids, the hedge and the residual's AR(1) fitted once on
the history the framework hands it, the z-score scaled by the AR(1)'s
innovation, and a band rule whose exits the framework's executor handles. The
pairs package calls the same `fit_hedge` and `fit_ou` (there is one hedge ratio
in the repository), and adds what a pair study on bars needs and the plugin does
not have: the cointegration test with its null, the screen across many pairs
with Holm's correction, the walk-forward refit, the z-score scaled by the
residual's own standard deviation on formation, stops and a maximum hold,
next-bar fills and the cost of both legs. The plugin is unchanged.

## Checked against a planted answer

`trp verify pairs` (also part of `trp verify all`; `--quick` runs one path)
generates, per path, a planted pair (`log a = 0.5 + 0.8 log b + s`, `s` an
AR(1) with half-life 6 bars and sd 60 bp), two independent random walks of the
same volatility, and a pair whose residual becomes a random walk halfway. It
requires: the screen ranks the planted pair first and passes it after Holm on
every path, and passes the independent walks on none; the hedge ratio within
0.05 and the half-life within 25%; no decision moves when later prices are
perturbed; the planted backtest positive before costs (day-clustered t >= 3);
on independent walks nothing before costs (|t| < 3) and nothing significant after
costs (t < 3); after the break nothing significant after costs. On the
independent walks the mean net per trade is reported beside; with a few dozen
trades on random-walk spreads its sign is noise, so the criterion is the
repository's significance bar rather than the sign. On the default seed it
passes in about a second.

Nothing here is evidence that any two real instruments cointegrate.
