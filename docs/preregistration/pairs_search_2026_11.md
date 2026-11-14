# Registration: pairs trading on real perpetuals, chosen on D and read once on H

| | |
|---|---|
| **Committed** | 15 November 2026, before any pair below was screened or backtested on any month |
| **What it is** | `trp search pairs`: the repository's pairs tools (`pairs/`), until now checked only on planted data, run on fifteen real perpetuals: four rules on the development months, the best chosen, read once on the held-out months |
| **What kind of number** | A measurement on past data with the choice made on D alone and H read once. Not a forecast |
| **Searched** | The four rules below, on D only |
| **How it changes** | Only by a dated amendment appended at the end, which reports both numbers |

## Why

`trp pairs` screens pairs for cointegration and walks one forward, and
`trp verify pairs` shows it finds a planted pair and nothing in random walks.
It has never been run on real instruments. This asks whether trading the most
cointegrated pairs among real perpetuals, chosen each week from the month
before, pays after four taker fills and both legs' funding.

## The rule

[`search_engine/pairs.py`](../../src/trading_research/search_engine/pairs.py):

- **Universe.** Fifteen Binance USD-M perpetuals of the carry search, BTC and
  ETH left out (`funding_carry_2026` holds them out in months that would cut
  every pair's clock): BNB, SOL, XRP, ADA, DOGE, DOT, LINK, LTC, AVAX, ATOM,
  TRX, BCH, ETC, FIL, XLM. Every pair of them: 105.
- **Bars.** Hourly perpetual closes; an hour the archive misses carries the
  last close; a bar's open is the close before it.
- **Walk.** `pairs.backtest.backtest` on each pair: the hedge, the mean and the
  sd fitted on the formation window, used unchanged on the next 168 hours,
  stepped weekly. Bands: entry `entry_z`, exit 0.5, stop 4, at most 72 hours
  held; a decision on a close fills at the next bar's open.
- **Selection.** Each week, the pairs whose formation Engle-Granger p-value
  (from the null simulated at the window's length) survives Holm's correction
  across all 105 at 5%; of those, the five with the smallest p-values are
  traded. A week with none trades nothing.
- **Costs.** Four taker fills a trade at Binance USD-M's base 5 bp plus 2 bp
  of slippage each, on each leg's notional; both legs' funding at every
  settlement held (long pays the rate, short receives it).
- **Capital.** Five slots of a fifth each; a slot's fifth covers both legs'
  notional. Monthly returns by the month a trade closes.

## The grid, searched on D

`entry_z` ∈ {2.0, 2.5} × formation ∈ {30, 60} days: four rules, chosen by the
highest monthly Sharpe ratio on D (ties by total).

## Samples

- **D**, development: January 2021 to June 2023, walked on bars up to the end
  of June 2023 only.
- **H**, held out: July 2023 to June 2025, walked once after the ledger is
  written, the first formation ending at 1 July 2023.
- **Seen before.** These perpetuals' prices were read by the carry replays;
  no pair, hedge or spread was computed on them.

## Verdict

- **Placebos.** Nineteen portfolios that trade as many pairs each week, drawn
  at random from all 105, with the same bands and costs.
- **Positive** when the chosen rule's total on H is above zero **and** above
  all nineteen placebos. Otherwise **killed**, with the first failure.
- Reported beside, with no status: every rule on D, the trades, gross and net
  per trade, and the portfolio of all 105 pairs traded every week.

## Expected outcome, stated now

**Killed: not positive.** A round trip of four fills costs about 28 bp on each
leg; crypto pairs' hourly spreads mean-revert, when they do, by less than
that within three days, and a pair that was cointegrated last month often is
not this month. The prior is a gross per trade near zero and a net of about
the cost below it.
