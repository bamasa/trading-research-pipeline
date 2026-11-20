# Registration: ML and DL ranking a hundred perpetuals a week (W1, W2)

| | |
|---|---|
| **Committed** | 20 November 2026, before any daily bar or funding rate of the universe below was fetched, and before any model was fitted |
| **What it registers** | Eight rankers of the week ahead across the hundred most traded Binance USDT perpetuals, three rules and five fitted models; a long-short portfolio rebalanced weekly at taker fees with funding; a walk-forward replay on 2021 to 2024, reported as a measurement; and two statuses read once on January 2025 to October 2026 |
| **Searched** | **Nothing.** Every setting below is written down before the first fit |
| **How it changes** | Only by a dated amendment appended at the end, which cannot change a status already read |

## Why

Every model this repository has fitted lost to its costs where it traded
often: at ten minutes the edge per trade was a fraction of a basis point
against twelve ([`xs_deep`](xs_deep_forward_2026_12.md)). A week moves a coin
by several hundred basis points against a cross-section, and a weekly round
trip costs fourteen. The author's Numerai Crypto model, a tree ensemble on
daily price and volume features, ranked about three hundred tokens' next
twenty days with a mean correlation of 0.144 on August 2024 to October 2026,
about as well as the negated 60-day volatility alone (0.132). The question
here is whether that kind of ranking survives as a trade on perpetuals, at
their fees and funding, and whether a fitted model, deep or not, adds to the
plainest rule.

**What was known before this was written.** The Numerai read above covers most
of H and says a low-volatility effect was present then; this study's H is not
blind to it. The carry and cross-venue studies read fifteen of these coins'
hourly prices and funding from July 2025 to July 2026. No study here has read
the daily cross-section of the universe below.

## Data and universe

Binance USD-M perpetuals ending in USDT, from `data.binance.vision`'s monthly
archives, delisted ones included: daily klines (open, close, quote volume)
and every funding settlement. BTC and ETH are left out (held out by
[`funding_carry_2026`](funding_carry_2026.md)), and so are the stablecoins
quoted in USDT (USDC, FDUSD, TUSD, BUSD, USDP, DAI).

- **Decision:** every Monday 00:00 UTC, from daily bars to the Sunday's close.
- **Universe at a decision:** the hundred coins with the highest mean daily
  quote volume over the previous 30 days, among those with at least 120 days
  of bars before it.
- **Target:** the coin's log return from the Monday's open to the next
  Monday's open, minus the universe's mean; models are fitted on its
  cross-sectional z-score, clipped at 3.
- **Training rows:** every day (not only Mondays) whose seven-day target ended
  before the decision, with the same universe rule applied that day.

## The eight rankers

| name | kind | score |
|---|---|---|
| `momentum` | rule | the 28-day relative return |
| `reversal` | rule | minus the 7-day relative return |
| `low_vol` | rule | minus the 28-day volatility of the daily relative return |
| `ridge` | ML | twelve features, standardised, penalty 10 |
| `lightgbm` | ML | the twelve; 300 rounds, learning rate 0.03, 15 leaves, at least 200 rows a leaf, 80% of features a tree, seed 0 |
| `mlp` | DL | the twelve; `xs_deep`'s MLP (two layers of 64) |
| `tcn` | DL | the last 60 days of four series per coin; `xs_deep`'s causal TCN |
| `xs_transformer` | DL | the same encoder under attention across the day's coins; `xs_deep`'s |

The twelve features: the relative return over 1, 7, 28 and 60 days; the
volatility of the daily relative return over 7, 28 and 60 days; the universe's
return over 7 and 28 days; the log of the 7-day mean quote volume over the
60-day mean; the mean funding a day over 7 and 28 days, bp. The four series:
the daily relative return, the universe's daily return, the log quote volume
against its 60-day mean, the day's funding in bp. Inputs are standardised on
the training rows and clipped at 5.

The networks are [`xs_deep/nets.py`](../../src/trading_research/xs_deep/nets.py)'s,
imported unchanged, width 32; Adam at 0.001, weight decay 0.0001, 6 epochs,
batches of 64 days, seed 0, on the CPU. Every model is refitted every fourth
Monday on all training rows before it; between refits the last fit scores.
Models choose from the first Monday with 365 days of training rows.

## The portfolio

At each decision, long the ten coins ranked highest and short the ten ranked
lowest, equal weight, half the account long and half short, unlevered. Each
coin is held to the next Monday's open. A coin's leg pays Binance's USD-M
taker fee, 5 bp, plus 2 bp of slippage at each side it trades; a coin kept on
the same side pays nothing. Funding is paid by a long and received by a short
at every settlement, on the leg's notional. A coin delisted while held is
closed at its last close.

## Blocks

- **Development, D:** daily data from January 2020; decisions from July 2021 to
  December 2024. A walk-forward replay, every model fitted only on rows before
  each decision: a measurement, with no status.
- **Held out, H:** decisions from 6 January 2025 to 26 October 2026, about 95
  weeks. Fetched only after the code freeze is committed and read once by
  `experiments/xs_weekly.py read`, which refuses a second read.

## W1, W2

Weeks are the observations; one-sided *t*-tests; Holm's adjustment over the
five models within each.

- **W1, a model's portfolio pays.** For each of `ridge`, `lightgbm`, `mlp`,
  `tcn`, `xs_transformer`: **passes** when its mean weekly net return on H is
  positive with Holm-adjusted *p* below 0.05. Otherwise **killed**.
- **W2, a model beats the plainest rule.** For each model: its weekly net
  return minus `low_vol`'s. **Passes** as W1. Otherwise **killed**.

Reported beside, with no status: every ranker's total, return a year, Sharpe
ratio, worst drawdown; the result split into price, funding and cost; the
mean weekly rank correlation of each ranker's scores with the target; the
turnover.

## Expected outcome, stated now

**W1: uncertain; the tree models most likely to pass.** A low-volatility
ranking of crypto was present in H (above), and a week is long enough for a
ranking with a correlation of a few hundredths to clear fourteen basis points.
Against it: the short side holds the most volatile coins, which squeeze, and
whose funding is often negative, so shorts pay it. **W2: killed for all five.**
The models should mostly learn the volatility ranking, and 95 weeks resolve
only a large difference.
