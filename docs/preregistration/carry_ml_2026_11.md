# Registration: machine learning choosing the carry's coin (C1, C2)

| | |
|---|---|
| **Committed** | 20 November 2026, before any model below was fitted, and before any price or funding rate of the held-out months (July 2025 to July 2026) was fetched |
| **What it registers** | Five ways of choosing the one coin the funding carry holds each month, two rules and three fitted models, their settings fixed here; a walk-forward replay on the carry search's months, reported as a measurement; and two statuses read once on thirteen months no study here has read |
| **Searched** | **Nothing.** Every setting below is written down before the first fit |
| **How it changes** | Only by a dated amendment appended at the end, which cannot change a status already read |

## Why

The carry search ([`carry_search_2026_11`](carry_search_2026_11.md)) found that
the carry pays and that its choice among sixteen rules does not survive the
search's size: PBO 0.69, and on its held-out months the chosen rule ranked 14th
of 16 while all 16 were positive
([`carry_overfitting`](../../experiments/results/carry_overfitting/README.md)).
The weak part is the choice of coin. The chosen rule, the one the live journal
follows, forecasts next month's funding with the mean of the last three. This
study asks whether a model fitted on the coins' own history forecasts it
better, and whether a better forecast chooses a better coin.

The trade is monthly, so its cost is a few basis points a month, not twelve a
trade as at ten minutes ([`xs_deep`](xs_deep_forward_2026_12.md)): if a
forecast is better, the cost does not hide it.

## Data

Binance's public archives for the seventeen coins of the carry search
(`strategies/carry.py`, `UNIVERSE`): hourly spot and USD-M perpetual closes and
every funding settlement, by [`strategies/carry.load`](../../src/trading_research/strategies/carry.py).
BTC and ETH are held out by [`funding_carry_2026`](funding_carry_2026.md) from
December 2020 to June 2021, February to June 2024 and from July 2025 on; in
those months they are not candidates and their data is not fetched.

## The forecast

At the start of each month, for each candidate coin, from data before the
month alone:

| feature | |
|---|---|
| `f1`, `f3` | funding a day, bp, over the last month and the last three |
| `f_trend` | `f1 - f3` |
| `f_sd` | standard deviation of the daily funding, bp, over the last three months |
| `f_pos` | share of the last three months' settlements above zero |
| `basis_1` | mean of `perp / spot - 1`, bp, over the last month's hours |
| `basis_end` | the same over the last month's last 24 hours |
| `ret_1`, `ret_3` | the spot's log return over the last month and three |
| `vol_1` | standard deviation of the spot's hourly log return over the last month, bp |
| `rank_f3` | `f3`'s rank among the month's candidates, from 0 to 1 |

**Target:** the coin's funding a day, bp, over the month.

## The five choosers

Each holds the one coin it scores highest, long spot and short the perpetual,
the whole account, rebalanced monthly through the search's engine
(`strategies/carry.replay`): Binance Regular fees, 2 bp of slippage a leg and
side, a coin kept pays no round trip.

| name | score |
|---|---|
| `rule_lb3` | `f3`: the search's chosen rule and the journal's, `top1-lb3-floor_none` |
| `rule_lb1` | `f1` |
| `ridge` | linear, the eleven features standardised, penalty 1 |
| `lightgbm` | 200 rounds, learning rate 0.05, 7 leaves, at least 20 rows a leaf, seed 0 |
| `mlp` | two hidden layers of 32, ReLU; Adam, learning rate 0.001, weight decay 0.001, 300 full-batch epochs, seed 0, on the CPU |

The models are refitted every month on every (coin, month) row whose month has
ended, pooled across coins, with the features standardised on those rows. A
model chooses once it has at least twelve months of rows; the replays and both
statuses compare the five on the months where all five choose.

## Blocks

- **Development, D:** January 2021 to June 2025, read before by the carry
  search and the walk-forward replay. A walk-forward replay of the five here,
  compared from January 2022: **a measurement**, reported with no status.
- **Held out, H: July 2025 to July 2026**, thirteen months, fifteen coins (BTC
  and ETH held out). Nothing after June 2025 has been read by any study for
  these coins; the live journal read August to October 2026's funding, after
  H. The models fit walk-forward into H on months already ended, as in D. H is
  fetched only after this registration and the code freeze are committed, and
  read once by `experiments/carry_ml.py read`, which refuses a second read.

## C1, C2

Months are the observations; one-sided *t*-tests over H's thirteen months,
Holm's adjustment over the three models within each.

- **C1, a model chooses a better coin.** For each of `ridge`, `lightgbm`,
  `mlp`: the month's return of its replay minus `rule_lb3`'s. **Passes** when
  the mean is positive and the Holm-adjusted *p* is below 0.05. Otherwise
  **killed**.
- **C2, a model forecasts the funding better.** A month's forecast quality is
  the Spearman correlation across the month's candidates of the score and the
  realised funding. For each model: its correlation minus `rule_lb3`'s.
  **Passes** as C1. Otherwise **killed**.

Reported beside, with no status: every chooser's total return, return a year,
Sharpe ratio and worst month-end drawdown on H; its coins; nineteen placebos
holding a coin drawn at random each month; and how often each model holds
`rule_lb3`'s coin.

## Expected outcome, stated now

**C1 and C2: all three killed.** Funding is persistent and the three-month
mean is already a good forecast; the models have a few hundred to a thousand
rows. They should mostly hold the rule's coin, and thirteen months resolve
only a large difference. **`rule_lb3` positive on H**, low single digits a
year; any carry earns when funding is mostly positive.

## Amendment, 20 November 2026: code freeze, before H is fetched

The study's code is [`src/trading_research/carry_ml/`](../../src/trading_research/carry_ml/__init__.py),
written after the registration above and at its settings; the replay and the read are
[`experiments/carry_ml.py`](../../experiments/carry_ml.py). The four sources are frozen by sha256
there (`FROZEN`): `fetch` and `read` refuse if any differs, and `read` writes its result once.
Nothing departs from the registration.

### What was seen on development, after the code was written

The walk-forward replay, January 2022 to June 2025, 42 months
([`experiments/results/carry_ml/`](../../experiments/results/carry_ml/README.md)), a measurement:

- `rule_lb3` +19.8% (+5.3% a year, Sharpe 1.91, worst month-end drawdown −0.1%); `rule_lb1`
  +11.8%; `ridge` +8.9%, `lightgbm` +5.0%, `mlp` +7.1%, with drawdowns of −3% to −5%;
- forecast correlation: `rule_lb3` 0.71, `rule_lb1` 0.69, `ridge` 0.61, `lightgbm` 0.52,
  `mlp` 0.52; every model's difference from `rule_lb3` negative with *t* from −3.5 to −6.3;
- the models hold `rule_lb3`'s coin in 14% to 29% of months and change coin 29 to 35 times
  against its 12;
- nineteen random-coin placebos: −1.6% to +4.5%.

None of this changes a setting, a block or a status. The expected outcome stands, and on these
numbers C1 and C2 lean killed with more confidence than when it was written.

## Results, read once 20 November 2026

From the frozen code, by `experiments/carry_ml.py read`, after the code freeze above was
committed; the result is [`experiments/results/carry_ml_heldout/`](../../experiments/results/carry_ml_heldout/README.md)
(`result.json`). H: July 2025 to July 2026, thirteen months, fifteen coins, every day of every
coin on disk.

**C1: killed for all three.** The month's return minus `rule_lb3`'s: `ridge` −0.076% (*t* −3.81),
`lightgbm` −0.169% (*t* −3.54), `mlp` −0.061% (*t* −1.86); every Holm *p* 1.0.

**C2: killed for all three.** Forecast correlation minus `rule_lb3`'s: `ridge` −0.061 (*t* −1.74),
`lightgbm` −0.226 (*t* −3.43), `mlp` −0.104 (*t* −2.45).

Beside, with no status:

- `rule_lb3` +1.5% over the thirteen months (+1.4% a year, Sharpe 3.9, worst month-end drawdown
  −0.03%, two losing months), holding LTC, XRP, ADA, then LINK from January 2026; `rule_lb1`
  +0.9%; `ridge` +0.5%, `mlp` +0.7%, `lightgbm` −0.7%;
- forecast correlation: `rule_lb3` 0.68, `rule_lb1` 0.65, `ridge` 0.62, `mlp` 0.58, `lightgbm` 0.46;
- the models held `rule_lb3`'s coin in 8% to 15% of months;
- the nineteen random-coin placebos all lost, −3.7% to −1.4%: on H, choosing by funding is what
  keeps the carry positive.

Against the expectation stated in advance: C1 and C2 killed, as expected; `rule_lb3` positive in
low single digits a year, as expected. The level is a quarter of D's: funding was lower on these
months for every coin.
