# Registration: funding carry, chosen month by month from the past alone, January 2021 to June 2025

| | |
|---|---|
| **Committed** | 13 November 2026, before any funding rate or price of the universe below was fetched or read for this study |
| **What it is** | A walk-forward replay: if the pipeline had been switched on at the start of 2021 and left to choose its own instruments every month from what it had seen, what would it have earned? |
| **What kind of number** | A measurement on past data, read once, with the rule fixed here. Not a forecast; not a confirmation |
| **Searched** | **Nothing.** Every value below is fixed now |
| **How it changes** | Only by a dated amendment appended at the end, which reports both numbers |

## Why

`funding_carry_2026` registers an always-on carry on two coins and reads its
verdict in 2027. A measurement on its development months found the carry
positive on every range (`experiments/funding_carry_dev.py`), but an always-on
position on a fixed coin is not what a pipeline does. A pipeline looks at
what it has seen, picks where to trade, and moves on when the reason goes.
This replay asks whether that, done mechanically from the past alone, would
have earned money.

## The rule

- **Universe.** Seventeen coins with a Binance spot USDT pair and a USD-M
  perpetual listed before 2021: BTC, ETH, BNB, SOL, XRP, ADA, DOGE, DOT, LINK,
  LTC, AVAX, ATOM, TRX, BCH, ETC, FIL, XLM.
- **Decision.** At 00:00 UTC on the first day of each month from January 2021
  to June 2025, using only the previous calendar month: each coin's funding
  **per day** (the sum of its settlements over the month, divided by the
  month's days).
- **Choice.** The four coins with the highest funding per day, among those
  whose funding per day is above **1.27 bp** (a round trip's 38 bp spread over
  thirty days). Fewer than four qualify: the rest of the capital stays in cash
  at 0%.
- **Position.** Each chosen coin gets a quarter of the capital, as a
  delta-neutral carry: long spot, short the perpetual in equal quantity, the
  perpetual unlevered, so a coin's slot is split half on each leg. It is held
  for the month. A coin chosen again the next month is kept without a round
  trip; a coin dropped is closed; a new coin is opened.
- **Costs.** Binance Regular, no BNB discount: spot 10 bp and perpetual 5 bp
  a side, and 2 bp of slippage a leg and side, as `funding_carry_2026`
  registers. Funding is received or paid at every settlement, on the
  perpetual's quantity at its price at the settlement hour. The basis is
  carried hour by hour.
- **Variant with the kill switch** (reported beside, no status of its own):
  each position runs
  [`validation/kill_switch.py`](../../src/trading_research/validation/kill_switch.py)
  at its textbook values on its hourly result (funding at its settlement hour,
  the basis's change), from the hour it opened; when it fires, the position is
  closed at the next hour, paying the exit, and that coin is not reopened
  before the next decision.

## What is held out, and so not read

`funding_carry_2026` holds BTC and ETH out from December 2020 to June 2021,
from February to June 2024 and from July 2025 on. Their data for those months
is not fetched; in a month whose decision or holding needs it, BTC and ETH are
not candidates. Nothing after June 2025 is read for any coin. No other study
holds out a coin of this universe in these years.

## What is reported

The monthly return on the whole capital, the equity curve, the total and
yearly returns, the worst drawdown, the months in cash, the coins chosen each
month, and two baselines: holding BTC spot, and the always-on carry on BTC
alone, over the months BTC is readable.

## Expected outcome, stated now

Positive over the whole span, most of it in 2021, when funding was high, with
a flat or slightly negative 2022 as funding fell and some months chose cash.
A yearly return in single digits over the span, and a worst drawdown of a few
percent: a carry earns the funding and gives some of it back in basis and fees,
and the threshold keeps it out of the worst months.

## Results, read once 13 November 2026

[`experiments/results/carry_walkforward/results.md`](../../experiments/results/carry_walkforward/results.md),
from the code committed before the read. A first run stopped before writing
anything, on hours the price archive misses; the fix carries the last close
known from the whole history (committed, then the read).

| | total | a year | worst drawdown, month ends | losing months | months wholly in cash |
|---|--:|--:|--:|--:|--:|
| **the rule** | **+47.8%** | **+9.1%** | −0.6% | 10 of 54 | 1 |
| the rule with the kill switch | +30.3% | +6.1% | −0.4% | 11 | 1 |
| BTC carry, always on, where readable | +13.9% | +2.9% | −0.1% | 3 | 13 unreadable |

By year: 2021 **+29.7%**, 2022 +0.7%, 2023 +3.5%, 2024 +9.2%, January to June
2025 +0.2%. The worst month was May 2022 (−0.42%, the LUNA collapse); the best
four were in the first four months of 2021. The coins chosen most often were
FIL, LINK, XRP, LTC and DOGE; BTC was rarely among the four highest.

As stated in advance: positive over the span, most of it in 2021, 2022 flat.

- **The kill switch hurt here.** A carry's hourly result is mostly the basis's
  noise around a steady funding income; the switch read the noise as a lost
  edge, closed positions that kept paying and paid to reopen them the next
  month: +30.3% against +47.8%. A kill switch suits a strategy whose edge can
  vanish within the hour, as the reversion rule's did, not one paid on a
  schedule.
- **What this is.** A replay on past data, the choice made each month from the
  month before alone, at an account's base fees. The drawdown is measured at
  month ends; within a month the basis moves the position's value further. The
  perpetual is unlevered, so the capital is twice each position's notional, and
  the result carries the exchange's own risk, which no backtest prices.
