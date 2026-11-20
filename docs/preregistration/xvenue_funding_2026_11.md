# Registration: the funding spread between Binance and Bybit (V1, V2)

| | |
|---|---|
| **Committed** | 20 November 2026, before any Bybit price or funding rate below was fetched, beyond three rows of FILUSDT's first hours (29 June 2021) and its listing record, read to check the endpoints |
| **What it registers** | One rule that holds the same coin's perpetual on both venues, short where the funding is higher and long where it is lower; a replay on development months, reported as a measurement; and two statuses read once on thirteen held-out months |
| **Searched** | **Nothing.** The rule's settings are written here before any spread is computed |
| **How it changes** | Only by a dated amendment appended at the end, which cannot change a status already read |

## Why

The one thing this repository found that pays is the funding carry
([findings](../findings.md)), and the carry's risk is the coin's basis against
spot. The same coin's perpetual trades on several venues, each with its own
funding. Shorting it where the funding is higher and buying it where it is
lower collects the difference with no spot leg and no exposure to the coin's
price, only to the gap between two perpetuals' prices. If the gap in funding
is persistent and wider than two venues' round trip, the position pays. It
trades once a month at most, so the cost is spread over the month.

## Data

The fifteen coins of the carry search's universe (`strategies/carry.py`,
`UNIVERSE`) other than BTC and ETH, which `funding_carry_2026` holds out on
Binance. On each venue, hourly closes of the USDT perpetual and every funding
settlement:

- **Binance** USD-M, from `data.binance.vision`, by `strategies/carry.load`;
- **Bybit** linear, from the public REST endpoints (`/v5/market/kline`,
  interval 60; `/v5/market/funding/history`), without credentials.

Every coin was listed on both venues by October 2021.

## The rule

At each month's first hour, for each coin, its **spread**: Binance's funding a
day minus Bybit's, bp, over the previous three months, where both venues have
every settlement.

- **Candidates:** coins whose spread's size exceeds the pair's round trip
  spread over thirty days, 29 / 30 bp a day.
- **Held:** the four candidates with the largest size, each short on the venue
  whose funding was higher and long on the other, equal quantity; fewer than
  four leaves the slots in cash.
- **Capital:** the account split in four slots; a slot's two legs each carry
  half of it, unlevered.
- **Cost:** a taker on both venues, Binance USD-M 5 bp and Bybit linear 5.5 bp
  a side, plus 2 bp of slippage a leg and side: 29 bp a round trip on one leg's
  notional. A pair kept in the same direction pays nothing; one dropped or
  reversed pays its round trip.
- **Result each hour:** the long leg's price change minus the short leg's, on
  the quantity, plus the funding the short leg receives and minus the funding
  the long leg pays at each venue's settlements.

## Blocks

- **Development, D: January 2022 to June 2025.** Binance's side was read by the
  carry studies; Bybit's has not been read for these coins. A replay of the
  rule, reported as a measurement with no status.
- **Held out, H: July 2025 to July 2026**, thirteen months. Bybit's side has
  not been fetched. Binance's side was read on 20 November 2026 by
  [`carry_ml_2026_11`](carry_ml_2026_11.md) (each coin's funding and its
  carry's return); no spread between venues was computed on any month of it.
  H is fetched after the code freeze, read once by
  `experiments/xvenue_funding.py read`, which refuses a second read.

## V1, V2

- **V1, the spread pays.** **Passes** when the rule's mean monthly return on H
  is positive with a one-sided *t* *p* below 0.05, months as observations.
  Otherwise **killed**.
- **V2, the direction matters.** Nineteen placebos hold the rule's coins each
  month with the direction drawn at random. **Passes** when the rule's total on
  H is above all nineteen. Otherwise **killed**.

Reported beside, with no status: total, return a year, Sharpe ratio, worst
month-end drawdown; the result split into funding, price gap and cost; the
months with no pair held; each coin's mean spread.

## Expected outcome, stated now

**V1 killed.** Two venues' funding on the same coin move together; the spread
is a fraction of a basis point a day for most coins and months, and few
months should clear 29 bp a round trip by much. **V2 passes** if the rule holds
anything: a spread's sign that has held three months should mostly hold the
next.

## Amendment, 20 November 2026: code freeze, before H is fetched

The rule is [`src/trading_research/xvenue/engine.py`](../../src/trading_research/xvenue/engine.py),
Bybit's closes come from [`data/bybit_klines.py`](../../src/trading_research/data/bybit_klines.py),
and the replay and read are [`experiments/xvenue_funding.py`](../../experiments/xvenue_funding.py),
which freezes the three sources by sha256 (`FROZEN`); `fetch` and `read` refuse if any differs.

**A fault found on development, corrected before the freeze.** Binance's archive has empty
FILUSDT perpetual files for 26 to 28 February 2022. The first replay carried Binance's last
price through the gap, booked the gap against Bybit's moving price as a loss of 1.6% of the
account, and never booked its return, because each month's marking started at the month's first
hour and a pair closed at a month's start was not marked to it. The engine now marks a pair held
across months from the previous month's last hour and marks a pair closed at a month's start to
that hour; two tests check both. The rule, its settings and the blocks are unchanged.

The carry search's engine, `strategies/carry.replay`, which is frozen with earlier studies,
marks its coins the same way: one hour of basis change at each month's boundary is not booked
for a coin kept or dropped. On spot against perpetual the hour's change is small; its effect on
those studies' numbers is not measured here.

### What was seen on development, after the correction

January 2022 to June 2025 ([`experiments/results/xvenue_funding/`](../../experiments/results/xvenue_funding/README.md)),
a measurement: **+3.0%** (+0.85% a year, Sharpe 0.94, worst month-end drawdown −1.1%); funding
+4.7%, price gap +0.2%, cost −1.8%; mean +0.072% a month, *t* +1.76. Bybit's funding was the
higher on all fifteen coins on average, by 0.4 to 1.3 bp a day, so the rule was mostly long on
Binance and short on Bybit. The nineteen random-direction placebos lost −4.9% to −0.9%.

None of this changes a setting, a block or a status. The expected outcome is unchanged.
