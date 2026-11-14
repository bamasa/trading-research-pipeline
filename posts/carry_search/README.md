# The pipeline finds a strategy that pays

A post: the research pipeline's own search, given seventeen coins and the
years 2021 to 2025, chooses how to run a funding carry from the first two and a
half years alone, and the choice earns money on the two years it never saw.
Every number is in [`experiments/results/search_carry`](../../experiments/results/search_carry/),
written by one command, `trp search carry`; the search is registered in
[`docs/preregistration/carry_search_2026_11.md`](../../docs/preregistration/carry_search_2026_11.md).
[`tests/test_carry_search.py`](../../tests/test_carry_search.py) fails if this
page and those files disagree.

## The strategy, in one paragraph

Buy a coin on the spot market and sell the same quantity of its perpetual
future. The two legs cancel: if the price goes up or down, one leg gains what
the other loses. What is left is the **funding**: every eight hours, the
perpetual's longs pay its shorts (or the other way round) to keep its price
near the spot price. In a market that wants to be long, the longs pay, and a
short perpetual collects it. That is the whole edge: not a forecast of the
price, a fee paid by the other side.

## What the pipeline did

1. **Searched sixteen ways of choosing.** Hold the 1, 2, 4 or 8 coins paying
   the most funding; measure it over the last 1 or 3 months; require it to pay
   more than the cost of getting in and out, or not. Rebalance every month; a
   coin kept pays no new fees.
2. **Chose on the past alone.** It replayed all sixteen on January 2021 to June
   2023 and picked the best risk-adjusted one: **hold the single coin that paid
   the most funding over the last three months.** All sixteen were positive on
   those months, so the choice is not one lucky setting among losers.
3. **Locked the choice.** The choice and a fingerprint of the development
   results went into a ledger before anything later was read.
4. **Read the next two years once.** July 2023 to June 2025, at Binance's
   base fees (spot 10 bp, perpetual 5 bp a side) and 2 bp of slippage a leg.

## What it earned on the two years it never saw

| | |
|---|---|
| Total, July 2023 to June 2025 | **+15.8%** |
| A year | **+7.6%** |
| Worst drawdown, month ends | **−0.09%** |
| Losing months | 3 of 24, each under 0.1% |
| By year | 2023 (from July) +3.1%, 2024 +11.9%, 2025 (to June) +0.5% |
| Against a random coin each month | beat all 19 placebos (they made +0.5% to +7.3%) |

The coin it held was FIL for seventeen of the twenty-four months, then LTC,
LINK, ETC and DOGE.

## How to read it

- **This is a backtest on past data.** It says what the pipeline would have
  done, not what it will do. Funding pays most in rising markets: 2021 and
  2024 were good years, the first half of 2025 was flat.
- **The held-out years were not unseen by the author.** An earlier replay of a
  neighbouring rule had read them. The pipeline's choice was made on the first
  two and a half years alone; the grid it chose from was written by someone who
  had seen the later years. Both are stated in the registration.
- **One coin at a time is concentrated.** A coin's own events, a delisting, a
  funding spike against the short, the exchange itself, are not spread across
  others, and no backtest prices them.
- **The capital counts both legs, unlevered.** With leverage the return per
  dollar rises and so does the risk of the short leg being liquidated in a
  spike.

## LinkedIn text

> **I asked my research pipeline to find a strategy. It found one that paid on two years it never saw.**
>
> The pipeline searched sixteen ways of running a funding carry on seventeen crypto coins: buy the coin, short its perpetual future, and collect the funding the longs pay the shorts every eight hours. No price forecast, just a fee paid by the other side.
>
> It chose using only January 2021 to June 2023, locked the choice in a ledger, and then read July 2023 to June 2025 once.
>
> Result on the unseen years, after fees: +15.8%, about +7.6% a year, worst month-end drawdown −0.09%. Holding a random coin each month instead made +0.5% to +7.3%; the pipeline's choice beat all nineteen tries.
>
> It is a backtest, it is concentrated in one coin at a time, and a rising market pays the most. But it is the first strategy in this project that pays after fees on months its choice never touched.
>
> Code, registration and every number: github.com/bamasa/trading-research-pipeline/tree/main/posts/carry_search
