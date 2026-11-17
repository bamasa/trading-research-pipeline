# Registration: the automatic search for a funding carry (`trp search carry`)

| | |
|---|---|
| **Committed** | 13 November 2026, with the search's code, before the search was run |
| **What it registers** | The pipeline's own search for a funding carry: what it searches, how it chooses on the development months, what it reads once on the held-out months, and what counts as a result |
| **Searched** | Sixteen rules, on the development months only, by the rule below |
| **How it changes** | Only by a dated amendment appended at the end |

## The search

[`search_engine/carry.py`](../../src/trading_research/search_engine/carry.py)
on the engine [`strategies/carry.py`](../../src/trading_research/strategies/carry.py).

- **Universe.** The seventeen coins of
  [`carry_walkforward_2026_11`](carry_walkforward_2026_11.md), each held as a
  delta-neutral carry (long spot, short the USD-M perpetual, unlevered) at
  Binance Regular fees and 2 bp of slippage a leg and side.
- **Rules searched.** Every combination of: hold the **1, 2, 4 or 8** coins
  paying the most funding a day; measured over the previous **1 or 3**
  months; above a floor of **the round trip's cost** (1.27 bp a day) **or
  none**. Sixteen rules, rebalanced monthly; a coin kept pays no round trip.
- **Development months, D:** January 2021 to June 2023. Every rule is
  replayed on D; the search **chooses the rule with the highest monthly Sharpe
  ratio on D**, ties broken by D's total return.
- **Held-out months, H:** July 2023 to June 2025. The choice and the sha256 of
  D's results are written to the run's ledger first; then the chosen rule is
  replayed on H, **once**. The command refuses a second read.
- **Placebos.** Nineteen replays of the chosen rule on each block with its
  coins drawn at random each month from those it could have held: what choosing
  by funding adds over holding any carry.
- **The held-out months of `funding_carry_2026`** (BTC and ETH, December 2020
  to June 2021 and February to June 2024) and everything after June 2025 are
  not read.

## What counts

- **Positive** if the chosen rule's total return on H is above zero;
  **killed** otherwise.
- **Choosing adds** if it also beats all nineteen placebos on H; reported
  beside, with no status of its own.

## What was known before this was written

This search is not blind on H, and says so. On 13 November 2026 the
registered replay [`carry_walkforward_2026_11`](carry_walkforward_2026_11.md)
read the same coins from January 2021 to June 2025 with one of these sixteen
rules (four coins, one month, the cost floor) and found it positive in every
year of H's span (2023 +3.5%, 2024 +9.2%, January to June 2025 +0.2%). H is
inside the information set of whoever wrote the grid. What this search adds is
the choice: the rule is picked by the pipeline from D alone, not by the author.

## Expected outcome, stated now

The search chooses a rule holding several coins (four or eight) on one month
of funding with the cost floor, since a floor keeps it out of the months that
pay less than the round trip and several coins spread the basis's noise. H is
**positive**, in single digits a year. Choosing by funding beats most random
placebos on H but not all nineteen, because in a market where most coins' funding
is positive most of the time, any carry earns.

## Results, read once 13 November 2026

From the code committed before the run (`trp search carry`), in
[`experiments/results/search_carry`](../../experiments/results/search_carry/):
`development.json`, `ledger.json` (choice and D's sha256 `8b831780…`, written
before H was read) and `held_out.json`.

**On D, every one of the sixteen rules was positive**, with monthly Sharpe
ratios from 1.76 to 1.99 and returns from +11.1% to +19.2% a year. The search
chose **`top1-lb3-floor_none`**: hold the one coin paying the most funding a
day over the previous three months, with no floor (Sharpe 1.99 on D, +15.8% a
year, worst month-end drawdown −0.1%). Its nineteen placebos on D returned
+8.4% to +39.0% over the thirty months, against its +44.3%.

**H: positive.** +15.8% over July 2023 to June 2025, **+7.6% a year**, worst
month-end drawdown −0.09%, three losing months of twenty-four (each above
−0.1%). By year: July to December 2023 +3.1%, 2024 +11.9%, January to June
2025 +0.5%. **Choosing adds:** it beat all nineteen random-pick placebos, which
returned +0.5% to +7.3%. The coin held was FIL in seventeen of the twenty-four
months, then LTC, LINK, ETC and DOGE.

Against the expectation stated in advance: H positive and in single digits a
year, as expected; the choice was **one** coin, not four or eight, and it
**beat all** nineteen placebos, not most.

- **One coin at a time** concentrates the position's risk: a coin's own
  events (a delisting, a funding spike against the short, an exchange's
  failure) are not spread. The replay prices none of them.
- H was read by the walk-forward replay of a neighbouring rule before this
  search was written (above); the choice among the sixteen was made on D alone.
