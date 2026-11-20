# The funding spread between Binance and Bybit: 2022-01 to 2025-06

Written by `experiments/xvenue_report.py` from `development.json`
([registration](../../../docs/preregistration/xvenue_funding_2026_11.md)). Each month the rule holds
up to four of fifteen coins' perpetuals on both venues, short where the last three months' funding
was higher and long where it was lower, at taker fees on both. **A measurement on development months**; no status.

![cumulative result and spreads](figure.png)

| total % | a year % | Sharpe | worst drawdown % | losing months | mean a month % | *t* |
|--:|--:|--:|--:|--:|--:|--:|
| +3.02 | +0.85 | 0.94 | -1.08 | 12 of 42 | +0.072 | +1.76 |

- **Where it came from:** funding +4.66%, the price gap between the venues
  +0.20%, fees and slippage -1.84%; 1 month(s)
  with nothing held.
- **Bybit's funding is the higher one** on 15 of 15 coins on
  average, so the rule is mostly long on Binance and short on Bybit.
- **The direction matters:** the same coins with a random direction each month ended between
  -4.90% and -1.57%.
