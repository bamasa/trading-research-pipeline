# The carry search's choice against the size of its search

**What it says.** The carry survives the size of the search; the choice among the
sixteen does not. Every check on the chosen rule's own months passes with room: its
Sharpe ratio after Harvey and Liu's haircut is still 1.40 a year, and even counting the
months as normal it needs about 11 of them to be told from zero, against 30 on D and 24
on H. But choosing the best Sharpe ratio among rules whose Sharpe ratios differ by
a few hundredths is a coin toss: PBO 0.69, and on H the choice ranks 14 of 16
while all sixteen are positive. The edge is the carry, not the rule; holding more coins
would have done as well with a fraction of the drawdown.

Written by `experiments/carry_overfitting.py`. The search chose `top1-lb3-floor_none` among 16 rules on 30 development months (D) and read it once on
24 held-out months (H). Monthly returns after costs; Sharpe ratios annualised by √12.

The 16 rules are 1.0 independent trials (their monthly returns are
correlated); the best of that many rules of no skill would show a Sharpe ratio of about -0.04 a year.

| | D (chosen here) | H (read once) |
|---|--:|--:|
| months | 30 | 24 |
| Sharpe ratio, a year | 1.99 | 2.16 |
| skewness, kurtosis | +2.20, 6.9 | +3.19, 13.6 |
| probability the Sharpe ratio is above 0 (PSR) | 1.000 | 1.000 |
| deflated Sharpe ratio, against the best of 1.0 null rules | 1.000 | 1.000 |
| deflated Sharpe ratio, counting all 16 as independent | 1.000 | 1.000 |
| months needed to be above 0 at 95% | 3 | 3 |
| the same, if the months were normal (no skew, no fat tails) | 11 | 9 |
| months needed to be above the best of 16 null rules at 95% | 3 | 3 |

Harvey and Liu's haircut of the D Sharpe ratio, for having looked at sixteen:

| adjustment | adjusted *p* | Sharpe ratio left, a year | haircut |
|---|--:|--:|--:|
| bonferroni | 0.0264 | 1.40 | 29% |
| holm | 0.0264 | 1.40 | 29% |
| bhy | 0.0165 | 1.52 | 24% |

Probability of backtest overfitting on D (CSCV, 10 blocks of three months, 252 splits): **0.69**, median logit -0.61.

After the fact, all sixteen rules replayed on H (the search read only the choice):
the choice ranks 14 of 16 by Sharpe ratio there; the best on H is `top8-lb3-floor_none` (3.03).

| rule | D Sharpe | H Sharpe | H total |
|---|--:|--:|--:|
| `top8-lb3-floor_none` | 1.80 | 3.03 | +14.9% |
| `top8-lb3-floor_cost` | 1.85 | 2.82 | +14.1% |
| `top4-lb3-floor_none` | 1.83 | 2.81 | +14.2% |
| `top4-lb3-floor_cost` | 1.82 | 2.70 | +13.9% |
| `top8-lb1-floor_none` | 1.85 | 2.59 | +14.7% |
| `top4-lb1-floor_none` | 1.93 | 2.51 | +11.9% |
| `top2-lb3-floor_none` | 1.88 | 2.47 | +15.2% |
| `top2-lb3-floor_cost` | 1.87 | 2.44 | +15.1% |
| `top8-lb1-floor_cost` | 1.88 | 2.33 | +13.7% |
| `top4-lb1-floor_cost` | 1.90 | 2.33 | +11.3% |
| `top1-lb1-floor_none` | 1.76 | 2.28 | +9.5% |
| `top1-lb1-floor_cost` | 1.76 | 2.27 | +9.5% |
| `top2-lb1-floor_none` | 1.89 | 2.16 | +12.9% |
| `top1-lb3-floor_cost` | 1.97 | 2.16 | +15.8% |
| `top1-lb3-floor_none` (chosen) | 1.99 | 2.16 | +15.8% |
| `top2-lb1-floor_cost` | 1.88 | 2.14 | +12.8% |
