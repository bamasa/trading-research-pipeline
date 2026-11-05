# The benchmark: score any strategy the same way

A backtest of one strategy on one span says little. Too much was chosen along
the way: the span, the fee, the parameters, the way fills are counted. The
benchmark fixes all of it. It runs seven tracks and applies one scoring rule,
written down before any board was computed, to every strategy, whether it is
the repository's or yours. Two cards can then be compared line by line.

**Version 1, frozen on 4 November 2026.** The tracks, the seeds, the rule and
the verdict are fixed, and so is the fingerprint of each track's data
(`bench/manifest.py`). Changing any of them makes a version 2; version 1 stays
as written, so a number quoted from a version-1 card keeps its meaning. How it
compares with the published benchmarks is in
[`benchmarks_sota.md`](benchmarks_sota.md); how to cite it is in
[`CITATION.cff`](../CITATION.cff).

```bash
trading-research bench list
trading-research bench fetch
trading-research bench verify
trading-research bench state
trading-research bench run reversion
trading-research bench board
```

`fetch` downloads the recorded tracks from Bybit's free archives (about 2,000
instrument-days of top-of-book; days already on disk are kept, and the dry run
says what it would fetch). `verify` fingerprints each track's data and
compares it with version 1's. `state` measures the market state the reversion
needs on each track's scored span. `run` takes any registered strategy or model and
writes its card under `artifacts/bench/<name>/` (`card.md`, `card.json`), with
the benchmark version, the code version and, for each track, whether its data
matches version 1. `board` does the same for every registered strategy
(`--kind model` or `both`, `--workers` to score several at once) and writes
`board.json`.

## Your own strategy

A strategy is one file with a decorator, `fit(history)` and `decide(frame)`
([`extending.md`](extending.md)). Write it from the template, pass the
conformance checks (no look-ahead, its declared look-back, every execution, the
planted-edge test), then score it:

```bash
trading-research strategies new my_rule
trading-research strategies check my_rule
trading-research bench run my_rule
```

A model is scored the same way: it is fitted on the history, its confidence
threshold chosen on the history's last 30%, and it trades through the same
rule (`plugins/model_strategy.py`).

## The tracks

| track | kind | what it asks | a good strategy |
|---|---|---|---|
| `planted-book` | synthetic | Does the strategy find an order-book edge that is known to be there? Three markets (three seeds) of twelve hours of a 10-level book at 200 ms, queue-imbalance signal planted | a rule reading the touch imbalance earns before fees; others may miss it |
| `null-book` | synthetic | The same market with nothing planted | nothing distinguishable from zero |
| `planted-reversion` | synthetic | Does a cross-sectional rule find a reversion known to be there? Three universes (three seeds) of twelve instruments, eight days of 5 s, a transient planted in the market and in each instrument | a rule trading against the index move earns before fees |
| `null-reversion` | synthetic | The same universe with the transients replaced by a random walk of the same variance | nothing distinguishable from zero |
| `bybit-2024-02` | recorded | Bybit top-of-book for 26 USDT perpetuals; fitted 1–25 February, scored 26 February to 10 March 2024: the fortnight where the ten-minute reversion paid (index autocorrelation about −0.10) | a rule betting on short reversion does its best here |
| `bybit-2024-04` | recorded | The same instruments; fitted 12 March to 5 April, scored 6 to 19 April 2024, after the state changed (autocorrelation near zero) | the same rule loses or stands aside; a rule that only works in one state shows it here |
| `bybit-2026-11` | forward | The same instruments (POL for MATIC); fitted 9 October to 5 November, scored 6 to 19 November 2026: days that had not happened when the track was registered. Sealed until 21 November, then read once | nothing is expected; an edge after costs shows here, where nothing could be fitted to it |

The synthetic tracks are generated from fixed seeds, so anyone gets the same
numbers without a download; a test checks two of their fingerprints against
version 1's on every run of the suite. The reversion tracks' prices sit on a
tick grid of about a millionth of the price, as a venue's do: `exp` differs in
its last bits between a Mac and a Linux machine, and the grid rounds those bits
away, so the tracks are the same bytes on both. The recorded tracks read one parquet per
instrument and day under `data/universe` and `data/universe_fresh`, which
`fetch` writes. A fresh download of two of those days was compared with the
days this repository had stored since 2024 and matched value for value.
Without the data the recorded tracks are skipped, never faked.

**The recorded spans were chosen after this repository had read them.** They
are a benchmark, labelled as one: a known favourable state and a known
unfavourable one. A strategy's result on them shows how it behaves in each
state. It is not evidence that either state will come back.

### The forward track

The six other tracks are measurements: the recorded spans were picked after
they were read, and the synthetic answers were planted by the people who
wrote the strategies. The forward track is the test none of that touches,
the way the G-Research competition froze its models and scored them on
months that had not happened. It was registered on 5 November 2026, before its
first scored day, with the twelve strategies of the board frozen at their
sources
([pre-registration](preregistration/bench_forward_2026_11.md)). Until 21
November `bench` refuses to build it; then it is fetched, read once and
committed as it comes.

The registration also states in advance what the reversion should do. Its
condition is the index's ten-minute autocorrelation, which `bench state`
measures: −0.1323 on the 2024-02 track's scored span, where the rule paid,
and +0.0219 on the 2024-04 track's, where it lost. At −0.05 or below on the
forward fortnight the reversion should be positive after the base fee; above
−0.02 it should not be.

## The scoring rule

For each instrument the strategy is fitted on the history and decides on every
row. Its decisions after the start are traded as a taker at Bybit's base fee:
5.5 bp a side, 0.5 bp slippage, the spread paid on both legs, one position at a
time, and no sooner after a close than the strategy's `cooldown` if it declares
one. The attempts of every instrument are then pooled.

| number | what it is |
|---|---|
| gross, cost, net | bp per trade, before costs, the costs, and after them |
| *t* | the day-clustered *t* of net (hour-clustered where a group's scored span is shorter than five days, as on the synthetic tracks): trades on one day share a market, so they are not independent. Clusters never cross seeds: each seed is a market of its own |
| placebo | the share of nineteen placebos whose gross is below the strategy's; each placebo is the same decisions rotated in time and traded the same way, so an edge a rotation keeps is drift, not timing |
| break-even fee | the taker fee per side at which net would be zero, with the spread and slippage still paid |
| fee ladder | net at the base fee (5.5), VIP 1 (4.0), the lowest taker fee Bybit publishes (Pro 6, 1.5), and none at all, which no account has: the hypothetical case of a firm that pays nothing to trade |
| Holm *p* | on a board only: the one-sided *p* of net > 0 from *t*, adjusted by Holm's step-down across the board's strategies, track by track. The best of twelve is selected, so its *t* has to clear more than one strategy's would |

The verdict, the same for every strategy (`bench/score.py`, `verdict`):

| track | ✓ | ✗ | · |
|---|---|---|---|
| planted | found it: gross *t* ≥ 3 and beats 95% of the placebos | | missed it; not a failure, since a rule built for another effect is not expected to find this one |
| null | quiet: gross *t* below 3 | earned where nothing was planted, so the strategy reads the future or fits noise | |
| recorded | pays the base fee: net > 0, *t* ≥ 2, beats 95% of the placebos | an edge smaller than the fee; positive but not distinguishable from chance; nothing beyond chance | |

A card shows one mark per track, in the order of the table above: ✓, ✗, · or -
(not run, with the reason).

## The board

Every registered strategy and six of the models, scored under version 1 on
4 November 2026 (`trading-research bench board` with `--kind both --workers 6`, code
`0.1.0 2933b52`); every track's data `matches`. The numbers are in
[`experiments/results/bench_board.json`](../experiments/results/bench_board.json).
Marks run in the order of the tracks above. Net and break-even fee are bp per
trade and bp per side; Holm *p* is the family-adjusted *p* of net > 0 across
the twelve; the last column is the ladder's hypothetical rung, no fee at all,
with the spread and slippage still paid.

| strategy | kind | marks | 2024-02: net, t | break-even fee | Holm *p* | 2024-04: net, t | break-even fee | Holm *p* | net at no fee, 02 / 04 |
|---|---|:-:|--:|--:|--:|--:|--:|--:|--:|
| `book_imbalance` | strategy | ✓✓·✓✗✗ | -12.09, -112.68 | -0.54 | 1.00 | -11.74, -120.30 | -0.37 | 1.00 | -1.09 / -0.74 |
| `ev_gate` | strategy | ·✓·✓✗✗ | -10.90, -2.49 | +0.05 | 1.00 | +15.80, -0.02 | +13.40 | 1.00 | +0.10 / +26.80 |
| `hold_first` | strategy | ✓✓✓✓✗✗ | -13.20, -99.16 | -1.10 | 1.00 | -13.71, -44.38 | -1.35 | 1.00 | -2.20 / -2.71 |
| `relative` | strategy | ·✓✓✓✗✗ | -13.92, -132.45 | -1.46 | 1.00 | -13.21, -87.60 | -1.10 | 1.00 | -2.92 / -2.21 |
| `reversion` | strategy | ·✓✓✓✗✗ | +7.00, +1.21 | +9.00 | 1.00 | -46.96, -1.70 | -17.98 | 1.00 | +18.00 / -35.96 |
| `spread` | strategy | ·✓·✓✗✗ | -13.70, -85.08 | -1.35 | 1.00 | -13.93, -50.64 | -1.46 | 1.00 | -2.70 / -2.93 |
| `breakout` | model | ✓✓·✓✗✗ | -14.25, -8.39 | -1.63 | 1.00 | -10.63, -3.26 | +0.19 | 1.00 | -3.25 / +0.37 |
| `logistic` | model | ✓✓✓✓✗✗ | -15.06, -15.66 | -2.03 | 1.00 | -16.57, -10.20 | -2.79 | 1.00 | -4.06 / -5.57 |
| `mean_reversion` | model | ·✓✓✓✗✗ | -14.44, -14.11 | -1.72 | 1.00 | -16.14, -8.27 | -2.57 | 1.00 | -3.44 / -5.14 |
| `momentum` | model | ✓✓·✓✗✗ | -13.25, -15.66 | -1.12 | 1.00 | -13.63, -8.16 | -1.32 | 1.00 | -2.25 / -2.63 |
| `order_flow` | model | ✓✓·✓✗✗ | -12.15, -90.68 | -0.57 | 1.00 | -12.09, -131.29 | -0.55 | 1.00 | -1.15 / -1.09 |
| `xgboost` | model | ✓✓·✓✗✗ | -12.62, -40.00 | -0.81 | 1.00 | -17.00, -7.65 | -3.00 | 1.00 | -1.62 / -6.00 |

What the board says:

- **No strategy pays the base fee on either recorded track.** Every recorded
  mark is ✗, and no Holm *p* is below 1.
- **The checks do their job on synthetic data.** Each planted effect is found
  by rules built for it: the book edge by `book_imbalance`, `order_flow`,
  `hold_first`, `logistic`, `xgboost`, `breakout` and `momentum` (a queue
  imbalance moves the price the way it leans, so a momentum model rides it);
  the reversion by `reversion`, `hold_first`, `relative`, `logistic` and
  `mean_reversion`. `xgboost` found the reversion (gross *t* 3.68) before the
  synthetic prices were put on the tick grid and misses it after (*t* 2.89):
  moving prices by a millionth moves its decisions, so its mark on that track
  is fragile. Every strategy is quiet on both nulls. On a
  single seed, a draft of this board had flagged `xgboost` on the book null
  (gross *t* 3.86 on 42 trades); on three seeds it is quiet (*t* 0.71 on 225).
  The flag was one draw, which is why version 1 scores three.
- **Real timing, too small to trade.** `book_imbalance` and `order_flow` time
  the market on both recorded spans: gross *t* from 8 to 17, above every
  placebo. But the edge is 1.3 to 1.7 bp a trade, less than the spread and
  slippage alone, so they lose even with no fee at all. Crossing the spread is
  what kills them, not the fee; a firm with zero fees would have to rest at the
  touch instead, and resting orders meet adverse selection
  ([`findings.md`](findings.md#maker-execution)).
- **The reversion, in its state and out of it.** On 2024-02 it makes +7.00 bp
  a trade after the base fee (*t* 1.21 by day, Holm *p* 1), with a break-even
  fee of 9 bp a side, so it would pay at every published tier and +18 bp at
  none. On 2024-04 it loses 46.96 bp a trade. The same rule, the same
  parameters: the market's state decides, as
  [§27](results.md#27-cross-sectional-reversion-and-the-market-state-it-requires)
  found. These numbers repeat §27's (+6.89 pooled there) only because the
  benchmark's taker honours the rule's cooldown; without it the rule re-enters
  while the index is still extended and loses on 2024-02 too.
- **One positive mean that is not an edge.** `ev_gate` averages +15.80 bp on
  2024-04 over 163 trades, but its day-clustered *t* is −0.02: a few days carry
  the mean and the other days lose.

## Submitting a result

A card is reproducible by construction: the plugin's source sha256, the
benchmark version, the code version and each track's data fingerprint are in
it. To put a strategy on the board, open a pull request with the plugin file
(under `strategies/contrib/` or `models/contrib/`, its test beside it) and its
`card.json`. The card is accepted when `trading-research bench run` on the
pull request reproduces it with every track's data marked `matches`. A
strategy tuned on the recorded tracks is welcome on the board, labelled as
tuned: the tracks are public, so a card cannot show it was not.

## What a pass would still need

A pass on this benchmark is a measurement, not a strategy to trade. To trade one
that passes with real money, it still needs:

1. **A test it was not tuned on.** A registered study ([`preregistration/`](preregistration/)):
   the kill conditions committed before a held-out block is read, the block
   read once, then a forward block. The tracks here are public and fixed, so a
   strategy tuned to them proves nothing.
2. **A fill model as hard as the market.** The top of the book on a five-second
   grid is optimistic. A strategy that pays here must also pay in the event-time
   taker that walks the book with latency (`event_taker`), and a passive one in
   the queue simulator ([`execution_assumptions.md`](execution_assumptions.md)).
3. **The fee the account really has.** The ladder's last rung, no fee at all,
   belongs to no account. Most accounts pay the base tier; Pro tiers need
   hundreds of millions of USDT a month.
4. **The state it needs.** If it only pays in one state of the market, as the
   reversion does, that state has to be detected before trading, not after. This
   repository tried three ways to do that and none worked one day ahead
   ([`findings.md`](findings.md)).
5. **A paper run on live data** before money ([`paper_trading.md`](paper_trading.md)),
   and a size small enough that the depth at the touch does not move against it.

This repository's own studies, which did all of the above, found nothing that
pays Bybit's published taker fees outside the one conditional reversion.
Nothing here is investment advice.
