# Registration: weekly rankers fitted on the return net of funding (N1, N2)

| | |
|---|---|
| **Committed** | 20 November 2026, after [`xs_weekly_2026_11`](xs_weekly_2026_11.md)'s development replay and before any fit of this study, and before any day of the held-out block was fetched |
| **What it registers** | The same universe, features, models and book as `xs_weekly_2026_11`, with one change: the target is the week's return **net of the funding** a long pays, so a ranking is fitted to what a leg earns. One rule is added. A development replay, and two statuses read once on the same held-out block |
| **Searched** | **The target, once, on D.** The change was chosen after `xs_weekly`'s development replay showed the short side paying the most volatile coins' negative funding (its amendment); it is counted here as one more configuration seen on D. Nothing else is chosen |
| **How it changes** | Only by a dated amendment appended at the end, which cannot change a status already read |

## What changes from `xs_weekly_2026_11`

- **Target:** the coin's log return from Monday's open to the next Monday's,
  minus the funding a long pays over the week (the sum of the week's
  settlements), minus the universe's mean of the same. A short's net is minus
  this, so ranking it ranks both sides' net. Models are fitted on its
  cross-sectional z-score, clipped at 3.
- **Rankers:** `momentum`, `low_vol`, a new rule `funding` (minus the 28-day
  mean funding: long the coins whose longs are paid, short those whose shorts
  are paid), and the five models, at `xs_weekly_2026_11`'s settings.
  `reversal` is dropped.

Everything else, the data, the universe, the twelve features and four series,
the refits, the book of ten a side with taker fees and funding, the blocks D
and H, is `xs_weekly_2026_11`'s, by the same code.

## N1, N2

Weeks are the observations; one-sided *t*-tests; Holm over the five models.

- **N1, a model's portfolio pays.** As `xs_weekly`'s W1, on this study's models.
- **N2, a model beats the best rule seen.** Its weekly net return minus
  `momentum`'s, the highest of `xs_weekly`'s rules on D. **Passes** as N1.

Reported beside, with no status: as `xs_weekly`, and each model's net against
the same model in `xs_weekly` on the same weeks.

## Expected outcome, stated now

**N1: killed for most; one or two may pass.** Fitting to the net return should
stop the models shorting coins that charge their shorts, and the `funding`
rule is a cross-sectional carry that has paid in this repository's other
carry studies; but the week's price move is far noisier than its funding and
the models may not learn to weigh the two. **N2: killed for all five.** On D
`momentum` made +43%, mostly from its funding (+30%), and 95 weeks resolve only
a large difference.

## Amendment, 20 November 2026: what development showed

The development replay, 182 weeks from July 2021 to December 2024
([`experiments/results/xs_weekly_net/`](../../experiments/results/xs_weekly_net/README.md)), a
measurement, by the code frozen below. Fitting on the net return cut the models' funding drag
(from −28% to −40% in `xs_weekly` to −19% to +1%) but not their losses: totals −24.5%
(`lightgbm`) to −48.8% (`tcn`), with fees of −18% to −21% at a turnover of 140% to 163% a week.
The `funding` rule made **+90.2%** (+20.2% a year, Sharpe 0.93, worst drawdown −25.6%; funding
+64.8%, price +16.3%, fees −7.8%); `momentum` +43.3%. Every model's weekly net minus
`momentum`'s is negative (*t* −0.78 to −1.16).

None of this changes a setting, a block or a status. The expected outcome stands; N1 now leans
killed for all five.

## Results, read once 20 November 2026

From the frozen code, by `experiments/xs_weekly.py read --study net`, after the freeze recorded in
[`xs_weekly_2026_11`](xs_weekly_2026_11.md) was committed; the result is
[`experiments/results/xs_weekly_net_heldout/`](../../experiments/results/xs_weekly_net_heldout/README.md)
(`result.json`). H: 94 weeks, 6 January 2025 to 19 October 2026.

**N1: killed for all five.** Every model's mean weekly net return is positive, none significantly:
`lightgbm` and `mlp` *t* +1.69 (Holm *p* 0.235), `xs_transformer` +0.95, `tcn` +0.89, `ridge`
+0.88 (Holm *p* 0.515).

**N2: killed for all five.** Weekly net minus `momentum`'s: *t* −0.72 to −0.14, every Holm *p* 1.

Beside, with no status. Totals: `momentum` +154.2%, `mlp` +134.0% (+60% a year, Sharpe 1.26, worst
drawdown −47%), `lightgbm` +122.0% (Sharpe 1.26, −41%), `xs_transformer` +46.7%, `ridge` +41.6%,
`tcn` +38.3%, `low_vol` −9.8%, `funding` −9.9%. Fitting on the net return cut the models' funding
drag from −97% to −141% in `xs_weekly` to −9% to −44%, and every model ended above its `xs_weekly`
twin. But the gain is not spread over H: from January 2025 to June 2026 every model was flat or
lost (`lightgbm` −0.7%, `mlp` −9.3%, the others −27% to −35%), and from July to October 2026 all
made +90% to +158%. The `funding` rule, +90% on D, made −9.9%: its funding paid +155%, its price
lost −135%.

Against the expectation stated in advance: N1 killed for all five, where one or two were expected
possible; N2 killed for all five, as expected. Sixteen good weeks at the end of a 94-week block are
not evidence of an edge; a forward read on weeks after October 2026 is what would be.
