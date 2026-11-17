# The benchmark against the state of the art

What the published benchmarks for trading models score, what each guards
against, and what this benchmark ([`benchmarks.md`](benchmarks.md)) took from
them or does differently. Surveyed in November 2026; the sources are at the end.

## What exists

| benchmark | what it scores | data | costs | guard against overfitting |
|---|---|---|---|---|
| **FI-2010** (Ntakaris et al., 2018) | classification of the mid-price's direction (F1) | ten days of five Nasdaq Nordic stocks, 2010 | none in the score | a fixed split; labels smoothed over future prices |
| **LOBCAST** (Prata et al., *AI Review* 2024) | fifteen deep models on the same direction task, plus a profit analysis | FI-2010 and LOBSTER stocks | in the profit analysis | re-tests on unseen stocks: every model drops sharply |
| **TLOB** (Berti and Kasneci, 2025) | a transformer for the same task | FI-2010, NASDAQ, Bitcoin | trend labels set against the average spread | reports that cost-aware labels lower every model's score |
| **LOB-Bench** (Nagy et al., ICML 2025) | how realistic *generated* order-book messages are: distributions, a discriminator, price impact | LOBSTER | not a trading score | a trained discriminator against real data |
| **TradeMaster** (NeurIPS 2023) and **PRUDEX-Compass** (TMLR 2023) | reinforcement-learning traders on six axes (profitability, risk, universality, diversity, reliability, explainability), 17 measures | four long real-world datasets, over 15 years | in the environments | many measures, several markets |
| **FinRL-Meta** (NeurIPS 2022) | market environments for reinforcement learning, reproductions of papers, community competitions | pipelines from public sources | in the environments | names low signal-to-noise, survivorship bias and backtest overfitting as the field's problems |
| **QuantBench** (2025) | a standard pipeline for quantitative-investment models, aligned with industry testing | standardised datasets | in the pipeline | finds distribution shift and overfitting at low signal-to-noise |
| **FinTSB** (2025) | stock forecasting across four classes of price movement, standard metrics on three dimensions | stock data | transaction fees modelled | diversity of market patterns |
| **G-Research Crypto Forecasting** (Kaggle, 2021–22) | weighted correlation of 15-minute residual returns | crypto assets | none in the score | models frozen, then scored on three months of live data that did not exist when they were built |
| **PBO / CSCV** and the **deflated Sharpe ratio** (Bailey and López de Prado) | not benchmarks: corrections for choosing the best of many backtests | | | the probability that the in-sample winner is below median out of sample; a Sharpe deflated by the number of trials |
| **ABIDES** (Byrd et al.) | an agent-based exchange simulator | simulated, or replayed | in the simulation | shows what a replay cannot: how the market answers one's own orders |

## What the field has learned, in three lines

1. **Accuracy is not money.** The order-book benchmarks score F1 on a direction
   label. When the label is set against the spread (TLOB), every model's score
   falls. This repository found the same from the other side: the direction is
   predictable, and the edge is smaller than the cost of trading it
   ([`findings.md`](findings.md)).
2. **New data breaks most models.** LOBCAST's models drop on unseen stocks;
   QuantBench and FinTSB name distribution shift as the main problem. FI-2010's
   labels are smoothed over the future, and this repository measured that the
   standard smoothing correlates +0.586 with a quantity known before the
   decision ([`findings.md`](findings.md#fi-2010-smoothed-labels)).
3. **The only clean test is one the model could not have seen.** G-Research
   froze the models and scored them on months that had not happened yet.
   Everything short of that needs a correction for how many things were tried
   (PBO, the deflated Sharpe ratio, Holm).

## What this benchmark took, and what it adds

| from | taken | here |
|---|---|---|
| LOBCAST, QuantBench | test where the model was not tuned | two recorded spans in two market states; a strategy fitted on each span's first weeks and scored on the rest |
| TLOB, FinTSB | costs inside the score | every number is net of Bybit's published base fee, with the spread and slippage paid; a fee ladder and the break-even fee |
| PRUDEX-Compass | reliability as an axis | three seeds per synthetic track, nineteen time-shifted placebos per strategy, day-clustered *t* |
| PBO, deflated Sharpe | the family is part of the test | Holm's adjustment across a board's strategies, track by track |
| FinRL-Meta | data from the source, by a pipeline | `trading-research bench fetch` downloads the recorded tracks from Bybit's free archives; a fresh download was checked against the stored days and matched to the byte |
| G-Research | a test on the future | the forward track `bybit-2026-11`, registered 5 November 2026 before its first day, sealed until its fortnight has been published, read once |

Two things none of the benchmarks surveyed here offer:

- **Tracks with a known answer.** An edge is planted in a synthetic market and
  removed from its twin. A strategy built for the effect must find it, and
  every strategy must stay quiet on the null. A strategy that earns on a null
  is reading the future or fitting noise, and this is caught before any real
  data is involved. Three seeds a track keep one unlucky draw from deciding: a
  single-seed draft flagged `xgboost` on the book null, and three seeds showed
  it was the draw.
- **A fingerprint of the data in every card.** Each card records the sha256 of
  the numbers each track held; `trading-research bench verify` compares a
  checkout's data with version 1's. Two cards are comparable only when their
  fingerprints match.

## Where it is weaker, and version 2

- **A replay, not a market.** Like every benchmark above except ABIDES and
  LOB-Bench, the recorded tracks replay history: one's own orders move
  nothing. The taker fill at the top of the book on a five-second grid is
  optimistic. Version 2 should add the event-time taker that walks the book with
  latency, and the queue simulator for passive orders, as tracks of their own.
- **One forward fortnight.** The forward track is fourteen days of one market
  state; a strategy's verdict on it rests on fourteen day-clusters. A rolling
  forward track, a new sealed fortnight registered as each one opens, would
  make it a series.
- **Survivorship.** The 26 instruments traded through all of both spans; a
  listing or a delisting inside a span is not represented.
- **One axis.** Profitability and reliability are scored; risk (drawdown, tail
  loss) and capacity are not. PRUDEX-Compass's axes are the model.
- **Synthetic realism unmeasured.** The generators plant one effect each; how
  realistic their order flow is has not been measured. LOB-Bench's statistics
  could score it.

## Sources

- FI-2010 and its labels: [DeepLOB (Zhang, Zohren, Roberts)](https://arxiv.org/pdf/1808.03668)
- LOBCAST: [LOB-Based Deep Learning Models for Stock Price Trend Prediction: A Benchmark Study](https://arxiv.org/pdf/2308.01915.pdf), [*Artificial Intelligence Review* 57, 116 (2024)](https://link.springer.com/10.1007/s10462-024-10715-4)
- TLOB: [arXiv 2502.15757](https://arxiv.org/html/2502.15757v3)
- LOB-Bench: [arXiv 2502.09172](https://arxiv.org/pdf/2502.09172), [ICML 2025](https://proceedings.mlr.press/v267/nagy25a.html)
- TradeMaster: [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b8f6f7f2ba4137124ac976286eacb611-Abstract.html); PRUDEX-Compass: [arXiv 2302.00586](https://arxiv.org/abs/2302.00586v1)
- FinRL-Meta: [arXiv 2211.03107](https://arxiv.org/pdf/2211.03107), [NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0bf54b80686d2c4dc0808c2e98d430f7-Abstract.html)
- QuantBench: [arXiv 2504.18600](https://arxiv.org/pdf/2504.18600)
- FinTSB: [arXiv 2502.18834](https://arxiv.org/abs/2502.18834)
- G-Research Crypto Forecasting: [the organiser's wrap-up](https://www.gresearch.com/news/wrapping-up-the-g-research-crypto-forecasting-competition/)
- PBO and CSCV: [Bailey, Borwein, López de Prado, Zhu (SSRN 2326253)](https://papers.ssrn.com/abstract=2326253)
- ABIDES: [arXiv 1904.12066](https://arxiv.org/abs/1904.12066v1); replay against interactive simulation: [arXiv 1906.12010](https://arxiv.org/pdf/1906.12010)
