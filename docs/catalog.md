# Catalogue

Everything the repository can run, written by `trp catalog --write` from where each part is defined. Do not edit by hand.

- [Instruments and data: where to get them](#instruments-and-data-where-to-get-them)
- [Synthetic and simulated markets](#synthetic-and-simulated-markets)
- [Strategies](#strategies)
- [Models](#models)
- [Features](#features)
- [Feature filters and checks](#feature-filters-and-checks)
- [Labels](#labels)
- [Executions and backtests](#executions-and-backtests)
- [Evaluation: how a result is judged](#evaluation-how-a-result-is-judged)
- [Regime and structural-break detectors](#regime-and-structural-break-detectors)
- [The automatic search](#the-automatic-search)
- [Checks against planted answers](#checks-against-planted-answers)
- [Fee tables](#fee-tables)
- [Registered studies](#registered-studies)
- [Paper trading](#paper-trading)
- [Test cases](#test-cases)

## Instruments and data: where to get them

| venue | instrument type | datasets | book | where | fetch |
|---|---|---|---|---|---|
| Binance | spot (e.g. BTCUSDT, BTCFDUSD) | aggTrades, klines (bars, any interval) | none | https://data.binance.vision | `trp data download --market spot --kind aggTrades --symbol BTCFDUSD --start 2025-06-01 --end 2025-06-07` |
| Binance | USD-M perpetual (linear, e.g. BTCUSDT, XRPUSDT) | aggTrades, bookTicker, bookDepth, metrics, fundingRate, klines | the touch (bookTicker) | https://data.binance.vision | `trp data download --market futures-um --kind bookTicker --grid 100ms --symbol BTCUSDT --start 2024-02-01 --end 2024-02-07` |
| Binance | COIN-M futures (inverse, e.g. BTCUSD_PERP) | aggTrades, bookTicker, klines | the touch (bookTicker) | https://data.binance.vision | `trp data download --market futures-cm --kind klines --interval 1h --symbol BTCUSD_PERP --start 2024-01-01 --end 2024-01-31` |
| Bybit | linear perpetual (e.g. BTCUSDT, ETHUSDT, any listed contract) | order book replayed from snapshots and deltas | 500 levels to 2025-08-20, 200 levels after | https://quote-saver.bycsi.com/orderbook/linear/ | `trp data download-book --symbol ETHUSDT --depth 10 --grid-ms 100 --start 2024-03-01 --end 2024-03-03` |
| Bybit | linear perpetual prints | every trade with its aggressor | none | https://public.bybit.com/trading/ | `python: trading_research.data.bybit_trades.download_range` |
| Bybit | linear perpetual funding | every funding settlement | none | https://api.bybit.com/v5/market/funding/history | `python: trading_research.data.bybit_funding.download_range` |
| Bybit | spot (e.g. BTCUSDT) | order book replayed from snapshots and deltas | the archive's levels | https://quote-saver.bycsi.com/orderbook/spot/ | `python: trading_research.spot.bybit_book.fetch_day` |
| the venues' websockets | Binance and Bybit, spot and perpetual, live | the touch and prints, recorded going forward | the touch, or the depth the stream carries | public streams, no key | `trp paper record --venue bybit --market linear --symbol BTCUSDT --minutes 60` |

Not fetched yet:

- **Bybit inverse contracts (e.g. XRPUSD)**: no fetcher; the linear archive's layout is the model for one.
- **Bitfinex spot**: no fetcher and no fee table.
- **Options, equities, FX**: out of scope; every contract above is crypto.

Every command of the data step:

| command | what it does |
|---|---|
| `trp data download` | Download public Binance market data and convert it to the project contract. |
| `trp data download-book` | Fetch multi-level order books from Bybit's free archives. |
| `trp data generate-demo-data` | Generate a synthetic dataset for tests, CI and the quickstart. |
| `trp data validate-data` | Check a dataset against its contract and report quality problems. |
| `trp data describe-schema` | Print a data contract, so the expected columns and units are never guesswork. |

## Synthetic and simulated markets

Planted answers to test a strategy on before any real day: each has a null with the effect removed.

| command | what it writes |
|---|---|
| `trp synthetic book-days` | Order-book days (10 levels, 100 ms) and prints with a planted short-horizon edge. |
| `trp synthetic break-days` | Whole days of book with one regime break at a known time (or none, with --null). |
| `trp synthetic cross-section` | A universe on 5 s bars with a planted reversion or momentum. |
| `trp synthetic order-flow` | An order-level book for the market-making simulator, with the queue's truth. |
| `trp synthetic funding` | A funding-rate series with a planted persistence, in the venue's file layout. |

## Strategies

| strategy | origin | what it does |
|---|---|---|
| `book_imbalance` | contrib | Go with the heavier side of the touch once its imbalance is in the top share; exit on the clock. |
| `ev_gate` | built-in | Enter where the predicted move's lower bound clears fees, the walk and a buffer. |
| `hold_first` | built-in | The hold-first rule on a model's class probabilities, thresholds set from a trade rate. |
| `relative` | built-in | Long the most depressed name and short the most extended: the traded leg's sign. |
| `reversion` | built-in | Trade against the peers' index move, the frozen ten-minute reversion rule. |
| `spread` | built-in | The pair against its first peer: a hedge and a mean-reverting residual fitted on history. |

Add one: `trp strategies new NAME`, then `trp strategies check NAME`.

## Models

| model | family | what it is |
|---|---|---|
| `always_hold` | baseline | Never trades. |
| `class_prior` | baseline | Predicts the training class frequencies, ignoring the features. |
| `agree` | composite (agreement or gate) | Agreement: order_flow, logistic |
| `agree_all` | composite (agreement or gate) | Agreement: order_flow, logistic, xgboost |
| `model_gated_by_rule` | composite (agreement or gate) | Gated: logistic, order_flow |
| `rule_gated_by_model` | composite (agreement or gate) | Gated: order_flow, logistic |
| `bound_mixed` | confidence bound | Confidence bound: mean from ridge, error variance from xgboost_regressor |
| `bound_ridge` | confidence bound | Confidence bound: mean from ridge, error variance from ridge |
| `bound_tree` | confidence bound | Confidence bound: mean from xgboost_regressor, error variance from xgboost_regressor |
| `blend` | ensemble (average) | Average of order_flow, logistic |
| `ensemble` | ensemble (average) | Average of logistic, xgboost |
| `ensemble_all` | ensemble (average) | Average of logistic, xgboost, tcn |
| `xgboost` | gradient-boosted trees | XGBoost multi-class classifier over the feature frame. |
| `logistic` | linear | Multinomial logistic regression on standardised features. |
| `meta_model_model` | meta-labelled | Meta-labelled: logistic picks the side, xgboost decides |
| `meta_rule_model` | meta-labelled | Meta-labelled: order_flow picks the side, logistic decides |
| `meta_rule_tree` | meta-labelled | Meta-labelled: order_flow picks the side, xgboost decides |
| `tcn` | neural network | Causal dilated convolutional classifier over a window of past rows. |
| `tcn_gaussian` | neural network | Causal dilated network predicting a move in bp and the log of its variance. |
| `ridge` | regression | Linear, regularised. The regression counterpart of the logistic baseline. |
| `xgboost_regressor` | regression | Gradient boosting on the basis-point target. |
| `breakout` | rule (fitted to nothing) | Trade a move that clears the recent range. |
| `mean_reversion` | rule (fitted to nothing) | Trade against a move away from the recent average. |
| `momentum` | rule (fitted to nothing) | Trade in the direction the price has just moved. |
| `order_flow` | rule (fitted to nothing) | Trade with the imbalance at the touch. |
| `spread_capture` | rule (fitted to nothing) | Trade only when the spread is unusually tight, in the imbalance direction. |
| `extra_trees` | tree ensemble | Random forest with random split thresholds as well as random features. |
| `lightgbm` | tree ensemble | Leaf-wise histogram boosting — a second booster that errs differently. |
| `random_forest` | tree ensemble | Bagged trees with feature subsampling at each split. |

Add one: `trp strategies new NAME --kind model`; the search then takes it with `trp search register --models logistic,NAME`.

## Features

| feature set | what it holds |
|---|---|
| `book` | Every registered book-plane feature (``features/book.py``, ``features/microstructure.py``). |
| `depth` | The depth builder over every level the book carries (``features/depth.py``). |
| `flow` | The flow builder on prints bucketed by the backward rule (``features/flow.py``). |
| `generated` | The wide generated set on the book: lags, windows, z-scores (``features/generated.py``). |
| `microstructure_2025` | The short-horizon study's set, winsorised on its trailing hour (``features/study_set.py``). |
| `trade_stats` | Trailing trade statistics in time windows, joined backward onto the book (``features/align.py``). |

Single registered features, each with the look-back it declares and the look-ahead check enforces:

| feature | plane | look-back rows | what it measures |
|---|---|---|---|
| `imbalance_slope_10` | book | 10 | Whether queue imbalance is building or decaying. |
| `log_depth_ratio` | book | 0 | log(bid size / ask size) at the touch. |
| `log_total_depth` | book | 0 | log of total size resting at the touch. |
| `micro_drift_5` | book | 5 | Microprice movement over five rows, in basis points. |
| `microprice_dev_bp` | book | 0 | Size-weighted mid minus arithmetic mid, in basis points. |
| `mid_return_1_bp` | book | 1 | Log return of mid over one observation, in basis points. |
| `mid_return_20_bp` | book | 20 | Log return of mid over 20 observations, in basis points. |
| `mid_reversal_10` | book | 10 | Whether the last move undid the one before it. |
| `ofi_1` | book | 1 | Order flow imbalance at the touch over one update. |
| `ofi_20` | book | 20 | Order flow imbalance summed over the last 20 updates, in size units. |
| `ofi_20_norm` | book | 70 | Order flow imbalance over 20 updates, divided by typical resting depth. |
| `queue_imbalance` | book | 0 | (bid size - ask size) / (bid size + ask size) at the touch, in [-1, 1]. |
| `quote_intensity_20` | book | 20 | How often the touch actually changed over the last 20 rows. |
| `realised_vol_20` | book | 20 | Short-window realised volatility of the mid, in basis points. |
| `realized_vol_50_bp` | book | 51 | Standard deviation of one-step mid returns over 50 observations, in basis points. |
| `size_shock_10` | book | 10 | Sudden change in resting size at the touch. |
| `spread_bp` | book | 0 | Quoted spread in basis points of mid. |
| `spread_bp_ratio_50` | book | 50 | Current spread divided by its mean over the last 50 observations. |
| `spread_pressure_20` | book | 20 | Current spread against its recent typical value. |
| `touch_persistence_20` | book | 20 | How long the current best bid and ask have survived. |

Add a set: `trp strategies new NAME --kind features`.

## Feature filters and checks

| filter | what it does | implemented by |
|---|---|---|
| variance filter | drops columns that barely move, fitted on training rows only | `trading_research.features.selection:FeatureSelector` |
| correlation filter | keeps one of each group of near-duplicate columns | `trading_research.features.selection:FeatureSelector` |
| mutual information ranking | keeps the columns that say most about the label | `trading_research.features.selection:FeatureSelector` |
| rolling normalisation | z-scores on a trailing window, no statistic from the future | `trading_research.features.normalise:RollingNormaliser` |
| pooled normalisation | one scale across instruments, floors fitted on training days | `trading_research.features.normalise:normalise_pooled` |
| look-ahead check | truncates the data and requires that no earlier value moves | `trading_research.validation.leakage` |
| information audit | the linear ceiling and mutual information of a source against the move | `trading_research.evaluation.information:audit` |

## Labels

| label | what it asks the model to predict |
|---|---|
| `control_move` | The smoothed control move: mean mid after the hold over mean mid before, bp. |
| `direction` | Sign of the move where it clears the cost |
| `directional` | Forward mid return over the horizon, cut into sell, hold and buy at a threshold. |
| `forward_smoothed` | Future averaged, referenced to the price now |
| `magnitude` | The forward return in basis points |
| `net_pnl` | What a long nets after the round trip |
| `normalised_magnitude` | Smoothed move in units of trailing volatility |
| `smoothed_direction` | Direction between smoothed prices, fi-2010 style |
| `taker_forward_move` | The mid's move between the entry and exit snapshots of a taker round trip, bp. |
| `triple_barrier` | Which barrier is touched first: take-profit, stop, or the clock |
| `walked_taker` | The walked-book taker label: long, short or hold after both walks and fees. |

Add one: `trp strategies new NAME --kind label`.

## Executions and backtests

| execution | how a decision becomes fills |
|---|---|
| `event_taker` | The clip walked across the visible book at the decision and back after the hold. |
| `maker` | One clip at the touch per decision in the event-time simulator; a taker exit. |
| `passive` | Post at the touch and fill when the queue ahead has traded, on the grid. |
| `taker` | Cross on the decision and cross back after the hold, on the grid. |

Backtest commands:

| command | what it does |
|---|---|
| `trp backtest backtest` | Turn probabilities into trades and profit. |
| `trp backtest mm-backtest` | Quote both sides of one instrument in event time, day by day. |
| `trp backtest exit-search` | Search how to leave a trade: clock, take-profit, stop, trail, signal. |
| `trp backtest retrain-search` | Search how often to retrain: window length, apply length, step. |

Add an execution: `trp strategies new NAME --kind execution`.

## Evaluation: how a result is judged

| measure | what it answers | implemented by |
|---|---|---|
| net per trade, gross, cost, hit rate, drawdown | the backtest's summary of a trade list | `trading_research.backtest.metrics:summarise` |
| day-clustered t | each day one observation, so one day's many trades are not many proofs | `trading_research.evaluation.significance:assess` |
| Newey-West t | a t robust to autocorrelation in a daily series | `trading_research.evaluation.significance:newey_west_t` |
| effective count | instruments that move together counted as fewer | `trading_research.evaluation.significance:effective_count` |
| minimum detectable effect and power | what a block of this length could have found | `trading_research.evaluation.significance:minimum_detectable_effect` |
| deflated Sharpe ratio | the Sharpe a search of this many trials would reach by chance | `trading_research.evaluation.overfitting:deflated_sharpe_ratio` |
| probability of backtest overfitting | combinatorially symmetric cross-validation | `trading_research.evaluation.overfitting:cscv_pbo` |
| Holm's procedure | a family of claims tested together | `trading_research.evaluation.family:family_holm` |
| placebos | the same decisions moved in time; the result must beat their 95th percentile | `trading_research.walkthrough.verdict:placebo_values` |
| kill conditions | written before the read, applied in order | `trading_research.walkthrough.verdict:KILLS` |
| fee ladder | the same trades re-costed at every published tier | `trading_research.walkthrough.execution:at_fee` |
| headroom | the cost, or the share of trades a forecast must get right, at which the net is zero | `trading_research.walkthrough.verdict:headroom` |
| bias-variance decomposition | how much of a model's error is the model and how much the data | `trading_research.evaluation.bias_variance:decompose` |
| scorecard | every measure of one result on one page | `trading_research.evaluation.scorecard:build_scorecard` |

## Regime and structural-break detectors

| detector | what it finds | implemented by |
|---|---|---|
| CUSUM | a persistent drift in a standardised statistic, in either direction | `trading_research.validation.changepoint:detect` |
| structural-break monitors | the statistics of a break-detection library on a whitened stream, calibrated on a null | `trading_research.validation.structural_breaks:detect_breaks` |

Run: `trp regimes breaks`.

## The automatic search

| command | what it does |
|---|---|
| `trp search register` | Write the search's registration from the dataset's file names, before any day is read. |
| `trp search instruments` | Screen every instrument on the search block; registers the search first unless it is. |
| `trp search frequency` | The hold and the label threshold, by their neighbourhood on validation; the peak reported. |
| `trp search method` | Every rule against every learned model at that frequency. |
| `trp search model` | The chosen family's candidates: the rules, or a model's parameters. |
| `trp search execution` | The signal as a taker, a passive entry and a market maker on the search block; all frozen. |
| `trp search read` | The held-out block, once, through the run's ledger; the verdict as registered. |
| `trp search run` | Every step in order, the read last. |
| `trp search status` | Which steps are complete, what each chose, and the ledger's reads. |

## Checks against planted answers

| command | what it checks |
|---|---|
| `trp verify edge` | A planted edge through features, labels, model or rule and the taker executor. |
| `trp verify detector` | The regime detectors on a planted break and on the same days without it. |
| `trp verify simulator` | The market-making simulator's markouts and fills against the planted truth. |
| `trp verify reversion` | The cross-sectional and index reversion rules on a planted cross-section. |
| `trp verify all` | Every check with its defaults; one summary page beside each check's own. |

## Fee tables

| venue | tiers |
|---|---|
| binance_spot_BTCFDUSD | regular, vip1, vip2_to_vip9, spot_liquidity_provider, vip3, vip9 |
| binance_spot_BTCUSDT | regular, vip1, vip2, vip3 |
| binance_usdm_futures | standard |
| bybit_linear_perpetual | base, vip1, vip2, vip3, vip4, vip5, supreme_vip, pro1_g1, pro1_altcoin, pro2_g1, pro2_altcoin, pro3_g1, pro3_altcoin, pro4_g1, pro4_altcoin, pro5_g1, pro5_altcoin, pro6_g1, pro6_altcoin, mm_programme, maker_costs_default, programme, zero_fee |
| bybit_spot_BTCUSDT | non_vip, vip1, vip2, vip3, vip4, vip5, supreme_vip, pro6_institutional |
| synthetic | demo_config |

Every row with its period and source: `trp fees`.

## Registered studies

| study | what it registered |
|---|---|
| `short_horizon_2025` | BTCUSDT perpetual, 2025: taker and passive cells searched on D, the finalists read on H, F and P through the ledger |
| `spot_2026` | S25 on spot, Binance BTC/FDUSD and Bybit BTCUSDT, 2026: nothing searched, H read once, P for a candidate |

Start one: `trp study new NAME`.

## Paper trading

| command | what it does |
|---|---|
| `trp paper replay` | Replay a stored day through the paper loop. |
| `trp paper run` | Paper-trade a public stream for N minutes. No key, no order. |
| `trp paper record` | Record a public stream to day files for replay and the batch pipeline. |
| `trp paper status` | Where a run stands: running or finished, its PnL, position, gaps and kill switch. |
| `trp paper parity` | Streaming against batch features on a stored day: the largest deviation per feature. |

## Test cases

2286 test functions in 233 files, by area (the first word of the file name):

| area | files | tests |
|---|---|---|
| agent | 1 | 4 |
| agents | 7 | 84 |
| allocator | 1 | 26 |
| api | 1 | 6 |
| binance | 2 | 40 |
| book | 1 | 6 |
| bybit | 3 | 37 |
| catalog | 1 | 6 |
| changepoint | 1 | 11 |
| choice | 1 | 23 |
| cli | 3 | 22 |
| config | 1 | 12 |
| costs | 1 | 32 |
| demo | 2 | 10 |
| diagnostics | 1 | 12 |
| discovery | 2 | 16 |
| disk | 1 | 6 |
| ensembles | 1 | 22 |
| ensure | 1 | 16 |
| execution | 2 | 30 |
| exit | 1 | 28 |
| exits | 1 | 17 |
| export | 2 | 9 |
| family | 1 | 9 |
| feature | 1 | 12 |
| features | 1 | 21 |
| fees | 1 | 5 |
| fetch | 3 | 27 |
| fold | 1 | 6 |
| gating | 1 | 21 |
| generated | 1 | 21 |
| grand | 1 | 17 |
| information | 1 | 15 |
| leakage | 1 | 26 |
| maker | 1 | 16 |
| meta | 1 | 16 |
| mm | 24 | 246 |
| newey | 1 | 11 |
| normalise | 1 | 15 |
| overfitting | 1 | 12 |
| paper | 5 | 30 |
| placebo | 1 | 24 |
| plugins | 3 | 33 |
| positioning | 1 | 10 |
| readme | 1 | 6 |
| regime | 4 | 50 |
| report | 1 | 13 |
| retraining | 1 | 41 |
| reversion | 1 | 15 |
| schema | 1 | 13 |
| screen | 1 | 11 |
| search | 9 | 82 |
| sh | 46 | 341 |
| sh2025 | 1 | 38 |
| short | 1 | 5 |
| significance | 1 | 17 |
| sim | 1 | 4 |
| splits | 1 | 32 |
| spot | 54 | 305 |
| store | 1 | 10 |
| strategies | 1 | 19 |
| structural | 1 | 15 |
| studies | 1 | 6 |
| study | 2 | 25 |
| synthetic | 5 | 50 |
| taker | 1 | 9 |
| targets | 1 | 24 |
| tcn | 2 | 29 |
| uncertainty | 1 | 15 |
| validate | 1 | 23 |
| walkthrough | 1 | 10 |
| windows | 1 | 10 |
