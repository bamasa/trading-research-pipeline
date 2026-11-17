"""A replication of the author's 2025 ETH result on public data.

The author's 2025 research trained a temporal convolutional network on ten days
of ETH/USDT spot books (2025-01-28 to 2025-02-06), gated its three-class
probabilities with an expected-value rule written in basis points (its inputs
are z-scores; see :mod:`.gate`), and backtested the signals on the last two days
with 0.01 ETH. Of about thirty gate configurations run on those two days, three
ended positive after the backtester's fee (about 2.2 bp of turnover): sweep
cells 206 and 218 at +1 USDT rounded, and REF cell 48 at +3.1 USDT before the
fee and +0.9 after, the best of them by profit per million.

Those books are not public. This package rebuilds the recipe on Bybit's public
ETHUSDT linear perpetual archive (book and prints), which is a different venue
and a different instrument; every number here is that venue's. The modules:

* :mod:`.grid` puts a replayed book and the day's prints on the original's
  100 ms grid.
* :mod:`.features` re-types the original's sixteen auxiliary features, its
  volatility filter, its three strategy inputs, its forty normalised book
  features and its labels.
* :mod:`.net` is the original's network and training loop.
* :mod:`.gate` is the original's gate in basis points and its one-trade-per-
  event rule.
* :mod:`.execution` turns signals into taker round trips through
  :func:`trading_research.backtest.taker.execute_day`.
"""
