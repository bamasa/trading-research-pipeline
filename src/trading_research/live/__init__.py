"""Paper trading: the backtest's features, model, execution and fees on a public stream.

The deployment step of the pipeline, with no order ever sent anywhere. A
session reads a public market-data stream (or a stored day replayed through
the same interface), computes the registered features from a window sized by
their declared look-backs, scores a model at each boundary, carries the
decision out as paper orders filled against public prints by the repository's
fill rules, and keeps a paper account at the dated fee with limits, a kill
switch and a mark at every row.

Modules, in the order data flows through them:

``ws``        a standard-library websocket client (public streams only)
``stream``    Bybit and Binance public books and prints as rows and prints, with
              sequence checks, reconnects and a gap log
``record``    a stream to day files: a raw log, and the archive layouts
``replay``    a stored day through the same interface
``features``  streaming features, calling the batch functions on a window
``parity``    the streaming-versus-batch comparison on a replayed day
``decide``    the model's decision and the taker, passive and maker policies
``fees``      the dated fee tables, looked up per fill
``ledger``    the paper account, limits and kill switch
``guard``     the refusal of anything that looks like an exchange key
``session``   the loop
``page``      the run's page, tables and manifest

The commands are ``trading-research paper replay | run | record | status | parity``.
"""
