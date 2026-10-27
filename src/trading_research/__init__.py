"""Leakage-aware ML research and backtesting for limit order books.

The package is organised as a pipeline, and each stage is importable and
testable on its own:

``trading_research.data``
    Event and order-book schemas, a synthetic generator, and loaders for public
    market data.
``trading_research.features``
    A registry of small pure feature functions, each declaring how far back it
    looks so that look-ahead can be checked mechanically.
``trading_research.labels``
    Forward-looking targets, which declare their horizon so that validation can
    embargo the right amount of data.
``trading_research.validation``
    Chronological splits with purging and embargo, plus leakage probes.
``trading_research.models``
    Baselines and models behind one interface.
``trading_research.backtest``
    Cost model, execution simulation and trading metrics.
``trading_research.reporting``
    Run manifests, plots and reports.

The pipeline end to end from Python is :mod:`trading_research.api`, whose
three objects are re-exported here and imported on first use:
:class:`~trading_research.api.Pipeline` (with
:class:`~trading_research.api.PipelineConfig`),
:class:`~trading_research.api.Backtest` and :class:`~trading_research.api.Study`.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["Backtest", "Pipeline", "PipelineConfig", "Study", "__version__"]

#: Imported from :mod:`trading_research.api` on first use, so that importing the
#: package (or its command line) does not load the whole pipeline.
_API = frozenset({"Backtest", "Pipeline", "PipelineConfig", "Study"})


def __getattr__(name: str) -> object:
    if name in _API:
        from trading_research import api

        return getattr(api, name)
    raise AttributeError(f"module 'trading_research' has no attribute {name!r}")
