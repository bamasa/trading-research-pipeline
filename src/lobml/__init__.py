"""Leakage-aware ML research and backtesting for limit order books.

The package is organised as a pipeline, and each stage is importable and
testable on its own:

``lobml.data``
    Event and order-book schemas, a synthetic generator, and loaders for public
    market data.
``lobml.features``
    A registry of small pure feature functions, each declaring how far back it
    looks so that look-ahead can be checked mechanically.
``lobml.labels``
    Forward-looking targets, which declare their horizon so that validation can
    embargo the right amount of data.
``lobml.validation``
    Chronological splits with purging and embargo, plus leakage probes.
``lobml.models``
    Baselines and models behind one interface.
``lobml.backtest``
    Cost model, execution simulation and trading metrics.
``lobml.reporting``
    Run manifests, plots and reports.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
