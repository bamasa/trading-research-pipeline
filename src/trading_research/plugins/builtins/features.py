"""The feature builders of ``features/``, registered as feature-set plugins.

Each reads the book plane (``timestamp`` and the level columns of
:mod:`trading_research.data.schema`) and, where it needs prints, the trades
plane, and returns columns on the book's index. Look-backs are in rows of the
book handed in; the builders whose windows are in time (``trade_stats``,
``microstructure_2025``) are built for a grid of ``grid_seconds`` and declare
their look-back in rows of that grid.

Not registered, because they read more than one instrument's book and prints:
``cross`` (a universe panel), ``positioning`` (the venue's open interest and
funding) and ``lob_tensor`` (a calibration fitted on the warm-up block).
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from trading_research.plugins.market import level_columns, to_prints
from trading_research.plugins.registry import register_features

BOOK_INPUTS: tuple[str, ...] = (
    "timestamp",
    "bid_price_*",
    "bid_size_*",
    "ask_price_*",
    "ask_size_*",
)


@register_features("book")
class BookSet:
    """Every registered book-plane feature (``features/book.py``, ``features/microstructure.py``)."""

    inputs = BOOK_INPUTS

    def __init__(self) -> None:
        from trading_research.features import REGISTRY

        self.names = REGISTRY.names("book")
        self.lookback = REGISTRY.max_lookback(self.names)

    def compute(self, book: pd.DataFrame, trades: pd.DataFrame | None = None) -> pd.DataFrame:  # noqa: ARG002
        from trading_research.features import build

        return build(book, self.names, plane="book").drop(columns=["timestamp"])


@register_features("depth")
class DepthSet:
    """The depth builder over every level the book carries (``features/depth.py``)."""

    inputs = BOOK_INPUTS
    lookback = 0

    def compute(self, book: pd.DataFrame, trades: pd.DataFrame | None = None) -> pd.DataFrame:  # noqa: ARG002
        from trading_research.features.depth import build_depth_features

        return build_depth_features(book)


@register_features("flow")
class FlowSet:
    """The flow builder on prints bucketed by the backward rule (``features/flow.py``)."""

    inputs = (*BOOK_INPUTS, "trades")

    def __init__(self) -> None:
        from trading_research.features.study_set import FLOW_LOOKBACK_ROWS

        self.lookback = FLOW_LOOKBACK_ROWS

    def compute(self, book: pd.DataFrame, trades: pd.DataFrame | None = None) -> pd.DataFrame:
        from trading_research.features.flow import build_flow_features
        from trading_research.features.prints import bucket_prints

        if trades is None:
            raise ValueError("the flow features read prints; no trades were given")
        return build_flow_features(bucket_prints(to_prints(trades), book))


@register_features("generated")
class GeneratedSet:
    """The wide generated set on the book: lags, windows, z-scores (``features/generated.py``)."""

    inputs = BOOK_INPUTS

    def __init__(self) -> None:
        from trading_research.features.generated import DEFAULT_LAGS, DEFAULT_WINDOWS

        # A difference over the longest window reads one row more than it.
        self.lookback = max(*DEFAULT_WINDOWS, *DEFAULT_LAGS) + 1

    def compute(self, book: pd.DataFrame, trades: pd.DataFrame | None = None) -> pd.DataFrame:  # noqa: ARG002
        from trading_research.features.generated import generate

        return generate(book)


@register_features("trade_stats")
class TradeStatsSet:
    """Trailing trade statistics in time windows, joined backward onto the book (``features/align.py``)."""

    inputs = (*BOOK_INPUTS, "trades")

    def __init__(self, *, grid_seconds: float = 1.0) -> None:
        # The longest window (300 s) plus the join's 60 s tolerance, and the
        # impact term's 300-row shift, whichever is longer, in rows of the grid.
        self.lookback = max(math.ceil((300 + 60) / grid_seconds) + 1, 300)

    def compute(self, book: pd.DataFrame, trades: pd.DataFrame | None = None) -> pd.DataFrame:
        from trading_research.features.align import build_trade_features

        if trades is None:
            raise ValueError("the trade statistics read the trades plane; none was given")
        return build_trade_features(book, trades)


@register_features("microstructure_2025")
class Microstructure2025:
    """The short-horizon study's set, winsorised on its trailing hour (``features/study_set.py``)."""

    inputs = (*BOOK_INPUTS, "trades")

    def __init__(self, *, grid_seconds: float = 1.0, winsorise: bool = True) -> None:
        from trading_research.features.study_set import microstructure_registry

        self.registry = microstructure_registry(winsorise=winsorise)
        self.grid_ms = round(grid_seconds * 1000)
        self.lookback = self.registry.max_rows(grid_ms=self.grid_ms)

    def compute(self, book: pd.DataFrame, trades: pd.DataFrame | None = None) -> pd.DataFrame:
        from trading_research.features.timed import build_timed

        prints = None if trades is None else to_prints(trades)
        rows = book[["timestamp", *level_columns(10)]] if "bid_price_9" in book else book
        out: Any = build_timed(rows, self.registry.names(), prints=prints, registry=self.registry)
        frame: pd.DataFrame = out.drop(columns=["timestamp"])
        return frame
