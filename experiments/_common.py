"""Shared plumbing for the experiment scripts.

Every script here answers one question from the results document, and they all
need the same three things: prepared days for an instrument, the cost model, and
a place to write a table. Putting that here keeps each script short enough to
read in one sitting, which is the point — a script nobody reads is no better
evidence than a number with no script at all.

Nothing in this module is imported by the package. The dependency runs one way:
experiments use ``trading_research``, never the reverse. An experiment is a
record of a question that was asked once, and the library should not carry
weight for it.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from trading_research.backtest.costs import TakerCosts
from trading_research.pipeline.stages import StageError, load_prepared

#: Where ``prepare`` writes by default in these scripts. Gitignored.
PREPARED = Path("artifacts")

#: Results land here as CSV, next to the numbers they justify.
RESULTS = Path("experiments/results")

#: The published Binance USD-M taker rate, plus the slippage assumed throughout.
COSTS = TakerCosts(fee_bp_per_side=5.0, slippage_bp=0.5)

#: Round trip per instrument, at the median spread. Quoted in the documents, so
#: kept in one place rather than retyped into each script.
ROUND_TRIP_BP = {"BTCUSDT": 11.02, "XRPUSDT": 12.68}

#: 100 ms observations, subsampled by 50, so one row is five seconds.
GRID_MS = 100
SUBSAMPLE = 50


def prepared_dir(symbol: str, root: Path = PREPARED) -> Path:
    return root / f"prepared_{symbol}"


def load(symbol: str, *, root: Path = PREPARED, days: int | None = None) -> pd.DataFrame:
    """Read prepared days for one instrument.

    ``days`` truncates to the first N, which is how two instruments are compared
    over the same window. Comparing them over different windows measures the
    window — an earlier version of the ceiling table did exactly that and made
    XRPUSDT look twice as tradeable as it is.
    """
    directory = prepared_dir(symbol, root)
    if not directory.exists():
        raise StageError(
            f"{directory} does not exist. Run:\n"
            f"  uv run trading-research prepare --symbol {symbol} "
            f"--horizon 1200 --subsample {SUBSAMPLE} -o {directory}"
        )
    frame = load_prepared(directory)
    if days is not None:
        keep = sorted(set(frame["timestamp"].dt.date))[:days]
        frame = frame[frame["timestamp"].dt.date.isin(set(keep))].reset_index(drop=True)
    return frame


def available_days(symbol: str, *, root: Path = PREPARED) -> list[date]:
    return [
        date.fromisoformat(f.stem) for f in sorted(prepared_dir(symbol, root).glob("*.parquet"))
    ]


def round_trip(symbol: str, frame: pd.DataFrame | None = None) -> float:
    """Cost of a round trip, from the table or measured from the data."""
    if symbol in ROUND_TRIP_BP:
        return ROUND_TRIP_BP[symbol]
    if frame is None:
        raise KeyError(f"no published round trip for {symbol}; pass a frame to measure one")
    return float(COSTS.round_trip_bp(float(frame["spread_bp_now"].median())))


def emit(table: pd.DataFrame, name: str, *, index: bool = False) -> Path:
    """Print a table and write it beside the document it supports."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.csv"
    table.to_csv(path, index=index)
    pd.set_option("display.width", 200)
    print(table.round(4).to_string(index=index))
    print(f"\n-> {path}")
    return path


def parser(
    description: str, *, symbols: Sequence[str] = ("BTCUSDT", "XRPUSDT")
) -> argparse.ArgumentParser:
    """The arguments every script accepts, so they are driven the same way."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--symbols", nargs="+", default=list(symbols))
    p.add_argument("--prepared-root", type=Path, default=PREPARED)
    p.add_argument("--days", type=int, default=None, help="Use only the first N days.")
    p.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    return p


def load_book_features(
    symbol: str,
    features: Sequence[str],
    *,
    raw_root: Path = Path("data/raw"),
    days: int | None = None,
    subsample: int = SUBSAMPLE,
) -> pd.DataFrame:
    """Build named registry features straight from the raw book.

    ``prepare`` writes the *generated* feature set — lags, rolling statistics and
    ratios derived from a handful of primitives. A few features named in the
    results document, order-flow imbalance among them, are registry features
    computed from bid and ask sizes, and those sizes do not survive into the
    prepared files. Reproducing that table therefore needs the raw stream.

    Returns the requested features plus ``timestamp`` and ``mid``, subsampled the
    same way ``prepare`` subsamples so the two are comparable.
    """
    from trading_research.features import build

    directory = raw_root / symbol / "bookTicker"
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise StageError(f"no raw book data under {directory}")
    if days is not None:
        files = files[:days]

    parts = []
    for file in files:
        book = pd.read_parquet(file)
        for column in ("symbol", "source"):
            if column in book.columns:
                book[column] = book[column].astype("string")
        built = build(book, list(features))[list(features)]
        built["timestamp"] = book["timestamp"]
        built["mid"] = (book["bid_price_0"] + book["ask_price_0"]) / 2
        parts.append(built.iloc[::subsample])
        del book, built
    return pd.concat(parts, ignore_index=True)
