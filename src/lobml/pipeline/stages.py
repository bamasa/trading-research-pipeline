"""The pipeline as separate stages, each writing its output to disk.

Five stages, each runnable on its own:

1. ``prepare``  — raw book and trades to a feature matrix, one file per day
2. ``select``   — choose features on a training window, write the list
3. ``train``    — fit a model on a training window, write the model
4. ``predict``  — run a model over a period, write probabilities
5. ``backtest`` — turn probabilities into trades and profit

Why the files, rather than one function that does everything: a stage can be
re-run without repeating what came before, which matters when feature building
takes five minutes and model fitting takes thirty seconds. More importantly,
every intermediate is inspectable. A pipeline that only produces a final number
is a pipeline whose middle nobody checks, and the middle is where the mistakes
in this kind of work actually live.

Each stage writes a manifest beside its output recording what produced it — the
configuration, the input files, the code version. That is what makes a result
traceable back to its inputs rather than to a memory of how it was run.

Ordering guarantee
------------------
Stages that fit anything — ``select`` and ``train`` — take an explicit date
range and read only that range. They cannot see later data because they are
never handed it. ``predict`` and ``backtest`` do not fit anything at all.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lobml import __version__

MANIFEST = "manifest.json"


class StageError(RuntimeError):
    """A stage cannot run with the inputs it was given."""


@dataclass
class StageManifest:
    """Provenance for one stage's output."""

    stage: str
    lobml_version: str = __version__
    inputs: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)

    def write(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / MANIFEST
        path.write_text(json.dumps(asdict(self), indent=2, default=str) + "\n", encoding="utf-8")
        return path

    @staticmethod
    def read(directory: Path) -> dict[str, Any]:
        path = Path(directory) / MANIFEST
        if not path.exists():
            raise StageError(f"no {MANIFEST} in {directory}; run the earlier stage first")
        loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return loaded


# ---------------------------------------------------------------------------
# Stage 1: prepare
# ---------------------------------------------------------------------------


def prepare(
    raw_dir: Path | str,
    out_dir: Path | str,
    symbol: str,
    *,
    horizon: int,
    subsample: int = 1,
    warmup: int = 1200,
    overwrite: bool = False,
) -> Path:
    """Build the feature matrix, one file per day.

    Processed a day at a time because the full series does not fit: 30M rows at
    ~200 columns is tens of gigabytes, and an earlier version that built it in
    one go drove the machine into swap.

    Rolling windows are warmed from the tail of the previous day rather than
    restarted at each boundary, so a day break does not produce a block of
    missing values or, worse, a window that quietly means something different
    from every other window.
    """
    from lobml.features.generated import generate
    from lobml.features.registry import REGISTRY

    raw = Path(raw_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    book_files = sorted((raw / symbol / "bookTicker").glob("*.parquet"))
    if not book_files:
        raise StageError(f"no book data under {raw / symbol / 'bookTicker'}")
    trade_files = {p.stem: p for p in (raw / symbol / "aggTrades").glob("*.parquet")}

    written: list[str] = []
    tail: pd.DataFrame | None = None

    for book_file in book_files:
        day = book_file.stem
        target = out / f"{day}.parquet"
        if target.exists() and not overwrite:
            written.append(day)
            tail = None  # cannot warm from a day that was not read
            continue

        book = pd.read_parquet(book_file)
        for column in ("symbol", "source"):
            book[column] = book[column].astype("string")

        warm_rows = 0
        if tail is not None:
            warm_rows = len(tail)
            book = pd.concat([tail, book], ignore_index=True)
        tail = book.tail(warmup).copy()

        trades = None
        if day in trade_files:
            trades = pd.read_parquet(trade_files[day])
            for column in ("symbol", "source"):
                trades[column] = trades[column].astype("string")

        features = generate(book, trades).iloc[warm_rows:]
        mid = ((book["bid_price_0"] + book["ask_price_0"]) / 2).to_numpy()
        forward = np.full(len(mid), np.nan)
        if horizon < len(mid):
            forward[:-horizon] = np.log(mid[horizon:] / mid[:-horizon]) * 1e4

        features.insert(0, "timestamp", book["timestamp"].iloc[warm_rows:])
        features["forward_bp"] = forward[warm_rows:]
        features["spread_bp_now"] = REGISTRY.get("spread_bp")(book).iloc[warm_rows:]

        features.iloc[::subsample].reset_index(drop=True).to_parquet(
            target, compression="zstd", index=False
        )
        written.append(day)
        del book, trades, features

    StageManifest(
        stage="prepare",
        inputs={"raw_dir": str(raw), "symbol": symbol},
        params={"horizon": horizon, "subsample": subsample, "warmup": warmup},
        outputs={"days": written, "n_days": len(written)},
    ).write(out)
    return out


def load_prepared(
    directory: Path | str,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """Read prepared days, optionally restricted to a date range.

    The range is applied by filename before anything is read, so a stage asked
    for a training window never has later data in memory at all. That is a
    stronger guarantee than filtering after loading: code cannot accidentally
    use what was never there.
    """
    path = Path(directory)
    files = sorted(path.glob("*.parquet"))
    if not files:
        raise StageError(f"no prepared data in {path}; run prepare first")

    if start is not None or end is not None:
        chosen = []
        for file in files:
            day = date.fromisoformat(file.stem)
            if start is not None and day < start:
                continue
            if end is not None and day > end:
                continue
            chosen.append(file)
        files = chosen

    if not files:
        raise StageError(f"no prepared days in {path} between {start} and {end}")

    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """The feature columns of a prepared frame, excluding bookkeeping."""
    reserved = {"timestamp", "forward_bp", "spread_bp_now", "label"}
    return [c for c in frame.columns if c not in reserved]
