"""Reading and writing datasets, with provenance attached.

A dataset is a directory, not a file: one Parquet file per plane plus a
``manifest.json`` describing where the data came from. Keeping the manifest
beside the data rather than in a separate registry means a directory copied to
another machine still knows what it is — including the seed and generator
version for synthetic data, which is the difference between a reproducible demo
and a plausible-looking one.

Reading restores dtypes to the contract. Parquet round-trips most things
faithfully but not all of them: pandas string columns can come back as
``object``, and a timezone can be dropped by an intermediate tool. Normalising
on read means a frame is either contract-compliant or raises, and never
silently degraded.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from lobml.data.schema import (
    BOOK_SCHEMA,
    TRADE_SCHEMA,
    DatasetManifest,
    Schema,
    book_schema,
    infer_depth,
)

MANIFEST_NAME = "manifest.json"
TRADES_FILE = "trades.parquet"
BOOK_FILE = "book.parquet"
LATENT_FILE = "latent.parquet"


class DatasetNotFoundError(FileNotFoundError):
    """A dataset directory is missing or does not contain the requested plane."""


def write_dataset(
    directory: Path | str,
    *,
    trades: pd.DataFrame | None = None,
    book: pd.DataFrame | None = None,
    latent: pd.DataFrame | None = None,
    manifests: dict[str, DatasetManifest] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write the planes that were supplied, plus a combined manifest."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)

    written: dict[str, str] = {}
    for name, frame, filename in (
        ("trades", trades, TRADES_FILE),
        ("book", book, BOOK_FILE),
        ("latent", latent, LATENT_FILE),
    ):
        if frame is None:
            continue
        frame.to_parquet(path / filename, index=False)
        written[name] = filename

    if not written:
        raise ValueError("nothing to write: supply at least one of trades, book or latent")

    payload: dict[str, Any] = {
        "files": written,
        "planes": {name: _as_dict(m) for name, m in (manifests or {}).items()},
    }
    if extra:
        payload.update(extra)

    (path / MANIFEST_NAME).write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return path


def read_manifest(directory: Path | str) -> dict[str, Any]:
    path = Path(directory) / MANIFEST_NAME
    if not path.exists():
        raise DatasetNotFoundError(f"no {MANIFEST_NAME} in {directory}")
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def read_trades(directory: Path | str) -> pd.DataFrame:
    """Load the trade plane, restoring contract dtypes."""
    df = _read(Path(directory) / TRADES_FILE)
    return _coerce(df, TRADE_SCHEMA)


def read_book(directory: Path | str) -> pd.DataFrame:
    """Load the book plane, restoring contract dtypes at whatever depth is stored."""
    df = _read(Path(directory) / BOOK_FILE)
    depth = infer_depth(df)
    schema = BOOK_SCHEMA if depth == 10 else book_schema(max(depth, 1))
    return _coerce(df, schema)


def read_latent(directory: Path | str) -> pd.DataFrame:
    """Load the generator's latent state, if the dataset is synthetic."""
    df = _read(Path(directory) / LATENT_FILE)
    df["timestamp"] = _to_utc(df["timestamp"])
    return df


def has_plane(directory: Path | str, plane: str) -> bool:
    filename = {"trades": TRADES_FILE, "book": BOOK_FILE, "latent": LATENT_FILE}[plane]
    return (Path(directory) / filename).exists()


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise DatasetNotFoundError(f"{path} does not exist")
    return pd.read_parquet(path)


def _coerce(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    """Restore the dtypes the contract requires, then verify the result."""
    for column in schema.columns:
        if column.name not in df.columns:
            continue
        series = df[column.name]
        if column.dtype.startswith("datetime64"):
            df[column.name] = _to_utc(series)
        elif column.dtype == "string":
            df[column.name] = series.astype("string")
        elif column.dtype.startswith("int"):
            df[column.name] = series.astype("int64")
        elif column.dtype.startswith("float"):
            df[column.name] = series.astype("float64")
        elif column.dtype == "bool":
            df[column.name] = series.astype("bool")

    schema.validate(df)
    return df


def _to_utc(series: pd.Series) -> pd.Series:
    """Return a timezone-aware UTC series, whatever the stored representation."""
    converted = pd.to_datetime(series)
    if converted.dt.tz is None:
        return converted.dt.tz_localize("UTC")
    return converted.dt.tz_convert("UTC")


def _as_dict(manifest: DatasetManifest | dict[str, Any]) -> dict[str, Any]:
    if isinstance(manifest, DatasetManifest):
        return manifest.to_dict()
    if is_dataclass(manifest) and not isinstance(manifest, type):
        return asdict(manifest)
    return dict(manifest)
