"""Tests for dataset persistence.

The round trip is the contract: what comes back off disk must satisfy the same
schema as what went in. Parquet preserves most dtypes but not all of them, and a
timezone quietly lost in storage would break every time-ordering guarantee
downstream without raising anything.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from trading_research.data import store
from trading_research.data.schema import TRADE_SCHEMA, book_schema
from trading_research.data.validate import validate_book, validate_trades


@pytest.fixture
def written(tmp_path: Path, dataset) -> Path:
    return store.write_dataset(
        tmp_path / "ds",
        trades=dataset.trades,
        book=dataset.book,
        latent=dataset.latent,
        manifests=dataset.manifests,
    )


def test_round_trip_preserves_the_trade_contract(written: Path, dataset) -> None:
    restored = store.read_trades(written)
    TRADE_SCHEMA.validate(restored)
    pd.testing.assert_frame_equal(restored, dataset.trades)


def test_round_trip_preserves_the_book_contract(written: Path, dataset) -> None:
    restored = store.read_book(written)
    book_schema(dataset.config.depth).validate(restored)
    pd.testing.assert_frame_equal(restored, dataset.book)


def test_restored_data_still_validates(written: Path) -> None:
    assert validate_trades(store.read_trades(written)).ok
    assert validate_book(store.read_book(written)).ok


def test_timestamps_come_back_as_utc(written: Path) -> None:
    assert str(store.read_trades(written)["timestamp"].dt.tz) == "UTC"
    assert str(store.read_latent(written)["timestamp"].dt.tz) == "UTC"


def test_manifest_records_provenance(written: Path, dataset) -> None:
    manifest = store.read_manifest(written)
    assert set(manifest["files"]) == {"trades", "book", "latent"}
    book_manifest = manifest["planes"]["book"]
    assert book_manifest["source"] == "synthetic"
    assert book_manifest["seed"] == dataset.config.seed
    assert book_manifest["depth"] == dataset.config.depth
    assert book_manifest["generator"]


def test_planes_are_detected(written: Path) -> None:
    assert store.has_plane(written, "trades")
    assert store.has_plane(written, "book")


def test_partial_dataset_writes_only_what_it_has(tmp_path: Path, dataset) -> None:
    path = store.write_dataset(tmp_path / "trades_only", trades=dataset.trades)
    assert store.has_plane(path, "trades")
    assert not store.has_plane(path, "book")


def test_writing_nothing_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        store.write_dataset(tmp_path / "empty")


def test_reading_an_absent_plane_raises(tmp_path: Path, dataset) -> None:
    path = store.write_dataset(tmp_path / "trades_only", trades=dataset.trades)
    with pytest.raises(store.DatasetNotFoundError):
        store.read_book(path)


def test_reading_an_absent_manifest_raises(tmp_path: Path) -> None:
    with pytest.raises(store.DatasetNotFoundError):
        store.read_manifest(tmp_path)
