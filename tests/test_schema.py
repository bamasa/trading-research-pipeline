"""Tests for the data contracts."""

from __future__ import annotations

import pandas as pd
import pytest

from lobml.data.schema import (
    SCHEMA_VERSION,
    TRADE_SCHEMA,
    DatasetManifest,
    SchemaError,
    ask_price_col,
    bid_price_col,
    book_schema,
    infer_depth,
    level_columns,
    mid_price,
    signed_quantity,
    spread,
)


def test_valid_frame_passes(trades: pd.DataFrame) -> None:
    TRADE_SCHEMA.validate(trades)


def test_missing_column_is_reported_by_name(trades: pd.DataFrame) -> None:
    with pytest.raises(SchemaError, match="price"):
        TRADE_SCHEMA.validate(trades.drop(columns=["price"]))


def test_extra_columns_allowed_unless_strict(trades: pd.DataFrame) -> None:
    """Feature building adds columns; the original contract must still hold."""
    widened = trades.assign(some_feature=1.0)
    TRADE_SCHEMA.validate(widened)
    with pytest.raises(SchemaError, match="some_feature"):
        TRADE_SCHEMA.validate(widened, strict=True)


def test_timezone_naive_timestamp_is_rejected(trades: pd.DataFrame) -> None:
    """The single most common silent failure in a time-series pipeline."""
    naive = trades.assign(timestamp=trades["timestamp"].dt.tz_localize(None))
    with pytest.raises(SchemaError, match="timestamp"):
        TRADE_SCHEMA.validate(naive)


def test_integer_width_is_tolerated(trades: pd.DataFrame) -> None:
    """int32 from a Parquet round trip is not a contract violation."""
    narrowed = trades.assign(trade_id=trades["trade_id"].astype("int32"))
    TRADE_SCHEMA.validate(narrowed)


def test_float_column_typed_as_int_is_rejected(trades: pd.DataFrame) -> None:
    wrong = trades.assign(price=trades["price"].astype("int64"))
    with pytest.raises(SchemaError, match="price"):
        TRADE_SCHEMA.validate(wrong)


def test_book_schema_depth_must_be_positive() -> None:
    with pytest.raises(ValueError, match="depth"):
        book_schema(0)


def test_level_columns_are_ordered_best_first() -> None:
    assert level_columns(2) == (
        "bid_price_0",
        "bid_size_0",
        "ask_price_0",
        "ask_size_0",
        "bid_price_1",
        "bid_size_1",
        "ask_price_1",
        "ask_size_1",
    )


def test_infer_depth_counts_only_complete_levels(book: pd.DataFrame) -> None:
    assert infer_depth(book) == 5
    truncated = book.drop(columns=["ask_size_4"])
    assert infer_depth(truncated) == 4


def test_signed_quantity_follows_the_aggressor() -> None:
    """A resting buyer means the aggressor sold, so flow is negative."""
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01", "2026-01-01"], utc=True),
            "symbol": pd.array(["X", "X"], dtype="string"),
            "price": [100.0, 100.0],
            "quantity": [2.0, 3.0],
            "is_buyer_maker": [False, True],
            "trade_id": [1, 2],
            "source": pd.array(["t", "t"], dtype="string"),
        }
    )
    assert list(signed_quantity(df)) == [2.0, -3.0]


def test_mid_and_spread_are_derived_from_the_touch() -> None:
    df = pd.DataFrame({bid_price_col(0): [99.0], ask_price_col(0): [101.0]})
    assert mid_price(df).iloc[0] == 100.0
    assert spread(df).iloc[0] == 2.0


def test_manifest_omits_absent_optional_fields() -> None:
    manifest = DatasetManifest(
        plane="trades",
        symbol="X",
        source="synthetic",
        schema_version=SCHEMA_VERSION,
        rows=10,
        start="a",
        end="b",
    )
    payload = manifest.to_dict()
    assert "seed" not in payload
    assert payload["schema_version"] == SCHEMA_VERSION


def test_manifest_carries_seed_and_generator_when_present() -> None:
    manifest = DatasetManifest(
        plane="book",
        symbol="X",
        source="synthetic",
        schema_version=SCHEMA_VERSION,
        rows=10,
        start="a",
        end="b",
        seed=42,
        generator="g",
        extra={"note": "n"},
    )
    payload = manifest.to_dict()
    assert payload["seed"] == 42
    assert payload["generator"] == "g"
    assert payload["note"] == "n"
