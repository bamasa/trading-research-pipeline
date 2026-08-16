"""Data contracts for trade events and order-book snapshots.

Everything downstream — features, labels, splits, the backtest — reads one of
the two frames defined here. Pinning the contract in one place is what lets the
rest of the pipeline stay honest: a feature cannot quietly depend on a column
that only exists for one data source, and a new source cannot silently change
the meaning of a column.

Two planes, deliberately separate
---------------------------------
``TradeSchema``
    One row per trade (or aggregated trade). Available from public exchange
    archives, so this is the plane the first real-data experiments run on.
``BookSchema``
    One row per order-book snapshot, ``depth`` levels per side. A full book
    cannot be reconstructed from public trade archives, so real data for this
    plane needs a dedicated collector. It is defined and exercised from day one
    against synthetic data, so that book features and their tests exist before
    the collector does rather than after.

Conventions that hold for both planes
-------------------------------------
- ``timestamp`` is timezone-aware UTC with millisecond resolution or finer, and
  denotes the moment the information became observable. Every look-ahead check
  in the project is stated in terms of this column.
- Prices and quantities are in the instrument's quote and base units
  respectively. No implicit scaling, no ticks-as-integers.
- ``SCHEMA_VERSION`` is written into every manifest. Any change to the column
  set or to the meaning of a column bumps it, so that a stored dataset can
  never be silently misread by a later version of the code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal

import pandas as pd

SCHEMA_VERSION: Final = "1.0"

#: Order-book side. ``bid`` is the buy side, ``ask`` the sell side.
Side = Literal["bid", "ask"]

#: Which plane a frame belongs to.
Plane = Literal["trades", "book"]


class SchemaError(ValueError):
    """A frame does not satisfy its declared contract."""


@dataclass(frozen=True)
class Column:
    """One column of a data contract."""

    name: str
    dtype: str
    description: str
    unit: str = ""


@dataclass(frozen=True)
class Schema:
    """A named set of required columns, plus the index convention.

    ``validate`` only checks structure: presence, dtype and index. Semantic
    checks — monotone timestamps, positive prices, crossed books — live in
    :mod:`trading_research.data.validate`, because those are properties of a *dataset*
    rather than of its type, and they need to report how badly a rule is broken
    rather than merely that it is.
    """

    name: str
    plane: Plane
    columns: tuple[Column, ...]
    version: str = SCHEMA_VERSION

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)

    def dtypes(self) -> dict[str, str]:
        return {c.name: c.dtype for c in self.columns}

    def validate(self, df: pd.DataFrame, *, strict: bool = False) -> None:
        """Raise :class:`SchemaError` if ``df`` does not match this contract.

        With ``strict=True`` any column beyond the contract is also an error.
        The default is permissive because feature building legitimately adds
        columns to a frame that must still satisfy its original contract.
        """
        missing = [c for c in self.column_names if c not in df.columns]
        if missing:
            raise SchemaError(f"{self.name}: missing required column(s): {', '.join(missing)}")

        if strict:
            extra = [c for c in df.columns if c not in self.column_names]
            if extra:
                raise SchemaError(f"{self.name}: unexpected column(s): {', '.join(extra)}")

        wrong: list[str] = []
        for col in self.columns:
            actual = str(df[col.name].dtype)
            if not _dtype_compatible(actual, col.dtype):
                wrong.append(f"{col.name} is {actual}, expected {col.dtype}")
        if wrong:
            raise SchemaError(f"{self.name}: wrong dtype(s): {'; '.join(wrong)}")

    def describe(self) -> pd.DataFrame:
        """Return the contract as a table, for docs and for run manifests."""
        return pd.DataFrame(
            [
                {
                    "column": c.name,
                    "dtype": c.dtype,
                    "unit": c.unit,
                    "description": c.description,
                }
                for c in self.columns
            ]
        )


def _dtype_compatible(actual: str, expected: str) -> bool:
    """Compare dtypes leniently enough to be useful, strictly enough to catch bugs.

    Integer width is not worth failing over — a column read back from Parquet
    may come out as ``int32`` where it was written as ``int64``. Timezone is
    worth failing over: a naive timestamp column is the single most common way
    a time-series pipeline silently goes wrong.
    """
    if expected.startswith("datetime64"):
        return actual.startswith("datetime64") and "UTC" in actual
    if expected.startswith("int"):
        return actual.startswith(("int", "uint"))
    if expected.startswith("float"):
        return actual.startswith("float")
    return actual == expected


# ---------------------------------------------------------------------------
# Trade plane
# ---------------------------------------------------------------------------

TRADE_SCHEMA: Final = Schema(
    name="trades",
    plane="trades",
    columns=(
        Column(
            "timestamp",
            "datetime64[ns, UTC]",
            "Exchange match time, timezone-aware UTC.",
            "UTC",
        ),
        Column("symbol", "string", "Instrument identifier, e.g. BTCUSDT.", ""),
        Column("price", "float64", "Execution price.", "quote"),
        Column("quantity", "float64", "Executed size, always positive.", "base"),
        Column(
            "is_buyer_maker",
            "bool",
            "True when the buy side was resting. The aggressor is therefore the "
            "seller, so signed flow is negative. This is the exchange's own "
            "convention and is kept rather than pre-signed, so that the sign "
            "rule stays visible at the point of use.",
            "",
        ),
        Column(
            "trade_id",
            "int64",
            "Monotone per-symbol trade or aggregate-trade identifier. Used to "
            "detect gaps and duplicates.",
            "",
        ),
        Column(
            "source",
            "string",
            "Where the row came from, e.g. 'synthetic' or 'binance-aggtrades'. "
            "Kept per row so that mixed frames stay auditable.",
            "",
        ),
    ),
)


def signed_quantity(df: pd.DataFrame) -> pd.Series:
    """Return trade size signed by aggressor: positive for buys, negative for sells.

    Defined once, here, rather than inline in every feature. The convention is
    easy to invert by accident and an inverted sign flips the meaning of every
    order-flow feature downstream without making anything fail.
    """
    TRADE_SCHEMA.validate(df)
    sign = pd.Series(1.0, index=df.index).where(~df["is_buyer_maker"], -1.0)
    return df["quantity"] * sign


# ---------------------------------------------------------------------------
# Book plane
# ---------------------------------------------------------------------------


def bid_price_col(level: int) -> str:
    return f"bid_price_{level}"


def bid_size_col(level: int) -> str:
    return f"bid_size_{level}"


def ask_price_col(level: int) -> str:
    return f"ask_price_{level}"


def ask_size_col(level: int) -> str:
    return f"ask_size_{level}"


def level_columns(depth: int) -> tuple[str, ...]:
    """Return the price/size column names for ``depth`` levels, best level first."""
    cols: list[str] = []
    for level in range(depth):
        cols += [
            bid_price_col(level),
            bid_size_col(level),
            ask_price_col(level),
            ask_size_col(level),
        ]
    return tuple(cols)


def book_schema(depth: int) -> Schema:
    """Build the order-book contract for a given number of levels per side.

    Depth is a parameter rather than a constant because it is a property of the
    data source: an exchange stream capped at five levels and a full-depth
    reconstruction are both legitimate inputs, and a feature that needs ten
    levels should fail loudly on a five-level frame rather than quietly produce
    ``NaN``.
    """
    if depth < 1:
        raise ValueError(f"depth must be at least 1, got {depth}")

    columns: list[Column] = [
        Column(
            "timestamp",
            "datetime64[ns, UTC]",
            "Time the snapshot became observable, timezone-aware UTC.",
            "UTC",
        ),
        Column("symbol", "string", "Instrument identifier.", ""),
        Column(
            "sequence_id",
            "int64",
            "Exchange update identifier. Strictly increasing within a symbol; a "
            "jump means the local book missed updates and the segment is "
            "unreliable.",
            "",
        ),
        Column("source", "string", "Origin of the snapshot.", ""),
    ]
    for level in range(depth):
        ordinal = "best" if level == 0 else f"level {level}"
        columns += [
            Column(bid_price_col(level), "float64", f"Bid price at {ordinal}.", "quote"),
            Column(bid_size_col(level), "float64", f"Resting bid size at {ordinal}.", "base"),
            Column(ask_price_col(level), "float64", f"Ask price at {ordinal}.", "quote"),
            Column(ask_size_col(level), "float64", f"Resting ask size at {ordinal}.", "base"),
        ]

    return Schema(name=f"book_l{depth}", plane="book", columns=tuple(columns))


def infer_depth(df: pd.DataFrame) -> int:
    """Return how many complete levels per side ``df`` carries.

    A level counts only when all four of its columns are present, so a truncated
    frame reports the depth that is actually usable rather than the highest
    level number that happens to appear.
    """
    depth = 0
    while all(
        col in df.columns
        for col in (
            bid_price_col(depth),
            bid_size_col(depth),
            ask_price_col(depth),
            ask_size_col(depth),
        )
    ):
        depth += 1
    return depth


#: Convenience contract for the depth used by the synthetic generator and demo.
DEFAULT_BOOK_DEPTH: Final = 10
BOOK_SCHEMA: Final = book_schema(DEFAULT_BOOK_DEPTH)


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------
#
# These are not features. They are the handful of definitions that features,
# labels and the backtest all need to agree on, so they live with the schema
# rather than in the feature registry. `mid` in particular is the reference
# price for labelling and for mark-to-market, and those two must never drift
# apart.


def best_bid(df: pd.DataFrame) -> pd.Series:
    return df[bid_price_col(0)]


def best_ask(df: pd.DataFrame) -> pd.Series:
    return df[ask_price_col(0)]


def mid_price(df: pd.DataFrame) -> pd.Series:
    """Arithmetic mid of the best quotes."""
    return (df[bid_price_col(0)] + df[ask_price_col(0)]) / 2.0


def spread(df: pd.DataFrame) -> pd.Series:
    """Absolute quoted spread, in quote units."""
    return df[ask_price_col(0)] - df[bid_price_col(0)]


@dataclass(frozen=True)
class DatasetManifest:
    """Provenance for one stored dataset.

    Written next to the Parquet files by every command that produces data. The
    point is reproducibility: a result is only meaningful if the exact inputs
    that produced it can be identified later, and a seed is only useful if it is
    recorded alongside the generator version that consumed it.
    """

    plane: Plane
    symbol: str
    source: str
    schema_version: str
    rows: int
    start: str
    end: str
    depth: int | None = None
    seed: int | None = None
    generator: str | None = None
    trading_research_version: str | None = None
    extra: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "plane": self.plane,
            "symbol": self.symbol,
            "source": self.source,
            "schema_version": self.schema_version,
            "rows": self.rows,
            "start": self.start,
            "end": self.end,
        }
        for key, value in (
            ("depth", self.depth),
            ("seed", self.seed),
            ("generator", self.generator),
            ("trading_research_version", self.trading_research_version),
        ):
            if value is not None:
                out[key] = value
        out.update(self.extra)
        return out
