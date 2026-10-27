"""What flows through the paper-trading loop: book rows, prints and gaps, and their frames.

A live stream and a replayed day produce the same three kinds of event, so
everything downstream (features, decisions, the ledger, the page) cannot tell
them apart:

* :class:`BookRow`: one row of the 100 ms grid with ten levels a side, in the
  book contract's units (prices in quote, sizes in base). ``ts_ns`` is the
  venue's stamp of the update that opened the row's interval, exactly as
  :func:`trading_research.data.bybit.reconstruct` stamps a replayed archive;
  where a venue sends no stamp (Binance spot's partial depth) it is the local
  receipt, and the page says so.
* :class:`Print`: one trade with the aggressor's side (+1 a buyer crossed, -1 a
  seller did), the convention of :mod:`trading_research.data.bybit_trades`.
* :class:`Gap`: a break in what the stream can vouch for: a sequence jump, a
  reconnect, a stale feed. Gaps are reported, never repaired.

``recv_ns`` is the local wall clock when the message arrived (equal to
``ts_ns`` for events built from stored frames that carry no receipt time).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading_research.data.schema import TRADE_SCHEMA, book_schema

#: Levels a side on every row the loop sees: what the archive replay keeps.
DEPTH = 10
#: The grid the archive replay photographs the book on.
GRID_MS = 100


@dataclass(frozen=True, slots=True)
class BookRow:
    """One grid row: the book right after the first update of its 100 ms interval."""

    ts_ns: int
    recv_ns: int
    sequence_id: int
    bid_px: tuple[float, ...]
    bid_sz: tuple[float, ...]
    ask_px: tuple[float, ...]
    ask_sz: tuple[float, ...]

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid_px[0] + self.ask_px[0])

    @property
    def spread_bp(self) -> float:
        return (self.ask_px[0] - self.bid_px[0]) / self.mid * 1e4


@dataclass(frozen=True, slots=True)
class Print:
    """One trade, signed by its aggressor."""

    ts_ns: int
    recv_ns: int
    price: float
    size: float
    aggressor: int
    trade_id: str
    #: Bybit's retail-price-improvement prints: no order of ours could match them,
    #: so they never fill a paper order (they stay in the features).
    rpi: bool = False


@dataclass(frozen=True, slots=True)
class Gap:
    """A stretch the stream cannot vouch for, with what was seen on either side."""

    ts_ns: int
    recv_ns: int
    kind: str  # "sequence", "trade_sequence", "reconnect", "stale", "late_print"
    detail: str


Event = BookRow | Print | Gap


def _ns(stamps: pd.Series) -> np.ndarray:
    index = pd.DatetimeIndex(stamps).as_unit("ns")
    return np.asarray(index.to_numpy(dtype="datetime64[ns]").view(np.int64))


def rows_to_frame(
    rows: Sequence[BookRow], *, symbol: str, source: str, depth: int = DEPTH
) -> pd.DataFrame:
    """Book rows as a frame in the book contract (``book_schema(depth)``)."""
    n = len(rows)
    data: dict[str, object] = {
        "timestamp": pd.to_datetime(np.array([r.ts_ns for r in rows], dtype=np.int64), utc=True),
        "symbol": pd.array([symbol] * n, dtype="string"),
        "sequence_id": np.array([r.sequence_id for r in rows], dtype=np.int64),
        "source": pd.array([source] * n, dtype="string"),
    }
    for side, attr_px, attr_sz in (("bid", "bid_px", "bid_sz"), ("ask", "ask_px", "ask_sz")):
        prices = np.array([getattr(r, attr_px) for r in rows], dtype=np.float64).reshape(n, -1)
        sizes = np.array([getattr(r, attr_sz) for r in rows], dtype=np.float64).reshape(n, -1)
        for level in range(depth):
            data[f"{side}_price_{level}"] = prices[:, level] if n else np.zeros(0)
            data[f"{side}_size_{level}"] = sizes[:, level] if n else np.zeros(0)
    frame = pd.DataFrame(data)
    ordered = list(book_schema(depth).column_names)
    return frame[ordered]


def frame_to_rows(frame: pd.DataFrame, *, depth: int = DEPTH) -> list[BookRow]:
    """A book frame (contract columns) as rows; ``recv_ns`` is the stamp."""
    stamps = _ns(frame["timestamp"])
    sequence = (
        frame["sequence_id"].to_numpy(dtype=np.int64)
        if "sequence_id" in frame.columns
        else np.arange(len(frame), dtype=np.int64)
    )
    planes = {
        name: np.column_stack(
            [frame[f"{name.split('_')[0]}_{kind}_{lv}"].to_numpy("float64") for lv in range(depth)]
        )
        for name, kind in (
            ("bid_px", "price"),
            ("bid_sz", "size"),
            ("ask_px", "price"),
            ("ask_sz", "size"),
        )
    }
    out = []
    for i, ts in enumerate(stamps.tolist()):
        out.append(
            BookRow(
                int(ts),
                int(ts),
                int(sequence[i]),
                tuple(planes["bid_px"][i].tolist()),
                tuple(planes["bid_sz"][i].tolist()),
                tuple(planes["ask_px"][i].tolist()),
                tuple(planes["ask_sz"][i].tolist()),
            )
        )
    return out


def prints_to_frame(prints: Sequence[Print]) -> pd.DataFrame:
    """Prints in the Bybit print contract: ``timestamp``, ``price``, ``size``,
    ``aggressor`` and ``match_id`` (what :mod:`trading_research.features.prints` reads)."""
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                np.array([p.ts_ns for p in prints], dtype=np.int64), utc=True
            ),
            "price": np.array([p.price for p in prints], dtype=np.float64),
            "size": np.array([p.size for p in prints], dtype=np.float64),
            "aggressor": np.array([p.aggressor for p in prints], dtype=np.int64),
            "match_id": pd.array([p.trade_id for p in prints], dtype="string"),
            "rpi": np.array([p.rpi for p in prints], dtype=bool),
        }
    )


def prints_to_trades(prints: pd.DataFrame, *, symbol: str, source: str) -> pd.DataFrame:
    """Prints (Bybit contract) as the trade plane (``TRADE_SCHEMA``).

    ``is_buyer_maker`` is True when the seller crossed. ``trade_id`` is the
    venue's identifier when it is an integer (Binance's aggregate id) and the
    print's position in the frame otherwise (Bybit's ids are strings).
    """
    ids = prints["match_id"].astype("string") if "match_id" in prints.columns else None
    numeric = ids is not None and bool(ids.str.fullmatch(r"\d+").fillna(False).all())
    trade_id = (
        ids.astype("int64").to_numpy() if numeric and ids is not None else np.arange(len(prints))
    )
    out = pd.DataFrame(
        {
            "timestamp": prints["timestamp"],
            "symbol": pd.array([symbol] * len(prints), dtype="string"),
            "price": prints["price"].astype("float64"),
            "quantity": prints["size"].astype("float64"),
            "is_buyer_maker": prints["aggressor"].to_numpy() < 0,
            "trade_id": np.asarray(trade_id, dtype=np.int64),
            "source": pd.array([source] * len(prints), dtype="string"),
        }
    )
    TRADE_SCHEMA.validate(out)
    return out


def trades_to_prints(trades: pd.DataFrame) -> pd.DataFrame:
    """The trade plane (``TRADE_SCHEMA``) as prints in the Bybit contract."""
    return pd.DataFrame(
        {
            "timestamp": trades["timestamp"],
            "price": trades["price"].astype("float64"),
            "size": trades["quantity"].astype("float64"),
            "aggressor": np.where(trades["is_buyer_maker"].to_numpy(dtype=bool), -1, 1),
            "match_id": trades["trade_id"].astype("int64").astype("string"),
        }
    )


def frame_to_prints(prints: pd.DataFrame) -> list[Print]:
    """A print frame (Bybit contract) as events; ``recv_ns`` is the stamp."""
    stamps = _ns(prints["timestamp"])
    ids = (
        prints["match_id"].astype("string").fillna("").tolist()
        if "match_id" in prints.columns
        else [str(i) for i in range(len(prints))]
    )
    price = prints["price"].to_numpy(dtype=np.float64).tolist()
    size = prints["size"].to_numpy(dtype=np.float64).tolist()
    side = prints["aggressor"].to_numpy(dtype=np.int64).tolist()
    rpi = (
        prints["rpi"].to_numpy(dtype=bool).tolist()
        if "rpi" in prints.columns
        else [False] * len(prints)
    )
    return [
        Print(int(t), int(t), price[i], size[i], int(side[i]), str(ids[i]), bool(rpi[i]))
        for i, t in enumerate(stamps.tolist())
    ]


def merge_by_time(rows: Sequence[BookRow], prints: Sequence[Print]) -> list[Event]:
    """Rows and prints in stamp order; a print stamped at a row's stamp comes first.

    A print belongs to the first row stamped at or after it (the backward rule of
    :mod:`trading_research.features.prints`), so it must be known when that row
    is processed.
    """
    keyed: list[tuple[int, int, int, Event]] = []
    for i, p in enumerate(prints):
        keyed.append((p.ts_ns, 0, i, p))
    for i, r in enumerate(rows):
        keyed.append((r.ts_ns, 1, i, r))
    keyed.sort(key=lambda item: (item[0], item[1], item[2]))
    return [item[3] for item in keyed]
