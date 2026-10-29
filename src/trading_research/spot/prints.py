"""The print contract both venues' parsers write, and its check.

One row per print (Bybit) or aggregated print (Binance):

==============  =======================  ==============================================
Column          Type                     Meaning
==============  =======================  ==============================================
``timestamp``   ``datetime64[ns, UTC]``  the venue's match time
``price``       ``float64``              quote per BTC, positive
``size``        ``float64``              BTC, positive
``aggressor``   ``int8``                 +1 buyer-initiated, -1 seller-initiated
``trade_id``    ``int64``                the archive's own id
``rpi``         ``bool``                 a print no order can reach (never filled
                                         against; kept in the features)
``source``      ``string``               the archive's tag
==============  =======================  ==============================================

Rows are in time order, sorted stably, so prints sharing a stamp keep the
order the archive wrote them in: the print rule's sweep walks them in that
order. A frame that breaks the contract raises :class:`PrintContractError`,
which the fetch records as a replay failure: the day is unreadable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNS: dict[str, str] = {
    "timestamp": "datetime64[ns, UTC]",
    "price": "float64",
    "size": "float64",
    "aggressor": "int8",
    "trade_id": "int64",
    "rpi": "bool",
    "source": "string",
}

BYBIT_SOURCE = "bybit-spot-prints"
BINANCE_SOURCE = "binance-spot-aggtrades"


class PrintContractError(ValueError):
    """A parsed day of prints does not satisfy the contract."""


def check(frame: pd.DataFrame) -> pd.DataFrame:
    """Return ``frame`` unchanged if it satisfies the contract; raise otherwise."""
    if list(frame.columns) != list(COLUMNS):
        raise PrintContractError(f"columns {list(frame.columns)}, expected {list(COLUMNS)}")
    wrong = {c: str(frame[c].dtype) for c, t in COLUMNS.items() if str(frame[c].dtype) != t}
    if wrong:
        raise PrintContractError(f"column types {wrong}, expected {COLUMNS}")
    if frame.empty:
        return frame
    if not frame["aggressor"].isin((-1, 1)).all():
        raise PrintContractError("an aggressor is neither +1 nor -1")
    for column in ("price", "size"):
        values = frame[column].to_numpy()
        if not (np.isfinite(values).all() and (values > 0).all()):
            raise PrintContractError(f"a {column} is not a positive finite number")
    if not frame["timestamp"].is_monotonic_increasing:
        raise PrintContractError("timestamps are not in time order")
    return frame


def assemble(
    stamps: pd.Series,
    price: pd.Series,
    size: pd.Series,
    aggressor: np.ndarray,
    trade_id: pd.Series,
    rpi: np.ndarray,
    source: str,
) -> pd.DataFrame:
    """Build a contract frame from parsed columns, in time order, ties in input order."""
    frame = pd.DataFrame(
        {
            "timestamp": pd.DatetimeIndex(stamps).as_unit("ns"),
            "price": np.asarray(price, dtype=np.float64),
            "size": np.asarray(size, dtype=np.float64),
            "aggressor": np.asarray(aggressor, dtype=np.int8),
            "trade_id": np.asarray(trade_id, dtype=np.int64),
            "rpi": np.asarray(rpi, dtype=bool),
            "source": pd.array([source] * len(price), dtype="string"),
        }
    )
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    return check(frame)


def empty(source: str) -> pd.DataFrame:
    """A day without a print, in the contract's columns and types."""
    return assemble(
        pd.Series(pd.to_datetime([], utc=True)),
        pd.Series([], dtype=np.float64),
        pd.Series([], dtype=np.float64),
        np.zeros(0, dtype=np.int8),
        pd.Series([], dtype=np.int64),
        np.zeros(0, dtype=bool),
        source,
    )
