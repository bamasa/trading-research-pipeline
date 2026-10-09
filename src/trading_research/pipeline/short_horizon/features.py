"""The study's inputs and targets for one day: features for the stores, labels for the fits.

**Features.** The 58 timed microstructure features (``features/study_set``,
winsorised on their own trailing hour) and the ten-level tensor scaled by W's
calibration, one row per book row. Rolling state may read the previous
calendar day when it is usable and in the same block or the adjacent earlier
one (W before D); otherwise the day starts **cold** and its first N minutes are
never scored (:func:`scored_from_ns`). Raw-size columns are scaled by W's
touch size when a store writes them.

**Targets**, from the book's float64 prices at the every-tenth rows the fits
take their targets on: for each hold, the walked-book taker label's long and
short P&L (a label fee ``f`` takes ``2 f`` off both, so one walk serves every
fee), the class at each fee, and the forward move in bp.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date, timedelta
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from trading_research.data import study_access as sa
from trading_research.features.lob_tensor import TensorCalibration, lob_tensor, tensor_columns
from trading_research.features.study_set import RAW_SIZE_COLUMNS, microstructure_registry
from trading_research.features.timed import TimedRegistry, build_timed, stamps_ns
from trading_research.labels.directional import BUY, HOLD, SELL
from trading_research.labels.taker_vwap import WalkedTakerLabel, forward_move_bp
from trading_research.pipeline.short_horizon.plan import (
    LONGEST_LOOKBACK_MINUTES,
    NS_PER_S,
    TARGET_STRIDE,
    midnight_ns,
)

TIME = "timestamp"
#: Rolling history read from the previous day: N plus a minute of margin.
HISTORY_NS = (LONGEST_LOOKBACK_MINUTES + 1) * 60 * NS_PER_S
TENSOR = tuple(tensor_columns())
RAW_SIZES = tuple(RAW_SIZE_COLUMNS)
#: Blocks whose previous day a day may read rolling state from (the same block,
#: and W before D); every other first day is cold.
_ADJACENT = {"D": {"D", "W"}, "W": {"W"}, "H": {"H"}, "F": {"F"}, "P": {"P", "F"}}


@cache
def registry() -> TimedRegistry:
    return microstructure_registry()


def micro_columns() -> tuple[str, ...]:
    return tuple(registry().names())


def previous_day(day: date, usable: Sequence[date]) -> date | None:
    """The day whose rolling state ``day`` may read, or None when ``day`` is cold."""
    before = day - timedelta(days=1)
    block, earlier = sa.block_of(day), sa.block_of(before)
    if before not in set(usable) or block is None or earlier not in _ADJACENT.get(block, set()):
        return None
    if day in sa.COLD_START_DAYS:
        return None
    return before


def scored_from_ns(day: date, cold: bool) -> int:
    """The first stamp a decision may be scored at: N minutes after a cold midnight."""
    start = midnight_ns(day)
    return start + LONGEST_LOOKBACK_MINUTES * 60 * NS_PER_S if cold else start


def _tail(frame: pd.DataFrame, since_ns: int) -> pd.DataFrame:
    return frame[stamps_ns(frame) >= since_ns] if len(frame) else frame


def day_frame(
    rows: pd.DataFrame,
    prints: pd.DataFrame,
    calibration: TensorCalibration,
    history: tuple[pd.DataFrame, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Every input of one day, one row per book row, with its ``timestamp``."""
    head = 0
    if history is not None:
        since = stamps_ns(rows)[0] - HISTORY_NS
        before, before_prints = _tail(history[0], since), _tail(history[1], since)
        head = len(before)
        rows = pd.concat([before, rows], ignore_index=True)
        prints = pd.concat([before_prints, prints], ignore_index=True)
    reg = registry()
    micro = build_timed(rows, reg.names(), prints=prints, registry=reg)
    tensor = lob_tensor(rows, calibration)
    frame = pd.concat([micro.reset_index(drop=True), tensor.reset_index(drop=True)], axis=1)
    frame.insert(0, TIME, rows[TIME].reset_index(drop=True))
    return frame.iloc[head:].reset_index(drop=True)


def target_rows(n_rows: int) -> np.ndarray:
    """Positions 0, 10, 20, ... of a day's rows: where the fits take targets."""
    return np.arange(0, n_rows, TARGET_STRIDE, dtype=np.int64)


def label_column(hold_s: float, fee_bp: float) -> str:
    return f"label_{hold_s:g}_{fee_bp:g}"


def forward_column(hold_s: float) -> str:
    return f"forward_{hold_s:g}"


def day_targets(
    rows: pd.DataFrame, holds_s: Sequence[float], fees_bp: Sequence[float], clip: float
) -> pd.DataFrame:
    """Every target at the every-tenth rows: classes per hold and fee, forward moves,
    and the market gate's two inputs (the rows a fold's gate levels are fitted on)."""
    from trading_research.search_engine.market_gate import gate_inputs

    at = target_rows(len(rows))
    columns: dict[str, np.ndarray] = {"row": at, TIME: stamps_ns(rows)[at]}
    gates = gate_inputs(rows).iloc[at]
    for name in gates.columns:
        columns[name] = gates[name].to_numpy(dtype=np.float64)
    for hold in holds_s:
        outcome = WalkedTakerLabel(hold, clip=clip).outcomes(rows).iloc[at]
        long_bp, short_bp = outcome["long_bp"].to_numpy(), outcome["short_bp"].to_numpy()
        unknown = np.isnan(long_bp) | outcome["entry_exceeds"].to_numpy()
        for fee in fees_bp:
            label = np.full(len(at), float(HOLD))
            label[long_bp - 2 * fee > 0] = float(BUY)
            label[short_bp - 2 * fee > 0] = float(SELL)
            label[unknown] = np.nan
            columns[label_column(hold, fee)] = label
        columns[forward_column(hold)] = forward_move_bp(rows, hold).to_numpy()[at]
    return pd.DataFrame(columns)


def read_or_build(path: Path, build: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    """A cached day's frame, built once."""
    if path.exists():
        return pd.read_parquet(path)
    frame = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    frame.to_parquet(partial, compression="zstd", index=False)
    partial.replace(path)
    return frame
