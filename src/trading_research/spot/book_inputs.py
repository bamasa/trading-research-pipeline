"""S25-BB's and S25-T's inputs, and S25-BB's label, on Bybit's spot book, from the 2025 code.

**Inputs.** S25's 40 micro-selected features (the 2025 configuration's
``features.selected``, not selected again), computed as the 2025 study
computes them: the 2025 microstructure registry
(``features/study_set.microstructure_registry``, winsorised on each feature's
trailing hour) on the spot book's rows and Bybit's spot prints, unchanged.
A warm day reads its rolling state from the previous usable day's last 62
minutes (``N`` plus a minute), as in 2025; a cold day starts at its own
midnight. ``ofi_1`` and ``ofi_20`` are divided by a touch size
(``data/feature_store.scale_raw_sizes``): for S25-BB, the median touch size of
Bybit's spot book pooled over W's rows (``data/feature_store.median_touch_size``,
the function the 2025 study's stage 1 used); for S25-T, the 2025 study's own,
the ``touch_size`` of its fitted-values file, so that its frozen fit sees its
inputs on the scale it was trained on.

**Label and forward move.** The walked-book taker label
(``labels/taker_vwap.WalkedTakerLabel``), unchanged, for the clip, at label
fees 0 and 10 bp for S25-BB (one walk per hold serves every fee, a fee ``f``
taking ``2 f`` off both sides), walking the depth the D amendment freezes (10,
or 200 if stage 0 finds the clip beyond ten levels in more than 0.1% of
snapshots); the forward move mid to mid (``labels/taker_vwap.forward_move_bp``)
for the IC. Each row also reports the exit snapshot's stamp, for the purge.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from functools import cache
from typing import Any

import numpy as np
import pandas as pd

from trading_research.data.feature_store import median_touch_size, scale_raw_sizes
from trading_research.features.study_set import RAW_SIZE_COLUMNS, microstructure_registry
from trading_research.features.timed import TimedRegistry, build_timed, stamps_ns
from trading_research.labels.directional import BUY, HOLD, SELL
from trading_research.labels.taker_vwap import WalkedTakerLabel, forward_move_bp
from trading_research.spot.print_label import forward_column, label_column

#: S25's inputs: configs/short_horizon_2025.yaml features.selected, in its order.
S25_SELECTED: tuple[str, ...] = (
    "queue_imbalance", "depth_imbalance", "log_depth_ratio", "concentration_imbalance",
    "impact_imbalance_bp", "microprice_dev_bp", "ofi_20", "slope_imbalance", "ofi_20_norm",
    "imbalance_slope_10", "ask_impact_bp", "ask_concentration", "bid_concentration",
    "bid_impact_bp", "ask_slope_bp", "ofi_1", "bid_slope_bp", "ask0_log_change_bp_1s",
    "micro_drift_5", "mid_return_20_bp", "flow_persistence_24", "flow_imbalance_24",
    "flow_imbalance_6", "mid_return_1_bp", "vwap_deviation_bp", "trade_run_length_60s",
    "ask0_log_change_bp_5s", "size_shock_10", "flow_imbalance_120", "realized_vol_50_bp",
    "turn_count_10s", "realised_vol_20", "turn_count_5s", "trade_intensity_120",
    "trade_intensity_6", "trade_intensity_24", "volatility_1s_bp", "spread_bp", "turn_count_2s",
    "quote_intensity_20",
)  # fmt: skip

#: Rolling history a warm day reads from the previous day, as in 2025: N plus a minute.
HISTORY_NS: int = (61 + 1) * 60 * 1_000_000_000

#: The key of the touch size in the 2025 study's fitted-values file.
TOUCH_SIZE_KEY = "touch_size"


@cache
def book_registry() -> TimedRegistry:
    """The 2025 microstructure registry, winsorised, unchanged."""
    return microstructure_registry()


def _tail(frame: pd.DataFrame, since_ns: int) -> pd.DataFrame:
    return frame[stamps_ns(frame) >= since_ns] if len(frame) else frame


def book_day_frame(
    rows: pd.DataFrame,
    prints: pd.DataFrame,
    *,
    names: Sequence[str] = S25_SELECTED,
    history: tuple[pd.DataFrame, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """``timestamp`` and the named inputs, unscaled, at every book row of a day.

    ``history`` is the previous usable day's ``(rows, prints)`` on a warm day.
    """
    head = 0
    if history is not None:
        since = int(stamps_ns(rows)[0]) - HISTORY_NS
        before, before_prints = _tail(history[0], since), _tail(history[1], since)
        head = len(before)
        rows = pd.concat([before, rows], ignore_index=True)
        prints = pd.concat([before_prints, prints], ignore_index=True)
    frame = build_timed(rows, list(names), prints=prints, registry=book_registry())
    return frame.iloc[head:].reset_index(drop=True)


def spot_touch_size(w_books: Iterable[pd.DataFrame]) -> float:
    """S25-BB's touch size: the median of both touch sizes over every W row."""
    return median_touch_size(w_books)


def touch_size_2025(fitted_values: Mapping[str, Any]) -> float:
    """S25-T's touch size, from the 2025 study's fitted-values file (as read)."""
    value = float(fitted_values[TOUCH_SIZE_KEY])
    if not value > 0:
        raise ValueError(f"the 2025 touch size is {value}, not a positive scale")
    return value


def scale_inputs(frame: pd.DataFrame, touch_size: float) -> pd.DataFrame:
    """``frame`` with ``ofi_1`` and ``ofi_20`` divided by ``touch_size``."""
    present = [c for c in RAW_SIZE_COLUMNS if c in frame.columns]
    return scale_raw_sizes(frame, present, touch_size)


def book_targets(
    rows: pd.DataFrame,
    holds_s: Sequence[float],
    fees_bp: Sequence[float],
    *,
    clip: float = 0.010,
    depth: int = 10,
    at: np.ndarray | None = None,
) -> pd.DataFrame:
    """Classes per hold and fee, forward moves, and the latest stamp read, at rows ``at``."""
    stamps = stamps_ns(rows)
    picked = np.arange(len(rows)) if at is None else np.asarray(at, dtype=np.int64)
    columns: dict[str, np.ndarray] = {"row": picked, "timestamp": stamps[picked]}
    reads = np.full(len(picked), -1, dtype=np.int64)
    for hold in holds_s:
        outcome = WalkedTakerLabel(hold, clip=clip, depth=depth).outcomes(rows).iloc[picked]
        long_bp, short_bp = outcome["long_bp"].to_numpy(), outcome["short_bp"].to_numpy()
        unknown = np.isnan(long_bp) | outcome["entry_exceeds"].to_numpy()
        for fee in fees_bp:
            # The 2025 rule, in its order: a crossed book could make both pay.
            label = np.full(len(picked), float(HOLD))
            label[long_bp - 2 * fee > 0] = float(BUY)
            label[short_bp - 2 * fee > 0] = float(SELL)
            label[unknown] = np.nan
            columns[label_column(hold, fee)] = label
        columns[forward_column(hold)] = forward_move_bp(rows, hold).to_numpy()[picked]
        exit_index = outcome["exit_index"].to_numpy()
        known = exit_index < len(rows)
        reads[known] = np.maximum(reads[known], stamps[exit_index[known]])
    columns["reads_until_ns"] = reads
    return pd.DataFrame(columns)
