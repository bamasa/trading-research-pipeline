"""The short-horizon study's simulator settings, its market maker and its passive cells.

Three execution paths reach the event-time simulator in this study, all under
one set of settings (:func:`study_config`, the registration's ``passive_fill``
block): the signal-entry quoter (S-P, S25-Z, A0 and FILL), and S1 leaning on a
signal (S-M). The taker slots use :mod:`trading_research.backtest.taker`.

**S-M** is round one's S1 at its frozen values (:data:`S1_FROZEN`) with the
signal lean of :class:`~trading_research.pipeline.market_making.SignalLean`, a
fixed clip, a soft limit of one clip, and the regime guard (``pull``, or
``widen`` as a measurement) at a window of 60 x the regime window in hours
(:func:`lean_plan`). The guard is set by
:func:`trading_research.pipeline.regime.maker_plan` alone, from the same
:class:`~trading_research.pipeline.regime.RegimePolicy` the search's regime
axis uses, so the market maker's guard has one path. Its ``sigma_ref`` is the
median of the simulator's own ``vol_bp_1m`` over the warm-up block's rows
(:func:`sigma_ref_from_days`), fitted once and written to the fitted-values
file.

Regime flags reach either quoter as the ``flag_s`` tape (:func:`flag_tape`):
each alarm effective at the close of its hour, read strictly before each
snapshot.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import time
from functools import partial
from typing import Any

import numpy as np

from trading_research.backtest.costs import BYBIT_BASE
from trading_research.market_making.events import DayEvents
from trading_research.market_making.quoters import FLAG_SIGNAL
from trading_research.market_making.signal_entry import SignalEntryQuoter
from trading_research.market_making.signals import SignalTape
from trading_research.market_making.simulator import Cell, SimConfig, vol_at_rows
from trading_research.pipeline.market_making import MakerPlan
from trading_research.pipeline.regime import REGIME_WINDOWS_H, RegimePolicy, maker_plan

#: S1's frozen values from round one (``configs/mm_prereg.yaml``).
S1_FROZEN: dict[str, float] = {"skew_bp": 2.0, "k": 0.5, "min_edge_bp": 4.0}
#: S-M's lean neighbourhood (the cell is 1) and the regime windows, hours.
LEANS: tuple[float, ...] = (0.5, 1.0, 2.0)
GUARD_WINDOWS_H: tuple[int, ...] = REGIME_WINDOWS_H
CLIP_BTC = 0.010


def study_config(max_sequence_gaps: int, **changes: Any) -> SimConfig:
    """The registered simulator settings; ``max_sequence_gaps`` is the frozen value.

    ``changes`` are the robustness axes (fill rule, queue brackets, fee tier,
    latency), never choices.
    """
    config = SimConfig(
        clip_btc=CLIP_BTC,
        fees=BYBIT_BASE,
        order_latency_ns=10_000_000,
        cancel_latency_ns=10_000_000,
        feed_latency_ns=0,
        soft_limit_clips=1.0,
        flatten_slippage_bp=0.0,
        warmup_s=60.0,
        stop_quoting_at=time(23, 58),
        flatten_at=time(23, 59),
        suspend_after_pause_s=5.0,
        touch_window_s=3600,
        vol_half_life_s=60.0,
        markout_horizons_s=(1.0, 5.0, 10.0, 30.0),
        decomposition_horizon_s=5.0,
        max_sequence_gaps=int(max_sequence_gaps),
    )
    return config.with_(**changes) if changes else config


def sigma_ref_from_days(days: Iterable[DayEvents], config: SimConfig) -> float:
    """The median ``vol_bp_1m`` over every book row of ``days`` (one day held at a time)."""
    values = [vol_at_rows(events, config) for events in days]
    if not values:
        raise ValueError("sigma_ref needs at least one day")
    pooled = np.concatenate(values)
    pooled = pooled[np.isfinite(pooled)]
    if not len(pooled):
        raise ValueError("no row has a volatility")
    return float(np.median(pooled))


def lean_plan(
    config: SimConfig,
    sigma_ref: float,
    *,
    lean: float,
    hold_s: float,
    policy: RegimePolicy | None = None,
) -> MakerPlan:
    """S-M: S1 at its frozen values leaning on the signal, guarded under ``policy``.

    ``policy`` is ``none`` (the default), ``pull_<w>h`` or ``widen_<w>h``;
    :func:`~trading_research.pipeline.regime.maker_plan` sets the guard.
    """
    if config.clip_btc is None:
        raise ValueError("S-M runs on a fixed clip_btc")
    base = MakerPlan(
        "S1",
        config,
        dict(S1_FROZEN),
        sigma_ref=float(sigma_ref),
        lean=float(lean),
        lean_window_s=float(hold_s),
    )
    return maker_plan(base, policy or RegimePolicy())


def signal_entry_cell(
    label: str,
    config: SimConfig,
    *,
    hold_s: float,
    passive_exit: bool = False,
    cancel_on_flag: bool = False,
) -> Cell:
    """A simulator cell for the signal-entry quoter (a fresh quoter each day).

    ``cancel_on_flag`` (a slot with a regime window above 0) makes the simulator
    cancel a resting entry at the first event after each flag of the cell's
    ``flag_s`` tape (:func:`flag_tape`); the decisions themselves are guarded
    upstream (:func:`trading_research.pipeline.regime.guard_decisions`).
    """
    make = partial(SignalEntryQuoter, hold_s, passive_exit=passive_exit)
    return Cell(label, make, config.with_(cancel_entries_on_flag=cancel_on_flag))


def flag_tape(alarm_ns: np.ndarray) -> dict[str, SignalTape]:
    """The guard's tape from alarm times (ns, each the close of its hour)."""
    ts = np.unique(np.asarray(alarm_ns, dtype=np.int64))
    return {FLAG_SIGNAL: SignalTape(FLAG_SIGNAL, ts, ts / 1e9)}
