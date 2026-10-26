"""The original's gate in basis points and its one-trade-per-event rule.

Re-typed from the author's 2025 strategy layer (its ``futures_taker`` scenario,
which the backtested signals came from). On the rows that pass the volatility
filter, in time order:

* the strategy inputs are ``rv_bp = 1e4 * rv_50`` and
  ``spread_bp = 1e4 * spread_rolling_10 / mid_mean_50``, both clipped at zero;
  as stored, all three are rolling z-scores, so these are not basis points,
  and the replication keeps them as the original computed them;
* a row trades up if ``(p_up - p_down) * rv_bp - cost`` beats the mirror, where
  ``cost = 2 * 5 + spread_bp + 3``, and holds if the better of the two is at
  most ``buffer_bp``, if ``1 - p_flat < tau_move``, if
  ``|p_up - p_down| < min_margin``, or if ``rv_bp / max(spread_bp, 1e-9)`` is
  below ``t_min``;
* consecutive rows with one label and no more than ``event_max_gap`` whole
  seconds between them are one event; only the first row of an up or down
  event trades, and none starts within ``cooldown_seconds`` of the end of the
  last event that traded.

The original grouped events after an unstable sort on whole seconds, which can
reorder the rows inside a second; here the rows keep their time order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: The original's taker scenario: fee per side and slippage, in its units.
FEE_BP_PER_SIDE, SLIPPAGE_BP = 5.0, 3.0
DOWN, FLAT, UP = 0, 1, 2


@dataclass(frozen=True)
class GateConfig:
    """One configuration of the gate; ``name`` says where the original ran it."""

    name: str
    tau_move: float
    min_margin: float
    buffer_bp: float
    t_min: float
    event_max_gap: float
    cooldown_seconds: float


def strategy_inputs(
    rv_50: np.ndarray, mid_mean_50: np.ndarray, spread_rolling_10: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``(rv_bp, spread_bp)`` from the stored columns of consecutive rows."""
    rv_bp = np.round(np.clip(np.nan_to_num(1e4 * rv_50), 0.0, None), 6)
    ref = np.where(mid_mean_50 == 0, np.nan, mid_mean_50)
    spread = 1e4 * spread_rolling_10 / (ref + 1e-12)
    valid = np.isfinite(spread)
    if not valid.any():
        return rv_bp, np.zeros_like(spread)
    index = np.where(valid, np.arange(len(spread)), 0)
    index = np.maximum.accumulate(index)  # forward fill
    filled = spread[index]
    filled[: np.argmax(valid)] = spread[np.argmax(valid)]  # back fill the head
    return rv_bp, np.round(np.clip(filled, 0.0, None), 3)


def gate(
    probs: np.ndarray, rv_bp: np.ndarray, spread_bp: np.ndarray, cfg: GateConfig
) -> np.ndarray:
    """0 down / 1 hold / 2 up per row, before the event rule."""
    p = np.clip(probs.astype(np.float64), 1e-12, 1.0)
    p /= p.sum(axis=1, keepdims=True)
    edge = p[:, UP] - p[:, DOWN]
    cost = 2.0 * FEE_BP_PER_SIDE + spread_bp + SLIPPAGE_BP
    ev_up, ev_down = edge * rv_bp - cost, -edge * rv_bp - cost
    out = np.where(ev_up > ev_down, UP, DOWN)
    tradable = rv_bp / np.maximum(spread_bp, 1e-9) >= cfg.t_min
    hold = np.maximum(ev_up, ev_down) <= cfg.buffer_bp
    hold |= (1.0 - p[:, FLAT]) < cfg.tau_move
    hold |= np.abs(edge) < cfg.min_margin
    hold |= ~tradable
    out[hold] = FLAT
    return out


def one_per_event(labels: np.ndarray, seconds: np.ndarray, cfg: GateConfig) -> np.ndarray:
    """``labels`` with every row but each traded event's first set to hold."""
    labels = np.asarray(labels)
    n = len(labels)
    out = np.full(n, FLAT, dtype=labels.dtype)
    if n == 0:
        return out
    gap = np.diff(seconds) > cfg.event_max_gap
    starts = np.flatnonzero(np.concatenate([[True], (labels[1:] != labels[:-1]) | gap]))
    ends = np.concatenate([starts[1:] - 1, [n - 1]])
    last_close = -np.inf
    for start, end in zip(starts, ends, strict=True):
        if labels[start] == FLAT or seconds[start] < last_close + cfg.cooldown_seconds:
            continue
        out[start] = labels[start]
        last_close = seconds[end]
    return out
