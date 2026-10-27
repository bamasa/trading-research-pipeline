"""The repository's two reversion rules, scored against a planted cross-section.

* :func:`relative_trades` runs :func:`trading_research.strategies.relative.backtest`
  (long the most depressed names, short the most extended, demeaned, every
  ``hold`` bars) and scores each basket against the planted transients;
* :func:`index_trades` runs the frozen rule of
  :mod:`trading_research.strategies.reversion` (trade each instrument against the
  equal-weighted index of the others, threshold set on the first days, one
  position at a time through :func:`trading_research.backtest.execution.thin`)
  on the remaining days.

Each trade carries the recovered gross (its realised log move, signed, in bp),
the planted gross (what the latent transients owed that position over that
hold, :meth:`CrossSectionSpec.expected_move`) and their difference, and the
statistics are clustered by day.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading_research.synthetic.cross_section import CrossSectionDataset


@dataclass(frozen=True)
class ReversionSettings:
    """What the two rules are run with; costs are stated per name and round trip."""

    cost_bp: float = 12.0
    basket: int = 3
    #: Rows of lookback and hold for the relative rule; None uses the half-life.
    relative_rows: int | None = None
    #: Days the index rule's threshold is set on, before the days it trades.
    train_days: int = 3


def relative_trades(
    dataset: CrossSectionDataset, settings: ReversionSettings | None = None
) -> pd.DataFrame:
    """One row per basket the relative rule formed, scored against the truth."""
    from trading_research.strategies.relative import CrossSection, backtest, deviations, weights

    settings = settings or ReversionSettings()
    spec = dataset.spec
    rows = settings.relative_rows or max(2, round(spec.half_life_bars))
    config = CrossSection(lookback=rows, basket=settings.basket, hold=rows, demean=True)
    panel = dataset.log_prices
    result = backtest(panel, config, cost_bp_per_name=settings.cost_bp)
    if result.empty:
        return result
    basket = weights(deviations(panel, config), config).to_numpy()
    owed = spec.expected_move(dataset.transient.to_numpy(), rows) * 1e4
    at = result["at"].to_numpy(dtype=np.int64)
    result["planted_bp"] = np.sum(basket[at] * owed[at], axis=1)
    result["residual_bp"] = result["gross_bp"] - result["planted_bp"]
    result["day"] = dataset.day_of_row(at)
    return result


def index_trades(
    dataset: CrossSectionDataset, settings: ReversionSettings | None = None
) -> pd.DataFrame:
    """The frozen index-reversion rule on each instrument, after its training days."""
    from trading_research.backtest.execution import ThinningRules, thin
    from trading_research.strategies.reversion import (
        ReversionConfig,
        decide,
        index_level,
        signal,
        threshold_for_rate,
    )

    settings = settings or ReversionSettings()
    spec = dataset.spec
    config = ReversionConfig(exclude=())
    values = dataset.log_prices.to_numpy()
    owed = spec.expected_move(dataset.transient.to_numpy(), config.hold) * 1e4
    cut = settings.train_days * spec.rows_per_day
    if cut >= len(values) - spec.rows_per_day:
        raise ValueError("the index rule needs at least one day after its training days")
    rules = ThinningRules(hold_periods=config.hold, cooldown_periods=config.cooldown)
    parts = []
    for j, symbol in enumerate(dataset.log_prices.columns):
        level = index_level(values, exclude=j)
        raw = signal(level, config)
        threshold = threshold_for_rate(raw[:cut], spec.rows_per_day, config)
        decision = decide(raw[cut:], threshold)
        forward = np.full(len(values) - cut, np.nan)
        price = values[cut:, j]
        forward[: -config.hold] = (price[config.hold :] - price[: -config.hold]) * 1e4
        trades = thin(decision, forward, np.zeros(len(forward)), rules)
        if not trades:
            continue
        entry = np.array([t.entry_index for t in trades], dtype=np.int64) + cut
        side = np.array([t.direction for t in trades], dtype=np.float64)
        gross = side * np.array([t.move_bp for t in trades])
        parts.append(
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "at": entry,
                    "side": side,
                    "gross_bp": gross,
                    "planted_bp": side * owed[entry, j],
                    "net_bp": gross - settings.cost_bp,
                    "threshold_bp": threshold,
                }
            )
        )
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out["residual_bp"] = out["gross_bp"] - out["planted_bp"]
    out["day"] = dataset.day_of_row(out["at"].to_numpy())
    return out


def clustered(trades: pd.DataFrame, value: str, series: str | None = None) -> tuple[float, float]:
    """Day-clustered mean and t of ``value``; NaN with fewer than two days."""
    from trading_research.evaluation.significance import SignificanceError, assess

    if trades.empty:
        return float("nan"), float("nan")
    try:
        result = assess(trades, value=value, cluster="day", series=series)
    except SignificanceError:
        return float(trades[value].mean()), float("nan")
    return result.cluster_mean_bp, result.cluster_t


def summary(trades: pd.DataFrame, series: str | None = None) -> dict[str, float]:
    if trades.empty:
        return {"trades": 0.0}
    out = {"trades": float(len(trades)), "days": float(trades["day"].nunique())}
    for name, column in (
        ("recovered", "gross_bp"),
        ("planted", "planted_bp"),
        ("residual", "residual_bp"),
        ("net", "net_bp"),
    ):
        mean, t = clustered(trades, column, series)
        out[f"{name}_bp"] = mean
        out[f"{name}_t"] = t
    means = trades.groupby("day")["residual_bp"].mean().to_numpy(dtype=np.float64)
    out["residual_se_bp"] = (
        float(np.std(means, ddof=1) / np.sqrt(len(means))) if len(means) > 1 else float("nan")
    )
    return out
