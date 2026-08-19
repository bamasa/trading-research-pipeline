"""Who is holding the risk, and what it costs them to hold it.

Every feature until now was computed from the book or the tape: what is quoted,
what traded. Neither says how much is *at risk* or who is carrying it, and those
are different questions with different answers. A thin book and a crowded
position unwinding produce similar prints and are not the same mechanism.

Binance publishes the missing half free, in the same archive as everything else:

``sum_open_interest``
    Contracts outstanding. Rising open interest with a rising price is new money
    taking a side; falling open interest with a rising price is shorts covering,
    and the two usually end differently.
``count_long_short_ratio`` and the top-trader variants
    How accounts are positioned, and separately how the *largest* accounts are.
    The gap between them is the one number here that is hard to get anywhere
    else: retail and size disagreeing is a different state from both leaning the
    same way.
``sum_taker_long_short_vol_ratio``
    Whether the aggressive flow is buying or selling.
``funding_rate``
    What a long pays a short every eight hours. Not a prediction — it is a fact
    about carry, and a large one is the market saying which side is crowded.

The frequency is the constraint
-------------------------------
Metrics arrive every five minutes and funding every eight hours, against a grid
of a tenth of a second. So these are *context*, not triggers: at a two-minute
horizon a metrics value is up to five minutes stale, and any feature pretending
to be timely on it would be fitting the staleness.

Everything here is therefore a level, a change over several observations, or a
position within a trailing distribution — quantities that mean the same thing
whether they are thirty seconds or three minutes old. Nothing differences two
adjacent observations, because at this resolution that measures the sampling
grid rather than the market.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _z(values: pd.Series, window: int) -> pd.Series:
    """Position within a trailing distribution, shifted so a row is excluded
    from its own reference."""
    # Capped at the window itself: a quarter of a short window rounds below
    # three, and pandas refuses a min_periods larger than the window it applies
    # to. Found by a test using a two-observation window.
    rolling = values.shift(1).rolling(window, min_periods=min(window, max(3, window // 4)))
    centre = rolling.median()
    spread = (rolling.quantile(0.75) - rolling.quantile(0.25)) / 1.349
    return (values - centre) / spread.where(spread > 0)


def build_positioning_features(
    metrics: pd.DataFrame,
    *,
    funding: pd.DataFrame | None = None,
    windows: tuple[int, ...] = (12, 72, 288),
) -> pd.DataFrame:
    """Positioning features on the metrics grid, ready to be joined to a book.

    ``windows`` are in observations, so on a five-minute grid they are one hour,
    six hours and a day. Three scales because a position that is crowded
    relative to the last hour and normal relative to the day is a different
    state from one crowded on both.

    The frame keeps its own timestamps; joining it to the book plane is
    :func:`trading_research.features.align.join_asof`'s job, and it must be a
    backward join — a five-minute metric published at 12:05 is not knowable at
    12:03.
    """
    if "timestamp" not in metrics.columns:
        raise KeyError("metrics frame needs a timestamp column")

    out = pd.DataFrame({"timestamp": metrics["timestamp"]})
    oi = metrics.get("sum_open_interest")

    if oi is not None:
        # Log, because open interest is a level that grows: a change of ten
        # thousand contracts means something different at fifty thousand and at
        # five hundred thousand.
        out["log_open_interest"] = np.log(oi.where(oi > 0))
        for window in windows:
            out[f"open_interest_change_{window}"] = out["log_open_interest"] - out[
                "log_open_interest"
            ].shift(window)
            out[f"open_interest_z{window}"] = _z(out["log_open_interest"], window)

    for name, column in (
        ("crowd", "count_long_short_ratio"),
        ("size", "sum_toptrader_long_short_ratio"),
        ("size_accounts", "count_toptrader_long_short_ratio"),
        ("taker", "sum_taker_long_short_vol_ratio"),
    ):
        values = metrics.get(column)
        if values is None:
            continue
        # Logged so that 2.0 and 0.5 — twice as many longs, twice as many
        # shorts — are equal and opposite rather than 1.0 apart and 0.5 apart.
        logged = np.log(values.where(values > 0))
        out[f"log_{name}_ratio"] = logged
        for window in windows:
            out[f"{name}_ratio_z{window}"] = _z(logged, window)

    if "log_crowd_ratio" in out and "log_size_ratio" in out:
        # The one quantity here that is genuinely hard to obtain elsewhere:
        # whether the crowd and the large accounts are on the same side.
        out["crowd_minus_size"] = out["log_crowd_ratio"] - out["log_size_ratio"]
        for window in windows:
            out[f"crowd_minus_size_z{window}"] = _z(out["crowd_minus_size"], window)

    if funding is not None and not funding.empty:
        rate = funding[["timestamp", "funding_rate"]].sort_values("timestamp")
        merged = pd.merge_asof(
            out.sort_values("timestamp"), rate, on="timestamp", direction="backward"
        )
        # In basis points, matching every other cost in the project, and per
        # eight-hour interval as published rather than annualised — the horizon
        # here is minutes, and an annual figure would be a different quantity
        # wearing the same name.
        merged["funding_bp"] = merged["funding_rate"] * 1e4
        merged["funding_bp_z288"] = _z(merged["funding_bp"], 288)
        out = merged.drop(columns="funding_rate")

    return out
