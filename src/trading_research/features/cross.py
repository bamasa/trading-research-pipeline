"""What the rest of the market did, as an input for one instrument.

Every feature elsewhere in this project reads a single instrument's own book.
That is a strong assumption and an unexamined one: crypto perpetuals move
together, the large ones move first, and a smaller instrument that has not yet
followed a move in BTCUSDT is in a measurably different state from one that has.

Lead-lag at seconds-to-minutes is among the best-documented effects at exactly
the horizons this project works at, and it is the one source of information the
searches never had. That omission is why they are being built now.

The features
------------
* **Leader return.** The recent move of the market's largest instrument, and of
  an equal-weighted index of the universe. This is what the instrument might
  follow.
* **Catch-up.** The leader's move minus this instrument's own over the same
  window. Positive means the leader has gone up and this one has not yet, which
  is the state the lead-lag story says should resolve.
* **Beta-adjusted catch-up.** The same, with the instrument's own sensitivity to
  the leader estimated on a trailing window, so a name that moves at half the
  leader's amplitude is not permanently flagged as lagging.
* **Dispersion.** The cross-sectional spread of returns. High dispersion means
  names are moving on their own news, and relative signals mean less.
* **Relative rank.** Where this instrument sits in the cross-section of recent
  returns, as a share in [0, 1].

Causality across instruments
----------------------------
All of it is computed from returns over windows ending at the row being
described, on a shared time grid. The subtle failure mode is the grid: if one
instrument's prices are forward-filled from a stale quote while another's are
current, a "lead-lag" appears that is really a difference in update rates. Rows
where an instrument's book has not updated are therefore marked, so the caller
can drop them rather than trade a stale relationship.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

#: Windows for the cross-sectional comparisons, in rows.
WINDOWS = (6, 24, 120)


def build_cross_features(
    log_prices: pd.DataFrame,
    symbol: str,
    *,
    leader: str | None = None,
    windows: Sequence[int] = WINDOWS,
    beta_window: int = 1200,
) -> pd.DataFrame:
    """Features for ``symbol`` derived from the rest of the panel.

    ``leader`` defaults to the first column, which callers are expected to order
    by liquidity. The index return excludes ``symbol`` itself, since an index
    that contains the instrument being predicted partly predicts it by
    construction.
    """
    if symbol not in log_prices.columns:
        raise KeyError(f"{symbol} is not in the panel: {list(log_prices.columns)[:6]}")
    if log_prices.shape[1] < 3:
        raise ValueError(f"need at least three instruments, got {log_prices.shape[1]}")

    leader = leader or next(c for c in log_prices.columns if c != symbol)
    others = [c for c in log_prices.columns if c != symbol]
    out = pd.DataFrame(index=log_prices.index)

    own_all = log_prices[symbol]
    leader_all = log_prices[leader]

    for window in windows:
        own = own_all.diff(window) * 1e4
        lead = leader_all.diff(window) * 1e4
        # The index excludes this instrument; including it would let the feature
        # read part of its own target.
        index = log_prices[others].diff(window).mean(axis=1) * 1e4

        out[f"leader_return_{window}"] = lead
        out[f"index_return_{window}"] = index
        # Positive: the market moved and this instrument has not followed yet.
        out[f"catch_up_leader_{window}"] = lead - own
        out[f"catch_up_index_{window}"] = index - own
        # Dispersion: when names are moving on their own news, a relative signal
        # is weaker, whatever its size.
        out[f"dispersion_{window}"] = log_prices[others].diff(window).std(axis=1) * 1e4
        # Where this instrument sits in the cross-section, as a share.
        ranked = log_prices.diff(window).rank(axis=1, pct=True)
        out[f"relative_rank_{window}"] = ranked[symbol]

    # Sensitivity to the leader, estimated on a trailing window. Without it a
    # low-amplitude instrument reads as permanently lagging.
    own_step = own_all.diff() * 1e4
    lead_step = leader_all.diff() * 1e4
    covariance = own_step.rolling(beta_window, min_periods=beta_window // 4).cov(lead_step)
    variance = lead_step.rolling(beta_window, min_periods=beta_window // 4).var()
    beta = (covariance / variance.where(variance > 0)).clip(-5.0, 5.0)
    out["beta_to_leader"] = beta

    middle = windows[len(windows) // 2]
    out["beta_adjusted_catch_up"] = (
        beta * (leader_all.diff(middle) * 1e4) - own_all.diff(middle) * 1e4
    )

    # A row where this instrument's price has not changed at all across the
    # longest window is one whose book is probably stale; a lead-lag measured
    # against a stale quote is a measurement of the update rate.
    out["own_stale"] = (own_all.diff(windows[-1]).abs() < 1e-12).astype(float)
    return out.replace([np.inf, -np.inf], np.nan)
