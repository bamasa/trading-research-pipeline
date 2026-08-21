"""A refuted candidate, kept with the number that refuted it.

This looked like the one thing in the project that worked, and it does not. It
is kept — parameters frozen, result attached — because a killed candidate in the
open is worth more than a deleted one: the next person to find a ten-minute
reversion in crypto perpetuals can read what happened to this one.

The register in ``docs/findings.md`` carries the full account.

The claim
---------
An equal-weighted index of USDT perpetuals overshoots over roughly ten minutes
and comes back. The index's recent return therefore predicts the *next* move of
each constituent with a negative sign, and trading against it at a ten-minute
holding period clears the taker round trip on the more volatile instruments.

What has been checked
---------------------
* **Consistency.** Negative information coefficient on 26 of 26 instruments,
  on both halves of the data. Median about -0.06 at a two-minute horizon and
  -0.098 at ten minutes.
* **Not a measurement artefact.** Shared price noise between the end of a
  lookback window and the start of a forward window manufactures negative
  correlation from nothing, and such an artefact dies the instant a gap is
  inserted between the windows. This one strengthens slightly out to a
  thirty-second gap and then decays over about ten minutes, which is an
  economic shape rather than an accounting one.
* **Market-wide, not cross-sectional.** Against a market-neutral target the
  coefficient collapses from -0.056 to +0.004 and the instruments stop agreeing.
  This matters for sizing: 26 instruments carry *one* bet, so trading them all
  is leverage rather than diversification.
* **Structure repeats across blocks.** The holding-period profile is the same
  on the search and held-out blocks — two minutes loses, ten minutes is best,
  longer decays — and configuration ranks correlate across blocks at +0.35.
* **The instrument ordering follows the identity.** Edge per trade is roughly
  IC times the dispersion of the move, so high-volatility alts should beat
  BTCUSDT and ETHUSDT at similar cost. They do, in that order.

What killed it
--------------
The test the whole project is built around: the same frozen configuration, with
thresholds carried over unchanged, on days neither block had seen. Run on 12
March to 20 April 2024, immediately after the original span.

Nothing was positive. Median -19.94 bp per trade at the frozen ten-minute hold,
0 instruments of 26, 3,637 trades, -69,283 bp in total.

Both kill conditions stated in advance fired, and the second is the informative
one. A median below zero is what a decayed edge looks like; the *profile
inverted* — the best holding period is now the shortest tested rather than ten
minutes, and at the frozen horizon the gross edge changed sign from +16.6 bp to
-5.31, before any cost. The shape that repeated across two blocks did not
survive a third.

The warning was already recorded while this was a candidate: a sixfold decay
between the original blocks, +17.2 bp median falling to +2.8. That is the same
signature §25 documented — a structure that looks like an interior optimum and
is a period.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

#: Seconds per row on the grid these numbers were measured on. Every window
#: below is a count of those rows, so changing the grid changes their meaning.
GRID_SECONDS = 5


@dataclass(frozen=True)
class ReversionConfig:
    """The frozen parameters, in rows of :data:`GRID_SECONDS` seconds.

    Frozen deliberately. The values are what the search block chose and what the
    held-out block was then read with; re-tuning them against a later result
    would turn the one honest test this candidate still has to pass into
    another selection step.
    """

    #: Rows of index history the signal reads.
    lookback: int = 120
    #: Rows a position is held. The axis that decided everything: the move grows
    #: with the square root of the horizon and the cost of a round trip does
    #: not grow at all, so two minutes loses and ten minutes clears.
    hold: int = 120
    #: Rows to stand aside after closing, so one signal is not traded twice.
    cooldown: int = 120
    #: Target trades a day, applied by taking the strongest signals.
    trades_per_day: float = 60.0
    #: Instruments whose dispersion is too small to clear their cost. Measured,
    #: not assumed: both were negative on the held-out block while twenty of
    #: the other twenty-four were positive.
    exclude: tuple[str, ...] = ("BTCUSDT", "ETHUSDT")

    @property
    def lookback_seconds(self) -> int:
        return self.lookback * GRID_SECONDS

    @property
    def hold_seconds(self) -> int:
        return self.hold * GRID_SECONDS

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


#: What the frozen configuration produced on days nothing had chosen. This is
#: the number that settled it, kept beside the one that did not.
FRESH_SPAN_RESULT = {
    "block": "2024-03-12 to 2024-04-20, chosen by nothing",
    "instruments": 26,
    "positive_instruments": 0,
    "trades": 3637,
    "median_net_bp_per_trade": -19.94,
    "median_gross_bp_per_trade": -5.31,
    "total_net_bp": -69_283,
    "best": {"symbol": "DOGEUSDT", "net_bp": -9.92},
    "verdict": "killed: both stated conditions fired",
}

#: What the configuration above produced when the original held-out block was
#: read once. Kept so the contrast is legible rather than remembered.
HELD_OUT_RESULT = {
    "block": "2024-02-26 to 2024-03-11, never searched",
    "instruments": 26,
    "positive_instruments": 20,
    "median_net_bp_per_trade": 6.86,
    "median_trades": 126,
    "best": {"symbol": "DOGEUSDT", "net_bp": 26.70, "two_se_bp": 28.44},
    "worst": {"symbol": "ETHUSDT", "net_bp": -6.59, "two_se_bp": 9.53},
    "search_block_median_net_bp": 17.22,
    "caveat": (
        "Instruments are not independent: one market-wide bet expressed many "
        "ways, so 20 of 26 is not 20 independent successes."
    ),
}


def index_level(log_prices: np.ndarray, exclude: int | None = None) -> np.ndarray:
    """Equal-weighted index of log prices, optionally dropping one column.

    The column being predicted must be excluded, or the index partly predicts it
    by construction. That is the one line in this strategy where a mistake would
    manufacture a result rather than merely lose money.
    """
    if log_prices.ndim != 2:
        raise ValueError(f"expected a panel, got an array with {log_prices.ndim} axes")
    if exclude is None:
        return log_prices.mean(axis=1)
    if not 0 <= exclude < log_prices.shape[1]:
        raise ValueError(f"column {exclude} outside a panel of {log_prices.shape[1]}")
    keep = [j for j in range(log_prices.shape[1]) if j != exclude]
    if not keep:
        raise ValueError("excluding the only column leaves no index")
    return log_prices[:, keep].mean(axis=1)


def signal(level: np.ndarray, config: ReversionConfig) -> np.ndarray:
    """The index's recent return, in basis points, aligned to the decision row.

    Positive means the market has risen, which under this strategy is a reason
    to be short. The sign flip happens in :func:`decide`, once, so that reading
    the signal and acting on it stay separable.
    """
    out = np.full(len(level), np.nan)
    if config.lookback < len(level):
        out[config.lookback :] = (level[config.lookback :] - level[: -config.lookback]) * 1e4
    return out


def decide(values: np.ndarray, threshold: float) -> np.ndarray:
    """Trade against the index move, once it is large enough to bother with.

    ``threshold`` must come from a period before the one being traded. Choosing
    it on the rows it acts on is the most direct way to manufacture an edge, and
    it is the mistake this project has caught itself making more than once.
    """
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError(f"threshold must be a non-negative number, got {threshold}")
    out = np.zeros(len(values), dtype=int)
    strong = np.isfinite(values) & (np.abs(values) >= threshold)
    out[strong] = -np.sign(values[strong]).astype(int)
    return out


def threshold_for_rate(values: np.ndarray, rows_per_day: float, config: ReversionConfig) -> float:
    """The signal size that yields the target trade rate on this stretch.

    Derived from a quantile of past signal magnitudes, so it is a statement
    about the period it is computed on and must be computed on a training one.
    """
    usable = values[np.isfinite(values)]
    if len(usable) < 1_000:
        raise ValueError(f"only {len(usable)} usable rows to set a threshold from")
    days = len(values) / rows_per_day
    wanted = max(1, int(config.trades_per_day * days))
    share = min(0.999, wanted / len(usable))
    return float(np.quantile(np.abs(usable), 1.0 - share))
