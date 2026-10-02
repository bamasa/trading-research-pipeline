"""Cross-sectional reversion at ten minutes: a conditional result, frozen.

An equal-weighted index of USDT perpetuals overshoots over roughly ten minutes
and comes back, so the index's recent return, excluding the instrument being
traded, predicts that instrument's next move with a negative sign. The rule
trades against large index moves and holds ten minutes. Its status in
``docs/findings.md`` is **conditional result**, reported in §27 of
``docs/results.md``: the effect and the market state it needs are both
measured, and the state is not forecastable by what was tried. The parameters
here are frozen; nothing re-tunes them.

Where it holds
--------------
On the held-out fortnight of 26 February to 11 March 2024, read once with the
configuration below, 22 of 26 instruments were positive at a median of +6.99 bp
per trade net of 12-16 bp of taker cost (by day, +6.18 bp over 14 days,
t = 1.18); :data:`HELD_OUT_RESULT` records slightly different figures for the
same block (20 of 26, +6.86 bp), and §27's are the ones quoted. The checks
behind it: a negative information coefficient on 26 of 26
instruments on both halves of the data; an effect that strengthens out to a
thirty-second gap between the windows and then decays, which an accounting
artefact would not do; a market-wide rather than cross-sectional effect (the 26
instruments carry one bet); and the instrument ordering the identity "edge is
IC times dispersion" predicts.

Where it does not, and why that is the condition
------------------------------------------------
On 12 March to 20 April 2024 the same frozen configuration lost on every
instrument: median -19.94 bp per trade, 0 of 26 positive, 3,637 trades, and the
holding-period profile inverted. The two spans differ in the quantity the rule
bets on: the index's ten-minute autocorrelation was -0.1014 where it pays and
-0.0003 where it does not. :func:`trailing_autocorrelation` measures it causally;
a gate on its trailing value (§27) admitted worse days than it rejected, so the
state is measurable but not forecastable one day ahead.

:data:`FRESH_SPAN_RESULT` and :data:`HELD_OUT_RESULT` keep both numbers beside
the configuration. :func:`reversion_tape` is the signal as an event-time
consumer (the market-making study) must read it: labelled by the end of each
grid bin, so that an as-of join on the label never reads a value before it was
observed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

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


#: Rows of the five-second grid in one day.
ROWS_PER_DAY = 86_400 // GRID_SECONDS


def trailing_autocorrelation(
    level: np.ndarray, lag: int, window_rows: int, *, known_at_open: bool = False
) -> np.ndarray:
    """Index autocorrelation over a trailing window, per row, strictly causal.

    The correlation of the index's ``lag``-row return with its next
    ``lag``-row return, over the ``window_rows`` rows that end before each day
    starts, computed once per day and held for that day: a value updating
    within the day would let the afternoon's behaviour decide the morning's
    trades. Days are blocks of :data:`ROWS_PER_DAY` rows from the start of
    ``level``; rows before the first full window are NaN, and so is a day whose
    window holds fewer than 500 usable pairs or no variation.

    By default the window's forward returns end up to ``lag`` rows after the
    window, so the value for a day reads the first ``lag`` rows of that day:
    the regime-gate experiment of §27 used exactly this definition, and it is
    kept rather than silently changed. ``known_at_open=True`` moves the window
    ``lag`` rows earlier, so every pair it uses ended before the day's first
    row and the value is known when the day opens.
    """
    n = len(level)
    past = np.full(n, np.nan)
    past[lag:] = (level[lag:] - level[:-lag]) * 1e4
    forward = np.full(n, np.nan)
    forward[:-lag] = (level[lag:] - level[:-lag]) * 1e4

    shift = lag if known_at_open else 0
    out = np.full(n, np.nan)
    for start in range(window_rows, n, ROWS_PER_DAY):
        # With the shift, the first window is clipped at the start of the data
        # rather than dropped.
        window = slice(max(0, start - window_rows - shift), start - shift)
        a, b = past[window], forward[window]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 500 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
            continue
        out[start : start + ROWS_PER_DAY] = float(np.corrcoef(a[ok], b[ok])[0, 1])
    return out


def reversion_tape(panel: pd.DataFrame, symbol: str, config: ReversionConfig) -> pd.Series:
    """The index's ten-minute return excluding ``symbol``, in bp, causally labelled.

    ``panel`` holds one column of log mids per instrument on the five-second
    grid, **labelled by the end of each bin** (``to_grid(..., label="right")``)
    and forward-filled, so a row's label is after every observation it carries.
    The value at a label is the return of the equal-weighted index of every
    other column over the :attr:`ReversionConfig.lookback` rows ending there:
    the quantity :func:`signal` computes, on a timeline where an as-of join on
    the label is causal. NaN until a full lookback has passed.

    Nothing after a label enters its value: the index at a row reads that row
    and earlier rows only, and the forward fill carries values forward, never
    back. Truncating the panel at any label leaves every earlier value of the
    tape unchanged.
    """
    if symbol not in panel.columns:
        raise KeyError(f"{symbol} is not in the panel")
    if not isinstance(panel.index, pd.DatetimeIndex) or not panel.index.is_monotonic_increasing:
        raise ValueError("the panel needs an increasing DatetimeIndex of bin-end labels")
    values = panel.to_numpy(dtype=np.float64)
    column = list(panel.columns).index(symbol)
    level = index_level(values, exclude=column)
    return pd.Series(signal(level, config), index=panel.index, name="index_return_bp")
