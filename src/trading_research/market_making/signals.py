"""External signals, joined into event time at the end of their bin.

A signal computed on the five-second grid — the reversion index, a regime
statistic — is a value per bin. :func:`trading_research.data.grid.to_grid`
labels a bin by its start by default and carries its last value, so joining it
into event time at that label hands a decision a value observed up to five
seconds later. A :class:`SignalTape` therefore always holds **bin-end** labels,
and a decision at time ``t`` reads the last value whose label is strictly
before ``t``.

The reversion signal
--------------------
The study's H2 reads §27's frozen signal, the ten-minute return of an
equal-weighted index of the universe excluding the instrument. Here it is built
for event time:

* :func:`load_universe_panel` reads the universe's top of book one instrument at
  a time, only on the days asked for and only if the
  :class:`~trading_research.market_making.prereg.Access` given permits every one
  of them, and puts it on a right-labelled five-second grid;
* :func:`reversion_tape` turns the panel into the tape S4 and X1 read;
* :func:`fit_theta` and :func:`fit_beta` are the two values fitted on D;
* :func:`lean_trigger_tape` marks where S4's window opens, and
  :func:`reversion_triggers` is the frozen rule's entries — thinned with §27's
  hold and cooldown — which X1 executes passively and :func:`taker_twin`
  crosses for, at the same times.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from trading_research.market_making.events import NS_PER_S, to_ns, within_day


@dataclass(frozen=True)
class SignalTape:
    """A causal signal: ``values[i]`` is known from ``ts[i]`` (ns) onward."""

    name: str
    ts: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        if len(self.ts) != len(self.values):
            raise ValueError("a tape needs one value per timestamp")
        if len(self.ts) > 1 and np.any(np.diff(self.ts) <= 0):
            raise ValueError("tape timestamps must be strictly increasing")

    @classmethod
    def from_grid(
        cls,
        name: str,
        grid: pd.DataFrame,
        column: str,
        *,
        seconds: int,
        label: Literal["left", "right"],
    ) -> SignalTape:
        """A tape from a gridded frame, relabelled to bin ends if need be.

        ``label`` must say how ``grid`` was labelled; there is no default,
        because guessing is how the look-ahead gets in. A left-labelled grid
        has every label moved one bin later.
        """
        if label not in ("left", "right"):
            raise ValueError(f"label must be 'left' or 'right', got {label!r}")
        stamps = to_ns(grid["timestamp"])
        if label == "left":
            stamps = stamps + seconds * NS_PER_S
        values = grid[column].to_numpy(dtype=np.float64)
        return cls(name, stamps, values)

    def strictly_before(self, at_ns: np.ndarray) -> np.ndarray:
        """The value known strictly before each time in ``at_ns``; NaN before the first."""
        at = np.asarray(at_ns, dtype=np.int64)
        index = np.searchsorted(self.ts, at, side="left") - 1
        out = np.full(len(at), np.nan)
        known = index >= 0
        out[known] = self.values[index[known]]
        return out

    def __len__(self) -> int:
        return len(self.ts)

    def between(self, start_ns: int, end_ns: int) -> SignalTape:
        """The part a window ``[start_ns, end_ns]`` can read.

        Keeps the last value before ``start_ns`` as well, since a decision at
        the window's start reads it. Used to send each simulated day only its
        own slice of a long tape.
        """
        first = max(int(np.searchsorted(self.ts, start_ns, side="left")) - 1, 0)
        last = int(np.searchsorted(self.ts, end_ns, side="right"))
        return SignalTape(self.name, self.ts[first:last].copy(), self.values[first:last].copy())

    def fingerprint(self) -> str:
        """A short hash of the contents, for cache keys."""
        import hashlib

        digest = hashlib.sha256(self.name.encode())
        digest.update(np.ascontiguousarray(self.ts).tobytes())
        digest.update(np.ascontiguousarray(self.values).tobytes())
        return digest.hexdigest()[:16]


# ---------------------------------------------------------------------------
# The reversion signal
# ---------------------------------------------------------------------------

#: Where the universe's top of book lives: one file per instrument-day.
UNIVERSE_ROOT = Path("data/universe")

#: Seconds per row of the reversion grid, as in §27.
GRID_SECONDS = 5


@dataclass(frozen=True)
class UniversePanel:
    """Log mids and spreads (bp) of the universe on a right-labelled grid.

    Columns are instruments, rows are bin-end labels (UTC), forward-filled
    across the union of every instrument's rows, with the rows before every
    instrument has a value dropped, as §27's panel was built.
    """

    log_mid: pd.DataFrame
    spread_bp: pd.DataFrame


def load_universe_panel(
    symbols: Sequence[str],
    days: Sequence[date],
    *,
    access: object,
    root: Path = UNIVERSE_ROOT,
    seconds: int = GRID_SECONDS,
) -> UniversePanel:
    """Read the universe on ``days`` only, after checking ``access`` permits them.

    The days are checked before any file is opened, and only files named for
    those days are opened, so a day outside the permitted blocks is never read;
    rows of a day's file stamped outside that day are dropped. An instrument
    with no file on any of the days is left out.
    """
    from trading_research.data.grid import to_grid
    from trading_research.market_making.prereg import Access

    if not isinstance(access, Access):
        raise TypeError("reading the universe needs a prereg.Access")
    days = sorted(set(days))
    access.require(days, what="read the universe for")
    columns = ["timestamp", "bid_price_0", "ask_price_0"]
    mids: dict[str, pd.Series] = {}
    spreads: dict[str, pd.Series] = {}
    for symbol in symbols:
        frames = []
        for day in days:
            path = Path(root) / symbol / f"{day.isoformat()}.parquet"
            if path.is_file() and path.stat().st_size > 0:
                frames.append(within_day(pd.read_parquet(path, columns=columns), day))
        if not frames:
            continue
        grid = to_grid(pd.concat(frames, ignore_index=True), seconds, label="right")
        index = pd.DatetimeIndex(grid["timestamp"])
        keep = ~index.duplicated()
        grid, index = grid[keep], index[keep]
        bid = grid["bid_price_0"].to_numpy(dtype=np.float64)
        ask = grid["ask_price_0"].to_numpy(dtype=np.float64)
        mid = (bid + ask) / 2.0
        mids[symbol] = pd.Series(np.log(mid), index=index)
        spreads[symbol] = pd.Series((ask - bid) / mid * 1e4, index=index)
        del frames, grid
    if not mids:
        raise FileNotFoundError(f"no universe files under {root} for the days asked")
    log_mid = pd.DataFrame(mids).sort_index().ffill().dropna()
    spread = pd.DataFrame(spreads).sort_index().ffill().reindex(log_mid.index)
    return UniversePanel(log_mid=log_mid, spread_bp=spread)


def reversion_tape(panel: UniversePanel, symbol: str, config: object) -> SignalTape:
    """§27's signal for ``symbol`` as a :class:`SignalTape` named ``index_return_bp``.

    The index's ten-minute return excluding ``symbol``, in bp, at each bin-end
    label: known from its label onward, and read strictly after it.
    """
    from trading_research.strategies import reversion

    if not isinstance(config, reversion.ReversionConfig):
        raise TypeError("config must be the frozen ReversionConfig")
    series = reversion.reversion_tape(panel.log_mid, symbol, config)
    known = np.isfinite(series.to_numpy())
    return SignalTape(
        "index_return_bp",
        to_ns(pd.DatetimeIndex(series.index[known])),
        series.to_numpy(dtype=np.float64)[known],
    )


def forward_return_bp(log_mid: pd.Series, rows: int) -> np.ndarray:
    """The instrument's return over the next ``rows`` rows, bp; NaN where unknown."""
    values = log_mid.to_numpy(dtype=np.float64)
    out = np.full(len(values), np.nan)
    if rows < len(values):
        out[:-rows] = (np.exp(values[rows:] - values[:-rows]) - 1.0) * 1e4
    return out


def fit_theta(tape: SignalTape, config: object) -> float:
    """S4's and X1's threshold: the frozen trade rate's quantile of |signal|.

    ``threshold_for_rate`` at :attr:`ReversionConfig.trades_per_day` over the
    tape's rows, with the five-second grid's rows per day. Fitted on D only.
    """
    from trading_research.strategies import reversion

    if not isinstance(config, reversion.ReversionConfig):
        raise TypeError("config must be the frozen ReversionConfig")
    return reversion.threshold_for_rate(tape.values, reversion.ROWS_PER_DAY, config)


def fit_beta(panel: UniversePanel, symbol: str, tape: SignalTape, theta: float) -> float:
    """S4's slope: OLS, no intercept, of the next ten-minute return on the signal.

    Over the rows with ``|signal| >= theta`` whose next ten minutes lie inside
    the panel: ``sum(s * y) / sum(s * s)``.
    """
    from trading_research.strategies.reversion import ReversionConfig

    hold = ReversionConfig().hold
    stamps = to_ns(pd.DatetimeIndex(panel.log_mid.index))
    position = np.searchsorted(stamps, tape.ts)
    if np.any(position >= len(stamps)) or np.any(stamps[position] != tape.ts):
        raise ValueError("the tape is not on the panel's grid")
    forward = forward_return_bp(panel.log_mid[symbol], hold)[position]
    strong = (np.abs(tape.values) >= theta) & np.isfinite(forward)
    if strong.sum() < 30:
        raise ValueError(f"only {int(strong.sum())} rows clear the threshold; no slope")
    s = tape.values[strong]  # noqa: PD011 - a SignalTape, not pandas
    return float(np.dot(s, forward[strong]) / np.dot(s, s))


def lean_trigger_tape(tape: SignalTape, theta: float) -> SignalTape:
    """Where S4's window opens: ``lean_trigger_s``, the time (epoch seconds) of
    the latest label at which ``|signal| >= theta``.

    A decision reads it strictly before its own time, as every tape, so the
    window it opens is known from the trigger's label onward.
    """
    strong = np.abs(tape.values) >= theta
    stamps = tape.ts[strong]
    return SignalTape("lean_trigger_s", stamps, stamps.astype(np.float64) / NS_PER_S)


#: X1 opens an attempt only between the end of the simulator's ten-minute
#: warm-up and 25 minutes before midnight, seconds after midnight UTC.
X1_FIRST_ENTRY_S = 600
X1_LAST_ENTRY_S = 86_400 - 25 * 60


@dataclass(frozen=True)
class Triggers:
    """The frozen rule's entries: when each is known, which way, and its signal.

    ``direction`` is the side the rule takes, against the index move: +1 buy
    after a fall, -1 sell after a rise.
    """

    ts: np.ndarray
    direction: np.ndarray
    signal_bp: np.ndarray

    def __len__(self) -> int:
        return len(self.ts)

    def tapes(self) -> dict[str, SignalTape]:
        """``x1_trigger_s`` and ``x1_direction``, for X1 to read."""
        return {
            "x1_trigger_s": SignalTape("x1_trigger_s", self.ts, self.ts / NS_PER_S),
            "x1_direction": SignalTape("x1_direction", self.ts, self.direction.astype(np.float64)),
        }

    def eligible(self, first_s: int = X1_FIRST_ENTRY_S, last_s: int = X1_LAST_ENTRY_S) -> Triggers:
        """The triggers X1 may act on: known at or after ``first_s`` and before
        ``last_s`` seconds after midnight. The taker twin is scored on the same
        set, so the two are paired trigger for trigger."""
        seconds = (self.ts // NS_PER_S) % 86_400
        keep = (seconds >= first_s) & (seconds < last_s)
        return Triggers(self.ts[keep], self.direction[keep], self.signal_bp[keep])

    def flipped(self) -> Triggers:
        """The same times, the other side: the placebo that follows the move."""
        return Triggers(self.ts, -self.direction, self.signal_bp)


def reversion_triggers(tape: SignalTape, theta: float, config: object) -> Triggers:
    """§27's entries at ``theta``, thinned with its hold and cooldown.

    ``decide`` turns each label into buy, sell or nothing; ``thin`` then keeps
    one position at a time and stands aside for the cooldown after each, on
    the five-second rows of the tape. An entry at a row depends on that row and
    earlier ones only, so the triggers are causal and do not depend on prices:
    they are the same for the passive execution and for the taker twin.
    """
    from trading_research.backtest.execution import ThinningRules, thin
    from trading_research.strategies import reversion

    if not isinstance(config, reversion.ReversionConfig):
        raise TypeError("config must be the frozen ReversionConfig")
    decision = reversion.decide(tape.values, theta)
    known = np.zeros(len(decision))
    taken = thin(
        decision,
        known,
        known,
        ThinningRules(hold_periods=config.hold, cooldown_periods=config.cooldown),
    )
    rows = np.array([t.entry_index for t in taken], dtype=np.int64)
    return Triggers(
        ts=tape.ts[rows] if len(rows) else np.zeros(0, dtype=np.int64),
        direction=np.array([t.direction for t in taken], dtype=np.int64),
        signal_bp=tape.values[rows] if len(rows) else np.zeros(0),  # noqa: PD011
    )


def taker_twin(
    triggers: Triggers,
    panel: UniversePanel,
    symbol: str,
    *,
    fee_bp_per_side: float = 5.5,
    slippage_bp: float = 0.5,
    hold_rows: int = 120,
) -> pd.DataFrame:
    """§27's taker execution of the same triggers, one row per trade.

    Entry at the trigger's label crossing the spread of the instrument's top of
    book, exit ``hold_rows`` later, ``TakerCosts(5.5, 0.5)``: gross is the
    direction times the move over the hold, cost the round trip at the entry
    spread. A trigger whose hold runs past the panel is dropped, as §27 dropped
    rows without a forward return.
    """
    from trading_research.backtest.costs import TakerCosts

    costs = TakerCosts(fee_bp_per_side=fee_bp_per_side, slippage_bp=slippage_bp)
    stamps = to_ns(pd.DatetimeIndex(panel.log_mid.index))
    position = np.searchsorted(stamps, triggers.ts)
    on_grid = (position < len(stamps)) & (
        stamps[np.minimum(position, len(stamps) - 1)] == triggers.ts
    )
    if not on_grid.all():
        raise ValueError("a trigger is not on the panel's grid")
    forward = forward_return_bp(panel.log_mid[symbol], hold_rows)[position]
    spread = panel.spread_bp[symbol].to_numpy(dtype=np.float64)[position]
    gross = triggers.direction * forward
    cost = np.asarray(costs.round_trip_bp(pd.Series(spread)), dtype=np.float64)
    frame = pd.DataFrame(
        {
            "symbol": symbol,
            "ts": triggers.ts,
            "day": pd.to_datetime(triggers.ts, utc=True).date,
            "direction": triggers.direction,
            "signal_bp": triggers.signal_bp,
            "entry_spread_bp": spread,
            "gross_bp": gross,
            "cost_bp": cost,
            "net_bp": gross - cost,
        }
    )
    return frame[np.isfinite(frame["gross_bp"])].reset_index(drop=True)
