"""Finding where the series stopped behaving the way it did before.

Every result in this project assumes that what a model learns on one stretch of
data still applies to the next. §25 is what happens when that assumption fails
quietly: a strategy fitted and validated across February looked profitable at
every trade rate, and the same procedure over the following two months lost
money at every trade rate. Nothing about the method changed. The market did.

The usual response is to refit more often, and §10 measured that — it helps and
it is not enough, because a fixed cadence is a guess about how long conditions
last. A refit every two days is wasteful if the regime holds for three weeks and
far too slow if it breaks after four hours.

The alternative is to let the data say when it changed, which is what this
module does, and then to treat the answer as a boundary: a model is expected to
apply until the next break, and no claim is made about it beyond that.

The detector
------------
A CUSUM filter, in the form López de Prado uses for event sampling. Track the
running sum of standardised deviations from a reference level in each direction:

    S⁺ₜ = max(0, S⁺ₜ₋₁ + zₜ),    S⁻ₜ = min(0, S⁻ₜ₋₁ + zₜ)

and declare a break when either exceeds a threshold ``h``, resetting both. A
single large move does not trigger it; a persistent drift in the monitored
statistic does. That is the property wanted here, because a break in regime is
not one unusual observation but a change in what ordinary observations look
like.

Blocks, not a rolling window
----------------------------
The first version of this monitored a rolling statistic at every observation
and cut a pure random walk into pieces — a false break roughly every twenty
thousand rows, on a series that by construction never changed.

The reason is that CUSUM assumes its increments are close to independent. A
rolling standard deviation over two thousand observations, evaluated at every
observation, shares all but one of its inputs with the value before it, so
consecutive z-scores are nearly the same number. Summing two thousand copies of
a mild deviation clears any threshold, and what fires is the overlap rather than
the market.

So the statistics are computed over **non-overlapping** blocks: one value per
``window`` observations, each summarising a stretch of the series that no other
value has seen. That restores the independence the filter assumes, and it is
also what makes the autocorrelation statistic affordable.

Where the threshold comes from
------------------------------
A CUSUM threshold is a false-alarm rate in disguise, and picking one by eye is
how a detector ends up reporting the analyst's expectations. This one was
calibrated against a null: forty random walks of two hundred thousand
observations each, generated with constant volatility, so every break reported
on them is by construction false.

| threshold | fires on null | detects a 1.5x volatility change |
|---|---|---|
| 6 | 40/40 | — |
| 10 | 28/40 | — |
| 14 | 6/40 | 20/20 |
| 20 | 1/40 | 20/20 |

The defaults are the last row: a break on roughly one quiet series in forty,
against detection of the smallest change worth acting on in all twenty trials,
at a median lag of one block. The reference length matters as much as the
threshold — at forty blocks the estimate of *normal* is stable enough that the
filter is not chasing its own start-up bias, which at ten blocks it does.

Causality
---------
Each block summarises observations that ended before it is scored, and the
standardisation uses the mean and standard deviation of *previous* blocks only,
expanding as the series runs — no block contributes to its own reference.

The consequence worth stating: a break inside block *k* is detectable no earlier
than the end of block *k*, and usually later, because CUSUM accumulates evidence
before it fires. Detection lag is the price of not looking ahead, and any
strategy conditioned on these breaks inherits it. Comparing these breaks against
breaks placed by a full-sample method would flatter this one; they should be
compared against nothing, and used only forward.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: Statistics monitored by default. Each is a trailing summary of a different
#: aspect of the market: how much it moves, what it costs to cross, how busy it
#: is, and whether its moves persist or reverse.
DEFAULT_STATISTICS: tuple[str, ...] = ("volatility", "spread", "intensity", "autocorrelation")


class ChangepointError(ValueError):
    """The series is too short to monitor, or the requested statistic is unknown."""


@dataclass(frozen=True)
class Break:
    """One detected regime break."""

    index: int
    statistic: str
    direction: int
    """+1 when the statistic drifted up, -1 when down."""
    excursion: float
    """How far the accumulator had run when it fired, in units of the threshold."""


@dataclass
class Segments:
    """A timeline cut at its breaks.

    ``bounds`` are half-open ``[start, end)`` index ranges covering the series
    exactly once, so a caller can iterate them without worrying about gaps or
    overlaps.
    """

    bounds: list[tuple[int, int]]
    breaks: list[Break] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.bounds)

    def of(self, index: int) -> int:
        """Which segment an index falls in."""
        for number, (start, end) in enumerate(self.bounds):
            if start <= index < end:
                return number
        raise ChangepointError(f"index {index} outside the series")

    def lengths(self) -> list[int]:
        return [end - start for start, end in self.bounds]

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for number, (start, end) in enumerate(self.bounds):
            trigger = next((b for b in self.breaks if b.index == start), None)
            rows.append(
                {
                    "segment": number,
                    "start": start,
                    "end": end,
                    "rows": end - start,
                    "opened_by": trigger.statistic if trigger else "start of series",
                    "direction": trigger.direction if trigger else 0,
                }
            )
        return pd.DataFrame(rows)


def _blocks(
    name: str, mid: np.ndarray, spread_bp: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray]:
    """One value of the statistic per non-overlapping block, with block end indices."""
    returns = np.diff(np.log(mid), prepend=np.nan)
    count = len(mid) // window
    if count < 2:
        raise ChangepointError(f"only {count} block(s) of {window} in {len(mid)} observations")
    ends = np.arange(1, count + 1) * window

    values = np.empty(count, dtype="float64")
    for i in range(count):
        block = returns[i * window : ends[i]]
        if name == "volatility":
            values[i] = np.nanstd(block)
        elif name == "spread":
            values[i] = np.nanmean(spread_bp[i * window : ends[i]])
        elif name == "intensity":
            # Share of observations where the price moved at all: falls in quiet
            # markets, rises in busy ones, and does not depend on tick size.
            values[i] = np.nanmean((np.abs(block) > 0).astype("float64"))
        elif name == "autocorrelation":
            # Whether moves persist or reverse — the statistic separating a
            # trending stretch from a choppy one, which is the distinction §25
            # turned out to have been measuring by accident.
            finite = block[np.isfinite(block)]
            if len(finite) < 3 or np.std(finite) == 0.0:
                values[i] = np.nan
            else:
                values[i] = float(np.corrcoef(finite[:-1], finite[1:])[0, 1])
        else:
            raise ChangepointError(f"unknown statistic {name!r}; known: {DEFAULT_STATISTICS}")
    return values, ends


def _standardise(values: np.ndarray, reference: int) -> np.ndarray:
    """Z-score each block against the blocks before it.

    Both the expanding mean and the expanding standard deviation stop one block
    short, so no block contributes to its own reference.
    """
    series = pd.Series(values)
    past = series.shift(1).expanding(min_periods=reference)
    scaled = (series - past.mean()) / past.std().replace(0.0, np.nan)
    return scaled.to_numpy()


def detect(
    mid: np.ndarray | pd.Series,
    spread_bp: np.ndarray | pd.Series,
    *,
    window: int = 2000,
    threshold: float = 20.0,
    reference: int = 40,
    minimum_gap: int = 20_000,
    statistics: Sequence[str] = DEFAULT_STATISTICS,
) -> list[Break]:
    """Run the CUSUM filter over each statistic and return the breaks it fires.

    ``window`` is the block size, in observations. ``reference`` is how many
    blocks must pass before monitoring starts — the filter needs a notion of
    normal before it can report a departure from it.

    ``threshold`` is in accumulated standard deviations of the block statistic.
    Higher means fewer, later, more certain breaks. ``minimum_gap`` is in
    observations and suppresses a second break too soon after the last: without
    it an unstable stretch fires continuously and cuts the series into pieces
    too small to fit anything on.
    """
    mid = np.asarray(mid, dtype="float64")
    spread_bp = np.asarray(spread_bp, dtype="float64")
    if len(mid) != len(spread_bp):
        raise ChangepointError(f"mid and spread differ in length: {len(mid)} vs {len(spread_bp)}")
    if len(mid) < window * (reference + 2):
        raise ChangepointError(
            f"need at least {window * (reference + 2)} observations to monitor, got {len(mid)}"
        )

    found: list[Break] = []
    for name in statistics:
        values, ends = _blocks(name, mid, spread_bp, window)
        z = _standardise(values, reference)
        up = down = 0.0
        last = -minimum_gap
        for i, value in enumerate(z):
            if not np.isfinite(value):
                continue
            up = max(0.0, up + value)
            down = min(0.0, down + value)
            at = int(ends[i])
            if at - last < minimum_gap:
                continue
            if up > threshold:
                found.append(Break(at, name, +1, up / threshold))
                up = down = 0.0
                last = at
            elif down < -threshold:
                found.append(Break(at, name, -1, abs(down) / threshold))
                up = down = 0.0
                last = at
    return sorted(found, key=lambda b: b.index)


def segment(
    mid: np.ndarray | pd.Series,
    spread_bp: np.ndarray | pd.Series,
    *,
    minimum_rows: int = 20_000,
    **kwargs: object,
) -> Segments:
    """Cut the series at its breaks, merging segments too short to fit a model on.

    ``minimum_rows`` is a modelling constraint rather than a statistical one: a
    break is real whether or not enough data follows it, but a segment of two
    thousand rows cannot support a fit, and pretending otherwise produces
    models trained on noise. Short segments are absorbed into the preceding one
    and the break that opened them is dropped from the record, so the returned
    bounds and breaks stay consistent with each other.
    """
    breaks = detect(mid, spread_bp, **kwargs)  # type: ignore[arg-type]
    return cut_at_breaks(breaks, len(np.asarray(mid)), minimum_rows=minimum_rows)


def cut_at_breaks(breaks: Sequence[Break], total: int, *, minimum_rows: int) -> Segments:
    """Turn a list of breaks into ``Segments`` over a series of ``total`` rows.

    Shared by both detectors so that a refit-at-breaks policy cuts the series
    the same way whichever found the breaks. Breaks closer than ``minimum_rows``
    to the previous cut are dropped, and so is a final cut that would leave a
    tail too short to fit on.
    """
    kept: list[Break] = []
    cuts: list[int] = []
    previous = 0
    for b in sorted(breaks, key=lambda b: b.index):
        if b.index - previous < minimum_rows:
            continue
        cuts.append(b.index)
        kept.append(b)
        previous = b.index
    # A final segment shorter than the minimum is merged backwards for the same
    # reason, which means dropping the cut that created it.
    if cuts and total - cuts[-1] < minimum_rows:
        cuts.pop()
        kept.pop()

    edges = [0, *cuts, total]
    bounds = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    return Segments(bounds=bounds, breaks=kept)
