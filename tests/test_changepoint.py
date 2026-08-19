"""Does the detector find a break that is there, and stay quiet when none is?"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from trading_research.validation.changepoint import (
    ChangepointError,
    Segments,
    detect,
    segment,
)


def _walk(steps: int, scale: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(0.0, scale, steps))


def _prices(returns: np.ndarray, start: float = 100.0) -> np.ndarray:
    return start * np.exp(returns)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_quiet_series_produces_no_breaks(seed: int) -> None:
    """A series whose behaviour never changes should not be cut.

    The defaults were calibrated to fire on about one null series in forty, so a
    handful of seeds passing is the expected behaviour rather than a guarantee.
    """
    mid = _prices(_walk(200_000, 1e-4, seed=seed))
    spread = np.full(len(mid), 1.0)
    assert detect(mid, spread, statistics=("volatility", "spread")) == []


def test_volatility_break_is_found_after_it_happens() -> None:
    """A tenfold change in volatility is the clearest break there is."""
    calm = _walk(150_000, 1e-5, seed=1)
    wild = calm[-1] + _walk(150_000, 1e-4, seed=2)
    mid = _prices(np.concatenate([calm, wild]))
    spread = np.full(len(mid), 1.0)

    breaks = detect(mid, spread, statistics=("volatility",))
    assert breaks, "a tenfold volatility change should be detected"
    first = breaks[0]
    assert first.direction == +1
    # Never before the change, because every input window ends at the point
    # being scored. Detection lag is expected and is the cost of causality.
    assert first.index >= 150_000


def test_spread_break_is_attributed_to_the_spread() -> None:
    """The reported statistic should name what actually moved."""
    mid = _prices(_walk(300_000, 1e-5, seed=3))
    spread = np.concatenate([np.full(150_000, 1.0), np.full(150_000, 6.0)])

    breaks = detect(mid, spread, statistics=("volatility", "spread"))
    assert breaks
    assert breaks[0].statistic == "spread"


def test_standardisation_never_uses_the_current_point() -> None:
    """Truncating the series must not change the breaks found before the cut.

    This is the property that makes the detector usable forward. If a reference
    window included the present, results would shift as data arrived.
    """
    mid = _prices(np.concatenate([_walk(150_000, 1e-5, seed=4), 1 + _walk(150_000, 2e-4, seed=5)]))
    spread = np.full(len(mid), 1.0)

    full = detect(mid, spread, statistics=("volatility",))
    cut = 220_000
    truncated = detect(mid[:cut], spread[:cut], statistics=("volatility",))
    assert [b.index for b in full if b.index <= cut] == [b.index for b in truncated]


def test_minimum_gap_is_respected() -> None:
    """Two breaks closer than the gap should collapse to one."""
    mid = _prices(np.concatenate([_walk(100_000, 1e-5, seed=6), 1 + _walk(200_000, 3e-4, seed=7)]))
    spread = np.full(len(mid), 1.0)

    breaks = detect(mid, spread, statistics=("volatility",), minimum_gap=10_000)
    indices = [b.index for b in breaks]
    assert all(b - a >= 10_000 for a, b in pairwise(indices))


def test_segments_cover_the_series_exactly_once() -> None:
    mid = _prices(np.concatenate([_walk(150_000, 1e-5, seed=8), 1 + _walk(150_000, 3e-4, seed=9)]))
    spread = np.full(len(mid), 1.0)

    cut = segment(mid, spread, minimum_rows=10_000, statistics=("volatility",))
    assert cut.bounds[0][0] == 0
    assert cut.bounds[-1][1] == len(mid)
    for (_, end), (start, _) in pairwise(cut.bounds):
        assert end == start
    assert sum(cut.lengths()) == len(mid)


def test_short_segments_are_merged_and_their_breaks_dropped() -> None:
    """Bounds and breaks must stay consistent after merging."""
    mid = _prices(
        np.concatenate([_walk(150_000, 1e-5, seed=10), 1 + _walk(150_000, 3e-4, seed=11)])
    )
    spread = np.full(len(mid), 1.0)

    cut = segment(mid, spread, minimum_rows=10_000_000, statistics=("volatility",))
    # Nothing can survive a minimum longer than the series, so it stays whole.
    assert len(cut) == 1
    assert cut.breaks == []
    assert cut.bounds == [(0, len(mid))]


def test_segment_lookup_reports_the_containing_segment() -> None:
    cut = Segments(bounds=[(0, 100), (100, 250)])
    assert cut.of(0) == 0
    assert cut.of(99) == 0
    assert cut.of(100) == 1
    with pytest.raises(ChangepointError):
        cut.of(250)


def test_series_too_short_is_refused_rather_than_guessed() -> None:
    mid = _prices(_walk(5_000, 1e-4, seed=12))
    with pytest.raises(ChangepointError, match="at least"):
        detect(mid, np.ones(len(mid)))


def test_unknown_statistic_is_named_in_the_error() -> None:
    mid = _prices(_walk(20_000, 1e-4, seed=13))
    with pytest.raises(ChangepointError, match="momentum"):
        detect(mid, np.ones(len(mid)), window=200, reference=10, statistics=("momentum",))


def test_mismatched_inputs_are_refused() -> None:
    with pytest.raises(ChangepointError, match="differ in length"):
        detect(np.ones(100), np.ones(50))
