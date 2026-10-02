"""Known-answer markets: the simulator finds the edge when there is one and
nothing when there is none.

The counterpart of the repository's injected-signal tests. With uninformed
flow the mid never moves, so a touch quoter's whole gross is the half-spread on
every unit filled. With fully informed flow each print moves the mid to the
price just traded, so the half-spread is given back at once and spread plus
adverse selection is exactly zero. Moving the mid further than that makes
every fill lose, and its markout carries the sign of the move.
"""

from __future__ import annotations

import numpy as np
import pytest

from trading_research.market_making.analysis import market_wide_markouts, summarise_markouts
from trading_research.market_making.quoters import TouchQuoter
from trading_research.market_making.simulator import SimConfig, simulate_day
from trading_research.market_making.synthetic import known_answer_market

#: A clip of one: a tenth of the ten-unit touch. No slippage on the flatten, so
#: the flatten costs exactly the half-spread.
CONFIG = SimConfig(clip_notional=1e12, warmup_s=0.0, flatten_slippage_bp=0.0)
HALF_SPREAD = 2 * 0.01  # four ticks of 0.01, halved


def _maker(result):
    return result.fills[result.fills["maker"]]


def _captured(fills) -> np.ndarray:
    side = fills["side"].to_numpy(dtype=float)
    return side * (fills["mid_ref"].to_numpy() - fills["price"].to_numpy())


def test_uninformed_flow_earns_the_half_spread() -> None:
    result = simulate_day(known_answer_market(move_ticks=0), TouchQuoter(), CONFIG)
    maker = _maker(result)
    assert len(maker) > 200
    np.testing.assert_allclose(_captured(maker), HALF_SPREAD, rtol=1e-9)
    parts = result.decomposition
    volume = maker["size"].sum()
    flattened = result.fills.loc[~result.fills["maker"], "size"].sum()
    assert parts["adverse"] == pytest.approx(0.0, abs=1e-9)
    assert parts["inventory"] == pytest.approx(0.0, abs=1e-9)
    assert parts["gross"] == pytest.approx(HALF_SPREAD * (volume - flattened), rel=1e-9)


def test_informed_flow_earns_nothing_before_fees() -> None:
    result = simulate_day(known_answer_market(move_ticks=2), TouchQuoter(), CONFIG)
    maker = _maker(result)
    assert len(maker) > 200
    later = maker["markout_5s"].to_numpy()
    np.testing.assert_allclose(later, 0.0, atol=1e-9)
    # Spread captured, then given back in full within five seconds.
    np.testing.assert_allclose(_captured(maker), HALF_SPREAD, rtol=1e-9)


def test_markouts_have_the_sign_of_the_move() -> None:
    """Uninformed fills are worth the half-spread afterwards; fills against flow
    that moves the mid a full spread lose the half-spread."""
    good = _maker(simulate_day(known_answer_market(move_ticks=0), TouchQuoter(), CONFIG))
    bad = _maker(simulate_day(known_answer_market(move_ticks=4), TouchQuoter(), CONFIG))
    for horizon in ("markout_1s", "markout_5s"):
        assert (good[horizon] > 0).all()
        assert (bad[horizon] < 0).all()
    price = bad["price"].to_numpy()
    np.testing.assert_allclose(bad["markout_5s"], -HALF_SPREAD / price * 1e4, rtol=1e-9)
    by_path = summarise_markouts(bad, (1.0, 5.0))
    assert by_path["markout_5s"].iloc[0] < 0


def test_the_market_wide_benchmark_scores_the_resting_side() -> None:
    calm = market_wide_markouts(known_answer_market(move_ticks=0), (1.0, 5.0))
    toxic = market_wide_markouts(known_answer_market(move_ticks=4), (1.0, 5.0))
    assert (calm["markout_bp"] > 0).all()
    assert (toxic["markout_bp"] < 0).all()
    assert (calm["prints"] == 300).all()
