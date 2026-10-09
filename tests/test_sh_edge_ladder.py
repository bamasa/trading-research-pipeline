"""S25's edge ladder: every step present at every stage, each labelled."""

from __future__ import annotations

import numpy as np
import pytest

from trading_research.backtest.costs import tiers_for
from trading_research.reporting.edge_ladder import (
    CHANGED,
    INFORMATION_SET,
    LADDER_COLUMNS,
    NOT_READ,
    edge_ladder,
)

TIERS = len(tiers_for("g1"))


def test_before_any_read_every_row_is_there_and_marked() -> None:
    ladder = edge_ladder()
    assert tuple(ladder.columns) == LADDER_COLUMNS
    assert len(ladder) == 4 + 2 + 3 + TIERS + 1 + 3
    assert ladder["value"].isna().all()
    assert ladder["label"].str.contains(NOT_READ).all()
    assert list(ladder["step"].unique()) == [1, 2, 3, 4, 5]
    for step, text in CHANGED.items():
        assert set(ladder.loc[ladder["step"] == step, "changed"]) == {text}


def test_the_values_land_on_their_steps() -> None:
    ladder = edge_ladder(
        a0_zero_fee={"crossing": 4.0, "touch": -1.0, "trade_through": 0.5, "queue": -2.0},
        s25_d=(3.0, -8.5),
        s25_h_window=-7.0,
        s25_h_all=-6.0,
        s25_f=-9.0,
        s25_break_even=2.0,
        s25z_net_zero_fee=-1.5,
        fill_ratios={"touch": 0.8, "trade_through": 0.2, "queue": 0.4},
    )
    by = ladder.set_index(["step", "row", "block"])["value"]
    assert by[(1, "A0, crossing", "O")] == 4.0
    assert by[(1, "A0, posted, trade-through", "O")] == 0.5
    assert by[(2, "S25, net", "D")] == -8.5
    assert by[(3, "S25, net, status window", "H before 2025-08-01")] == -7.0
    assert by[(5, "fill ratio, queue", "O")] == 0.4
    assert not ladder["label"].str.contains(NOT_READ).any()
    held = ladder.loc[(ladder["step"] == 3) & (ladder["block"].isin(["H", "F"]))]
    assert set(held["label"]) == {INFORMATION_SET}
    assert "on development data" in ladder.loc[ladder["step"] == 2, "label"].iloc[0]
    assert "zero-fee venue" in ladder.loc[ladder["row"] == "S25-Z, net at 0/0", "label"].iloc[0]


def test_the_break_even_is_set_against_every_published_tier() -> None:
    ladder = edge_ladder(s25_break_even=2.0)
    fees = ladder.loc[ladder["step"] == 4].set_index("row")
    assert len(fees) == TIERS and set(fees["unit"]) == {"bp per side"}
    assert fees.loc["S25, break-even per side, base", "reference"] == 5.5
    assert fees.loc["S25, break-even per side, base", "pays"] is False
    assert fees.loc["S25, break-even per side, pro6_g1", "pays"] is True
    assert np.isnan(ladder.loc[ladder["step"] == 1, "value"]).all()


def test_an_unknown_rule_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown"):
        edge_ladder(a0_zero_fee={"maker": 1.0})
    with pytest.raises(ValueError, match="unknown"):
        edge_ladder(fill_ratios={"crossing": 1.0})
