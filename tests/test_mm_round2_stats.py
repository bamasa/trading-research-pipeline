"""The second round's registered statistics on hand-made numbers.

The primary metric is net per 100 USDT of clip a day, pooled as the mean over
the instruments with a usable day; a paired difference is taken only where
both days are usable. K-queue fires on a positive verdict the joint pessimistic
bracket does not keep; K-nbhd on a neighbourhood whose median is not positive;
and the gate's choice follows the registered neighbourhood rule.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from experiments.market_making_round2_verdicts import (
    Verdict,
    first_statuses,
    later_statuses,
    not_tested,
    taken_forward,
)
from trading_research.market_making import round2_stats as stats
from trading_research.market_making.gate import GateCell, gate_cells


def rows(values: list[tuple[str, str, float]], excluded: tuple[str, ...] = ()) -> pd.DataFrame:
    """Simulator rows: (symbol, day, net), with ``excluded`` instrument-days
    flagged and one more not simulated at all."""
    frame = pd.DataFrame(values, columns=["symbol", "day", "net"])
    frame["status"] = "ok"
    frame["excluded"] = [
        f"{s} {d}" in excluded for s, d in zip(frame["symbol"], frame["day"], strict=True)
    ]
    missing = pd.DataFrame(
        [{"symbol": "AUSDT", "day": "d9", "net": np.nan, "status": "no book", "excluded": True}]
    )
    return pd.concat([frame, missing], ignore_index=True)


CLIPS = {"AUSDT": 20.0, "BUSDT": 400.0}


def test_the_metric_is_net_per_100_usdt_of_clip() -> None:
    frame = rows([("AUSDT", "d1", 2.0), ("BUSDT", "d1", 40.0), ("AUSDT", "d2", -1.0)])
    y = stats.per_clip(frame, CLIPS)
    assert y.set_index(["symbol", "day"])["y"].to_dict() == {
        ("AUSDT", "d1"): 10.0,
        ("BUSDT", "d1"): 10.0,
        ("AUSDT", "d2"): -5.0,
    }
    with pytest.raises(KeyError, match="no frozen clip"):
        stats.per_clip(frame, {"AUSDT": 20.0})


def test_pooling_weights_each_instrument_equally_over_its_usable_days() -> None:
    # In USDT, BUSDT's clip would carry the day; per 100 of clip both count once.
    frame = rows(
        [("AUSDT", "d1", 2.0), ("BUSDT", "d1", -20.0), ("AUSDT", "d2", 1.0), ("BUSDT", "d2", 8.0)],
        excluded=("AUSDT d2",),
    )
    daily = stats.pooled_daily(frame, CLIPS)
    assert daily.to_dict() == {"d1": pytest.approx((10.0 - 5.0) / 2), "d2": pytest.approx(2.0)}
    assert stats.usdt_daily(frame).to_dict() == {"d1": -18.0, "d2": 8.0}


def test_a_paired_difference_needs_both_days_usable() -> None:
    gated = rows(
        [("AUSDT", "d1", 3.0), ("BUSDT", "d1", 4.0), ("AUSDT", "d2", 1.0), ("BUSDT", "d2", 8.0)],
        excluded=("BUSDT d2",),
    )
    base = rows(
        [("AUSDT", "d1", 1.0), ("BUSDT", "d1", 8.0), ("AUSDT", "d2", 2.0), ("BUSDT", "d2", 0.0)],
        excluded=("AUSDT d1",),
    )
    paired = stats.paired_daily(gated, base, CLIPS)
    # d1: only BUSDT is usable in both, (4 - 8) * 100 / 400 = -1; d2: only AUSDT, -5.
    assert paired.to_dict() == {"d1": pytest.approx(-1.0), "d2": pytest.approx(-5.0)}


def test_the_minimum_detectable_effect() -> None:
    daily = pd.Series([1.0, 3.0, 2.0, np.nan, 6.0])
    out = stats.mde(daily, 13)
    sd = float(np.std([1.0, 3.0, 2.0, 6.0], ddof=1))
    assert out["days"] == 4 and out["sigma_d"] == pytest.approx(sd)
    assert out["mde"] == pytest.approx(2 * sd / math.sqrt(13))


@pytest.mark.parametrize(
    ("default", "joint", "fires"),
    [
        ([1.0, 0.5], [0.8, 0.1], False),  # survives the bracket
        ([1.0, 0.5], [0.8, -0.1], True),  # one claim lost under the bracket
        ([1.0, 0.5], [0.8, 0.0], True),  # zero is not positive
        ([1.0, -0.5], [-0.8, -0.1], False),  # not positive by default: nothing to kill
        ([-1.0], [-2.0], False),
    ],
)
def test_k_queue_fires_only_on_a_positive_verdict_the_bracket_loses(
    default: list[float], joint: list[float], fires: bool
) -> None:
    assert stats.k_queue(default, joint) is fires


def test_k_nbhd_takes_the_median_of_the_neighbourhood() -> None:
    assert stats.k_nbhd([0.5, -0.1, 0.2]) == (0.2, False)
    assert stats.k_nbhd([0.5, -0.1, -0.2]) == (-0.1, True)
    assert stats.k_nbhd([1.0, -1.0]) == (0.0, True)
    assert stats.k_nbhd([np.nan, 0.3]) == (0.3, False)
    median, fires = stats.k_nbhd([np.nan])
    assert math.isnan(median) and fires
    # Per instrument: each instrument's median, then their mean.
    pooled, fires = stats.k_nbhd_per_instrument({"AUSDT": [1.0, 2.0, 3.0], "BUSDT": [-4.0, -1.0]})
    assert pooled == pytest.approx((2.0 - 2.5) / 2) and fires
    pooled, fires = stats.k_nbhd_per_instrument({"AUSDT": [1.0, 2.0, 3.0], "BUSDT": [0.0, -1.0]})
    assert pooled == pytest.approx((2.0 - 0.5) / 2) and not fires


def test_the_gate_is_chosen_by_its_neighbourhood_among_cells_that_fill() -> None:
    cells = gate_cells()
    values = dict.fromkeys(cells, -1.0)
    fills = dict.fromkeys(cells, 10.0)
    peak = GateCell("S1", "trailing", 15, 2.0)  # a lone peak in a corner
    values[peak] = 5.0
    region = [GateCell("S1_touch", "both", w, m) for w in (60, 240) for m in (0.0, 1.0)]
    for cell in region:
        values[cell] = 1.0
    chosen, record = stats.choose_gate_cell(values, fills)
    assert chosen in region and record["median"] == 1.0
    assert record["peak"][:4] == ["S1", "trailing", 15, 2] and record["peak"][4] == 5.0
    # Ties of the median are broken by the cell's own value.
    values[GateCell("S1_touch", "both", 240, 1.0)] = 1.5
    chosen, _ = stats.choose_gate_cell(values, fills)
    assert chosen == GateCell("S1_touch", "both", 240, 1.0)
    # A cell under five passive fills a day takes no part, and is no neighbour.
    for cell in region:
        fills[cell] = 4.9
    chosen, record = stats.choose_gate_cell(values, fills)
    assert chosen not in region and record["cells_taking_part"] == 42 - len(region)
    with pytest.raises(ValueError, match="no gate cell"):
        stats.choose_gate_cell(values, dict.fromkeys(cells, 0.0))


# -- the status rules ---------------------------------------------------------------


def verdict(name: str, p: float, *, fired: str | None = None, positive: bool = True) -> Verdict:
    v = Verdict(name, "metric", "H")
    v.p = p
    v.claims_positive = positive
    v.check("K1", "mean <= 0", 1.0, 0.0, fired == "K1")
    v.check("K5", "too few fills", 500.0, 100.0, fired == "K5", "inconclusive")
    return v


def test_the_first_read_statuses_follow_holm_across_all_five() -> None:
    verdicts = [
        verdict("B1", 0.001),
        verdict("B2", 0.001, fired="K1"),
        verdict("B3a", 0.04),
        verdict("B3b", 0.001, fired="K5"),
        not_tested("B4", "P"),
    ]
    out = first_statuses(verdicts)
    assert out["B1"][0] == "candidate"  # 0.001 <= 0.05 / 5
    assert out["B2"] == ("killed", "K1")
    assert out["B3a"] == ("inconclusive", "Holm")  # Holm stops before it
    assert out["B3b"] == ("inconclusive", "K5")
    assert out["B4"][0].startswith("not tested")


def test_later_reads_take_and_move_the_registered_hypotheses() -> None:
    previous = pd.DataFrame(
        {
            "hypothesis": ["B1", "B2", "B3a", "B3b"],
            "status": ["candidate", "inconclusive", "inconclusive", "killed"],
            "claims_positive": [True, True, False, True],
        }
    )
    assert taken_forward(previous, ["B1", "B2", "B3a", "B3b"]) == ["B1", "B2"]
    out = later_statuses([verdict("B1", 0.001), verdict("B2", 0.001)], previous, "basket_P")
    assert out == {"B1": ("confirmed", ""), "B2": ("candidate", "first significant on P")}
    out = later_statuses([verdict("B1", 0.2), verdict("B2", 0.2)], previous, "basket_P")
    assert out["B1"][0] == "candidate, not confirmed" and out["B2"][0] == "inconclusive"
    out = later_statuses([verdict("B1", 0.001, fired="K1")], previous, "basket_P")
    assert out["B1"][0] == "killed on P"
    # A fee-tier survivor whose base-fee twin did not survive is conditional.
    previous.loc[2, "status"] = "candidate"
    out = later_statuses([verdict("B1", 0.3), verdict("B3a", 0.001)], previous, "basket_P")
    assert out["B3a"][0] == "conditional on the fee tier"
    out = later_statuses([verdict("B1", 0.001)], previous, "Q")
    assert out["B1"] == ("confirmed", "")
