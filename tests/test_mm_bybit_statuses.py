"""The Bybit 2021 statuses on H: the kills in order, the precedence, Holm over {F, S0}."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from trading_research.mm_bybit import runner
from trading_research.mm_bybit import statuses as st

DAYS = 84


def primary(name: str = "S0", mean: float = 1.0, **changes: object) -> st.Primary:
    """A primary whose every series is positive and steady unless changed."""
    rng = np.random.default_rng(7)
    y = mean + rng.normal(0.0, 0.2, size=DAYS)
    hourly = np.repeat(y[:, None] / 24.0, 24, axis=1)
    base = st.Primary(
        name=name, cell="l0|none", y=y, making=y * 0.2, inventory=y * 0.1,
        funding=np.zeros(DAYS), ex_rebate=y * 0.5, touch=y * 2.0, delay=y * 0.9,
        latency150=y * 0.9, hourly=hourly, fills=10_000, capacity=np.full(DAYS, 0.01),
        neighbour_means=[mean, mean], mde=0.1,
    )  # fmt: skip
    return dataclasses.replace(base, **changes)  # type: ignore[arg-type]


def test_a_steady_positive_primary_meeting_the_bar_is_a_candidate() -> None:
    stats = st.assess([primary()])
    s = stats["S0"]
    assert s["status"] == "candidate" and s["first_kill"] is None
    assert s["holm_m"] == 2  # F counts in the family even when it is not run
    assert s["labels"] == {"paid_with_the_rebate": True, "paid_ex_rebate": True,
                           "paid_at_the_touch_only": False}  # fmt: skip


def test_the_kills_fire_in_the_registered_order() -> None:
    assert st.KILL_ORDER == ("K1", "K-dir", "K-delay", "K-latency", "K-fund", "K-drift",
                             "K-days", "K-nbhd", "K-placebo")  # fmt: skip
    p = primary(making=-np.ones(DAYS), delay=-np.ones(DAYS))
    s = st.assess([p])["S0"]
    assert s["status"] == "killed"
    assert s["first_kill"]["kill"] == "K-dir"
    assert [k["kill"] for k in s["kills"] if k["fires"]] == ["K-dir", "K-delay"]
    negative = st.assess([primary(mean=-1.0)])["S0"]
    assert negative["first_kill"]["kill"] == "K1"
    assert negative["labels"]["paid_at_the_touch_only"] is False


def test_conditional_kills_need_a_positive_result() -> None:
    p = primary(mean=-1.0, delay=-np.ones(DAYS), inventory=np.full(DAYS, 5.0))
    fired = {k["kill"] for k in st.kills(p) if k["fires"]}
    assert "K-delay" not in fired and "K-drift" not in fired and "K1" in fired
    drift = st.kills(primary(inventory=np.full(DAYS, 0.6)))
    assert next(k for k in drift if k["kill"] == "K-drift")["fires"]


def test_k_days_k_nbhd_and_k_placebo() -> None:
    half = np.where(np.arange(DAYS) % 2 == 0, 3.0, -1.0)
    days = st.kills(primary(y=half, hourly=np.repeat(half[:, None] / 24, 24, axis=1)))
    assert next(k for k in days if k["kill"] == "K-days")["fires"]
    nbhd = st.kills(primary(neighbour_means=[1.0, -2.0, -3.0]))
    assert next(k for k in nbhd if k["kill"] == "K-nbhd")["fires"]
    shifted = st.kills(primary(shifted_mean=5.0))
    placebo = next(k for k in shifted if k["kill"] == "K-placebo")
    assert placebo["fires"] and placebo["shifted_mean"] == 5.0


def test_the_precedence_void_then_too_few_fills_then_killed_then_capacity() -> None:
    assert st.assess([primary(void=["K-acct"])])["S0"]["status"] == "void (K-acct)"
    few = primary(fills=99, mean=-1.0)
    assert st.assess([few])["S0"]["status"] == "inconclusive (too few fills)"
    crowded = primary(capacity=np.full(DAYS, 0.2))
    assert st.assess([crowded])["S0"]["status"] == "inconclusive (capacity)"


def test_a_finalist_not_admitted_or_the_same_as_s0_carries_its_label_and_p_one() -> None:
    f = primary("F", holm_one="not admitted on D (iv)")
    stats = st.assess([f, primary()])
    assert stats["F"]["status"] == "not admitted on D (iv)"
    assert stats["F"]["status_as_a_primary"] == "inconclusive"
    assert stats["F"]["holm_p"] == 1.0
    assert stats["S0"]["status"] == "candidate"
    same = st.assess([primary("F", holm_one="same as S0"), primary()])
    assert same["F"]["status"] == "same as S0"


def test_holm_over_two_needs_the_smaller_p_under_alpha_over_two() -> None:
    z = np.random.default_rng(3).normal(size=DAYS)
    z = (z - z.mean()) / z.std(ddof=1)
    y = 1.8 / np.sqrt(DAYS) + z  # a day-level t of 1.8: p about 0.04
    s = st.assess([primary(y=y, hourly=np.repeat(y[:, None] / 24, 24, axis=1))])["S0"]
    assert s["t"] == pytest.approx(1.8)
    assert 0.025 < s["p_one_sided"] < 0.05
    assert s["holm_m"] == 2 and not s["holm_passes"]
    alone = st.assess([primary(y=y, hourly=np.repeat(y[:, None] / 24, 24, axis=1))],
                      family=("S0",))["S0"]  # fmt: skip
    assert alone["holm_passes"]


def test_the_headline_sentences_carry_two_decimals_and_units() -> None:
    p = primary(mean=-1.0)
    stats = st.assess([p])["S0"]
    measurements = {"net_bp_per_fill": -1.234, "ex_rebate_bp_per_fill": -3.734,
                    "making_bp_per_fill": -2.5}  # fmt: skip
    first, question, labels = runner.sentences("S0", stats, measurements)
    assert first.startswith("S0 (l0|none) on Bybit BTCUSDT, 8 April to 30 June 2021")
    assert "first kill K1 at -1.0" in first and "USDT per 100 USDT of clip a day" in first
    assert "lost 1.23 bp per fill" in question and "lost 3.73 bp per fill" in question
    assert "making part was -2.50 bp" in question
    assert labels == "S0: not paid with the rebate; not paid ex-rebate."


@pytest.mark.parametrize("value", [0.4, 0.5])
def test_k_days_fires_at_half_or_fewer_days_positive(value: float) -> None:
    n_pos = round(value * DAYS)
    y = np.where(np.arange(DAYS) < n_pos, 5.0, -0.1)
    kill = next(k for k in st.kills(primary(y=y)) if k["kill"] == "K-days")
    assert kill["fires"]
