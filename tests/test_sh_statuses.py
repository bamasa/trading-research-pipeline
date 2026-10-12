"""The status rules on a held-out block: precedence, Holm at m = 5, the routes through F and P, R."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from trading_research.pipeline.short_horizon import statuses as st

H_DAYS = [date(2025, 7, 8) + timedelta(days=i) for i in range(44)]


def slot_days(slot: str = "S-C", *, mean: float = 3.0, noise: float = 1.0, trades: int = 10,
              days: list[date] | None = None, seed: int = 0, **changes: object) -> st.SlotDays:  # fmt: skip
    """A taker slot whose every placebo, neighbour and re-run is given, all unremarkable."""
    days = days or H_DAYS
    n = len(days)
    rng = np.random.default_rng(seed)
    value = mean + noise * rng.standard_normal(n)
    found = st.SlotDays(
        slot, "taker", list(days), value, np.full(n, trades), gross=np.full(n, 10.0),
        turnover=np.full(n, 1_000.0), ic=np.full(n, 0.05), random=rng.normal(-1, 1, (20, n)),
        flipped=np.full(n, -2.0), shifted=np.full(n, -1.0), long_only=np.full(n, -1.0),
        short_only=np.full(n, -1.0), neighbours=np.full((8, n), 1.0), latency=value - 0.1,
    )  # fmt: skip
    for name, value_ in changes.items():
        setattr(found, name, value_)
    return found


def test_a_slot_that_pays_on_every_count_is_a_candidate() -> None:
    s = slot_days()
    found = st.family({"S-C": s}, {"S-C": s.mask(s.days)}, "H", {})
    assert found["S-C"].status == "candidate"
    assert {k: v.status for k, v in found.items() if k != "S-C"} == dict.fromkeys(
        ("S25", "S-R", "S-P", "S-M"), "not assessed"
    )


def test_void_comes_before_everything() -> None:
    s = slot_days(void=("K-acct",), trades=1)
    assert st.family({"S-C": s}, {"S-C": s.mask(s.days)}, "H", {})["S-C"].status == "void"


def test_too_few_trades_is_inconclusive_whatever_the_sign_and_no_kill_is_evaluated() -> None:
    s = slot_days(mean=-5.0, trades=2)
    found = st.family({"S-C": s}, {"S-C": s.mask(s.days)}, "H", {})["S-C"]
    assert found.status == "inconclusive (too few trades)" and found.kills == [] and found.p == 1.0


def test_a_negative_mean_is_killed_by_k1_first() -> None:
    s = slot_days(mean=-1.0)
    found = st.family({"S-C": s}, {"S-C": s.mask(s.days)}, "H", {})["S-C"]
    assert found.status == "killed" and found.first_kill is not None
    assert found.first_kill.name == "K1"


def test_a_positive_result_at_or_below_its_drift_is_killed_by_k_drift() -> None:
    s = slot_days(long_only=np.full(len(H_DAYS), 10.0))
    found = st.family({"S-C": s}, {"S-C": s.mask(s.days)}, "H", {})["S-C"]
    assert found.status == "killed" and found.first_kill.name == "K-drift"


def test_a_small_mean_below_the_bar_and_the_mde_is_labelled() -> None:
    value = np.r_[np.full(26, 2.0), np.full(18, -2.15)]  # mean 0.30, 59% of days positive
    s = slot_days(mde=5.0)
    s.value, s.latency = value, value - 0.1
    found = st.family({"S-C": s}, {"S-C": s.mask(s.days)}, "H", {})["S-C"]
    assert found.first_kill is None
    assert found.status == "inconclusive, below the minimum detectable effect"
    s.mde = 0.1
    assert st.family({"S-C": s}, {"S-C": s.mask(s.days)}, "H", {})["S-C"].status == "inconclusive"


def test_a_status_set_on_d_enters_holm_with_p_1() -> None:
    s = slot_days()
    found = st.family({"S-C": s}, {"S-C": s.mask(s.days)}, "H", {"S-R": "not admitted on D (i)"})
    assert found["S-R"].status == "not admitted on D (i)" and found["S-R"].p == 1.0
    assert found["S-C"].status == "candidate"


def test_a_slot_on_s25s_signal_is_at_best_a_candidate_before_august() -> None:
    s = slot_days("S25", on_s25_signal=True)
    window = s.mask([d for d in s.days if d <= st.S25_WINDOW_END])
    found = st.family({"S25": s}, {"S25": window}, "H", {})["S25"]
    assert found.status == "candidate (H before 2025-08-01)" and found.days == 24
    assert not st.taken_to_f(found, on_s25_signal=True)


def test_what_is_taken_to_f() -> None:
    assert st.taken_to_f(st.Assessment("S-C", "candidate", 0.01), False)
    assert st.taken_to_f(st.Assessment("S-C", "inconclusive", 0.2, mean=0.5), False)
    assert not st.taken_to_f(st.Assessment("S-C", "inconclusive", 0.2, mean=-0.5), False)
    assert not st.taken_to_f(
        st.Assessment("S-C", "inconclusive (too few trades)", 1.0, mean=5.0), False
    )
    assert not st.taken_to_f(st.Assessment("S-C", "killed", 1.0, mean=5.0), False)


@pytest.mark.parametrize(
    ("h", "f", "expected"),
    [
        ("candidate", "passes", "candidate"),
        ("candidate", "inconclusive", "candidate, not confirmed"),
        ("candidate", "killed", "killed on F"),
        ("inconclusive", "passes", "exploratory: significant on F"),
        ("inconclusive", "inconclusive (too few trades)", "inconclusive"),
        ("inconclusive", "void", "void"),
    ],
)
def test_the_route_through_f(h: str, f: str, expected: str) -> None:
    assert st.after_f(h, st.Assessment("S-C", f, 1.0)) == expected


@pytest.mark.parametrize(
    ("f", "p", "mean", "supported", "expected"),
    [
        ("candidate", "passes", 1.0, False, "confirmed"),
        ("candidate", "inconclusive", 1.0, False, "candidate, not confirmed"),
        ("candidate", "killed", 1.0, False, "killed on P"),
        ("candidate", "", -1.0, True, "confirmed until the break"),
        ("candidate", "", 1.0, True, "candidate; the break did not last"),
        ("exploratory: significant on F", "passes", 1.0, False, "confirmed on P, first significant on F"),
        ("exploratory: significant on F", "inconclusive", 1.0, False, "exploratory: significant on F, not confirmed"),
    ],
)  # fmt: skip
def test_the_route_through_p(f: str, p: str, mean: float, supported: bool, expected: str) -> None:
    assert st.after_p(f, st.Assessment("S-C", p, 1.0, mean=mean), supported) == expected


def test_f_is_split_only_with_twenty_usable_days_before_the_alarm() -> None:
    days = [date(2025, 8, 21) + timedelta(days=i) for i in range(60)]
    assert st.split_f(days, None, False, False).on_all_of_f
    assert st.split_f(days, days[10], False, False).on_all_of_f
    found = st.split_f(days, days[30], False, False)
    assert not found.on_all_of_f and len(found.minus) == 30 and len(found.plus) == 29
    assert st.split_f(days, days[30], True, False).on_all_of_f


def test_r_is_untestable_without_enough_days_on_either_side() -> None:
    days = [date(2025, 8, 21) + timedelta(days=i) for i in range(25)]
    s = slot_days(days=days)
    split = st.split_f(days, days[22], False, False)
    found = st.r_test(s, days, split, False, True, lambda _: True)
    assert found["outcome"] == "untestable"


def test_r_is_supported_when_the_edge_stops_at_b_and_nowhere_else_as_sharply() -> None:
    days = [date(2025, 8, 21) + timedelta(days=i) for i in range(80)]
    b = days[50]
    value = np.where(np.array(days) < b, 4.0, -4.0) + 0.01 * np.arange(80)
    s = slot_days(days=days)
    s.value = value
    split = st.split_f(days, b, False, False)
    found = st.r_test(s, days, split, False, True, lambda _: True)
    assert found["outcome"] == "supported" and found["p_R"] <= 0.05


def test_k_nbhd_takes_its_median_over_the_whole_neighbourhood_the_chosen_cell_included() -> None:
    from trading_research.pipeline.short_horizon.page_numbers import slot_numbers

    n = len(H_DAYS)
    near = np.array([np.full(n, -1.0), np.full(n, 1.0)])  # the other cells' median is 0
    s = slot_days(neighbours=near)
    found = st.family({"S-C": s}, {"S-C": s.mask(s.days)}, "H", {})["S-C"]
    nbhd = next(k for k in found.kills if k.name == "K-nbhd")
    assert not nbhd.fired and nbhd.value == pytest.approx(1.0)
    assert slot_numbers(s, s.days)["k_nbhd_median"] == pytest.approx(1.0)
    silent = slot_days(neighbours=np.array([np.full(n, np.nan), np.full(n, -1.0)]))
    # the cell itself near 3, a cell without a trade scoring 0 and one at -1
    assert st.neighbourhood_median(silent, silent.mask(silent.days)) == 0.0


def test_every_kill_of_s25_z_reads_both_claims() -> None:
    gross = slot_days("S25-Z", on_s25_signal=True)
    net = slot_days("S25-Z", on_s25_signal=True, mean=2.0, seed=1)
    gross.net_claim, gross.second = net, net.value
    window = gross.mask([d for d in gross.days if d <= st.S25_WINDOW_END])
    clean = st.kills(gross, window)
    assert [k.name for k in clean][:2] == ["K1", "K2"] and not any(k.fired for k in clean)
    net.flipped = np.full(len(H_DAYS), 5.0)  # the net claim does not beat its sign-flipped placebo
    found = {k.name: k for k in st.kills(gross, window)}
    assert found["K-placebo"].fired and found["K-placebo"].test.endswith("(net at 0/0)")
    assert not found["K1"].fired and not found["K2"].fired
    assert st.s25_z(gross, window).status == "killed"
    net.value = np.full(len(H_DAYS), -1.0)
    assert {k.name for k in st.kills(gross, window) if k.fired} >= {"K2"}


def test_the_mde_on_f_and_p_uses_the_usable_days_of_the_series_assessed() -> None:
    f_days = [date(2025, 8, 21) + timedelta(days=i) for i in range(11)]
    value = np.r_[np.full(6, 3.0), np.full(5, -1.0)]  # mean 1.18: no kill, not the bar
    s = slot_days(days=f_days, mde=1.0)
    s.value, s.latency = value, value - 0.1
    on_f = st.family({"S-C": s}, {"S-C": s.mask(f_days)}, "F", {})["S-C"]
    assert on_f.bar["mde"] == pytest.approx(2.0)  # 1.0 x sqrt(44 / 11)
    assert on_f.status == "inconclusive, below the minimum detectable effect"
    assert st.block_mde(s, s.mask(f_days), "H") == 1.0
    s.on_s25_signal = True
    assert st.block_mde(s, s.mask(f_days[:6]), "P") == pytest.approx(2.0)  # 1.0 x sqrt(24 / 6)


def test_a_void_result_on_f_is_void_since_killed_on_f_names_a_kill_that_fired() -> None:
    from trading_research.market_making.verdicts import KillCheck
    from trading_research.reporting.headlines import HeadlineError, slot_on_block

    void = st.family({"S-C": slot_days(void=("K-cap",))}, {"S-C": np.ones(44, bool)}, "F", {})[
        "S-C"
    ]
    assert void.status == "void" and void.kills == [] and st.after_f("candidate", void) == "void"
    assert "void" in slot_on_block("S-C", "F", "void", void.kills)
    with pytest.raises(HeadlineError):  # a void result has no kill evaluated to name
        slot_on_block("S-C", "F", "killed on F", void.kills)
    k1 = KillCheck("S-C", "K1", "mean <= 0", -1.0, 0.0, True)
    assert "K1" in slot_on_block("S-C", "F", "killed on F", [k1])
