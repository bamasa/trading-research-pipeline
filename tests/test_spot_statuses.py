"""The spot study's status rules, every branch on synthetic per-day values."""

from __future__ import annotations

import math
import socket
from datetime import date, timedelta
from typing import Any

import numpy as np
import pytest

from trading_research.evaluation.significance import newey_west_t
from trading_research.market_making.verdicts import day_t
from trading_research.spot import statuses as st

FALLBACK = "(H inside S25's information set)"
CLIP_NOTIONAL = 1_000.0  # 0.010 BTC at 100,000 of quote


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise OSError("tests do not reach the network")

    monkeypatch.setattr(socket.socket, "connect", refuse)


def make(slot: str = "S25-BB", n: int = 75, mean: float = 5.0, sd: float = 1.0, *,
         trades: float | np.ndarray = 4.0, seed: int = 0, **changes: Any) -> st.SlotDays:  # fmt: skip
    """A slot whose series passes the bar on H unless ``changes`` say otherwise."""
    rng = np.random.default_rng(seed)
    days = [date(2026, 8, 8) + timedelta(days=i) for i in range(n)]
    value = mean + sd * rng.standard_normal(n)
    counts = np.broadcast_to(np.asarray(trades, dtype=float), (n,)).copy()
    notional = counts * CLIP_NOTIONAL
    found: dict[str, Any] = {
        "slot": slot, "days": days, "value": value, "trades": counts,
        "gross_bp_turnover": np.full(n, 15.0), "ic": np.full(n, 0.05),
        "random": sd * rng.standard_normal((st.RANDOM_DRAWS, n)), "flipped": -value,
        "shifted": sd * rng.standard_normal(n), "all_long": np.full(n, 0.5),
        "all_short": np.full(n, -0.5), "latency": value - 0.5,
        "neighbourhood": np.tile(value, (st.NEIGHBOURHOOD_CELLS, 1)), "capacity": np.full(n, 0.001),
        "mde": 1.0,
    }  # fmt: skip
    if slot in ("S25-PF", "S25-PB", "S25-PFT"):
        found["worse_of_two"] = st.FillDays(value * notional / 1e4, notional, notional, notional)
        found["fills"], found["beyond_sweep"] = 2 * counts, np.zeros(n)
    if slot in ("S25-PB", "S25-PFT"):
        found["neighbourhood"] = None
    found.update(changes)
    return st.SlotDays(**found)


def rows(**changes: st.SlotDays) -> dict[str, st.SlotDays]:
    found = {slot: make(slot, seed=i) for i, slot in enumerate(st.FAMILY)}
    return found | changes


def statuses(found: dict[str, st.Assessment]) -> dict[str, str]:
    return {slot: a.status for slot, a in found.items()}


def test_every_slot_passing_is_a_candidate_on_h_when_h_is_blind() -> None:
    found = st.on_h(rows(), blind=True, a_buy=0.5, a_sell=0.5)
    assert statuses(found) == dict.fromkeys(st.FAMILY, "candidate")
    for a in found.values():
        assert a.bar is not None and a.bar["passes"] and a.bar["holm"] and a.p < 1e-6
        assert a.trades == 300 and a.days == 75 and not any(k.fired for k in a.kills)
    assert [k.name for k in found["S25-PF"].kills] == list(st.KILL_ORDER["S25-PF"])


def test_under_the_fallback_a_candidate_carries_the_mark() -> None:
    found = st.on_h(rows(), blind=False, a_buy=0.5, a_sell=0.5)
    assert statuses(found) == dict.fromkeys(st.FAMILY, f"candidate {FALLBACK}")
    killed = st.on_h(rows(**{"S25-BB": make(mean=-1.0)}), blind=False, a_buy=0.5, a_sell=0.5)
    assert killed["S25-BB"].status == "killed"  # only the candidate is marked on H


def test_void_comes_first_and_enters_holm_with_p_one() -> None:
    every = st.on_h(rows(), blind=True, study_voids=("K-leak",), a_buy=0.5, a_sell=0.5)
    assert statuses(every) == dict.fromkeys(st.FAMILY, "void")
    assert all(a.void == ["K-leak"] and a.p == 1.0 and a.kills == [] for a in every.values())
    for condition in ("K-units", "K-acct", "K-cap", "K-inv"):
        found = st.on_h(rows(**{"S25-T": make("S25-T", mean=-3.0, void=(condition,))}), blind=True,
                        a_buy=0.5, a_sell=0.5)  # fmt: skip
        assert found["S25-T"].status == "void" and found["S25-T"].void == [condition]
        assert found["S25-PF"].status == found["S25-BB"].status == "candidate"
    with pytest.raises(ValueError, match="void condition"):
        make(void=("K-label",))


def test_too_few_trades_is_inconclusive_before_any_kill() -> None:
    for mean in (5.0, -5.0):
        few = make(trades=1.0, n=75, mean=mean)  # 75 round trips
        found = st.on_h(rows(**{"S25-BB": few}), blind=True, a_buy=0.5, a_sell=0.5)["S25-BB"]
        assert (
            found.status == "inconclusive (too few trades)" and found.kills == [] and found.p == 1.0
        )
    edge = make(trades=np.where(np.arange(75) < 25, 2.0, 1.0))  # 100 round trips
    assert st.on_h(rows(**{"S25-BB": edge}), blind=True)["S25-BB"].status == "candidate"


@pytest.mark.parametrize(
    ("changes", "kill"),
    [
        ({"mean": -1.0}, "K1"),
        ({"ic": np.zeros(75)}, "K-label"),
        ({"flipped": np.full(75, 6.0)}, "K-placebo"),
        ({"shifted": np.full(75, 9.0)}, "K-placebo"),
        ({"random": np.full((200, 75), 7.0)}, "K-placebo"),
        ({"all_long": np.full(75, 6.0)}, "K-drift"),
        ({"all_short": np.full(75, 6.0)}, "K-drift"),
        ({"neighbourhood": np.full((12, 75), -1.0)}, "K-nbhd"),
        ({"gross_bp_turnover": np.full(75, 10.0)}, "K-fee"),
        ({"latency": np.full(75, -0.1)}, "K-latency"),
    ],
)
def test_each_kill_fires_alone(changes: dict[str, Any], kill: str) -> None:
    mean = changes.pop("mean", 5.0)
    slot = make(mean=mean, **changes)
    found = st.on_h(rows(**{"S25-BB": slot}), blind=True)["S25-BB"]
    assert found.status == "killed" and found.p == 1.0
    assert [k.name for k in found.kills if k.fired] == ([kill] if kill != "K1" else found_k1(found))
    assert found.first_kill is not None and found.first_kill.name == kill


def found_k1(found: st.Assessment) -> list[str]:
    """K1 fires with the conditions a negative mean drags along; K1 comes first."""
    fired = [k.name for k in found.kills if k.fired]
    assert fired[0] == "K1"
    return fired


def test_k_days_fires_on_a_positive_mean_with_half_the_days_positive() -> None:
    value = np.where(np.arange(75) % 2 == 0, -1.0, 30.0)  # 37 of 75 days positive
    found = st.on_h(rows(**{"S25-BB": make(value=value)}), blind=True)["S25-BB"]
    assert found.status == "killed" and found.first_kill is not None
    assert found.first_kill.name == "K-days" and found.first_kill.value == pytest.approx(37 / 75)


def test_k_fill_charges_the_thinned_calibrations_advantage() -> None:
    passing = st.on_h(rows(), blind=True, a_buy=2.0, a_sell=-4.0)["S25-PF"]  # only buys charged
    assert passing.status == "candidate"
    k_fill = passing.kills[-1]
    assert k_fill.name == "K-fill" and k_fill.value == pytest.approx(passing.mean - 2.0, abs=1e-9)
    killed = st.on_h(rows(), blind=True, a_buy=3.0, a_sell=3.0)["S25-PF"]
    assert killed.status == "killed" and killed.first_kill is not None
    assert killed.first_kill.name == "K-fill"
    undefined = st.on_h(rows(), blind=True)["S25-PF"]  # no calibration: K-fill fires
    assert undefined.status == "killed" and undefined.first_kill is not None
    assert undefined.first_kill.name == "K-fill"
    negative = make("S25-PF", mean=-1.0)
    checks = st.kills(negative, 0.0, 0.0)
    assert not next(k for k in checks if k.name == "K-fill").fired  # needs a positive result
    assert not next(k for k in checks if k.name == "K-latency").fired
    with pytest.raises(ValueError, match="worse-of-two"):
        st.kills(make("S25-PF", worse_of_two=None))


def test_the_first_kill_follows_the_kill_table() -> None:
    slot = make(gross_bp_turnover=np.full(75, 5.0), all_long=np.full(75, 9.0), ic=np.full(75, -0.1))
    found = st.on_h(rows(**{"S25-BB": slot}), blind=True)["S25-BB"]
    assert [k.name for k in found.kills if k.fired] == ["K-label", "K-drift", "K-fee"]
    assert found.first_kill is not None and found.first_kill.name == "K-label"


def test_caps_apply_after_holm_and_only_when_the_bar_is_met() -> None:
    beyond = make("S25-PF", beyond_sweep=np.full(75, 1.0))  # 1 of 8 fills a day: 12.5%
    found = st.on_h(rows(**{"S25-PF": beyond}), blind=True, a_buy=0.5, a_sell=0.5)
    assert found["S25-PF"].status == "inconclusive (fill beyond sweep)"
    assert found["S25-PF"].caps["fill_beyond_sweep_share"] == pytest.approx(0.125)
    assert found["S25-PF"].p < 1e-6 and found["S25-BB"].status == "candidate"
    crowded = make(capacity=np.full(75, 0.03))
    assert (
        st.on_h(rows(**{"S25-BB": crowded}), blind=True)["S25-BB"].status
        == "inconclusive (capacity)"
    )
    both = make("S25-PF", beyond_sweep=np.full(75, 1.0), capacity=np.full(75, 0.03))
    found = st.on_h(rows(**{"S25-PF": both}), blind=True, a_buy=0.5, a_sell=0.5)
    assert found["S25-PF"].status == "inconclusive (fill beyond sweep)"
    weak = make("S25-PF", mean=0.3, sd=3.0, beyond_sweep=np.full(75, 4.0))
    found = st.on_h(rows(**{"S25-PF": weak}), blind=True, a_buy=0.0, a_sell=0.0)["S25-PF"]
    assert found.status.startswith("inconclusive") and found.caps == {}
    under = make("S25-PF", beyond_sweep=np.full(75, 0.8))  # 10%: at the cap, not above
    assert (
        st.on_h(rows(**{"S25-PF": under}), blind=True, a_buy=0, a_sell=0)["S25-PF"].status
        == "candidate"
    )


def test_a_result_short_of_the_bar_is_inconclusive_or_below_the_mde() -> None:
    v = np.where(np.arange(75) % 5 < 3, 2.0, -2.4)  # mean 0.24, 60% of days positive
    weak = make(value=v, flipped=-v, latency=v, neighbourhood=np.tile(v, (12, 1)), mde=1.0,
                random=np.full((200, 75), -1.0), shifted=np.zeros(75), all_long=np.full(75, 0.1))  # fmt: skip
    found = st.on_h(rows(**{"S25-BB": weak}), blind=True)["S25-BB"]
    assert found.bar is not None and not found.bar["t_ok"] and found.mean > 0
    assert found.status == "inconclusive, below the minimum detectable effect"
    no_mde = st.on_h(rows(**{"S25-BB": replace_mde(weak, None)}), blind=True)["S25-BB"]
    assert no_mde.status == "inconclusive"
    small = replace_mde(weak, 0.1)
    assert st.on_h(rows(**{"S25-BB": small}), blind=True)["S25-BB"].status == "inconclusive"
    short = make(n=19, trades=6.0)  # strong, 114 round trips, but 19 days with a trade on H
    found = st.on_h(rows(**{"S25-BB": short}), blind=True)["S25-BB"]
    assert found.bar is not None and found.bar["t_ok"] and not found.bar["days_ok"]
    assert found.status == "inconclusive"  # its mean is above the MDE


def replace_mde(s: st.SlotDays, mde: float | None) -> st.SlotDays:
    from dataclasses import replace

    return replace(s, mde=mde)


def test_day_statistics_read_the_daily_series_only() -> None:
    values = np.array([1.0, np.nan, 2.0, 4.0, -1.0, 3.0, np.nan, 2.5])
    found = st.day_statistics(values)
    finite = values[np.isfinite(values)]
    assert found.days == 6 and found.mean == pytest.approx(finite.mean())
    assert found.t == pytest.approx(day_t(finite).t)
    assert found.p_one_sided == pytest.approx(day_t(finite).p_one_sided)
    assert found.newey_west_t == pytest.approx(newey_west_t(finite, 5).t)
    assert found.median == pytest.approx(np.median(finite))
    one = st.day_statistics(np.array([np.nan, 3.0]))
    assert one.days == 1 and one.p_one_sided == 1.0 and math.isnan(one.t)


def test_unreadable_and_failed_reads_on_h() -> None:
    found = st.on_h(rows(), blind=True, unreadable=["bybit-spot:BTCUSDT"], a_buy=0.5, a_sell=0.5)
    assert found["S25-BB"].status == found["S25-T"].status == "not assessed (H unreadable on Bybit)"
    assert found["S25-PF"].status == "candidate" and found["S25-BB"].p == 1.0
    failed = st.on_h(rows(), blind=True, read_failed=True)
    assert statuses(failed) == dict.fromkeys(st.FAMILY, "not assessed (H read failed)")
    gone = st.on_h(rows(), blind=True, unreadable=[st.sa.BINANCE])
    assert gone["S25-PF"].status == "not assessed (H unreadable on Binance)"


def test_only_candidates_are_taken_to_p() -> None:
    h = {"S25-PF": "candidate", "S25-BB": "killed", "S25-T": "inconclusive"}
    found = st.on_p(rows(), h, a_buy=0.5, a_sell=0.5)
    assert found["S25-PF"].status == "passes" and found["S25-PF"].block == "P"
    assert found["S25-BB"].status == found["S25-T"].status == "not taken to the block"
    assert found["S25-BB"].p == 1.0
    gone = st.on_p(rows(), h, unreadable=[st.sa.BINANCE])["S25-PF"]
    assert gone.status == "not assessed (P unreadable on Binance)"
    assert st.final_status("candidate", gone, p_state="read", blind=True) == (
        "candidate, not confirmed (P unreadable)"
    )
    marked = {**h, "S25-PF": f"candidate {FALLBACK}"}
    assert st.on_p(rows(), marked, a_buy=0.5, a_sell=0.5)["S25-PF"].status == "passes"


def test_p_needs_ten_days_and_scales_the_mde() -> None:
    nine = make("S25-BB", n=9, trades=12.0)
    found = st.family(
        {"S25-BB": nine}, "P", blind=True, preset={"S25-PF": "not taken to the block"}
    )
    assert found["S25-BB"].bar is not None and not found["S25-BB"].bar["days_ok"]
    assert found["S25-T"].status == "not taken to the block"
    assert found["S25-BB"].mde == pytest.approx(1.0 * math.sqrt(75 / 9))
    assert st.block_mde(2.0, "H", 40) == 2.0 and st.block_mde(None, "P", 40) is None
    ten = make("S25-BB", n=10, trades=12.0)
    assert st.family({"S25-BB": ten}, "P", blind=True)["S25-BB"].status == "passes"


@pytest.mark.parametrize(
    ("p_status", "state", "final"),
    [
        ("passes", "read", "confirmed"),
        ("inconclusive", "read", "candidate, not confirmed"),
        ("inconclusive (too few trades)", "read", "candidate, not confirmed"),
        ("inconclusive (capacity)", "read", "candidate, not confirmed"),
        ("killed", "read", "killed on P"),
        ("void", "read", "void"),
        (None, "read failed", "candidate, not confirmed (P read failed)"),
        (None, "not read by its deadline", "candidate, not confirmed (P not read by its deadline)"),
        (None, "not yet read", "candidate (P not yet read)"),
    ],
)
def test_the_p_table(p_status: str | None, state: str, final: str) -> None:
    on_p = None if p_status is None else st.Assessment("S25-PF", "P", p_status)
    assert st.final_status("candidate", on_p, p_state=state, blind=True) == final
    marked = st.final_status(f"candidate {FALLBACK}", on_p, p_state=state, blind=False)
    assert marked == f"{final} {FALLBACK}"
    st.check_status(final)
    st.check_status(marked)


def test_any_other_status_is_final_and_carries_the_fallback() -> None:
    for status in ("killed", "inconclusive", "void", "not assessed (H read failed)"):
        assert st.final_status(status, p_state="not yet read", blind=True) == status
        assert (
            st.final_status(status, p_state="not yet read", blind=False) == f"{status} {FALLBACK}"
        )
    defect = st.defect_status("H")
    assert defect == "void (defect found after the H read)"
    assert st.final_status(defect, p_state="not yet read", blind=False) == defect
    with pytest.raises(ValueError, match="P's read"):
        st.final_status("candidate", p_state="opened", blind=True)
    with pytest.raises(ValueError):
        st.defect_status("D")


def test_p_deadline_is_the_later_of_two_dates() -> None:
    assert st.p_deadline(date(2026, 11, 20)) == date(2027, 1, 31)
    assert st.p_deadline(date(2027, 1, 15)) == date(2027, 2, 14)


def test_measured_rows_report_their_kills_without_a_status() -> None:
    row = make("S25-PB", mean=-1.0)
    checks = st.measured_kills(row, 0.5, 0.5)
    assert [k.name for k in checks] == list(st.MEASURED_KILLS)
    assert checks[0].fired and st.first_kill("S25-PB", checks) is not None
    walked = make("S25-PB", worse_of_two=None)  # its decisions priced by the book walk
    assert "K-fill" not in [k.name for k in st.measured_kills(walked)]
    with pytest.raises(ValueError, match="family"):
        st.kills(row)


def test_results_record_and_candidates() -> None:
    found = st.on_h(rows(**{"S25-BB": make(mean=-1.0)}), blind=True, a_buy=0.5, a_sell=0.5)
    record = st.results_record("H", found, stamp="first read", session=("H", "abc"), blind=True)
    assert record["statuses"]["S25-BB"]["first_kill"] == "K1"
    assert st.candidates_in(record) == ["S25-PF", "S25-T"]
    assert st.candidates_in({**record, "stamp": "second read"}) == []
    with pytest.raises(ValueError, match="H's results"):
        st.candidates_in({**record, "block": "P"})


def test_slot_days_are_checked() -> None:
    with pytest.raises(ValueError, match="one value per day"):
        make(value=np.zeros(3))
    with pytest.raises(ValueError, match="draws x days"):
        make(random=np.zeros(75))
    with pytest.raises(ValueError, match="neighbourhood"):
        st.neighbourhood_median(make(neighbourhood=np.zeros((11, 75))))
    with pytest.raises(ValueError, match="capacity"):
        st.on_h(rows(**{"S25-BB": make(capacity=None)}), blind=True)
    with pytest.raises(ValueError, match="statuses are given"):
        st.family(rows(), "D", blind=True)
