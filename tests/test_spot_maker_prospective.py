"""P's gate (its deadline, its last day published, what may change since H) and its fetch."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from trading_research.market_making.prereg import HeldOutLocked
from trading_research.spot import synthetic as syn
from trading_research.spot_maker import blocks as bk
from trading_research.spot_maker import prospective as pr
from trading_research.spot_maker import reads as rd


def test_the_deadline_and_the_published_last_day() -> None:
    assert pr.deadline(date(2026, 11, 1)) == date(2027, 1, 31)
    assert pr.deadline(date(2027, 1, 15)) == date(2027, 2, 14)
    with pytest.raises(HeldOutLocked, match="deadline"):
        pr.check_open(date(2027, 2, 1), date(2026, 11, 1), lambda: True)
    with pytest.raises(HeldOutLocked, match="not published"):
        pr.check_open(date(2027, 1, 5), date(2026, 11, 1), lambda: False)
    pr.check_open(date(2027, 1, 5), date(2026, 11, 1), lambda: True)
    venue = syn.SpotVenue(syn.SpotMarket(frozenset({date(2026, 12, 31)})))
    assert pr.published(date(2026, 12, 31), venue)
    assert not pr.published(date(2026, 12, 30), venue)


def test_only_results_docs_and_the_fee_amendment_change_since_h() -> None:
    ok = ["experiments/results/spot_maker_2026_H.json", "docs/spot_maker_2026_run_log.md",
          "README.md", "configs/spot_maker_2026.yaml", "docs/preregistration/spot_maker_2026.md"]  # fmt: skip
    pr.check_since_h(ok, "a", "a", "text and more", "text")
    with pytest.raises(HeldOutLocked, match="changed since H"):
        pr.check_since_h(["src/trading_research/spot_maker/day.py"], "a", "a", "x", "x")
    with pytest.raises(HeldOutLocked, match="configuration"):
        pr.check_since_h(ok, "b", "a", "x", "x")
    with pytest.raises(HeldOutLocked, match="appending"):
        pr.check_since_h(ok, "a", "a", "new text", "old text")
    assert rd.block_mde({"MK-S": 1.0}, "P", 75 // 4) == {"MK-S": pytest.approx(75**0.5 / 18**0.5)}
    assert rd.block_mde({"MK-S": 1.0}, "H", 70) == {"MK-S": 1.0}


def test_p_days_are_fetched_inside_the_sealed_access(tmp_path: Path) -> None:
    days = [date(2026, 11, 1), date(2026, 11, 2)]
    venue = syn.SpotVenue(syn.SpotMarket(frozenset(days[:1])))
    access = bk.MakerAccess(bk.opened_blocks("P"), "first read", ("P", "c", "s", "f"), seal=bk._SEAL)
    with syn.no_network():
        found = pr.fetch_days(tmp_path, days, access, opener=venue, sleep=lambda _: None)
    assert found["2026-11-01"]["status"] == "fetched" and bk.day_file(tmp_path, days[0]).exists()
    assert found["2026-11-02"]["status"] == "absent"
    with pytest.raises(HeldOutLocked):
        pr.fetch_days(tmp_path, days, bk.development_access(), opener=venue)
