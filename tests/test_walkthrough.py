"""The search's pieces: views, the registration, the order of the steps, the ledger, the verdict's arithmetic."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_research.backtest.costs import fee_tier
from trading_research.pipeline.discovery import _with_neighbourhood
from trading_research.pipeline.execution import ExecutionMode, ExecutionResult
from trading_research.walkthrough import execution as ex
from trading_research.walkthrough import signal as sg
from trading_research.walkthrough import steps
from trading_research.walkthrough import verdict as vd
from trading_research.walkthrough.layout import (
    Dataset,
    SyntheticInstrument,
    protected_overlap,
    split_days,
    view,
    write_synthetic,
)
from trading_research.walkthrough.record import ReadRefused, SearchError, SearchRun

FIRST = date(2031, 3, 3)


@pytest.fixture(scope="module")
def data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("walk")
    write_synthetic(root, [SyntheticInstrument("SYNAUSDT", 1, 0.3), SyntheticInstrument("SYNBUSDT", 2, 0.0)],
                    first_day=FIRST, days=4, overrides={"step_ms": 5000})  # fmt: skip
    return root


def test_the_synthetic_market_is_written_in_every_planes_layout(data: Path) -> None:
    dataset = Dataset.under(data)
    assert dataset.symbols() == ["SYNAUSDT", "SYNBUSDT"]
    assert dataset.days("SYNAUSDT") == [FIRST + timedelta(days=i) for i in range(4)]
    assert dataset.has_event_time("SYNAUSDT", dataset.days("SYNAUSDT"))
    prints = pd.read_parquet(dataset.trades / "SYNAUSDT" / f"{FIRST}.parquet")  # type: ignore[operator]
    raw = pd.read_parquet(dataset.raw / "SYNAUSDT" / "aggTrades" / f"{FIRST}.parquet")
    assert (prints["aggressor"] == np.where(raw["is_buyer_maker"], -1, 1)).all()
    record = json.loads((data / "synthetic.json").read_text())
    assert [i["seed"] for i in record["instruments"]] == [1, 2]


def test_a_view_holds_only_the_days_it_is_given(data: Path, tmp_path: Path) -> None:
    dataset = Dataset.under(data)
    search, held = split_days(dataset.days("SYNAUSDT"), 0.65)
    assert (len(search), len(held)) == (3, 1) and max(search) < min(held)
    seen = view(dataset, ["SYNAUSDT"], search, tmp_path / "v")
    for plane in (seen.raw / "SYNAUSDT" / "bookTicker", seen.book / "SYNAUSDT"):  # type: ignore[operator]
        assert sorted(p.stem for p in plane.glob("*.parquet")) == [str(d) for d in search]
    assert not list((tmp_path / "v").rglob(f"*{held[0]}*"))
    again = view(dataset, ["SYNAUSDT"], search[:1], tmp_path / "v")  # made afresh
    assert len(list((again.raw / "SYNAUSDT" / "bookTicker").glob("*.parquet"))) == 1


def test_a_registered_range_is_refused_before_anything_is_read(tmp_path: Path) -> None:
    # The calendar is every registration on every local branch, so another study
    # registered on these days (the spot maker's H, on its own branch) may join it.
    assert "spot_2026:H" in protected_overlap([date(2026, 8, 10)])
    assert protected_overlap([FIRST]) == []
    root = tmp_path / "data"
    write_synthetic(root, [SyntheticInstrument("SYNAUSDT", 1, 0.3)], first_day=date(2026, 8, 9),
                    days=2, overrides={"step_ms": 60_000})  # fmt: skip
    with pytest.raises(SearchError, match="spot_2026:H"):
        steps.register(SearchRun(tmp_path / "run"), Dataset.under(root))
    assert not (tmp_path / "run" / "registration.json").exists()


def test_the_steps_run_in_order_and_the_registration_does_not_move(
    data: Path, tmp_path: Path
) -> None:
    run = SearchRun(tmp_path / "run")
    with pytest.raises(SearchError, match=r"run the instruments step first|trp search instruments"):
        steps.frequency(run)
    steps.register(run, Dataset.under(data), family=("taker",))
    with pytest.raises(SearchError, match="registered once"):
        steps.register(run, Dataset.under(data))
    registration = json.loads(run.registration_path.read_text())
    assert registration["holm"]["m"] == 1 and [k["name"] for k in registration["kill_conditions"]] == [
        "K1", "K-placebo", "K-days",
    ]  # fmt: skip
    steps.instruments(run)
    chosen = json.loads((run.step_dir("instruments") / "chosen.json").read_text())
    assert chosen["symbol"] == "SYNAUSDT" and chosen["held_out_days"] == [
        str(FIRST + timedelta(days=3))
    ]
    run.registration_path.write_text(run.registration_path.read_text().replace('"m": 1', '"m": 2'))
    with pytest.raises(SearchError, match="changed after instruments ran"):
        steps.frequency(run)


def test_the_ledger_records_a_read_before_it_and_refuses_a_second(tmp_path: Path) -> None:
    run = SearchRun(tmp_path / "run")
    run.register({"family": ["taker"]})
    frozen = tmp_path / "frozen.json"
    with pytest.raises(ReadRefused, match="freezes every choice"):
        run.open_held_out(frozen)
    frozen.write_text("{}")
    first = run.open_held_out(frozen)
    assert first["read"] == "first read" and len(first["frozen_sha256"]) == 64
    with pytest.raises(ReadRefused, match="second read needs --second-read"):
        run.open_held_out(frozen)
    second = run.open_held_out(frozen, second_read=True)
    assert second["read"] == "second read"
    assert len(json.loads(run.ledger_path.read_text())["H"]) == 2


def test_the_neighbourhood_is_reproduces_rule_on_named_axes() -> None:
    rng = np.random.default_rng(0)
    table = pd.DataFrame(
        [{"hold_s": h, "threshold_bp": t, "net_per_trade_bp": float(rng.normal())}
         for h in (30, 60, 120) for t in (0.0, 5.0, 10.0)]
    )  # fmt: skip
    ours = sg.neighbourhood(table, "threshold_bp", "hold_s")
    theirs = _with_neighbourhood(
        table.rename(columns={"threshold_bp": "lookback_s"}).assign(rate=0.0)
    )
    np.testing.assert_allclose(ours["neighbourhood_bp"], theirs["neighbourhood_bp"])
    winner, peak = sg.choose(ours)
    assert peak["net_per_trade_bp"] == table["net_per_trade_bp"].max()
    assert winner["neighbourhood_bp"] == ours["neighbourhood_bp"].max()


def test_relabel_moves_the_mid_within_each_day_only() -> None:
    stamps = pd.date_range("2031-03-03 23:59:40", periods=6, freq="5s", tz="UTC")
    frame = pd.DataFrame({"timestamp": stamps, "mid": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]})
    out = sg.relabel(frame, 2)
    assert out["forward_bp"].iloc[0] == pytest.approx(np.log(102 / 100) * 1e4)
    assert np.isnan(out["forward_bp"].iloc[3:]).all()  # no move reaches across midnight


def _result(net: list[float], days: list[str], *, path: str = "hold") -> ExecutionResult:
    attempts = pd.DataFrame({"symbol": "X", "day": days, "ts": 0, "filled": True,
                             "gross_bp": np.array(net) + 12.0, "cost_bp": 12.0, "net_bp": net, "path": path})  # fmt: skip
    return ExecutionResult(attempts, {}, ExecutionMode.TAKER)


def test_the_fee_ladder_re_costs_the_same_attempts() -> None:
    base = ex.Costs.named("base")
    result = _result([1.0, 3.0], ["2031-03-09", "2031-03-10"])
    assert ex.at_fee(result, "taker", base, fee_tier("base")) == pytest.approx(2.0)
    assert ex.at_fee(result, "taker", base, fee_tier("vip1")) == pytest.approx(
        2.0 + 2 * (5.5 - 4.0)
    )
    passive = _result([1.0, 3.0], ["2031-03-09", "2031-03-10"])
    # a maker fee in, a taker fee out when the exit crossed
    expected = 2.0 + (2.0 - 1.8) + (5.5 - 4.0)
    assert ex.at_fee(passive, "passive", base, fee_tier("vip1")) == pytest.approx(expected)


def test_the_verdict_applies_the_kills_in_order_and_holm_across_the_family() -> None:
    good = _result(
        [5.0, 6.0, 5.5, 6.5, 5.2, 6.1] * 3, [f"2031-03-{9 + i % 6:02d}" for i in range(18)]
    )
    bad = _result([-1.0, -2.0, -1.5] * 2, [f"2031-03-{9 + i % 3:02d}" for i in range(6)])
    modes = {
        "taker": vd.assess_mode("taker", good.attempts, [0.0, 0.5, 1.0], min_days=2),
        "passive": vd.assess_mode("passive", bad.attempts, [0.0, 0.5, 1.0], min_days=2),
    }
    decided = vd.decide(modes)
    assert decided["taker"]["status"] == "candidate" and decided["taker"]["holm"]
    assert decided["passive"]["status"] == "killed" and decided["passive"]["decided_by"][0] == "K1"
    beaten = vd.decide({"taker": vd.assess_mode("taker", good.attempts, [9.0, 10.0], min_days=2)})
    assert beaten["taker"]["decided_by"] == ["K-placebo"]
    few = vd.decide({"taker": vd.assess_mode("taker", good.attempts, [0.0], min_days=30)})
    assert few["taker"]["status"] == "inconclusive" and few["taker"]["decided_by"] == ["K-days"]


def test_placebo_shifts_move_the_decisions_by_at_least_an_hour() -> None:
    found = vd.shifts(4 * 17_280, 20, 7)
    assert len(found) == 20 and all(720 <= s <= 4 * 17_280 - 720 for s in found)
    assert vd.shifts(4 * 17_280, 20, 7) == found  # seeded
    decisions = np.zeros(10, dtype=int)
    decisions[2] = 1
    assert vd.rolled(decisions, 3)[5] == 1
