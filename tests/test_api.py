"""The Python API on synthetic data: Backtest in each mode, the Pipeline's first steps, Study."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import trading_research
from trading_research.api import Backtest, Pipeline, PipelineConfig, Study
from trading_research.market_making.prereg import HeldOutLocked
from trading_research.walkthrough.layout import SyntheticInstrument, write_synthetic

FIRST = date(2031, 3, 3)


@pytest.fixture(scope="module")
def data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("api")
    write_synthetic(root, [SyntheticInstrument("SYNAUSDT", 3, 0.3), SyntheticInstrument("SYNBUSDT", 4, 0.0)],
                    first_day=FIRST, days=4)  # fmt: skip
    return root


def test_the_package_exports_the_api_on_first_use() -> None:
    assert trading_research.Pipeline is Pipeline and trading_research.Study is Study
    assert trading_research.Backtest is Backtest and "PipelineConfig" in trading_research.__all__
    with pytest.raises(AttributeError):
        _ = trading_research.Nonesuch  # type: ignore[attr-defined]


def rule_signal(bt: Backtest, day: date) -> tuple[pd.DataFrame, np.ndarray]:
    """Every minute, the side the touch's imbalance points to: a rule fitted to nothing."""
    book = pd.read_parquet(bt.dataset.raw / bt.symbol / "bookTicker" / f"{day}.parquet")
    rows = book.iloc[::60].reset_index(drop=True)
    imbalance = (rows["bid_size_0"] - rows["ask_size_0"]) / (
        rows["bid_size_0"] + rows["ask_size_0"]
    )
    return rows, np.sign(imbalance.to_numpy()).astype(int)


def test_a_taker_and_a_passive_backtest_on_one_signal(data: Path) -> None:
    bt = Backtest(data, "SYNAUSDT")
    day = FIRST + timedelta(days=1)
    frame, decisions = rule_signal(bt, day)
    signal = bt.signal(frame, decisions, hold_rows=12)
    taker = bt.run("taker", day, day, signal=signal, hold_rows=12)
    passive = bt.run("passive", day, day, signal=signal, hold_rows=12)
    assert len(taker.attempts) > 100 and taker.attempts["filled"].all()
    cost = taker.attempts["cost_bp"]
    assert (cost >= 2 * 5.5 + 2 * 0.5).all()  # the base taker fee both ways, plus slippage
    assert 0 < passive.summary["fill_rate"] < 1  # misses are attempts that earned nothing
    assert set(passive.attempts["path"]) >= {"unfilled"}
    assert taker.summary["days"] == passive.summary["days"] == 1
    with pytest.raises(ValueError, match="executes a signal"):
        bt.run("taker", day, day)


def test_the_market_maker_quotes_in_event_time_at_the_stated_tier(data: Path) -> None:
    bt = Backtest(data, "SYNAUSDT", fee="vip1")
    day = FIRST + timedelta(days=2)
    result = bt.run("maker", day, day, reference_days=[FIRST])
    assert result.days is not None and result.days["status"].tolist() == ["ok"]
    assert len(result.attempts) == 1 and result.summary["days"] == 1
    assert {"spread", "adverse", "inventory", "fees", "funding"} <= set(result.summary)


def test_the_market_maker_refuses_a_held_out_day_of_its_study(tmp_path: Path) -> None:
    write_synthetic(tmp_path, [SyntheticInstrument("SYNAUSDT", 5, 0.0)], first_day=date(2024, 2, 26),
                    days=1, overrides={"step_ms": 5000})  # fmt: skip
    with pytest.raises(HeldOutLocked, match="allow-heldout"):
        Backtest(tmp_path, "SYNAUSDT").run("maker", date(2024, 2, 26), date(2024, 2, 26))


def test_a_pipeline_registers_first_then_checks_the_data_and_screens(
    data: Path, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="execution must be one of"):
        PipelineConfig(data=data, execution="sometimes")
    config = PipelineConfig(data=data, out=tmp_path / "run", execution="taker")
    assert config.family == ("taker",)
    pipeline = Pipeline(config)
    pipeline.register()
    registration = json.loads(pipeline.search.registration_path.read_text())
    assert registration["family"] == ["taker"] and registration["fee"]["taker_bp"] == 5.5
    assert pipeline.data()["passed"]
    regimes = pipeline.regimes()
    assert set(regimes) == {"SYNAUSDT", "SYNBUSDT"} and all(
        r["exit_code"] == 0 for r in regimes.values()
    )
    closes = pd.read_csv(tmp_path / "run" / "regimes" / "SYNAUSDT" / "closes.csv")
    last = pd.Timestamp(
        date.fromisoformat(registration["days"]["search"][-1]) + timedelta(days=1), tz="UTC"
    )
    assert pd.to_datetime(closes["timestamp"]).max() < last  # the search block's closes only
    chosen = json.loads((pipeline.instruments() / "chosen.json").read_text())
    assert chosen["symbol"] == "SYNAUSDT"


def test_a_study_from_python_is_its_entry_point() -> None:
    spot = Study("spot_2026")
    assert spot.stages()[:4] == ("0", "1", "D", "amend")
    assert spot.registration.name == "spot_2026.md" and spot.config.name == "spot_2026.yaml"
    assert Study.registered()[:2] == ["short_horizon_2025", "spot_2026"]
    with pytest.raises(KeyError, match="no registered study"):
        Study("nonesuch")
