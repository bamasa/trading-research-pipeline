"""The paper commands end to end on stored days, the key guard, and the recorder's layouts."""

from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from trading_research.cli import app
from trading_research.data import bybit, bybit_trades
from trading_research.live.guard import check_config
from trading_research.live.record import export_day, parse_raw_log
from trading_research.live.stream import connections

runner = CliRunner()
FIXTURES = Path(__file__).parent / "fixtures" / "paper"


def raw_log(venue: str, market: str) -> Path:
    return FIXTURES / f"{venue}-{market}-BTCUSDT-2026-10-27.jsonl"


def test_replay_of_a_recorded_day_writes_a_page_and_status(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    result = runner.invoke(
        app,
        ["paper", "replay", "--day", str(raw_log("bybit", "spot")), "--venue", "bybit",
         "--market", "spot", "--symbol", "BTCUSDT", "--execution", "maker", "--clip", "0.0001",
         "--run-dir", str(run_dir)],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    for name in ("page.md", "manifest.json", "summary.json", "fills.csv", "orders.csv",
                 "decisions.csv", "gaps.csv", "marks.parquet", "quotes.parquet", "status.json"):  # fmt: skip
        assert (run_dir / name).exists(), name
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["fees"]["tier"] == "non_vip"
    assert manifest["model"]["name"] == "always_hold"
    assert set(manifest["files"]) >= {"fills.csv", "marks.parquet"}
    page = (run_dir / "page.md").read_text()
    assert "No order was sent anywhere" in page and "PnL net at the fee used" in page
    status = runner.invoke(app, ["paper", "status", str(run_dir)])
    assert status.exit_code == 0 and "finished" in status.output


def test_replay_with_a_trained_model_on_the_demo_day(tmp_path: Path) -> None:
    from trading_research.models.linear import LogisticBaseline

    demo = tmp_path / "demo"
    assert runner.invoke(app, ["generate-demo-data", "-o", str(demo), "-n", "800"]).exit_code == 0
    book = pd.read_parquet(demo / "book.parquet")
    features = ["log_depth_ratio", "ofi_1", "ofi_20"]
    from trading_research.features import build

    x = build(book, features)[features].dropna()
    y = pd.Series([-1, 0, 1] * (len(x) // 3 + 1), index=range(len(x) + 3))[: len(x)]
    model = LogisticBaseline().fit(x.reset_index(drop=True), y.reset_index(drop=True))
    target = tmp_path / "model"
    target.mkdir()
    with (target / "model.pkl").open("wb") as handle:
        pickle.dump({"model": model, "features": features}, handle)
    run_dir = tmp_path / "run"
    result = runner.invoke(
        app,
        ["paper", "replay", "--day", str(demo), "--model", str(target), "--execution", "taker",
         "--fee-tier", "base", "--fee-asof", "2026-09-15", "--every", "5",
         "--min-confidence", "0.34", "--hold-s", "10", "--run-dir", str(run_dir)],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["model"]["feature_set"] == "registry"
    assert manifest["model"]["sha256"] and manifest["model"]["features"] == features
    decisions = pd.read_csv(run_dir / "decisions.csv")
    assert len(decisions) == 160 and (decisions["outcome"] == "entered").any()
    fills = pd.read_csv(run_dir / "fills.csv")
    assert len(fills) >= 2 and set(fills["fee_status"]) == {
        "sourced as of 2026-09-15T00:00:00+00:00"
    }


def test_a_day_the_fee_table_does_not_reach_is_refused_without_asof(tmp_path: Path) -> None:
    demo = tmp_path / "demo"
    runner.invoke(app, ["generate-demo-data", "-o", str(demo), "-n", "100"])
    result = runner.invoke(
        app, ["paper", "replay", "--day", str(demo), "--run-dir", str(tmp_path / "r")]
    )
    assert result.exit_code == 2 and "--fee-asof" in result.output


def test_the_guard_refuses_anything_that_looks_like_a_key(tmp_path: Path) -> None:
    urls = [c.url for c in connections("bybit", "spot", "BTCUSDT")]
    assert check_config({"venue": "bybit", "symbol": "BTCUSDT"}, urls) == []
    assert check_config({"x": "BYBIT_API_KEY"}, urls)
    assert check_config({"x": "${BINANCE_SECRET}"}, urls)
    assert check_config({"x": str(tmp_path / ".env")}, urls)
    config = tmp_path / "settings.yaml"
    config.write_text("venue: bybit\napi_key: abc\n")
    assert check_config({"config": str(config)}, urls)
    assert check_config({}, ["wss://stream.bybit.com/v5/private"])
    assert check_config({}, ["wss://stream.binance.com:9443/ws/abcdefListenKey"])
    for venue in ("bybit", "binance"):
        for market in ("spot", "linear"):
            assert check_config({}, [c.url for c in connections(venue, market, "BTCUSDT")]) == []


def test_run_refuses_a_key_like_configuration_before_connecting(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["paper", "run", "--venue", "bybit", "--market", "spot", "--symbol", "BTCUSDT",
         "--minutes", "1", "--model", "BYBIT_API_KEY", "--run-dir", str(tmp_path / "r")],
    )  # fmt: skip
    assert result.exit_code == 2 and "refuses to start" in result.output
    assert not (tmp_path / "r").exists()


def test_a_recorded_day_exports_to_the_archive_and_staged_layouts(tmp_path: Path) -> None:
    root = tmp_path / "paper"
    log = root / "bybit-linear" / "BTCUSDT" / "raw" / "2026-10-27.jsonl.gz"
    log.parent.mkdir(parents=True)
    log.write_bytes(gzip.compress(raw_log("bybit", "linear").read_bytes()))
    written = export_day(log, "bybit", "linear", "BTCUSDT", root)
    assert set(written) == {"archive", "book", "trades", "staged_book", "staged_trades"}
    rows, prints = parse_raw_log(log, "bybit", "linear")
    replayed = bybit.reconstruct(written["archive"], symbol="BTCUSDT")
    assert len(replayed) == len(rows)
    assert (replayed["bid_price_0"].to_numpy() == [r.bid_px[0] for r in rows]).all()
    trades = bybit_trades.load_day(
        "BTCUSDT", pd.Timestamp("2026-10-27").date(), root / "bybit-linear" / "BTCUSDT" / "trades"
    )
    assert len(trades) == len(prints)
    assert (trades["aggressor"].to_numpy() == [p.aggressor for p in prints]).all()
    from trading_research.pipeline.stages import load_prepared, prepare

    out = prepare(
        root / "bybit-linear" / "BTCUSDT" / "staged", tmp_path / "prepared", "BTCUSDT", horizon=5
    )
    assert len(load_prepared(out)) == len(rows)
