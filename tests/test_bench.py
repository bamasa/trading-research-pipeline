"""The benchmark: its tracks, its scoring rule, and the command that runs it."""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import pytest
from typer.testing import CliRunner

from trading_research.bench import DISCLAIMER, TRACKS, MarketCache, card_page, run
from trading_research.bench import tracks as tracks_module
from trading_research.bench.score import FEE_LADDER, TrackScore, score_track, verdict
from trading_research.cli import app
from trading_research.plugins import get


@pytest.fixture
def small_book(monkeypatch: pytest.MonkeyPatch) -> None:
    """The book tracks at the generator's smallest size, so a test runs in seconds."""
    monkeypatch.setattr(tracks_module, "BOOK_STEPS", 24_000)


def test_every_track_states_its_question_and_what_a_good_strategy_does() -> None:
    assert len(TRACKS) == 7
    for track in TRACKS.values():
        assert track.kind in ("synthetic", "recorded", "forward")
        assert track.question.strip() and track.expected.strip() and track.source.strip()
    planted = [t for t in TRACKS.values() if t.planted]
    nulls = [t for t in TRACKS.values() if t.kind == "synthetic" and not t.planted]
    assert len(planted) == len(nulls) == 2  # each planted effect has its null


def test_a_recorded_track_without_its_data_is_skipped_and_says_where_to_get_it(
    tmp_path: Path,
) -> None:
    score = score_track(
        TRACKS["bybit-2024-02"], lambda: get("reversion", "strategy").build(), data_root=tmp_path
    )
    assert score.status == "skipped"
    assert "data/universe" in score.reason


def test_the_fee_ladder_and_the_break_even_fee_agree_with_net(small_book: None) -> None:
    score = score_track(
        TRACKS["planted-book"], lambda: get("book_imbalance", "strategy").build(), data_root=Path()
    )
    assert score.status == "scored" and score.trades > 0
    assert score.net_at["base"] == pytest.approx(score.net_bp, abs=1e-9)
    other = score.cost_bp - 2 * FEE_LADDER[0][1]
    at_break_even = score.gross_bp - other - 2 * score.breakeven_taker_bp
    assert at_break_even == pytest.approx(0.0, abs=1e-9)
    assert score.net_at["none (hypothetical)"] > score.net_at["base"]
    assert 0.0 <= score.placebo <= 1.0


def test_a_strategy_missing_an_input_the_track_lacks_is_not_applicable(small_book: None) -> None:
    from trading_research.bench.tracks import TrackData

    built = TRACKS["planted-book"].markets(Path())
    stripped = TrackData(
        [
            [(m.__class__(**{**m.__dict__, "frame": m.frame.drop(columns=["bid_size_0"])}), s)]
            for m, s, _ in built
        ]
    )
    score = score_track(
        TRACKS["planted-book"],
        lambda: get("book_imbalance", "strategy").build(),
        data_root=Path(),
        markets=stripped,
    )
    assert score.status == "n/a" and "bid_size_0" in score.reason


def _scored(track: str, **numbers: float) -> TrackScore:
    base = {"gross_bp": 1.0, "net_bp": -10.0, "gross_t": 0.0, "net_t": 0.0, "placebo": 0.5}
    return TrackScore(track, TRACKS[track].kind, "scored", trades=100, **{**base, **numbers})


@pytest.mark.parametrize(
    ("track", "numbers", "expected", "passed"),
    [
        ("planted-book", {"gross_t": 4.0, "placebo": 1.0}, "found the planted", True),
        ("planted-book", {"gross_t": 4.0, "placebo": 0.5}, "did not find", None),
        ("planted-book", {"gross_t": 1.0, "placebo": 1.0}, "did not find", None),
        ("null-book", {"gross_t": 3.5}, "earned where nothing was planted", False),
        ("null-book", {"gross_t": 2.9}, "quiet", True),
        ("bybit-2024-02", {"net_bp": 2.0, "net_t": 2.5, "placebo": 1.0}, "pays the base fee", True),
        (
            "bybit-2024-02",
            {"net_bp": 2.0, "net_t": 1.0, "placebo": 1.0},
            "not distinguishable",
            False,
        ),
        ("bybit-2024-02", {"gross_bp": 3.0, "placebo": 0.95}, "smaller than the fee", False),
        ("bybit-2024-02", {"gross_bp": 3.0, "placebo": 0.6}, "nothing beyond chance", False),
    ],
)
def test_the_verdict_rule(
    track: str, numbers: dict[str, float], expected: str, passed: bool | None
) -> None:
    said, ok = verdict(TRACKS[track], _scored(track, **numbers))
    assert expected in said and ok is passed


def test_the_reversion_rule_declares_the_cooldown_the_benchmark_honours() -> None:
    from trading_research.strategies.reversion import ReversionConfig

    assert get("reversion", "strategy").build().cooldown == ReversionConfig().cooldown > 0


def test_a_card_carries_marks_a_fee_ladder_and_the_disclaimer(small_book: None) -> None:
    card = run(
        get("book_imbalance", "strategy"), {}, [TRACKS["planted-book"], TRACKS["bybit-2024-02"]],
        MarketCache(Path("/nonexistent")),
    )  # fmt: skip
    assert len(card.marks) == 2 and card.marks[1] == "-"
    page = card_page(card)
    assert "none (hypothetical)" in page and DISCLAIMER in page


def test_bench_run_writes_the_card(tmp_path: Path, small_book: None) -> None:
    result = CliRunner().invoke(
        app,
        ["bench", "run", "book_imbalance", "--tracks", "planted-book,null-book",
         "--output", str(tmp_path), "--data", str(tmp_path)],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    card = json.loads((tmp_path / "book_imbalance" / "card.json").read_text())
    assert [s["track"] for s in card["scores"]] == ["planted-book", "null-book"]
    assert (tmp_path / "book_imbalance" / "card.md").exists()


def test_bench_list_and_an_unknown_track() -> None:
    runner = CliRunner()
    listed = runner.invoke(app, ["bench", "list", "--data", "/nonexistent"])
    assert listed.exit_code == 0 and "planted-reversion" in listed.output
    refused = runner.invoke(app, ["bench", "run", "reversion", "--tracks", "nowhere"])
    assert refused.exit_code == 2


def test_nan_numbers_are_written_as_null() -> None:
    row = TrackScore("null-book", "synthetic", "n/a", reason="x").as_dict()
    assert row["gross_bp"] is None and not any(
        isinstance(v, float) and math.isnan(v) for v in row.values()
    )


def test_bench_board_writes_the_board_and_a_card_each(tmp_path: Path, small_book: None) -> None:
    result = CliRunner().invoke(
        app,
        ["bench", "board", "--names", "book_imbalance,reversion", "--tracks", "planted-book",
         "--output", str(tmp_path), "--data", str(tmp_path)],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    board = json.loads((tmp_path / "board.json").read_text())
    assert [r["name"] for r in board["rows"]] == ["book_imbalance", "reversion"]
    assert all(len(r["marks"]) == 1 for r in board["rows"])
    assert (tmp_path / "reversion" / "card.md").exists()


def test_the_synthetic_tracks_are_the_bytes_version_1_froze() -> None:
    from trading_research.bench.manifest import FINGERPRINTS

    for name in ("null-book", "planted-reversion"):
        data = TRACKS[name].markets(Path())
        assert data.fingerprint() == FINGERPRINTS[name], name
        assert len(data.parts) == 3  # three seeds, each its own group


def test_every_track_has_a_frozen_fingerprint_and_a_card_says_whether_it_matches() -> None:
    from trading_research.bench import data_check
    from trading_research.bench.manifest import FINGERPRINTS, FORWARD_TRACK, VERSION

    assert VERSION == "1" and set(FINGERPRINTS) == set(TRACKS) - {FORWARD_TRACK}
    same = TrackScore("null-book", "synthetic", "scored", data_sha256=FINGERPRINTS["null-book"])
    other = TrackScore("null-book", "synthetic", "scored", data_sha256="0" * 64)
    assert data_check(same) == "matches" and data_check(other) == "differs"
    assert data_check(TrackScore("null-book", "synthetic", "skipped")) == "unknown"


def test_trades_cluster_within_a_seed_never_across(small_book: None) -> None:
    score = score_track(
        TRACKS["null-book"], lambda: get("book_imbalance", "strategy").build(), data_root=Path()
    )
    # three seeds share one clock, so hour clusters would merge across them
    # if the group were ignored; each seed's hours count separately
    assert score.groups == 3 and score.clusters >= 3 * 1


def test_the_board_corrects_for_the_family_with_holm() -> None:
    from trading_research.bench import Card, board_rows

    cards = []
    for name, p in (("a", 0.01), ("b", 0.02), ("c", 0.5)):
        score = TrackScore("bybit-2024-02", "recorded", "scored", net_p=p)
        cards.append(Card(name, "strategy", "test", {}, None, [score]))
    rows = {r["name"]: r["bybit-2024-02"]["holm_p"] for r in board_rows(cards)}
    assert rows["a"] == pytest.approx(0.03) and rows["b"] == pytest.approx(0.04)
    assert rows["c"] == pytest.approx(0.5)


def test_fetch_says_what_it_would_download_and_verify_reports_the_data(tmp_path: Path) -> None:
    runner = CliRunner()
    planned = runner.invoke(app, ["bench", "fetch", "--dry-run", "--data", str(tmp_path)])
    assert planned.exit_code == 0, planned.output
    assert "26 instruments" in planned.output and "to fetch" in planned.output
    assert not any(tmp_path.iterdir())  # a dry run writes nothing
    checked = runner.invoke(
        app, ["bench", "verify", "--tracks", "null-book,bybit-2024-02", "--data", str(tmp_path)]
    )
    assert checked.exit_code == 0, checked.output
    assert "matches version 1" in checked.output and "not on disk" in checked.output


def test_a_synthetic_book_does_not_move_with_the_last_bits_of_exp() -> None:
    """Platforms differ in exp's last bits; the tick grid rounds them away."""
    import numpy as np
    import pandas as pd

    index = pd.date_range("2026-01-01", periods=20_000, freq="5s", tz="UTC")
    rng = np.random.default_rng(7)
    log_price = np.log(73.0) + np.cumsum(rng.normal(0.0, 3e-4, len(index)))
    exact = tracks_module._touch_from_log(pd.Series(log_price, index=index))
    for ulps in (1, 3):
        nudged = log_price
        for _ in range(ulps):
            nudged = np.nextafter(nudged, np.inf)
        moved = tracks_module._touch_from_log(pd.Series(nudged, index=index))
        assert exact.equals(moved), ulps


def test_the_forward_track_is_sealed_until_its_days_are_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    track = TRACKS["bybit-2026-11"]
    reversion = lambda: get("reversion", "strategy").build()  # noqa: E731
    monkeypatch.setattr(tracks_module, "today", lambda: date(2026, 11, 20))
    sealed = score_track(track, reversion, data_root=tmp_path)
    assert sealed.status == "skipped" and "sealed until 2026-11-21" in sealed.reason
    runner = CliRunner()
    assert (
        "sealed until 2026-11-21"
        in runner.invoke(app, ["bench", "fetch", "--dry-run", "--data", str(tmp_path)]).output
    )
    monkeypatch.setattr(tracks_module, "today", lambda: date(2026, 11, 21))
    opened = score_track(track, reversion, data_root=tmp_path)
    assert opened.status == "skipped" and "found 0 of 26" in opened.reason


def test_the_forward_track_trades_the_same_instruments_with_pol_for_matic() -> None:
    from trading_research.bench.manifest import FORWARD_REGISTERED
    from trading_research.bench.tracks import FORWARD_SYMBOLS, RECORDED, RECORDED_SYMBOLS

    assert len(FORWARD_SYMBOLS) == len(RECORDED_SYMBOLS) == 26
    assert set(FORWARD_SYMBOLS) ^ set(RECORDED_SYMBOLS) == {"POLUSDT", "MATICUSDT"}
    _, fitted, scored, last = RECORDED["bybit-2026-11"]
    registered = date.fromisoformat(FORWARD_REGISTERED)
    assert fitted < scored and registered < scored <= last < TRACKS["bybit-2026-11"].opens


def test_the_strategies_read_on_the_forward_track_are_the_ones_registered() -> None:
    """Changing one of the twelve before the read would make it a new strategy."""
    from trading_research.bench.manifest import FORWARD_STRATEGIES

    assert len(FORWARD_STRATEGIES) == 12
    for (name, kind), sha in FORWARD_STRATEGIES.items():
        assert get(name, kind).fingerprint() == sha, name


def test_the_state_measure_sees_the_planted_reversion_and_not_its_null() -> None:
    from trading_research.bench.state import index_autocorrelation

    planted = index_autocorrelation(TRACKS["planted-reversion"].markets(Path()))
    null = index_autocorrelation(TRACKS["null-reversion"].markets(Path()))
    assert planted < -0.1 and abs(null) < 0.05
    shown = CliRunner().invoke(app, ["bench", "state", "--tracks", "planted-reversion"])
    assert shown.exit_code == 0 and f"{planted:+.4f}" in shown.output


def test_the_fitted_window_reads_only_the_days_before_the_seal(tmp_path: Path) -> None:
    """The fitted days are fetched and fingerprinted early; a scored day is never read."""
    import pandas as pd

    from trading_research.bench.tracks import FORWARD_SYMBOLS, fitted_days, fitted_window

    days = fitted_days("bybit-2026-11")
    assert len(days) == 28 and days[0] == date(2026, 10, 9) and days[-1] == date(2026, 11, 5)

    def write(symbol: str, day: date, price: float) -> None:
        stamps = pd.date_range(pd.Timestamp(day, tz="UTC"), periods=4, freq="6h")
        frame = pd.DataFrame(
            {"timestamp": stamps, "bid_price_0": price, "bid_size_0": 1.0,
             "ask_price_0": price + 0.01, "ask_size_0": 1.0}
        )  # fmt: skip
        folder = tmp_path / "forward_2026_11" / symbol
        folder.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(folder / f"{day.isoformat()}.parquet")

    for symbol in FORWARD_SYMBOLS[:20]:
        for day in days:
            write(symbol, day, 10.0)
    before = fitted_window("bybit-2026-11", tmp_path).fingerprint()
    write(FORWARD_SYMBOLS[0], date(2026, 11, 6), 99.0)  # a scored day
    # and the first snapshot of the scored day, which the last fitted day's archive carries
    late = tmp_path / "forward_2026_11" / FORWARD_SYMBOLS[0] / "2026-11-05.parquet"
    frame = pd.read_parquet(late)
    spill = frame.iloc[[-1]].assign(
        timestamp=pd.Timestamp("2026-11-06 00:00:01", tz="UTC"), bid_price_0=99.0
    )
    pd.concat([frame, spill]).to_parquet(late)
    assert fitted_window("bybit-2026-11", tmp_path).fingerprint() == before
    with pytest.raises(FileNotFoundError):
        fitted_window("bybit-2026-11", tmp_path / "elsewhere")
