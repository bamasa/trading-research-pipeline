"""The reversion tape for event time: right-labelled, causal, and guarded.

A synthetic universe is written to disk in the layout the downloader leaves,
and the tape is built from it the way the study builds it. Causality is checked
by truncation: the tape built from less data agrees with the tape built from
more on every label the shorter one has, and rewriting data after a moment
changes nothing labelled before it.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_research.market_making import prereg, signals
from trading_research.market_making.events import NS_PER_S, day_start_ns, within_day
from trading_research.strategies.reversion import ReversionConfig, decide, threshold_for_rate

SYMBOLS = ("AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT", "EEEUSDT")
FIRST = date(2024, 2, 1)
DAYS = [FIRST + timedelta(days=i) for i in range(3)]
CONFIG = ReversionConfig()


def write_universe(
    root: Path, days: list[date], *, seed: int = 0, shock_after: int | None = None
) -> None:
    """Sparse top-of-book updates for each instrument: a common walk plus noise.

    ``shock_after`` rewrites every price after that many seconds into the last
    day, so a causality test can change the future and nothing else.
    """
    rng = np.random.default_rng(seed)
    for day in days:
        start = pd.Timestamp(day, tz="UTC")
        seconds = np.sort(rng.choice(86_400 * 10, size=12_000, replace=False)) / 10.0
        common = np.cumsum(rng.normal(0, 4e-4, len(seconds)))
        for k, symbol in enumerate(SYMBOLS):
            own = np.cumsum(rng.normal(0, 3e-4, len(seconds)))
            mid = 1.0 + 0.1 * k + common + own
            if shock_after is not None and day == days[-1]:
                mid = np.where(seconds > shock_after, mid * 1.05 + 0.01, mid)
            half = 0.0001 * (1 + k % 2)
            frame = pd.DataFrame(
                {
                    "timestamp": start + pd.to_timedelta(seconds, unit="s"),
                    "bid_price_0": np.round(mid - half, 6),
                    "ask_price_0": np.round(mid + half, 6),
                }
            )
            # The archive spills a row past midnight; it must never be read.
            spill = pd.DataFrame(
                {
                    "timestamp": [start + pd.Timedelta(days=1, milliseconds=50)],
                    "bid_price_0": [999.0],
                    "ask_price_0": [999.2],
                }
            )
            (root / symbol).mkdir(parents=True, exist_ok=True)
            pd.concat([frame, spill]).to_parquet(root / symbol / f"{day}.parquet", index=False)


@pytest.fixture(scope="module")
def universe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("universe")
    write_universe(root, DAYS)
    return root


def vals(tape: signals.SignalTape) -> np.ndarray:
    """A tape's values as an array (the attribute is not a pandas one)."""
    return np.asarray(tape.values)


def panel_of(root: Path, days: list[date]) -> signals.UniversePanel:
    return signals.load_universe_panel(SYMBOLS, days, access=prereg.development_access(), root=root)


def test_the_panel_is_labelled_by_bin_ends_and_reads_only_its_days(universe: Path) -> None:
    panel = panel_of(universe, DAYS)
    labels = pd.DatetimeIndex(panel.log_mid.index)
    assert pd.Timestamp(DAYS[-1], tz="UTC") < labels[-1]
    assert labels[-1] <= pd.Timestamp(DAYS[-1], tz="UTC") + pd.Timedelta(days=1)
    assert (labels.second % 5 == 0).all() and (labels.microsecond == 0).all()
    # The first label holds the last update of its five seconds, never a later one.
    first = pd.read_parquet(universe / SYMBOLS[0] / f"{DAYS[0]}.parquet")
    inside = first[first["timestamp"] < labels[0]]
    expected = np.log((inside["bid_price_0"].iloc[-1] + inside["ask_price_0"].iloc[-1]) / 2)
    assert panel.log_mid[SYMBOLS[0]].iloc[0] == pytest.approx(expected)
    # The row spilled past midnight was dropped: no price near 999 anywhere.
    assert float(np.exp(panel.log_mid.to_numpy()).max()) < 10.0


def test_the_tape_is_causal_under_truncation_and_rewritten_futures(
    universe: Path, tmp_path: Path
) -> None:
    full = signals.reversion_tape(panel_of(universe, DAYS), "AAAUSDT", CONFIG)
    short = signals.reversion_tape(panel_of(universe, DAYS[:2]), "AAAUSDT", CONFIG)
    assert len(short.ts) > 1000
    head = full.ts <= short.ts[-1]
    np.testing.assert_array_equal(full.ts[head], short.ts)
    np.testing.assert_array_equal(vals(full)[head], short.values)

    noon = 12 * 3600
    write_universe(tmp_path, DAYS, shock_after=noon)
    shocked = signals.reversion_tape(panel_of(tmp_path, DAYS), "AAAUSDT", CONFIG)
    cut = day_start_ns(DAYS[-1]) + noon * NS_PER_S
    before = full.ts <= cut
    np.testing.assert_array_equal(shocked.ts[before], full.ts[before])
    np.testing.assert_array_equal(vals(shocked)[before], vals(full)[before])
    assert not np.allclose(vals(shocked)[~before], vals(full)[~before])


def test_the_tape_excludes_its_own_instrument(universe: Path) -> None:
    panel = panel_of(universe, DAYS)
    tape = signals.reversion_tape(panel, "AAAUSDT", CONFIG)
    moved = signals.UniversePanel(
        panel.log_mid.assign(AAAUSDT=panel.log_mid["AAAUSDT"] * 3.0), panel.spread_bp
    )
    np.testing.assert_array_equal(
        signals.reversion_tape(moved, "AAAUSDT", CONFIG).values, tape.values
    )
    others = panel.log_mid.drop(columns="AAAUSDT").to_numpy()
    level = others.mean(axis=1)
    expected = (level[120:] - level[:-120]) * 1e4
    np.testing.assert_allclose(tape.values, expected)


def test_a_decision_reads_a_label_only_after_it(universe: Path) -> None:
    tape = signals.reversion_tape(panel_of(universe, DAYS), "AAAUSDT", CONFIG)
    at = tape.ts[500]
    read = tape.strictly_before(np.array([at, at + 1]))
    assert read[0] == vals(tape)[499]
    assert read[1] == vals(tape)[500]


def test_theta_and_beta_are_the_registered_rules(universe: Path) -> None:
    panel = panel_of(universe, DAYS)
    tape = signals.reversion_tape(panel, "AAAUSDT", CONFIG)
    theta = signals.fit_theta(tape, CONFIG)
    assert theta == threshold_for_rate(tape.values, 86_400 // 5, CONFIG)
    beta = signals.fit_beta(panel, "AAAUSDT", tape, theta)
    stamps = pd.DatetimeIndex(panel.log_mid.index).as_unit("ns").asi8
    position = np.searchsorted(stamps, tape.ts)
    log_mid = panel.log_mid["AAAUSDT"].to_numpy()
    forward = np.full(len(log_mid), np.nan)
    forward[:-120] = (np.exp(log_mid[120:] - log_mid[:-120]) - 1) * 1e4
    y = forward[position]
    mask = (np.abs(tape.values) >= theta) & np.isfinite(y)
    assert beta == pytest.approx(
        np.dot(vals(tape)[mask], y[mask]) / np.dot(vals(tape)[mask], vals(tape)[mask])
    )


def test_triggers_are_thinned_faded_and_causal(universe: Path) -> None:
    tape = signals.reversion_tape(panel_of(universe, DAYS), "AAAUSDT", CONFIG)
    theta = float(np.quantile(np.abs(vals(tape)), 0.5))
    triggers = signals.reversion_triggers(tape, theta, CONFIG)
    assert len(triggers) > 50
    rows = np.searchsorted(tape.ts, triggers.ts)
    assert np.all(np.diff(rows) >= CONFIG.hold + CONFIG.cooldown)
    np.testing.assert_array_equal(triggers.direction, decide(vals(tape)[rows], theta))
    assert set(np.unique(triggers.direction)) <= {-1, 1}
    cut = len(tape.ts) * 2 // 3
    early = signals.reversion_triggers(
        signals.SignalTape(tape.name, tape.ts[:cut], vals(tape)[:cut]), theta, CONFIG
    )
    np.testing.assert_array_equal(early.ts, triggers.ts[triggers.ts <= tape.ts[cut - 1]])
    flipped = triggers.flipped()
    np.testing.assert_array_equal(flipped.direction, -triggers.direction)
    tapes = triggers.tapes()
    assert vals(tapes["x1_trigger_s"])[0] == pytest.approx(triggers.ts[0] / NS_PER_S)


def test_eligible_triggers_avoid_the_warm_up_and_the_last_25_minutes() -> None:
    start = day_start_ns(FIRST)
    seconds = np.array([0, 599, 600, 3600, 84_899, 84_900, 86_399])
    triggers = signals.Triggers(start + seconds * NS_PER_S, np.ones(7, dtype=np.int64), np.zeros(7))
    kept = (triggers.eligible().ts - start) // NS_PER_S
    assert kept.tolist() == [600, 3600, 84_899]


def test_the_taker_twin_is_section_27s_rule_on_the_same_triggers(universe: Path) -> None:
    panel = panel_of(universe, DAYS)
    tape = signals.reversion_tape(panel, "AAAUSDT", CONFIG)
    triggers = signals.reversion_triggers(tape, signals.fit_theta(tape, CONFIG), CONFIG)
    twin = signals.taker_twin(triggers, panel, "AAAUSDT")
    assert len(twin) == len(triggers) or len(twin) == len(triggers) - 1  # the last may run out
    stamps = pd.DatetimeIndex(panel.log_mid.index).as_unit("ns").asi8
    row = int(np.searchsorted(stamps, twin["ts"].iloc[0]))
    log_mid = panel.log_mid["AAAUSDT"].to_numpy()
    move = (np.exp(log_mid[row + 120] - log_mid[row]) - 1) * 1e4
    first = twin.iloc[0]
    assert first["gross_bp"] == pytest.approx(first["direction"] * move)
    assert first["cost_bp"] == pytest.approx(2 * 5.5 + first["entry_spread_bp"] + 2 * 0.5)
    assert first["net_bp"] == pytest.approx(first["gross_bp"] - first["cost_bp"])


def test_the_lean_trigger_tape_marks_every_label_at_or_above_theta(universe: Path) -> None:
    tape = signals.reversion_tape(panel_of(universe, DAYS), "AAAUSDT", CONFIG)
    theta = float(np.quantile(np.abs(tape.values), 0.99))
    marks = signals.lean_trigger_tape(tape, theta)
    np.testing.assert_array_equal(marks.ts, tape.ts[np.abs(tape.values) >= theta])
    np.testing.assert_allclose(marks.values, marks.ts / NS_PER_S)


def test_a_held_out_day_is_refused_before_any_file_is_opened(
    universe: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def never(*args: object, **kwargs: object) -> None:
        raise AssertionError("a file was opened")

    monkeypatch.setattr(pd, "read_parquet", never)
    for day in (date(2024, 2, 26), date(2024, 3, 9), date(2024, 3, 12), date(2024, 1, 31)):
        with pytest.raises(prereg.HeldOutLocked):
            signals.load_universe_panel(
                SYMBOLS, [*DAYS, day], access=prereg.development_access(), root=universe
            )
    with pytest.raises(TypeError):
        signals.load_universe_panel(SYMBOLS, DAYS, access=None, root=universe)


def test_within_day_keeps_only_the_days_rows() -> None:
    start = pd.Timestamp(FIRST, tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": [
                start - pd.Timedelta(milliseconds=1),
                start,
                start + pd.Timedelta(hours=23, minutes=59, seconds=59),
                start + pd.Timedelta(days=1),
            ],
            "x": [0, 1, 2, 3],
        }
    )
    assert within_day(frame, FIRST)["x"].tolist() == [1, 2]


def test_a_tape_slice_keeps_the_value_standing_at_its_start() -> None:
    tape = signals.SignalTape("x", np.array([10, 20, 30, 40]), np.array([1.0, 2.0, 3.0, 4.0]))
    window = tape.between(25, 35)
    np.testing.assert_array_equal(window.ts, [20, 30])
    np.testing.assert_array_equal(window.strictly_before(np.array([26, 31])), [2.0, 3.0])
    assert tape.fingerprint() != window.fingerprint()
