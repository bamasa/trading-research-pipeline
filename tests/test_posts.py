"""Posts quote their numbers.json, and numbers.json agrees with the series it was cut from."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

MINUS = chr(0x2212)  # the typographic minus the page uses
POST = Path(__file__).resolve().parent.parent / "posts" / "btc_ath_2024"


@pytest.fixture(scope="module")
def numbers() -> dict:
    return json.loads((POST / "numbers.json").read_text(encoding="utf-8"))


def test_the_numbers_are_the_series_they_summarise(numbers: dict) -> None:
    data = json.loads((POST / "data.json").read_text(encoding="utf-8"))
    equity = dict(data["equity_pct"])
    peak_at = max(equity, key=equity.get)
    assert peak_at == numbers["peak"]["at"]
    assert round(equity[peak_at], 1) == numbers["peak"]["equity_pct"]
    assert round(data["equity_pct"][-1][1], 1) == numbers["end"]["equity_pct"]
    after_peak = [(t, v) for t, v in data["equity_pct"] if t > peak_at and v < 0]
    assert after_peak[0][0] == numbers["below_zero"]["at"]
    btc = dict(data["btc"])
    assert round(max(btc.values())) == numbers["btc_high"]["hourly_mid"]


def test_the_february_half_is_the_benchmarks_reversion_card(numbers: dict) -> None:
    board = json.loads(
        (POST.parent.parent / "experiments/results/bench_board.json").read_text(encoding="utf-8")
    )
    card = next(r for r in board["rows"] if r["name"] == "reversion")["bybit-2024-02"]
    assert numbers["before"]["trades"] == card["trades"]
    assert numbers["before"]["net_bp_per_trade"] == round(card["net_bp"], 2)


def test_the_page_quotes_its_numbers(numbers: dict) -> None:
    text = (POST / "README.md").read_text(encoding="utf-8")
    hours = (
        datetime.fromisoformat(numbers["below_zero"]["at"])
        - datetime.fromisoformat(numbers["peak"]["at"])
    ).total_seconds() / 3600
    quoted = [
        f"{numbers['before']['trades']:,} trades",
        f"{numbers['after']['trades']:,} trades",
        f"+{numbers['before']['gross_bp_per_trade']:.1f} bp a trade gross",
        f"+{numbers['after']['gross_bp_per_trade']:.1f} bp a trade gross",
        f"**+{numbers['before']['net_bp_per_trade']:.1f} bp**",
        f"**{MINUS}{-numbers['after']['net_bp_per_trade']:.1f} bp**",
        f"**+{numbers['peak']['equity_pct']:.1f}%**",
        f"**{MINUS}{-numbers['end']['equity_pct']:.1f}%**",
        f"**${numbers['btc_high']['hourly_mid']:,}**",
        f"{hours:.0f} hours after the peak",
    ]
    quoted += [f"{v:.3f}".replace("-", MINUS) for v in numbers["autocorrelation_mean"].values()]
    missing = [q for q in quoted if q not in text]
    assert not missing, missing
    assert 48 <= hours < 72, "the post says 'within two days'"
    assert "two days" in text


def test_the_animation_draws_from_its_files() -> None:
    page = (POST / "animation.html").read_text(encoding="utf-8")
    assert "<script>/*DATA*/</script>" in page
    for read in (
        "N.end.equity_pct",
        "MONI.stop_end_pct",
        "MONI.shadow_end_pct",
        "STOP.hours_after_peak",
    ):
        assert read in page, read


def test_the_kill_switch_is_what_the_page_says() -> None:
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("post_monitor", POST / "monitor.py")
    assert spec and spec.loader
    monitor = module_from_spec(spec)
    spec.loader.exec_module(monitor)
    assert (monitor.K, monitor.H, monitor.WARM_UP) == (0.5, 5.0, 72)
    recorded = json.loads((POST / "monitor.json").read_text(encoding="utf-8"))
    returns = monitor.hourly_returns()
    for name, restart in (("stop", False), ("shadow", True)):
        path, events = monitor.run(returns, restart)
        assert round(float(path[-1]), 1) == recorded[f"{name}_end_pct"]
        assert [e["event"] for e in events] == [e["event"] for e in recorded[f"{name}_events"]]
    first = recorded["stop_events"][0]
    text = (POST / "README.md").read_text(encoding="utf-8")
    assert f"{first['hours_after_peak']} hours after the peak" in text
    assert f"**+{recorded['stop_end_pct']:.1f}%**" in text
    assert f"**+{recorded['shadow_end_pct']:.1f}%**" in text
    assert len(recorded["shadow_events"]) == 3 and "restart 14 April, stop 14 April" in text


def test_the_break_detector_finds_what_the_page_says() -> None:
    pytest.importorskip("structural_break")
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("post_breaks", POST / "breaks.py")
    assert spec and spec.loader
    breaks = module_from_spec(spec)
    spec.loader.exec_module(breaks)
    recorded = json.loads((POST / "breaks.json").read_text(encoding="utf-8"))
    assert breaks.alarms() == recorded["alarms"]
    text = (POST / "README.md").read_text(encoding="utf-8")
    assert "| 15 March 09:10 | dependence rose |" in text
    assert [a["at"] for a in recorded["alarms"]][3] == "2024-03-15T09:10"
