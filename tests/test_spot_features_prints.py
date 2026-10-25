"""The nine print members and twelve touch-proxy members: declarations, values, leakage."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from test_spot_synthetic_prints import DAY, make_prints, no_network
from trading_research.features import REGISTRY as GLOBAL_REGISTRY
from trading_research.features.study_set import winsorise_trailing
from trading_research.features.timed import TIMED_REGISTRY, build_timed, stamps_ns
from trading_research.spot.features_prints import (
    PRINT_MEMBERS,
    TOUCH_PROXY_MEMBERS,
    day_frame,
    print_registry,
)
from trading_research.spot.grid import print_grid
from trading_research.validation.leakage_timed import assert_timed_honest


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    no_network(monkeypatch)


@pytest.fixture(scope="module")
def prints() -> pd.DataFrame:
    return make_prints(1_500, seed=5, mean_gap_s=0.08, quiet_at=700)


@pytest.fixture(scope="module")
def rows(prints: pd.DataFrame) -> pd.DataFrame:
    return print_grid(prints)


def test_the_registry_is_the_studys_own_with_the_registered_declarations() -> None:
    registry = print_registry()
    assert registry.names() == sorted([*PRINT_MEMBERS, *TOUCH_PROXY_MEMBERS])
    assert len(PRINT_MEMBERS) == 9 and len(TOUCH_PROXY_MEMBERS) == 12
    assert registry.longest_lookback_minutes(PRINT_MEMBERS) == 61
    assert registry.longest_lookback_minutes() == 61
    assert {registry.get(n).plane for n in registry.names()} == {"trades"}
    assert registry.get("flow_imbalance_120").lookback_rows == 120
    assert registry.get("trade_run_length_60s").lookback_s == 3660.0
    assert registry.get("realized_vol_50_bp").lookback_rows == 51
    assert registry.get("turn_count_10s").lookback_s == 3610.0
    # Nothing is added to the global registries.
    counts = (len(GLOBAL_REGISTRY.names()), len(TIMED_REGISTRY))
    print_registry()
    assert (len(GLOBAL_REGISTRY.names()), len(TIMED_REGISTRY)) == counts
    assert "flow_imbalance_6" not in GLOBAL_REGISTRY.names()


def test_vwap_deviation_divides_by_the_previous_rows_mid_proxy(rows, prints) -> None:
    built = build_timed(rows, ["vwap_deviation_bp", "flow_imbalance_6"], prints=prints,
                        registry=print_registry())  # fmt: skip
    stamps, times = stamps_ns(rows), stamps_ns(prints)
    owner = np.searchsorted(stamps, times, side="left")
    size, price = prints["size"].to_numpy(), prints["price"].to_numpy()
    vwap = np.bincount(owner, size * price, len(rows)) / np.bincount(owner, size, len(rows))
    previous = np.r_[np.nan, rows["mid_proxy"].to_numpy()[:-1]]
    raw = (vwap / previous - 1.0) * 1e4
    expected = winsorise_trailing(raw, stamps)
    np.testing.assert_allclose(built["vwap_deviation_bp"], expected, rtol=1e-9, atol=1e-9)
    signed = np.bincount(owner, size * prints["aggressor"].to_numpy(), len(rows))
    volume = np.bincount(owner, size, len(rows))
    six = (
        pd.Series(signed).rolling(6, min_periods=1).sum()
        / pd.Series(volume).rolling(6, min_periods=1).sum()
    )
    np.testing.assert_allclose(built["flow_imbalance_6"], winsorise_trailing(six.to_numpy(), stamps),
                               rtol=1e-12)  # fmt: skip


def test_touch_members_read_the_proxies(rows, prints) -> None:
    built = build_timed(rows, ["spread_bp", "mid_return_1_bp"], prints=prints,
                        registry=print_registry())  # fmt: skip
    ask, bid = rows["ask_proxy"].to_numpy(), rows["bid_proxy"].to_numpy()
    mid = (ask + bid) / 2
    stamps = stamps_ns(rows)
    spread = winsorise_trailing((ask - bid) / mid * 1e4, stamps)
    np.testing.assert_allclose(built["spread_bp"], spread, rtol=1e-12, equal_nan=True)
    ret = winsorise_trailing(np.log(mid / np.r_[np.nan, mid[:-1]]) * 1e4, stamps)
    np.testing.assert_allclose(built["mid_return_1_bp"], ret, rtol=1e-12, equal_nan=True)


def test_every_member_passes_truncation_and_streaming(rows, prints) -> None:
    positions = [*range(0, len(rows), 23), len(rows) - 1]
    for cut in (0.4, 0.7):
        assert_timed_honest(print_registry(), rows, prints, positions=positions, cut=cut)


def test_the_grid_rebuilt_from_cut_prints_changes_no_earlier_value(prints) -> None:
    """The proxies and members together, the grid rebuilt from prints cut mid-interval."""
    full = day_frame(prints, names=[*PRINT_MEMBERS, *TOUCH_PROXY_MEMBERS])
    times = stamps_ns(prints)
    bins = times // 100_000_000
    for share in (0.3, 0.55, 0.8):
        k = int(len(times) * share)
        # Print k-1 lies strictly inside its row's interval, and print k after it.
        while not (bins[k - 2] == bins[k] and times[k - 2] < times[k - 1] < times[k]):
            k += 1
        cut = day_frame(prints.iloc[:k], names=[*PRINT_MEMBERS, *TOUCH_PROXY_MEMBERS])
        assert stamps_ns(cut)[-1] < times[k - 1]
        pd.testing.assert_frame_equal(cut, full.iloc[: len(cut)], check_exact=False, rtol=1e-9)


def test_a_warm_day_equals_the_two_days_computed_together() -> None:
    before = make_prints(400, seed=6, day=DAY - timedelta(days=1), start_s=70_000.0,
                         mean_gap_s=30.0, quiet_at=None)  # fmt: skip
    day = make_prints(300, seed=7, start_s=5.0, mean_gap_s=2.0, quiet_at=None)
    names = [*PRINT_MEMBERS, *TOUCH_PROXY_MEMBERS]
    warm = day_frame(day, names=names, history=before)
    together = day_frame(pd.concat([before, day], ignore_index=True), names=names)
    together = together[together["timestamp"] >= day["timestamp"].iloc[0]].reset_index(drop=True)
    pd.testing.assert_frame_equal(warm, together)
    cold = day_frame(day, names=names)
    assert not np.allclose(cold["flow_imbalance_120"], warm["flow_imbalance_120"], equal_nan=True)
    with pytest.raises(ValueError, match="without a print"):
        day_frame(day.iloc[:0])


def test_the_cut_inside_an_interval_catches_the_forward_rule_on_the_print_grid(
    rows, prints
) -> None:
    from trading_research.features.flow import align_to_grid
    from trading_research.features.timed import TimedFeature
    from trading_research.validation.leakage_timed import check_timed_causal

    def forward(rows: pd.DataFrame, prints: pd.DataFrame | None) -> pd.Series:
        assert prints is not None
        stamps = rows["timestamp"].reset_index(drop=True)
        mid = rows["mid_proxy"].to_numpy()
        return pd.Series(align_to_grid(prints, stamps, mid)["trade_count"].to_numpy())

    ahead = TimedFeature("count_forward", "trades", 0.0, 0, forward)
    assert check_timed_causal(ahead, rows, prints).leaks
    assert not check_timed_causal(print_registry().get("trade_intensity_6"), rows, prints).leaks
