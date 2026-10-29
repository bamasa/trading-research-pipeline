"""The measurements a block read reports beside the statuses, from per-day values."""

from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from trading_research.pipeline.regime import random_flag_sets
from trading_research.pipeline.short_horizon import heldout_measures as hm
from trading_research.pipeline.short_horizon import heldout_passive as hp
from trading_research.pipeline.short_horizon import statuses as st
from trading_research.pipeline.short_horizon.passive import Decisions, SimCell

DAYS = [date(2025, 7, 8) + timedelta(days=i) for i in range(6)]
KEY = "BTCUSDT|C|gbm|micro|open|hold_s=50|trades_per_day=240|label_fee_bp=0|regime=pull_4h"


def slot(
    kind: str = "taker", days: list[date] = DAYS, window_h: int = 4, **extra: np.ndarray
) -> st.SlotDays:
    n = len(days)
    value = np.arange(n, dtype=np.float64) + 1.0
    return st.SlotDays(
        "S-C", kind, list(days), value, np.full(n, 10), np.full(n, 1.0), np.full(n, 100.0),
        np.full(n, 0.1), np.zeros((2, n)), -value, -value, -value, -value, np.ones((3, n)),
        value, window_h=window_h, extra=extra,
    )  # fmt: skip


def test_merge_puts_the_gap_days_back_in_day_order_with_nan_where_one_side_lacks_an_array() -> None:
    usable = slot(days=[DAYS[0], DAYS[2], DAYS[5]], twin=np.array([1.0, 2.0, 3.0]))
    gaps = slot(days=[DAYS[1], DAYS[4]], exit=np.array([7.0, 8.0]))
    gaps.void = ("K-cap",)
    merged = st.merge(usable, gaps)
    assert merged.days == [DAYS[0], DAYS[1], DAYS[2], DAYS[4], DAYS[5]]
    assert merged.value.tolist() == [1.0, 1.0, 2.0, 2.0, 3.0]
    assert merged.trades.tolist() == [10] * 5 and merged.random.shape == (2, 5)
    assert np.isnan(merged.extra["twin"][[1, 3]]).all() and merged.extra["twin"][
        [0, 2, 4]
    ].tolist() == [1.0, 2.0, 3.0]
    assert np.isnan(merged.extra["exit"][[0, 2, 4]]).all() and merged.extra["exit"][
        [1, 3]
    ].tolist() == [7.0, 8.0]
    assert merged.void == ("K-cap",) and merged.window_h == 4
    with pytest.raises(ValueError, match="overlap"):
        st.merge(usable, usable)


def test_attribution_is_the_guarded_mean_against_the_twin_and_the_random_flag_sets() -> None:
    n_flags, sets = 3, 4
    drawn = DAYS
    rows = np.array([[0.5] * 6, [1.5] * 6, [2.5] * 6, [3.5] * 6])
    s = slot(twin=np.full(6, 2.0), flagsets=rows, guarded=np.full(6, 5.0))
    flags = random_flag_sets(drawn, n_flags, sets=1, first_seed=7)[0]
    found = hm.attribution(s, flags, drawn, n_flags, "pull_4h")
    assert found is not None
    assert found["difference"] == pytest.approx(3.0) and found["window_h"] == 4
    assert found["draws"] == pytest.approx([-1.5, -0.5, 0.5, 1.5])
    assert found["random_p95"] == pytest.approx(np.quantile([-1.5, -0.5, 0.5, 1.5], 0.95))
    assert found["exceeds_random"] and found["flags"] == n_flags and found["sets"] == sets
    assert found["slot_mean"] == pytest.approx(3.5) and found["guarded_mean"] == pytest.approx(5.0)
    assert (
        hm.attribution(slot(window_h=0, twin=rows[0], flagsets=rows), flags, drawn, n_flags, "none")
        is None
    )
    assert hm.attribution(slot(), flags, drawn, n_flags, "pull_4h") is None
    sentence = hm.regime_sentence("S-C", found)
    assert sentence.startswith("S-C on H: its 4 h guard after detector alarms did add")


def test_measured_rows_report_each_regime_and_exit_series_and_the_control_ic() -> None:
    s = slot(**{"regime/none": np.full(6, 2.0), "regime/none/trades": np.full(6, 3.0),
                "exit/passive": np.full(6, -1.0), "control_ic_label": np.array([0.1, np.nan, 0.3, 0.1, 0.1, 0.1]),
                "flagsets": np.zeros((2, 6))})  # fmt: skip
    found = hm.measured_rows(s)
    assert set(found) == {"regime/none", "exit/passive", "control_ic_label"}
    assert found["regime/none"]["mean"] == 2.0 and found["regime/none"]["trades"] == 18
    assert found["exit/passive"]["mean"] == -1.0 and found["exit/passive"]["trades"] == 60
    assert found["control_ic_label"] == {"days": 5, "mean": pytest.approx(0.14)}


def _fill_rows() -> pd.DataFrame:
    paths = json.dumps(
        {"queue": {"fills": 2.0, "volume": 1.0, "markout_5s": 3.0, "markout_5s_weight": 1.0}}
    )
    rows = []
    for variant, posted, filled in (
        ("true", 10.0, 4.0),
        ("fill/touch", 10.0, 9.0),
        ("fill/trade_through", 10.0, 1.0),
    ):
        for day, excluded in ((DAYS[0], False), (DAYS[1], False), (DAYS[2], True)):
            rows.append({"run_slot": "S-P", "variant": variant, "day": day.isoformat(), "excluded": excluded,
                         "orders_posted": posted, "orders_filled": filled, "volume_posted": posted,
                         "volume_filled": filled / 2, "gross_usdt": 1.0, "fees_usdt": -0.5, "net_usdt": 0.5,
                         "fill_paths": paths})  # fmt: skip
    rows.append(rows[0] | {"run_slot": "S25-Z", "orders_filled": 0.0})
    return pd.DataFrame(rows)


def test_fill_rules_on_a_stream_read_the_three_rules_and_the_registered_prediction() -> None:
    found = hm.fill_rules(_fill_rows(), "S-P", DAYS[:2])
    by_rule = {r["rule"]: r for r in found["summary"]}
    assert list(by_rule) == ["touch", "trade_through", "queue"]
    assert by_rule["touch"]["fill_ratio"] == 0.9 and by_rule["queue"]["fill_ratio"] == 0.4
    assert by_rule["trade_through"]["bias"] == "pessimistic" and by_rule["queue"]["days"] == 2
    assert (
        by_rule["queue"]["volume_ratio"] == pytest.approx(0.2)
        and by_rule["queue"]["net_usdt"] == 1.0
    )
    assert by_rule["queue"]["markouts"] == [
        {"path": "queue", "fills": 4.0, "volume": 2.0, "markout_5s": 3.0}
    ]
    assert found["prediction"] == {"touch_fill_ratio": 0.9, "queue_fill_ratio": 0.4, "holds": True}
    assert hm.fill_rules(_fill_rows(), "S-P", DAYS[2:3])["summary"][0]["days"] == 0


def test_measured_runs_of_an_entry_slot_carry_the_exit_fill_rules_and_flag_sets() -> None:
    cell = SimCell("S-P", KEY, "entry", 50.0, policy="pull_4h")
    own = Decisions(np.array([1, 2]), np.array([1, -1]), np.zeros(2))
    twin = SimCell("S-P", KEY.replace("pull_4h", "none"), "entry", 50.0)
    sets = [(np.array([5]), own), (np.array([9]), own)]
    runs = hp.measured_runs("S-P", cell, own, [("regime/none", twin, own)], sets, fill=True)
    assert [r.variant for r in runs] == ["regime/none", "exit/passive", "fill/touch", "fill/trade_through",
                                         "flagset/0", "flagset/1"]  # fmt: skip
    assert runs[1].cell.passive_exit and runs[1].cell.label.endswith("|exit=passive")
    assert (
        runs[2].changes["fill_rule"].value == "touch"
        and runs[3].changes["fill_rule"].value == "trade_through"
    )
    assert (
        runs[4].flags_ns is not None
        and runs[4].flags_ns.tolist() == [5]
        and runs[0].flags_ns is None
    )
    maker = SimCell("S-M", KEY, "maker", 50.0, lean=1.0, policy="pull_4h")
    assert [r.variant for r in hp.measured_runs("S-M", maker, own, [], sets, fill=True)] == [
        "flagset/0",
        "flagset/1",
    ]
    assert [r.variant for r in hp.measured_runs("S-P", cell, own, [], [], fill=False)] == [
        "exit/passive"
    ]


class _Result:
    def __init__(self) -> None:
        self.orders = pd.DataFrame({"size": [1.0, 1.0, 2.0], "filled": [1.0, 0.0, 0.5]})
        self.fills = pd.DataFrame({"maker": [True, True, False], "path": ["queue", "through", "taker"],
                                   "size": [1.0, 0.5, 1.0], "markout_1s": [2.0, np.nan, 9.0],
                                   "markout_5s": [4.0, 6.0, 9.0]})  # fmt: skip


def test_fill_values_count_orders_and_volume_and_sum_the_markouts_by_path() -> None:
    found = hp.fill_values(_Result())
    assert found["orders_posted"] == 3.0 and found["orders_filled"] == 2.0
    assert found["volume_posted"] == 4.0 and found["volume_filled"] == 1.5
    paths = json.loads(found["fill_paths"])
    assert paths["queue"] == {"fills": 1.0, "volume": 1.0, "markout_1s": 2.0, "markout_1s_weight": 1.0,
                              "markout_5s": 4.0, "markout_5s_weight": 1.0}  # fmt: skip
    assert paths["through"]["markout_1s_weight"] == 0.0 and paths["through"]["markout_5s"] == 3.0
    assert hm._fill_paths([found["fill_paths"]] * 2) == [
        {"path": "queue", "fills": 2.0, "volume": 2.0, "markout_1s": 2.0, "markout_5s": 4.0},
        {"path": "through", "fills": 2.0, "volume": 1.0, "markout_1s": pytest.approx(np.nan, nan_ok=True),
         "markout_5s": 6.0}]  # fmt: skip
