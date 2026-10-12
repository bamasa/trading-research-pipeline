"""K-acct and K-cap on the passive slots: an accounting fault or a position past the cap voids the slot."""

from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from trading_research.market_making.accounting import IdentityError
from trading_research.pipeline.short_horizon import heldout as ho
from trading_research.pipeline.short_horizon import heldout_passive as hp
from trading_research.pipeline.short_horizon.passive import Decisions, SimCell

KEY = "BTCUSDT|C|gbm|micro|open|hold_s=50|trades_per_day=240|label_fee_bp=0|regime=none"
DAYS = [date(2025, 7, 8) + timedelta(days=i) for i in range(3)]
CLIP = 0.01


def _fills(*positions: float) -> pd.DataFrame:
    return pd.DataFrame({"position_after": list(positions)})


def test_the_cap_is_one_clip_for_the_quoter_and_two_for_s1() -> None:
    assert not hp.position_cap(_fills(0.01, 0.0, -0.01), CLIP, "entry")["cap_exceeded"]
    over = hp.position_cap(_fills(0.01, 0.015), CLIP, "entry")
    assert over["cap_exceeded"] and over["max_abs_position"] == pytest.approx(0.015)
    assert not hp.position_cap(_fills(-0.02), CLIP, "maker")["cap_exceeded"]
    assert hp.position_cap(_fills(-0.021), CLIP, "maker")["cap_exceeded"]
    assert hp.position_cap(_fills(), CLIP, "entry") == {"max_abs_position": 0.0, "cap_exceeded": False,
                                                        "accounting_failed": ""}  # fmt: skip
    # the simulator's own count is read when a run carries it (day/max_abs_position)
    counted = hp.position_cap(_fills(0.005), CLIP, "entry", counted=0.012)
    assert counted["cap_exceeded"] and counted["max_abs_position"] == pytest.approx(0.012)
    assert not hp.position_cap(_fills(0.015), CLIP, "entry", counted=0.009)["cap_exceeded"]


def test_an_accounting_fault_in_one_run_is_recorded_and_the_others_still_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trading_research.market_making import simulator

    def simulate(events: Any, quoter: Any, config: Any, tapes: Any) -> Any:
        if tapes.get("flipped"):
            raise IdentityError("decomposition net 1.0 differs from cash 2.0")
        return SimpleNamespace(fills=_fills(0.01, 0.0), counters={"max_abs_position": 0.01})

    monkeypatch.setattr(simulator, "simulate_day", simulate)
    monkeypatch.setattr(
        hp,
        "quoter_and_tapes",
        lambda _cell, decided, *_: (None, None, {"flipped": decided.side[0] < 0}),
    )
    monkeypatch.setattr(hp, "day_values", lambda *_: {"excluded": False, "net_per_100": 1.0})
    monkeypatch.setattr(hp, "fill_values", lambda _: {})
    events = SimpleNamespace(bid_px=np.ones((2, 1)), ask_px=np.ones((2, 1)), spec=SimpleNamespace(tick=0.1),
                             day=DAYS[0])  # fmt: skip
    cell = SimCell("S-P", KEY, "entry", 50.0)
    own = Decisions(np.array([1], np.int64), np.array([1]), np.array([1.0]))
    runs = (
        hp.Run("S-P", "true", cell, own),
        hp.Run("S-P", "flipped", cell, Decisions(own.stamps_ns, -own.side, own.strength_bp)),
    )
    rows = hp.day_rows(events, runs, np.zeros(0, np.int64), 9, 1.0, CLIP)
    assert [r["variant"] for r in rows] == ["true", "flipped"]
    assert rows[0]["accounting_failed"] == "" and rows[0]["max_abs_position"] == pytest.approx(0.01)
    assert rows[1]["accounting_failed"].startswith("IdentityError: decomposition net")


def _rows(**true: Any) -> pd.DataFrame:
    out = []
    for variant in ("true", "flipped"):
        for day in DAYS:
            out.append({"run_slot": "S-P", "variant": variant, "day": day.isoformat(), "excluded": False,
                        "net_per_100": 1.0, "trades": 2, "net_usdt": 1.0, "passive_notional": 10.0,
                        "accounting_failed": "", "cap_exceeded": False} | (true if variant == "true" else {}))  # fmt: skip
    return pd.DataFrame(out)


def test_the_passive_slot_is_void_under_k_acct_or_k_cap() -> None:
    p = ho.PassiveSlot("S-P", SimCell("S-P", KEY, "entry", 50.0), [], ic=[0.1] * 3)
    assert ho.passive_series(p, _rows(), DAYS, 0, 0, False).void == ()
    capped = _rows(cap_exceeded=True)
    assert ho.passive_series(p, capped, DAYS, 0, 0, False).void == ("K-cap",)
    failed = _rows()
    failed.loc[4, "accounting_failed"] = "RuntimeError: the filled attempts' cash"
    failed.loc[4, "net_per_100"] = np.nan
    assert ho.passive_series(p, failed, DAYS, 0, 0, False).void == ("K-acct",)
    assert ho.failures(failed) == [{"day": DAYS[1].isoformat(), "run_slot": "S-P", "variant": "flipped",
                                    "accounting_failed": "RuntimeError: the filled attempts' cash"}]  # fmt: skip


def test_simulator_workers_keep_six_peaks_within_24_gb() -> None:
    assert hp.simulator_workers(6, int(2.9e9)) == 6
    assert hp.simulator_workers(6, int(5e9)) == 4
    assert hp.simulator_workers(6, int(30e9)) == 1
    assert hp.simulator_workers(3, int(1e9)) == 3 and hp.simulator_workers(8, 0) == 6


def test_simulate_batched_sizes_each_batch_by_the_largest_peak_so_far(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The batching stage C's simulate_all and the reads share: one job alone first."""
    import concurrent.futures

    batches: list[int] = []

    class Inline:
        def __init__(self, max_workers: int, **_: Any) -> None:
            batches.append(max_workers)

        def __enter__(self) -> Inline:
            return self

        def __exit__(self, *_: Any) -> None:
            return None

        def map(self, fn: Any, jobs: list[Any]) -> list[Any]:
            return [fn(job) for job in jobs]

    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", Inline)
    found = hp.simulate_batched(
        list(range(5)), lambda j: (j * 10, int(7e9) if j == 0 else int(5e9)), 8
    )
    assert found == ([0, 10, 20, 30, 40], int(7e9), 3) and batches == [1, 3, 1]
    assert hp.simulate_batched([], lambda j: (j, 0), 6) == ([], 0, 1)


def test_the_first_day_measures_the_peak_and_sizes_every_later_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import concurrent.futures

    batches: list[int] = []

    class Inline:
        def __init__(self, max_workers: int, **_: Any) -> None:
            batches.append(max_workers)

        def __enter__(self) -> Inline:
            return self

        def __exit__(self, *_: Any) -> None:
            return None

        def map(self, fn: Any, jobs: list[Any]) -> list[Any]:
            return [fn(job) for job in jobs]

    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", Inline)
    monkeypatch.setattr(hp, "run_measured", lambda job: (job, int(7e9) if job == 3 else int(5e9)))
    block = SimpleNamespace(study=SimpleNamespace(workers=8), notes=[])
    assert ho.simulate_jobs(block, list(range(12))) == list(range(12))  # type: ignore[arg-type]
    assert batches == [1, 4, 3, 3, 1] and block.notes == [
        "simulator: peak 7000000000 bytes for one day, 3 workers"
    ]
