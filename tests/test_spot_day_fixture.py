"""A synthetic reader-test day of both venues on disk, loaded as the study loads a day.

Shared by the spot entry point's tests: the synthetic venue's archives fetched
under the reader-test access into a temporary data root (no socket), then loaded
through :func:`trading_research.spot.days.load_day`.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np

from trading_research.data import fetch_log as fl
from trading_research.spot import access as sa
from trading_research.spot import days as dy
from trading_research.spot import fetch_stage as fs
from trading_research.spot import synthetic as syn
from trading_research.spot.costs import FeeTable, load_fee_table

ROOT = Path(__file__).resolve().parents[1]
DAYS = sa.READER_TEST_DAYS
TABLE: FeeTable = load_fee_table(ROOT / "configs" / "spot_2026.yaml")


def fetch_days(
    base: Path, *, step_ms: int = 2_000, gap_days: frozenset[date] = frozenset()
) -> fl.FetchLog:
    """The reader-test days of every plane under ``base/spot``."""
    market = syn.SpotMarket(frozenset(DAYS), step_ms=step_ms, rpi_every=7, gap_days=gap_days)
    with syn.no_network():
        return fs.run_stage(0, fs.every_plane(DAYS), fs.SpotRoots.under(base / "spot"),
                            fl.FetchLog(base / "log.json"), access=sa.reader_test_access("dry run"),
                            disk_record=base / "disk.json", opener=syn.SpotVenue(market),
                            measure=lambda _: 1_000.0)  # fmt: skip


def load(base: Path, log: fl.FetchLog, day: date, max_gaps: int | None = None) -> dy.DayInputs:
    return dy.load_day(day, base / "spot", access=sa.reader_test_access("dry run"), logs=[log],
                       table=TABLE, max_sequence_gaps=max_gaps, depth=10)  # fmt: skip


class Stub:
    """A model whose probabilities follow one input's sign: long above zero, short below."""

    def __init__(self, column: str, columns: list[str]) -> None:
        self.column = column
        self.columns_ = columns

    def predict_proba(self, x: object) -> np.ndarray:
        values = np.asarray(x[self.column], dtype=float)  # type: ignore[index]
        p_long = np.where(values > 0, 0.6, 0.2)
        p_short = np.where(values < 0, 0.6, 0.2)
        return np.column_stack([p_short, 1.0 - p_long - p_short, p_long])


def probas(n: int, seed: int = 0, share: float = 0.05) -> np.ndarray:
    """Random class probabilities, a ``share`` of rows deciding long or short."""
    rng = np.random.default_rng(seed)
    out = np.tile([0.3, 0.4, 0.3], (n, 1)).astype(float)
    picked = rng.random(n) < share
    side = rng.choice([0, 2], size=n)
    out[picked] = [0.1, 0.2, 0.1]
    out[picked, side[picked]] = 0.7
    return out


def study_with_days(repo: Path, *, gap_days: frozenset[date] = frozenset()) -> object:
    """A study on a temporary repository holding the registration as registered, with
    the reader-test days fetched where stage 0 keeps them."""
    import shutil

    from trading_research.market_making.prereg import unfrozen_text
    from trading_research.spot import registration as sr
    from trading_research.spot.context import SpotStudy
    from trading_research.spot.plan import DRY_RUN

    for relative in (sr.CONFIG_PATH, sr.DOCUMENT_PATH):
        (repo / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, repo / relative)
    config = repo / sr.CONFIG_PATH
    config.write_text(unfrozen_text(config.read_text(encoding="utf-8")), encoding="utf-8")
    study = SpotStudy(repo, DRY_RUN, disk=lambda _: 1_000.0)
    market = syn.SpotMarket(frozenset(DAYS), step_ms=2_000, rpi_every=7, gap_days=gap_days)
    with syn.no_network():
        fs.run_stage(0, fs.every_plane(DAYS), study.roots, study.fetch_log(0),
                     access=sa.reader_test_access("dry run"), disk_record=study.disk_record,
                     opener=syn.SpotVenue(market), measure=lambda _: 1_000.0)  # fmt: skip
    return study


def stub_models() -> dict[str, object]:
    """Every row's fits as stubs on one of its inputs, thresholds at 0.5 on every cell."""
    from trading_research.spot.plan import ROWS
    from trading_research.spot.runs import RowModels
    from trading_research.strategies.hold_first import HoldFirstRule

    column = {"binance": "flow_imbalance_6", "bybit_prints": "flow_imbalance_6", "book": "ofi_1"}
    out = {}
    for name, row in ROWS.items():
        stub = Stub(column[row.store], [column[row.store]])
        out[name] = RowModels(name, dict.fromkeys(row.fits, stub),
                              dict.fromkeys(row.cells, HoldFirstRule(0.5, 0.5)),
                              dy.layout(row.store, 1.0))  # fmt: skip
    return out
