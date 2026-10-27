"""The study as the entry point runs it: its checkout, its source, its calendar and its stages.

:class:`MakerStudy` binds the checkout (``repo``), the spot study's checkout the
copies come from (``source``), the signal's binding (the registered sha256s, or
a dry run's stand-in), the calendar (the registered blocks, or a dry run's few
days of each), the worker count and the D run's number. Its stage machine is
``spot/state.StageState`` over this study's stages, under
``artifacts/spot_maker_2026/``. :func:`compute_days` computes day records in
worker processes (days in parallel), each worker loading its day inside the
access it is given; :func:`signal_days` computes only the signal on each day
(stage 1).
"""

from __future__ import annotations

import json
import multiprocessing
import shutil
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from trading_research.spot.costs import FeeTable
from trading_research.spot.state import StageState, plain, write_json
from trading_research.spot_maker import blocks as bk
from trading_research.spot_maker import copies as cp
from trading_research.spot_maker import registration as rg
from trading_research.spot_maker.signal import SignalBinding

STAGES: tuple[str, ...] = ("0", "1", "D", "amend")


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


@dataclass
class MakerStudy:
    repo: Path
    source: Path | None = None
    binding: SignalBinding = field(default_factory=SignalBinding)
    calendar: Mapping[str, tuple[date, ...]] | None = None
    workers: int = 1
    run: int = 1
    disk: Callable[[Path], float] = free_gb
    #: What serves P's archives and the published check (None: the network).
    opener: Any = None
    #: The data directory holding the calibration's book days, read in place.
    book_data: Path | None = None

    @property
    def state(self) -> StageState:
        return StageState(Path(self.repo) / rg.WORK_DIR, STAGES)

    @property
    def work(self) -> Path:
        return Path(self.repo) / rg.WORK_DIR

    def days(self, block: str) -> tuple[date, ...]:
        if self.calendar is not None:
            return tuple(self.calendar[block])
        return tuple(bk.block_days(block))

    def registration(self) -> rg.Registration:
        return rg.Registration.load(self.repo)

    def fees(self) -> dict[str, Any]:
        return dict(self.registration().config["fees"])

    def table(self) -> FeeTable:
        return FeeTable.from_config(self.fees())

    def require_source(self) -> Path:
        if self.source is None:
            raise ValueError("this stage copies files: name the spot study's checkout (--source)")
        return Path(self.source)

    def disk_check(self, stage: str) -> dict[str, Any]:
        """Free disk at a stage's start, appended to the record; refused below the margin."""
        free = float(self.disk(Path(self.repo)))
        needed = rg.DISK_BUDGET_GB + rg.DISK_MARGIN_GB
        found = {"stage": stage, "free_gb": free, "needed_gb": needed, "passed": free > needed}
        path = Path(self.repo) / "data/spot_maker_2026/disk.json"
        record = json.loads(path.read_text("utf-8")) if path.exists() else []
        write_json(path, [*record, found])
        if not found["passed"]:
            raise RuntimeError(f"stage {stage}: {free:.1f} GB free, {needed:.1f} GB needed")
        return found


def day_job(repo: Path, day: date, access: Any, binding: SignalBinding,
            fees: Mapping[str, Any]) -> dict[str, Any]:  # fmt: skip
    """One day's record, computed in a worker: the day loaded inside ``access``."""
    from trading_research.spot_maker import day as dd
    from trading_research.spot_maker.signal import day_decisions, load_signal
    from trading_research.spot_maker.touch import Proxies

    table = FeeTable.from_config(fees)
    loaded = next(iter(cp.walk(repo, [day], access, table)))
    head = {"day": day.isoformat(), "reasons": list(loaded.reasons), "stamp": access.stamp}
    if not loaded.usable:
        return {**head, "usable": False}
    signal = load_signal(repo, binding)
    decisions = day_decisions(signal, loaded.frame, loaded.history, loaded.scored_from)
    tape = loaded.tape()
    ctx = dd.DayContext(tape, Proxies.from_tape(tape, loaded.carried), decisions, table)
    return plain({**head, **dd.day_record(ctx, warm=loaded.warm)})


def signal_job(repo: Path, day: date, access: Any, binding: SignalBinding,
               fees: Mapping[str, Any]) -> dict[str, Any]:  # fmt: skip
    """The signal alone on one day: its decisions counted, nothing else computed."""
    from trading_research.spot_maker.signal import day_decisions, load_signal

    loaded = next(iter(cp.walk(repo, [day], access, FeeTable.from_config(fees))))
    head = {"day": day.isoformat(), "reasons": list(loaded.reasons), "stamp": access.stamp}
    if not loaded.usable:
        return {**head, "usable": False}
    found = day_decisions(load_signal(repo, binding), loaded.frame, loaded.history,
                          loaded.scored_from)  # fmt: skip
    return {**head, "usable": True, "warm": loaded.warm, "grid_rows": len(found.grid_ns),
            "scored_rows": found.scored_rows, "decisions": len(found.decision_ns)}  # fmt: skip


def compute_days(study: MakerStudy, days: Sequence[date], access: Any, folder: Path,
                 job: Callable[..., dict[str, Any]] = day_job) -> list[dict[str, Any]]:  # fmt: skip
    """Every day's record by ``job``, kept under ``folder`` so a stopped stage resumes."""
    folder.mkdir(parents=True, exist_ok=True)
    todo = [d for d in days if not (folder / f"{d}.json").exists()]
    args = [(study.repo, d, access, study.binding, study.fees()) for d in todo]
    if study.workers > 1 and len(todo) > 1:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(study.workers, mp_context=context) as pool:
            for record in pool.map(job, *zip(*args, strict=True)):
                write_json(folder / f"{record['day']}.json", record)
    else:
        for arg in args:
            record = job(*arg)
            write_json(folder / f"{record['day']}.json", record)
    return [json.loads((folder / f"{d}.json").read_text("utf-8")) for d in days]


def signal_days(study: MakerStudy, days: Sequence[date], access: Any, folder: Path) -> list[dict[str, Any]]:
    return compute_days(study, days, access, folder, signal_job)
