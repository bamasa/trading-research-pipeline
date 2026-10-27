"""The pipeline from Python: :class:`Pipeline`, :class:`Backtest` and :class:`Study`.

Three objects, each a composition of what the command line runs, so a result
from Python is the result the commands would give:

:class:`Pipeline`
    data → regimes → instruments → signal → execution → verdict on one
    dataset, as a run directory a reader can open step by step. The data is
    checked against its contract (``trp data validate-data``), the break
    detector runs on the search block's closes (``trp regimes breaks``), and the
    search chooses the instrument, the frequency, rule or model, the model and
    the execution on the search block before the held-out block is read once
    through the run's ledger (``trp search ...``). ``execution`` is ``taker``,
    ``passive``, ``maker`` or ``all``: the modes the verdict reads, the family
    Holm's m counts.

:class:`Backtest`
    One execution mode on a day range of data already on disk: a taker or a
    passive entry on the five-second grid, or the market maker in event time,
    through :func:`trading_research.pipeline.execution.execute` as
    ``reproduce`` and ``mm-backtest`` call it.

:class:`Study`
    A registered study, run through its own entry point
    (:mod:`trading_research.studies`).

Example, on the synthetic market the demo generates::

    from datetime import date
    from trading_research import Pipeline, PipelineConfig
    from trading_research.walkthrough.layout import SyntheticInstrument, write_synthetic

    write_synthetic(
        Path("data/walk"),
        [SyntheticInstrument("SYNAUSDT", 1, 0.3)],
        first_day=date(2031, 3, 3),
        days=10,
    )
    result = Pipeline(PipelineConfig(data=Path("data/walk"), out=Path("artifacts/walk"))).run()
    result.verdict["modes"]["taker"]["status"]
"""

from __future__ import annotations

import contextlib
import io
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from trading_research.pipeline.execution import ExecutionResult, SignalStream
from trading_research.studies import EntryPointStudy, StudyRun
from trading_research.studies import get as _get_study
from trading_research.walkthrough import execution as _execution
from trading_research.walkthrough import steps as _steps
from trading_research.walkthrough.layout import Dataset, split_days
from trading_research.walkthrough.record import SearchRun

EXECUTIONS: tuple[str, ...] = ("taker", "passive", "maker", "all")


def command(args: Sequence[str], *, quiet: bool = True) -> tuple[int, str]:
    """Run one ``trp`` command in this process; its exit code and what it printed."""
    from typer.main import get_command

    from trading_research.cli import app

    buffer = io.StringIO()
    target = buffer if quiet else None
    with contextlib.redirect_stdout(target) if target else contextlib.nullcontext():
        code = get_command(app).main(args=list(args), prog_name="trp", standalone_mode=False)
    return int(code or 0), buffer.getvalue()


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------


@dataclass
class Backtest:
    """One execution mode over a day range, on data already on disk.

    ``data`` is a dataset root in the walkthrough layout (``raw/``, and
    ``book/``, ``trades/``, ``funding/`` for the market maker; see
    :mod:`trading_research.walkthrough.layout`). ``fee`` names a row of Bybit's
    linear schedule (``trp fees``); ``slippage_bp`` is charged per crossing leg.

    >>> bt = Backtest(Path("data/demo/data"), "SYNAUSDT")  # doctest: +SKIP
    >>> bt.run("maker", date(2031, 3, 10), date(2031, 3, 11)).summary  # doctest: +SKIP
    >>> bt.run(
    ...     "taker", start, end, signal=bt.signal(frame, decisions, hold_rows=12)
    ... )  # doctest: +SKIP

    A taker or a passive entry executes a :class:`SignalStream` (decisions in
    event time, :meth:`signal` builds one from rows of predictions); the market
    maker quotes S1 by ``mm-backtest``'s defaults, its clip and volatility
    reference measured on ``reference_days`` (the range itself when not given),
    and leans on a signal when one is given. Days of the market-making study's
    held-out blocks are refused, as ``mm-backtest`` refuses them.
    """

    data: Path
    symbol: str
    fee: str = "base"
    slippage_bp: float = 0.5

    @property
    def dataset(self) -> Dataset:
        return Dataset.under(self.data)

    @property
    def costs(self) -> _execution.Costs:
        return _execution.Costs.named(self.fee, self.slippage_bp)

    def signal(
        self,
        frame: pd.DataFrame,
        decisions: Sequence[int],
        *,
        hold_rows: int,
        strength_bp: float = 1.0,
    ) -> SignalStream:
        """Decisions (-1, 0, +1) on rows stamped ``frame["timestamp"]`` as a signal stream."""
        import numpy as np

        return _execution.stream(self.symbol, frame, np.asarray(decisions), hold_rows, strength_bp)

    def run(
        self,
        mode: str,
        start: date,
        end: date,
        *,
        signal: SignalStream | None = None,
        hold_rows: int = 12,
        reference_days: Sequence[date] | None = None,
    ) -> ExecutionResult:
        """``mode`` (taker, passive or maker) over ``start``..``end`` inclusive."""
        days = list(pd.date_range(start, end, freq="D").date)
        plan = None
        if mode == "maker":
            plan = _execution.maker_plan(
                self.dataset,
                self.symbol,
                list(reference_days or days),
                self.costs,
                lean=signal is not None,
            )
        return _execution.run_mode(
            mode,
            self.dataset,
            self.symbol,
            days,
            self.costs,
            signal=signal,
            hold_rows=hold_rows,
            plan=plan,
        )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineConfig:
    """Everything a pipeline run decides before it reads data.

    ``data`` is a dataset root; ``out`` the run's directory (one folder per
    step). ``execution`` is the family the verdict reads: ``taker``,
    ``passive``, ``maker`` or ``all``. The fee is a tier of Bybit's linear
    schedule; the rest are the search's registration (see
    :func:`trading_research.walkthrough.steps.instruments`).
    """

    data: Path
    out: Path = Path("artifacts/pipeline")
    execution: str = "all"
    symbols: tuple[str, ...] | None = None
    fee: str = "base"
    slippage_bp: float = 0.5
    search_share: float = 0.65
    holds_s: tuple[int, ...] = (30, 60, 120, 300)
    minimum_trades: int = 30
    min_days: int = 2
    placebos: int = 40
    maker_placebos: int = 3
    seed: int = 20260101
    regime_bar: str = "5min"
    regime_history: int = 250
    regime_online: int = 60

    def __post_init__(self) -> None:
        if self.execution not in EXECUTIONS:
            raise ValueError(f"execution must be one of {EXECUTIONS}, got {self.execution!r}")

    @property
    def family(self) -> tuple[str, ...]:
        return _steps.FAMILY if self.execution == "all" else (self.execution,)


@dataclass
class PipelineResult:
    """What a run established, step by step, and how long each step took."""

    out: Path
    validation: dict[str, Any] = field(default_factory=dict)
    regimes: dict[str, Any] = field(default_factory=dict)
    chosen: dict[str, Any] = field(default_factory=dict)
    verdict: dict[str, Any] = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)

    @property
    def run(self) -> SearchRun:
        return SearchRun(self.out / "search")


class Pipeline:
    """data → regimes → instruments → signal → execution → verdict, on one dataset.

    Each method is one step and can be called alone once the steps before it
    have run; :meth:`run` calls them in order. The signal steps fit boosted trees,
    which the repository runs only in a process that has not imported torch. Everything is written under
    ``config.out``: ``validation.txt``, ``regimes/<symbol>/``, and the search's
    directory ``search/`` with one folder per step, its registration and its
    ledger.
    """

    def __init__(self, config: PipelineConfig, report: Callable[[str], None] | None = None) -> None:
        self.config = config
        self.report = report or (lambda _line: None)
        self.result = PipelineResult(config.out)

    @property
    def dataset(self) -> Dataset:
        return Dataset.under(self.config.data)

    @property
    def search(self) -> SearchRun:
        return SearchRun(self.config.out / "search")

    def _days(self) -> tuple[list[date], list[date]]:
        names = list(self.config.symbols or self.dataset.symbols())
        common = sorted(set.intersection(*(set(self.dataset.days(s)) for s in names)))
        return split_days(common, self.config.search_share)

    def data(self) -> dict[str, Any]:
        """The dataset against its contract: ``trp data validate-data`` on the raw planes."""
        code, text = command(["data", "validate-data", "--input", str(self.dataset.raw)])
        self.config.out.mkdir(parents=True, exist_ok=True)
        (self.config.out / "validation.txt").write_text(text)
        self.result.validation = {
            "passed": code == 0,
            "exit_code": code,
            "output": "validation.txt",
        }
        if code != 0:
            raise RuntimeError(
                f"the data does not meet its contract; see {self.config.out / 'validation.txt'}"
            )
        return self.result.validation

    def regimes(self) -> dict[str, Any]:
        """``trp regimes breaks`` on each instrument's closes over the search block only."""
        search_days, _ = self._days()
        found: dict[str, Any] = {}
        for symbol in self.config.symbols or self.dataset.symbols():
            out = self.config.out / "regimes" / symbol
            out.mkdir(parents=True, exist_ok=True)
            book = pd.concat(
                [pd.read_parquet(self.dataset.raw / symbol / "bookTicker" / f"{d}.parquet",
                                 columns=["timestamp", "bid_price_0", "ask_price_0"]) for d in search_days],
                ignore_index=True,
            )  # fmt: skip
            mid = ((book["bid_price_0"] + book["ask_price_0"]) / 2).set_axis(book["timestamp"])
            close = mid.resample(self.config.regime_bar).last().dropna()
            prices = out / "closes.csv"
            close.rename("close").rename_axis("timestamp").reset_index().to_csv(prices, index=False)
            code, _ = command([
                "regimes", "breaks", "--prices", str(prices), "--symbol", symbol,
                "--interval", self.config.regime_bar, "--history", str(self.config.regime_history),
                "--online", str(self.config.regime_online), "--output", str(out),
            ])  # fmt: skip
            breaks = (
                pd.read_csv(out / "breaks.csv") if (out / "breaks.csv").exists() else pd.DataFrame()
            )
            found[symbol] = {"exit_code": code, "closes": len(close), "breaks": len(breaks),
                             "flagged": breaks.get("timestamp", pd.Series(dtype=str)).astype(str).tolist()}  # fmt: skip
        self.result.regimes = found
        return found

    def register(self) -> str:
        """The search's registration, from the dataset's file names, before any day is read."""
        c = self.config
        return _steps.register(
            self.search, self.dataset, symbols=c.symbols, search_share=c.search_share, fee=c.fee,
            slippage_bp=c.slippage_bp, family=c.family, holds_s=c.holds_s,
            minimum_trades=c.minimum_trades, min_days=c.min_days, placebo_draws=c.placebos,
            maker_placebo_draws=c.maker_placebos, seed=c.seed, report=self.report,
        )  # fmt: skip

    def instruments(self) -> Path:
        """Every instrument screened on the search block; the first by edge over cost taken."""
        if not self.search.registration_path.exists():
            self.register()
        return _steps.instruments(self.search, report=self.report)

    def signal(self) -> Path:
        """Frequency, method and model, each chosen on the search block's validation days."""
        for step in ("frequency", "method", "model"):
            out = _steps.STEP_FUNCTIONS[step](self.search, report=self.report)
        return out

    def execution(self) -> Path:
        """The chosen signal in each mode on the search block; every choice frozen."""
        return _steps.execution(self.search, report=self.report)

    def verdict(self, *, second_read: bool = False) -> dict[str, Any]:
        """The held-out block, once, through the ledger; the registered verdict."""
        out = _steps.read(self.search, second_read=second_read, report=self.report)
        chosen = json.loads((out / "chosen.json").read_text())
        self.result.chosen = chosen
        self.result.verdict = json.loads(Path(chosen["results"]).read_text())
        return self.result.verdict

    def run(self) -> PipelineResult:
        """Every step in order; each step's wall time in ``result.seconds``."""
        for name, step in (("register", self.register), ("data", self.data), ("regimes", self.regimes),
                           ("instruments", self.instruments),
                           ("signal", self.signal), ("execution", self.execution), ("verdict", self.verdict)):  # fmt: skip
            started = time.perf_counter()
            self.report(f"== {name}")
            step()
            self.result.seconds[name] = time.perf_counter() - started
        return self.result

    def describe(self) -> dict[str, Any]:
        """The configuration as written into a manifest."""
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(self.config).items()}


# ---------------------------------------------------------------------------
# Study
# ---------------------------------------------------------------------------


class Study:
    """A registered study, through its own entry point.

    >>> spot = Study("spot_2026")  # doctest: +SKIP
    >>> spot.stages()  # doctest: +SKIP
    ('0', '1', 'D', 'amend', 'open H', 'read H', 'open P', 'read P')
    >>> spot.status().result  # doctest: +SKIP
    >>> spot.run("read H").refused  # doctest: +SKIP

    Every call runs ``python -m experiments.<name> ...`` in a fresh interpreter
    and returns a :class:`~trading_research.studies.StudyRun`; a refusal is the
    entry point's own, so a held-out block is never opened around its ledger.
    """

    def __init__(self, name: str, repo: Path | None = None) -> None:
        self._study: EntryPointStudy = _get_study(name, repo)

    @property
    def name(self) -> str:
        return self._study.name

    @property
    def registration(self) -> Path:
        return self._study.registration

    @property
    def config(self) -> Path:
        return self._study.config

    def stages(self) -> tuple[str, ...]:
        return self._study.stages()

    def run(self, stage: str, *args: str, **options: Any) -> StudyRun:
        return self._study.run(stage, *args, **options)

    def status(self) -> StudyRun:
        return self._study.status()

    def pages(self) -> StudyRun:
        return self._study.pages()

    def dry_run(self, keep: Path | None = None, **options: Any) -> StudyRun:
        return self._study.dry_run(keep, **options)

    @staticmethod
    def registered(repo: Path | None = None) -> list[str]:
        """The names of every registered study."""
        from trading_research.studies import studies

        return list(studies(repo))
