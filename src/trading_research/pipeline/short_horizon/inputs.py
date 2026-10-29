"""What stage 1 left for the later stages, read the same way by each of them.

:class:`Inputs` reads stage 1's record (usable and cold days, the gap
threshold, the selected features, W's fitted values and the stores' layout)
and gives every stage the same view of a day: its book through the study's
loader under the stage's access, its executor book with funding, its stored
features at every row or every tenth, and its targets.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd

from trading_research.backtest.taker import TakerBook
from trading_research.data import study_access as sa
from trading_research.data import study_loader as sl
from trading_research.data.feature_store import FeatureStore, StoreLayout, open_stores
from trading_research.features.lob_tensor import TensorCalibration
from trading_research.pipeline.short_horizon import features as ft
from trading_research.pipeline.short_horizon.context import Study
from trading_research.pipeline.short_horizon.plan import midnight_ns


@dataclass
class Inputs:
    """Stage 1's outputs for ``study``, read under ``access`` (development by default)."""

    study: Study
    access: sa.StudyAccess = field(default_factory=sa.development_access)
    #: Stage 1's record while stage 1 itself is still running (its leakage tests).
    given: dict[str, Any] | None = None

    @cached_property
    def record(self) -> dict[str, Any]:
        return self.given if self.given is not None else self.study.record("1")

    @property
    def threshold(self) -> int:
        return int(self.record["max_sequence_gaps"])

    @property
    def selected(self) -> list[str]:
        return list(self.record["selected"])

    def columns(self, inputs: str) -> list[str]:
        """A model's input columns: the micro-selected set, with the tensor when named."""
        if inputs == "micro":
            return self.selected
        if inputs in ("micro+tensor", "tensor+micro"):
            return [*self.selected, *ft.TENSOR]
        raise ValueError(f"unknown inputs {inputs!r}")

    @cached_property
    def calibration(self) -> TensorCalibration:
        return TensorCalibration.from_dict(self.record["fitted"]["tensor"])

    @cached_property
    def stores(self) -> dict[str, FeatureStore]:
        layout = StoreLayout.from_dict(self.record["stores"]["layout"])
        return open_stores(self.study.work / "1" / "features", layout)

    def usable(self, block: str = "D") -> list[date]:
        days = self.record["days"]
        return [d for d in self.study.scale.days(block) if days[d.isoformat()]["usable"]]

    def cold(self, day: date) -> bool:
        return bool(self.record["days"][day.isoformat()]["cold"])

    def scored_from(self, day: date) -> int:
        return ft.scored_from_ns(day, self.cold(day))

    def book(self, day: date) -> pd.DataFrame:
        found = sl.load_book_day(self.study.symbol, day, self.study.roots.book,
                                 max_sequence_gaps=self.threshold, access=self.access)  # fmt: skip
        if found.frame is None or not found.usable:
            raise RuntimeError(f"{day} is not a usable book day: {found.exclusions}")
        return found.frame

    def funding(self, day: date) -> pd.DataFrame:
        return sl.load_funding_day(
            self.study.symbol, day, self.study.roots.funding, access=self.access
        )

    def prints(self, day: date) -> pd.DataFrame:
        return sl.load_prints_day(
            self.study.symbol, day, self.study.roots.trades, access=self.access
        )

    def taker_book(self, day: date, frame: pd.DataFrame | None = None) -> TakerBook:
        rows = self.book(day) if frame is None else frame
        return TakerBook.from_frame(rows, day, funding=self.funding(day))

    def targets(self, day: date) -> pd.DataFrame:
        return pd.read_parquet(self.study.work / "1" / "targets" / f"{day}.parquet")

    def features(self, day: date, columns: Sequence[str], *, every_row: bool) -> pd.DataFrame:
        store = self.stores["every_row" if every_row else "every_tenth_row"]
        return store.read_day(day, columns=list(columns))

    def training(
        self,
        days: Sequence[date],
        columns: Sequence[str],
        target: str,
        before_ns: int,
        max_rows: int,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Every-tenth rows of ``days`` stamped before ``before_ns`` with a target,
        thinned evenly in time to ``max_rows``; float64 inputs."""
        xs, ys = [], []
        for day in days:
            targets = self.targets(day)
            x = self.features(day, columns, every_row=False)
            if len(x) != len(targets):
                raise RuntimeError(f"{day}: {len(x)} stored rows against {len(targets)} targets")
            keep = (targets[ft.TIME].to_numpy() < before_ns) & np.isfinite(
                targets[target].to_numpy()
            )
            xs.append(x[keep].astype(np.float64))
            ys.append(targets[target][keep])
        if not xs:
            return pd.DataFrame(columns=list(columns)), pd.Series(dtype=np.float64)
        x, y = pd.concat(xs, ignore_index=True), pd.concat(ys, ignore_index=True)
        if len(x) > max_rows:
            kept = np.floor(np.arange(max_rows) * len(x) / max_rows).astype(np.int64)
            x, y = x.iloc[kept].reset_index(drop=True), y.iloc[kept].reset_index(drop=True)
        return x, y

    def day_start(self, day: date) -> int:
        return midnight_ns(day)
