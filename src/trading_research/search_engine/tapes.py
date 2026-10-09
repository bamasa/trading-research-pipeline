"""Candidate-row tapes: per test day and signal configuration, where the loosest cell could enter.

A signal configuration's cells differ only in their decision threshold, gate
and regime window, and every stricter cell enters on a subset of the loosest
cell's rows: hold-first's thresholds at 240 trades a day (stratum C) or the EV
gate at k = 0 and beta = 0 (stratum R), with the gate open and no guard. Stage
A writes those rows once per configuration and test day, each with:

* its decision row in the day's book and its stamp, and the loosest side;
* its scores: ``p_long`` and ``p_short`` (C), or ``mu_bp``, ``sigma_bp`` and
  ``walk_cost_bp`` (R), all float64, so a stricter cell's threshold decides on
  exactly the values the loosest one did;
* its gate states (``quiet_out``, ``tight_only``; ``open`` passes every row),
  from levels fitted on the fold's training rows;
* its guard states: inside ``(flag, flag + w]`` of a fold-wise flag, for w of
  4, 24 and 120 hours (``pipeline/execution.guarded_rows``).

A row's side is the same in every cell that keeps it: a hold-first row is long
because ``p_long > p_short``, an EV-gate row because ``mu`` clears a positive
hurdle. A tape is one zstd parquet file whose metadata names the configuration,
the day and the loosest thresholds, checked when it is read.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from trading_research.labels.directional import BUY, SELL
from trading_research.models.base import CLASSES
from trading_research.pipeline.execution import guarded_rows
from trading_research.strategies.ev_gate import EVGate
from trading_research.strategies.hold_first import HoldFirstRule

#: The regime windows a tape carries a guard state for, in hours.
GUARD_WINDOWS_H: tuple[int, ...] = (4, 24, 120)
#: The gates a tape carries a state for; ``open`` passes every row.
TAPE_GATES: tuple[str, ...] = ("quiet_out", "tight_only")
LOOSEST_TRADES_PER_DAY = 240
SCORES: dict[str, tuple[str, ...]] = {
    "C": ("p_long", "p_short"),
    "R": ("mu_bp", "sigma_bp", "walk_cost_bp"),
}
_META_KEY = b"candidate_tape"
NS_PER_HOUR = 3_600_000_000_000


class TapeError(ValueError):
    """A tape is malformed or is not the one asked for."""


@dataclass(frozen=True)
class CandidateTape:
    """One configuration's candidate rows on one test day."""

    day: date
    configuration: str
    stratum: str
    rows: np.ndarray
    stamps: np.ndarray
    side: np.ndarray
    scores: dict[str, np.ndarray]
    gates: dict[str, np.ndarray]
    guards: dict[int, np.ndarray]
    #: The loosest cell's thresholds: ``theta_long``/``theta_short`` (C) or
    #: ``cost_bp`` with k = beta = 0 (R).
    loosest: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.stratum not in SCORES:
            raise TapeError(f"unknown stratum {self.stratum!r}")
        m = len(self.rows)
        arrays = [self.stamps, self.side, *self.scores.values(), *self.gates.values()]
        if any(len(a) != m for a in [*arrays, *self.guards.values()]):
            raise TapeError("every column of a tape needs one value per candidate row")
        if set(self.scores) != set(SCORES[self.stratum]):
            raise TapeError(f"stratum {self.stratum} tapes carry {SCORES[self.stratum]}")
        if set(self.gates) != set(TAPE_GATES) or set(self.guards) != set(GUARD_WINDOWS_H):
            raise TapeError("a tape carries every gate state and every guard state")
        if m and (np.any(np.diff(self.rows) <= 0) or self.rows[0] < 0):
            raise TapeError("candidate rows must be increasing row positions")
        if not np.isin(self.side, (-1, 1)).all():
            raise TapeError("a candidate row is long (+1) or short (-1)")

    def __len__(self) -> int:
        return len(self.rows)

    def to_frame(self) -> pd.DataFrame:
        columns: dict[str, Any] = {"row": self.rows, "stamp_ns": self.stamps, "side": self.side}
        columns |= {name: self.scores[name] for name in SCORES[self.stratum]}
        columns |= {f"gate_{name}": self.gates[name] for name in TAPE_GATES}
        columns |= {f"guard_{w}h": self.guards[w] for w in GUARD_WINDOWS_H}
        return pd.DataFrame(columns)

    def metadata(self) -> dict[str, Any]:
        return {
            "configuration": self.configuration,
            "day": self.day.isoformat(),
            "stratum": self.stratum,
            "loosest": self.loosest,
        }

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, metadata: Mapping[str, Any]) -> CandidateTape:
        stratum = str(metadata["stratum"])
        if stratum not in SCORES:
            raise TapeError(f"unknown stratum {stratum!r}")
        missing = [name for name in SCORES[stratum] if name not in frame.columns]
        if missing:
            raise TapeError(f"stratum {stratum} tapes carry {SCORES[stratum]}; {missing} missing")
        return cls(
            date.fromisoformat(str(metadata["day"])),
            str(metadata["configuration"]),
            stratum,
            frame["row"].to_numpy(dtype=np.int64),
            frame["stamp_ns"].to_numpy(dtype=np.int64),
            frame["side"].to_numpy(dtype=np.int8),
            {name: frame[name].to_numpy(dtype=np.float64) for name in SCORES[stratum]},
            {name: frame[f"gate_{name}"].to_numpy(dtype=bool) for name in TAPE_GATES},
            {w: frame[f"guard_{w}h"].to_numpy(dtype=bool) for w in GUARD_WINDOWS_H},
            {str(k): float(v) for k, v in dict(metadata["loosest"]).items()},
        )


def _tape(
    day: date,
    configuration: str,
    stratum: str,
    sides: np.ndarray,
    stamps: np.ndarray,
    scores: Mapping[str, np.ndarray],
    gates: Mapping[str, np.ndarray],
    flags_ns: np.ndarray,
    loosest: dict[str, float],
) -> CandidateTape:
    stamps = np.asarray(stamps, dtype=np.int64)
    if len(sides) != len(stamps) or any(len(g) != len(stamps) for g in gates.values()):
        raise TapeError("sides, stamps and gate states need one value per row of the day")
    rows = np.flatnonzero(np.asarray(sides) != 0).astype(np.int64)
    at = stamps[rows]
    return CandidateTape(
        day,
        configuration,
        stratum,
        rows,
        at,
        np.asarray(sides)[rows].astype(np.int8),
        {name: np.asarray(values, dtype=np.float64)[rows] for name, values in scores.items()},
        {name: np.asarray(gates[name], dtype=bool)[rows] for name in TAPE_GATES},
        {w: guarded_rows(at, flags_ns, w * NS_PER_HOUR) for w in GUARD_WINDOWS_H},
        loosest,
    )


def hold_first_tape(
    day: date,
    configuration: str,
    stamps: np.ndarray,
    proba: np.ndarray,
    rule: HoldFirstRule,
    gates: Mapping[str, np.ndarray],
    flags_ns: np.ndarray,
) -> CandidateTape:
    """Stratum C: the rows where hold-first at 240 trades a day (``rule``) enters.

    ``proba`` is ``(rows, 3)`` in the models' class order; ``gates`` holds each
    tape gate's state at every row; ``flags_ns`` the fold's flags."""
    proba = np.asarray(proba, dtype=np.float64)
    scores = {"p_long": proba[:, CLASSES.index(BUY)], "p_short": proba[:, CLASSES.index(SELL)]}
    sides = rule.decide(scores["p_long"], scores["p_short"])
    loosest = {"theta_long": float(rule.theta_long), "theta_short": float(rule.theta_short)}
    return _tape(day, configuration, "C", sides, stamps, scores, gates, flags_ns, loosest)


def ev_gate_tape(
    day: date,
    configuration: str,
    stamps: np.ndarray,
    mu_bp: np.ndarray,
    sigma_bp: np.ndarray,
    walk_cost_bp: np.ndarray,
    gates: Mapping[str, np.ndarray],
    flags_ns: np.ndarray,
    cost_bp: float = EVGate(0.0, 0.0).cost_bp,
) -> CandidateTape:
    """Stratum R: the rows where the EV gate at k = 0 and beta = 0 is open."""
    gate = EVGate(0.0, 0.0, cost_bp)
    sides = gate.sides(mu_bp, sigma_bp, walk_cost_bp)
    scores = {"mu_bp": mu_bp, "sigma_bp": sigma_bp, "walk_cost_bp": walk_cost_bp}
    plain = {name: np.asarray(values, dtype=np.float64) for name, values in scores.items()}
    return _tape(
        day, configuration, "R", sides, stamps, plain, gates, flags_ns, {"cost_bp": cost_bp}
    )


def tape_path(root: Path | str, configuration: str, day: date) -> Path:
    """``<root>/<configuration, made a file name>/<day>.parquet``."""
    folder = re.sub(r"[^A-Za-z0-9_.=-]", "_", configuration.replace("|", "__"))
    return Path(root) / folder / f"{day.isoformat()}.parquet"


def write_tape(tape: CandidateTape, path: Path | str) -> Path:
    """One zstd parquet file, the tape's identity in its schema metadata."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(tape.to_frame(), preserve_index=False)
    meta = dict(table.schema.metadata or {})
    meta[_META_KEY] = json.dumps(tape.metadata(), sort_keys=True).encode()
    pq.write_table(table.replace_schema_metadata(meta), target, compression="zstd")
    return target


def read_tape(
    path: Path | str, *, configuration: str | None = None, day: date | None = None
) -> CandidateTape:
    """Read a tape, refused if it is not ``configuration``'s on ``day``."""
    table = pq.read_table(Path(path))
    raw = (table.schema.metadata or {}).get(_META_KEY)
    if raw is None:
        raise TapeError(f"{path} is not a candidate tape")
    tape = CandidateTape.from_frame(table.to_pandas(), json.loads(raw))
    if configuration is not None and tape.configuration != configuration:
        raise TapeError(f"{path} holds {tape.configuration!r}, not {configuration!r}")
    if day is not None and tape.day != day:
        raise TapeError(f"{path} holds {tape.day}, not {day}")
    return tape
