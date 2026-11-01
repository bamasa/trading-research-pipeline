"""S25-PF, frozen: its files copied and bound by hash, and its decisions on a day of prints.

The spot study's frozen files (``models/spot_2026/slots/S25-PF.json`` and the
fit ``S25-PF__hold_s=50__label_fee_bp=0.pkl``, in its checkout) are copied to
``models/spot_maker_2026/signal/``, each sha256 checked against the registered
value at the copy (:func:`copy_signal`) and at every load (:func:`load_signal`),
and the slot file's thresholds for S25's cell against the registered θ.

A day's decisions (:func:`day_decisions`) are computed exactly as the spot
study's read computed S25-PF's: the store ``binance``'s inputs by
``spot/days.store_frame``'s path (``spot/features_prints.day_frame`` with the
previous usable day's prints as history on a warm day), the probabilities by
``spot/days.predict`` from the day's first scored stamp, and the sides by
``spot/runs.decisions`` with the frozen hold-first rule. Nothing is refitted.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_research.data.feature_store import StoreLayout
from trading_research.features.timed import stamps_ns
from trading_research.market_making.prereg import HeldOutLocked, sha256_bytes
from trading_research.spot import days as dy
from trading_research.spot.features_prints import day_frame
from trading_research.spot.fits import load_model
from trading_research.spot.plan import STORES
from trading_research.spot.runs import decisions
from trading_research.spot_maker import registration as rg
from trading_research.strategies.hold_first import HoldFirstRule

SOURCE_DIR = Path("models/spot_2026")
STORE = "binance"


@dataclass(frozen=True)
class SignalBinding:
    """The sha256s and thresholds the copies must have: the registered ones, or a dry
    run's stand-in."""

    slot_sha256: str = rg.SIGNAL_SLOT_SHA256
    fit_sha256: str = rg.SIGNAL_FIT_SHA256
    theta: Mapping[str, float] = field(default_factory=lambda: dict(rg.SIGNAL_THETA))
    cell: str = rg.SIGNAL_CELL

    def files(self) -> dict[str, tuple[str, str]]:
        """``{file name: (subdirectory at the source, sha256)}``."""
        return {rg.SIGNAL_SLOT_FILE: ("slots", self.slot_sha256),
                rg.SIGNAL_FIT: ("weights", self.fit_sha256)}  # fmt: skip


def copy_signal(source: Path, repo: Path, binding: SignalBinding) -> dict[str, str]:
    """Copy the frozen files from the spot study's checkout, each checked by sha256."""
    out = {}
    for name, (sub, digest) in binding.files().items():
        target = Path(repo) / rg.SIGNAL_DIR / name
        if not (target.exists() and sha256_bytes(target.read_bytes()) == digest):
            data = (Path(source) / SOURCE_DIR / sub / name).read_bytes()
            if sha256_bytes(data) != digest:
                raise HeldOutLocked(f"{sub}/{name} at the source is not at its frozen sha256 {digest}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        out[name] = digest
    verify_signal(repo, binding)
    return out


def verify_signal(repo: Path, binding: SignalBinding) -> None:
    for name, (_, digest) in binding.files().items():
        path = Path(repo) / rg.SIGNAL_DIR / name
        if not path.exists() or sha256_bytes(path.read_bytes()) != digest:
            raise HeldOutLocked(f"the signal's copy {path.name} is missing or not at {digest}")


@dataclass(frozen=True)
class FrozenSignal:
    model: Any
    rule: HoldFirstRule


def load_signal(repo: Path, binding: SignalBinding) -> FrozenSignal:
    verify_signal(repo, binding)
    slot = json.loads((Path(repo) / rg.SIGNAL_DIR / rg.SIGNAL_SLOT_FILE).read_text("utf-8"))
    theta = {k: float(v) for k, v in slot["cells"][binding.cell].items()}
    if theta != dict(binding.theta):
        raise HeldOutLocked(f"the slot file's θ {theta} are not the registered {binding.theta}")
    model = load_model(Path(repo) / rg.SIGNAL_DIR / rg.SIGNAL_FIT, binding.fit_sha256)
    return FrozenSignal(model, HoldFirstRule(theta["theta_long"], theta["theta_short"]))


@dataclass(frozen=True)
class DayDecisions:
    """A day's print grid and the signal's decisions on it."""

    grid_ns: np.ndarray
    decision_ns: np.ndarray
    decision_side: np.ndarray
    scored_rows: int


def day_features(prints: pd.DataFrame, history: pd.DataFrame | None) -> pd.DataFrame:
    """The store ``binance``'s inputs at every row of the day's print grid."""
    _, names = STORES[STORE]
    return day_frame(prints, names=names, history=history, registry=dy.registry())


def day_decisions(signal: FrozenSignal, prints: pd.DataFrame, history: pd.DataFrame | None,
                  scored_from: int) -> DayDecisions:  # fmt: skip
    """The frozen rule's decisions at every scored row of the day's print grid."""
    return decide(signal, day_features(prints, history), scored_from)


def decide(signal: FrozenSignal, frame: pd.DataFrame, scored_from: int) -> DayDecisions:
    """The probabilities and the hold-first decisions on a day's inputs."""
    proba = dy.predict(signal.model, frame, StoreLayout(1.0, ()), scored_from)
    rows, side = decisions(proba, signal.rule)
    grid = np.asarray(stamps_ns(frame), dtype=np.int64)
    scored = int(np.isfinite(proba).all(axis=1).sum())
    return DayDecisions(grid, grid[rows], np.asarray(side, dtype=np.int64), scored)
