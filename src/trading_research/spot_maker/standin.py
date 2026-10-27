"""Stand-ins for the tests and the dry run: a synthetic source checkout, never a result.

* :class:`StandInModel`, a deterministic three-class model on S25-PF's inputs
  (picklable), in place of the spot study's frozen fit;
* :func:`write_source`, a stand-in of the spot study's checkout: its signal
  files (``models/spot_2026/...``) and its converted BTCFDUSD day files
  (``data/spot/binance/prints/BTCFDUSD/<day>.parquet``), the prints drawn by the
  spot study's synthetic venue (``spot/synthetic.SpotMarket``) and converted by
  its Binance parser (``spot/binance.parse``), so they are in the print
  contract; it returns the :class:`~trading_research.spot_maker.signal.SignalBinding`
  of its files.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Iterable
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from trading_research.market_making.prereg import sha256_bytes
from trading_research.spot import binance
from trading_research.spot.features_prints import PRINT_MEMBERS
from trading_research.spot.synthetic import SpotMarket
from trading_research.spot_maker import blocks as bk
from trading_research.spot_maker import registration as rg
from trading_research.spot_maker.signal import SOURCE_DIR, SignalBinding

#: The stand-in's thresholds: a few hundred decisions on a synthetic day.
STAND_IN_THETA: dict[str, float] = {"theta_long": 0.47, "theta_short": 0.47}


class StandInModel:
    """``p_buy`` and ``p_sell`` from the six-row flow imbalance; ``p_hold`` the rest."""

    def __init__(self) -> None:
        self.columns_ = list(PRINT_MEMBERS)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        lean = np.tanh(3.0 * x["flow_imbalance_6"].to_numpy(dtype=np.float64))
        buy = 0.2 + 0.3 * (lean + 1.0) / 2.0
        sell = 0.2 + 0.3 * (1.0 - lean) / 2.0
        return np.column_stack([sell, 1.0 - buy - sell, buy])


def market(days: Iterable[date], seed: int = 2026) -> SpotMarket:
    return SpotMarket(frozenset(days), step_ms=2_000, seed=seed, print_rate=0.9)


def prints_frame(found: SpotMarket, day: date) -> pd.DataFrame:
    """One synthetic day of Binance prints in the print contract."""
    return binance.parse(found.binance_archive(day), day)


def write_source(root: Path, days: Iterable[date], seed: int = 2026) -> SignalBinding:
    """A stand-in source checkout under ``root``; the binding of its signal files."""
    wanted = sorted(set(days))
    found = market(wanted, seed)
    for day in wanted:
        path = bk.day_file(root, day)
        path.parent.mkdir(parents=True, exist_ok=True)
        prints_frame(found, day).to_parquet(path, index=False)
    slot = {"row": rg.SIGNAL_ROW, "stand_in": "dry run", "cells": {rg.SIGNAL_CELL: STAND_IN_THETA}}
    files = {("slots", rg.SIGNAL_SLOT_FILE): json.dumps(slot, sort_keys=True).encode(),
             ("weights", rg.SIGNAL_FIT): pickle.dumps(StandInModel())}  # fmt: skip
    digests = {}
    for (sub, name), data in files.items():
        target = Path(root) / SOURCE_DIR / sub / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        digests[name] = sha256_bytes(data)
    return SignalBinding(digests[rg.SIGNAL_SLOT_FILE], digests[rg.SIGNAL_FIT], STAND_IN_THETA)
