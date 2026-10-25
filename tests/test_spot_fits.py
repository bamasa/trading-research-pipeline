"""D's fits: training rows under the purge, the thresholds, and S25-T's 2025 files by sha256."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_research.market_making.prereg import HeldOutLocked
from trading_research.spot import access as sa
from trading_research.spot import fits
from trading_research.spot.context import Binding2025, SpotStudy
from trading_research.spot.ledger import file_sha256
from trading_research.spot.plan import NS_PER_S, folds, purge_ns


def frame(fold, n: int = 400) -> pd.DataFrame:  # type: ignore[no-untyped-def]
    start = fold.inner_start_ns - 2 * 86_400 * NS_PER_S
    stamps = np.linspace(start, fold.inner_start_ns + 86_400 * NS_PER_S, n).astype(np.int64)
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    return pd.DataFrame({
        "timestamp": pd.to_datetime(stamps, unit="ns", utc=True),
        "scored": True,
        "reads_until_ns": stamps + 60 * NS_PER_S,
        "a": x,
        "label_50_0": np.sign(np.round(x)),
    })  # fmt: skip


def test_training_rows_end_before_the_purge_and_are_thinned_evenly() -> None:
    fold = folds(sa.block_days("D"))["june"]
    data = frame(fold)
    x, y = fits.training_set(data, fold, 50.0, 0.0, ["a"], max_rows=1_000)
    stamps = data["timestamp"].astype("int64").to_numpy()
    expected = (stamps < fold.inner_start_ns - purge_ns(50.0)).sum()
    assert len(y) == expected and len(x) == len(y)
    x, y = fits.training_set(data.assign(scored=False), fold, 50.0, 0.0, ["a"], max_rows=1_000)
    assert len(y) == 0
    assert fits.thin_positions(10, 4).tolist() == [0, 2, 5, 7]


def test_a_fit_needs_two_classes_and_thresholds_need_rows() -> None:
    x = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    assert fits.fit_model(x, pd.Series([1.0, 1.0, 1.0]), {}) is None
    never = fits.rules_at(np.zeros((0, 3)), 7, [100.0, 240.0])
    assert all(math.isinf(r.theta_long) for r in never.values())
    assert fits.finite_theta(never[100.0]) == {"theta_long": 2.0, "theta_short": 2.0}
    proba = np.tile([0.2, 0.3, 0.5], (100, 1))
    assert fits.rules_at(proba, 1, [240.0])[240.0].theta_long == 0.5


def test_s25_t_is_refused_unless_every_2025_file_is_at_its_sha256(tmp_path: Path) -> None:
    config = tmp_path / "configs" / "short_horizon_2025.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("name: stand-in\n")
    weights = tmp_path / "models" / "short_horizon_2025" / "weights" / "fit.pkl"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"weights")
    files = {Path("models/short_horizon_2025/weights/fit.pkl"): file_sha256(weights)}
    study = SpotStudy(tmp_path, binding=Binding2025(Path("configs/short_horizon_2025.yaml"),
                                                    file_sha256(config), files))  # fmt: skip
    assert fits.check_2025(study) == tmp_path
    weights.write_bytes(b"changed")
    with pytest.raises(HeldOutLocked, match=r"fit\.pkl"):
        fits.check_2025(study)
    config.write_text("name: changed\n")
    with pytest.raises(HeldOutLocked, match="frozen sha256"):
        fits.check_2025(study)


def test_a_saved_fit_is_loaded_only_at_its_sha256(tmp_path: Path) -> None:
    digest = fits.save_model({"model": 1}, tmp_path / "m.pkl")
    assert fits.load_model(tmp_path / "m.pkl", digest) == {"model": 1}
    with pytest.raises(HeldOutLocked):
        fits.load_model(tmp_path / "m.pkl", "0" * 64)
