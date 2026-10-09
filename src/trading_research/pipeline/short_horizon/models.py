"""The study's four models, built at the run's scale, saved by file and hashed.

Stratum C fits ``gbm`` (``models/gbm.py``, three classes) and ``logistic``
(``models/linear.py``); stratum R fits ``tcn`` (the network with the Gaussian
head) and ``bound_ridge`` (a ridge for the mean and one for the variance,
``models/uncertainty.HeteroscedasticRegressor``). Hyperparameters are the
repository's defaults, overridden only by the dry run's scale.

**One process per model kind.** The network never shares a process with
XGBoost (``models/tcn.py``: both bring an OpenMP runtime, and the network hangs
once XGBoost has run). :func:`in_fresh_process` runs a function in a spawned
interpreter, which every network fit and timing goes through.
"""

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any, TypeVar

from trading_research.pipeline.short_horizon.plan import Scale

C_MODELS: tuple[str, ...] = ("gbm", "logistic")
R_MODELS: tuple[str, ...] = ("tcn", "bound_ridge")
T = TypeVar("T")


def classifier(name: str, scale: Scale) -> Any:
    """A stratum C model at the scale's settings."""
    if name == "gbm":
        from trading_research.models.gbm import GradientBoostedBaseline

        return GradientBoostedBaseline(**scale.params("gbm"))
    if name == "logistic":
        from trading_research.models.linear import LogisticBaseline

        return LogisticBaseline(**scale.params("logistic"))
    raise ValueError(f"{name!r} is not a stratum C model; they are {C_MODELS}")


def bound_ridge(scale: Scale) -> Any:
    """The bound regressor: a ridge for the mean and a ridge for the variance."""
    from trading_research.models.regression import RidgeBaseline
    from trading_research.models.uncertainty import HeteroscedasticRegressor

    params = scale.params("bound_ridge")
    return HeteroscedasticRegressor(RidgeBaseline(**params), RidgeBaseline(**params))


def network(scale: Scale, **changes: Any) -> Any:
    """The network with the Gaussian head; ``changes`` (``epochs``, ``device``) on top."""
    from trading_research.models.tcn import GaussianTCN

    return GaussianTCN(**{**scale.params("tcn"), **changes})


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def save(model: Any, path: Path) -> str:
    """Write a fitted model (the network through its own ``save``); its file's sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "weights_sha256") and hasattr(model, "save"):
        model.save(path)
    else:
        with path.open("wb") as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return file_sha256(path)


def load(path: Path, *, kind: str) -> Any:
    """A model written by :func:`save`; ``kind`` is the study's model name."""
    if kind == "tcn":
        from trading_research.models.tcn import GaussianTCN

        return GaussianTCN.load(path, device="cpu")
    with path.open("rb") as handle:
        return pickle.load(handle)  # the study's own files, under the ignored models/


def in_fresh_process(fn: Callable[..., T], *args: Any) -> T:
    """``fn(*args)`` in a spawned interpreter; its result comes back pickled."""
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as pool:
        return pool.submit(fn, *args).result()
