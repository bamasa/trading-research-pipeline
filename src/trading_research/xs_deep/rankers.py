"""The six rankers, with the settings the registration fixed.

A ranker sees a batch of decision times, each with every instrument's twelve
features and four series, and scores every instrument; only the order of the
scores within a time is used. ``reversal`` is not fitted. ``ridge`` and
``lightgbm`` are fitted on the rows (time, instrument) pooled; the three
networks on batches of 64 decision times, so ``xs_transformer`` sees each
time's instruments together.

Run LightGBM and torch in separate processes: each ships an OpenMP runtime
(see ``models/tcn.py`` for the hang two of them caused in one interpreter).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from trading_research.xs_deep.features import REL_10, TABULAR

NAMES = ("reversal", "ridge", "lightgbm", "mlp", "tcn", "xs_transformer")

RIDGE_PENALTY = 10.0
LIGHTGBM = {
    "objective": "regression",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "seed": 0,
    "deterministic": True,
    "verbose": -1,
}
LIGHTGBM_ROUNDS = 300
NET_LR, NET_WEIGHT_DECAY = 1e-3, 1e-4
NET_EPOCHS, NET_BATCH, NET_SEED, NET_WIDTH = 6, 64, 0, 32
THREADS = 8


@dataclass
class Batch:
    """Standardised inputs of B decision times by N instruments."""

    tab: np.ndarray  # (B, N, 12) float32
    seq: np.ndarray  # (B, N, WINDOW, 4) float32
    mask: np.ndarray  # (B, N) bool: the row is usable
    y: np.ndarray | None = None  # (B, N): the target's z-score, 0 where masked

    def take(self, idx: np.ndarray) -> Batch:
        y = None if self.y is None else self.y[idx]
        return Batch(self.tab[idx], self.seq[idx], self.mask[idx], y)


class Reversal:
    name = "reversal"

    def fit(self, batch: Batch) -> None:  # noqa: ARG002
        return None

    def score(self, batch: Batch) -> np.ndarray:
        return np.asarray(-batch.tab[..., REL_10], dtype="float64")


class Ridge:
    name = "ridge"

    def fit(self, batch: Batch) -> None:
        assert batch.y is not None
        x, y = batch.tab[batch.mask].astype("float64"), batch.y[batch.mask]
        self.mu, self.ymu = x.mean(axis=0), y.mean()
        xc = x - self.mu
        gram = xc.T @ xc + RIDGE_PENALTY * np.eye(len(TABULAR))
        self.w = np.linalg.solve(gram, xc.T @ (y - self.ymu))

    def score(self, batch: Batch) -> np.ndarray:
        return np.asarray((batch.tab.astype("float64") - self.mu) @ self.w + self.ymu)


class LightGBM:
    name = "lightgbm"

    def fit(self, batch: Batch) -> None:
        import lightgbm as lgb

        assert batch.y is not None
        data = lgb.Dataset(batch.tab[batch.mask], batch.y[batch.mask])
        params = {**LIGHTGBM, "num_threads": THREADS}
        self.model = lgb.train(params, data, num_boost_round=LIGHTGBM_ROUNDS)

    def score(self, batch: Batch) -> np.ndarray:
        b, n, f = batch.tab.shape
        return np.asarray(self.model.predict(batch.tab.reshape(b * n, f))).reshape(b, n)


class Network:
    """A torch module fitted with the registration's loop; see :mod:`.nets`."""

    def __init__(self, name: str) -> None:
        self.name = name

    def fit(self, batch: Batch) -> None:
        import torch

        from trading_research.xs_deep.nets import build

        assert batch.y is not None
        torch.manual_seed(NET_SEED)
        torch.set_num_threads(THREADS)
        rng = np.random.default_rng(NET_SEED)
        self.model = build(self.name, NET_WIDTH)
        opt = torch.optim.Adam(self.model.parameters(), lr=NET_LR, weight_decay=NET_WEIGHT_DECAY)
        self.model.train()
        for _ in range(NET_EPOCHS):
            order = rng.permutation(len(batch.mask))
            for start in range(0, len(order), NET_BATCH):
                part = batch.take(order[start : start + NET_BATCH])
                tab, seq, mask = _tensors(part)
                y = torch.from_numpy(np.asarray(part.y, dtype="float32"))
                pred = self.model(tab, seq, mask)
                weight = mask.float()
                loss = (((pred - y) ** 2) * weight).sum() / weight.sum().clamp(min=1.0)
                opt.zero_grad()
                loss.backward()
                opt.step()

    def score(self, batch: Batch) -> np.ndarray:
        import torch

        self.model.eval()
        out = []
        with torch.no_grad():
            for start in range(0, len(batch.mask), 256):
                part = batch.take(np.arange(start, min(start + 256, len(batch.mask))))
                out.append(self.model(*_tensors(part)).numpy())
        return np.concatenate(out).astype("float64")


def _tensors(batch: Batch) -> tuple[Any, Any, Any]:
    import torch

    return (
        torch.from_numpy(np.ascontiguousarray(batch.tab)),
        torch.from_numpy(np.ascontiguousarray(batch.seq)),
        torch.from_numpy(np.ascontiguousarray(batch.mask)),
    )


def make(name: str) -> Reversal | Ridge | LightGBM | Network:
    if name == "reversal":
        return Reversal()
    if name == "ridge":
        return Ridge()
    if name == "lightgbm":
        return LightGBM()
    if name in ("mlp", "tcn", "xs_transformer"):
        return Network(name)
    raise ValueError(f"unknown ranker {name!r}; one of {', '.join(NAMES)}")
