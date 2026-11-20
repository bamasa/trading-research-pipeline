"""The five choosers' scores, month by month, from the past alone.

The rules score a coin by its own funding (``f3``, ``f1``). The models are
refitted at every month on the rows of the months already ended, pooled
across coins, standardised on those rows, and score the month's candidates.
A model scores a month once it has :data:`MIN_TRAIN_MONTHS` months of rows.

LightGBM and torch each bring an OpenMP runtime; run them in separate processes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trading_research.carry_ml.features import FEATURES

NAMES = ("rule_lb3", "rule_lb1", "ridge", "lightgbm", "mlp")
MODELS = ("ridge", "lightgbm", "mlp")
MIN_TRAIN_MONTHS = 12
RIDGE_PENALTY = 1.0
LIGHTGBM = {
    "objective": "regression",
    "learning_rate": 0.05,
    "num_leaves": 7,
    "min_data_in_leaf": 20,
    "seed": 0,
    "deterministic": True,
    "num_threads": 4,
    "verbose": -1,
}
LIGHTGBM_ROUNDS = 200
MLP_HIDDEN, MLP_LR, MLP_WEIGHT_DECAY, MLP_EPOCHS, MLP_SEED = 32, 1e-3, 1e-3, 300, 0


def _standardise(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu, sd = train.mean(axis=0), train.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (train - mu) / sd, (test - mu) / sd


def ridge(x: np.ndarray, y: np.ndarray, test: np.ndarray) -> np.ndarray:
    xc, ymu = x - x.mean(axis=0), y.mean()
    w = np.linalg.solve(xc.T @ xc + RIDGE_PENALTY * np.eye(x.shape[1]), xc.T @ (y - ymu))
    return np.asarray((test - x.mean(axis=0)) @ w + ymu)


def lightgbm(x: np.ndarray, y: np.ndarray, test: np.ndarray) -> np.ndarray:
    import lightgbm as lgb

    model = lgb.train(LIGHTGBM, lgb.Dataset(x, y), num_boost_round=LIGHTGBM_ROUNDS)
    return np.asarray(model.predict(test))


def mlp(x: np.ndarray, y: np.ndarray, test: np.ndarray) -> np.ndarray:
    import torch
    from torch import nn

    torch.manual_seed(MLP_SEED)
    torch.set_num_threads(4)
    net = nn.Sequential(
        nn.Linear(x.shape[1], MLP_HIDDEN), nn.ReLU(),
        nn.Linear(MLP_HIDDEN, MLP_HIDDEN), nn.ReLU(),
        nn.Linear(MLP_HIDDEN, 1),
    )  # fmt: skip
    opt = torch.optim.Adam(net.parameters(), lr=MLP_LR, weight_decay=MLP_WEIGHT_DECAY)
    xt = torch.tensor(x, dtype=torch.float32)
    sd = float(y.std()) or 1.0
    yt = torch.tensor((y - y.mean()) / sd, dtype=torch.float32)
    for _ in range(MLP_EPOCHS):
        loss = ((net(xt).squeeze(-1) - yt) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        out = net(torch.tensor(test, dtype=torch.float32)).squeeze(-1).numpy()
    return np.asarray(out * sd + y.mean())


FIT: dict[str, Any] = {"ridge": ridge, "lightgbm": lightgbm, "mlp": mlp}


def scores(rows: pd.DataFrame, name: str, months: pd.PeriodIndex) -> pd.DataFrame:
    """(month, coin, score) for every month of ``months`` the chooser can score."""
    out = []
    for month in months:
        now = rows[rows["month"] == month]
        if now.empty:
            continue
        if name in ("rule_lb3", "rule_lb1"):
            s = now["f3" if name == "rule_lb3" else "f1"].to_numpy()
        else:
            past = rows[rows["month"] < month]
            if past["month"].nunique() < MIN_TRAIN_MONTHS:
                continue
            x, test = _standardise(
                past[list(FEATURES)].to_numpy(float), now[list(FEATURES)].to_numpy(float)
            )
            s = FIT[name](x, past["target"].to_numpy(float), test)
        out.append(pd.DataFrame({"month": month, "coin": now["coin"].to_numpy(), "score": s}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
