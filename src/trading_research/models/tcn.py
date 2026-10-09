"""A temporal convolutional network over feature windows.

The other models see one row at a time: whatever the feature engineering
managed to compress into that row is all they get. A sequence model sees the
last *L* rows and can find shape — an imbalance that has been building versus
one that just appeared — which no amount of rolling statistics fully captures.

Why a TCN rather than an LSTM: dilated causal convolutions reach a long window
with a shallow stack, they train in parallel across time rather than
sequentially, and causality is structural rather than something to be careful
about. Each layer's kernel only ever reaches backwards, so the network cannot
read the future even if the data handed to it were misaligned. On a problem
where look-ahead is the main hazard, an architecture that makes it impossible
is worth more than one that merely permits avoiding it.

Expectations, stated in advance
-------------------------------
Given what the simpler models found, this is unlikely to change the
conclusion. The expected edge per trade is roughly the information coefficient
times the volatility of the move, and that product sits near 0.4 bp at every
horizon while a round trip costs 11. A sequence model can raise the
coefficient; it cannot raise it by a factor of twenty-five. It is here because
"we tried a sequence model" is a question a reader will ask, and because a
negative result from a fair attempt is worth more than an untested assumption.

One process per model
---------------------
Do not fit this and XGBoost in the same interpreter. Both ship their own
OpenMP runtime, and two of them in one macOS process do not coexist: measured
here, the network fits 40k rows in 20 seconds on its own and never returns once
XGBoost has run in the same process — not slower, stuck.

The staged pipeline avoids this by construction, since ``train --model xgboost``
and ``train --model tcn`` are separate invocations. It is worth knowing before
writing a script that loops over models in one process, which is the obvious
thing to write and the one that hangs.

The Gaussian head
-----------------
:class:`GaussianTCN` puts the same causal stack under a regression head: an
expected move in basis points and the log of its variance, trained by Gaussian
negative log-likelihood. The variance is what a gate of the form
``mu - k * sigma > cost`` needs, and it comes from the same network that made
the prediction rather than from a second model fitted on its residuals. Its
windows come from :mod:`trading_research.models.windows`, gathered per batch,
and its checkpoint is read on an inner validation the caller supplies, never on
a slice of the training rows.

PyTorch is an optional dependency — ``uv sync --extra deep``.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_research.models.base import CLASSES, Model
from trading_research.models.regression import edge_to_proba
from trading_research.models.windows import WINDOW_ROWS, Day, WindowDataset


class TCNBaseline(Model):
    """Causal dilated convolutional classifier over a window of past rows.

    ``window`` is how many past observations each prediction sees. It is the
    parameter that matters most: too short and the model has no more context
    than the tabular baselines, too long and it spends its capacity on a
    sample that cannot support it.
    """

    def __init__(
        self,
        *,
        window: int = 64,
        channels: int = 32,
        levels: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        batch_size: int = 512,
        max_epochs: int = 20,
        patience: int = 3,
        validation_fraction: float = 0.15,
        balance_classes: bool = True,
        seed: int = 42,
        device: str | None = None,
    ) -> None:
        super().__init__(name="tcn")
        self.window = window
        self.channels = channels
        self.levels = levels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.balance_classes = balance_classes
        self.seed = seed
        self.device = device

        self.net_: Any = None
        self.columns_: list[str] | None = None
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.classes_: np.ndarray | None = None
        self.history_: list[dict[str, float]] = []

    # ------------------------------------------------------------------
    def fit(self, x: pd.DataFrame, y: pd.Series) -> TCNBaseline:
        torch, nn = _import_torch()
        torch.manual_seed(self.seed)

        self.columns_ = list(x.columns)
        values = x.to_numpy(dtype="float32")

        # Standardised here, inside fit, for the same reason the linear model
        # does it: statistics computed over the whole sample carry the test
        # period into training, and nothing about the result looks wrong.
        self.mean_ = values.mean(axis=0)
        self.scale_ = np.where(values.std(axis=0) > 0, values.std(axis=0), 1.0)
        values = (values - self.mean_) / self.scale_

        labels = y.to_numpy().astype(int)
        self.classes_ = np.array(sorted(set(labels)), dtype=int)
        encoded = np.searchsorted(self.classes_, labels)

        windows, targets = _windowise(values, encoded, self.window)
        if len(windows) < 100:
            raise ValueError(f"only {len(windows)} windows of length {self.window}; need more rows")

        # The validation split is the *tail* of the training block, not a
        # random sample. Random validation on a time series puts neighbouring,
        # near-identical rows on both sides and reports an optimistic loss that
        # early stopping then trusts.
        split = int(len(windows) * (1 - self.validation_fraction))
        train_x, val_x = windows[:split], windows[split:]
        train_y, val_y = targets[:split], targets[split:]

        device = torch.device(
            self.device or ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        self.net_ = _build_net(
            n_features=values.shape[1],
            n_classes=len(self.classes_),
            channels=self.channels,
            levels=self.levels,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
        ).to(device)

        weight = None
        if self.balance_classes:
            counts = np.bincount(train_y, minlength=len(self.classes_)).astype("float32")
            counts[counts == 0] = 1.0
            weight = torch.tensor(len(train_y) / (len(counts) * counts), device=device)

        loss_fn = nn.CrossEntropyLoss(weight=weight)
        optimiser = torch.optim.Adam(self.net_.parameters(), lr=self.learning_rate)

        train_tensor = torch.from_numpy(train_x)
        train_target = torch.from_numpy(train_y).long()
        val_tensor = torch.from_numpy(val_x).to(device)
        val_target = torch.from_numpy(val_y).long().to(device)

        best_loss = float("inf")
        best_state: dict[str, Any] | None = None
        bad_epochs = 0

        for epoch in range(self.max_epochs):
            self.net_.train()
            order = torch.randperm(len(train_tensor))
            total = 0.0
            for start in range(0, len(order), self.batch_size):
                idx = order[start : start + self.batch_size]
                batch = train_tensor[idx].to(device)
                target = train_target[idx].to(device)
                optimiser.zero_grad()
                loss = loss_fn(self.net_(batch), target)
                loss.backward()
                # Gradient clipping: the class imbalance here is extreme, so a
                # batch that happens to contain several rare-class examples can
                # otherwise produce a step that undoes an epoch of progress.
                torch.nn.utils.clip_grad_norm_(self.net_.parameters(), 1.0)
                optimiser.step()
                total += float(loss.detach()) * len(idx)

            self.net_.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(self.net_(val_tensor), val_target))
            self.history_.append(
                {"epoch": epoch, "train_loss": total / max(len(order), 1), "val_loss": val_loss}
            )

            if val_loss < best_loss - 1e-5:
                best_loss = val_loss
                best_state = {k: v.detach().clone() for k, v in self.net_.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break

        if best_state is not None:
            self.net_.load_state_dict(best_state)
        self.fitted_ = True
        return self

    # ------------------------------------------------------------------
    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        torch, _ = _import_torch()
        assert self.net_ is not None and self.classes_ is not None
        assert self.mean_ is not None and self.scale_ is not None

        if self.columns_ is not None and list(x.columns) != self.columns_:
            raise ValueError(f"{self.name}: feature columns differ from those seen in fit")

        values = (x.to_numpy(dtype="float32") - self.mean_) / self.scale_
        windows, _ = _windowise(values, np.zeros(len(values), dtype=int), self.window)

        device = next(self.net_.parameters()).device
        self.net_.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, len(windows), 4096):
                batch = torch.from_numpy(windows[start : start + 4096]).to(device)
                chunks.append(torch.softmax(self.net_(batch), dim=1).cpu().numpy())
        scored = np.concatenate(chunks) if chunks else np.zeros((0, len(self.classes_)))

        # The first `window - 1` rows have no full history. They are given the
        # neutral class rather than dropped, so the output stays aligned with
        # the input and the caller's row indexing keeps working.
        out = np.zeros((len(x), len(CLASSES)))
        out[:, CLASSES.index(0)] = 1.0
        lookup = {int(c): i for i, c in enumerate(self.classes_)}
        for target, cls in enumerate(CLASSES):
            source = lookup.get(int(cls))
            if source is None:
                continue
            out[self.window - 1 :, target] = scored[:, source]
        return out

    def describe(self) -> dict[str, Any]:
        out = super().describe()
        out.update(
            {
                "window": self.window,
                "channels": self.channels,
                "levels": self.levels,
                "kernel_size": self.kernel_size,
                "dropout": self.dropout,
                "epochs_run": len(self.history_),
                "best_val_loss": min((h["val_loss"] for h in self.history_), default=None),
            }
        )
        return out


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _import_torch() -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - exercised only without torch
        raise ImportError(
            "the TCN needs PyTorch, which is an optional dependency. "
            "Install it with: uv sync --extra deep"
        ) from exc
    return torch, nn


def _windowise(
    values: np.ndarray, targets: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``(n, features, window)`` sequences ending at each row.

    Built as a strided view rather than by copying: a 500k-row, 40-feature,
    64-step dataset is 5 GB copied and nothing as a view. The copy happens per
    batch instead, where it is small.

    Each window *ends* at its row, so the sequence for row *t* covers
    *t-window+1 … t* and never reaches past *t*.
    """
    n, features = values.shape
    if n < window:
        return np.zeros((0, features, window), dtype="float32"), np.zeros(0, dtype=int)

    strided = np.lib.stride_tricks.sliding_window_view(values, window, axis=0)
    # sliding_window_view gives (n - window + 1, features, window) already in
    # the layout a Conv1d wants: (batch, channels, time).
    return np.ascontiguousarray(strided, dtype="float32"), targets[window - 1 :]


def _build_net(
    *,
    n_features: int,
    n_classes: int,
    channels: int,
    levels: int,
    kernel_size: int,
    dropout: float,
) -> Any:
    """Build the network.

    Everything is defined inside the function so that importing this module
    does not require torch — the tabular models must stay usable without a deep
    learning stack installed.
    """
    _, nn = _import_torch()

    class Chomp(nn.Module):
        """Trim the padding a causal convolution adds on the right.

        Conv1d pads both ends. The left padding is what makes the layer causal;
        the right padding would let the kernel see later steps, so it is cut
        off. Without this the network reads the future — and would look
        excellent doing it.
        """

        def __init__(self, size: int) -> None:
            super().__init__()
            self.size = size

        def forward(self, x: Any) -> Any:
            return x[:, :, : -self.size] if self.size > 0 else x

    class Block(nn.Module):
        """One residual block of two dilated causal convolutions."""

        def __init__(self, n_in: int, n_out: int, kernel: int, dilation: int, drop: float) -> None:
            super().__init__()
            pad = (kernel - 1) * dilation
            self.body = nn.Sequential(
                nn.utils.parametrizations.weight_norm(
                    nn.Conv1d(n_in, n_out, kernel, padding=pad, dilation=dilation)
                ),
                Chomp(pad),
                nn.ReLU(),
                nn.Dropout(drop),
                nn.utils.parametrizations.weight_norm(
                    nn.Conv1d(n_out, n_out, kernel, padding=pad, dilation=dilation)
                ),
                Chomp(pad),
                nn.ReLU(),
                nn.Dropout(drop),
            )
            self.shortcut = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
            self.relu = nn.ReLU()

        def forward(self, x: Any) -> Any:
            residual = x if self.shortcut is None else self.shortcut(x)
            return self.relu(self.body(x) + residual)

    class Net(nn.Module):
        """Dilated causal stack, classified at the final timestep."""

        def __init__(self) -> None:
            super().__init__()
            self.body = nn.Sequential(
                *[
                    Block(
                        n_in=n_features if level == 0 else channels,
                        n_out=channels,
                        kernel=kernel_size,
                        # Doubling the dilation each level: the receptive field
                        # grows exponentially, so a handful of layers covers the
                        # whole window.
                        dilation=2**level,
                        drop=dropout,
                    )
                    for level in range(levels)
                ]
            )
            self.head = nn.Linear(channels, n_classes)

        def forward(self, x: Any) -> Any:
            # Only the last timestep is classified: it is the one the prediction
            # is for, and the earlier positions exist to give it context.
            return self.head(self.body(x)[:, :, -1])

    return Net()


# ---------------------------------------------------------------------------
# The Gaussian head: helpers
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> np.random.Generator:
    """Seed every generator a fit draws from, and return the one for shuffles.

    Python's and torch's global generators are seeded (torch's for every
    device, so dropout on MPS is seeded too); the shuffles draw from the
    returned numpy generator alone, so nothing else in the process moves them.
    With the network built on the CPU after this call, two fits with one seed on
    one device start from the same weights and see the same batches.
    """
    torch, _ = _import_torch()
    random.seed(seed)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)


def gaussian_nll_loss(
    mu: Any, log_variance: Any, target: Any, *, log_v: float, prior_weight: float = 0.01
) -> Any:
    """Gaussian negative log-likelihood plus a prior pulling log-variance to ``log v``.

    ``mean(0.5 * (log s2 + (y - mu)^2 / s2)) + prior_weight * mean((log s2 - log v)^2)``,
    with ``v`` the variance of the fit's training targets. The constant
    ``0.5 * log(2 pi)`` is left out; it moves no gradient. Without the prior a
    network can buy a lower loss on a few rows by shrinking their variance
    towards zero, which is exactly the overconfidence a gate on ``sigma``
    cannot afford.
    """
    torch, _ = _import_torch()
    nll = 0.5 * (log_variance + (target - mu) ** 2 * torch.exp(-log_variance))
    return nll.mean() + prior_weight * ((log_variance - log_v) ** 2).mean()


def rank_ic(prediction: np.ndarray, outcome: np.ndarray) -> float:
    """Spearman correlation over the rows where both are finite; NaN if undefined."""
    prediction = np.asarray(prediction, dtype=np.float64)
    outcome = np.asarray(outcome, dtype=np.float64)
    ok = np.isfinite(prediction) & np.isfinite(outcome)
    if ok.sum() < 3 or np.ptp(prediction[ok]) == 0 or np.ptp(outcome[ok]) == 0:
        return float("nan")
    return float(pd.Series(prediction[ok]).corr(pd.Series(outcome[ok]), method="spearman"))


def _input_moments(data: WindowDataset) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation of the inputs at the window ends, streamed by day.

    Merged day by day with Chan's pairwise update rather than as
    ``E[x^2] - E[x]^2``, which loses its digits on inputs far from zero.
    """
    count, mean, m2 = 0, np.zeros(0), np.zeros(0)
    for i, day in enumerate(data.days):
        values, offset = data.extended(i)
        ends = np.asarray(day.rows)[data.complete(i, values, offset)] + offset
        rows = np.asarray(values[ends], dtype=np.float64)
        if not len(rows):
            continue
        n_b, mean_b = len(rows), rows.mean(axis=0)
        m2_b = ((rows - mean_b) ** 2).sum(axis=0)
        if count == 0:
            count, mean, m2 = n_b, mean_b, m2_b
            continue
        delta, total = mean_b - mean, count + n_b
        mean = mean + delta * n_b / total
        m2 = m2 + m2_b + delta**2 * count * n_b / total
        count = total
    if count == 0:
        raise ValueError("no training row has a complete window")
    std = np.sqrt(m2 / count)
    return mean, np.where(np.isfinite(std) & (std > 0), std, 1.0)


def _build_gaussian_net(
    *,
    n_features: int,
    channels: int,
    levels: int,
    kernel_size: int,
    dropout: float,
    mean: np.ndarray,
    scale: np.ndarray,
    log_v: float,
    band: float,
) -> Any:
    """The causal stack of :func:`_build_net` with a head for ``mu`` and ``log s2``.

    The input standardisation, the head's scale (``sqrt(v)`` for ``mu``, an
    offset of ``log v`` for the log-variance) and the clamp of the log-variance
    to ``log v +- band`` are buffers of the module, so the weights file and an
    exported graph carry them and read raw store rows.
    """
    torch, nn = _import_torch()
    core = _build_net(
        n_features=n_features,
        n_classes=2,
        channels=channels,
        levels=levels,
        kernel_size=kernel_size,
        dropout=dropout,
    )

    class GaussianNet(nn.Module):
        """Raw windows in, ``(mu, log s2)`` in basis points out."""

        def __init__(self) -> None:
            super().__init__()
            self.core = core
            # Starting at the prior: mu = 0 and s2 = v until the data says otherwise.
            nn.init.zeros_(self.core.head.weight)
            nn.init.zeros_(self.core.head.bias)
            for name, value in (("in_mean", mean), ("in_scale", scale)):
                buffer = torch.tensor(np.asarray(value, dtype=np.float32)).reshape(1, -1, 1)
                self.register_buffer(name, buffer)
            self.register_buffer("log_v", torch.tensor(log_v, dtype=torch.float32))
            self.register_buffer("band", torch.tensor(band, dtype=torch.float32))

        def forward(self, x: Any) -> tuple[Any, Any]:
            raw = self.core((x - self.in_mean) / self.in_scale)
            mu = raw[:, 0] * torch.exp(0.5 * self.log_v)
            log_var = torch.minimum(
                torch.maximum(self.log_v + raw[:, 1], self.log_v - self.band),
                self.log_v + self.band,
            )
            return mu, log_var

    return GaussianNet()


# ---------------------------------------------------------------------------
# The Gaussian head: the model
# ---------------------------------------------------------------------------


class GaussianTCN(Model):
    """Causal dilated network predicting a move in bp and the log of its variance.

    The architecture and the optimiser settings are :class:`TCNBaseline`'s
    defaults; only the head and the loss differ. ``prior_weight`` and
    ``log_variance_band`` are the loss's pull of ``log s2`` towards ``log v`` and
    the clamp around it, ``v`` being the variance of the fit's training targets.

    The checkpoint and early stopping read the rank correlation of ``mu`` with
    the validation days' targets, which the caller sets to the realised forward
    move whatever the network was trained on. There is no internal split. With
    ``epochs`` set, exactly that many epochs are trained and the last is kept:
    with the same seed, data and device this reproduces the weights of a fit
    whose checkpoint was that epoch, without its validation days.
    """

    def __init__(
        self,
        *,
        window: int = WINDOW_ROWS,
        channels: int = 32,
        levels: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        batch_size: int = 512,
        max_epochs: int = 20,
        patience: int = 3,
        prior_weight: float = 0.01,
        log_variance_band: float = 6.0,
        days_per_shard: int = 4,
        epochs: int | None = None,
        scale_bp: float = 11.0,
        seed: int = 42,
        device: str | None = None,
    ) -> None:
        super().__init__(name="tcn_gaussian")
        if epochs is not None and epochs < 1:
            raise ValueError(f"epochs must be at least 1, got {epochs}")
        self.window, self.channels, self.levels = window, channels, levels
        self.kernel_size, self.dropout = kernel_size, dropout
        self.learning_rate, self.batch_size = learning_rate, batch_size
        self.max_epochs, self.patience, self.epochs = max_epochs, patience, epochs
        self.prior_weight, self.log_variance_band = prior_weight, log_variance_band
        self.days_per_shard, self.scale_bp = days_per_shard, scale_bp
        self.seed, self.device = seed, device

        self.net_: Any = None
        self.columns_: list[str] | None = None
        self.n_features_: int | None = None
        self.target_variance_: float | None = None
        self.checkpoint_epoch_: int | None = None
        self.history_: list[dict[str, float]] = []

    # ------------------------------------------------------------------
    def fit(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        *,
        validation: tuple[pd.DataFrame, pd.Series] | None = None,
    ) -> GaussianTCN:
        """Fit on one contiguous block of every-row inputs; NaN targets are not trained on.

        ``x`` must hold consecutive 100 ms rows, as at inference, or the windows
        will not be the ones the network is later asked about.
        """
        self.columns_ = list(x.columns)
        held = None if validation is None else [self._frame_day(*validation)]
        return self.fit_days([self._frame_day(x, y)], held)

    def fit_days(
        self, train: Sequence[Day], validation: Sequence[Day] | None = None
    ) -> GaussianTCN:
        """Fit on days streamed from the every-row store; the study's path."""
        torch, _ = _import_torch()
        if validation is None and self.epochs is None:
            raise ValueError("the checkpoint is read on validation days: pass them, or set epochs")
        rng = seed_everything(self.seed)
        data = WindowDataset(train, window=self.window)
        given = [np.asarray(d.targets, dtype=np.float64) for d in train if d.targets is not None]
        targets = np.concatenate(given) if given else np.zeros(0)
        targets = targets[np.isfinite(targets)]
        if len(targets) < 100 or not np.var(targets) > 0:
            raise ValueError(f"only {len(targets)} usable training targets, or no variance")
        self.target_variance_ = float(np.var(targets))
        log_v = math.log(self.target_variance_)
        mean, scale = _input_moments(data)
        self.n_features_ = len(mean)

        device = torch.device(
            self.device or ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        net = _build_gaussian_net(
            n_features=self.n_features_,
            channels=self.channels,
            levels=self.levels,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            mean=mean,
            scale=scale,
            log_v=log_v,
            band=self.log_variance_band,
        )
        self.net_ = net.to(device)
        optimiser = torch.optim.Adam(self.net_.parameters(), lr=self.learning_rate)
        self.history_ = []
        best_score, best_state, bad_epochs = -math.inf, None, 0
        for epoch in range(1, (self.epochs or self.max_epochs) + 1):
            self.net_.train()
            total, count = 0.0, 0
            for windows, target in data.training_batches(self.batch_size, rng, self.days_per_shard):
                yb = torch.from_numpy(target).to(device)
                optimiser.zero_grad()
                mu, log_var = self.net_(torch.from_numpy(windows).to(device))
                loss = gaussian_nll_loss(
                    mu, log_var, yb, log_v=log_v, prior_weight=self.prior_weight
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net_.parameters(), 1.0)
                optimiser.step()
                total, count = total + float(loss.detach()) * len(target), count + len(target)
            record = {"epoch": float(epoch), "train_loss": total / max(count, 1)}
            if validation is not None:
                record["validation_ic"] = self._validation_ic(validation)
            self.history_.append(record)
            if self.epochs is not None:
                continue
            score = record["validation_ic"]
            score = score if math.isfinite(score) else -math.inf
            if best_state is None or score > best_score:
                best_score, bad_epochs, self.checkpoint_epoch_ = score, 0, epoch
                best_state = {k: v.detach().clone() for k, v in self.net_.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break
        if best_state is not None:
            self.net_.load_state_dict(best_state)
        else:
            self.checkpoint_epoch_ = len(self.history_)
        self.fitted_ = True
        return self

    # ------------------------------------------------------------------
    def _frame_day(self, x: pd.DataFrame, y: pd.Series | None = None) -> Day:
        if self.columns_ is not None and list(x.columns) != self.columns_:
            raise ValueError(f"{self.name}: feature columns differ from those seen in fit")
        values = x.to_numpy(dtype="float32")
        targets = None if y is None else y.to_numpy(dtype="float32")
        return Day("frame", lambda: values, np.arange(len(values)), targets)

    def _validation_ic(self, days: Sequence[Day]) -> float:
        predicted = self._predict_days(days)
        mu = np.concatenate([p[0] for p in predicted])
        outcome = np.concatenate(
            [np.asarray(d.targets, dtype=np.float64) for d in days if d.targets is not None]
        )
        if len(outcome) != len(mu):
            raise ValueError("every validation day needs its realised forward move as targets")
        return rank_ic(mu, outcome)

    def _predict_days(
        self, days: Sequence[Day], batch_size: int = 4096
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        torch, _ = _import_torch()
        out = [(np.full(len(d.rows), np.nan), np.full(len(d.rows), np.nan)) for d in days]
        device = next(self.net_.parameters()).device
        self.net_.eval()
        with torch.no_grad():
            for i, positions, windows in WindowDataset(days, window=self.window).inference_batches(
                batch_size
            ):
                if windows.shape[1] != self.n_features_:
                    raise ValueError(
                        f"{days[i].key}: {windows.shape[1]} inputs, fitted on {self.n_features_}"
                    )
                mu, log_var = self.net_(torch.from_numpy(windows).to(device))
                out[i][0][positions] = mu.cpu().numpy()
                out[i][1][positions] = log_var.cpu().numpy()
        return out

    def predict_days(
        self, days: Sequence[Day], batch_size: int = 4096
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """``(mu_bp, log_variance)`` per day, aligned with each day's ``rows``.

        A row without a complete window (a cold-start day's first rows, or a
        window holding a non-finite input) gets NaN: it is unscored, not guessed.
        """
        self._check_fitted()
        return self._predict_days(days, batch_size)

    def predict_gaussian(self, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """``mu`` and ``sigma`` in bp for every row of a contiguous frame."""
        self._check_fitted()
        [(mu, log_var)] = self._predict_days([self._frame_day(x)])
        return mu, np.exp(0.5 * log_var)

    def predict_mean_bp(self, x: pd.DataFrame) -> np.ndarray:
        return self.predict_gaussian(x)[0]

    def predict_sigma_bp(self, x: pd.DataFrame) -> np.ndarray:
        return self.predict_gaussian(x)[1]

    def predict_edge_bp(self, x: pd.DataFrame) -> np.ndarray:
        return self.predict_mean_bp(x)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """The expected move mapped onto probabilities as the regression models do it.

        Rows without a window are given the neutral class, so the output stays
        aligned with the input.
        """
        mu = self.predict_mean_bp(x)
        out = edge_to_proba(mu, self.scale_bp)
        out[~np.isfinite(mu)] = [1.0 if c == 0 else 0.0 for c in CLASSES]
        return out

    # ------------------------------------------------------------------
    def weights_sha256(self) -> str:
        """sha256 over the network's state, buffers included, in name order."""
        self._check_fitted()
        digest = hashlib.sha256()
        for name, tensor in sorted(self.net_.state_dict().items()):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def save(self, path: Path | str) -> Path:
        """Write the weights with what is needed to rebuild the network."""
        self._check_fitted()
        torch, _ = _import_torch()
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        state = {k: v.detach().cpu() for k, v in self.net_.state_dict().items()}
        torch.save({"describe": self.describe(), "columns": self.columns_, "state": state}, out)
        return out

    @classmethod
    def load(cls, path: Path | str, *, device: str | None = None) -> GaussianTCN:
        """Rebuild a saved network for inference."""
        torch, _ = _import_torch()
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        info = payload["describe"]
        keys = (
            *("window", "channels", "levels", "kernel_size", "dropout"),
            *("prior_weight", "log_variance_band", "scale_bp", "seed"),
        )
        model = cls(**{k: info[k] for k in keys}, device=device)
        state = payload["state"]
        model.n_features_ = int(state["in_mean"].shape[1])
        model.columns_ = payload["columns"]
        model.target_variance_ = info["target_variance"]
        model.checkpoint_epoch_ = info["checkpoint_epoch"]
        net = _build_gaussian_net(
            n_features=model.n_features_,
            channels=model.channels,
            levels=model.levels,
            kernel_size=model.kernel_size,
            dropout=model.dropout,
            mean=np.zeros(model.n_features_),
            scale=np.ones(model.n_features_),
            log_v=0.0,
            band=model.log_variance_band,
        )
        net.load_state_dict(state)
        model.net_ = net.to(torch.device(device or "cpu"))
        model.fitted_ = True
        return model

    def describe(self) -> dict[str, Any]:
        out = super().describe()
        out.update(
            {
                "window": self.window,
                "channels": self.channels,
                "levels": self.levels,
                "kernel_size": self.kernel_size,
                "dropout": self.dropout,
                "prior_weight": self.prior_weight,
                "log_variance_band": self.log_variance_band,
                "scale_bp": self.scale_bp,
                "seed": self.seed,
                "target_variance": self.target_variance_,
                "checkpoint_epoch": self.checkpoint_epoch_,
                "epochs_run": len(self.history_),
            }
        )
        return out
