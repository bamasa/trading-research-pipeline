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

PyTorch is an optional dependency — ``uv sync --extra deep``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trading_research.models.base import CLASSES, Model


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
