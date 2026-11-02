"""The original's network: five residual blocks of 128 channels over 50 rows.

Re-typed from the author's 2025 training notebook. Each block is two
weight-normalised causal convolutions (kernel 3, dilation 1, 2, 4, 8, 16), each
followed by ReLU and channel dropout at 0.5, with a 1x1 convolution on the
residual path where the width changes and a ReLU after the sum. The head reads
the last step and the mean over the window, concatenated, through one linear
layer to three logits (down, flat, up).

The original's initialiser set every convolution's bias to zero and drew the
1x1 convolution from a Kaiming normal; its Kaiming draw on the weight-normalised
convolutions changed a tensor recomputed on every forward pass and so had no
effect. Those keep PyTorch's default initialisation here too.

The repository's :mod:`trading_research.models.tcn` differs in two places that
matter (element-wise dropout, a head on the last step only), so the network is
built here; its windows come from :func:`trading_research.models.windows.
gather_windows`.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

CHANNELS, LEVELS, KERNEL, DROPOUT, WINDOW, CLASSES = 128, 5, 3, 0.5, 50, 3


class Chomp(nn.Module):
    """Cut the right padding, so each step sees only itself and the past."""

    def __init__(self, size: int) -> None:
        super().__init__()
        self.size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.size] if self.size > 0 else x


class Block(nn.Module):
    def __init__(self, n_in: int, n_out: int, dilation: int, dropout: float) -> None:
        super().__init__()
        pad = (KERNEL - 1) * dilation
        wn = nn.utils.parametrizations.weight_norm
        self.body = nn.Sequential(
            wn(nn.Conv1d(n_in, n_out, KERNEL, padding=pad, dilation=dilation)),
            Chomp(pad),
            nn.ReLU(),
            nn.Dropout1d(dropout),
            wn(nn.Conv1d(n_out, n_out, KERNEL, padding=pad, dilation=dilation)),
            Chomp(pad),
            nn.ReLU(),
            nn.Dropout1d(dropout),
        )
        self.shortcut = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
        for module in self.modules():
            if isinstance(module, nn.Conv1d) and module.bias is not None:
                nn.init.zeros_(module.bias)
        if self.shortcut is not None:
            nn.init.kaiming_normal_(self.shortcut.weight, nonlinearity="relu")
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.shortcut is None else self.shortcut(x)
        return self.relu(self.body(x) + residual)


class DeepTCN(nn.Module):
    """``(batch, inputs, steps)`` to ``(batch, 3)`` logits."""

    def __init__(self, n_inputs: int, dropout: float = DROPOUT) -> None:
        super().__init__()
        blocks = [
            Block(n_inputs if level == 0 else CHANNELS, CHANNELS, 2**level, dropout)
            for level in range(LEVELS)
        ]
        self.tcn = nn.Sequential(*blocks)
        self.fc = nn.Linear(2 * CHANNELS, CLASSES)

    def forward(self, x: torch.Tensor) -> Any:
        y = self.tcn(x)
        return self.fc(torch.cat([y[:, :, -1], y.mean(dim=2)], dim=1))
