"""The three networks. Each maps (tab, seq, mask) of B times by N instruments to (B, N).

- ``mlp`` reads each instrument's twelve features alone;
- ``tcn`` reads each instrument's last 60 bars of four series alone, through
  dilated causal convolutions, so no output can depend on a later bar;
- ``xs_transformer`` puts the same encoder under two layers of self-attention
  across the instruments of one time. Instruments carry no position: the
  layer sees a set, so a score depends on what the others did, not on which
  column they sit in. Masked instruments are neither attended to nor scored.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from trading_research.xs_deep.features import SERIES, TABULAR

KERNEL, DILATIONS = 3, (1, 2, 4, 8)


class CausalConv(nn.Module):
    """A dilated causal convolution over (M, L, C), written as taps and one linear map.

    The same function as ``nn.Conv1d`` with left padding; gathering the taps
    and multiplying once runs about ten times faster on this CPU than the
    library's dilated convolution.
    """

    def __init__(self, c_in: int, c_out: int, dilation: int) -> None:
        super().__init__()
        self.dilation = dilation
        self.linear = nn.Linear(c_in * KERNEL, c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = (KERNEL - 1) * self.dilation
        length = x.shape[1]
        xp = nn.functional.pad(x, (0, 0, pad, 0))
        taps = [xp[:, k * self.dilation : k * self.dilation + length] for k in range(KERNEL)]
        return torch.as_tensor(self.linear(torch.cat(taps, dim=-1)))


class CausalBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, dilation: int) -> None:
        super().__init__()
        self.conv1 = CausalConv(c_in, c_out, dilation)
        self.conv2 = CausalConv(c_out, c_out, dilation)
        self.skip = nn.Linear(c_in, c_out) if c_in != c_out else nn.Identity()
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.conv2(self.act(self.conv1(x))))
        return torch.as_tensor(self.act(h + self.skip(x)))


class Encoder(nn.Module):
    """(M, WINDOW, 4) -> (M, width): the causal stack's output at the last bar."""

    def __init__(self, width: int) -> None:
        super().__init__()
        chans = [len(SERIES)] + [width] * len(DILATIONS)
        self.blocks = nn.Sequential(
            *[CausalBlock(chans[i], chans[i + 1], d) for i, d in enumerate(DILATIONS)]
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(self.blocks(seq)[:, -1])


class MLP(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        hidden = 2 * width
        self.net = nn.Sequential(
            nn.Linear(len(TABULAR), hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )  # fmt: skip

    def forward(self, tab: torch.Tensor, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        return torch.as_tensor(self.net(tab).squeeze(-1))


class TCN(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.encoder = Encoder(width)
        self.head = nn.Linear(width, 1)

    def forward(self, tab: torch.Tensor, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        b, n, w, c = seq.shape
        h = self.encoder(seq.reshape(b * n, w, c))
        return torch.as_tensor(self.head(h).reshape(b, n))


class XSTransformer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.encoder = Encoder(width)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=4, dim_feedforward=2 * width, dropout=0.1, batch_first=True
        )
        self.mix = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.head = nn.Linear(width, 1)

    def forward(self, tab: torch.Tensor, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        b, n, w, c = seq.shape
        h = self.encoder(seq.reshape(b * n, w, c)).reshape(b, n, -1)
        # A time with no usable instrument would attend to nothing: let it
        # attend to everything; its scores are masked out downstream anyway.
        pad = ~mask & mask.any(dim=1, keepdim=True)
        h = self.mix(h, src_key_padding_mask=pad)
        return torch.as_tensor(self.head(h).squeeze(-1))


def build(name: str, width: int) -> Any:
    torch.manual_seed(0)
    return {"mlp": MLP, "tcn": TCN, "xs_transformer": XSTransformer}[name](width)
