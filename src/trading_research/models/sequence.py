"""More sequence models: a transformer, an LSTM, a GRU and a window MLP.

Each one sees the same thing the TCN sees, the last ``window`` rows of
standardised features ending at the row it predicts, and is fitted by the same
loop (:class:`~trading_research.models.tcn.TCNBaseline`): standardisation inside
``fit``, the tail of the training block as validation, early stopping on its
loss, class weights against the imbalance. Only the network differs, so a
difference between them is a difference of architecture, not of how carefully
each was trained.

- :class:`TransformerClassifier`: a causal self-attention encoder. A learned
  position embedding, a mask that lets each step attend only to itself and the
  steps before it, and a head on the last step. Attention can weight any past
  row directly, where a convolution or a recurrence passes it along.
- :class:`LSTMClassifier` and :class:`GRUClassifier`: recurrent networks read
  forward through the window; the head reads the last hidden state. Causal by
  construction, since a recurrence only carries the past.
- :class:`WindowMLP`: the window flattened and passed through two dense layers.
  The control: whatever the sequence models find beyond it is what order and
  shape add over the same numbers seen at once.

The window ends at its row in every case (``tcn._windowise``), so none of them
can read past the row it predicts, whatever its attention or recurrence does.

PyTorch is an optional dependency, ``uv sync --extra deep``.
"""

from __future__ import annotations

from typing import Any

from trading_research.models.tcn import TCNBaseline, _import_torch


class _Sequence(TCNBaseline):
    """The TCN's fitting loop with another network; ``channels`` is the width."""

    model_name = "sequence"

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.name = self.model_name


class TransformerClassifier(_Sequence):
    """Causal self-attention encoder over a window of past rows.

    ``channels`` is the model width, ``levels`` the number of encoder layers,
    ``heads`` the attention heads (it must divide ``channels``).
    """

    model_name = "transformer"

    def __init__(self, *, heads: int = 4, levels: int = 2, **params: Any) -> None:
        super().__init__(levels=levels, **params)
        if self.channels % heads:
            raise ValueError(f"heads ({heads}) must divide channels ({self.channels})")
        self.heads = heads

    def _network(self, n_features: int, n_classes: int) -> Any:
        torch, nn = _import_torch()
        width, window, drop = self.channels, self.window, self.dropout
        layers, heads = self.levels, self.heads

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed = nn.Linear(n_features, width)
                self.position = nn.Parameter(torch.zeros(1, window, width))
                layer = nn.TransformerEncoderLayer(
                    width, heads, dim_feedforward=2 * width, dropout=drop, batch_first=True
                )
                self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
                self.norm = nn.LayerNorm(width)
                self.head = nn.Linear(width, n_classes)
                # True above the diagonal: step t never attends to a later step.
                mask = torch.triu(torch.ones(window, window, dtype=torch.bool), diagonal=1)
                self.register_buffer("mask", mask)

            def forward(self, x: Any) -> Any:
                h = self.embed(x.transpose(1, 2)) + self.position  # (batch, window, width)
                h = self.encoder(h, mask=self.mask)
                return self.head(self.norm(h[:, -1]))

        return Net()

    def describe(self) -> dict[str, Any]:
        out = super().describe()
        out["heads"] = self.heads
        return out


class _Recurrent(_Sequence):
    cell = "LSTM"

    def __init__(self, *, levels: int = 1, **params: Any) -> None:
        super().__init__(levels=levels, **params)

    def _network(self, n_features: int, n_classes: int) -> Any:
        _, nn = _import_torch()
        width, layers, drop, cell = self.channels, self.levels, self.dropout, self.cell

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rnn = getattr(nn, cell)(
                    n_features, width, num_layers=layers, batch_first=True,
                    dropout=drop if layers > 1 else 0.0,
                )  # fmt: skip
                self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(width, n_classes))

            def forward(self, x: Any) -> Any:
                out, _ = self.rnn(x.transpose(1, 2))
                return self.head(out[:, -1])

        return Net()


class LSTMClassifier(_Recurrent):
    """A long short-term memory network read forward through the window."""

    model_name, cell = "lstm", "LSTM"


class GRUClassifier(_Recurrent):
    """A gated recurrent unit network read forward through the window."""

    model_name, cell = "gru", "GRU"


class WindowMLP(_Sequence):
    """The window flattened into one vector, two dense layers: the order-blind control."""

    model_name = "mlp"

    def _network(self, n_features: int, n_classes: int) -> Any:
        _, nn = _import_torch()
        width, drop = 4 * self.channels, self.dropout
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_features * self.window, width), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(width, width), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(width, n_classes),
        )  # fmt: skip


__all__ = ["GRUClassifier", "LSTMClassifier", "TransformerClassifier", "WindowMLP"]
