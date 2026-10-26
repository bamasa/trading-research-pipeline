"""The original's training loop and inference, over windows of concatenated days.

As in the author's 2025 notebook: windows of 50 rows end at every row of a
split (the split's days concatenated, its first 49 rows ending no window); a
window's target is its last row's 5 s label, or ignored (-100) where that row
fails the volatility filter; batches of 128 in a fresh random order each epoch,
a batch with no target skipped; cross-entropy with inverse-frequency class
weights (normalised to mean one) and label smoothing 0.05; AdamW at 3e-5 with
weight decay 1e-5; gradients clipped at norm 1; the learning rate halved after
two epochs without a lower validation loss; five epochs; the weights kept from
the epoch with the best validation accuracy on the targeted rows.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import torch
from torch import nn

from trading_research.models.tcn import seed_everything
from trading_research.models.windows import gather_windows
from trading_research.replication_eth.net import WINDOW, DeepTCN

IGNORE = -100
BATCH, EPOCHS, LR, WEIGHT_DECAY, SMOOTHING, CLIP_NORM = 128, 5, 3e-5, 1e-5, 0.05, 1.0


def targets(label: np.ndarray, volatility: np.ndarray, min_volatility: float) -> np.ndarray:
    """1 / 2 / 3 labels to 0 / 1 / 2 targets, ignored below the filter."""
    out = np.asarray(label, dtype=np.int64) - 1
    out[np.asarray(volatility) < min_volatility] = IGNORE
    return out


def class_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y[y != IGNORE], minlength=3).astype(np.float64).clip(min=1)
    w = counts.sum() / (3 * counts)
    return w / w.mean()


def _batches(x: np.ndarray, ends: np.ndarray, size: int) -> Iterator[tuple[np.ndarray, Any]]:
    for i in range(0, len(ends), size):
        chunk = ends[i : i + size]
        yield chunk, torch.from_numpy(gather_windows(x, chunk, WINDOW))


def evaluate(
    model: nn.Module, x: np.ndarray, y: np.ndarray, loss_fn: Any, device: str
) -> tuple[float, float]:
    """Mean batch loss and accuracy on targeted rows, as the original measured them."""
    model.eval()
    losses, right, total = [], 0, 0
    with torch.no_grad():
        for ends, xb in _batches(x, np.arange(WINDOW - 1, len(x)), BATCH):
            yb = torch.from_numpy(y[ends]).to(device)
            if not bool((yb != IGNORE).any()):
                continue
            logits = model(xb.to(device))
            losses.append(float(loss_fn(logits, yb)))
            keep = yb != IGNORE
            right += int((logits.argmax(1)[keep] == yb[keep]).sum())
            total += int(keep.sum())
    return float(np.mean(losses)), right / max(total, 1)


def fit(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    seed: int,
    device: str,
    log: Callable[[dict[str, float]], None] = print,
) -> tuple[DeepTCN, list[dict[str, float]]]:
    rng = seed_everything(seed)
    model = DeepTCN(x_train.shape[1]).to(device)
    weight = torch.tensor(class_weights(y_train[WINDOW - 1 :]), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(
        weight=weight.to(device), label_smoothing=SMOOTHING, ignore_index=IGNORE
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    best, best_state, history = -1.0, copy.deepcopy(model.state_dict()), []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for ends, xb in _batches(
            x_train, rng.permutation(np.arange(WINDOW - 1, len(x_train))), BATCH
        ):
            yb = torch.from_numpy(y_train[ends]).to(device)
            if not bool((yb != IGNORE).any()):
                continue
            optimizer.zero_grad()
            loss = loss_fn(model(xb.to(device)), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            optimizer.step()
            losses.append(float(loss.detach()))
        val_loss, val_acc = evaluate(model, x_val, y_val, loss_fn, device)
        plateau.step(val_loss)
        if val_acc > best:
            best, best_state = val_acc, copy.deepcopy(model.state_dict())
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "kept": float(val_acc >= best),
        }
        history.append(row)
        log(row)
    model.load_state_dict(best_state)
    return model, history


def predict(model: nn.Module, x: np.ndarray, device: str, batch: int = 4096) -> np.ndarray:
    """``(len(x), 3)`` softmax probabilities; the first 49 rows (no window) are NaN."""
    model.eval()
    out = np.full((len(x), 3), np.nan, dtype=np.float32)
    with torch.no_grad():
        for ends, xb in _batches(x, np.arange(WINDOW - 1, len(x)), batch):
            out[ends] = torch.softmax(model(xb.to(device)), dim=1).cpu().numpy()
    return out
