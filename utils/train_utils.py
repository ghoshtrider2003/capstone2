"""Training utilities: checkpointing, early stopping, and confusion matrix."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


@dataclass
class EarlyStopping:
    patience: int = 10
    min_delta: float = 0.0
    mode: str = "max"
    best_score: Optional[float] = None
    counter: int = 0
    should_stop: bool = False

    def step(self, current: float) -> bool:
        if self.best_score is None:
            self.best_score = current
            return False

        improvement = current - self.best_score if self.mode == "max" else self.best_score - current

        if improvement > self.min_delta:
            self.best_score = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def save_checkpoint(state: dict, checkpoint_path: str) -> None:
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import torch

    torch.save(state, path)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_path: str,
) -> None:
    """Plot and persist confusion matrix for publication-ready reporting."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
