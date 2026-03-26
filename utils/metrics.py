"""Evaluation utilities for HMSAN training and testing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


@dataclass
class MetricTracker:
    y_true: list[int]
    y_pred: list[int]

    def __init__(self) -> None:
        self.y_true = []
        self.y_pred = []

    def update(self, targets, predictions) -> None:
        self.y_true.extend(targets.tolist())
        self.y_pred.extend(predictions.tolist())

    def compute(self) -> Dict[str, float]:
        y_true = np.array(self.y_true)
        y_pred = np.array(self.y_pred)
        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }


def format_metrics(metrics: Dict[str, float]) -> str:
    return " | ".join(f"{key}: {value:.4f}" for key, value in metrics.items())
