from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(model: Any, x: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    start = time.perf_counter()
    predictions = model.predict(x)
    predict_ms_per_item = (time.perf_counter() - start) * 1000 / max(len(x), 1)
    score = None
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(x)[:, 1]
    elif hasattr(model, "decision_function"):
        score = model.decision_function(x)
    return metrics_from_predictions(y, predictions, score, predict_ms_per_item)


def metrics_from_predictions(
    y: pd.Series,
    predictions: np.ndarray,
    scores: np.ndarray | None,
    predict_ms_per_item: float,
) -> dict[str, Any]:
    try:
        auc = float(roc_auc_score(y, scores)) if scores is not None and len(np.unique(y)) == 2 else None
    except ValueError:
        auc = None
    matrix = confusion_matrix(y, predictions, labels=[0, 1]).tolist()
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": auc,
        "confusion_matrix": matrix,
        "prediction_ms_per_item": float(predict_ms_per_item),
    }


def save_confusion_matrix(matrix: list[list[int]], title: str, destination: Path) -> None:
    figure, axis = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["authentic", "tampered"],
        yticklabels=["authentic", "tampered"],
        ax=axis,
    )
    axis.set(title=title, xlabel="Predicted label", ylabel="True label")
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def save_comparison_plot(results: dict[str, dict[str, Any]], destination: Path) -> None:
    rows = [
        {"model": name, "metric": metric, "value": values[metric]}
        for name, values in results.items()
        for metric in ("precision", "recall", "f1", "roc_auc")
        if values.get(metric) is not None
    ]
    if not rows:
        return
    frame = pd.DataFrame(rows)
    figure, axis = plt.subplots(figsize=(10, 5))
    sns.barplot(data=frame, x="model", y="value", hue="metric", ax=axis)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Held-out test metric")
    axis.set_xlabel("")
    axis.tick_params(axis="x", rotation=18)
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def save_roc_plot(curves: dict[str, tuple[np.ndarray, np.ndarray]], destination: Path) -> None:
    if not curves:
        return
    figure, axis = plt.subplots(figsize=(7, 6))
    for name, (labels, scores) in curves.items():
        from sklearn.metrics import roc_curve

        false_positive, true_positive, _ = roc_curve(labels, scores)
        axis.plot(false_positive, true_positive, label=name)
    axis.plot([0, 1], [0, 1], linestyle="--", color="#777777", label="Chance")
    axis.set(title="Held-out test ROC curves", xlabel="False positive rate", ylabel="True positive rate")
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def save_json(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
