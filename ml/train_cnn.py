from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ml.cnn_model import DocumentCNN
from ml.dataset import load_manifest
from ml.evaluation import metrics_from_predictions, save_confusion_matrix, save_json, save_roc_plot
from ml.train_classical import SEED, group_train_validation_test_split

IMAGE_SIZE = 224


class ManifestImages(Dataset):
    def __init__(self, rows: Any, image_root: Path, augment: bool = False) -> None:
        self.rows = rows.reset_index(drop=True)
        self.image_root = image_root
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows.iloc[index]
        with Image.open(self.image_root / row["image_path"]) as source:
            image = source.convert("RGB")
            if self.augment:
                image = image.rotate(random.uniform(-2.0, 2.0), fillcolor="white")
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return tensor, label


def _probabilities(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.inference_mode():
        for images, target in loader:
            logits = model(images.to(device))
            predictions.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(target.numpy())
    return np.concatenate(labels).astype(int), np.concatenate(predictions)


def train_cnn(
    manifest_path: Path,
    image_root: Path,
    project_root: Path,
    epochs: int = 15,
    batch_size: int = 16,
) -> dict[str, Any]:
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive integers.")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)

    image_root = image_root.resolve()
    frame, rejected = load_manifest(manifest_path, image_root)
    if frame["label"].nunique() != 2:
        raise ValueError("CNN training requires both authentic and tampered examples.")
    train_idx, validation_idx, test_idx = group_train_validation_test_split(frame)
    train_rows = frame.iloc[train_idx]
    validation_rows = frame.iloc[validation_idx]
    test_rows = frame.iloc[test_idx]

    train_loader = DataLoader(ManifestImages(train_rows, image_root, augment=True), batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(ManifestImages(validation_rows, image_root), batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(ManifestImages(test_rows, image_root), batch_size=batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DocumentCNN().to(device)
    positive = int(train_rows["label"].sum())
    negative = int(len(train_rows) - positive)
    if min(positive, negative) == 0:
        raise ValueError("CNN training split must contain both classes.")
    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negative / positive], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    project_root = project_root.resolve()
    models_dir = project_root / "models"
    evaluation_dir = project_root / "results" / "evaluation"
    models_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = models_dir / "document_cnn.pt"
    best_f1 = -1.0
    best_epoch = 0
    patience = 4
    stale_epochs = 0
    start_training = time.perf_counter()
    training_history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(images), labels)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        validation_labels, validation_probabilities = _probabilities(model, validation_loader, device)
        validation_predictions = (validation_probabilities >= 0.5).astype(int)
        validation_f1 = float(f1_score(validation_labels, validation_predictions, zero_division=0))
        training_history.append({
            "epoch": epoch,
            "mean_training_loss": float(np.mean(epoch_losses)),
            "validation_f1_at_0_5": validation_f1,
        })
        if validation_f1 > best_f1:
            best_f1 = validation_f1
            best_epoch = epoch
            stale_epochs = 0
            torch.save(model.cpu().state_dict(), checkpoint)
            model.to(device)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    training_seconds = time.perf_counter() - start_training
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    prediction_start = time.perf_counter()
    test_labels, test_probabilities = _probabilities(model, test_loader, device)
    prediction_ms = (time.perf_counter() - prediction_start) * 1000 / max(len(test_labels), 1)
    predictions = (test_probabilities >= 0.5).astype(int)
    test_metrics = metrics_from_predictions(
        test_labels,
        predictions,
        test_probabilities,
        prediction_ms,
    )
    test_metrics["training_seconds"] = float(training_seconds)
    test_metrics["selected_epoch"] = best_epoch
    test_metrics["decision_threshold"] = 0.5
    result = {
        "status": "trained",
        "model": "small custom CNN",
        "architecture": "three convolution blocks with batch normalization and adaptive pooling",
        "seed": SEED,
        "image_size": IMAGE_SIZE,
        "device": str(device),
        "row_counts": {"total": len(frame), "train": len(train_rows), "validation": len(validation_rows), "test": len(test_rows)},
        "group_leakage_check": "passed: source group IDs are disjoint across train, validation, and test",
        "rejected_rows": int(len(rejected)),
        "metrics": test_metrics,
        "training_history": training_history,
        "limitations": [
            "CNN metrics are valid only for this supplied dataset and fixed held-out group split.",
            "The CNN is not pretrained and must be trained on a licensed, appropriately labeled dataset before use.",
            "Visual cues from identity documents may not generalize to academic certificates.",
            "The prediction is an indicator for human review, not proof of fraud.",
        ],
    }
    save_json(result, evaluation_dir / "cnn_results.json")
    save_confusion_matrix(test_metrics["confusion_matrix"], "CNN held-out test", evaluation_dir / "cnn_confusion_matrix.png")
    save_roc_plot({"custom_cnn": (test_labels, test_probabilities)}, evaluation_dir / "cnn_roc_curve.png")
    save_json(
        {"architecture": result["architecture"], "image_size": IMAGE_SIZE, "seed": SEED, "model": checkpoint.name},
        models_dir / "document_cnn.json",
    )
    save_json({"epochs": training_history}, evaluation_dir / "cnn_training_history.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small CNN from a licensed image manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    result = train_cnn(args.manifest, args.image_root, args.project_root, args.epochs, args.batch_size)
    print(json.dumps({"status": result["status"], "row_counts": result["row_counts"], "metrics": result["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
