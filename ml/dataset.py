from __future__ import annotations

import hashlib
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "log_file_bytes",
    "width",
    "height",
    "aspect_ratio",
    "gray_mean",
    "gray_std",
    "laplacian_variance",
    "edge_density",
    "entropy",
]
LABEL_MAP = {"authentic": 0, "tampered": 1}


def extract_image_features(path: Path) -> dict[str, float]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not decode image: {path.name}")
    return _features_from_image(image, path.stat().st_size)


def extract_image_features_from_bytes(
    data: bytes,
    filename: str,
    original_file_size: int | None = None,
) -> dict[str, float]:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not decode image: {filename}")
    return _features_from_image(image, original_file_size if original_file_size is not None else len(data))


def _features_from_image(image: np.ndarray, file_size: int) -> dict[str, float]:
    height, width = image.shape[:2]
    if min(width, height) < 16:
        raise ValueError(f"Image dimensions are too small for analysis: {path.name}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    probabilities = histogram / max(float(histogram.sum()), 1.0)
    entropy = -float(np.sum(probabilities[probabilities > 0] * np.log2(probabilities[probabilities > 0])))
    edges = cv2.Canny(gray, 80, 160)
    return {
        "log_file_bytes": math.log1p(file_size),
        "width": float(width),
        "height": float(height),
        "aspect_ratio": float(width / height),
        "gray_mean": float(gray.mean()),
        "gray_std": float(gray.std()),
        "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "edge_density": float(np.count_nonzero(edges) / edges.size),
        "entropy": entropy,
    }


def load_manifest(manifest_path: Path, image_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load image_path,label,group_id rows; return usable feature rows and rejected rows."""
    manifest_path = manifest_path.resolve()
    image_root = image_root.resolve()
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    required = {"image_path", "label", "group_id"}
    missing_columns = required - set(manifest.columns)
    if missing_columns:
        raise ValueError(f"Manifest is missing required columns: {', '.join(sorted(missing_columns))}")
    manifest = manifest[list(sorted(required))].copy()
    manifest["label"] = manifest["label"].str.strip().str.lower()
    manifest["group_id"] = manifest["group_id"].str.strip()
    rejected: list[dict[str, str]] = []
    records: list[dict[str, object]] = []
    hashes: dict[str, str] = {}
    seen: set[str] = set()

    for row_number, row in manifest.iterrows():
        relative = row["image_path"].strip()
        if not relative or row["label"] not in LABEL_MAP or not row["group_id"]:
            rejected.append({"row": str(row_number + 2), "image_path": relative, "reason": "missing path/group or invalid label"})
            continue
        image_path = (image_root / relative).resolve()
        if not image_path.is_relative_to(image_root):
            rejected.append({"row": str(row_number + 2), "image_path": relative, "reason": "path escapes dataset root"})
            continue
        if not image_path.is_file():
            rejected.append({"row": str(row_number + 2), "image_path": relative, "reason": "file is missing"})
            continue
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        previous_label = hashes.get(digest)
        if previous_label is not None and previous_label != row["label"]:
            raise ValueError(f"Identical image bytes have conflicting labels at manifest row {row_number + 2}.")
        if digest in seen:
            rejected.append({"row": str(row_number + 2), "image_path": relative, "reason": "exact duplicate removed"})
            continue
        hashes[digest] = row["label"]
        seen.add(digest)
        try:
            features = extract_image_features(image_path)
        except (OSError, ValueError, cv2.error) as exc:
            rejected.append({"row": str(row_number + 2), "image_path": relative, "reason": str(exc)})
            continue
        records.append({
            **features,
            "image_path": relative,
            "label": LABEL_MAP[row["label"]],
            "group_id": row["group_id"],
            "sha256": digest,
        })
    if not records:
        raise ValueError("No readable, labeled images remain after manifest validation.")
    return pd.DataFrame(records), pd.DataFrame(rejected, columns=["row", "image_path", "reason"])
