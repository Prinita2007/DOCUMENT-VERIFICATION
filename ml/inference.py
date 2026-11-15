from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import pymupdf
from PIL import Image

from ml.dataset import FEATURE_COLUMNS, extract_image_features_from_bytes

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIRECTORY = PROJECT_ROOT / "models"
SUPERVISED_MODELS = (
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "support_vector_machine",
)


def _first_page_image(data: bytes, extension: str) -> tuple[bytes, str]:
    if extension == ".pdf":
        with pymupdf.open(stream=data, filetype="pdf") as document:
            if not document.page_count:
                raise ValueError("Cannot create image features from a PDF with no pages.")
            image_data = document[0].get_pixmap(matrix=pymupdf.Matrix(1, 1), alpha=False).tobytes("png")
        return image_data, "document.png"
    return data, f"document{extension}"


def predict_classical_models(data: bytes, extension: str) -> dict[str, Any]:
    available = [
        name for name in SUPERVISED_MODELS
        if (MODEL_DIRECTORY / f"{name}.joblib").is_file()
    ]
    if not available:
        return {"status": "not_trained", "predictions": []}
    try:
        image_data, filename = _first_page_image(data, extension)
        features = extract_image_features_from_bytes(image_data, filename, original_file_size=len(data))
    except Exception:
        return {"status": "error", "predictions": []}
    row = pd.DataFrame([[features[name] for name in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    predictions: list[dict[str, Any]] = []
    for name in available:
        model = joblib.load(MODEL_DIRECTORY / f"{name}.joblib")
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(row)[0]
            classes = list(model.classes_)
            positive_index = classes.index(1)
            score = float(probabilities[positive_index])
            predictions.append({
                "algorithm": name,
                "prediction": "tampered_indicator" if score >= 0.5 else "authentic_like",
                "score_type": "tampered_probability",
                "score": round(score, 4),
            })
        else:
            score = float(model.decision_function(row)[0])
            predictions.append({
                "algorithm": name,
                "prediction": "tampered_indicator" if score >= 0 else "authentic_like",
                "score_type": "decision_margin",
                "score": round(score, 4),
            })
    return {"status": "available", "predictions": predictions}
