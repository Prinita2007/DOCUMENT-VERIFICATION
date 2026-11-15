from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Document

FEATURE_COLUMNS = (
    "log_file_size",
    "width",
    "height",
    "blur_variance",
    "text_length",
    "page_count",
    "ocr_confidence",
)
MINIMUM_BASELINE = 10


def extract_features(
    file_size: int,
    quality: dict[str, Any],
    text_length: int,
    ocr_confidence: float | None,
) -> dict[str, float]:
    return {
        "log_file_size": float(np.log1p(max(file_size, 0))),
        "width": float(quality.get("width", 0)),
        "height": float(quality.get("height", 0)),
        "blur_variance": float(quality.get("blur", 0)),
        "text_length": float(max(text_length, 0)),
        "page_count": float(quality.get("pages", 0)),
        "ocr_confidence": float(ocr_confidence or 0),
    }


def detect_anomaly(features: dict[str, float], db: Session) -> dict[str, Any]:
    rows = db.scalars(
        select(Document.features)
        .where(Document.status == "verified")
        .order_by(Document.id.desc())
        .limit(500)
    ).all()
    baseline = [
        row
        for item in rows
        if isinstance((row := item), dict) and all(key in row for key in FEATURE_COLUMNS)
    ]
    if len(baseline) < MINIMUM_BASELINE:
        return {
            "status": "insufficient_baseline",
            "prediction": None,
            "decision_score": None,
            "baseline_count": len(baseline),
            "message": (
                f"Isolation Forest not run: {len(baseline)}/{MINIMUM_BASELINE} "
                "manually verified feature records are available."
            ),
        }

    frame = pd.DataFrame(baseline, columns=FEATURE_COLUMNS).replace([np.inf, -np.inf], np.nan)
    frame = frame.fillna(0.0)
    current = pd.DataFrame([[features.get(key, 0.0) for key in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
        n_jobs=1,
    )
    model.fit(frame)
    prediction = int(model.predict(current)[0])
    decision_score = float(model.decision_function(current)[0])
    return {
        "status": "outlier" if prediction == -1 else "within_baseline",
        "prediction": prediction,
        "decision_score": round(decision_score, 6),
        "baseline_count": len(baseline),
        "message": (
            "Feature profile is an outlier relative to manually verified documents; "
            "this is an indicator requiring human review."
            if prediction == -1
            else "Feature profile falls within the model's learned baseline; this does not prove authenticity."
        ),
    }
