from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

from ml.dataset import FEATURE_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
EVALUATION_DIR = PROJECT_ROOT / "results" / "evaluation"
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"

FEATURE_FRIENDLY_NAMES = {
    "log_file_bytes": "File Size & Compression",
    "width": "Document Width (px)",
    "height": "Document Height (px)",
    "aspect_ratio": "Page Aspect Ratio",
    "gray_mean": "Background Luminance / Brightness",
    "gray_std": "Grayscale Contrast Variation",
    "laplacian_variance": "Text Edge Sharpness / Blur",
    "edge_density": "High-Frequency Edge / Texture Density",
    "entropy": "Information Entropy / Noise Irregularity",
}

FEATURE_DESCRIPTIONS = {
    "log_file_bytes": "Anomalies in uncompressed byte payload compared to typical scans.",
    "laplacian_variance": "Localized blur or resampling softness often introduced during digital cut-and-paste.",
    "edge_density": "Unnatural high-frequency pixel transitions around tampered letters or pasted grades.",
    "entropy": "Information distribution variations that reveal mismatched noise floors from splicing.",
    "gray_mean": "Luminance inconsistency between authentic paper background and digital inserts.",
    "gray_std": "Discrepancy in contrast across document patches.",
    "aspect_ratio": "Abnormal document page dimensions.",
    "width": "Pixel dimension deviation from standard DPI scans.",
    "height": "Pixel dimension deviation from standard DPI scans.",
}

_explainer_cache: tuple[Any, shap.Explainer, pd.DataFrame] | None = None


def _get_explainer() -> tuple[Any, shap.Explainer, pd.DataFrame]:
    global _explainer_cache
    if _explainer_cache is not None:
        return _explainer_cache

    model_path = MODELS_DIR / "logistic_regression.joblib"
    if not model_path.is_file():
        model_path = MODELS_DIR / "random_forest.joblib"
    if not model_path.is_file():
        raise FileNotFoundError("No trained classical model found for SHAP explanations.")

    model = joblib.load(model_path)

    # Load background data from split manifest or processed features
    features_csv = PROCESSED_DIR / "image_features.csv"
    if features_csv.is_file():
        df = pd.read_csv(features_csv)
        background = df[FEATURE_COLUMNS].sample(min(20, len(df)), random_state=42)
    else:
        # Fallback synthetic background
        background = pd.DataFrame(
            np.zeros((5, len(FEATURE_COLUMNS))),
            columns=FEATURE_COLUMNS,
        )

    def predict_fn(values: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(values, columns=FEATURE_COLUMNS)
        if hasattr(model, "predict_proba"):
            return np.asarray(model.predict_proba(frame))[:, 1]
        return np.asarray(model.decision_function(frame))

    explainer = shap.Explainer(
        predict_fn,
        background.to_numpy(),
        algorithm="permutation",
        feature_names=FEATURE_COLUMNS,
    )
    _explainer_cache = (model, explainer, background)
    return _explainer_cache


def explain_document_features(features: dict[str, float]) -> dict[str, Any]:
    """Generate local SHAP explanations for a single document's extracted features."""
    try:
        model, explainer, background = _get_explainer()
    except (FileNotFoundError, Exception) as exc:
        return {
            "status": "unavailable",
            "message": f"SHAP explainer unavailable: {exc}",
            "feature_contributions": [],
        }

    # Prepare single row
    row = pd.DataFrame([[features.get(col, 0.0) for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)

    # Predict probability
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(row)[0, 1])
    else:
        prob = float(1.0 / (1.0 + np.exp(-model.decision_function(row)[0])))

    # Compute SHAP values
    shap_vals = explainer(row.to_numpy(), max_evals=2 * len(FEATURE_COLUMNS) + 1)
    values = np.asarray(shap_vals.values)[0]
    if values.ndim > 1:
        values = values[..., 0]

    base_val = float(shap_vals.base_values[0]) if hasattr(shap_vals, "base_values") else 0.5

    contributions = []
    for idx, col in enumerate(FEATURE_COLUMNS):
        sv = float(values[idx])
        val = float(row.iloc[0, idx])
        abs_sv = abs(sv)
        impact = "high" if abs_sv >= 0.1 else "medium" if abs_sv >= 0.03 else "low"
        direction = "increases_risk" if sv > 0 else "reduces_risk"

        contributions.append({
            "feature": col,
            "friendly_name": FEATURE_FRIENDLY_NAMES.get(col, col),
            "value": round(val, 3),
            "shap_value": round(sv, 4),
            "impact": impact,
            "direction": direction,
            "description": FEATURE_DESCRIPTIONS.get(col, ""),
        })

    # Sort by absolute SHAP value descending
    contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    # Generate narrative summary
    top_risk = [c for c in contributions if c["direction"] == "increases_risk" and c["impact"] in ("high", "medium")]
    top_safe = [c for c in contributions if c["direction"] == "reduces_risk" and c["impact"] in ("high", "medium")]

    narrative_parts = []
    if top_risk:
        names = ", ".join(f"{c['friendly_name']} ({c['shap_value']:+.2f})" for c in top_risk[:2])
        narrative_parts.append(f"Primary factors elevating risk: {names}.")
    if top_safe:
        names = ", ".join(f"{c['friendly_name']} ({c['shap_value']:+.2f})" for c in top_safe[:2])
        narrative_parts.append(f"Factors supporting authenticity: {names}.")
    if not narrative_parts:
        narrative_parts.append("All extracted statistical features are near baseline expectations.")

    narrative = " ".join(narrative_parts)

    # Generate inline base64 plot
    plot_base64 = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 3.8))
        names = [c["friendly_name"] for c in contributions[::-1]]
        y_pos = np.arange(len(names))
        vals = [c["shap_value"] for c in contributions[::-1]]
        colors = ["#d9534f" if v > 0 else "#2e8540" for v in vals]

        ax.barh(y_pos, vals, color=colors, align="center", height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("SHAP Value (Impact on Tampered Prediction)", fontsize=8)
        ax.set_title(f"Local Explanation: Predicted Tampered Probability = {prob:.1%}", fontsize=9, fontweight="bold")
        fig.tight_layout()

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=140)
        plt.close(fig)
        buffer.seek(0)
        plot_base64 = f"data:image/png;base64,{base64.b64encode(buffer.read()).decode('utf-8')}"
    except Exception:
        pass

    return {
        "status": "available",
        "predicted_probability": round(prob, 4),
        "prediction": "tampered_indicator" if prob >= 0.5 else "authentic_like",
        "base_value": round(base_val, 4),
        "feature_contributions": contributions,
        "narrative_explanation": narrative,
        "plot_base64": plot_base64,
        "model_used": "Logistic Regression / Random Forest (Permutation SHAP)",
    }
