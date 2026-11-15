from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf
import torch
from PIL import Image

from ml.cnn_model import DocumentCNN
from ml.train_cnn import IMAGE_SIZE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "document_cnn.pt"


def predict_document_image(data: bytes, extension: str) -> dict[str, Any]:
    if not MODEL_PATH.is_file():
        return {
            "status": "not_trained",
            "message": "CNN model is not trained; no CNN prediction is available.",
        }
    try:
        if extension == ".pdf":
            with pymupdf.open(stream=data, filetype="pdf") as document:
                if not document.page_count:
                    raise ValueError("Cannot run CNN on a PDF with no pages.")
                image = Image.open(io.BytesIO(document[0].get_pixmap(matrix=pymupdf.Matrix(1, 1), alpha=False).tobytes("png"))).convert("RGB")
        else:
            with Image.open(io.BytesIO(data)) as source:
                image = source.convert("RGB")
    except Exception:
        return {
            "status": "error",
            "message": "Cannot decode image for CNN inference.",
        }
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    tensor = torch.from_numpy((np.asarray(image, dtype=np.float32) / 255.0).transpose(2, 0, 1).copy()).unsqueeze(0)
    model = DocumentCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    with torch.inference_mode():
        probability = float(torch.sigmoid(model(tensor))[0])
    return {
        "status": "available",
        "prediction": "tampered_indicator" if probability >= 0.5 else "authentic_like",
        "tampered_probability": round(probability, 4),
        "threshold": 0.5,
        "message": "CNN output is an image-pattern indicator only; it does not prove document fraud.",
    }
