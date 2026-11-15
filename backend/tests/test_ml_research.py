import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import pytest
import torch
from sklearn.datasets import make_classification
from sklearn.ensemble import IsolationForest

from ml.cnn_model import DocumentCNN
from ml.cnn_inference import predict_document_image
from ml.dataset import FEATURE_COLUMNS, load_manifest
from ml.inference import predict_classical_models
from ml.train_classical import _save_shap, _supervised_models, group_train_validation_test_split, train


def _create_image(path: Path, index: int, tampered: bool) -> None:
    image = Image.new(
        "RGB",
        (96 + index % 7, 128 + index % 11),
        (250, 250, 250) if not tampered else (238, 238, 238),
    )
    draw = ImageDraw.Draw(image)
    draw.text((8 + index % 20, 12), f"Academic sample {index}", fill="black")
    if tampered:
        draw.rectangle((8, 72, 82, 88), fill="red")
    image.save(path)


def _manifest_dataset(root: Path, count_per_class: int = 20) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, tampered in (("authentic", False), ("tampered", True)):
        for index in range(count_per_class):
            name = f"{label}_{index}.png"
            _create_image(root / name, index + (1000 if tampered else 0), tampered)
            rows.append({"image_path": name, "label": label, "group_id": f"{label}-source-{index}"})
    manifest = root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=["image_path", "label", "group_id"])
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def test_manifest_processing_rejects_missing_and_deduplicates(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    _create_image(root / "one.png", 1, False)
    _create_image(root / "two.png", 2, True)
    (root / "duplicate.png").write_bytes((root / "one.png").read_bytes())
    manifest = root / "manifest.csv"
    manifest.write_text(
        "image_path,label,group_id\none.png,authentic,source-one\n"
        "duplicate.png,authentic,source-one\ntwo.png,tampered,source-two\n"
        "missing.png,authentic,source-three\n",
        encoding="utf-8",
    )
    features, rejected = load_manifest(manifest, root)
    assert len(features) == 2
    assert set(FEATURE_COLUMNS).issubset(features.columns)
    assert len(rejected) == 2
    assert any("duplicate" in reason for reason in rejected["reason"])
    assert any("missing" in reason for reason in rejected["reason"])


def test_manifest_rejects_conflicting_labels_for_identical_image(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    _create_image(root / "one.png", 1, False)
    (root / "same.png").write_bytes((root / "one.png").read_bytes())
    manifest = root / "manifest.csv"
    manifest.write_text(
        "image_path,label,group_id\none.png,authentic,source-one\nsame.png,tampered,source-one\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="conflicting labels"):
        load_manifest(manifest, root)


def test_group_split_has_disjoint_source_groups():
    rows = [
        {"label": label, "group_id": f"{label}-{index}"}
        for label in (0, 1)
        for index in range(20)
    ]
    import pandas as pd

    frame = pd.DataFrame(rows)
    train_index, validation_index, test_index = group_train_validation_test_split(frame)
    split_groups = [set(frame.iloc[index]["group_id"]) for index in (train_index, validation_index, test_index)]
    assert split_groups[0].isdisjoint(split_groups[1])
    assert split_groups[0].isdisjoint(split_groups[2])
    assert split_groups[1].isdisjoint(split_groups[2])


def test_five_classical_algorithms_fit_and_predict_on_test_fixture():
    x, y = make_classification(
        n_samples=90,
        n_features=len(FEATURE_COLUMNS),
        n_informative=6,
        n_redundant=1,
        random_state=21,
    )
    import pandas as pd

    frame = pd.DataFrame(x, columns=FEATURE_COLUMNS)
    for name, (pipeline, _) in _supervised_models(len(FEATURE_COLUMNS)).items():
        pipeline.fit(frame, y)
        assert pipeline.predict(frame[:3]).shape == (3,), name
    anomaly = IsolationForest(random_state=42, n_estimators=10).fit(frame)
    assert anomaly.predict(frame[:3]).shape == (3,)


def test_shap_explainer_generates_real_attributions_on_test_fixture(tmp_path):
    x, y = make_classification(
        n_samples=80,
        n_features=len(FEATURE_COLUMNS),
        n_informative=6,
        n_redundant=1,
        random_state=17,
    )
    import pandas as pd

    frame = pd.DataFrame(x, columns=FEATURE_COLUMNS)
    model = _supervised_models(len(FEATURE_COLUMNS))["logistic_regression"][0]
    model.fit(frame[:60], y[:60])
    result = _save_shap(model, frame[:60], frame[60:], tmp_path, "test_model")
    assert result["explained_sample_count"] > 0
    assert {item["name"] for item in result["features"]} == set(FEATURE_COLUMNS)
    assert (tmp_path / "shap_summary.png").is_file()


def test_cnn_forward_pass_returns_binary_logits():
    model = DocumentCNN().eval()
    with torch.inference_mode():
        logits = model(torch.zeros((2, 3, 224, 224)))
    assert tuple(logits.shape) == (2,)


def test_untrained_inference_is_reported_as_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr("ml.inference.MODEL_DIRECTORY", tmp_path)
    monkeypatch.setattr("ml.cnn_inference.MODEL_PATH", tmp_path / "missing.pt")
    assert predict_classical_models(b"image", ".png")["status"] == "not_trained"
    assert predict_document_image(b"image", ".png")["status"] == "not_trained"


def test_complete_classical_pipeline_runs_on_ephemeral_unit_fixture(tmp_path):
    manifest = _manifest_dataset(tmp_path / "raw", count_per_class=20)
    result = train(manifest, manifest.parent, tmp_path / "project")
    assert result["status"] == "trained"
    assert set(result["models"]) == {
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "support_vector_machine",
        "isolation_forest",
    }
    assert "passed" in result["group_leakage_check"]
    assert (tmp_path / "project" / "results" / "evaluation" / "model_comparison.json").is_file()


def test_cnn_training_path_runs_on_ephemeral_unit_fixture(tmp_path):
    from ml.train_cnn import train_cnn

    manifest = _manifest_dataset(tmp_path / "raw", count_per_class=10)
    result = train_cnn(manifest, manifest.parent, tmp_path / "project", epochs=1, batch_size=4)
    assert result["status"] == "trained"
    assert result["metrics"]["confusion_matrix"]
    assert (tmp_path / "project" / "models" / "document_cnn.pt").is_file()
