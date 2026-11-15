from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ml.dataset import FEATURE_COLUMNS, load_manifest
from ml.eda import create_eda
from ml.evaluation import (
    classification_metrics,
    metrics_from_predictions,
    save_comparison_plot,
    save_confusion_matrix,
    save_json,
    save_roc_plot,
)

SEED = 42


def group_train_validation_test_split(frame: pd.DataFrame, seed: int = SEED) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.model_selection import GroupShuffleSplit

    labels = frame["label"].to_numpy()
    groups = frame["group_id"].to_numpy()
    for attempt_seed in range(seed, seed + 100):
        outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=attempt_seed)
        train_validation, test = next(outer.split(frame, labels, groups))
        inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=attempt_seed + 1)
        train_relative, validation_relative = next(
            inner.split(frame.iloc[train_validation], labels[train_validation], groups[train_validation])
        )
        train = train_validation[train_relative]
        validation = train_validation[validation_relative]
        splits = (train, validation, test)
        if all(set(labels[indices]) == {0, 1} for indices in splits):
            group_sets = [set(groups[indices]) for indices in splits]
            if group_sets[0].isdisjoint(group_sets[1]) and group_sets[0].isdisjoint(group_sets[2]) and group_sets[1].isdisjoint(group_sets[2]):
                return splits
    raise ValueError(
        "Could not make train/validation/test splits with both classes and disjoint groups. "
        "Add more distinct source group_id values for authentic and tampered examples."
    )


def _pipeline(classifier: Any, feature_count: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif, k="all")),
            ("classifier", classifier),
        ]
    )


def _supervised_models(feature_count: int) -> dict[str, tuple[Pipeline, dict[str, list[Any]]]]:
    return {
        "logistic_regression": (
            _pipeline(LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED), feature_count),
            {"select__k": ["all", min(feature_count, 5)], "classifier__C": [0.1, 1.0, 10.0]},
        ),
        "decision_tree": (
            _pipeline(DecisionTreeClassifier(class_weight="balanced", random_state=SEED), feature_count),
            {"select__k": ["all", min(feature_count, 5)], "classifier__max_depth": [3, 8, None], "classifier__min_samples_leaf": [1, 3]},
        ),
        "random_forest": (
            _pipeline(RandomForestClassifier(class_weight="balanced", n_estimators=200, random_state=SEED, n_jobs=1), feature_count),
            {"select__k": ["all", min(feature_count, 5)], "classifier__max_depth": [None, 12], "classifier__min_samples_leaf": [1, 3]},
        ),
        "support_vector_machine": (
            _pipeline(SVC(class_weight="balanced", random_state=SEED), feature_count),
            {"select__k": ["all", min(feature_count, 5)], "classifier__C": [0.1, 1.0, 10.0], "classifier__kernel": ["linear", "rbf"]},
        ),
    }


def _positive_score(model: Any, values: pd.DataFrame) -> tuple[np.ndarray, str]:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(values))[:, 1], "positive_class_probability"
    return np.asarray(model.decision_function(values)), "decision_margin"


def _save_shap(model: Any, x_train: pd.DataFrame, x_test: pd.DataFrame, out_dir: Path, model_name: str) -> dict[str, Any]:
    import shap
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    background = x_train.sample(min(25, len(x_train)), random_state=SEED)
    sample = x_test.sample(min(12, len(x_test)), random_state=SEED)
    predict = lambda values: _positive_score(model, pd.DataFrame(values, columns=FEATURE_COLUMNS))[0]
    explainer = shap.Explainer(
        predict,
        background.to_numpy(),
        algorithm="permutation",
        feature_names=FEATURE_COLUMNS,
    )
    explanation = explainer(sample.to_numpy(), max_evals=2 * len(FEATURE_COLUMNS) + 1)
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        values = values[..., 0]
    contribution_means = np.mean(np.abs(values), axis=0)
    order = np.argsort(contribution_means)[::-1]
    summary = {
        "model": model_name,
        "method": "SHAP permutation explainer on model output",
        "output_scale": _positive_score(model, sample)[1],
        "explained_sample_count": int(len(sample)),
        "background_sample_count": int(len(background)),
        "features": [
            {"name": FEATURE_COLUMNS[index], "mean_absolute_shap": float(contribution_means[index])}
            for index in order
        ],
    }
    save_json(summary, out_dir / "shap_explanation.json")
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.barh(
        [FEATURE_COLUMNS[index] for index in order[::-1]],
        contribution_means[order[::-1]],
        color="#4677a8",
    )
    axis.set(title=f"SHAP global feature contributions: {model_name}", xlabel="Mean absolute SHAP value")
    figure.tight_layout()
    figure.savefig(out_dir / "shap_summary.png", dpi=160)
    plt.close(figure)
    return summary


def train(manifest_path: Path, image_root: Path, project_root: Path) -> dict[str, Any]:
    frame, rejected = load_manifest(manifest_path, image_root)
    if frame["label"].nunique() != 2:
        raise ValueError("Supervised training requires authentic and tampered labels.")
    if frame.groupby("label")["group_id"].nunique().min() < 3:
        raise ValueError("At least three independent source groups per class are needed for grouped CV.")

    processed_dir = project_root / "datasets" / "processed"
    eda_dir = project_root / "results" / "eda"
    evaluation_dir = project_root / "results" / "evaluation"
    models_dir = project_root / "models"
    for directory in (processed_dir, eda_dir, evaluation_dir, models_dir):
        directory.mkdir(parents=True, exist_ok=True)

    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    frame.to_csv(processed_dir / "image_features.csv", index=False)
    rejected.to_csv(processed_dir / "rejected_manifest_rows.csv", index=False)
    eda = create_eda(frame, eda_dir)
    train_idx, validation_idx, test_idx = group_train_validation_test_split(frame)
    split = np.full(len(frame), "train", dtype=object)
    split[validation_idx] = "validation"
    split[test_idx] = "test"
    frame.assign(split=split).to_csv(processed_dir / "split_manifest.csv", index=False)

    x = frame[FEATURE_COLUMNS]
    y = frame["label"].astype(int)
    groups = frame["group_id"]
    x_train, y_train, group_train = x.iloc[train_idx], y.iloc[train_idx], groups.iloc[train_idx]
    x_validation, y_validation = x.iloc[validation_idx], y.iloc[validation_idx]
    x_test, y_test = x.iloc[test_idx], y.iloc[test_idx]

    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    results: dict[str, dict[str, Any]] = {}
    roc_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    best_model = None
    best_model_name = ""
    best_validation_f1 = -1.0
    supervised = _supervised_models(len(FEATURE_COLUMNS))
    for name, (pipeline, parameters) in supervised.items():
        start = time.perf_counter()
        search = GridSearchCV(
            pipeline,
            parameters,
            scoring="f1",
            cv=cv,
            n_jobs=1,
            refit=True,
            error_score="raise",
            return_train_score=False,
        )
        search.fit(x_train, y_train, groups=group_train)
        training_seconds = time.perf_counter() - start
        model = search.best_estimator_
        validation_metrics = classification_metrics(model, x_validation, y_validation)
        test_metrics = classification_metrics(model, x_test, y_test)
        test_metrics["training_seconds_including_cv"] = float(training_seconds)
        test_metrics["validation_f1"] = validation_metrics["f1"]
        test_metrics["best_parameters"] = search.best_params_
        results[name] = test_metrics
        joblib.dump(model, models_dir / f"{name}.joblib")
        cv_frame = pd.DataFrame(search.cv_results_)
        cv_frame.sort_values("rank_test_score").head(30).to_csv(evaluation_dir / f"{name}_cv_results.csv", index=False)
        save_confusion_matrix(test_metrics["confusion_matrix"], f"{name} held-out test", evaluation_dir / f"{name}_confusion_matrix.png")
        roc_curves[name] = (y_test.to_numpy(), _positive_score(model, x_test)[0])
        if validation_metrics["f1"] > best_validation_f1:
            best_validation_f1 = validation_metrics["f1"]
            best_model = model
            best_model_name = name

    baseline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("detector", IsolationForest(random_state=SEED, n_estimators=300, contamination="auto", n_jobs=1)),
    ])
    authentic = y_train == 0
    if authentic.sum() < 3:
        raise ValueError("Isolation Forest requires at least three authentic training examples.")
    baseline_grid = []
    selected_contamination = None
    selected_score = -1.0
    for contamination in (0.01, 0.05, 0.1, "auto"):
        candidate = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("detector", IsolationForest(
                random_state=SEED,
                n_estimators=300,
                contamination=contamination,
                n_jobs=1,
            )),
        ])
        start = time.perf_counter()
        candidate.fit(x_train.loc[authentic])
        predictions = (candidate.predict(x_validation) == -1).astype(int)
        score = float(f1_score(y_validation, predictions, zero_division=0))
        baseline_grid.append({"contamination": contamination, "validation_f1": score})
        if score > selected_score:
            selected_score = score
            selected_contamination = contamination
            baseline = candidate
    start = time.perf_counter()
    baseline.fit(x_train.loc[authentic])
    training_seconds = time.perf_counter() - start
    start = time.perf_counter()
    baseline_predictions = (baseline.predict(x_test) == -1).astype(int)
    baseline_scores = -baseline.decision_function(x_test)
    prediction_ms = (time.perf_counter() - start) * 1000 / max(len(x_test), 1)
    baseline_metrics = metrics_from_predictions(y_test, baseline_predictions, baseline_scores, prediction_ms)
    baseline_metrics["training_seconds"] = float(training_seconds)
    baseline_metrics["selected_contamination"] = selected_contamination
    baseline_metrics["validation_candidates"] = baseline_grid
    results["isolation_forest"] = baseline_metrics
    joblib.dump(baseline, models_dir / "isolation_forest.joblib")
    save_confusion_matrix(baseline_metrics["confusion_matrix"], "Isolation Forest held-out test", evaluation_dir / "isolation_forest_confusion_matrix.png")
    roc_curves["isolation_forest"] = (y_test.to_numpy(), baseline_scores)

    if best_model is None:
        raise RuntimeError("No supervised model was selected from validation data.")
    importance = permutation_importance(
        best_model,
        x_validation,
        y_validation,
        scoring="f1",
        n_repeats=5,
        random_state=SEED,
        n_jobs=1,
    )
    importance_frame = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std,
    }).sort_values("importance_mean", ascending=False)
    importance_frame.to_csv(eda_dir / "feature_importance.csv", index=False)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.barh(
        importance_frame["feature"].iloc[::-1],
        importance_frame["importance_mean"].iloc[::-1],
        xerr=importance_frame["importance_std"].iloc[::-1],
        color="#4677a8",
    )
    axis.set(title=f"Validation permutation importance: {best_model_name}", xlabel="Mean F1 decrease after feature permutation")
    figure.tight_layout()
    figure.savefig(eda_dir / "feature_importance.png", dpi=160)
    plt.close(figure)
    shap_summary = _save_shap(best_model, x_train, x_test, evaluation_dir, best_model_name)
    comparison = {
        "status": "trained",
        "experiment": "academic-document-image-manifest",
        "seed": SEED,
        "manifest_sha256": manifest_digest,
        "feature_columns": FEATURE_COLUMNS,
        "label_mapping": {"authentic": 0, "tampered": 1},
        "row_counts": {"total": len(frame), "train": len(train_idx), "validation": len(validation_idx), "test": len(test_idx)},
        "group_leakage_check": "passed: source group IDs are disjoint across train, validation, and test",
        "selected_supervised_model": best_model_name,
        "models": results,
        "eda": eda,
        "rejected_rows": int(len(rejected)),
        "shap": shap_summary,
        "limitations": [
            "Metrics apply only to the supplied dataset and fixed held-out group split.",
            "This dataset is not a representative sample of academic certificates unless its source labels and domain justify that claim.",
            "Model predictions are indicators for human review, not proof of fraud.",
        ],
    }
    save_json(comparison, evaluation_dir / "model_comparison.json")
    save_comparison_plot(results, evaluation_dir / "model_comparison.png")
    save_roc_plot(roc_curves, evaluation_dir / "roc_curves.png")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the five document-analysis algorithms.")
    parser.add_argument("--manifest", type=Path, required=True, help="CSV with image_path,label,group_id columns")
    parser.add_argument("--image-root", type=Path, required=True, help="Root directory containing dataset images")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    result = train(args.manifest, args.image_root, args.project_root.resolve())
    print(json.dumps({
        "status": result["status"],
        "rows": result["row_counts"],
        "selected_supervised_model": result["selected_supervised_model"],
        "results_path": str(args.project_root.resolve() / "results" / "evaluation" / "model_comparison.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
