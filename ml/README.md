# Research training pipeline

The five required algorithms are implemented in `train_classical.py`: Logistic Regression, Decision Tree, Random Forest, SVM, and Isolation Forest. Supervised models use group-aware train/validation/test splits, three-fold `StratifiedGroupKFold` cross-validation, `GridSearchCV`, imputation, scaling, and `SelectKBest`. Isolation Forest is fit only on the authentic portion of the training split; its contamination setting is selected on validation labels, never on the test set. Held-out metrics, timings, confusion matrices, ROC curves, CV results, and a SHAP permutation explanation are written to `results/evaluation/`. EDA outputs are written to `results/eda/`.

`train_cnn.py` contains a real three-block PyTorch CNN training loop with validation-F1 early stopping and a held-out test evaluation. `cnn_inference.py` loads the saved CPU-safe state dictionary for inference in the FastAPI verification pipeline. It remains unavailable until it has been trained with a suitable, licensed, labeled image corpus.

## Input manifest

Provide CSV columns:

```csv
image_path,label,group_id
images/example_001.png,authentic,source_document_001
images/example_001_edited.png,tampered,source_document_001
```

Labels must be exactly `authentic` or `tampered`. Each source document and all scans/edits derived from it must share a `group_id`. The loader rejects paths outside the given root, missing images, unreadable files, invalid labels, and exact duplicates. Conflicting labels for identical bytes stop the run. Review `datasets/dataset_sources.csv` and the original dataset terms before preparing a manifest.

## Commands

Run from the project root using the configured backend virtual environment:

```powershell
.\backend\venv\Scripts\python.exe -m ml.train_classical --manifest .\datasets\raw\approved_manifest.csv --image-root .\datasets\raw
.\backend\venv\Scripts\python.exe -m ml.train_cnn --manifest .\datasets\raw\approved_manifest.csv --image-root .\datasets\raw
```

The manifest and dataset paths are examples; create them only after obtaining permission and mapping labels correctly. No model metrics are present until these commands complete against a real dataset. A test with random tensors or synthetic examples is not an experiment and is not reported as one.
