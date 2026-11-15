from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest

from ml.dataset import FEATURE_COLUMNS


def create_eda(features: pd.DataFrame, output_dir: Path, seed: int = 42) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    numeric = features[FEATURE_COLUMNS]
    missing = numeric.isna().sum().sort_values(ascending=False)
    figure, axis = plt.subplots(figsize=(10, 4))
    sns.barplot(x=missing.index, y=missing.values, color="#4472c4", ax=axis)
    axis.set(title="Missing values by engineered feature", ylabel="Missing count", xlabel="")
    axis.tick_params(axis="x", rotation=35)
    figure.tight_layout()
    figure.savefig(output_dir / "missing_values.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 4))
    sns.countplot(data=features, x="label", hue="label", legend=False, palette=["#3c8d70", "#c65f5f"], ax=axis)
    axis.set(title="Manifest class counts", xlabel="Label (0 authentic, 1 tampered)", ylabel="Image count")
    figure.tight_layout()
    figure.savefig(output_dir / "class_distribution.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 8))
    sns.heatmap(features[FEATURE_COLUMNS + ["label"]].corr(numeric_only=True), cmap="vlag", center=0, ax=axis)
    axis.set(title="Feature correlation (descriptive EDA; not model selection)")
    figure.tight_layout()
    figure.savefig(output_dir / "correlation_heatmap.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(11, 5))
    sns.boxplot(data=features.melt(id_vars="label", value_vars=FEATURE_COLUMNS), x="variable", y="value", hue="label", ax=axis)
    axis.set(title="Feature distributions by label", xlabel="", ylabel="Raw feature value")
    axis.tick_params(axis="x", rotation=35)
    figure.tight_layout()
    figure.savefig(output_dir / "feature_distributions.png", dpi=160)
    plt.close(figure)

    imputed = numeric.fillna(numeric.median())
    detector = IsolationForest(random_state=seed, contamination="auto")
    outliers = detector.fit_predict(imputed) == -1
    outlier_data = features[["image_path", "label", "group_id"]].copy()
    outlier_data["outlier_flag"] = outliers
    outlier_data.to_csv(output_dir / "outlier_flags.csv", index=False)
    figure, axis = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=features.assign(outlier_flag=outliers),
        x="laplacian_variance",
        y="entropy",
        hue="outlier_flag",
        style="label",
        palette={False: "#4472c4", True: "#c65f5f"},
        ax=axis,
    )
    axis.set(title="Unsupervised outlier flags (EDA only; no row removal)")
    figure.tight_layout()
    figure.savefig(output_dir / "outlier_visualization.png", dpi=160)
    plt.close(figure)
    return {"rows": int(len(features)), "outlier_flags": int(outliers.sum())}
