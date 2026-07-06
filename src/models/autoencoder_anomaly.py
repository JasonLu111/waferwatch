"""
Autoencoder-style reconstruction-error anomaly detection baseline for WaferWatch.

This module trains a lightweight neural reconstruction model using scikit-learn
MLPRegressor. It acts as an autoencoder-style anomaly detection baseline without
introducing a heavyweight deep learning dependency.

Design:
- Fit StandardScaler and MLPRegressor only on normal-reference training lots.
- Use reconstruction error as the anomaly risk score.
- Use labels only for held-out evaluation.
- Select a threshold from the normal-reference training reconstruction-error
  distribution.
"""

from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings

from src.models.isolation_forest import (
    evaluate_anomaly_detector,
    load_feature_table,
    prepare_xy,
)
from src.utils.config import DATA_PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, ensure_directories_exist
from src.utils.logger import get_logger


logger = get_logger(__name__)


def calculate_reconstruction_error(
    model: MLPRegressor,
    X_scaled: np.ndarray,
) -> np.ndarray:
    """
    Calculate per-row reconstruction error.

    Larger values indicate higher anomaly risk.
    """

    X_reconstructed = model.predict(X_scaled)
    errors = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)

    return errors


def build_markdown_report(report: dict[str, Any]) -> str:
    """
    Build Markdown report for the autoencoder-style anomaly detection baseline.
    """

    metrics = report["evaluation_metrics"]

    lines: list[str] = []

    lines.append("# WaferWatch Autoencoder Anomaly Detection Baseline Report")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append(
        "This report documents a lightweight autoencoder-style reconstruction-error anomaly detection baseline."
    )
    lines.append(
        "The model is fitted only on normal-reference training lots. Labels are used only for held-out evaluation."
    )
    lines.append("")
    lines.append("## 2. Model Configuration")
    lines.append("")
    lines.append(f"- Model file: `{report['model_file']}`")
    lines.append(f"- Feature table: `{report['feature_table']}`")
    lines.append(f"- Training rows: `{report['training_rows']}`")
    lines.append(f"- Normal-reference training rows: `{report['normal_reference_training_rows']}`")
    lines.append(f"- Test rows: `{report['test_rows']}`")
    lines.append(f"- Number of features: `{report['n_features']}`")
    lines.append(f"- Hidden layers: `{report['model_parameters']['hidden_layer_sizes']}`")
    lines.append(f"- Activation: `{report['model_parameters']['activation']}`")
    lines.append(f"- Solver: `{report['model_parameters']['solver']}`")
    lines.append(f"- Max iterations: `{report['model_parameters']['max_iter']}`")
    lines.append(f"- Threshold quantile: `{report['threshold_quantile']}`")
    lines.append(f"- Reconstruction-error threshold: `{report['reconstruction_error_threshold']:.6f}`")
    lines.append("")
    lines.append("## 3. Evaluation Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")

    metric_order = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
        "matthews_corrcoef",
        "roc_auc",
        "pr_auc",
        "precision_at_k",
        "recall_at_k",
        "false_alarms_per_100_lots",
    ]

    for metric_name in metric_order:
        lines.append(f"| {metric_name} | {metrics[metric_name]:.6f} |")

    lines.append("")
    lines.append("## 4. Confusion Matrix")
    lines.append("")
    lines.append("| Item | Count |")
    lines.append("|---|---:|")

    for key, value in metrics["confusion_matrix"].items():
        lines.append(f"| {key} | {value} |")

    lines.append("")
    lines.append("## 5. Top Suspicious Lots")
    lines.append("")
    lines.append("| Rank | Lot ID | True Label | Reconstruction Error | Predicted Label |")
    lines.append("|---:|---|---:|---:|---:|")

    for row in report["top_suspicious_lots"]:
        lines.append(
            f"| {row['rank']} | `{row['lot_id']}` | {row['true_label']} | "
            f"{row['reconstruction_error']:.6f} | {row['predicted_label']} |"
        )

    lines.append("")
    lines.append("## 6. Interpretation")
    lines.append("")
    lines.append(report["interpretation_note"])
    lines.append("")

    return "\n".join(lines)


def run_autoencoder_anomaly_baseline(
    feature_table_file_name: str = "demo_spc_selected_feature_table.csv",
    model_file_name: str = "autoencoder_anomaly_normal_reference.joblib",
    report_file_name: str = "autoencoder_anomaly_report.json",
    markdown_file_name: str = "autoencoder_anomaly_report.md",
    scored_lots_file_name: str = "demo_autoencoder_anomaly_scored_lots.csv",
    threshold_quantile: float = 0.95,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Train and evaluate an autoencoder-style reconstruction-error anomaly detector.
    """

    ensure_directories_exist()

    df = load_feature_table(feature_table_file_name)
    X, y, feature_columns = prepare_xy(df)

    lot_ids = (
        df["lot_id"].astype(str)
        if "lot_id" in df.columns
        else pd.Series([f"ROW_{i:04d}" for i in range(len(df))])
    )

    X_train, X_test, y_train, y_test, lot_train, lot_test = train_test_split(
        X,
        y,
        lot_ids,
        test_size=0.30,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    normal_mask = y_train.to_numpy() == 0
    X_train_normal_scaled = X_train_scaled[normal_mask]

    logger.info("Training autoencoder-style anomaly detector on normal-reference training lots.")

    hidden_layer_sizes = (4, 2, 4)

    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=0.0001,
        learning_rate_init=0.001,
        max_iter=3000,
        random_state=random_state,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(X_train_normal_scaled, X_train_normal_scaled)

    train_normal_errors = calculate_reconstruction_error(
        model=model,
        X_scaled=X_train_normal_scaled,
    )

    reconstruction_threshold = float(
        np.quantile(train_normal_errors, threshold_quantile)
    )

    test_errors = calculate_reconstruction_error(
        model=model,
        X_scaled=X_test_scaled,
    )

    predicted_label = (test_errors >= reconstruction_threshold).astype(int)
    y_test_array = y_test.to_numpy()

    metrics = evaluate_anomaly_detector(
        y_true=y_test_array,
        predicted_label=predicted_label,
        risk_scores=test_errors,
        top_k=10,
    )

    scored = pd.DataFrame(
        {
            "lot_id": lot_test.to_numpy(),
            "true_label": y_test_array.astype(int),
            "reconstruction_error": test_errors.astype(float),
            "predicted_label": predicted_label.astype(int),
        }
    )

    scored = scored.sort_values("reconstruction_error", ascending=False).reset_index(drop=True)
    scored["risk_rank"] = np.arange(1, len(scored) + 1)

    scored_path = DATA_PROCESSED_DIR / scored_lots_file_name
    scored.to_csv(scored_path, index=False)

    logger.info("Saved autoencoder anomaly scored lots to: %s", scored_path)

    top_suspicious_lots = []

    for _, row in scored.head(10).iterrows():
        top_suspicious_lots.append(
            {
                "rank": int(row["risk_rank"]),
                "lot_id": str(row["lot_id"]),
                "true_label": int(row["true_label"]),
                "reconstruction_error": float(row["reconstruction_error"]),
                "predicted_label": int(row["predicted_label"]),
            }
        )

    model_package = {
        "scaler": scaler,
        "model": model,
        "feature_columns": feature_columns,
        "reconstruction_error_threshold": reconstruction_threshold,
        "threshold_quantile": threshold_quantile,
        "training_design": "fit_on_normal_training_lots_only",
    }

    model_path = MODELS_DIR / model_file_name
    joblib.dump(model_package, model_path)

    logger.info("Saved autoencoder anomaly model package to: %s", model_path)

    report: dict[str, Any] = {
        "model_name": "Autoencoder-style reconstruction-error anomaly detection baseline",
        "model_file": str(model_path),
        "feature_table": str(DATA_PROCESSED_DIR / feature_table_file_name),
        "training_rows": int(len(X_train)),
        "normal_reference_training_rows": int(len(X_train_normal_scaled)),
        "test_rows": int(len(X_test)),
        "n_features": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "model_parameters": {
            "hidden_layer_sizes": list(hidden_layer_sizes),
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.0001,
            "learning_rate_init": 0.001,
            "max_iter": 3000,
            "random_state": random_state,
        },
        "threshold_quantile": float(threshold_quantile),
        "reconstruction_error_threshold": reconstruction_threshold,
        "training_loss": float(model.loss_),
        "n_iter": int(model.n_iter_),
        "evaluation_metrics": metrics,
        "top_suspicious_lots": top_suspicious_lots,
        "scored_lots_file": str(scored_path),
        "interpretation_note": (
            "This lightweight autoencoder-style baseline uses a neural reconstruction model "
            "trained only on normal-reference lots. Lots with high reconstruction error are "
            "treated as suspicious. In this controlled synthetic SPC demo, this baseline is "
            "used to compare neural reconstruction-error anomaly detection against PCA and "
            "Isolation Forest. The result should validate the workflow rather than imply real "
            "production performance."
        ),
    }

    report_path = REPORTS_DIR / report_file_name

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    logger.info("Saved autoencoder anomaly report JSON to: %s", report_path)

    markdown_path = REPORTS_DIR / markdown_file_name
    markdown_text = build_markdown_report(report)

    with markdown_path.open("w", encoding="utf-8") as file:
        file.write(markdown_text)

    logger.info("Saved autoencoder anomaly report Markdown to: %s", markdown_path)

    print("Autoencoder anomaly detection baseline summary")
    print("----------------------------------------------")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print()
    print("Top suspicious lots")
    print("-------------------")
    print(scored.head(10)[[
        "risk_rank",
        "lot_id",
        "true_label",
        "reconstruction_error",
        "predicted_label",
    ]])

    return report


def _demo() -> None:
    """
    Run the autoencoder anomaly detection baseline demo.
    """

    run_autoencoder_anomaly_baseline()


if __name__ == "__main__":
    _demo()