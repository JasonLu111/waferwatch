"""
Isolation Forest anomaly detection baseline for WaferWatch.

This module trains an unsupervised anomaly detection baseline on the selected
SPC-enhanced feature table.

Important design choice:
- The Isolation Forest is fitted only on normal training lots.
- The pass/fail label is used only for evaluation, not for fitting.

This simulates a common manufacturing situation where confirmed failure labels
are rare, delayed, or incomplete.
"""

from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils.config import DATA_PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, ensure_directories_exist
from src.utils.logger import get_logger


logger = get_logger(__name__)


LABEL_COLUMN = "pass_fail_label"
NON_FEATURE_COLUMNS = ["lot_id", "timestamp", LABEL_COLUMN]


def load_feature_table(file_name: str) -> pd.DataFrame:
    """
    Load the selected feature table.
    """

    path = DATA_PROCESSED_DIR / file_name

    if not path.exists():
        raise FileNotFoundError(f"Feature table not found: {path}")

    logger.info("Loading feature table from: %s", path)
    return pd.read_csv(path)


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Prepare feature matrix and label vector.

    Labels are used only for evaluation.
    """

    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Missing required label column: {LABEL_COLUMN}")

    feature_columns = [
        column
        for column in df.columns
        if column not in NON_FEATURE_COLUMNS
    ]

    X = df[feature_columns].copy()
    y = df[LABEL_COLUMN].astype(int).copy()

    return X, y, feature_columns


def precision_at_k(y_true: np.ndarray, risk_scores: np.ndarray, k: int) -> float:
    """
    Calculate precision among the top-k highest-risk lots.
    """

    if len(y_true) == 0:
        return 0.0

    k = min(k, len(y_true))
    top_indices = np.argsort(risk_scores)[::-1][:k]

    return float(np.mean(y_true[top_indices]))


def recall_at_k(y_true: np.ndarray, risk_scores: np.ndarray, k: int) -> float:
    """
    Calculate recall captured by the top-k highest-risk lots.
    """

    positives = int(np.sum(y_true))

    if positives == 0:
        return 0.0

    k = min(k, len(y_true))
    top_indices = np.argsort(risk_scores)[::-1][:k]
    captured_positives = int(np.sum(y_true[top_indices]))

    return float(captured_positives / positives)


def evaluate_anomaly_detector(
    y_true: np.ndarray,
    predicted_label: np.ndarray,
    risk_scores: np.ndarray,
    top_k: int = 10,
) -> dict[str, Any]:
    """
    Evaluate unsupervised anomaly detection results using held-out labels.
    """

    cm = confusion_matrix(y_true, predicted_label, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, predicted_label)),
        "precision": float(precision_score(y_true, predicted_label, zero_division=0)),
        "recall": float(recall_score(y_true, predicted_label, zero_division=0)),
        "f1": float(f1_score(y_true, predicted_label, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted_label)),
        "matthews_corrcoef": float(matthews_corrcoef(y_true, predicted_label)),
        "roc_auc": float(roc_auc_score(y_true, risk_scores)),
        "pr_auc": float(average_precision_score(y_true, risk_scores)),
        "precision_at_k": precision_at_k(y_true, risk_scores, top_k),
        "recall_at_k": recall_at_k(y_true, risk_scores, top_k),
        "top_k": int(min(top_k, len(y_true))),
        "false_alarms_per_100_lots": float(fp / len(y_true) * 100),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }

    return metrics


def build_markdown_report(report: dict[str, Any]) -> str:
    """
    Build a Markdown report for the Isolation Forest baseline.
    """

    metrics = report["evaluation_metrics"]

    lines: list[str] = []

    lines.append("# WaferWatch Isolation Forest Baseline Report")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append(
        "This report documents an unsupervised Isolation Forest anomaly detection baseline."
    )
    lines.append(
        "The model is fitted only on normal training lots. Labels are used only for held-out evaluation."
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
    lines.append(f"- Number of estimators: `{report['model_parameters']['n_estimators']}`")
    lines.append(f"- Contamination setting: `{report['model_parameters']['contamination']}`")
    lines.append(f"- Random state: `{report['model_parameters']['random_state']}`")
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
    lines.append("| Rank | Lot ID | True Label | Risk Score | Predicted Label |")
    lines.append("|---:|---|---:|---:|---:|")

    for row in report["top_suspicious_lots"]:
        lines.append(
            f"| {row['rank']} | `{row['lot_id']}` | {row['true_label']} | "
            f"{row['risk_score']:.6f} | {row['predicted_label']} |"
        )

    lines.append("")
    lines.append("## 6. Interpretation")
    lines.append("")
    lines.append(report["interpretation_note"])
    lines.append("")

    return "\n".join(lines)


def run_isolation_forest_baseline(
    feature_table_file_name: str = "demo_spc_selected_feature_table.csv",
    model_file_name: str = "isolation_forest_normal_reference.joblib",
    report_file_name: str = "isolation_forest_report.json",
    markdown_file_name: str = "isolation_forest_report.md",
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Train and evaluate the Isolation Forest baseline.
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

    logger.info("Training Isolation Forest on normal-reference training lots.")

    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=random_state,
    )

    model.fit(X_train_normal_scaled)

    # In scikit-learn Isolation Forest:
    # - predict returns 1 for normal and -1 for anomaly.
    # - decision_function is higher for normal observations.
    raw_prediction = model.predict(X_test_scaled)
    predicted_label = np.where(raw_prediction == -1, 1, 0)

    decision_scores = model.decision_function(X_test_scaled)
    risk_scores = -decision_scores

    y_test_array = y_test.to_numpy()

    metrics = evaluate_anomaly_detector(
        y_true=y_test_array,
        predicted_label=predicted_label,
        risk_scores=risk_scores,
        top_k=10,
    )

    top_indices = np.argsort(risk_scores)[::-1][: min(10, len(risk_scores))]

    top_suspicious_lots = []

    for rank, index in enumerate(top_indices, start=1):
        top_suspicious_lots.append(
            {
                "rank": int(rank),
                "lot_id": str(lot_test.iloc[index]),
                "true_label": int(y_test_array[index]),
                "risk_score": float(risk_scores[index]),
                "predicted_label": int(predicted_label[index]),
            }
        )

    model_package = {
        "scaler": scaler,
        "model": model,
        "feature_columns": feature_columns,
        "training_design": "fit_on_normal_training_lots_only",
    }

    model_path = MODELS_DIR / model_file_name
    joblib.dump(model_package, model_path)

    logger.info("Saved Isolation Forest model package to: %s", model_path)

    report: dict[str, Any] = {
        "model_name": "Isolation Forest normal-reference anomaly detection baseline",
        "model_file": str(model_path),
        "feature_table": str(DATA_PROCESSED_DIR / feature_table_file_name),
        "training_rows": int(len(X_train)),
        "normal_reference_training_rows": int(len(X_train_normal_scaled)),
        "test_rows": int(len(X_test)),
        "n_features": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "model_parameters": {
            "n_estimators": 300,
            "contamination": "auto",
            "random_state": random_state,
        },
        "evaluation_metrics": metrics,
        "top_suspicious_lots": top_suspicious_lots,
        "interpretation_note": (
            "Isolation Forest is used here as an unsupervised anomaly detection baseline. "
            "The model is fitted only on normal-reference training lots, and held-out labels "
            "are used only for evaluation. This baseline is useful for discussing situations "
            "where confirmed failure labels are rare, delayed, or incomplete. Because this is "
            "still a controlled synthetic SPC demo, the result should validate the workflow "
            "rather than imply real production performance."
        ),
    }

    report_path = REPORTS_DIR / report_file_name

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    logger.info("Saved Isolation Forest report JSON to: %s", report_path)

    markdown_path = REPORTS_DIR / markdown_file_name
    markdown_text = build_markdown_report(report)

    with markdown_path.open("w", encoding="utf-8") as file:
        file.write(markdown_text)

    logger.info("Saved Isolation Forest report Markdown to: %s", markdown_path)

    print("Isolation Forest baseline summary")
    print("---------------------------------")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print()
    print("Top suspicious lots")
    print("-------------------")
    print(pd.DataFrame(top_suspicious_lots))

    return report


def _demo() -> None:
    """
    Run the Isolation Forest baseline demo.
    """

    run_isolation_forest_baseline()


if __name__ == "__main__":
    _demo()