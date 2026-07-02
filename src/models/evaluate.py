"""
Model evaluation utilities for WaferWatch.

This module evaluates a saved binary classification model using metrics that are
more suitable for imbalanced manufacturing anomaly detection than accuracy alone.

Metrics include:
- accuracy
- precision
- recall
- F1
- ROC-AUC
- PR-AUC
- balanced accuracy
- Matthews correlation coefficient
- confusion matrix
- Precision@K
- Recall@K
- false alarms per 100 lots
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
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

from src.models.train import prepare_xy
from src.utils.config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, ensure_directories_exist
from src.utils.logger import get_logger


logger = get_logger(__name__)


def calculate_precision_at_k(
    y_true: pd.Series,
    y_score: list[float],
    k: int,
) -> float:
    """
    Calculate Precision@K.

    Precision@K answers:
    Among the top K highest-risk lots, how many are truly risky?
    """

    if k <= 0:
        raise ValueError("k must be greater than 0.")

    result_df = pd.DataFrame(
        {
            "y_true": y_true.reset_index(drop=True),
            "y_score": y_score,
        }
    )

    top_k = result_df.sort_values("y_score", ascending=False).head(k)

    if len(top_k) == 0:
        return 0.0

    return float(top_k["y_true"].sum() / len(top_k))


def calculate_recall_at_k(
    y_true: pd.Series,
    y_score: list[float],
    k: int,
) -> float:
    """
    Calculate Recall@K.

    Recall@K answers:
    Among all truly risky lots, how many are captured in the top K list?
    """

    total_positive = int(y_true.sum())

    if total_positive == 0:
        return 0.0

    result_df = pd.DataFrame(
        {
            "y_true": y_true.reset_index(drop=True),
            "y_score": y_score,
        }
    )

    top_k = result_df.sort_values("y_score", ascending=False).head(k)

    return float(top_k["y_true"].sum() / total_positive)


def evaluate_model(
    y_true: pd.Series,
    y_pred: list[int],
    y_score: list[float],
    top_k: int = 10,
) -> dict[str, Any]:
    """
    Evaluate binary classification predictions.

    Parameters
    ----------
    y_true:
        True labels.
    y_pred:
        Predicted class labels.
    y_score:
        Predicted risk scores or probabilities for positive class.
    top_k:
        Number of highest-risk lots used for Precision@K and Recall@K.

    Returns
    -------
    dict
        Evaluation metrics.
    """

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    true_negative = int(cm[0, 0])
    false_positive = int(cm[0, 1])
    false_negative = int(cm[1, 0])
    true_positive = int(cm[1, 1])

    n_lots = len(y_true)
    false_alarms_per_100_lots = (
        float(false_positive / n_lots * 100) if n_lots > 0 else 0.0
    )

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "matthews_corrcoef": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc": None,
        "pr_auc": None,
        "precision_at_k": calculate_precision_at_k(y_true, y_score, top_k),
        "recall_at_k": calculate_recall_at_k(y_true, y_score, top_k),
        "top_k": int(top_k),
        "false_alarms_per_100_lots": false_alarms_per_100_lots,
        "confusion_matrix": {
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_positive": true_positive,
        },
    }

    if len(set(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_score))

    return metrics


def evaluate_saved_model(
    feature_table_file_name: str = "demo_training_feature_table.csv",
    model_file_name: str = "logistic_regression_baseline.joblib",
    report_file_name: str = "baseline_evaluation_report.json",
    label_column: str = "pass_fail_label",
    id_column: str = "lot_id",
    test_size: float = 0.30,
    random_state: int = 42,
    top_k: int = 10,
) -> dict[str, Any]:
    """
    Load a saved model and evaluate it on the same deterministic test split.
    """

    ensure_directories_exist()

    feature_table_path = PROCESSED_DATA_DIR / feature_table_file_name
    model_path = MODELS_DIR / model_file_name

    if not feature_table_path.exists():
        raise FileNotFoundError(
            f"Feature table not found: {feature_table_path}. "
            "Please run src.models.train first."
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Please run src.models.train first."
        )

    logger.info("Loading feature table from: %s", feature_table_path)
    df = pd.read_csv(feature_table_path)

    X, y = prepare_xy(
        df=df,
        label_column=label_column,
        id_column=id_column,
    )

    stratify = y if y.value_counts().min() >= 2 else None

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    logger.info("Loading model from: %s", model_path)
    model = joblib.load(model_path)

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1].tolist()
    else:
        y_score = y_pred.tolist()

    effective_top_k = min(top_k, len(y_test))

    metrics = evaluate_model(
        y_true=y_test,
        y_pred=y_pred.tolist(),
        y_score=y_score,
        top_k=effective_top_k,
    )

    report: dict[str, Any] = {
        "model_file": str(model_path),
        "feature_table_file": str(feature_table_path),
        "n_test_rows": int(len(y_test)),
        "n_features": int(X_test.shape[1]),
        "feature_columns": X_test.columns.tolist(),
        "test_size": test_size,
        "random_state": random_state,
        "metrics": metrics,
    }

    report_path = REPORTS_DIR / report_file_name

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    logger.info("Saved evaluation report to: %s", report_path)

    return report


def _demo() -> None:
    """
    Run evaluation demo using the baseline model created by src.models.train.
    """

    report = evaluate_saved_model()

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _demo()