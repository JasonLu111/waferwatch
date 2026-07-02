"""
Baseline model training utilities for WaferWatch.

This module trains a simple Logistic Regression baseline model.

At this stage, the goal is not to achieve high model performance.
The goal is to build a reproducible training script that can:
- load a feature table
- split data into train/test sets
- train a baseline classifier
- evaluate basic classification metrics
- save the trained model
- save a training report
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, ensure_directories_exist
from src.utils.logger import get_logger


logger = get_logger(__name__)


def prepare_xy(
    df: pd.DataFrame,
    label_column: str = "pass_fail_label",
    id_column: str = "lot_id",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare feature matrix X and target vector y.

    Parameters
    ----------
    df:
        Feature table.
    label_column:
        Target label column.
    id_column:
        Identifier column that should not be used for training.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.Series]
        X feature matrix and y target vector.
    """

    if label_column not in df.columns:
        raise ValueError(f"Label column not found: {label_column}")

    columns_to_drop = [label_column]

    if id_column in df.columns:
        columns_to_drop.append(id_column)

    X = df.drop(columns=columns_to_drop)
    y = df[label_column]

    # Keep numeric columns only for this first baseline.
    X = X.select_dtypes(include=["number"])

    if X.empty:
        raise ValueError("No numeric feature columns available for training.")

    return X, y


def train_logistic_regression_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """
    Train a Logistic Regression baseline model.

    Logistic Regression benefits from feature scaling, so we use a pipeline:
    StandardScaler -> LogisticRegression.
    """

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    logger.info("Training Logistic Regression baseline model.")

    model.fit(X_train, y_train)

    logger.info("Model training completed.")

    return model


def evaluate_binary_classifier(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """
    Evaluate a binary classifier using basic metrics.

    Returns accuracy, precision, recall, F1, and ROC-AUC when possible.
    """

    y_pred = model.predict(X_test)

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    if hasattr(model, "predict_proba") and len(set(y_test)) == 2:
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
    else:
        metrics["roc_auc"] = None

    return metrics


def train_from_feature_table(
    input_file_name: str,
    label_column: str = "pass_fail_label",
    id_column: str = "lot_id",
    model_file_name: str = "logistic_regression_baseline.joblib",
    report_file_name: str = "baseline_training_report.json",
) -> dict[str, Any]:
    """
    Load a feature table, train a baseline model, save the model, and save a report.
    """

    ensure_directories_exist()

    input_path = PROCESSED_DATA_DIR / input_file_name

    if not input_path.exists():
        raise FileNotFoundError(
            f"Feature table not found: {input_path}. "
            "Please run feature engineering first."
        )

    logger.info("Loading feature table from: %s", input_path)

    df = pd.read_csv(input_path)
    X, y = prepare_xy(df, label_column=label_column, id_column=id_column)

    logger.info("Feature matrix shape: %s rows, %s columns", X.shape[0], X.shape[1])
    logger.info("Label distribution: %s", y.value_counts().to_dict())

    stratify = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=stratify,
    )

    model = train_logistic_regression_baseline(X_train, y_train)

    metrics = evaluate_binary_classifier(model, X_test, y_test)

    model_path = MODELS_DIR / model_file_name
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    logger.info("Saved model to: %s", model_path)

    report: dict[str, Any] = {
        "model_name": "Logistic Regression Baseline",
        "input_file": str(input_path),
        "model_path": str(model_path),
        "n_rows": int(df.shape[0]),
        "n_features": int(X.shape[1]),
        "feature_columns": X.columns.tolist(),
        "label_distribution": {
            str(label): int(count)
            for label, count in y.value_counts().to_dict().items()
        },
        "test_size": 0.30,
        "random_state": 42,
        "metrics": metrics,
    }

    report_path = REPORTS_DIR / report_file_name

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    logger.info("Saved training report to: %s", report_path)

    return report


def create_demo_training_feature_table(
    output_file_name: str = "demo_training_feature_table.csv",
    n_samples: int = 60,
) -> Path:
    """
    Create a larger demo feature table for model training.

    The earlier demo_feature_table.csv has only 3 rows, which is too small
    for a meaningful train/test split. This function creates a small synthetic
    training dataset so we can test the training pipeline safely.
    """

    ensure_directories_exist()

    rng = np.random.default_rng(seed=42)

    sensor_mean = rng.normal(loc=3.3, scale=0.35, size=n_samples)
    sensor_std = rng.normal(loc=2.8, scale=0.25, size=n_samples)
    sensor_min = sensor_mean - rng.uniform(1.5, 2.2, size=n_samples)
    sensor_max = sensor_mean + rng.uniform(1.5, 2.2, size=n_samples)
    sensor_missing_count = rng.integers(low=0, high=3, size=n_samples)
    sensor_missing_ratio = sensor_missing_count / 20

    # Create a synthetic risk rule.
    risk_score = (
        0.9 * sensor_mean
        + 0.7 * sensor_std
        + 0.5 * sensor_missing_count
        + rng.normal(loc=0.0, scale=0.5, size=n_samples)
    )

    threshold = np.quantile(risk_score, 0.75)
    pass_fail_label = (risk_score > threshold).astype(int)

    demo_df = pd.DataFrame(
        {
            "lot_id": [f"LOT_{i:03d}" for i in range(1, n_samples + 1)],
            "sensor_mean": sensor_mean,
            "sensor_std": sensor_std,
            "sensor_min": sensor_min,
            "sensor_max": sensor_max,
            "sensor_missing_count": sensor_missing_count,
            "sensor_missing_ratio": sensor_missing_ratio,
            "pass_fail_label": pass_fail_label,
        }
    )

    output_path = PROCESSED_DATA_DIR / output_file_name
    demo_df.to_csv(output_path, index=False)

    logger.info("Created demo training feature table at: %s", output_path)

    return output_path


def _demo() -> None:
    """
    Run a model training demo.
    """

    create_demo_training_feature_table()

    report = train_from_feature_table(
        input_file_name="demo_training_feature_table.csv",
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _demo()