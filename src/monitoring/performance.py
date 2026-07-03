"""
Performance monitoring utilities for WaferWatch.

This module monitors model performance after labels become available.

It compares a reference period and a current period using:
- accuracy
- precision
- recall
- F1
- ROC-AUC
- PR-AUC
- Precision@K
- Recall@K
- false alarms per 100 lots
- confusion matrix

This is different from drift monitoring:
- Drift monitoring asks whether feature distributions changed.
- Performance monitoring asks whether the model is still making good decisions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.models.evaluate import evaluate_model
from src.utils.config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, ensure_directories_exist
from src.utils.logger import get_logger


logger = get_logger(__name__)


def load_feature_columns_from_report(
    report_file_name: str = "spc_selected_training_report.json",
) -> list[str]:
    """
    Load the feature columns used during model training.
    """

    report_path = REPORTS_DIR / report_file_name

    if not report_path.exists():
        raise FileNotFoundError(
            f"Training report not found: {report_path}. "
            "Please run src.models.compare first."
        )

    with report_path.open("r", encoding="utf-8") as file:
        report = json.load(file)

    feature_columns = report.get("feature_columns")

    if not feature_columns:
        raise ValueError(
            f"No feature_columns found in {report_path}."
        )

    return feature_columns


def prepare_xy_for_monitoring(
    df: pd.DataFrame,
    feature_columns: list[str],
    label_column: str = "pass_fail_label",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare X and y for performance monitoring.
    """

    if label_column not in df.columns:
        raise ValueError(f"Label column not found: {label_column}")

    missing_features = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing_features:
        raise ValueError(f"Missing required feature columns: {missing_features}")

    X = df[feature_columns].copy()
    y = df[label_column].copy()

    return X, y


def evaluate_model_on_period(
    model: Any,
    period_df: pd.DataFrame,
    feature_columns: list[str],
    threshold: float = 0.50,
    label_column: str = "pass_fail_label",
    top_k: int = 10,
) -> dict[str, Any]:
    """
    Evaluate model performance on one monitoring period.
    """

    X, y = prepare_xy_for_monitoring(
        df=period_df,
        feature_columns=feature_columns,
        label_column=label_column,
    )

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X)[:, 1].tolist()
    else:
        y_score = model.predict(X).tolist()

    y_pred = [
        1 if score >= threshold else 0
        for score in y_score
    ]

    metrics = evaluate_model(
        y_true=y,
        y_pred=y_pred,
        y_score=y_score,
        top_k=min(top_k, len(y)),
    )

    return metrics


def simulate_current_performance_degradation(
    current_df: pd.DataFrame,
    label_column: str = "pass_fail_label",
) -> pd.DataFrame:
    """
    Simulate current-period performance degradation.

    This is only for demo purposes.

    We create a situation where SPC signals become less aligned with labels:
    - Some normal lots receive stronger SPC anomaly-like signals.
    - Some risky lots lose part of their SPC anomaly signal.

    In a real deployment, we would not simulate this. We would evaluate
    actual current data once labels become available.
    """

    degraded_df = current_df.copy()

    if "spc_violation_count" not in degraded_df.columns:
        return degraded_df

    if "spc_max_abs_zscore" not in degraded_df.columns:
        return degraded_df

    normal_indices = degraded_df[degraded_df[label_column] == 0].index.tolist()
    risky_indices = degraded_df[degraded_df[label_column] == 1].index.tolist()

    normal_to_shift = normal_indices[: max(1, len(normal_indices) // 3)]
    risky_to_shift = risky_indices[: max(1, len(risky_indices) // 3)]

    degraded_df.loc[normal_to_shift, "spc_violation_count"] = (
        degraded_df.loc[normal_to_shift, "spc_violation_count"] + 1
    )

    degraded_df.loc[normal_to_shift, "spc_max_abs_zscore"] = (
        degraded_df.loc[normal_to_shift, "spc_max_abs_zscore"] + 2.0
    )

    degraded_df.loc[risky_to_shift, "spc_violation_count"] = 0

    degraded_df.loc[risky_to_shift, "spc_max_abs_zscore"] = (
        degraded_df.loc[risky_to_shift, "spc_max_abs_zscore"] * 0.50
    )

    logger.info(
        "Simulated current performance degradation. Shifted %s normal lots and %s risky lots.",
        len(normal_to_shift),
        len(risky_to_shift),
    )

    return degraded_df


def compare_period_performance(
    reference_metrics: dict[str, Any],
    current_metrics: dict[str, Any],
    min_current_recall: float = 0.70,
    min_current_pr_auc: float = 0.70,
    max_recall_drop: float = 0.15,
    max_pr_auc_drop: float = 0.15,
    max_false_alarm_increase_per_100_lots: float = 5.0,
) -> dict[str, Any]:
    """
    Compare reference and current performance and decide whether to alert.
    """

    recall_drop = reference_metrics["recall"] - current_metrics["recall"]

    reference_pr_auc = reference_metrics["pr_auc"]
    current_pr_auc = current_metrics["pr_auc"]

    if reference_pr_auc is None or current_pr_auc is None:
        pr_auc_drop = None
    else:
        pr_auc_drop = reference_pr_auc - current_pr_auc

    false_alarm_increase = (
        current_metrics["false_alarms_per_100_lots"]
        - reference_metrics["false_alarms_per_100_lots"]
    )

    alert_reasons: list[str] = []

    if current_metrics["recall"] < min_current_recall:
        alert_reasons.append("current_recall_below_minimum")

    if recall_drop >= max_recall_drop:
        alert_reasons.append("recall_drop_exceeds_threshold")

    if current_pr_auc is not None and current_pr_auc < min_current_pr_auc:
        alert_reasons.append("current_pr_auc_below_minimum")

    if pr_auc_drop is not None and pr_auc_drop >= max_pr_auc_drop:
        alert_reasons.append("pr_auc_drop_exceeds_threshold")

    if false_alarm_increase >= max_false_alarm_increase_per_100_lots:
        alert_reasons.append("false_alarm_increase_exceeds_threshold")

    return {
        "performance_alert": bool(len(alert_reasons) > 0),
        "alert_reasons": alert_reasons,
        "metric_deltas": {
            "accuracy_delta_current_minus_reference": (
                current_metrics["accuracy"] - reference_metrics["accuracy"]
            ),
            "precision_delta_current_minus_reference": (
                current_metrics["precision"] - reference_metrics["precision"]
            ),
            "recall_delta_current_minus_reference": (
                current_metrics["recall"] - reference_metrics["recall"]
            ),
            "f1_delta_current_minus_reference": (
                current_metrics["f1"] - reference_metrics["f1"]
            ),
            "pr_auc_delta_current_minus_reference": (
                None
                if reference_pr_auc is None or current_pr_auc is None
                else current_pr_auc - reference_pr_auc
            ),
            "false_alarms_per_100_lots_delta_current_minus_reference": (
                false_alarm_increase
            ),
        },
        "thresholds": {
            "min_current_recall": min_current_recall,
            "min_current_pr_auc": min_current_pr_auc,
            "max_recall_drop": max_recall_drop,
            "max_pr_auc_drop": max_pr_auc_drop,
            "max_false_alarm_increase_per_100_lots": (
                max_false_alarm_increase_per_100_lots
            ),
        },
    }


def run_performance_monitoring_demo(
    feature_table_file_name: str = "demo_spc_selected_feature_table.csv",
    model_file_name: str = "spc_selected_logistic_regression.joblib",
    report_file_name: str = "performance_monitoring_report.json",
    threshold: float = 0.50,
) -> dict[str, Any]:
    """
    Run performance monitoring demo.
    """

    ensure_directories_exist()

    feature_table_path = PROCESSED_DATA_DIR / feature_table_file_name
    model_path = MODELS_DIR / model_file_name

    if not feature_table_path.exists():
        raise FileNotFoundError(
            f"Feature table not found: {feature_table_path}. "
            "Please run src.features.feature_selection first."
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Please run src.models.compare first."
        )

    logger.info("Loading selected feature table from: %s", feature_table_path)

    df = pd.read_csv(feature_table_path)

    midpoint = len(df) // 2

    reference_df = df.iloc[:midpoint].copy()
    current_df = df.iloc[midpoint:].copy()

    current_df = simulate_current_performance_degradation(current_df)

    logger.info("Loading model from: %s", model_path)

    model = joblib.load(model_path)

    feature_columns = load_feature_columns_from_report()

    reference_metrics = evaluate_model_on_period(
        model=model,
        period_df=reference_df,
        feature_columns=feature_columns,
        threshold=threshold,
    )

    current_metrics = evaluate_model_on_period(
        model=model,
        period_df=current_df,
        feature_columns=feature_columns,
        threshold=threshold,
    )

    comparison = compare_period_performance(
        reference_metrics=reference_metrics,
        current_metrics=current_metrics,
    )

    report: dict[str, Any] = {
        "model_file": str(model_path),
        "feature_table_file": str(feature_table_path),
        "reference_rows": int(len(reference_df)),
        "current_rows": int(len(current_df)),
        "threshold": threshold,
        "feature_columns": feature_columns,
        "reference_metrics": reference_metrics,
        "current_metrics": current_metrics,
        "performance_comparison": comparison,
        "interpretation_note": (
            "This demo intentionally simulates current-period performance degradation. "
            "In a real deployment, performance monitoring should use actual labels "
            "when they become available."
        ),
    }

    report_path = REPORTS_DIR / report_file_name

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    logger.info("Saved performance monitoring report to: %s", report_path)

    print("Performance monitoring summary")
    print("------------------------------")
    print("Reference metrics:")
    print(json.dumps(reference_metrics, indent=2, ensure_ascii=False))
    print()
    print("Current metrics:")
    print(json.dumps(current_metrics, indent=2, ensure_ascii=False))
    print()
    print("Comparison:")
    print(json.dumps(comparison, indent=2, ensure_ascii=False))

    return report


def _demo() -> None:
    """
    Run performance monitoring demo.
    """

    run_performance_monitoring_demo()


if __name__ == "__main__":
    _demo()