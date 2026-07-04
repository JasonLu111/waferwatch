"""
Model comparison utilities for WaferWatch.

This module supports two comparison workflows:

1. Feature strategy comparison:
   - Sensor aggregate only
   - Sensor aggregate + SPC
   - Sensor aggregate + SPC + feature selection

2. Model family comparison:
   - Logistic Regression on selected SPC features
   - Random Forest on selected SPC features

The purpose is to compare both feature engineering choices and model family
choices under the same controlled synthetic demo setting.
"""

from __future__ import annotations

import json
from typing import Any

from src.features.build_features import (
    build_combined_features_from_csv,
    build_features_from_csv,
)
from src.features.feature_selection import select_features_from_csv
from src.models.evaluate import evaluate_saved_model
from src.models.random_forest import run_random_forest_baseline
from src.models.train import train_from_feature_table
from src.utils.config import REPORTS_DIR, ensure_directories_exist
from src.utils.logger import get_logger


logger = get_logger(__name__)


KEY_METRICS = [
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


def extract_key_metrics(evaluation_report: dict[str, Any]) -> dict[str, Any]:
    """
    Extract key metrics from an evaluation report.

    Supports both:
    - evaluate_saved_model() reports with key "metrics"
    - Random Forest reports with key "evaluation_metrics"
    """

    if "metrics" in evaluation_report:
        metrics = evaluation_report["metrics"]
    elif "evaluation_metrics" in evaluation_report:
        metrics = evaluation_report["evaluation_metrics"]
    else:
        raise KeyError("Evaluation report must contain 'metrics' or 'evaluation_metrics'.")

    return {
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "matthews_corrcoef": metrics["matthews_corrcoef"],
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "precision_at_k": metrics["precision_at_k"],
        "recall_at_k": metrics["recall_at_k"],
        "false_alarms_per_100_lots": metrics["false_alarms_per_100_lots"],
        "confusion_matrix": metrics["confusion_matrix"],
    }


def calculate_metric_differences(
    baseline_metrics: dict[str, Any],
    comparison_metrics: dict[str, Any],
) -> dict[str, Any]:
    """
    Calculate metric differences between two models.

    Difference = comparison model - baseline model.
    """

    differences: dict[str, Any] = {}

    for metric_name in KEY_METRICS:
        baseline_value = baseline_metrics[metric_name]
        comparison_value = comparison_metrics[metric_name]

        if baseline_value is not None and comparison_value is not None:
            differences[metric_name] = comparison_value - baseline_value
        else:
            differences[metric_name] = None

    return differences


def compare_feature_strategies(
    report_file_name: str = "model_comparison_with_selection_report.json",
) -> dict[str, Any]:
    """
    Compare three feature strategies:
    A. Aggregate only
    B. Aggregate + SPC
    C. Aggregate + SPC + feature selection
    """

    ensure_directories_exist()

    logger.info("Building aggregate-only feature table.")

    aggregate_feature_df, aggregate_feature_report = build_features_from_csv(
        input_file_name="demo_spc_sensor_data_processed.csv",
        output_file_name="demo_spc_aggregate_feature_table.csv",
    )

    logger.info("Building combined aggregate + SPC feature table.")

    combined_feature_df, combined_feature_report = build_combined_features_from_csv(
        input_file_name="demo_spc_sensor_data_processed.csv",
        output_file_name="demo_spc_combined_feature_table.csv",
    )

    logger.info("Building selected aggregate + SPC feature table.")

    selected_feature_df, selected_feature_report = select_features_from_csv(
        input_file_name="demo_spc_combined_feature_table.csv",
        output_file_name="demo_spc_selected_feature_table.csv",
    )

    logger.info("Training aggregate-only baseline model.")

    aggregate_training_report = train_from_feature_table(
        input_file_name="demo_spc_aggregate_feature_table.csv",
        model_file_name="spc_aggregate_logistic_regression.joblib",
        report_file_name="spc_aggregate_training_report.json",
    )

    logger.info("Training SPC-enhanced baseline model.")

    combined_training_report = train_from_feature_table(
        input_file_name="demo_spc_combined_feature_table.csv",
        model_file_name="spc_combined_logistic_regression.joblib",
        report_file_name="spc_combined_training_report.json",
    )

    logger.info("Training feature-selected SPC baseline model.")

    selected_training_report = train_from_feature_table(
        input_file_name="demo_spc_selected_feature_table.csv",
        model_file_name="spc_selected_logistic_regression.joblib",
        report_file_name="spc_selected_training_report.json",
    )

    logger.info("Evaluating aggregate-only baseline model.")

    aggregate_evaluation_report = evaluate_saved_model(
        feature_table_file_name="demo_spc_aggregate_feature_table.csv",
        model_file_name="spc_aggregate_logistic_regression.joblib",
        report_file_name="spc_aggregate_evaluation_report.json",
    )

    logger.info("Evaluating SPC-enhanced baseline model.")

    combined_evaluation_report = evaluate_saved_model(
        feature_table_file_name="demo_spc_combined_feature_table.csv",
        model_file_name="spc_combined_logistic_regression.joblib",
        report_file_name="spc_combined_evaluation_report.json",
    )

    logger.info("Evaluating feature-selected SPC baseline model.")

    selected_evaluation_report = evaluate_saved_model(
        feature_table_file_name="demo_spc_selected_feature_table.csv",
        model_file_name="spc_selected_logistic_regression.joblib",
        report_file_name="spc_selected_evaluation_report.json",
    )

    aggregate_metrics = extract_key_metrics(aggregate_evaluation_report)
    combined_metrics = extract_key_metrics(combined_evaluation_report)
    selected_metrics = extract_key_metrics(selected_evaluation_report)

    combined_minus_aggregate = calculate_metric_differences(
        baseline_metrics=aggregate_metrics,
        comparison_metrics=combined_metrics,
    )

    selected_minus_aggregate = calculate_metric_differences(
        baseline_metrics=aggregate_metrics,
        comparison_metrics=selected_metrics,
    )

    selected_minus_combined = calculate_metric_differences(
        baseline_metrics=combined_metrics,
        comparison_metrics=selected_metrics,
    )

    report: dict[str, Any] = {
        "comparison_name": "Feature strategy comparison",
        "dataset": "demo_spc_sensor_data_processed.csv",
        "model_a": {
            "name": "Sensor Aggregate Logistic Regression",
            "feature_table": "demo_spc_aggregate_feature_table.csv",
            "model_file": "spc_aggregate_logistic_regression.joblib",
            "n_rows": aggregate_feature_report["output_rows"],
            "n_columns": aggregate_feature_report["output_columns"],
            "training_report": aggregate_training_report,
            "evaluation_metrics": aggregate_metrics,
        },
        "model_b": {
            "name": "Sensor Aggregate + SPC Logistic Regression",
            "feature_table": "demo_spc_combined_feature_table.csv",
            "model_file": "spc_combined_logistic_regression.joblib",
            "n_rows": combined_feature_report["output_rows"],
            "n_columns": combined_feature_report["output_columns"],
            "spc_summary": combined_feature_report["spc_summary"],
            "training_report": combined_training_report,
            "evaluation_metrics": combined_metrics,
        },
        "model_c": {
            "name": "Sensor Aggregate + SPC + Feature Selection Logistic Regression",
            "feature_table": "demo_spc_selected_feature_table.csv",
            "model_file": "spc_selected_logistic_regression.joblib",
            "n_rows": selected_feature_report["output_rows"],
            "n_columns": selected_feature_report["output_columns"],
            "n_final_features": selected_feature_report["n_final_features"],
            "final_feature_columns": selected_feature_report["final_feature_columns"],
            "removed_columns": {
                "high_missing": selected_feature_report["removed_high_missing_columns"],
                "non_numeric": selected_feature_report[
                    "removed_non_numeric_columns_for_baseline"
                ],
                "constant": selected_feature_report["removed_constant_columns"],
                "near_constant": selected_feature_report["removed_near_constant_columns"],
                "high_correlation": selected_feature_report[
                    "removed_high_correlation_columns"
                ],
            },
            "training_report": selected_training_report,
            "evaluation_metrics": selected_metrics,
        },
        "metric_differences": {
            "model_b_minus_model_a": combined_minus_aggregate,
            "model_c_minus_model_a": selected_minus_aggregate,
            "model_c_minus_model_b": selected_minus_combined,
        },
        "interpretation_note": (
            "This is a controlled demo experiment using synthetic SPC anomalies. "
            "Strong performance should be interpreted as evidence that SPC-derived "
            "features can encode the injected anomaly mechanism, not as proof of "
            "real fab deployment performance."
        ),
    }

    report_path = REPORTS_DIR / report_file_name

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    logger.info("Saved feature strategy comparison report to: %s", report_path)

    return report


def build_model_family_markdown_report(
    report: dict[str, Any],
) -> str:
    """
    Build a Markdown report for model family comparison.
    """

    logistic_metrics = report["logistic_regression"]["evaluation_metrics"]
    random_forest_metrics = report["random_forest"]["evaluation_metrics"]
    differences = report["metric_differences_random_forest_minus_logistic_regression"]

    lines: list[str] = []

    lines.append("# WaferWatch Model Family Comparison Report")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append(
        "This report compares Logistic Regression and Random Forest on the same selected SPC-enhanced feature table."
    )
    lines.append(
        "The goal is to test whether a nonlinear tabular model improves over the current linear baseline."
    )
    lines.append("")
    lines.append("## 2. Compared Models")
    lines.append("")
    lines.append("| Model | Feature table | Model file |")
    lines.append("|---|---|---|")
    lines.append(
        f"| Logistic Regression | `{report['feature_table']}` | `{report['logistic_regression']['model_file']}` |"
    )
    lines.append(
        f"| Random Forest | `{report['feature_table']}` | `{report['random_forest']['model_file']}` |"
    )
    lines.append("")
    lines.append("## 3. Metrics")
    lines.append("")
    lines.append("| Metric | Logistic Regression | Random Forest | RF - LR |")
    lines.append("|---|---:|---:|---:|")

    for metric_name in KEY_METRICS:
        lr_value = logistic_metrics.get(metric_name)
        rf_value = random_forest_metrics.get(metric_name)
        diff_value = differences.get(metric_name)

        lr_text = "None" if lr_value is None else f"{lr_value:.6f}"
        rf_text = "None" if rf_value is None else f"{rf_value:.6f}"
        diff_text = "None" if diff_value is None else f"{diff_value:.6f}"

        lines.append(
            f"| {metric_name} | {lr_text} | {rf_text} | {diff_text} |"
        )

    lines.append("")
    lines.append("## 4. Random Forest Feature Importance")
    lines.append("")
    lines.append("| Rank | Feature | Importance |")
    lines.append("|---:|---|---:|")

    for rank, row in enumerate(report["random_forest"]["feature_importance"], start=1):
        lines.append(
            f"| {rank} | `{row['feature']}` | {row['importance']:.6f} |"
        )

    lines.append("")
    lines.append("## 5. Interpretation")
    lines.append("")
    lines.append(report["interpretation_note"])
    lines.append("")

    return "\n".join(lines)


def compare_model_families(
    report_file_name: str = "model_family_comparison_report.json",
    markdown_file_name: str = "model_family_comparison_report.md",
) -> dict[str, Any]:
    """
    Compare Logistic Regression and Random Forest on selected SPC features.
    """

    ensure_directories_exist()

    logger.info("Training selected-feature Logistic Regression model.")

    logistic_training_report = train_from_feature_table(
        input_file_name="demo_spc_selected_feature_table.csv",
        model_file_name="spc_selected_logistic_regression.joblib",
        report_file_name="spc_selected_training_report.json",
    )

    logger.info("Evaluating selected-feature Logistic Regression model.")

    logistic_evaluation_report = evaluate_saved_model(
        feature_table_file_name="demo_spc_selected_feature_table.csv",
        model_file_name="spc_selected_logistic_regression.joblib",
        report_file_name="spc_selected_evaluation_report.json",
    )

    logger.info("Training and evaluating selected-feature Random Forest model.")

    random_forest_report = run_random_forest_baseline(
        feature_table_file_name="demo_spc_selected_feature_table.csv",
        model_file_name="random_forest_selected_baseline.joblib",
        report_file_name="random_forest_report.json",
        markdown_file_name="random_forest_report.md",
    )

    logistic_metrics = extract_key_metrics(logistic_evaluation_report)
    random_forest_metrics = extract_key_metrics(random_forest_report)

    metric_differences = calculate_metric_differences(
        baseline_metrics=logistic_metrics,
        comparison_metrics=random_forest_metrics,
    )

    report: dict[str, Any] = {
        "comparison_name": "Model family comparison",
        "feature_table": "demo_spc_selected_feature_table.csv",
        "logistic_regression": {
            "name": "Logistic Regression selected-feature baseline",
            "model_file": "spc_selected_logistic_regression.joblib",
            "training_report": logistic_training_report,
            "evaluation_metrics": logistic_metrics,
        },
        "random_forest": {
            "name": "Random Forest selected-feature baseline",
            "model_file": "random_forest_selected_baseline.joblib",
            "model_parameters": random_forest_report["model_parameters"],
            "evaluation_metrics": random_forest_metrics,
            "feature_importance": random_forest_report["feature_importance"],
        },
        "metric_differences_random_forest_minus_logistic_regression": metric_differences,
        "interpretation_note": (
            "In the current controlled synthetic demo, both Logistic Regression and "
            "Random Forest achieve perfect test metrics on the selected SPC-enhanced "
            "feature table. This suggests the selected SPC features strongly encode "
            "the injected anomaly mechanism. It does not prove production performance. "
            "Random Forest additionally provides feature importance, showing that "
            "SPC violation count and maximum absolute SPC z-score dominate the demo signal."
        ),
    }

    report_path = REPORTS_DIR / report_file_name

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    logger.info("Saved model family comparison report JSON to: %s", report_path)

    markdown_path = REPORTS_DIR / markdown_file_name
    markdown_text = build_model_family_markdown_report(report)

    with markdown_path.open("w", encoding="utf-8") as file:
        file.write(markdown_text)

    logger.info("Saved model family comparison report Markdown to: %s", markdown_path)

    print("Model family comparison summary")
    print("-------------------------------")
    print("Logistic Regression metrics:")
    print(json.dumps(logistic_metrics, indent=2, ensure_ascii=False))
    print()
    print("Random Forest metrics:")
    print(json.dumps(random_forest_metrics, indent=2, ensure_ascii=False))
    print()
    print("Metric differences: Random Forest - Logistic Regression")
    print(json.dumps(metric_differences, indent=2, ensure_ascii=False))

    return report


def _demo() -> None:
    """
    Run comparison demos.
    """

    feature_report = compare_feature_strategies()
    family_report = compare_model_families()

    print()
    print("Comparison reports saved.")
    print(
        json.dumps(
            {
                "feature_strategy_comparison": feature_report["comparison_name"],
                "model_family_comparison": family_report["comparison_name"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    _demo()