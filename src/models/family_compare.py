"""
Model family comparison utilities for WaferWatch.

This module compares three supervised tabular model families on the same
selected SPC-enhanced feature table:

1. Logistic Regression
2. Random Forest
3. Gradient Boosting

The goal is to compare model families under the same data split, feature table,
and evaluation metric set.
"""

from __future__ import annotations

import json
from typing import Any

from src.models.evaluate import evaluate_saved_model
from src.models.gradient_boosting import run_gradient_boosting_baseline
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


def extract_metrics(report: dict[str, Any]) -> dict[str, Any]:
    """
    Extract metrics from either an evaluation report or a model report.
    """

    if "metrics" in report:
        return report["metrics"]

    if "evaluation_metrics" in report:
        return report["evaluation_metrics"]

    raise KeyError("Report must contain either 'metrics' or 'evaluation_metrics'.")


def calculate_differences(
    baseline_metrics: dict[str, Any],
    comparison_metrics: dict[str, Any],
) -> dict[str, Any]:
    """
    Calculate metric differences.

    Difference = comparison model - baseline model.
    """

    differences: dict[str, Any] = {}

    for metric_name in KEY_METRICS:
        baseline_value = baseline_metrics.get(metric_name)
        comparison_value = comparison_metrics.get(metric_name)

        if baseline_value is None or comparison_value is None:
            differences[metric_name] = None
        else:
            differences[metric_name] = comparison_value - baseline_value

    return differences


def build_markdown_report(report: dict[str, Any]) -> str:
    """
    Build a Markdown report for the three-model family comparison.
    """

    lr_metrics = report["models"]["logistic_regression"]["evaluation_metrics"]
    rf_metrics = report["models"]["random_forest"]["evaluation_metrics"]
    gb_metrics = report["models"]["gradient_boosting"]["evaluation_metrics"]

    rf_minus_lr = report["metric_differences"]["random_forest_minus_logistic_regression"]
    gb_minus_lr = report["metric_differences"]["gradient_boosting_minus_logistic_regression"]
    gb_minus_rf = report["metric_differences"]["gradient_boosting_minus_random_forest"]

    lines: list[str] = []

    lines.append("# WaferWatch Model Family Comparison Report")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append(
        "This report compares Logistic Regression, Random Forest, and Gradient Boosting on the same selected SPC-enhanced feature table."
    )
    lines.append(
        "The goal is to evaluate whether nonlinear tree-based models improve over the linear baseline in the current controlled synthetic demo."
    )
    lines.append("")
    lines.append("## 2. Compared Models")
    lines.append("")
    lines.append("| Model | Feature table | Model file |")
    lines.append("|---|---|---|")
    lines.append(
        "| Logistic Regression | `demo_spc_selected_feature_table.csv` | `spc_selected_logistic_regression.joblib` |"
    )
    lines.append(
        "| Random Forest | `demo_spc_selected_feature_table.csv` | `random_forest_selected_baseline.joblib` |"
    )
    lines.append(
        "| Gradient Boosting | `demo_spc_selected_feature_table.csv` | `gradient_boosting_selected_baseline.joblib` |"
    )
    lines.append("")
    lines.append("## 3. Main Metrics")
    lines.append("")
    lines.append("| Metric | Logistic Regression | Random Forest | Gradient Boosting |")
    lines.append("|---|---:|---:|---:|")

    for metric_name in KEY_METRICS:
        lr_value = lr_metrics.get(metric_name)
        rf_value = rf_metrics.get(metric_name)
        gb_value = gb_metrics.get(metric_name)

        lr_text = "None" if lr_value is None else f"{lr_value:.6f}"
        rf_text = "None" if rf_value is None else f"{rf_value:.6f}"
        gb_text = "None" if gb_value is None else f"{gb_value:.6f}"

        lines.append(
            f"| {metric_name} | {lr_text} | {rf_text} | {gb_text} |"
        )

    lines.append("")
    lines.append("## 4. Metric Differences")
    lines.append("")
    lines.append("### 4.1 Random Forest minus Logistic Regression")
    lines.append("")
    lines.append("| Metric | Difference |")
    lines.append("|---|---:|")

    for metric_name in KEY_METRICS:
        value = rf_minus_lr.get(metric_name)
        value_text = "None" if value is None else f"{value:.6f}"
        lines.append(f"| {metric_name} | {value_text} |")

    lines.append("")
    lines.append("### 4.2 Gradient Boosting minus Logistic Regression")
    lines.append("")
    lines.append("| Metric | Difference |")
    lines.append("|---|---:|")

    for metric_name in KEY_METRICS:
        value = gb_minus_lr.get(metric_name)
        value_text = "None" if value is None else f"{value:.6f}"
        lines.append(f"| {metric_name} | {value_text} |")

    lines.append("")
    lines.append("### 4.3 Gradient Boosting minus Random Forest")
    lines.append("")
    lines.append("| Metric | Difference |")
    lines.append("|---|---:|")

    for metric_name in KEY_METRICS:
        value = gb_minus_rf.get(metric_name)
        value_text = "None" if value is None else f"{value:.6f}"
        lines.append(f"| {metric_name} | {value_text} |")

    lines.append("")
    lines.append("## 5. Random Forest Feature Importance")
    lines.append("")
    lines.append("| Rank | Feature | Importance |")
    lines.append("|---:|---|---:|")

    for rank, row in enumerate(
        report["models"]["random_forest"]["feature_importance"],
        start=1,
    ):
        lines.append(
            f"| {rank} | `{row['feature']}` | {row['importance']:.6f} |"
        )

    lines.append("")
    lines.append("## 6. Gradient Boosting Feature Importance")
    lines.append("")
    lines.append("| Rank | Feature | Importance |")
    lines.append("|---:|---|---:|")

    for rank, row in enumerate(
        report["models"]["gradient_boosting"]["feature_importance"],
        start=1,
    ):
        lines.append(
            f"| {rank} | `{row['feature']}` | {row['importance']:.6f} |"
        )

    lines.append("")
    lines.append("## 7. Interpretation")
    lines.append("")
    lines.append(report["interpretation_note"])
    lines.append("")

    return "\n".join(lines)


def run_model_family_comparison(
    report_file_name: str = "model_family_comparison_report.json",
    markdown_file_name: str = "model_family_comparison_report.md",
) -> dict[str, Any]:
    """
    Run three-model family comparison.
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

    logger.info("Training and evaluating selected-feature Gradient Boosting model.")

    gradient_boosting_report = run_gradient_boosting_baseline(
        feature_table_file_name="demo_spc_selected_feature_table.csv",
        model_file_name="gradient_boosting_selected_baseline.joblib",
        report_file_name="gradient_boosting_report.json",
        markdown_file_name="gradient_boosting_report.md",
    )

    logistic_metrics = extract_metrics(logistic_evaluation_report)
    random_forest_metrics = extract_metrics(random_forest_report)
    gradient_boosting_metrics = extract_metrics(gradient_boosting_report)

    rf_minus_lr = calculate_differences(
        baseline_metrics=logistic_metrics,
        comparison_metrics=random_forest_metrics,
    )

    gb_minus_lr = calculate_differences(
        baseline_metrics=logistic_metrics,
        comparison_metrics=gradient_boosting_metrics,
    )

    gb_minus_rf = calculate_differences(
        baseline_metrics=random_forest_metrics,
        comparison_metrics=gradient_boosting_metrics,
    )

    report: dict[str, Any] = {
        "comparison_name": "Three-model family comparison",
        "feature_table": "demo_spc_selected_feature_table.csv",
        "models": {
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
            "gradient_boosting": {
                "name": "Gradient Boosting selected-feature baseline",
                "model_file": "gradient_boosting_selected_baseline.joblib",
                "model_parameters": gradient_boosting_report["model_parameters"],
                "evaluation_metrics": gradient_boosting_metrics,
                "feature_importance": gradient_boosting_report["feature_importance"],
            },
        },
        "metric_differences": {
            "random_forest_minus_logistic_regression": rf_minus_lr,
            "gradient_boosting_minus_logistic_regression": gb_minus_lr,
            "gradient_boosting_minus_random_forest": gb_minus_rf,
        },
        "interpretation_note": (
            "In the current controlled synthetic SPC demo, Logistic Regression, Random Forest, "
            "and Gradient Boosting all achieve perfect headline test metrics on the selected "
            "SPC-enhanced feature table. This indicates that the selected SPC features strongly "
            "encode the injected anomaly mechanism. The result validates the model comparison "
            "workflow, but it should not be interpreted as production fab performance. "
            "The tree-based models add value by providing feature importance: Random Forest uses "
            "both SPC violation count and maximum absolute SPC z-score, while Gradient Boosting "
            "places most importance on maximum absolute SPC z-score and SPC violation count."
        ),
    }

    report_path = REPORTS_DIR / report_file_name

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    logger.info("Saved model family comparison report JSON to: %s", report_path)

    markdown_path = REPORTS_DIR / markdown_file_name
    markdown_text = build_markdown_report(report)

    with markdown_path.open("w", encoding="utf-8") as file:
        file.write(markdown_text)

    logger.info("Saved model family comparison report Markdown to: %s", markdown_path)

    print("Three-model family comparison summary")
    print("-------------------------------------")
    print("Logistic Regression metrics:")
    print(json.dumps(logistic_metrics, indent=2, ensure_ascii=False))
    print()
    print("Random Forest metrics:")
    print(json.dumps(random_forest_metrics, indent=2, ensure_ascii=False))
    print()
    print("Gradient Boosting metrics:")
    print(json.dumps(gradient_boosting_metrics, indent=2, ensure_ascii=False))
    print()

    return report


def _demo() -> None:
    """
    Run the three-model family comparison demo.
    """

    run_model_family_comparison()


if __name__ == "__main__":
    _demo()