"""
Model family and anomaly baseline comparison utilities for WaferWatch.

This module compares four model families on the same selected SPC-enhanced
feature table:

1. Logistic Regression
2. Random Forest
3. Gradient Boosting
4. Isolation Forest

The first three models are supervised classifiers. Isolation Forest is an
unsupervised anomaly detection baseline fitted on normal-reference training lots.
"""

from __future__ import annotations

import json
from typing import Any

from src.models.evaluate import evaluate_saved_model
from src.models.gradient_boosting import run_gradient_boosting_baseline
from src.models.isolation_forest import run_isolation_forest_baseline
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


def format_metric(value: Any) -> str:
    """
    Format metric value for Markdown tables.
    """

    if value is None:
        return "None"

    return f"{value:.6f}"


def build_markdown_report(report: dict[str, Any]) -> str:
    """
    Build a Markdown report for the four-model comparison.
    """

    lr_metrics = report["models"]["logistic_regression"]["evaluation_metrics"]
    rf_metrics = report["models"]["random_forest"]["evaluation_metrics"]
    gb_metrics = report["models"]["gradient_boosting"]["evaluation_metrics"]
    if_metrics = report["models"]["isolation_forest"]["evaluation_metrics"]

    rf_minus_lr = report["metric_differences"]["random_forest_minus_logistic_regression"]
    gb_minus_lr = report["metric_differences"]["gradient_boosting_minus_logistic_regression"]
    if_minus_lr = report["metric_differences"]["isolation_forest_minus_logistic_regression"]

    lines: list[str] = []

    lines.append("# WaferWatch Model Family and Anomaly Baseline Comparison Report")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append(
        "This report compares Logistic Regression, Random Forest, Gradient Boosting, and Isolation Forest on the same selected SPC-enhanced feature table."
    )
    lines.append(
        "Logistic Regression, Random Forest, and Gradient Boosting are supervised classifiers. Isolation Forest is an unsupervised anomaly detection baseline fitted only on normal-reference training lots."
    )
    lines.append("")
    lines.append("## 2. Compared Models")
    lines.append("")
    lines.append("| Model | Learning type | Training label usage | Feature table | Model file |")
    lines.append("|---|---|---|---|---|")
    lines.append(
        "| Logistic Regression | Supervised classification | Uses labels during training | `demo_spc_selected_feature_table.csv` | `spc_selected_logistic_regression.joblib` |"
    )
    lines.append(
        "| Random Forest | Supervised classification | Uses labels during training | `demo_spc_selected_feature_table.csv` | `random_forest_selected_baseline.joblib` |"
    )
    lines.append(
        "| Gradient Boosting | Supervised classification | Uses labels during training | `demo_spc_selected_feature_table.csv` | `gradient_boosting_selected_baseline.joblib` |"
    )
    lines.append(
        "| Isolation Forest | Unsupervised anomaly detection | Labels used only for evaluation | `demo_spc_selected_feature_table.csv` | `isolation_forest_normal_reference.joblib` |"
    )
    lines.append("")
    lines.append("## 3. Main Metrics")
    lines.append("")
    lines.append("| Metric | Logistic Regression | Random Forest | Gradient Boosting | Isolation Forest |")
    lines.append("|---|---:|---:|---:|---:|")

    for metric_name in KEY_METRICS:
        lr_text = format_metric(lr_metrics.get(metric_name))
        rf_text = format_metric(rf_metrics.get(metric_name))
        gb_text = format_metric(gb_metrics.get(metric_name))
        if_text = format_metric(if_metrics.get(metric_name))

        lines.append(
            f"| {metric_name} | {lr_text} | {rf_text} | {gb_text} | {if_text} |"
        )

    lines.append("")
    lines.append("## 4. Metric Differences")
    lines.append("")
    lines.append(
        "Differences are calculated against the Logistic Regression supervised baseline."
    )
    lines.append("")
    lines.append("### 4.1 Random Forest minus Logistic Regression")
    lines.append("")
    lines.append("| Metric | Difference |")
    lines.append("|---|---:|")

    for metric_name in KEY_METRICS:
        lines.append(
            f"| {metric_name} | {format_metric(rf_minus_lr.get(metric_name))} |"
        )

    lines.append("")
    lines.append("### 4.2 Gradient Boosting minus Logistic Regression")
    lines.append("")
    lines.append("| Metric | Difference |")
    lines.append("|---|---:|")

    for metric_name in KEY_METRICS:
        lines.append(
            f"| {metric_name} | {format_metric(gb_minus_lr.get(metric_name))} |"
        )

    lines.append("")
    lines.append("### 4.3 Isolation Forest minus Logistic Regression")
    lines.append("")
    lines.append("| Metric | Difference |")
    lines.append("|---|---:|")

    for metric_name in KEY_METRICS:
        lines.append(
            f"| {metric_name} | {format_metric(if_minus_lr.get(metric_name))} |"
        )

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
    lines.append("## 7. Isolation Forest Top Suspicious Lots")
    lines.append("")
    lines.append("| Rank | Lot ID | True Label | Risk Score | Predicted Label |")
    lines.append("|---:|---|---:|---:|---:|")

    for row in report["models"]["isolation_forest"]["top_suspicious_lots"]:
        lines.append(
            f"| {row['rank']} | `{row['lot_id']}` | {row['true_label']} | "
            f"{row['risk_score']:.6f} | {row['predicted_label']} |"
        )

    lines.append("")
    lines.append("## 8. Interpretation")
    lines.append("")
    lines.append(report["interpretation_note"])
    lines.append("")

    return "\n".join(lines)


def run_model_family_comparison(
    report_file_name: str = "model_family_comparison_report.json",
    markdown_file_name: str = "model_family_comparison_report.md",
) -> dict[str, Any]:
    """
    Run four-model comparison.
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

    logger.info("Training and evaluating Isolation Forest anomaly detection baseline.")

    isolation_forest_report = run_isolation_forest_baseline(
        feature_table_file_name="demo_spc_selected_feature_table.csv",
        model_file_name="isolation_forest_normal_reference.joblib",
        report_file_name="isolation_forest_report.json",
        markdown_file_name="isolation_forest_report.md",
    )

    logistic_metrics = extract_metrics(logistic_evaluation_report)
    random_forest_metrics = extract_metrics(random_forest_report)
    gradient_boosting_metrics = extract_metrics(gradient_boosting_report)
    isolation_forest_metrics = extract_metrics(isolation_forest_report)

    rf_minus_lr = calculate_differences(
        baseline_metrics=logistic_metrics,
        comparison_metrics=random_forest_metrics,
    )

    gb_minus_lr = calculate_differences(
        baseline_metrics=logistic_metrics,
        comparison_metrics=gradient_boosting_metrics,
    )

    if_minus_lr = calculate_differences(
        baseline_metrics=logistic_metrics,
        comparison_metrics=isolation_forest_metrics,
    )

    report: dict[str, Any] = {
        "comparison_name": "Four-model family and anomaly baseline comparison",
        "feature_table": "demo_spc_selected_feature_table.csv",
        "models": {
            "logistic_regression": {
                "name": "Logistic Regression selected-feature baseline",
                "learning_type": "supervised_classification",
                "model_file": "spc_selected_logistic_regression.joblib",
                "training_report": logistic_training_report,
                "evaluation_metrics": logistic_metrics,
            },
            "random_forest": {
                "name": "Random Forest selected-feature baseline",
                "learning_type": "supervised_classification",
                "model_file": "random_forest_selected_baseline.joblib",
                "model_parameters": random_forest_report["model_parameters"],
                "evaluation_metrics": random_forest_metrics,
                "feature_importance": random_forest_report["feature_importance"],
            },
            "gradient_boosting": {
                "name": "Gradient Boosting selected-feature baseline",
                "learning_type": "supervised_classification",
                "model_file": "gradient_boosting_selected_baseline.joblib",
                "model_parameters": gradient_boosting_report["model_parameters"],
                "evaluation_metrics": gradient_boosting_metrics,
                "feature_importance": gradient_boosting_report["feature_importance"],
            },
            "isolation_forest": {
                "name": "Isolation Forest normal-reference anomaly detection baseline",
                "learning_type": "unsupervised_anomaly_detection",
                "model_file": "isolation_forest_normal_reference.joblib",
                "model_parameters": isolation_forest_report["model_parameters"],
                "evaluation_metrics": isolation_forest_metrics,
                "top_suspicious_lots": isolation_forest_report["top_suspicious_lots"],
            },
        },
        "metric_differences": {
            "random_forest_minus_logistic_regression": rf_minus_lr,
            "gradient_boosting_minus_logistic_regression": gb_minus_lr,
            "isolation_forest_minus_logistic_regression": if_minus_lr,
        },
        "interpretation_note": (
            "In the current controlled synthetic SPC demo, the three supervised classifiers "
            "achieve perfect headline test metrics on the selected SPC-enhanced feature table. "
            "This suggests that the selected SPC features strongly encode the injected anomaly "
            "mechanism. Isolation Forest also ranks all held-out failed lots at the top, producing "
            "perfect ROC-AUC and PR-AUC, but its default anomaly threshold creates more false alarms "
            "than the supervised models. This is a useful manufacturing-style trade-off: an "
            "unsupervised anomaly detector can be valuable when labels are rare or delayed, but it "
            "requires threshold tuning, top-K review, and false-alarm budget control before it can "
            "be used operationally. These results validate the demo workflow and should not be "
            "interpreted as production fab performance."
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

    print("Four-model family comparison summary")
    print("------------------------------------")
    print("Logistic Regression metrics:")
    print(json.dumps(logistic_metrics, indent=2, ensure_ascii=False))
    print()
    print("Random Forest metrics:")
    print(json.dumps(random_forest_metrics, indent=2, ensure_ascii=False))
    print()
    print("Gradient Boosting metrics:")
    print(json.dumps(gradient_boosting_metrics, indent=2, ensure_ascii=False))
    print()
    print("Isolation Forest metrics:")
    print(json.dumps(isolation_forest_metrics, indent=2, ensure_ascii=False))
    print()

    return report


def _demo() -> None:
    """
    Run the four-model comparison demo.
    """

    run_model_family_comparison()


if __name__ == "__main__":
    _demo()