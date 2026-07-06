"""
Repeated-seed robustness experiments for WaferWatch.

This module repeats the main six-model baseline comparison across multiple
train-test split random seeds.

Purpose:
- Test whether conclusions depend on one lucky train-test split.
- Report mean and standard deviation of key metrics.
- Strengthen the robustness section of the project.
"""

from __future__ import annotations

import json
from typing import Any, Callable

import pandas as pd
from sklearn.model_selection import train_test_split

from src.models import robustness_ablation as ra
from src.models.isolation_forest import load_feature_table, prepare_xy
from src.utils.config import DATA_PROCESSED_DIR, REPORTS_DIR, ensure_directories_exist
from src.utils.logger import get_logger


logger = get_logger(__name__)


DEFAULT_SEEDS = [7, 21, 42, 84, 168]

MODEL_FUNCTIONS: list[tuple[str, str, Callable[..., dict[str, Any]]]] = [
    (
        "Logistic Regression",
        "supervised_classification",
        ra.train_evaluate_logistic_regression,
    ),
    (
        "Random Forest",
        "supervised_classification",
        ra.train_evaluate_random_forest,
    ),
    (
        "Gradient Boosting",
        "supervised_classification",
        ra.train_evaluate_gradient_boosting,
    ),
    (
        "Isolation Forest",
        "unsupervised_anomaly_detection",
        ra.train_evaluate_isolation_forest,
    ),
    (
        "PCA Anomaly",
        "unsupervised_anomaly_detection",
        ra.train_evaluate_pca_anomaly,
    ),
    (
        "Autoencoder Anomaly",
        "unsupervised_anomaly_detection",
        ra.train_evaluate_autoencoder_anomaly,
    ),
]

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


def run_single_seed(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
) -> list[dict[str, Any]]:
    """
    Run the six-model baseline comparison for one train-test split seed.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=seed,
        stratify=y,
    )

    rows: list[dict[str, Any]] = []

    original_random_state = ra.RANDOM_STATE
    ra.RANDOM_STATE = seed

    try:
        for model_name, learning_type, model_function in MODEL_FUNCTIONS:
            logger.info(
                "Running repeated-seed experiment with seed=%s, model='%s'.",
                seed,
                model_name,
            )

            metrics = model_function(X_train, X_test, y_train, y_test)
            cm = metrics["confusion_matrix"]

            row = {
                "seed": int(seed),
                "model_name": model_name,
                "learning_type": learning_type,
            }

            for metric_name in KEY_METRICS:
                row[metric_name] = metrics[metric_name]

            row.update(
                {
                    "true_negative": cm["true_negative"],
                    "false_positive": cm["false_positive"],
                    "false_negative": cm["false_negative"],
                    "true_positive": cm["true_positive"],
                }
            )

            rows.append(row)

    finally:
        ra.RANDOM_STATE = original_random_state

    return rows


def build_summary_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build mean, standard deviation, min, and max metrics by model.
    """

    aggregation = {
        metric_name: ["mean", "std", "min", "max"]
        for metric_name in KEY_METRICS
    }

    summary_df = results_df.groupby(
        ["model_name", "learning_type"],
        as_index=False,
    ).agg(aggregation)

    flat_columns = []

    for column in summary_df.columns:
        if isinstance(column, tuple):
            clean_name = "_".join([part for part in column if part])
            flat_columns.append(clean_name)
        else:
            flat_columns.append(column)

    summary_df.columns = flat_columns
    summary_df = summary_df.fillna(0.0)

    model_order = {name: index for index, (name, _, _) in enumerate(MODEL_FUNCTIONS)}
    summary_df["model_order"] = summary_df["model_name"].map(model_order)
    summary_df = summary_df.sort_values("model_order").drop(columns=["model_order"])

    return summary_df.reset_index(drop=True)


def format_value(value: Any) -> str:
    """
    Format numeric values for Markdown.
    """

    return f"{float(value):.6f}"


def build_markdown_report(
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    seeds: list[int],
) -> str:
    """
    Build Markdown report for repeated-seed robustness experiments.
    """

    lines: list[str] = []

    lines.append("# WaferWatch Repeated-Seed Robustness Experiment Report")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append(
        "This report repeats the main six-model baseline comparison under multiple random seeds."
    )
    lines.append(
        "The goal is to test whether the conclusions are stable or dependent on one train-test split."
    )
    lines.append("")
    lines.append("## 2. Experiment Design")
    lines.append("")
    lines.append(f"- Seeds: `{seeds}`")
    lines.append("- Split: stratified 70 percent train / 30 percent test")
    lines.append("- Dataset: selected SPC-enhanced feature table")
    lines.append("- Models per seed: 6")
    lines.append(f"- Total result rows: `{len(results_df)}`")
    lines.append("")
    lines.append("## 3. Aggregate Results Across Seeds")
    lines.append("")
    lines.append(
        "| Model | Precision mean | Precision std | Recall mean | Recall std | F1 mean | F1 std | PR-AUC mean | PR-AUC std | False alarms mean | False alarms std |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for _, row in summary_df.iterrows():
        lines.append(
            f"| {row['model_name']} | "
            f"{format_value(row['precision_mean'])} | "
            f"{format_value(row['precision_std'])} | "
            f"{format_value(row['recall_mean'])} | "
            f"{format_value(row['recall_std'])} | "
            f"{format_value(row['f1_mean'])} | "
            f"{format_value(row['f1_std'])} | "
            f"{format_value(row['pr_auc_mean'])} | "
            f"{format_value(row['pr_auc_std'])} | "
            f"{format_value(row['false_alarms_per_100_lots_mean'])} | "
            f"{format_value(row['false_alarms_per_100_lots_std'])} |"
        )

    lines.append("")
    lines.append("## 4. Worst-Case Checks")
    lines.append("")
    lines.append(
        "| Model | Recall min | Precision min | F1 min | PR-AUC min | False alarms max |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")

    for _, row in summary_df.iterrows():
        lines.append(
            f"| {row['model_name']} | "
            f"{format_value(row['recall_min'])} | "
            f"{format_value(row['precision_min'])} | "
            f"{format_value(row['f1_min'])} | "
            f"{format_value(row['pr_auc_min'])} | "
            f"{format_value(row['false_alarms_per_100_lots_max'])} |"
        )

    lines.append("")
    lines.append("## 5. Per-Seed Results")
    lines.append("")
    lines.append(
        "| Seed | Model | Precision | Recall | F1 | PR-AUC | False alarms per 100 lots | TP | FP | FN | TN |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    ordered_results = results_df.sort_values(["seed", "model_name"])

    for _, row in ordered_results.iterrows():
        lines.append(
            f"| {int(row['seed'])} | {row['model_name']} | "
            f"{format_value(row['precision'])} | "
            f"{format_value(row['recall'])} | "
            f"{format_value(row['f1'])} | "
            f"{format_value(row['pr_auc'])} | "
            f"{format_value(row['false_alarms_per_100_lots'])} | "
            f"{int(row['true_positive'])} | "
            f"{int(row['false_positive'])} | "
            f"{int(row['false_negative'])} | "
            f"{int(row['true_negative'])} |"
        )

    lines.append("")
    lines.append("## 6. Interpretation")
    lines.append("")
    lines.append(
        "Repeated-seed evaluation makes the experiment more defensible because it reduces dependence on one lucky train-test split."
    )
    lines.append(
        "If a model has high mean performance and low standard deviation, its result is more stable. If the standard deviation is large, the model may be sensitive to how lots are split into train and test sets."
    )
    lines.append(
        "These repeated-seed results should still be interpreted as controlled synthetic evidence, not production performance."
    )
    lines.append("")

    return "".join(lines)


def to_json_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Convert DataFrame to JSON-safe records.
    """

    return json.loads(df.to_json(orient="records"))


def run_repeated_seed_robustness_experiments(
    feature_table_file_name: str = "demo_spc_selected_feature_table.csv",
    seeds: list[int] | None = None,
    report_file_name: str = "repeated_seed_robustness_report.json",
    markdown_file_name: str = "repeated_seed_robustness_report.md",
    results_file_name: str = "demo_repeated_seed_robustness_results.csv",
) -> dict[str, Any]:
    """
    Run repeated-seed robustness experiments.
    """

    ensure_directories_exist()

    if seeds is None:
        seeds = DEFAULT_SEEDS

    df = load_feature_table(feature_table_file_name)
    X, y, feature_columns = prepare_xy(df)

    all_rows: list[dict[str, Any]] = []

    for seed in seeds:
        seed_rows = run_single_seed(X=X, y=y, seed=seed)
        all_rows.extend(seed_rows)

    results_df = pd.DataFrame(all_rows)
    summary_df = build_summary_table(results_df)

    results_path = DATA_PROCESSED_DIR / results_file_name
    results_df.to_csv(results_path, index=False)

    logger.info("Saved repeated-seed results CSV to: %s", results_path)

    report: dict[str, Any] = {
        "experiment_name": "Repeated-seed robustness experiments",
        "feature_table": str(DATA_PROCESSED_DIR / feature_table_file_name),
        "results_file": str(results_path),
        "seeds": seeds,
        "n_seeds": int(len(seeds)),
        "n_rows": int(len(df)),
        "n_features": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "n_models": int(len(MODEL_FUNCTIONS)),
        "n_result_rows": int(len(results_df)),
        "results": to_json_records(results_df),
        "summary": to_json_records(summary_df),
        "interpretation_note": (
            "Repeated-seed evaluation tests whether model conclusions remain stable across "
            "multiple stratified train-test splits. Mean and standard deviation summarize "
            "typical performance and split sensitivity."
        ),
    }

    report_path = REPORTS_DIR / report_file_name

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    logger.info("Saved repeated-seed report JSON to: %s", report_path)

    markdown_path = REPORTS_DIR / markdown_file_name
    markdown_text = build_markdown_report(
        results_df=results_df,
        summary_df=summary_df,
        seeds=seeds,
    )

    with markdown_path.open("w", encoding="utf-8") as file:
        file.write(markdown_text)

    logger.info("Saved repeated-seed report Markdown to: %s", markdown_path)

    print("Repeated-seed robustness experiment summary")
    print("-------------------------------------------")
    print(f"Seeds: {seeds}")
    print(f"Models per seed: {len(MODEL_FUNCTIONS)}")
    print(f"Result rows: {len(results_df)}")
    print()
    print("Aggregate summary:")
    print(
        summary_df[
            [
                "model_name",
                "precision_mean",
                "precision_std",
                "recall_mean",
                "recall_std",
                "f1_mean",
                "f1_std",
                "false_alarms_per_100_lots_mean",
                "false_alarms_per_100_lots_std",
            ]
        ]
    )

    return report


def _demo() -> None:
    """
    Run repeated-seed robustness experiments.
    """

    run_repeated_seed_robustness_experiments()


if __name__ == "__main__":
    _demo()