"""
Calibration analysis utilities for WaferWatch.

This module evaluates whether model risk scores are reasonably calibrated.

Calibration matters because WaferWatch uses model probabilities for:
- risk scoring
- cost-sensitive thresholding
- escalation decisions
- monitoring
- engineering review prioritization

A model can rank lots well but still produce poorly calibrated probabilities.
This module adds Brier score, log loss, reliability table, expected calibration
error, and maximum calibration error.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.utils.config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, ensure_directories_exist
from src.utils.logger import get_logger


logger = get_logger(__name__)


def load_feature_columns_from_training_report(
    report_file_name: str = "spc_selected_training_report.json",
) -> list[str]:
    """
    Load feature columns from a training report.
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
        raise ValueError(f"No feature_columns found in {report_path}.")

    return feature_columns


def prepare_calibration_data(
    df: pd.DataFrame,
    feature_columns: list[str],
    label_column: str = "pass_fail_label",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare X and y for calibration analysis.
    """

    if label_column not in df.columns:
        raise ValueError(f"Label column not found: {label_column}")

    missing_features = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")

    X = df[feature_columns].copy()
    y = df[label_column].copy()

    return X, y


def create_reliability_table(
    y_true: pd.Series,
    y_score: list[float],
    n_bins: int = 10,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Create a reliability table and calibration summary metrics.

    Each bin compares:
    - average predicted probability
    - observed positive rate
    """

    if n_bins <= 0:
        raise ValueError("n_bins must be greater than 0.")

    calibration_df = pd.DataFrame(
        {
            "y_true": y_true.reset_index(drop=True),
            "y_score": y_score,
        }
    )

    calibration_df["y_score"] = calibration_df["y_score"].clip(0.0, 1.0)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    calibration_df["probability_bin"] = pd.cut(
        calibration_df["y_score"],
        bins=bin_edges,
        include_lowest=True,
        labels=False,
    )

    rows: list[dict[str, Any]] = []
    n_total = len(calibration_df)

    expected_calibration_error = 0.0
    maximum_calibration_error = 0.0

    for bin_id in range(n_bins):
        bin_df = calibration_df[calibration_df["probability_bin"] == bin_id]

        bin_lower = float(bin_edges[bin_id])
        bin_upper = float(bin_edges[bin_id + 1])

        n_samples = int(len(bin_df))

        if n_samples == 0:
            avg_predicted_probability = None
            observed_positive_rate = None
            absolute_calibration_error = None
            weighted_calibration_error = 0.0
        else:
            avg_predicted_probability = float(bin_df["y_score"].mean())
            observed_positive_rate = float(bin_df["y_true"].mean())
            absolute_calibration_error = abs(
                avg_predicted_probability - observed_positive_rate
            )
            weighted_calibration_error = absolute_calibration_error * n_samples / n_total

            expected_calibration_error += weighted_calibration_error
            maximum_calibration_error = max(
                maximum_calibration_error,
                absolute_calibration_error,
            )

        rows.append(
            {
                "bin_id": bin_id,
                "bin_lower": bin_lower,
                "bin_upper": bin_upper,
                "n_samples": n_samples,
                "avg_predicted_probability": avg_predicted_probability,
                "observed_positive_rate": observed_positive_rate,
                "absolute_calibration_error": absolute_calibration_error,
                "weighted_calibration_error": weighted_calibration_error,
            }
        )

    reliability_table = pd.DataFrame(rows)

    summary = {
        "expected_calibration_error": float(expected_calibration_error),
        "maximum_calibration_error": float(maximum_calibration_error),
    }

    return reliability_table, summary


def evaluate_probability_quality(
    y_true: pd.Series,
    y_score: list[float],
) -> dict[str, Any]:
    """
    Evaluate probability quality and ranking quality.
    """

    clipped_scores = np.clip(np.array(y_score), 1e-15, 1 - 1e-15)

    metrics: dict[str, Any] = {
        "brier_score": float(brier_score_loss(y_true, y_score)),
        "log_loss": float(log_loss(y_true, clipped_scores)),
        "roc_auc": None,
        "pr_auc": None,
    }

    if len(set(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_score))

    return metrics


def build_calibration_markdown(
    report: dict[str, Any],
) -> str:
    """
    Build a Markdown calibration report.
    """

    metrics = report["calibration_metrics"]
    reliability_rows = report["reliability_table"]

    lines: list[str] = []

    lines.append("# WaferWatch Calibration Report")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append(
        "This report evaluates whether WaferWatch risk scores are reasonably calibrated."
    )
    lines.append(
        "Calibration matters because the project uses predicted risk scores for thresholding, escalation, and monitoring."
    )
    lines.append("")
    lines.append("## 2. Model and Data")
    lines.append("")
    lines.append(f"- Model file: `{report['model_file']}`")
    lines.append(f"- Feature table: `{report['feature_table_file']}`")
    lines.append(f"- Test rows: `{report['n_test_rows']}`")
    lines.append(f"- Number of bins: `{report['n_bins']}`")
    lines.append("")
    lines.append("## 3. Calibration Metrics")
    lines.append("")
    lines.append("| Metric | Value | Interpretation |")
    lines.append("|---|---:|---|")
    lines.append(
        f"| Brier score | {metrics['brier_score']:.6f} | Lower is better; measures mean squared probability error |"
    )
    lines.append(
        f"| Log loss | {metrics['log_loss']:.6f} | Lower is better; penalizes confident wrong predictions |"
    )
    lines.append(
        f"| Expected calibration error | {metrics['expected_calibration_error']:.6f} | Lower is better; weighted average calibration gap |"
    )
    lines.append(
        f"| Maximum calibration error | {metrics['maximum_calibration_error']:.6f} | Largest bin-level calibration gap |"
    )

    if metrics["roc_auc"] is not None:
        lines.append(
            f"| ROC-AUC | {metrics['roc_auc']:.6f} | Ranking quality across thresholds |"
        )

    if metrics["pr_auc"] is not None:
        lines.append(
            f"| PR-AUC | {metrics['pr_auc']:.6f} | Ranking quality under class imbalance |"
        )

    lines.append("")
    lines.append("## 4. Reliability Table")
    lines.append("")
    lines.append("| Bin | Range | N | Avg predicted probability | Observed positive rate | Abs calibration error |")
    lines.append("|---:|---|---:|---:|---:|---:|")

    for row in reliability_rows:
        if row["n_samples"] == 0:
            avg_prob = ""
            obs_rate = ""
            abs_error = ""
        else:
            avg_prob = f"{row['avg_predicted_probability']:.6f}"
            obs_rate = f"{row['observed_positive_rate']:.6f}"
            abs_error = f"{row['absolute_calibration_error']:.6f}"

        lines.append(
            f"| {row['bin_id']} | [{row['bin_lower']:.1f}, {row['bin_upper']:.1f}] | "
            f"{row['n_samples']} | {avg_prob} | {obs_rate} | {abs_error} |"
        )

    lines.append("")
    lines.append("## 5. Interpretation")
    lines.append("")
    lines.append(report["interpretation_note"])
    lines.append("")
    lines.append("## 6. Limitations")
    lines.append("")
    lines.append("- The current calibration analysis uses synthetic demo data.")
    lines.append("- The test set is small.")
    lines.append("- Calibration results should not be interpreted as production probability quality.")
    lines.append("- Future work should evaluate calibration on larger and time-split datasets.")
    lines.append("")

    return "\n".join(lines)


def run_calibration_analysis(
    feature_table_file_name: str = "demo_spc_selected_feature_table.csv",
    model_file_name: str = "spc_selected_logistic_regression.joblib",
    training_report_file_name: str = "spc_selected_training_report.json",
    report_file_name: str = "calibration_report.json",
    markdown_file_name: str = "calibration_report.md",
    label_column: str = "pass_fail_label",
    test_size: float = 0.30,
    random_state: int = 42,
    n_bins: int = 10,
) -> dict[str, Any]:
    """
    Run calibration analysis on the selected SPC model.
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

    logger.info("Loading feature table from: %s", feature_table_path)
    df = pd.read_csv(feature_table_path)

    feature_columns = load_feature_columns_from_training_report(
        report_file_name=training_report_file_name,
    )

    X, y = prepare_calibration_data(
        df=df,
        feature_columns=feature_columns,
        label_column=label_column,
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

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1].tolist()
    else:
        y_score = model.predict(X_test).tolist()

    probability_metrics = evaluate_probability_quality(
        y_true=y_test,
        y_score=y_score,
    )

    reliability_table, calibration_summary = create_reliability_table(
        y_true=y_test,
        y_score=y_score,
        n_bins=n_bins,
    )

    calibration_metrics = {
        **probability_metrics,
        **calibration_summary,
    }

    report: dict[str, Any] = {
        "model_file": str(model_path),
        "feature_table_file": str(feature_table_path),
        "training_report_file": str(REPORTS_DIR / training_report_file_name),
        "n_test_rows": int(len(y_test)),
        "test_size": test_size,
        "random_state": random_state,
        "n_bins": n_bins,
        "feature_columns": feature_columns,
        "calibration_metrics": calibration_metrics,
        "reliability_table": reliability_table.to_dict(orient="records"),
        "interpretation_note": (
            "This calibration report evaluates the probability quality of the current "
            "demo model. Because the dataset is synthetic and small, the result should "
            "be used to validate the calibration workflow rather than to claim production "
            "probability reliability."
        ),
    }

    report_path = REPORTS_DIR / report_file_name

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    logger.info("Saved calibration report JSON to: %s", report_path)

    markdown_path = REPORTS_DIR / markdown_file_name
    markdown_text = build_calibration_markdown(report)

    with markdown_path.open("w", encoding="utf-8") as file:
        file.write(markdown_text)

    logger.info("Saved calibration report Markdown to: %s", markdown_path)

    print("Calibration analysis summary")
    print("----------------------------")
    print(json.dumps(calibration_metrics, indent=2, ensure_ascii=False))
    print()
    print("Reliability table")
    print("-----------------")
    print(reliability_table)

    return report


def _demo() -> None:
    """
    Run calibration analysis demo.
    """

    run_calibration_analysis()


if __name__ == "__main__":
    _demo()