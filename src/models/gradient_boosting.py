"""
Gradient Boosting baseline utilities for WaferWatch.

This module trains and evaluates a Gradient Boosting classifier on the selected
SPC-enhanced feature table.

The purpose is to add a boosting-style tabular baseline before introducing
external libraries such as LightGBM or XGBoost.
"""

from __future__ import annotations

import json
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from src.models.evaluate import evaluate_model
from src.models.train import prepare_xy
from src.utils.config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, ensure_directories_exist
from src.utils.logger import get_logger


logger = get_logger(__name__)


def train_gradient_boosting_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    max_depth: int = 3,
    random_state: int = 42,
) -> GradientBoostingClassifier:
    """
    Train a Gradient Boosting baseline classifier.

    GradientBoostingClassifier does not directly accept class_weight.
    Therefore, balanced sample weights are passed during fitting.
    """

    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
    )

    sample_weight = compute_sample_weight(
        class_weight="balanced",
        y=y_train,
    )

    logger.info("Training Gradient Boosting baseline model.")
    model.fit(X_train, y_train, sample_weight=sample_weight)
    logger.info("Gradient Boosting training completed.")

    return model


def get_feature_importance(
    model: GradientBoostingClassifier,
    feature_columns: list[str],
) -> list[dict[str, Any]]:
    """
    Extract Gradient Boosting feature importances.
    """

    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    return importance_df.to_dict(orient="records")


def build_gradient_boosting_markdown_report(
    report: dict[str, Any],
) -> str:
    """
    Build a Markdown report for the Gradient Boosting baseline.
    """

    metrics = report["evaluation_metrics"]
    feature_importance = report["feature_importance"]

    lines: list[str] = []

    lines.append("# WaferWatch Gradient Boosting Baseline Report")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append(
        "This report documents the Gradient Boosting baseline model trained on the selected SPC-enhanced feature table."
    )
    lines.append(
        "The purpose is to add a boosting-style tabular baseline before introducing external libraries such as LightGBM or XGBoost."
    )
    lines.append("")
    lines.append("## 2. Model Configuration")
    lines.append("")
    lines.append(f"- Model file: `{report['model_file']}`")
    lines.append(f"- Feature table: `{report['feature_table_file']}`")
    lines.append(f"- Training rows: `{report['n_train_rows']}`")
    lines.append(f"- Test rows: `{report['n_test_rows']}`")
    lines.append(f"- Number of features: `{report['n_features']}`")
    lines.append(f"- Number of estimators: `{report['model_parameters']['n_estimators']}`")
    lines.append(f"- Learning rate: `{report['model_parameters']['learning_rate']}`")
    lines.append(f"- Max depth: `{report['model_parameters']['max_depth']}`")
    lines.append(f"- Sample weighting: `{report['model_parameters']['sample_weighting']}`")
    lines.append("")
    lines.append("## 3. Evaluation Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")

    metric_names = [
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

    for metric_name in metric_names:
        value = metrics.get(metric_name)
        value_text = "None" if value is None else f"{value:.6f}"
        lines.append(f"| {metric_name} | {value_text} |")

    lines.append("")
    lines.append("## 4. Confusion Matrix")
    lines.append("")
    lines.append("| Item | Count |")
    lines.append("|---|---:|")

    for key, value in metrics["confusion_matrix"].items():
        lines.append(f"| {key} | {value} |")

    lines.append("")
    lines.append("## 5. Feature Importance")
    lines.append("")
    lines.append("| Rank | Feature | Importance |")
    lines.append("|---:|---|---:|")

    for rank, row in enumerate(feature_importance, start=1):
        lines.append(
            f"| {rank} | `{row['feature']}` | {row['importance']:.6f} |"
        )

    lines.append("")
    lines.append("## 6. Interpretation")
    lines.append("")
    lines.append(report["interpretation_note"])
    lines.append("")

    return "\n".join(lines)


def run_gradient_boosting_baseline(
    feature_table_file_name: str = "demo_spc_selected_feature_table.csv",
    model_file_name: str = "gradient_boosting_selected_baseline.joblib",
    report_file_name: str = "gradient_boosting_report.json",
    markdown_file_name: str = "gradient_boosting_report.md",
    label_column: str = "pass_fail_label",
    id_column: str = "lot_id",
    test_size: float = 0.30,
    random_state: int = 42,
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    max_depth: int = 3,
    top_k: int = 10,
) -> dict[str, Any]:
    """
    Train and evaluate a Gradient Boosting baseline on the selected feature table.
    """

    ensure_directories_exist()

    feature_table_path = PROCESSED_DATA_DIR / feature_table_file_name
    model_path = MODELS_DIR / model_file_name

    if not feature_table_path.exists():
        raise FileNotFoundError(
            f"Feature table not found: {feature_table_path}. "
            "Please run src.features.feature_selection first."
        )

    logger.info("Loading feature table from: %s", feature_table_path)
    df = pd.read_csv(feature_table_path)

    X, y = prepare_xy(
        df=df,
        label_column=label_column,
        id_column=id_column,
    )

    stratify = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    model = train_gradient_boosting_baseline(
        X_train=X_train,
        y_train=y_train,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
    )

    y_pred = model.predict(X_test).tolist()
    y_score = model.predict_proba(X_test)[:, 1].tolist()

    metrics = evaluate_model(
        y_true=y_test,
        y_pred=y_pred,
        y_score=y_score,
        top_k=min(top_k, len(y_test)),
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    logger.info("Saved Gradient Boosting model to: %s", model_path)

    feature_importance = get_feature_importance(
        model=model,
        feature_columns=X.columns.tolist(),
    )

    report: dict[str, Any] = {
        "model_name": "Gradient Boosting Selected Feature Baseline",
        "model_file": str(model_path),
        "feature_table_file": str(feature_table_path),
        "n_rows": int(df.shape[0]),
        "n_train_rows": int(len(X_train)),
        "n_test_rows": int(len(X_test)),
        "n_features": int(X.shape[1]),
        "feature_columns": X.columns.tolist(),
        "label_distribution": {
            str(label): int(count)
            for label, count in y.value_counts().to_dict().items()
        },
        "model_parameters": {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "sample_weighting": "balanced",
            "random_state": random_state,
        },
        "test_size": test_size,
        "evaluation_metrics": metrics,
        "feature_importance": feature_importance,
        "interpretation_note": (
            "This Gradient Boosting model is a boosting-style nonlinear tabular baseline "
            "trained on controlled synthetic SPC-enhanced demo data. Results should be used "
            "to compare model behavior in the demo pipeline, not to claim production fab performance."
        ),
    }

    report_path = REPORTS_DIR / report_file_name

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    logger.info("Saved Gradient Boosting report JSON to: %s", report_path)

    markdown_path = REPORTS_DIR / markdown_file_name
    markdown_text = build_gradient_boosting_markdown_report(report)

    with markdown_path.open("w", encoding="utf-8") as file:
        file.write(markdown_text)

    logger.info("Saved Gradient Boosting report Markdown to: %s", markdown_path)

    print("Gradient Boosting baseline summary")
    print("----------------------------------")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print()
    print("Feature importance")
    print("------------------")
    print(pd.DataFrame(feature_importance))

    return report


def _demo() -> None:
    """
    Run Gradient Boosting baseline demo.
    """

    run_gradient_boosting_baseline()


if __name__ == "__main__":
    _demo()