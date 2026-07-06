"""
Robustness and ablation experiments for WaferWatch.

This module tests whether strong demo results remain stable when:

1. Key SPC features are removed.
2. The injected anomaly signal is weakened.

The goal is not to claim production performance. The goal is to create a
stronger experimental design and defend against overclaiming perfect metrics
from a controlled synthetic SPC demo.
"""

from __future__ import annotations

import json
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
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
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from src.models.isolation_forest import load_feature_table, prepare_xy
from src.utils.config import DATA_PROCESSED_DIR, REPORTS_DIR, ensure_directories_exist
from src.utils.logger import get_logger


logger = get_logger(__name__)


RANDOM_STATE = 42
TOP_K = 10


def calculate_precision_recall_at_k(
    y_true: np.ndarray,
    risk_scores: np.ndarray,
    top_k: int = TOP_K,
) -> tuple[float, float]:
    """
    Calculate precision@K and recall@K from risk scores.
    """

    k = min(top_k, len(y_true))

    if k <= 0:
        return 0.0, 0.0

    order = np.argsort(risk_scores)[::-1]
    top_indices = order[:k]

    top_true = y_true[top_indices]
    true_positive_at_k = int(top_true.sum())
    total_positive = int(y_true.sum())

    precision_at_k = true_positive_at_k / k
    recall_at_k = true_positive_at_k / total_positive if total_positive > 0 else 0.0

    return float(precision_at_k), float(recall_at_k)


def evaluate_predictions(
    y_true: np.ndarray,
    predicted_label: np.ndarray,
    risk_scores: np.ndarray,
    top_k: int = TOP_K,
) -> dict[str, Any]:
    """
    Evaluate binary anomaly predictions.
    """

    cm = confusion_matrix(y_true, predicted_label, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision_at_k, recall_at_k = calculate_precision_recall_at_k(
        y_true=y_true,
        risk_scores=risk_scores,
        top_k=top_k,
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, predicted_label)),
        "precision": float(precision_score(y_true, predicted_label, zero_division=0)),
        "recall": float(recall_score(y_true, predicted_label, zero_division=0)),
        "f1": float(f1_score(y_true, predicted_label, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted_label)),
        "matthews_corrcoef": float(matthews_corrcoef(y_true, predicted_label)),
        "roc_auc": float(roc_auc_score(y_true, risk_scores)),
        "pr_auc": float(average_precision_score(y_true, risk_scores)),
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "top_k": int(top_k),
        "false_alarms_per_100_lots": float(fp / len(y_true) * 100),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }

    return metrics


def shrink_anomaly_signal(
    X: pd.DataFrame,
    y: pd.Series,
    severity: float,
) -> pd.DataFrame:
    """
    Shrink failed-lot feature values toward normal-lot means.

    severity = 1.00 keeps the original anomaly signal.
    severity = 0.50 makes anomaly rows halfway closer to the normal mean.
    severity = 0.25 makes anomaly rows much harder to separate.
    """

    X_modified = X.copy()
    normal_mean = X.loc[y == 0].mean(axis=0)

    anomaly_mask = y == 1
    X_modified.loc[anomaly_mask, :] = normal_mean + severity * (
        X_modified.loc[anomaly_mask, :] - normal_mean
    )

    return X_modified


def train_evaluate_logistic_regression(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, Any]:
    """
    Train and evaluate Logistic Regression.
    """

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
        random_state=RANDOM_STATE,
    )

    model.fit(X_train_scaled, y_train)

    risk_scores = model.predict_proba(X_test_scaled)[:, 1]
    predicted_label = (risk_scores >= 0.5).astype(int)

    return evaluate_predictions(
        y_true=y_test.to_numpy(),
        predicted_label=predicted_label,
        risk_scores=risk_scores,
    )


def train_evaluate_random_forest(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, Any]:
    """
    Train and evaluate Random Forest.
    """

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    risk_scores = model.predict_proba(X_test)[:, 1]
    predicted_label = (risk_scores >= 0.5).astype(int)

    return evaluate_predictions(
        y_true=y_test.to_numpy(),
        predicted_label=predicted_label,
        risk_scores=risk_scores,
    )


def train_evaluate_gradient_boosting(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, Any]:
    """
    Train and evaluate Gradient Boosting.
    """

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train, sample_weight=sample_weight)

    risk_scores = model.predict_proba(X_test)[:, 1]
    predicted_label = (risk_scores >= 0.5).astype(int)

    return evaluate_predictions(
        y_true=y_test.to_numpy(),
        predicted_label=predicted_label,
        risk_scores=risk_scores,
    )


def train_evaluate_isolation_forest(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, Any]:
    """
    Train and evaluate Isolation Forest on normal-reference training lots.
    """

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    normal_mask = y_train.to_numpy() == 0
    X_train_normal_scaled = X_train_scaled[normal_mask]

    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=RANDOM_STATE,
    )

    model.fit(X_train_normal_scaled)

    risk_scores = -model.score_samples(X_test_scaled)
    predicted_label = (model.predict(X_test_scaled) == -1).astype(int)

    return evaluate_predictions(
        y_true=y_test.to_numpy(),
        predicted_label=predicted_label,
        risk_scores=risk_scores,
    )


def train_evaluate_pca_anomaly(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    threshold_quantile: float = 0.95,
) -> dict[str, Any]:
    """
    Train and evaluate PCA reconstruction-error anomaly detection.
    """

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    normal_mask = y_train.to_numpy() == 0
    X_train_normal_scaled = X_train_scaled[normal_mask]

    pca = PCA(
        n_components=0.95,
        svd_solver="full",
        random_state=RANDOM_STATE,
    )

    pca.fit(X_train_normal_scaled)

    train_normal_reconstructed = pca.inverse_transform(
        pca.transform(X_train_normal_scaled)
    )
    train_normal_errors = np.mean(
        (X_train_normal_scaled - train_normal_reconstructed) ** 2,
        axis=1,
    )

    threshold = float(np.quantile(train_normal_errors, threshold_quantile))

    test_reconstructed = pca.inverse_transform(pca.transform(X_test_scaled))
    risk_scores = np.mean((X_test_scaled - test_reconstructed) ** 2, axis=1)
    predicted_label = (risk_scores >= threshold).astype(int)

    return evaluate_predictions(
        y_true=y_test.to_numpy(),
        predicted_label=predicted_label,
        risk_scores=risk_scores,
    )


def build_autoencoder_hidden_layers(n_features: int) -> tuple[int, int, int]:
    """
    Build a small autoencoder-style hidden-layer structure.
    """

    outer_width = max(2, min(4, n_features))
    bottleneck_width = max(1, min(2, n_features - 1))

    return (outer_width, bottleneck_width, outer_width)


def train_evaluate_autoencoder_anomaly(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    threshold_quantile: float = 0.95,
) -> dict[str, Any]:
    """
    Train and evaluate autoencoder-style reconstruction-error anomaly detection.
    """

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    normal_mask = y_train.to_numpy() == 0
    X_train_normal_scaled = X_train_scaled[normal_mask]

    hidden_layer_sizes = build_autoencoder_hidden_layers(n_features=X_train.shape[1])

    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=0.0001,
        learning_rate_init=0.001,
        max_iter=3000,
        random_state=RANDOM_STATE,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(X_train_normal_scaled, X_train_normal_scaled)

    train_normal_reconstructed = model.predict(X_train_normal_scaled)
    train_normal_errors = np.mean(
        (X_train_normal_scaled - train_normal_reconstructed) ** 2,
        axis=1,
    )

    threshold = float(np.quantile(train_normal_errors, threshold_quantile))

    test_reconstructed = model.predict(X_test_scaled)
    risk_scores = np.mean((X_test_scaled - test_reconstructed) ** 2, axis=1)
    predicted_label = (risk_scores >= threshold).astype(int)

    return evaluate_predictions(
        y_true=y_test.to_numpy(),
        predicted_label=predicted_label,
        risk_scores=risk_scores,
    )


def build_experiment_scenarios(
    X: pd.DataFrame,
    y: pd.Series,
) -> list[dict[str, Any]]:
    """
    Build feature-ablation and anomaly-severity scenarios.
    """

    all_features = list(X.columns)

    spc_features = [
        feature for feature in all_features
        if feature in ["spc_violation_count", "spc_max_abs_zscore"]
    ]

    sensor_features = [
        feature for feature in all_features
        if feature not in spc_features
    ]

    scenarios: list[dict[str, Any]] = []

    scenarios.append(
        {
            "scenario_name": "baseline_full_features",
            "scenario_type": "baseline",
            "severity": 1.00,
            "feature_columns": all_features,
            "X": X[all_features].copy(),
        }
    )

    for severity in [0.75, 0.50, 0.25]:
        scenarios.append(
            {
                "scenario_name": f"anomaly_severity_{severity:.2f}",
                "scenario_type": "anomaly_severity_stress",
                "severity": severity,
                "feature_columns": all_features,
                "X": shrink_anomaly_signal(X[all_features], y, severity),
            }
        )

    if "spc_violation_count" in all_features:
        features = [feature for feature in all_features if feature != "spc_violation_count"]
        scenarios.append(
            {
                "scenario_name": "ablation_without_spc_violation_count",
                "scenario_type": "feature_ablation",
                "severity": 1.00,
                "feature_columns": features,
                "X": X[features].copy(),
            }
        )

    if "spc_max_abs_zscore" in all_features:
        features = [feature for feature in all_features if feature != "spc_max_abs_zscore"]
        scenarios.append(
            {
                "scenario_name": "ablation_without_spc_max_abs_zscore",
                "scenario_type": "feature_ablation",
                "severity": 1.00,
                "feature_columns": features,
                "X": X[features].copy(),
            }
        )

    if spc_features:
        scenarios.append(
            {
                "scenario_name": "ablation_without_all_spc_features",
                "scenario_type": "feature_ablation",
                "severity": 1.00,
                "feature_columns": sensor_features,
                "X": X[sensor_features].copy(),
            }
        )

        scenarios.append(
            {
                "scenario_name": "ablation_spc_features_only",
                "scenario_type": "feature_ablation",
                "severity": 1.00,
                "feature_columns": spc_features,
                "X": X[spc_features].copy(),
            }
        )

    return scenarios


def run_single_scenario(
    scenario: dict[str, Any],
    y: pd.Series,
) -> list[dict[str, Any]]:
    """
    Run all model families for one scenario.
    """

    X_scenario = scenario["X"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_scenario,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model_functions = [
        (
            "Logistic Regression",
            "supervised_classification",
            train_evaluate_logistic_regression,
        ),
        (
            "Random Forest",
            "supervised_classification",
            train_evaluate_random_forest,
        ),
        (
            "Gradient Boosting",
            "supervised_classification",
            train_evaluate_gradient_boosting,
        ),
        (
            "Isolation Forest",
            "unsupervised_anomaly_detection",
            train_evaluate_isolation_forest,
        ),
        (
            "PCA Anomaly",
            "unsupervised_anomaly_detection",
            train_evaluate_pca_anomaly,
        ),
        (
            "Autoencoder Anomaly",
            "unsupervised_anomaly_detection",
            train_evaluate_autoencoder_anomaly,
        ),
    ]

    rows: list[dict[str, Any]] = []

    for model_name, learning_type, model_function in model_functions:
        logger.info(
            "Running scenario '%s' with model '%s'.",
            scenario["scenario_name"],
            model_name,
        )

        metrics = model_function(X_train, X_test, y_train, y_test)
        cm = metrics["confusion_matrix"]

        rows.append(
            {
                "scenario_name": scenario["scenario_name"],
                "scenario_type": scenario["scenario_type"],
                "severity": float(scenario["severity"]),
                "model_name": model_name,
                "learning_type": learning_type,
                "n_features": int(len(scenario["feature_columns"])),
                "feature_columns": ", ".join(scenario["feature_columns"]),
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
                "true_negative": cm["true_negative"],
                "false_positive": cm["false_positive"],
                "false_negative": cm["false_negative"],
                "true_positive": cm["true_positive"],
            }
        )

    return rows


def build_markdown_report(results_df: pd.DataFrame) -> str:
    """
    Build Markdown report for robustness and ablation experiments.
    """

    lines: list[str] = []

    lines.append("# WaferWatch Robustness and Ablation Experiment Report")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append(
        "This report tests whether strong demo results remain stable when key SPC features are removed or when the anomaly signal is weakened."
    )
    lines.append(
        "The purpose is to strengthen the experimental design and avoid overclaiming perfect metrics from a controlled synthetic SPC demo."
    )
    lines.append("")
    lines.append("## 2. Experiment Design")
    lines.append("")
    lines.append("| Experiment type | Scenario | Purpose |")
    lines.append("|---|---|---|")
    lines.append(
        "| Baseline | Full selected feature table | Establish the reference result |"
    )
    lines.append(
        "| Feature ablation | Remove one or more SPC features | Test whether models depend too heavily on engineered SPC signals |"
    )
    lines.append(
        "| Anomaly severity stress test | Shrink failed-lot features toward normal-lot means | Test whether models remain stable when anomaly signals become weaker |"
    )
    lines.append("")
    lines.append("## 3. Baseline Full-Feature Results")
    lines.append("")

    baseline = results_df[results_df["scenario_name"] == "baseline_full_features"]
    lines.extend(format_results_table(baseline))

    lines.append("")
    lines.append("## 4. Feature Ablation Results")
    lines.append("")

    ablation = results_df[results_df["scenario_type"] == "feature_ablation"]
    lines.extend(format_results_table(ablation))

    lines.append("")
    lines.append("## 5. Anomaly Severity Stress Test Results")
    lines.append("")

    stress = results_df[results_df["scenario_type"] == "anomaly_severity_stress"]
    lines.extend(format_results_table(stress))

    lines.append("")
    lines.append("## 6. Interpretation")
    lines.append("")
    lines.append(
        "These experiments make the demo more defensible. If performance remains high after removing SPC features or weakening the anomaly signal, the workflow is more robust. If performance drops sharply, the result is still useful because it identifies which engineered signals drive the model."
    )
    lines.append(
        "The current results should be interpreted as controlled synthetic evidence, not production performance. The main value is the experimental design: supervised baselines, unsupervised anomaly detectors, feature ablation, severity stress tests, and false-alarm analysis are evaluated under one consistent framework."
    )
    lines.append("")

    return "\n".join(lines)


def format_results_table(df: pd.DataFrame) -> list[str]:
    """
    Format selected metrics as a Markdown table.
    """

    columns = [
        "scenario_name",
        "model_name",
        "n_features",
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "false_alarms_per_100_lots",
    ]

    lines: list[str] = []

    if df.empty:
        lines.append("No rows available.")
        return lines

    lines.append(
        "| Scenario | Model | Features | Precision | Recall | F1 | PR-AUC | False alarms per 100 lots |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")

    for _, row in df[columns].iterrows():
        lines.append(
            f"| {row['scenario_name']} | {row['model_name']} | "
            f"{int(row['n_features'])} | "
            f"{row['precision']:.6f} | "
            f"{row['recall']:.6f} | "
            f"{row['f1']:.6f} | "
            f"{row['pr_auc']:.6f} | "
            f"{row['false_alarms_per_100_lots']:.6f} |"
        )

    return lines


def run_robustness_ablation_experiments(
    feature_table_file_name: str = "demo_spc_selected_feature_table.csv",
    report_file_name: str = "robustness_ablation_report.json",
    markdown_file_name: str = "robustness_ablation_report.md",
    results_file_name: str = "demo_robustness_ablation_results.csv",
) -> dict[str, Any]:
    """
    Run robustness and ablation experiments.
    """

    ensure_directories_exist()

    df = load_feature_table(feature_table_file_name)
    X, y, feature_columns = prepare_xy(df)

    scenarios = build_experiment_scenarios(X=X, y=y)

    all_rows: list[dict[str, Any]] = []

    for scenario in scenarios:
        scenario_rows = run_single_scenario(scenario=scenario, y=y)
        all_rows.extend(scenario_rows)

    results_df = pd.DataFrame(all_rows)

    results_path = DATA_PROCESSED_DIR / results_file_name
    results_df.to_csv(results_path, index=False)

    logger.info("Saved robustness and ablation results CSV to: %s", results_path)

    report: dict[str, Any] = {
        "experiment_name": "Robustness and ablation experiments",
        "feature_table": str(DATA_PROCESSED_DIR / feature_table_file_name),
        "results_file": str(results_path),
        "n_rows": int(len(df)),
        "n_original_features": int(len(feature_columns)),
        "original_feature_columns": feature_columns,
        "n_scenarios": int(len(scenarios)),
        "n_result_rows": int(len(results_df)),
        "scenario_names": sorted(results_df["scenario_name"].unique().tolist()),
        "model_names": sorted(results_df["model_name"].unique().tolist()),
        "results": results_df.to_dict(orient="records"),
        "interpretation_note": (
            "The robustness and ablation experiments test whether strong demo results remain stable "
            "when SPC features are removed or when anomaly severity is reduced. These tests make the "
            "project more defensible by showing whether performance comes from broad signal, specific "
            "SPC engineered features, or a strong synthetic anomaly mechanism."
        ),
    }

    report_path = REPORTS_DIR / report_file_name

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    logger.info("Saved robustness and ablation report JSON to: %s", report_path)

    markdown_path = REPORTS_DIR / markdown_file_name
    markdown_text = build_markdown_report(results_df)

    with markdown_path.open("w", encoding="utf-8") as file:
        file.write(markdown_text)

    logger.info("Saved robustness and ablation report Markdown to: %s", markdown_path)

    print("Robustness and ablation experiment summary")
    print("------------------------------------------")
    print(f"Scenarios: {len(scenarios)}")
    print(f"Models per scenario: 6")
    print(f"Result rows: {len(results_df)}")
    print()
    print("Baseline full-feature results:")
    print(
        results_df[
            results_df["scenario_name"] == "baseline_full_features"
        ][
            [
                "model_name",
                "precision",
                "recall",
                "f1",
                "pr_auc",
                "false_alarms_per_100_lots",
            ]
        ]
    )

    return report


def _demo() -> None:
    """
    Run robustness and ablation experiments.
    """

    run_robustness_ablation_experiments()


if __name__ == "__main__":
    _demo()