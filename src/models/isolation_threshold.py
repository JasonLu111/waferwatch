"""
Isolation Forest threshold tuning and false-alarm budget analysis.

This module evaluates different operational policies for converting
Isolation Forest anomaly scores into engineer review decisions.

Main idea:
- Isolation Forest gives a continuous anomaly risk score.
- The default threshold may create too many false alarms.
- A fab-style workflow often needs an escalation budget:
  review only the top-K highest-risk lots, or cap the escalation rate.
"""

from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.models.isolation_forest import (
    evaluate_anomaly_detector,
    load_feature_table,
    prepare_xy,
)
from src.utils.config import DATA_PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, ensure_directories_exist
from src.utils.logger import get_logger


logger = get_logger(__name__)


def predict_top_k(risk_scores: np.ndarray, k: int) -> np.ndarray:
    """
    Predict anomaly label 1 for the top-k highest risk scores.
    """

    k = max(1, min(k, len(risk_scores)))
    prediction = np.zeros(len(risk_scores), dtype=int)
    top_indices = np.argsort(risk_scores)[::-1][:k]
    prediction[top_indices] = 1

    return prediction


def predict_by_threshold(risk_scores: np.ndarray, threshold: float) -> np.ndarray:
    """
    Predict anomaly label 1 when risk score is greater than or equal to threshold.
    """

    return (risk_scores >= threshold).astype(int)


def build_scored_test_set(
    feature_table_file_name: str = "demo_spc_selected_feature_table.csv",
    model_file_name: str = "isolation_forest_threshold_tuned.joblib",
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Fit Isolation Forest on normal-reference training lots and return scored test rows.
    """

    df = load_feature_table(feature_table_file_name)
    X, y, feature_columns = prepare_xy(df)

    lot_ids = (
        df["lot_id"].astype(str)
        if "lot_id" in df.columns
        else pd.Series([f"ROW_{i:04d}" for i in range(len(df))])
    )

    X_train, X_test, y_train, y_test, lot_train, lot_test = train_test_split(
        X,
        y,
        lot_ids,
        test_size=0.30,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    normal_mask = y_train.to_numpy() == 0
    X_train_normal_scaled = X_train_scaled[normal_mask]

    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=random_state,
    )

    logger.info("Training Isolation Forest for threshold analysis.")
    model.fit(X_train_normal_scaled)

    raw_prediction = model.predict(X_test_scaled)
    default_predicted_label = np.where(raw_prediction == -1, 1, 0)

    decision_scores = model.decision_function(X_test_scaled)
    risk_scores = -decision_scores

    scored = pd.DataFrame(
        {
            "lot_id": lot_test.to_numpy(),
            "true_label": y_test.to_numpy().astype(int),
            "risk_score": risk_scores.astype(float),
            "default_predicted_label": default_predicted_label.astype(int),
        }
    )

    scored = scored.sort_values("risk_score", ascending=False).reset_index(drop=True)
    scored["risk_rank"] = np.arange(1, len(scored) + 1)

    model_package = {
        "scaler": scaler,
        "model": model,
        "feature_columns": feature_columns,
        "training_design": "fit_on_normal_training_lots_only",
        "threshold_analysis": "top_k_and_escalation_budget",
    }

    model_path = MODELS_DIR / model_file_name
    joblib.dump(model_package, model_path)

    metadata = {
        "feature_table": str(DATA_PROCESSED_DIR / feature_table_file_name),
        "model_file": str(model_path),
        "training_rows": int(len(X_train)),
        "normal_reference_training_rows": int(len(X_train_normal_scaled)),
        "test_rows": int(len(X_test)),
        "n_features": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "model_parameters": {
            "n_estimators": 300,
            "contamination": "auto",
            "random_state": random_state,
        },
    }

    return scored, metadata


def evaluate_default_policy(scored: pd.DataFrame) -> dict[str, Any]:
    """
    Evaluate the Isolation Forest default decision threshold.
    """

    y_true = scored["true_label"].to_numpy()
    risk_scores = scored["risk_score"].to_numpy()
    predicted_label = scored["default_predicted_label"].to_numpy()

    return {
        "policy_name": "default_isolation_forest_threshold",
        "review_count": int(predicted_label.sum()),
        "review_rate": float(predicted_label.mean()),
        "metrics": evaluate_anomaly_detector(
            y_true=y_true,
            predicted_label=predicted_label,
            risk_scores=risk_scores,
            top_k=10,
        ),
    }


def evaluate_top_k_policies(
    scored: pd.DataFrame,
    top_k_values: list[int],
) -> list[dict[str, Any]]:
    """
    Evaluate fixed top-k engineer review policies.
    """

    y_true = scored["true_label"].to_numpy()
    risk_scores = scored["risk_score"].to_numpy()

    results: list[dict[str, Any]] = []

    for top_k in top_k_values:
        predicted_label = predict_top_k(risk_scores, top_k)

        results.append(
            {
                "policy_name": f"top_{top_k}_review",
                "review_count": int(predicted_label.sum()),
                "review_rate": float(predicted_label.mean()),
                "metrics": evaluate_anomaly_detector(
                    y_true=y_true,
                    predicted_label=predicted_label,
                    risk_scores=risk_scores,
                    top_k=top_k,
                ),
            }
        )

    return results


def evaluate_escalation_rate_policies(
    scored: pd.DataFrame,
    escalation_rates: list[float],
) -> list[dict[str, Any]]:
    """
    Evaluate policies that escalate a fixed percentage of lots.
    """

    results: list[dict[str, Any]] = []
    n_rows = len(scored)

    for rate in escalation_rates:
        top_k = max(1, int(np.ceil(rate * n_rows)))
        predicted_label = predict_top_k(scored["risk_score"].to_numpy(), top_k)

        results.append(
            {
                "policy_name": f"top_{int(rate * 100)}pct_review",
                "target_review_rate": float(rate),
                "review_count": int(predicted_label.sum()),
                "review_rate": float(predicted_label.mean()),
                "metrics": evaluate_anomaly_detector(
                    y_true=scored["true_label"].to_numpy(),
                    predicted_label=predicted_label,
                    risk_scores=scored["risk_score"].to_numpy(),
                    top_k=top_k,
                ),
            }
        )

    return results


def flatten_policy_results(policy_results: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Convert nested policy results into a flat table.
    """

    rows: list[dict[str, Any]] = []

    for result in policy_results:
        metrics = result["metrics"]

        rows.append(
            {
                "policy_name": result["policy_name"],
                "review_count": result["review_count"],
                "review_rate": result["review_rate"],
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
                "true_negative": metrics["confusion_matrix"]["true_negative"],
                "false_positive": metrics["confusion_matrix"]["false_positive"],
                "false_negative": metrics["confusion_matrix"]["false_negative"],
                "true_positive": metrics["confusion_matrix"]["true_positive"],
            }
        )

    return pd.DataFrame(rows)


def select_best_policy_under_false_alarm_budget(
    policy_table: pd.DataFrame,
    max_false_alarms_per_100_lots: float = 10.0,
) -> dict[str, Any]:
    """
    Select the best policy under a false-alarm budget.

    Selection rule:
    1. Keep policies with false alarms per 100 lots <= budget.
    2. Maximize recall.
    3. Break ties by higher precision.
    4. Break ties by lower review count.
    """

    candidates = policy_table[
        policy_table["false_alarms_per_100_lots"] <= max_false_alarms_per_100_lots
    ].copy()

    if candidates.empty:
        return {
            "budget": float(max_false_alarms_per_100_lots),
            "selected_policy": None,
            "reason": "No policy satisfies the false-alarm budget.",
        }

    candidates = candidates.sort_values(
        by=["recall", "precision", "review_count"],
        ascending=[False, False, True],
    )

    best = candidates.iloc[0].to_dict()

    return {
        "budget": float(max_false_alarms_per_100_lots),
        "selected_policy": best,
        "reason": (
            "Selected the highest-recall policy under the false-alarm budget, "
            "breaking ties by precision and review count."
        ),
    }


def build_markdown_report(report: dict[str, Any]) -> str:
    """
    Build Markdown report for Isolation Forest threshold tuning.
    """

    policy_table = pd.DataFrame(report["policy_table"])
    top_lots = report["top_scored_lots"]
    selected = report["selected_policy_under_false_alarm_budget"]

    lines: list[str] = []

    lines.append("# WaferWatch Isolation Forest Threshold Tuning Report")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append(
        "This report evaluates threshold and review-budget policies for Isolation Forest anomaly scores."
    )
    lines.append(
        "The goal is to reduce false alarms while preserving high recall for risky lots."
    )
    lines.append("")
    lines.append("## 2. Why Threshold Tuning Matters")
    lines.append("")
    lines.append(
        "Isolation Forest produces continuous anomaly scores. The default model threshold may be too conservative or too aggressive for an operational monitoring workflow."
    )
    lines.append(
        "A review-budget policy converts risk scores into a practical engineer triage queue, such as reviewing only the top-K highest-risk lots."
    )
    lines.append("")
    lines.append("## 3. Policy Comparison")
    lines.append("")
    lines.append("| Policy | Review Count | Review Rate | Precision | Recall | F1 | False Alarms per 100 Lots | TP | FP | FN | TN |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for _, row in policy_table.iterrows():
        lines.append(
            f"| {row['policy_name']} | "
            f"{int(row['review_count'])} | "
            f"{row['review_rate']:.6f} | "
            f"{row['precision']:.6f} | "
            f"{row['recall']:.6f} | "
            f"{row['f1']:.6f} | "
            f"{row['false_alarms_per_100_lots']:.6f} | "
            f"{int(row['true_positive'])} | "
            f"{int(row['false_positive'])} | "
            f"{int(row['false_negative'])} | "
            f"{int(row['true_negative'])} |"
        )

    lines.append("")
    lines.append("## 4. Selected Policy Under False-Alarm Budget")
    lines.append("")

    if selected["selected_policy"] is None:
        lines.append(f"- Budget: `{selected['budget']}` false alarms per 100 lots")
        lines.append("- Selected policy: `None`")
        lines.append(f"- Reason: {selected['reason']}")
    else:
        best = selected["selected_policy"]
        lines.append(f"- Budget: `{selected['budget']}` false alarms per 100 lots")
        lines.append(f"- Selected policy: `{best['policy_name']}`")
        lines.append(f"- Precision: `{best['precision']:.6f}`")
        lines.append(f"- Recall: `{best['recall']:.6f}`")
        lines.append(f"- False alarms per 100 lots: `{best['false_alarms_per_100_lots']:.6f}`")
        lines.append(f"- Review count: `{int(best['review_count'])}`")

    lines.append("")
    lines.append("## 5. Top Scored Lots")
    lines.append("")
    lines.append("| Rank | Lot ID | True Label | Risk Score |")
    lines.append("|---:|---|---:|---:|")

    for row in top_lots:
        lines.append(
            f"| {row['risk_rank']} | `{row['lot_id']}` | "
            f"{row['true_label']} | {row['risk_score']:.6f} |"
        )

    lines.append("")
    lines.append("## 6. Interpretation")
    lines.append("")
    lines.append(report["interpretation_note"])
    lines.append("")

    return "\n".join(lines)


def run_isolation_threshold_analysis(
    feature_table_file_name: str = "demo_spc_selected_feature_table.csv",
    report_file_name: str = "isolation_threshold_report.json",
    markdown_file_name: str = "isolation_threshold_report.md",
    scored_lots_file_name: str = "demo_isolation_threshold_scored_lots.csv",
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Run Isolation Forest threshold and false-alarm budget analysis.
    """

    ensure_directories_exist()

    scored, metadata = build_scored_test_set(
        feature_table_file_name=feature_table_file_name,
        random_state=random_state,
    )

    default_policy = evaluate_default_policy(scored)

    top_k_policies = evaluate_top_k_policies(
        scored=scored,
        top_k_values=[3, 5, 8, 10, 12, 15],
    )

    escalation_rate_policies = evaluate_escalation_rate_policies(
        scored=scored,
        escalation_rates=[0.10, 0.20, 0.30, 0.40, 0.50],
    )

    all_policy_results = [default_policy] + top_k_policies + escalation_rate_policies
    policy_table = flatten_policy_results(all_policy_results)

    selected_policy = select_best_policy_under_false_alarm_budget(
        policy_table=policy_table,
        max_false_alarms_per_100_lots=10.0,
    )

    scored_path = DATA_PROCESSED_DIR / scored_lots_file_name
    scored.to_csv(scored_path, index=False)

    logger.info("Saved scored Isolation Forest lots to: %s", scored_path)

    report: dict[str, Any] = {
        "analysis_name": "Isolation Forest threshold tuning and false-alarm budget analysis",
        "metadata": metadata,
        "policy_table": policy_table.to_dict(orient="records"),
        "selected_policy_under_false_alarm_budget": selected_policy,
        "top_scored_lots": scored.head(10).to_dict(orient="records"),
        "scored_lots_file": str(scored_path),
        "interpretation_note": (
            "The default Isolation Forest threshold captures all held-out failed lots in this demo, "
            "but it also produces several false alarms. Top-K and escalation-rate policies provide "
            "a more operational way to control review workload. In this controlled synthetic demo, "
            "a smaller top-K review budget can preserve high recall while reducing false alarms. "
            "This result should be interpreted as a workflow validation rather than evidence of "
            "real production performance."
        ),
    }

    report_path = REPORTS_DIR / report_file_name

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    logger.info("Saved Isolation Forest threshold report JSON to: %s", report_path)

    markdown_path = REPORTS_DIR / markdown_file_name
    markdown_text = build_markdown_report(report)

    with markdown_path.open("w", encoding="utf-8") as file:
        file.write(markdown_text)

    logger.info("Saved Isolation Forest threshold report Markdown to: %s", markdown_path)

    print("Isolation Forest threshold analysis summary")
    print("-------------------------------------------")
    print(policy_table[[
        "policy_name",
        "review_count",
        "precision",
        "recall",
        "f1",
        "false_alarms_per_100_lots",
        "true_positive",
        "false_positive",
        "false_negative",
        "true_negative",
    ]])
    print()
    print("Selected policy under false-alarm budget")
    print("----------------------------------------")
    print(json.dumps(selected_policy, indent=2, ensure_ascii=False))

    return report


def _demo() -> None:
    """
    Run the Isolation Forest threshold analysis demo.
    """

    run_isolation_threshold_analysis()


if __name__ == "__main__":
    _demo()