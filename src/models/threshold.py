"""
Cost-sensitive thresholding utilities for WaferWatch.

This module turns model risk scores into operational escalation decisions.

Instead of using the default 0.5 threshold blindly, this module compares:
- default threshold
- cost-sensitive threshold
- best threshold by realized operational cost
- top-K escalation list

This is central to WaferWatch because manufacturing anomaly detection should
balance missed-risk cost, false alarm burden, and engineer review capacity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.models.evaluate import evaluate_model
from src.models.train import prepare_xy
from src.utils.config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, ensure_directories_exist
from src.utils.logger import get_logger


logger = get_logger(__name__)


def calculate_operational_cost(
    y_true: list[int],
    y_pred: list[int],
    cost_true_positive: float = 1.0,
    cost_false_positive: float = 3.0,
    cost_false_negative: float = 20.0,
    cost_true_negative: float = 0.0,
) -> dict[str, Any]:
    """
    Calculate realized operational cost from classification decisions.

    Cost interpretation:
    - True positive: risky lot escalated correctly, still consumes review effort.
    - False positive: normal lot escalated, creates unnecessary review burden.
    - False negative: risky lot missed, most expensive outcome.
    - True negative: normal lot released, no extra cost.
    """

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    counts = {
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
        "true_negative": 0,
    }

    total_cost = 0.0

    for true_label, predicted_label in zip(y_true, y_pred):
        if true_label == 1 and predicted_label == 1:
            counts["true_positive"] += 1
            total_cost += cost_true_positive
        elif true_label == 0 and predicted_label == 1:
            counts["false_positive"] += 1
            total_cost += cost_false_positive
        elif true_label == 1 and predicted_label == 0:
            counts["false_negative"] += 1
            total_cost += cost_false_negative
        else:
            counts["true_negative"] += 1
            total_cost += cost_true_negative

    n_lots = len(y_true)
    cost_per_100_lots = float(total_cost / n_lots * 100) if n_lots > 0 else 0.0

    return {
        "counts": counts,
        "total_cost": float(total_cost),
        "cost_per_100_lots": cost_per_100_lots,
        "cost_assumptions": {
            "cost_true_positive": cost_true_positive,
            "cost_false_positive": cost_false_positive,
            "cost_false_negative": cost_false_negative,
            "cost_true_negative": cost_true_negative,
        },
    }


def apply_threshold(
    y_score: list[float],
    threshold: float,
) -> list[int]:
    """
    Convert risk scores into binary escalation decisions.

    1 means escalate.
    0 means release.
    """

    return [1 if score >= threshold else 0 for score in y_score]


def derive_simple_cost_sensitive_threshold(
    cost_review: float = 1.0,
    cost_false_negative: float = 20.0,
) -> float:
    """
    Derive a simple threshold using expected loss logic.

    Escalate if:

        p_fail × cost_false_negative > cost_review

    Therefore:

        p_fail > cost_review / cost_false_negative

    Example:
    If review cost = 1 and false negative cost = 20,
    threshold = 1 / 20 = 0.05.
    """

    if cost_false_negative <= 0:
        raise ValueError("cost_false_negative must be greater than 0.")

    threshold = cost_review / cost_false_negative

    return float(min(max(threshold, 0.0), 1.0))


def search_best_threshold_by_cost(
    y_true: list[int],
    y_score: list[float],
    thresholds: list[float] | None = None,
    cost_true_positive: float = 1.0,
    cost_false_positive: float = 3.0,
    cost_false_negative: float = 20.0,
    cost_true_negative: float = 0.0,
) -> dict[str, Any]:
    """
    Search over candidate thresholds and select the one with the lowest cost.
    """

    if thresholds is None:
        thresholds = [round(float(x), 2) for x in np.arange(0.01, 1.00, 0.01)]

    rows: list[dict[str, Any]] = []

    for threshold in thresholds:
        y_pred = apply_threshold(y_score, threshold)

        cost_report = calculate_operational_cost(
            y_true=y_true,
            y_pred=y_pred,
            cost_true_positive=cost_true_positive,
            cost_false_positive=cost_false_positive,
            cost_false_negative=cost_false_negative,
            cost_true_negative=cost_true_negative,
        )

        metrics = evaluate_model(
            y_true=pd.Series(y_true),
            y_pred=y_pred,
            y_score=y_score,
            top_k=min(10, len(y_true)),
        )

        row = {
            "threshold": threshold,
            "total_cost": cost_report["total_cost"],
            "cost_per_100_lots": cost_report["cost_per_100_lots"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "false_alarms_per_100_lots": metrics["false_alarms_per_100_lots"],
            "true_positive": cost_report["counts"]["true_positive"],
            "false_positive": cost_report["counts"]["false_positive"],
            "false_negative": cost_report["counts"]["false_negative"],
            "true_negative": cost_report["counts"]["true_negative"],
        }

        rows.append(row)

    results_df = pd.DataFrame(rows)
    best_row = results_df.sort_values(
        by=["total_cost", "false_negative", "false_positive"],
        ascending=[True, True, True],
    ).iloc[0]

    return {
        "best_threshold": float(best_row["threshold"]),
        "best_total_cost": float(best_row["total_cost"]),
        "best_cost_per_100_lots": float(best_row["cost_per_100_lots"]),
        "threshold_search_results": rows,
    }


def build_escalation_list(
    lot_ids: list[str],
    y_true: list[int],
    y_score: list[float],
    threshold: float,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Build an operational lot escalation list.

    The output is sorted by risk_score from high to low.
    """

    result_df = pd.DataFrame(
        {
            "lot_id": lot_ids,
            "true_label": y_true,
            "risk_score": y_score,
        }
    )

    result_df["escalate_by_threshold"] = result_df["risk_score"] >= threshold

    result_df["priority_rank"] = (
        result_df["risk_score"].rank(method="first", ascending=False).astype(int)
    )

    result_df["recommended_action"] = np.where(
        result_df["escalate_by_threshold"],
        "Escalate for engineer review",
        "Release / monitor",
    )

    result_df = result_df.sort_values("risk_score", ascending=False)

    return result_df.head(top_k).reset_index(drop=True)


def run_thresholding_demo(
    feature_table_file_name: str = "demo_training_feature_table.csv",
    model_file_name: str = "logistic_regression_baseline.joblib",
    report_file_name: str = "thresholding_report.json",
    escalation_file_name: str = "demo_escalation_list.csv",
    label_column: str = "pass_fail_label",
    id_column: str = "lot_id",
    test_size: float = 0.30,
    random_state: int = 42,
    top_k: int = 10,
) -> dict[str, Any]:
    """
    Run cost-sensitive thresholding using the saved baseline model.
    """

    ensure_directories_exist()

    feature_table_path = PROCESSED_DATA_DIR / feature_table_file_name
    model_path = MODELS_DIR / model_file_name

    if not feature_table_path.exists():
        raise FileNotFoundError(
            f"Feature table not found: {feature_table_path}. "
            "Please run src.models.train first."
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Please run src.models.train first."
        )

    logger.info("Loading feature table from: %s", feature_table_path)
    df = pd.read_csv(feature_table_path)

    X, y = prepare_xy(
        df=df,
        label_column=label_column,
        id_column=id_column,
    )

    lot_ids = df[id_column] if id_column in df.columns else pd.Series(range(len(df)))

    stratify = y if y.value_counts().min() >= 2 else None

    _, X_test, _, y_test, _, lot_ids_test = train_test_split(
        X,
        y,
        lot_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    logger.info("Loading model from: %s", model_path)
    model = joblib.load(model_path)

    y_score = model.predict_proba(X_test)[:, 1].tolist()

    default_threshold = 0.50
    simple_cost_threshold = derive_simple_cost_sensitive_threshold(
        cost_review=1.0,
        cost_false_negative=20.0,
    )

    default_pred = apply_threshold(y_score, default_threshold)
    simple_cost_pred = apply_threshold(y_score, simple_cost_threshold)

    default_metrics = evaluate_model(
        y_true=y_test,
        y_pred=default_pred,
        y_score=y_score,
        top_k=min(top_k, len(y_test)),
    )

    simple_cost_metrics = evaluate_model(
        y_true=y_test,
        y_pred=simple_cost_pred,
        y_score=y_score,
        top_k=min(top_k, len(y_test)),
    )

    default_cost = calculate_operational_cost(
        y_true=y_test.tolist(),
        y_pred=default_pred,
    )

    simple_cost = calculate_operational_cost(
        y_true=y_test.tolist(),
        y_pred=simple_cost_pred,
    )

    best_threshold_report = search_best_threshold_by_cost(
        y_true=y_test.tolist(),
        y_score=y_score,
    )

    best_threshold = best_threshold_report["best_threshold"]

    escalation_df = build_escalation_list(
        lot_ids=lot_ids_test.astype(str).tolist(),
        y_true=y_test.tolist(),
        y_score=y_score,
        threshold=best_threshold,
        top_k=min(top_k, len(y_test)),
    )

    escalation_path = PROCESSED_DATA_DIR / escalation_file_name
    escalation_df.to_csv(escalation_path, index=False)

    logger.info("Saved escalation list to: %s", escalation_path)

    report: dict[str, Any] = {
        "model_file": str(model_path),
        "feature_table_file": str(feature_table_path),
        "n_test_rows": int(len(y_test)),
        "cost_assumptions": {
            "cost_true_positive": 1.0,
            "cost_false_positive": 3.0,
            "cost_false_negative": 20.0,
            "cost_true_negative": 0.0,
            "cost_review_for_simple_rule": 1.0,
        },
        "default_threshold": {
            "threshold": default_threshold,
            "metrics": default_metrics,
            "cost": default_cost,
        },
        "simple_cost_sensitive_threshold": {
            "threshold": simple_cost_threshold,
            "metrics": simple_cost_metrics,
            "cost": simple_cost,
            "decision_rule": "Escalate if p_fail * cost_false_negative > cost_review.",
        },
        "best_threshold_by_realized_cost": {
            "threshold": best_threshold,
            "total_cost": best_threshold_report["best_total_cost"],
            "cost_per_100_lots": best_threshold_report["best_cost_per_100_lots"],
        },
        "escalation_list_file": str(escalation_path),
        "top_k": int(min(top_k, len(y_test))),
    }

    report_path = REPORTS_DIR / report_file_name

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    logger.info("Saved thresholding report to: %s", report_path)

    print("Top-K escalation list:")
    print(escalation_df)

    return report


def _demo() -> None:
    """
    Run cost-sensitive thresholding demo.
    """

    report = run_thresholding_demo()

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _demo()