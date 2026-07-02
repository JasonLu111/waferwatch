"""
Prediction utilities for WaferWatch.

This module loads a trained model and generates lot-level risk scores.

It supports:
- loading a saved model artifact
- loading a feature table
- generating predicted risk scores
- applying an escalation threshold
- saving prediction outputs
- saving a prediction report

This module will later be reused by FastAPI, batch inference, dashboard,
and monitoring pipelines.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.utils.config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, ensure_directories_exist
from src.utils.logger import get_logger


logger = get_logger(__name__)


def load_threshold_from_report(
    report_file_name: str = "thresholding_report.json",
    fallback_threshold: float = 0.50,
) -> float:
    """
    Load the best threshold from the thresholding report.

    If the report does not exist, use fallback_threshold.
    """

    report_path = REPORTS_DIR / report_file_name

    if not report_path.exists():
        logger.warning(
            "Threshold report not found: %s. Using fallback threshold: %s",
            report_path,
            fallback_threshold,
        )
        return fallback_threshold

    with report_path.open("r", encoding="utf-8") as file:
        report = json.load(file)

    threshold = report.get("best_threshold_by_realized_cost", {}).get(
        "threshold",
        fallback_threshold,
    )

    logger.info("Loaded threshold from report: %s", threshold)

    return float(threshold)


def load_feature_columns_from_training_report(
    report_file_name: str = "baseline_training_report.json",
) -> list[str]:
    """
    Load feature column names from the training report.

    This ensures prediction uses the same feature columns and order
    as model training.
    """

    report_path = REPORTS_DIR / report_file_name

    if not report_path.exists():
        raise FileNotFoundError(
            f"Training report not found: {report_path}. "
            "Please run src.models.train first."
        )

    with report_path.open("r", encoding="utf-8") as file:
        report = json.load(file)

    feature_columns = report.get("feature_columns")

    if not feature_columns:
        raise ValueError(
            "No feature_columns found in training report. "
            "Please check baseline_training_report.json."
        )

    return feature_columns


def prepare_prediction_features(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Prepare feature matrix for prediction.

    The input DataFrame may contain lot_id or pass_fail_label,
    but prediction only uses the feature columns used during training.
    """

    missing_columns = [column for column in feature_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(
            "Feature table is missing required columns: "
            f"{missing_columns}"
        )

    X = df[feature_columns].copy()

    return X


def predict_risk_scores(
    model: Any,
    X: pd.DataFrame,
) -> list[float]:
    """
    Generate risk scores from a trained model.

    For binary classification models with predict_proba(),
    the risk score is the probability of class 1.
    """

    if hasattr(model, "predict_proba"):
        risk_scores = model.predict_proba(X)[:, 1].tolist()
    else:
        risk_scores = model.predict(X).tolist()

    return [float(score) for score in risk_scores]


def build_prediction_table(
    input_df: pd.DataFrame,
    risk_scores: list[float],
    threshold: float,
    id_column: str = "lot_id",
    label_column: str = "pass_fail_label",
) -> pd.DataFrame:
    """
    Build a lot-level prediction table.

    Output columns include:
    - lot_id
    - risk_score
    - predicted_label
    - recommended_action
    - optional true label if available
    """

    prediction_df = pd.DataFrame()

    if id_column in input_df.columns:
        prediction_df[id_column] = input_df[id_column]
    else:
        prediction_df[id_column] = [f"ROW_{i:06d}" for i in range(len(input_df))]

    prediction_df["risk_score"] = risk_scores
    prediction_df["threshold"] = threshold
    prediction_df["predicted_label"] = (
        prediction_df["risk_score"] >= threshold
    ).astype(int)

    prediction_df["recommended_action"] = prediction_df["predicted_label"].map(
        {
            1: "Escalate for engineer review",
            0: "Release / monitor",
        }
    )

    if label_column in input_df.columns:
        prediction_df["true_label"] = input_df[label_column]

    prediction_df = prediction_df.sort_values(
        "risk_score",
        ascending=False,
    ).reset_index(drop=True)

    prediction_df["priority_rank"] = prediction_df.index + 1

    return prediction_df


def run_batch_prediction(
    feature_table_file_name: str = "demo_training_feature_table.csv",
    model_file_name: str = "logistic_regression_baseline.joblib",
    output_file_name: str = "demo_predictions.csv",
    report_file_name: str = "prediction_report.json",
    id_column: str = "lot_id",
    label_column: str = "pass_fail_label",
) -> dict[str, Any]:
    """
    Run batch prediction using a saved model and feature table.
    """

    ensure_directories_exist()

    feature_table_path = PROCESSED_DATA_DIR / feature_table_file_name
    model_path = MODELS_DIR / model_file_name

    if not feature_table_path.exists():
        raise FileNotFoundError(
            f"Feature table not found: {feature_table_path}. "
            "Please run feature engineering or training first."
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Please run src.models.train first."
        )

    logger.info("Loading feature table from: %s", feature_table_path)
    input_df = pd.read_csv(feature_table_path)

    logger.info("Loading model from: %s", model_path)
    model = joblib.load(model_path)

    feature_columns = load_feature_columns_from_training_report()
    threshold = load_threshold_from_report()

    X = prepare_prediction_features(
        df=input_df,
        feature_columns=feature_columns,
    )

    risk_scores = predict_risk_scores(
        model=model,
        X=X,
    )

    prediction_df = build_prediction_table(
        input_df=input_df,
        risk_scores=risk_scores,
        threshold=threshold,
        id_column=id_column,
        label_column=label_column,
    )

    output_path = PROCESSED_DATA_DIR / output_file_name
    prediction_df.to_csv(output_path, index=False)

    logger.info("Saved prediction table to: %s", output_path)

    n_lots = int(len(prediction_df))
    n_escalated = int(prediction_df["predicted_label"].sum())
    escalation_rate = float(n_escalated / n_lots) if n_lots > 0 else 0.0

    report: dict[str, Any] = {
        "model_file": str(model_path),
        "feature_table_file": str(feature_table_path),
        "prediction_file": str(output_path),
        "n_lots_scored": n_lots,
        "n_escalated": n_escalated,
        "escalation_rate": escalation_rate,
        "threshold": threshold,
        "feature_columns": feature_columns,
        "top_10_lots": prediction_df.head(10).to_dict(orient="records"),
    }

    report_path = REPORTS_DIR / report_file_name

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    logger.info("Saved prediction report to: %s", report_path)

    print("Top 10 predicted high-risk lots:")
    print(prediction_df.head(10))

    return report


def _demo() -> None:
    """
    Run batch prediction demo.
    """

    report = run_batch_prediction()

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _demo()