"""FastAPI entry point for the WaferWatch local MLOps prototype."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)
REGISTERED_MODEL_NAME = os.getenv(
    "WAFERWATCH_MODEL_NAME",
    "WaferWatchRiskModel",
)
MODEL_ALIAS = os.getenv(
    "WAFERWATCH_MODEL_ALIAS",
    "champion",
)
THRESHOLD_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "spc_selected_thresholding_report.json"
)

FEATURE_COLUMNS = [
    "sensor_mean",
    "sensor_std",
    "sensor_min",
    "sensor_max",
    "spc_violation_count",
    "spc_max_abs_zscore",
]

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_registry_uri(TRACKING_URI)

app = FastAPI(
    title="WaferWatch API",
    version="0.1.0",
)


class PredictionRequest(BaseModel):
    """One lot's selected SPC feature values for risk scoring."""

    lot_id: str | None = Field(
        default=None,
        description="Optional lot identifier supplied by the API caller.",
    )
    features: dict[str, float] = Field(
        description="Selected SPC feature name-value pairs.",
    )


class PredictionResponse(BaseModel):
    """Cost-sensitive lot escalation decision."""

    lot_id: str
    risk_score: float
    threshold: float
    predicted_label: int
    recommended_action: str
    model_version: str
    model_run_id: str
    model_uri: str


def get_registry_client() -> MlflowClient:
    """Create an MLflow client for tracking and registry operations."""

    return MlflowClient(
        tracking_uri=TRACKING_URI,
        registry_uri=TRACKING_URI,
    )


def get_current_model_version() -> Any:
    """Resolve the model version assigned to the configured alias."""

    try:
        return get_registry_client().get_model_version_by_alias(
            REGISTERED_MODEL_NAME,
            MODEL_ALIAS,
        )
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "Unable to resolve the current MLflow champion model: "
                f"{error}"
            ),
        ) from error


@lru_cache
def load_champion_model() -> Any:
    """Load and cache the current champion model from MLflow Registry."""

    model_uri = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"

    try:
        return mlflow.sklearn.load_model(model_uri)
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "Unable to load the current MLflow champion model: "
                f"{error}"
            ),
        ) from error


@lru_cache
def load_champion_threshold() -> float:
    """Load the cost-sensitive threshold for the registered champion."""

    if not THRESHOLD_REPORT_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "Champion threshold report is unavailable: "
                f"{THRESHOLD_REPORT_PATH}"
            ),
        )

    report = json.loads(
        THRESHOLD_REPORT_PATH.read_text(encoding="utf-8")
    )

    threshold = report.get(
        "best_threshold_by_realized_cost",
        {},
    ).get("threshold")

    if not isinstance(threshold, (int, float)):
        raise HTTPException(
            status_code=503,
            detail="Champion threshold report has no valid threshold.",
        )

    threshold = float(threshold)

    if not 0.0 <= threshold <= 1.0:
        raise HTTPException(
            status_code=503,
            detail="Champion threshold must be between 0 and 1.",
        )

    return threshold


@app.get("/")
def root() -> dict[str, str]:
    """Return basic API identification."""

    return {
        "service": "WaferWatch API",
        "status": "ok",
    }


@app.get("/health")
def health() -> dict[str, str]:
    """Return service health status."""

    return {"status": "ok"}


@app.get("/model/current")
def current_model() -> dict[str, str]:
    """Return the MLflow model assigned to the champion alias."""

    version = get_current_model_version()

    return {
        "registered_model_name": REGISTERED_MODEL_NAME,
        "alias": MODEL_ALIAS,
        "version": str(version.version),
        "run_id": version.run_id,
        "model_uri": (
            f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
        ),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Score one lot and return its cost-sensitive escalation decision."""

    missing_columns = [
        column
        for column in FEATURE_COLUMNS
        if column not in request.features
    ]
    unexpected_columns = sorted(
        set(request.features) - set(FEATURE_COLUMNS)
    )

    if missing_columns or unexpected_columns:
        raise HTTPException(
            status_code=422,
            detail={
                "missing_feature_columns": missing_columns,
                "unexpected_feature_columns": unexpected_columns,
                "required_feature_columns": FEATURE_COLUMNS,
            },
        )

    feature_frame = pd.DataFrame(
        [[request.features[column] for column in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS,
    )

    model = load_champion_model()
    threshold = load_champion_threshold()
    version = get_current_model_version()

    if not hasattr(model, "predict_proba"):
        raise HTTPException(
            status_code=503,
            detail=(
                "Current champion model does not support probability "
                "predictions."
            ),
        )

    risk_score = float(model.predict_proba(feature_frame)[0, 1])
    predicted_label = int(risk_score >= threshold)

    return PredictionResponse(
        lot_id=request.lot_id or "UNSPECIFIED_LOT",
        risk_score=risk_score,
        threshold=threshold,
        predicted_label=predicted_label,
        recommended_action=(
            "Escalate for engineer review"
            if predicted_label == 1
            else "Release / monitor"
        ),
        model_version=str(version.version),
        model_run_id=version.run_id,
        model_uri=(
            f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
        ),
    )