"""FastAPI entry point for the WaferWatch local MLOps prototype."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient
from pydantic import BaseModel, Field


from src.rag.generate import answer_question as local_rag_answer
from src.rag.llm_generate import answer_question_with_openai


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
DRIFT_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "drift_monitoring_report.json"
)
DIRECT_ARTIFACT_LOAD = os.getenv(
    "WAFERWATCH_DIRECT_ARTIFACT_LOAD",
    "false",
).lower() == "true"

LOCAL_ARTIFACT_ROOT = Path(
    os.getenv(
        "WAFERWATCH_LOCAL_ARTIFACT_ROOT",
        str(PROJECT_ROOT / "mlflow_data" / "artifacts"),
    )
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


class BatchPredictionRequest(BaseModel):
    """A bounded batch of lot feature records."""

    lots: list[PredictionRequest] = Field(
        min_length=1,
        max_length=1000,
        description="One to 1,000 lots to score in request order.",
    )


class BatchPredictionResponse(BaseModel):
    """Batch scoring result and shared model metadata."""

    predictions: list[PredictionResponse]
    n_lots_scored: int
    n_escalated: int
    threshold: float
    model_version: str
    model_run_id: str
    model_uri: str

class RagQueryRequest(BaseModel):
    """Evidence-grounded RCA question submitted to the RAG service."""

    query: str = Field(
        min_length=1,
        max_length=1000,
        description="Question about a lot, tool, chamber, or recipe.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of retrieved evidence items to consider.",
    )
    generation_mode: Literal["local", "openai"] = Field(
        default="local",
        description=(
            "Use local deterministic generation by default. "
            "OpenAI generation is used only when explicitly requested."
        ),
    )


class RagQueryResponse(BaseModel):
    """Evidence-grounded RAG answer."""

    answer: str
    generation_mode: str
    top_k: int

class DriftMonitoringResponse(BaseModel):
    """Latest persisted feature-drift monitoring result."""

    report: dict[str, Any]


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

def resolve_local_champion_artifact_path(
    version: Any,
) -> Path:
    """Resolve a mounted MLflow 3 logged-model artifact path."""

    source = str(version.source)

    if not source.startswith("models:/"):
        raise ValueError(
            "Champion model source is not an MLflow logged-model URI: "
            f"{source}"
        )

    model_id = source.removeprefix("models:/").split("/", 1)[0]

    if not model_id.startswith("m-"):
        raise ValueError(
            f"Unable to extract a logged-model ID from source: {source}"
        )

    run = get_registry_client().get_run(version.run_id)
    experiment_id = str(run.info.experiment_id)

    artifact_path = (
        LOCAL_ARTIFACT_ROOT
        / experiment_id
        / "models"
        / model_id
        / "artifacts"
    )

    if not (artifact_path / "MLmodel").is_file():
        raise FileNotFoundError(
            "Champion MLmodel file was not found at: "
            f"{artifact_path}"
        )

    return artifact_path



@lru_cache
@lru_cache
def load_champion_model() -> Any:
    """Load the champion model through Registry or local mounted artifacts."""

    version = get_current_model_version()
    model_uri = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"

    if DIRECT_ARTIFACT_LOAD:
        try:
            artifact_path = resolve_local_champion_artifact_path(
                version,
            )
            return mlflow.sklearn.load_model(str(artifact_path))
        except Exception as error:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Unable to load the champion model from mounted "
                    f"MLflow artifacts: {error}"
                ),
            ) from error

    try:
        return mlflow.sklearn.load_model(model_uri)
    except Exception as registry_error:
        try:
            artifact_path = resolve_local_champion_artifact_path(
                version,
            )
            return mlflow.sklearn.load_model(str(artifact_path))
        except Exception as local_error:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Unable to load the current MLflow champion model. "
                    f"Registry load failed: {registry_error}. "
                    f"Local artifact fallback failed: {local_error}"
                ),
            ) from local_error


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


def validate_feature_columns(features: dict[str, float]) -> None:
    """Reject incomplete or unexpected feature payloads."""

    missing_columns = [
        column
        for column in FEATURE_COLUMNS
        if column not in features
    ]
    unexpected_columns = sorted(
        set(features) - set(FEATURE_COLUMNS)
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


def score_lot(
    request: PredictionRequest,
    model: Any,
    threshold: float,
    version: Any,
) -> PredictionResponse:
    """Score one lot using already-resolved model resources."""

    validate_feature_columns(request.features)

    feature_frame = pd.DataFrame(
        [[request.features[column] for column in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS,
    )

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

    model = load_champion_model()
    threshold = load_champion_threshold()
    version = get_current_model_version()

    return score_lot(
        request=request,
        model=model,
        threshold=threshold,
        version=version,
    )


@app.post(
    "/batch_predict",
    response_model=BatchPredictionResponse,
)
def batch_predict(
    request: BatchPredictionRequest,
) -> BatchPredictionResponse:
    """Score a batch of lots with one consistent champion model version."""

    model = load_champion_model()
    threshold = load_champion_threshold()
    version = get_current_model_version()

    predictions = [
        score_lot(
            request=lot,
            model=model,
            threshold=threshold,
            version=version,
        )
        for lot in request.lots
    ]

    return BatchPredictionResponse(
        predictions=predictions,
        n_lots_scored=len(predictions),
        n_escalated=sum(
            prediction.predicted_label
            for prediction in predictions
        ),
        threshold=threshold,
        model_version=str(version.version),
        model_run_id=version.run_id,
        model_uri=(
            f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
        ),
    )


@app.post("/rag/query", response_model=RagQueryResponse)
def rag_query(request: RagQueryRequest) -> RagQueryResponse:
    """Answer an RCA query using retrieved WaferWatch evidence only."""

    try:
        if request.generation_mode == "openai":
            answer = answer_question_with_openai(
                request.query,
                top_k=request.top_k,
            )
        else:
            answer = local_rag_answer(
                request.query,
                top_k=request.top_k,
            )
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail=f"RAG knowledge base is unavailable: {error}",
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"RAG query failed: {error}",
        ) from error

    return RagQueryResponse(
        answer=answer,
        generation_mode=request.generation_mode,
        top_k=request.top_k,
    )


@app.get(
    "/monitoring/drift",
    response_model=DriftMonitoringResponse,
)
def monitoring_drift() -> DriftMonitoringResponse:
    """Return the latest persisted drift monitoring report."""

    if not DRIFT_REPORT_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "Drift monitoring report is unavailable: "
                f"{DRIFT_REPORT_PATH}"
            ),
        )

    try:
        report = json.loads(
            DRIFT_REPORT_PATH.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as error:
        raise HTTPException(
            status_code=503,
            detail="Drift monitoring report is not valid JSON.",
        ) from error

    required_fields = {
        "overall_drift_detected",
        "n_features_with_drift",
        "drifted_features",
        "thresholds",
    }

    missing_fields = sorted(required_fields - set(report))

    if missing_fields:
        raise HTTPException(
            status_code=503,
            detail=(
                "Drift monitoring report is missing required fields: "
                f"{missing_fields}"
            ),
        )

    return DriftMonitoringResponse(report=report)