"""FastAPI entry point for the WaferWatch local MLOps prototype."""

from __future__ import annotations

import os

import mlflow
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient


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

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_registry_uri(TRACKING_URI)

app = FastAPI(
    title="WaferWatch API",
    version="0.1.0",
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
    """Return the MLflow model currently assigned to the champion alias."""

    try:
        client = MlflowClient(
            tracking_uri=TRACKING_URI,
            registry_uri=TRACKING_URI,
        )
        version = client.get_model_version_by_alias(
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

    return {
        "registered_model_name": REGISTERED_MODEL_NAME,
        "alias": MODEL_ALIAS,
        "version": str(version.version),
        "run_id": version.run_id,
        "model_uri": (
            f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
        ),
    }
