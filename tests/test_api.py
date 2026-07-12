"""Offline API tests for the WaferWatch FastAPI service."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api import main as api_main


class FakeChampionModel:
    """Small deterministic model substitute for API tests."""

    def predict_proba(self, features: Any) -> np.ndarray:
        probabilities = []

        for _, row in features.iterrows():
            if row["spc_violation_count"] >= 1:
                probabilities.append([0.18, 0.82])
            else:
                probabilities.append([0.96, 0.04])

        return np.array(probabilities)


@pytest.fixture(autouse=True)
def clear_api_caches() -> None:
    """Keep tests independent of cached runtime model resources."""

    api_main.load_champion_model.cache_clear()
    api_main.load_champion_threshold.cache_clear()

    yield

    api_main.load_champion_model.cache_clear()
    api_main.load_champion_threshold.cache_clear()


@pytest.fixture
def client() -> TestClient:
    """Create a FastAPI test client without starting Uvicorn."""

    return TestClient(api_main.app)


@pytest.fixture
def feature_values() -> dict[str, float]:
    """Return one valid selected-SPC feature payload."""

    return {
        "sensor_mean": 0.0,
        "sensor_std": 0.0,
        "sensor_min": 0.0,
        "sensor_max": 0.0,
        "spc_violation_count": 0.0,
        "spc_max_abs_zscore": 0.0,
    }


@pytest.fixture
def mock_model_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    """Replace MLflow-dependent resources with deterministic test doubles."""

    model_version = SimpleNamespace(
        version="2",
        run_id="test-champion-run-id",
    )

    monkeypatch.setattr(
        api_main,
        "get_current_model_version",
        lambda: model_version,
    )
    monkeypatch.setattr(
        api_main,
        "load_champion_model",
        lambda: FakeChampionModel(),
    )
    monkeypatch.setattr(
        api_main,
        "load_champion_threshold",
        lambda: 0.05,
    )

    return model_version


def test_root_and_health(client: TestClient) -> None:
    """The basic service endpoints should return healthy responses."""

    root_response = client.get("/")
    health_response = client.get("/health")

    assert root_response.status_code == 200
    assert root_response.json() == {
        "service": "WaferWatch API",
        "status": "ok",
    }
    assert health_response.status_code == 200
    assert health_response.json() == {"status": "ok"}


def test_current_model_returns_champion_metadata(
    client: TestClient,
    mock_model_dependencies: SimpleNamespace,
) -> None:
    """The current-model endpoint should expose champion metadata."""

    response = client.get("/model/current")

    assert response.status_code == 200
    assert response.json() == {
        "registered_model_name": "WaferWatchRiskModel",
        "alias": "champion",
        "version": "2",
        "run_id": "test-champion-run-id",
        "model_uri": "models:/WaferWatchRiskModel@champion",
    }


def test_predict_returns_release_decision(
    client: TestClient,
    feature_values: dict[str, float],
    mock_model_dependencies: SimpleNamespace,
) -> None:
    """A low-risk lot should be released or monitored."""

    response = client.post(
        "/predict",
        json={
            "lot_id": "LOT_API_LOW_RISK",
            "features": feature_values,
        },
    )

    payload = response.json()

    assert response.status_code == 200
    assert payload["lot_id"] == "LOT_API_LOW_RISK"
    assert payload["risk_score"] == 0.04
    assert payload["threshold"] == 0.05
    assert payload["predicted_label"] == 0
    assert payload["recommended_action"] == "Release / monitor"
    assert payload["model_version"] == "2"


def test_batch_predict_returns_consistent_decisions(
    client: TestClient,
    feature_values: dict[str, float],
    mock_model_dependencies: SimpleNamespace,
) -> None:
    """Batch prediction should preserve order and share model metadata."""

    high_risk_features = dict(feature_values)
    high_risk_features["spc_violation_count"] = 1.0

    response = client.post(
        "/batch_predict",
        json={
            "lots": [
                {
                    "lot_id": "LOT_API_LOW_RISK",
                    "features": feature_values,
                },
                {
                    "lot_id": "LOT_API_HIGH_RISK",
                    "features": high_risk_features,
                },
            ]
        },
    )

    payload = response.json()

    assert response.status_code == 200
    assert payload["n_lots_scored"] == 2
    assert payload["n_escalated"] == 1
    assert payload["threshold"] == 0.05
    assert payload["model_version"] == "2"
    assert payload["predictions"][0]["lot_id"] == "LOT_API_LOW_RISK"
    assert payload["predictions"][0]["predicted_label"] == 0
    assert payload["predictions"][1]["lot_id"] == "LOT_API_HIGH_RISK"
    assert payload["predictions"][1]["predicted_label"] == 1
    assert (
        payload["predictions"][1]["recommended_action"]
        == "Escalate for engineer review"
    )


def test_predict_rejects_missing_feature_column(
    client: TestClient,
    feature_values: dict[str, float],
    mock_model_dependencies: SimpleNamespace,
) -> None:
    """The API should reject an incomplete selected-feature payload."""

    incomplete_features = dict(feature_values)
    incomplete_features.pop("sensor_std")

    response = client.post(
        "/predict",
        json={
            "lot_id": "LOT_API_INVALID",
            "features": incomplete_features,
        },
    )

    payload = response.json()

    assert response.status_code == 422
    assert "sensor_std" in payload["detail"]["missing_feature_columns"]


def test_rag_query_uses_local_grounded_generator(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The local RAG endpoint should return its grounded answer unchanged."""

    grounded_answer = (
        "Question:\nWhy was LOT_TEST escalated?\n\n"
        "Evidence IDs:\n- CASE_TEST"
    )

    monkeypatch.setattr(
        api_main,
        "local_rag_answer",
        lambda question, top_k: grounded_answer,
    )

    response = client.post(
        "/rag/query",
        json={
            "query": "Why was LOT_TEST escalated?",
            "top_k": 3,
            "generation_mode": "local",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": grounded_answer,
        "generation_mode": "local",
        "top_k": 3,
    }


def test_drift_endpoint_returns_persisted_report(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """The drift endpoint should return a valid persisted report."""

    report = {
        "reference_rows": 40,
        "current_rows": 40,
        "n_features_monitored": 6,
        "n_features_with_drift": 1,
        "drifted_features": ["sensor_mean"],
        "thresholds": {
            "psi_threshold": 0.25,
            "mean_shift_threshold": 1.0,
            "missing_rate_shift_threshold": 0.10,
        },
        "overall_drift_detected": True,
    }

    report_path = tmp_path / "drift_monitoring_report.json"
    report_path.write_text(
        json.dumps(report),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        api_main,
        "DRIFT_REPORT_PATH",
        report_path,
    )

    response = client.get("/monitoring/drift")

    assert response.status_code == 200
    assert response.json() == {"report": report}