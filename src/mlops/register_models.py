"""
Track existing WaferWatch supervised models with MLflow and register a champion.

The script reads the existing model-family comparison report instead of
retraining models. It logs each deployment-eligible supervised model as an
MLflow run, registers the selected champion, and assigns the registry alias
"champion".

Champion selection order:

1. Higher PR-AUC
2. Higher recall
3. Higher precision
4. Higher F1
5. Lower false alarms per 100 lots
6. Smaller serialized model artifact

The final size-based rule provides a deterministic deployment tie-breaker when
the evaluation metrics are equal.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from numbers import Real
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MLFLOW_DATA_DIR = PROJECT_ROOT / "mlflow_data"

MODEL_COMPARISON_REPORT = (
    REPORTS_DIR / "model_family_comparison_report.json"
)
MODEL_COMPARISON_MARKDOWN = (
    REPORTS_DIR / "model_family_comparison_report.md"
)
FEATURE_TABLE_PATH = (
    PROCESSED_DATA_DIR / "demo_spc_selected_feature_table.csv"
)

DEFAULT_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)
DEFAULT_EXPERIMENT_NAME = "WaferWatch-R4-Model-Tracking"
DEFAULT_REGISTERED_MODEL_NAME = "WaferWatchRiskModel"
DEFAULT_MODEL_ALIAS = "champion"

LABEL_COLUMN = "pass_fail_label"

FEATURE_COLUMNS = [
    "sensor_mean",
    "sensor_std",
    "sensor_min",
    "sensor_max",
    "spc_violation_count",
    "spc_max_abs_zscore",
]

DEPLOYMENT_CANDIDATES = {
    "logistic_regression": {
        "learning_type": "supervised_classification",
        "deployment_role": "risk_probability_classifier",
    },
    "random_forest": {
        "learning_type": "supervised_classification",
        "deployment_role": "risk_probability_classifier",
    },
    "gradient_boosting": {
        "learning_type": "supervised_classification",
        "deployment_role": "risk_probability_classifier",
    },
}


def load_json(path: Path) -> dict[str, Any]:
    """Load and validate a JSON object."""

    if not path.exists():
        raise FileNotFoundError(f"Required JSON file not found: {path}")

    content = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(content, dict):
        raise ValueError(f"Expected a JSON object in: {path}")

    return content


def load_feature_table() -> pd.DataFrame:
    """Load and validate the selected SPC feature table."""

    if not FEATURE_TABLE_PATH.exists():
        raise FileNotFoundError(
            f"Feature table not found: {FEATURE_TABLE_PATH}"
        )

    frame = pd.read_csv(FEATURE_TABLE_PATH)

    required_columns = [
        "lot_id",
        "timestamp",
        LABEL_COLUMN,
        *FEATURE_COLUMNS,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in frame.columns
    ]

    if missing_columns:
        raise ValueError(
            "Feature table is missing required columns: "
            + ", ".join(missing_columns)
        )

    if frame.empty:
        raise ValueError(
            f"Feature table contains no rows: {FEATURE_TABLE_PATH}"
        )

    return frame


def resolve_model_path(model_info: dict[str, Any]) -> Path:
    """Resolve a model file from the comparison report."""

    model_file = model_info.get("model_file")

    if not model_file:
        raise ValueError(
            "Model comparison report entry has no model_file value."
        )

    model_path = MODELS_DIR / Path(str(model_file)).name

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {model_path}"
        )

    return model_path


def metric_value(
    metrics: dict[str, Any],
    name: str,
    default: float,
) -> float:
    """Return one numeric metric as a float."""

    value = metrics.get(name, default)

    if isinstance(value, bool) or not isinstance(value, Real):
        return default

    return float(value)


def build_selection_key(
    model_info: dict[str, Any],
    model_path: Path,
) -> tuple[float, float, float, float, float, float]:
    """
    Build a tuple used to select the deployment champion.

    Python selects the maximum tuple lexicographically. Metrics that should be
    minimized are negated.
    """

    metrics = model_info.get("evaluation_metrics", {})

    if not isinstance(metrics, dict):
        raise ValueError(
            "evaluation_metrics must be a JSON object."
        )

    false_alarm_rate = metric_value(
        metrics,
        "false_alarms_per_100_lots",
        float("inf"),
    )

    return (
        metric_value(metrics, "pr_auc", float("-inf")),
        metric_value(metrics, "recall", float("-inf")),
        metric_value(metrics, "precision", float("-inf")),
        metric_value(metrics, "f1", float("-inf")),
        -false_alarm_rate,
        -float(model_path.stat().st_size),
    )


def select_champion(
    report: dict[str, Any],
) -> tuple[str, dict[str, Any], Path]:
    """Select the champion from supervised deployment candidates."""

    report_models = report.get("models")

    if not isinstance(report_models, dict):
        raise ValueError(
            "Model comparison report has no valid models object."
        )

    available_candidates: list[
        tuple[str, dict[str, Any], Path]
    ] = []

    for model_key in DEPLOYMENT_CANDIDATES:
        model_info = report_models.get(model_key)

        if not isinstance(model_info, dict):
            raise ValueError(
                f"Missing model report entry: {model_key}"
            )

        model_path = resolve_model_path(model_info)

        model = joblib.load(model_path)

        if not hasattr(model, "predict_proba"):
            raise TypeError(
                f"Deployment candidate {model_key} does not implement "
                "predict_proba()."
            )

        available_candidates.append(
            (model_key, model_info, model_path)
        )

    return max(
        available_candidates,
        key=lambda candidate: build_selection_key(
            candidate[1],
            candidate[2],
        ),
    )


def flatten_parameters(
    values: dict[str, Any],
    prefix: str = "",
) -> dict[str, str]:
    """Convert nested model parameters into MLflow-compatible strings."""

    flattened: dict[str, str] = {}

    for key, value in values.items():
        parameter_name = f"{prefix}.{key}" if prefix else str(key)

        if isinstance(value, dict):
            flattened.update(
                flatten_parameters(
                    value,
                    prefix=parameter_name,
                )
            )
        elif isinstance(value, (list, tuple, set)):
            flattened[parameter_name] = json.dumps(
                list(value),
                default=str,
                sort_keys=True,
            )
        elif value is None:
            flattened[parameter_name] = "None"
        else:
            flattened[parameter_name] = str(value)

    return flattened


def extract_numeric_metrics(
    model_info: dict[str, Any],
) -> dict[str, float]:
    """Extract metrics that MLflow can log."""

    source_metrics = model_info.get("evaluation_metrics", {})

    if not isinstance(source_metrics, dict):
        raise ValueError(
            "evaluation_metrics must be a JSON object."
        )

    numeric_metrics: dict[str, float] = {}

    for metric_name, metric_value_raw in source_metrics.items():
        if (
            isinstance(metric_value_raw, Real)
            and not isinstance(metric_value_raw, bool)
        ):
            numeric_metrics[str(metric_name)] = float(
                metric_value_raw
            )

    return numeric_metrics


def wait_for_registered_version(
    client: MlflowClient,
    registered_model_name: str,
    run_id: str,
    timeout_seconds: int = 60,
) -> str:
    """Wait until the registry version associated with a run is available."""

    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        versions = client.search_model_versions(
            f"name='{registered_model_name}'"
        )

        matching_versions = [
            version
            for version in versions
            if version.run_id == run_id
        ]

        if matching_versions:
            latest = max(
                matching_versions,
                key=lambda version: int(version.version),
            )
            return str(latest.version)

        time.sleep(1)

    raise TimeoutError(
        "Timed out while waiting for registered model version "
        f"for run {run_id}."
    )


def print_dry_run_summary(
    report: dict[str, Any],
    tracking_uri: str,
    experiment_name: str,
    registered_model_name: str,
    alias: str,
) -> None:
    """Validate inputs and print the future registration plan."""

    frame = load_feature_table()

    champion_key, champion_info, champion_path = (
        select_champion(report)
    )

    report_models = report["models"]

    print("MLFLOW_DRY_RUN_OK")
    print(f"Tracking URI: {tracking_uri}")
    print(f"Experiment: {experiment_name}")
    print(f"Registered model: {registered_model_name}")
    print(f"Registry alias: {alias}")
    print(f"Feature table rows: {len(frame)}")
    print(f"Feature count: {len(FEATURE_COLUMNS)}")
    print("Feature columns:")

    for feature_name in FEATURE_COLUMNS:
        print(f"  - {feature_name}")

    print("")
    print("Deployment candidates:")

    for model_key in DEPLOYMENT_CANDIDATES:
        model_info = report_models[model_key]
        model_path = resolve_model_path(model_info)
        model = joblib.load(model_path)
        metrics = extract_numeric_metrics(model_info)

        print(
            f"  - {model_key}: "
            f"class={model.__class__.__name__}, "
            f"pr_auc={metrics.get('pr_auc')}, "
            f"recall={metrics.get('recall')}, "
            f"precision={metrics.get('precision')}, "
            f"false_alarms_per_100_lots="
            f"{metrics.get('false_alarms_per_100_lots')}, "
            f"size_bytes={model_path.stat().st_size}"
        )

    print("")
    print(f"Selected champion key: {champion_key}")
    print(f"Selected champion name: {champion_info.get('name')}")
    print(f"Selected champion artifact: {champion_path}")
    print(
        "Selection rule: performance metrics first; "
        "smaller artifact resolves an exact metric tie."
    )


def register_models(
    report: dict[str, Any],
    tracking_uri: str,
    experiment_name: str,
    registered_model_name: str,
    alias: str,
) -> dict[str, Any]:
    """Log supervised candidate models and register the champion."""

    frame = load_feature_table()

    X = frame[FEATURE_COLUMNS].copy()
    input_example = X.head(5)

    champion_key, champion_info, champion_path = (
        select_champion(report)
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    client = MlflowClient(
        tracking_uri=tracking_uri,
        registry_uri=tracking_uri,
    )

    report_models = report["models"]
    logged_runs: dict[str, dict[str, Any]] = {}
    champion_run_id: str | None = None

    for model_key in DEPLOYMENT_CANDIDATES:
        model_info = report_models[model_key]
        model_path = resolve_model_path(model_info)
        model = joblib.load(model_path)

        predictions = model.predict(input_example)
        signature = infer_signature(
            input_example,
            predictions,
        )

        model_parameters = model_info.get(
            "model_parameters",
            {},
        )

        if not isinstance(model_parameters, dict):
            model_parameters = {}

        mlflow_parameters = {
            "model_key": model_key,
            "model_name": str(
                model_info.get("name", model_key)
            ),
            "model_class": model.__class__.__name__,
            "model_file": model_path.name,
            "feature_table": FEATURE_TABLE_PATH.name,
            "label_column": LABEL_COLUMN,
            "n_features": str(len(FEATURE_COLUMNS)),
            "artifact_size_bytes": str(
                model_path.stat().st_size
            ),
            **flatten_parameters(
                model_parameters,
                prefix="model",
            ),
        }

        mlflow_metrics = extract_numeric_metrics(model_info)

        is_champion = model_key == champion_key

        with mlflow.start_run(
            run_name=f"r4-{model_key}"
        ) as active_run:
            run_id = active_run.info.run_id

            mlflow.set_tags(
                {
                    "project": "WaferWatch",
                    "phase": "R4",
                    "learning_type": (
                        DEPLOYMENT_CANDIDATES[
                            model_key
                        ]["learning_type"]
                    ),
                    "deployment_role": (
                        DEPLOYMENT_CANDIDATES[
                            model_key
                        ]["deployment_role"]
                    ),
                    "deployment_eligible": "true",
                    "selected_champion": str(
                        is_champion
                    ).lower(),
                    "data_scope": (
                        "controlled_synthetic_spc_demo"
                    ),
                }
            )

            mlflow.log_params(mlflow_parameters)
            mlflow.log_metrics(mlflow_metrics)

            mlflow.log_artifact(
                str(MODEL_COMPARISON_REPORT),
                artifact_path="evaluation",
            )

            if MODEL_COMPARISON_MARKDOWN.exists():
                mlflow.log_artifact(
                    str(MODEL_COMPARISON_MARKDOWN),
                    artifact_path="evaluation",
                )

            log_model_arguments: dict[str, Any] = {
                "sk_model": model,
                "name": "model",
                "input_example": input_example,
                "signature": signature,
            }

            if is_champion:
                log_model_arguments[
                    "registered_model_name"
                ] = registered_model_name

            model_info_result = mlflow.sklearn.log_model(
                **log_model_arguments
            )

            logged_runs[model_key] = {
                "run_id": run_id,
                "model_uri": model_info_result.model_uri,
                "is_champion": is_champion,
            }

            if is_champion:
                champion_run_id = run_id

    if champion_run_id is None:
        raise RuntimeError(
            "No champion run was logged."
        )

    champion_version = wait_for_registered_version(
        client=client,
        registered_model_name=registered_model_name,
        run_id=champion_run_id,
    )

    client.set_registered_model_alias(
        name=registered_model_name,
        alias=alias,
        version=champion_version,
    )

    client.set_model_version_tag(
        name=registered_model_name,
        version=champion_version,
        key="project",
        value="WaferWatch",
    )

    client.set_model_version_tag(
        name=registered_model_name,
        version=champion_version,
        key="phase",
        value="R4",
    )

    client.set_model_version_tag(
        name=registered_model_name,
        version=champion_version,
        key="champion_selection",
        value="performance_then_artifact_size",
    )

    MLFLOW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {
        "tracking_uri": tracking_uri,
        "experiment_name": experiment_name,
        "registered_model_name": registered_model_name,
        "registered_model_alias": alias,
        "champion_model_key": champion_key,
        "champion_model_name": champion_info.get("name"),
        "champion_source_artifact": str(champion_path),
        "champion_run_id": champion_run_id,
        "champion_version": champion_version,
        "champion_model_uri": (
            f"models:/{registered_model_name}@{alias}"
        ),
        "feature_columns": FEATURE_COLUMNS,
        "logged_runs": logged_runs,
    }

    summary_path = (
        MLFLOW_DATA_DIR / "registration_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print("MLFLOW_REGISTRATION_COMPLETE")
    print(f"Experiment: {experiment_name}")
    print(f"Champion model key: {champion_key}")
    print(f"Champion run ID: {champion_run_id}")
    print(f"Registered model: {registered_model_name}")
    print(f"Registered version: {champion_version}")
    print(f"Alias: {alias}")
    print(
        "Registry model URI: "
        f"models:/{registered_model_name}@{alias}"
    )
    print(f"Summary: {summary_path}")

    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Track WaferWatch models and register the "
            "deployment champion."
        )
    )

    parser.add_argument(
        "--tracking-uri",
        default=DEFAULT_TRACKING_URI,
        help=(
            "MLflow tracking and registry URI. "
            "Default: MLFLOW_TRACKING_URI or "
            "http://127.0.0.1:5000"
        ),
    )

    parser.add_argument(
        "--experiment-name",
        default=DEFAULT_EXPERIMENT_NAME,
        help="MLflow experiment name.",
    )

    parser.add_argument(
        "--registered-model-name",
        default=DEFAULT_REGISTERED_MODEL_NAME,
        help="MLflow registered model name.",
    )

    parser.add_argument(
        "--alias",
        default=DEFAULT_MODEL_ALIAS,
        help="Alias assigned to the selected model version.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate report, feature table, artifacts, and "
            "champion selection without contacting MLflow."
        ),
    )

    return parser


def main() -> None:
    """Run dry-run validation or actual MLflow registration."""

    parser = build_argument_parser()
    arguments = parser.parse_args()

    report = load_json(MODEL_COMPARISON_REPORT)

    if arguments.dry_run:
        print_dry_run_summary(
            report=report,
            tracking_uri=arguments.tracking_uri,
            experiment_name=arguments.experiment_name,
            registered_model_name=(
                arguments.registered_model_name
            ),
            alias=arguments.alias,
        )
        return

    register_models(
        report=report,
        tracking_uri=arguments.tracking_uri,
        experiment_name=arguments.experiment_name,
        registered_model_name=(
            arguments.registered_model_name
        ),
        alias=arguments.alias,
    )


if __name__ == "__main__":
    main()