"""
SPC feature utilities for WaferWatch.

SPC means Statistical Process Control.

This module creates manufacturing-oriented features such as:
- sensor-level 3-sigma control limit violations
- total SPC violation count per lot
- SPC violation ratio
- maximum absolute z-score
- top violating sensor

These features can later be used by:
- ML models
- anomaly detection
- dashboard explanations
- RAG evidence retrieval
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import PROCESSED_DATA_DIR, REPORTS_DIR, ensure_directories_exist
from src.utils.logger import get_logger


logger = get_logger(__name__)


def find_sensor_columns(df: pd.DataFrame, prefix: str = "sensor_") -> list[str]:
    """
    Find sensor columns by prefix.
    """

    sensor_columns = [column for column in df.columns if column.startswith(prefix)]

    if not sensor_columns:
        raise ValueError(f"No sensor columns found with prefix: {prefix}")

    logger.info("Found %s sensor columns for SPC.", len(sensor_columns))

    return sensor_columns


def calculate_control_limits(
    df: pd.DataFrame,
    sensor_columns: list[str],
    reference_mask: pd.Series | None = None,
    sigma_width: float = 3.0,
) -> dict[str, dict[str, float]]:
    """
    Calculate mean, standard deviation, and control limits for each sensor.

    Parameters
    ----------
    df:
        Input DataFrame.
    sensor_columns:
        Sensor columns used for SPC.
    reference_mask:
        Optional boolean mask. If provided, control limits are calculated
        only from reference rows, such as normal lots.
    sigma_width:
        Number of standard deviations used for control limits.

    Returns
    -------
    dict
        Control limits for each sensor.
    """

    if reference_mask is not None:
        reference_df = df.loc[reference_mask, sensor_columns]
    else:
        reference_df = df[sensor_columns]

    if reference_df.empty:
        raise ValueError("Reference data for control limit calculation is empty.")

    limits: dict[str, dict[str, float]] = {}

    for column in sensor_columns:
        mean_value = float(reference_df[column].mean())
        std_value = float(reference_df[column].std(ddof=0))

        if np.isnan(std_value):
            std_value = 0.0

        lower_limit = mean_value - sigma_width * std_value
        upper_limit = mean_value + sigma_width * std_value

        limits[column] = {
            "mean": mean_value,
            "std": std_value,
            "lower_control_limit": float(lower_limit),
            "upper_control_limit": float(upper_limit),
        }

    logger.info("Calculated SPC control limits.")

    return limits


def build_spc_feature_table(
    df: pd.DataFrame,
    id_column: str = "lot_id",
    timestamp_column: str = "timestamp",
    label_column: str = "pass_fail_label",
    sigma_width: float = 3.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Build SPC features from cleaned sensor data.

    Returns
    -------
    tuple[pandas.DataFrame, dict]
        SPC feature table and report.
    """

    logger.info("Starting SPC feature engineering.")

    sensor_columns = find_sensor_columns(df)

    if label_column in df.columns:
        reference_mask = df[label_column] == 0
        reference_rows = int(reference_mask.sum())
    else:
        reference_mask = None
        reference_rows = int(len(df))

    limits = calculate_control_limits(
        df=df,
        sensor_columns=sensor_columns,
        reference_mask=reference_mask,
        sigma_width=sigma_width,
    )

    mean_series = pd.Series(
        {column: limits[column]["mean"] for column in sensor_columns}
    )

    std_series = pd.Series(
        {
            column: max(limits[column]["std"], 1e-12)
            for column in sensor_columns
        }
    )

    lower_series = pd.Series(
        {
            column: limits[column]["lower_control_limit"]
            for column in sensor_columns
        }
    )

    upper_series = pd.Series(
        {
            column: limits[column]["upper_control_limit"]
            for column in sensor_columns
        }
    )

    sensor_df = df[sensor_columns]

    z_scores = (sensor_df - mean_series) / std_series
    abs_z_scores = z_scores.abs().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    violations = (sensor_df < lower_series) | (sensor_df > upper_series)

    spc_df = pd.DataFrame(index=df.index)

    if id_column in df.columns:
        spc_df[id_column] = df[id_column]

    if timestamp_column in df.columns:
        spc_df[timestamp_column] = df[timestamp_column]

    spc_df["spc_violation_count"] = violations.sum(axis=1).astype(int)
    spc_df["spc_violation_ratio"] = violations.mean(axis=1)
    spc_df["spc_max_abs_zscore"] = abs_z_scores.max(axis=1)
    spc_df["spc_any_violation"] = (spc_df["spc_violation_count"] > 0).astype(int)

    top_sensor = abs_z_scores.idxmax(axis=1)
    spc_df["spc_top_violating_sensor"] = np.where(
        spc_df["spc_max_abs_zscore"] > sigma_width,
        top_sensor,
        "none",
    )

    if label_column in df.columns:
        spc_df[label_column] = df[label_column]

    created_features = [
        "spc_violation_count",
        "spc_violation_ratio",
        "spc_max_abs_zscore",
        "spc_any_violation",
        "spc_top_violating_sensor",
    ]

    report: dict[str, Any] = {
        "input_rows": int(df.shape[0]),
        "input_columns": int(df.shape[1]),
        "sensor_columns_found": int(len(sensor_columns)),
        "reference_rows_used_for_limits": reference_rows,
        "sigma_width": float(sigma_width),
        "output_rows": int(spc_df.shape[0]),
        "output_columns": int(spc_df.shape[1]),
        "created_features": created_features,
        "lots_with_any_spc_violation": int(spc_df["spc_any_violation"].sum()),
        "spc_violation_rate": float(spc_df["spc_any_violation"].mean()),
        "control_limits_preview": {
            column: limits[column]
            for column in sensor_columns[:10]
        },
    }

    logger.info("SPC feature engineering completed.")

    return spc_df, report


def build_spc_features_from_csv(
    input_file_name: str,
    output_file_name: str,
    input_dir: str | Path = PROCESSED_DATA_DIR,
    output_dir: str | Path = PROCESSED_DATA_DIR,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Load a processed sensor CSV, build SPC features, and save the output.
    """

    ensure_directories_exist()

    input_path = Path(input_dir) / input_file_name
    output_path = Path(output_dir) / output_file_name

    if not input_path.exists():
        raise FileNotFoundError(
            f"Processed sensor CSV not found: {input_path}"
        )

    logger.info("Loading processed sensor CSV from: %s", input_path)

    df = pd.read_csv(input_path)

    spc_df, report = build_spc_feature_table(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    spc_df.to_csv(output_path, index=False)

    report["output_file"] = str(output_path)

    logger.info("Saved SPC feature table to: %s", output_path)

    return spc_df, report


def save_spc_report(
    report: dict[str, Any],
    output_path: str | Path | None = None,
) -> Path:
    """
    Save SPC feature engineering report as JSON.
    """

    ensure_directories_exist()

    if output_path is None:
        output_path = REPORTS_DIR / "spc_feature_report.json"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    logger.info("Saved SPC feature report to: %s", output_path)

    return output_path


def create_demo_spc_sensor_dataset(
    output_file_name: str = "demo_spc_sensor_data_processed.csv",
    n_samples: int = 80,
    n_sensors: int = 5,
) -> Path:
    """
    Create a demo processed sensor dataset for SPC testing.

    Some lots are injected with abnormal sensor shifts so SPC violations
    can be detected.
    """

    ensure_directories_exist()

    rng = np.random.default_rng(seed=42)

    timestamps = pd.date_range(
        start="2026-01-01 08:00:00",
        periods=n_samples,
        freq="h",
    )

    data: dict[str, Any] = {
        "lot_id": [f"LOT_SPC_{i:03d}" for i in range(1, n_samples + 1)],
        "timestamp": timestamps.astype(str),
    }

    for sensor_idx in range(1, n_sensors + 1):
        data[f"sensor_{sensor_idx:03d}"] = rng.normal(
            loc=10.0 + sensor_idx,
            scale=0.5,
            size=n_samples,
        )

    labels = np.zeros(n_samples, dtype=int)

    anomaly_indices = rng.choice(
        np.arange(20, n_samples),
        size=int(n_samples * 0.20),
        replace=False,
    )

    sensor_names = [f"sensor_{i:03d}" for i in range(1, n_sensors + 1)]

    for row_idx in anomaly_indices:
        selected_sensor = rng.choice(sensor_names)
        shift_direction = rng.choice([-1, 1])
        data[selected_sensor][row_idx] += shift_direction * rng.normal(
            loc=3.0,
            scale=0.4,
        )
        labels[row_idx] = 1

    data["pass_fail_label"] = labels

    demo_df = pd.DataFrame(data)

    output_path = PROCESSED_DATA_DIR / output_file_name
    demo_df.to_csv(output_path, index=False)

    logger.info("Created demo SPC sensor dataset at: %s", output_path)

    return output_path


def _demo() -> None:
    """
    Run SPC feature demo.
    """

    create_demo_spc_sensor_dataset()

    spc_df, report = build_spc_features_from_csv(
        input_file_name="demo_spc_sensor_data_processed.csv",
        output_file_name="demo_spc_features.csv",
    )

    print(spc_df.head(15))
    print(json.dumps(report, indent=2, ensure_ascii=False))

    save_spc_report(report)


if __name__ == "__main__":
    _demo()