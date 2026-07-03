"""
Feature engineering utilities for WaferWatch.

This module converts cleaned sensor data into model-ready feature tables.

It supports:
- sensor aggregate features
- SPC features
- combined ML feature tables

The combined feature table will later be used for model comparison,
cost-sensitive thresholding, explainability, and monitoring.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.features.spc_features import build_spc_feature_table
from src.utils.config import PROCESSED_DATA_DIR, REPORTS_DIR, ensure_directories_exist
from src.utils.logger import get_logger


logger = get_logger(__name__)


def find_sensor_columns(df: pd.DataFrame, prefix: str = "sensor_") -> list[str]:
    """
    Find sensor columns by prefix.

    Parameters
    ----------
    df:
        Input DataFrame.
    prefix:
        Prefix used by sensor columns. Default is 'sensor_'.

    Returns
    -------
    list[str]
        List of sensor column names.
    """

    sensor_columns = [column for column in df.columns if column.startswith(prefix)]

    logger.info("Found %s sensor columns.", len(sensor_columns))

    return sensor_columns


def build_sensor_aggregate_features(
    df: pd.DataFrame,
    sensor_columns: list[str],
) -> pd.DataFrame:
    """
    Build aggregate features from sensor columns.

    Parameters
    ----------
    df:
        Input DataFrame.
    sensor_columns:
        List of sensor columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing aggregate sensor features.
    """

    if not sensor_columns:
        raise ValueError("No sensor columns were provided.")

    sensor_df = df[sensor_columns]

    feature_df = pd.DataFrame(index=df.index)

    feature_df["sensor_mean"] = sensor_df.mean(axis=1)
    feature_df["sensor_std"] = sensor_df.std(axis=1)
    feature_df["sensor_min"] = sensor_df.min(axis=1)
    feature_df["sensor_max"] = sensor_df.max(axis=1)
    feature_df["sensor_missing_count"] = sensor_df.isna().sum(axis=1)
    feature_df["sensor_missing_ratio"] = sensor_df.isna().mean(axis=1)

    logger.info("Built sensor aggregate features.")

    return feature_df


def build_feature_table(
    df: pd.DataFrame,
    id_column: str = "lot_id",
    label_column: str = "pass_fail_label",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Build a model-ready feature table using sensor aggregate features only.

    Parameters
    ----------
    df:
        Cleaned input DataFrame.
    id_column:
        Lot identifier column.
    label_column:
        Target label column.

    Returns
    -------
    tuple[pandas.DataFrame, dict]
        Feature table and feature engineering report.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    logger.info("Starting basic feature engineering.")

    sensor_columns = find_sensor_columns(df)

    feature_df = build_sensor_aggregate_features(
        df=df,
        sensor_columns=sensor_columns,
    )

    if id_column in df.columns:
        feature_df.insert(0, id_column, df[id_column])

    if label_column in df.columns:
        feature_df[label_column] = df[label_column]

    report: dict[str, Any] = {
        "feature_table_type": "sensor_aggregate_only",
        "input_rows": int(df.shape[0]),
        "input_columns": int(df.shape[1]),
        "sensor_columns_found": len(sensor_columns),
        "sensor_columns": sensor_columns,
        "output_rows": int(feature_df.shape[0]),
        "output_columns": int(feature_df.shape[1]),
        "created_features": [
            "sensor_mean",
            "sensor_std",
            "sensor_min",
            "sensor_max",
            "sensor_missing_count",
            "sensor_missing_ratio",
        ],
        "id_column_used": id_column if id_column in df.columns else None,
        "label_column_used": label_column if label_column in df.columns else None,
    }

    logger.info("Basic feature engineering completed.")

    return feature_df, report


def build_combined_feature_table(
    df: pd.DataFrame,
    id_column: str = "lot_id",
    timestamp_column: str = "timestamp",
    label_column: str = "pass_fail_label",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Build a combined feature table with sensor aggregate features and SPC features.

    The output keeps:
    - lot_id
    - optional timestamp
    - sensor aggregate features
    - SPC features
    - label column if available
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    logger.info("Starting combined feature engineering.")

    sensor_columns = find_sensor_columns(df)

    aggregate_df = build_sensor_aggregate_features(
        df=df,
        sensor_columns=sensor_columns,
    )

    spc_df, spc_report = build_spc_feature_table(
        df=df,
        id_column=id_column,
        timestamp_column=timestamp_column,
        label_column=label_column,
    )

    combined_df = pd.DataFrame(index=df.index)

    if id_column in df.columns:
        combined_df[id_column] = df[id_column]

    if timestamp_column in df.columns:
        combined_df[timestamp_column] = df[timestamp_column]

    combined_df = pd.concat(
        [
            combined_df,
            aggregate_df,
            spc_df.drop(
                columns=[
                    column
                    for column in [id_column, timestamp_column, label_column]
                    if column in spc_df.columns
                ],
                errors="ignore",
            ),
        ],
        axis=1,
    )

    if label_column in df.columns:
        combined_df[label_column] = df[label_column]

    created_aggregate_features = [
        "sensor_mean",
        "sensor_std",
        "sensor_min",
        "sensor_max",
        "sensor_missing_count",
        "sensor_missing_ratio",
    ]

    created_spc_features = [
        "spc_violation_count",
        "spc_violation_ratio",
        "spc_max_abs_zscore",
        "spc_any_violation",
        "spc_top_violating_sensor",
    ]

    report: dict[str, Any] = {
        "feature_table_type": "combined_sensor_aggregate_plus_spc",
        "input_rows": int(df.shape[0]),
        "input_columns": int(df.shape[1]),
        "sensor_columns_found": len(sensor_columns),
        "sensor_columns": sensor_columns,
        "output_rows": int(combined_df.shape[0]),
        "output_columns": int(combined_df.shape[1]),
        "created_aggregate_features": created_aggregate_features,
        "created_spc_features": created_spc_features,
        "id_column_used": id_column if id_column in df.columns else None,
        "timestamp_column_used": timestamp_column if timestamp_column in df.columns else None,
        "label_column_used": label_column if label_column in df.columns else None,
        "spc_summary": {
            "reference_rows_used_for_limits": spc_report["reference_rows_used_for_limits"],
            "lots_with_any_spc_violation": spc_report["lots_with_any_spc_violation"],
            "spc_violation_rate": spc_report["spc_violation_rate"],
            "sigma_width": spc_report["sigma_width"],
        },
    }

    logger.info("Combined feature engineering completed.")

    return combined_df, report


def build_features_from_csv(
    input_file_name: str,
    output_file_name: str,
    input_dir: str | Path = PROCESSED_DATA_DIR,
    output_dir: str | Path = PROCESSED_DATA_DIR,
    id_column: str = "lot_id",
    label_column: str = "pass_fail_label",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Load a processed CSV, build basic sensor aggregate features, and save the table.
    """

    ensure_directories_exist()

    input_path = Path(input_dir) / input_file_name
    output_path = Path(output_dir) / output_file_name

    if not input_path.exists():
        raise FileNotFoundError(
            f"Processed CSV not found: {input_path}. "
            "Please run the cleaning step first."
        )

    logger.info("Loading processed CSV from: %s", input_path)

    df = pd.read_csv(input_path)

    feature_df, report = build_feature_table(
        df=df,
        id_column=id_column,
        label_column=label_column,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(output_path, index=False)

    logger.info("Saved basic feature table to: %s", output_path)

    return feature_df, report


def build_combined_features_from_csv(
    input_file_name: str,
    output_file_name: str,
    input_dir: str | Path = PROCESSED_DATA_DIR,
    output_dir: str | Path = PROCESSED_DATA_DIR,
    id_column: str = "lot_id",
    timestamp_column: str = "timestamp",
    label_column: str = "pass_fail_label",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Load a processed sensor CSV, build combined features, and save the table.
    """

    ensure_directories_exist()

    input_path = Path(input_dir) / input_file_name
    output_path = Path(output_dir) / output_file_name

    if not input_path.exists():
        raise FileNotFoundError(
            f"Processed sensor CSV not found: {input_path}. "
            "Please run the SPC demo or cleaning step first."
        )

    logger.info("Loading processed sensor CSV from: %s", input_path)

    df = pd.read_csv(input_path)

    combined_df, report = build_combined_feature_table(
        df=df,
        id_column=id_column,
        timestamp_column=timestamp_column,
        label_column=label_column,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)

    logger.info("Saved combined feature table to: %s", output_path)

    return combined_df, report


def save_feature_report(
    report: dict[str, Any],
    output_path: str | Path | None = None,
) -> Path:
    """
    Save the feature engineering report as a JSON file.
    """

    ensure_directories_exist()

    if output_path is None:
        output_path = REPORTS_DIR / "feature_engineering_report.json"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    logger.info("Saved feature engineering report to: %s", output_path)

    return output_path


def _demo() -> None:
    """
    Run a combined feature engineering demo using the SPC demo sensor dataset.

    This requires:
    data/processed/demo_spc_sensor_data_processed.csv

    If it does not exist, run:
    python -m src.features.spc_features
    """

    combined_df, report = build_combined_features_from_csv(
        input_file_name="demo_spc_sensor_data_processed.csv",
        output_file_name="demo_combined_feature_table.csv",
    )

    print(combined_df.head(15))
    print(json.dumps(report, indent=2, ensure_ascii=False))

    save_feature_report(
        report,
        output_path=REPORTS_DIR / "combined_feature_engineering_report.json",
    )


if __name__ == "__main__":
    _demo()