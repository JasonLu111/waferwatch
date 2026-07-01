"""
Feature engineering utilities for WaferWatch.

This module converts cleaned sensor data into a model-ready feature table.

At this early stage, we build simple and interpretable features:
- sensor mean
- sensor standard deviation
- sensor minimum
- sensor maximum
- sensor missing count
- sensor missing ratio

These features will later support baseline modeling, anomaly detection,
SPC features, and lot-level explanations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

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
    Build a model-ready feature table.

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

    logger.info("Starting feature engineering.")

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

    logger.info("Feature engineering completed.")

    return feature_df, report


def build_features_from_csv(
    input_file_name: str,
    output_file_name: str,
    input_dir: str | Path = PROCESSED_DATA_DIR,
    output_dir: str | Path = PROCESSED_DATA_DIR,
    id_column: str = "lot_id",
    label_column: str = "pass_fail_label",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Load a processed CSV, build features, and save the feature table.

    Parameters
    ----------
    input_file_name:
        Input CSV file under data/processed/.
    output_file_name:
        Output feature CSV file under data/processed/.
    input_dir:
        Input directory.
    output_dir:
        Output directory.
    id_column:
        Lot identifier column.
    label_column:
        Target label column.

    Returns
    -------
    tuple[pandas.DataFrame, dict]
        Feature table and report.
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

    logger.info("Saved feature table to: %s", output_path)

    return feature_df, report


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
    Run a small feature engineering demo using the processed demo dataset.
    """

    feature_df, report = build_features_from_csv(
        input_file_name="demo_sensor_data_processed.csv",
        output_file_name="demo_feature_table.csv",
    )

    print(feature_df)
    print(json.dumps(report, indent=2, ensure_ascii=False))

    save_feature_report(report)


if __name__ == "__main__":
    _demo()