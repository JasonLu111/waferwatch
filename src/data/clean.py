"""
Data cleaning utilities for WaferWatch.

This module provides reusable functions for basic manufacturing data cleaning:
- remove duplicated rows
- remove columns with too many missing values
- impute numeric missing values with median
- save cleaned data to data/processed/
- save a cleaning report to reports/

At this stage, we keep the logic simple and transparent.
More advanced cleaning strategies will be added later.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import (
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    ensure_directories_exist,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)


def clean_dataframe(
    df: pd.DataFrame,
    label_column: str | None = None,
    high_missing_threshold: float = 0.60,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Clean a pandas DataFrame and return the cleaned DataFrame plus a report.

    Parameters
    ----------
    df:
        Input pandas DataFrame.
    label_column:
        Optional target label column. This column will not be imputed.
    high_missing_threshold:
        Columns with missing ratio greater than this threshold will be removed.

    Returns
    -------
    tuple[pandas.DataFrame, dict]
        Cleaned DataFrame and cleaning report.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    logger.info("Starting data cleaning.")

    original_rows = int(df.shape[0])
    original_columns = int(df.shape[1])

    cleaned_df = df.copy()

    # 1. Remove duplicated rows
    duplicate_rows_before = int(cleaned_df.duplicated().sum())
    cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)

    # 2. Remove columns with excessive missing values
    missing_ratio = cleaned_df.isna().mean()
    columns_to_drop = missing_ratio[missing_ratio > high_missing_threshold].index.tolist()

    if label_column in columns_to_drop:
        columns_to_drop.remove(label_column)

    cleaned_df = cleaned_df.drop(columns=columns_to_drop)

    # 3. Impute numeric missing values with median
    numeric_columns = cleaned_df.select_dtypes(include=["number"]).columns.tolist()

    if label_column in numeric_columns:
        numeric_columns.remove(label_column)

    imputed_columns: list[str] = []

    for column in numeric_columns:
        if cleaned_df[column].isna().sum() > 0:
            median_value = cleaned_df[column].median()
            cleaned_df[column] = cleaned_df[column].fillna(median_value)
            imputed_columns.append(column)

    # 4. Build cleaning report
    report: dict[str, Any] = {
        "original_rows": original_rows,
        "original_columns": original_columns,
        "cleaned_rows": int(cleaned_df.shape[0]),
        "cleaned_columns": int(cleaned_df.shape[1]),
        "removed_duplicate_rows": duplicate_rows_before,
        "dropped_high_missing_columns": columns_to_drop,
        "median_imputed_numeric_columns": imputed_columns,
        "remaining_missing_values": int(cleaned_df.isna().sum().sum()),
        "high_missing_threshold": high_missing_threshold,
    }

    logger.info("Data cleaning completed.")

    return cleaned_df, report


def clean_csv(
    input_file_name: str,
    output_file_name: str,
    label_column: str | None = None,
    input_dir: str | Path = INTERIM_DATA_DIR,
    output_dir: str | Path = PROCESSED_DATA_DIR,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Clean a CSV file from data/interim/ and save the cleaned version to data/processed/.

    Parameters
    ----------
    input_file_name:
        Input CSV file name under data/interim/.
    output_file_name:
        Output CSV file name under data/processed/.
    label_column:
        Optional target label column.
    input_dir:
        Input directory. Default is data/interim/.
    output_dir:
        Output directory. Default is data/processed/.

    Returns
    -------
    tuple[pandas.DataFrame, dict]
        Cleaned DataFrame and cleaning report.
    """

    ensure_directories_exist()

    input_path = Path(input_dir) / input_file_name
    output_path = Path(output_dir) / output_file_name

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {input_path}. "
            "Please run the ingestion step first."
        )

    logger.info("Loading interim CSV from: %s", input_path)

    df = pd.read_csv(input_path)

    cleaned_df, report = clean_dataframe(
        df=df,
        label_column=label_column,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)

    logger.info("Saved cleaned CSV to: %s", output_path)

    return cleaned_df, report


def save_cleaning_report(
    report: dict[str, Any],
    output_path: str | Path | None = None,
) -> Path:
    """
    Save the cleaning report as a JSON file.

    Parameters
    ----------
    report:
        Cleaning report dictionary.
    output_path:
        Optional output path. If not provided, the report is saved to
        reports/cleaning_report.json.

    Returns
    -------
    pathlib.Path
        Path to the saved report.
    """

    ensure_directories_exist()

    if output_path is None:
        output_path = REPORTS_DIR / "cleaning_report.json"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    logger.info("Saved cleaning report to: %s", output_path)

    return output_path


def _demo() -> None:
    """
    Run a small cleaning demo using the interim demo dataset.
    """

    cleaned_df, report = clean_csv(
        input_file_name="demo_sensor_data_interim.csv",
        output_file_name="demo_sensor_data_processed.csv",
        label_column="pass_fail_label",
    )

    print(cleaned_df)
    print(json.dumps(report, indent=2, ensure_ascii=False))

    save_cleaning_report(report)


if __name__ == "__main__":
    _demo()