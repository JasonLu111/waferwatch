"""
Data validation utilities for WaferWatch.

This module checks basic data quality issues such as:
- number of rows and columns
- duplicated rows
- missing values
- data types
- expected columns
- label distribution

It is designed to be reused by data ingestion, preprocessing,
model training, and monitoring pipelines.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import REPORTS_DIR, ensure_directories_exist
from src.utils.logger import get_logger


logger = get_logger(__name__)


def validate_dataframe(
    df: pd.DataFrame,
    label_column: str | None = None,
    expected_columns: list[str] | None = None,
) -> dict[str, Any]:
    """
    Validate a pandas DataFrame and return a structured report.

    Parameters
    ----------
    df:
        Input pandas DataFrame.
    label_column:
        Optional name of the target label column.
    expected_columns:
        Optional list of columns expected to exist in the DataFrame.

    Returns
    -------
    dict
        A dictionary containing data validation results.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    logger.info("Starting DataFrame validation.")

    report: dict[str, Any] = {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "total_missing_values": int(df.isna().sum().sum()),
        "missing_values_by_column": {
            column: int(count)
            for column, count in df.isna().sum().items()
            if int(count) > 0
        },
        "dtypes": {
            column: str(dtype)
            for column, dtype in df.dtypes.items()
        },
    }

    if expected_columns is not None:
        expected_set = set(expected_columns)
        actual_set = set(df.columns)

        report["missing_expected_columns"] = sorted(expected_set - actual_set)
        report["unexpected_columns"] = sorted(actual_set - expected_set)

    if label_column is not None:
        if label_column in df.columns:
            label_counts = df[label_column].value_counts(dropna=False).to_dict()
            report["label_distribution"] = {
                str(label): int(count)
                for label, count in label_counts.items()
            }
        else:
            report["label_distribution_error"] = (
                f"Label column '{label_column}' was not found."
            )

    logger.info("DataFrame validation completed.")

    return report


def validate_csv(
    csv_path: str | Path,
    label_column: str | None = None,
    expected_columns: list[str] | None = None,
) -> dict[str, Any]:
    """
    Load a CSV file and validate it.

    Parameters
    ----------
    csv_path:
        Path to the CSV file.
    label_column:
        Optional name of the target label column.
    expected_columns:
        Optional list of columns expected to exist in the CSV file.

    Returns
    -------
    dict
        A dictionary containing data validation results.
    """

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    logger.info("Loading CSV file: %s", csv_path)

    df = pd.read_csv(csv_path)

    return validate_dataframe(
        df=df,
        label_column=label_column,
        expected_columns=expected_columns,
    )


def save_validation_report(
    report: dict[str, Any],
    output_path: str | Path | None = None,
) -> Path:
    """
    Save a validation report as a JSON file.

    Parameters
    ----------
    report:
        Validation report dictionary.
    output_path:
        Optional output path. If not provided, the report is saved to
        reports/data_validation_report.json.

    Returns
    -------
    Path
        Path to the saved JSON report.
    """

    ensure_directories_exist()

    if output_path is None:
        output_path = REPORTS_DIR / "data_validation_report.json"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    logger.info("Saved validation report to: %s", output_path)

    return output_path


def _demo() -> None:
    """
    Run a small demonstration using synthetic data.

    This lets us test validate.py before downloading the real dataset.
    """

    demo_df = pd.DataFrame(
        {
            "lot_id": [
                "LOT_001",
                "LOT_002",
                "LOT_003",
                "LOT_003",
                "LOT_004",
            ],
            "sensor_001": [1.20, 1.35, None, None, 1.10],
            "sensor_002": [5.10, 5.30, 5.80, 5.80, None],
            "pass_fail_label": [0, 0, 1, 1, 0],
        }
    )

    expected_columns = [
        "lot_id",
        "sensor_001",
        "sensor_002",
        "pass_fail_label",
    ]

    report = validate_dataframe(
        df=demo_df,
        label_column="pass_fail_label",
        expected_columns=expected_columns,
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))

    save_validation_report(report)


if __name__ == "__main__":
    _demo()