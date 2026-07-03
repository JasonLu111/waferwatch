"""
Feature selection utilities for WaferWatch.

This module removes low-quality or redundant features from a feature table.

It supports:
- removing columns with high missingness
- removing constant or near-constant numeric features
- removing highly correlated numeric features
- preserving identifier, timestamp, and label columns

The goal is not to blindly reduce dimensionality. The goal is to make the
feature table cleaner, more stable, and easier to model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import PROCESSED_DATA_DIR, REPORTS_DIR, ensure_directories_exist
from src.utils.logger import get_logger


logger = get_logger(__name__)


def select_features(
    df: pd.DataFrame,
    id_column: str = "lot_id",
    timestamp_column: str = "timestamp",
    label_column: str = "pass_fail_label",
    high_missing_threshold: float = 0.60,
    near_constant_threshold: float = 0.98,
    high_correlation_threshold: float = 0.95,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Select cleaner features from a feature table.

    Parameters
    ----------
    df:
        Input feature table.
    id_column:
        Identifier column to preserve.
    timestamp_column:
        Timestamp column to preserve if available.
    label_column:
        Target label column to preserve.
    high_missing_threshold:
        Drop feature columns with missing ratio greater than this threshold.
    near_constant_threshold:
        Drop columns where the most common value appears more than this ratio.
    high_correlation_threshold:
        Drop one feature from each pair of highly correlated numeric features.

    Returns
    -------
    tuple[pandas.DataFrame, dict]
        Selected feature table and feature selection report.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    logger.info("Starting feature selection.")

    preserved_columns = [
        column
        for column in [id_column, timestamp_column, label_column]
        if column in df.columns
    ]

    candidate_columns = [
        column
        for column in df.columns
        if column not in preserved_columns
    ]

    selected_df = df[preserved_columns + candidate_columns].copy()

    # 1. Remove high-missingness candidate features.
    missing_ratio = selected_df[candidate_columns].isna().mean()

    high_missing_columns = missing_ratio[
        missing_ratio > high_missing_threshold
    ].index.tolist()

    selected_df = selected_df.drop(columns=high_missing_columns)

    candidate_columns = [
        column
        for column in candidate_columns
        if column not in high_missing_columns
    ]

    # 2. Keep numeric candidate features for this baseline ML pipeline.
    # Non-numeric features may be encoded later, but we exclude them for now.
    numeric_candidate_columns = selected_df[candidate_columns].select_dtypes(
        include=["number"]
    ).columns.tolist()

    non_numeric_columns = [
        column
        for column in candidate_columns
        if column not in numeric_candidate_columns
    ]

    selected_df = selected_df[preserved_columns + numeric_candidate_columns].copy()

    # 3. Remove constant and near-constant numeric features.
    constant_columns: list[str] = []
    near_constant_columns: list[str] = []

    for column in numeric_candidate_columns:
        value_counts = selected_df[column].value_counts(dropna=False, normalize=True)

        if len(value_counts) <= 1:
            constant_columns.append(column)
        elif float(value_counts.iloc[0]) >= near_constant_threshold:
            near_constant_columns.append(column)

    low_variance_columns = sorted(set(constant_columns + near_constant_columns))

    selected_df = selected_df.drop(columns=low_variance_columns)

    numeric_candidate_columns = [
        column
        for column in numeric_candidate_columns
        if column not in low_variance_columns
    ]

    # 4. Remove highly correlated numeric features.
    high_correlation_removed_columns: list[str] = []

    if len(numeric_candidate_columns) >= 2:
        corr_matrix = selected_df[numeric_candidate_columns].corr().abs()

        columns_to_remove = set()

        for i, column_i in enumerate(corr_matrix.columns):
            for column_j in corr_matrix.columns[i + 1:]:
                corr_value = corr_matrix.loc[column_i, column_j]

                if pd.notna(corr_value) and corr_value > high_correlation_threshold:
                    columns_to_remove.add(column_j)

        high_correlation_removed_columns = sorted(columns_to_remove)

        selected_df = selected_df.drop(columns=high_correlation_removed_columns)

    final_feature_columns = [
        column
        for column in selected_df.columns
        if column not in preserved_columns
    ]

    report: dict[str, Any] = {
        "input_rows": int(df.shape[0]),
        "input_columns": int(df.shape[1]),
        "output_rows": int(selected_df.shape[0]),
        "output_columns": int(selected_df.shape[1]),
        "preserved_columns": preserved_columns,
        "initial_candidate_columns": candidate_columns,
        "removed_high_missing_columns": high_missing_columns,
        "removed_non_numeric_columns_for_baseline": non_numeric_columns,
        "removed_constant_columns": constant_columns,
        "removed_near_constant_columns": near_constant_columns,
        "removed_high_correlation_columns": high_correlation_removed_columns,
        "final_feature_columns": final_feature_columns,
        "n_final_features": int(len(final_feature_columns)),
        "thresholds": {
            "high_missing_threshold": high_missing_threshold,
            "near_constant_threshold": near_constant_threshold,
            "high_correlation_threshold": high_correlation_threshold,
        },
        "interpretation_note": (
            "This feature selection step is intentionally conservative. "
            "Non-numeric features are excluded for the current baseline model, "
            "but they can be encoded and tested in later experiments."
        ),
    }

    logger.info("Feature selection completed.")

    return selected_df, report


def select_features_from_csv(
    input_file_name: str,
    output_file_name: str,
    input_dir: str | Path = PROCESSED_DATA_DIR,
    output_dir: str | Path = PROCESSED_DATA_DIR,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Load a feature table, select features, and save the selected table.
    """

    ensure_directories_exist()

    input_path = Path(input_dir) / input_file_name
    output_path = Path(output_dir) / output_file_name

    if not input_path.exists():
        raise FileNotFoundError(
            f"Feature table not found: {input_path}. "
            "Please run feature engineering first."
        )

    logger.info("Loading feature table from: %s", input_path)

    df = pd.read_csv(input_path)

    selected_df, report = select_features(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_df.to_csv(output_path, index=False)

    report["input_file"] = str(input_path)
    report["output_file"] = str(output_path)

    logger.info("Saved selected feature table to: %s", output_path)

    return selected_df, report


def save_feature_selection_report(
    report: dict[str, Any],
    output_path: str | Path | None = None,
) -> Path:
    """
    Save feature selection report as JSON.
    """

    ensure_directories_exist()

    if output_path is None:
        output_path = REPORTS_DIR / "feature_selection_report.json"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    logger.info("Saved feature selection report to: %s", output_path)

    return output_path


def _demo() -> None:
    """
    Run feature selection demo using the combined feature table.

    This requires:
    data/processed/demo_spc_combined_feature_table.csv

    If it does not exist, run:
    python -m src.models.compare
    """

    selected_df, report = select_features_from_csv(
        input_file_name="demo_spc_combined_feature_table.csv",
        output_file_name="demo_spc_selected_feature_table.csv",
    )

    print(selected_df.head(15))
    print(json.dumps(report, indent=2, ensure_ascii=False))

    save_feature_selection_report(report)


if __name__ == "__main__":
    _demo()