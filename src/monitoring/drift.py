"""
Drift monitoring utilities for WaferWatch.

This module detects feature distribution drift between a reference period
and a current period.

It supports:
- mean shift
- standard deviation shift
- missing rate shift
- Population Stability Index (PSI)
- feature-level drift flags
- overall drift summary report

This is important because manufacturing data is not static. Tool conditions,
recipe mix, maintenance events, and process variation can all change the
data distribution after deployment.
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


def get_numeric_feature_columns(
    df: pd.DataFrame,
    id_column: str = "lot_id",
    timestamp_column: str = "timestamp",
    label_column: str = "pass_fail_label",
) -> list[str]:
    """
    Get numeric feature columns for drift monitoring.

    Identifier, timestamp, and label columns are excluded.
    """

    excluded_columns = {
        id_column,
        timestamp_column,
        label_column,
    }

    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()

    feature_columns = [
        column
        for column in numeric_columns
        if column not in excluded_columns
    ]

    if not feature_columns:
        raise ValueError("No numeric feature columns found for drift monitoring.")

    logger.info("Found %s numeric feature columns for drift monitoring.", len(feature_columns))

    return feature_columns


def calculate_population_stability_index(
    reference_values: pd.Series,
    current_values: pd.Series,
    n_bins: int = 10,
) -> float:
    """
    Calculate Population Stability Index (PSI).

    PSI is commonly used to compare whether a feature distribution changed
    between a reference period and a current period.

    Rough practical interpretation:
    - PSI < 0.10: small change
    - 0.10 <= PSI < 0.25: moderate change
    - PSI >= 0.25: large change
    """

    reference_values = reference_values.dropna()
    current_values = current_values.dropna()

    if reference_values.empty or current_values.empty:
        return 0.0

    if reference_values.nunique() <= 1:
        return 0.0

    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(reference_values, quantiles)
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) < 3:
        return 0.0

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    reference_counts = pd.cut(
        reference_values,
        bins=bin_edges,
        include_lowest=True,
    ).value_counts(sort=False)

    current_counts = pd.cut(
        current_values,
        bins=bin_edges,
        include_lowest=True,
    ).value_counts(sort=False)

    reference_pct = reference_counts / reference_counts.sum()
    current_pct = current_counts / current_counts.sum()

    epsilon = 1e-6

    reference_pct = reference_pct.replace(0, epsilon)
    current_pct = current_pct.replace(0, epsilon)

    psi_values = (current_pct - reference_pct) * np.log(current_pct / reference_pct)

    return float(psi_values.sum())


def evaluate_feature_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_columns: list[str],
    psi_threshold: float = 0.25,
    mean_shift_threshold: float = 1.0,
    missing_rate_shift_threshold: float = 0.10,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Evaluate drift for each feature.

    Parameters
    ----------
    reference_df:
        Reference period data.
    current_df:
        Current period data.
    feature_columns:
        Numeric feature columns to monitor.
    psi_threshold:
        Feature is flagged if PSI is greater than or equal to this threshold.
    mean_shift_threshold:
        Feature is flagged if absolute standardized mean shift is greater than this.
    missing_rate_shift_threshold:
        Feature is flagged if missing rate changes more than this.

    Returns
    -------
    tuple[pandas.DataFrame, dict]
        Feature-level drift table and summary report.
    """

    rows: list[dict[str, Any]] = []

    for column in feature_columns:
        reference_mean = float(reference_df[column].mean())
        current_mean = float(current_df[column].mean())

        reference_std = float(reference_df[column].std(ddof=0))
        current_std = float(current_df[column].std(ddof=0))

        reference_missing_rate = float(reference_df[column].isna().mean())
        current_missing_rate = float(current_df[column].isna().mean())

        if np.isnan(reference_std) or reference_std == 0:
            standardized_mean_shift = 0.0
        else:
            standardized_mean_shift = float(
                abs(current_mean - reference_mean) / reference_std
            )

        std_ratio = float(current_std / reference_std) if reference_std > 0 else 1.0

        missing_rate_shift = float(
            abs(current_missing_rate - reference_missing_rate)
        )

        psi = calculate_population_stability_index(
            reference_values=reference_df[column],
            current_values=current_df[column],
        )

        drift_detected = (
            psi >= psi_threshold
            or standardized_mean_shift >= mean_shift_threshold
            or missing_rate_shift >= missing_rate_shift_threshold
        )

        rows.append(
            {
                "feature": column,
                "reference_mean": reference_mean,
                "current_mean": current_mean,
                "standardized_mean_shift": standardized_mean_shift,
                "reference_std": reference_std,
                "current_std": current_std,
                "std_ratio": std_ratio,
                "reference_missing_rate": reference_missing_rate,
                "current_missing_rate": current_missing_rate,
                "missing_rate_shift": missing_rate_shift,
                "psi": psi,
                "drift_detected": bool(drift_detected),
            }
        )

    drift_df = pd.DataFrame(rows).sort_values(
        by=["drift_detected", "psi", "standardized_mean_shift"],
        ascending=[False, False, False],
    )

    drifted_features = drift_df.loc[
        drift_df["drift_detected"],
        "feature",
    ].tolist()

    report: dict[str, Any] = {
        "reference_rows": int(reference_df.shape[0]),
        "current_rows": int(current_df.shape[0]),
        "n_features_monitored": int(len(feature_columns)),
        "n_features_with_drift": int(len(drifted_features)),
        "drifted_features": drifted_features,
        "thresholds": {
            "psi_threshold": psi_threshold,
            "mean_shift_threshold": mean_shift_threshold,
            "missing_rate_shift_threshold": missing_rate_shift_threshold,
        },
        "overall_drift_detected": bool(len(drifted_features) > 0),
        "interpretation_note": (
            "A drift alert means the current feature distribution differs from "
            "the reference period. It does not automatically mean the model is wrong, "
            "but it indicates that model performance should be checked when labels "
            "become available."
        ),
    }

    return drift_df, report


def create_drifted_current_data(
    df: pd.DataFrame,
    feature_columns: list[str],
    drift_strength: float = 1.5,
    drift_fraction: float = 0.40,
) -> pd.DataFrame:
    """
    Create a synthetic current-period dataset with injected drift.

    This is only for demo testing. In a real setting, current data would come
    from newly processed lots.
    """

    current_df = df.copy()

    n_drift_features = max(1, int(len(feature_columns) * drift_fraction))
    drift_features = feature_columns[:n_drift_features]

    for column in drift_features:
        feature_std = current_df[column].std(ddof=0)

        if pd.isna(feature_std) or feature_std == 0:
            continue

        current_df[column] = current_df[column] + drift_strength * feature_std

    logger.info("Injected synthetic drift into features: %s", drift_features)

    return current_df


def run_drift_monitoring_demo(
    input_file_name: str = "demo_spc_selected_feature_table.csv",
    drift_table_file_name: str = "demo_drift_table.csv",
    report_file_name: str = "drift_monitoring_report.json",
) -> dict[str, Any]:
    """
    Run a drift monitoring demo using the selected SPC feature table.
    """

    ensure_directories_exist()

    input_path = PROCESSED_DATA_DIR / input_file_name

    if not input_path.exists():
        raise FileNotFoundError(
            f"Selected feature table not found: {input_path}. "
            "Please run src.features.feature_selection first."
        )

    logger.info("Loading selected feature table from: %s", input_path)

    df = pd.read_csv(input_path)

    feature_columns = get_numeric_feature_columns(df)

    midpoint = len(df) // 2

    reference_df = df.iloc[:midpoint].copy()
    current_df = df.iloc[midpoint:].copy()

    current_df = create_drifted_current_data(
        df=current_df,
        feature_columns=feature_columns,
    )

    drift_df, report = evaluate_feature_drift(
        reference_df=reference_df,
        current_df=current_df,
        feature_columns=feature_columns,
    )

    drift_table_path = PROCESSED_DATA_DIR / drift_table_file_name
    drift_df.to_csv(drift_table_path, index=False)

    logger.info("Saved drift table to: %s", drift_table_path)

    report["input_file"] = str(input_path)
    report["drift_table_file"] = str(drift_table_path)

    report_path = REPORTS_DIR / report_file_name

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    logger.info("Saved drift monitoring report to: %s", report_path)

    print("Drift monitoring summary")
    print("------------------------")
    print(drift_df)
    print()
    print(json.dumps(report, indent=2, ensure_ascii=False))

    return report


def _demo() -> None:
    """
    Run drift monitoring demo.
    """

    run_drift_monitoring_demo()


if __name__ == "__main__":
    _demo()