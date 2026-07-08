from pathlib import Path

import pandas as pd

from src.data.ingest_secom import load_labels, load_sensor_data, check_required_files


REPORTS_DIR = Path("reports")
DATA_QUALITY_REPORT = REPORTS_DIR / "secom_data_quality_report.md"
CLASS_IMBALANCE_REPORT = REPORTS_DIR / "secom_class_imbalance_report.md"


def build_missing_value_report(sensor_df: pd.DataFrame, labels_df: pd.DataFrame) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    missing_counts = sensor_df.isna().sum()
    missing_rates = missing_counts / len(sensor_df)

    missing_summary = (
        pd.DataFrame(
            {
                "feature": sensor_df.columns,
                "missing_count": missing_counts.values,
                "missing_rate": missing_rates.values,
            }
        )
        .sort_values("missing_rate", ascending=False)
        .reset_index(drop=True)
    )

    total_missing_values = int(missing_counts.sum())
    features_with_missing = int((missing_counts > 0).sum())
    all_missing_features = int((missing_rates == 1.0).sum())
    high_missing_features = int((missing_rates >= 0.50).sum())
    moderate_missing_features = int(((missing_rates >= 0.05) & (missing_rates < 0.50)).sum())
    timestamp_missing = int(labels_df["timestamp"].isna().sum())

    top_missing = missing_summary.head(20)

    lines = []
    lines.append("# SECOM Data Quality Report")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Number of rows: {sensor_df.shape[0]}")
    lines.append(f"- Number of sensor features: {sensor_df.shape[1]}")
    lines.append(f"- Total missing sensor values: {total_missing_values}")
    lines.append(f"- Features with at least one missing value: {features_with_missing}")
    lines.append(f"- Features with 100% missing values: {all_missing_features}")
    lines.append(f"- Features with missing rate >= 50%: {high_missing_features}")
    lines.append(f"- Features with missing rate between 5% and 50%: {moderate_missing_features}")
    lines.append(f"- Missing timestamps: {timestamp_missing}")
    lines.append("")
    lines.append("## Top 20 Features by Missing Rate")
    lines.append("")
    lines.append("| Rank | Feature | Missing Count | Missing Rate |")
    lines.append("|---:|---|---:|---:|")

    for index, row in top_missing.iterrows():
        lines.append(
            f"| {index + 1} | `{row['feature']}` | "
            f"{int(row['missing_count'])} | {row['missing_rate']:.4f} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "The SECOM dataset contains substantial missing sensor values, which is expected "
        "for high-dimensional manufacturing sensor data. Phase R1 records this issue "
        "explicitly before cleaning, instead of silently imputing values."
    )

    DATA_QUALITY_REPORT.write_text("\n".join(lines), encoding="utf-8")


def build_class_imbalance_report(labels_df: pd.DataFrame) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    label_counts = labels_df["pass_fail_label"].value_counts().sort_index()
    label_rates = labels_df["pass_fail_label"].value_counts(normalize=True).sort_index()

    pass_count = int(label_counts.get(-1, 0))
    fail_count = int(label_counts.get(1, 0))
    total_count = int(len(labels_df))

    pass_rate = float(label_rates.get(-1, 0.0))
    fail_rate = float(label_rates.get(1, 0.0))

    imbalance_ratio = pass_count / fail_count if fail_count > 0 else float("inf")

    lines = []
    lines.append("# SECOM Class Imbalance Report")
    lines.append("")
    lines.append("## Label Meaning")
    lines.append("")
    lines.append("- `-1`: normal / pass")
    lines.append("- `1`: abnormal / fail")
    lines.append("")
    lines.append("## Class Distribution")
    lines.append("")
    lines.append("| Label | Meaning | Count | Rate |")
    lines.append("|---:|---|---:|---:|")
    lines.append(f"| -1 | Pass | {pass_count} | {pass_rate:.4f} |")
    lines.append(f"| 1 | Fail | {fail_count} | {fail_rate:.4f} |")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total samples: {total_count}")
    lines.append(f"- Pass samples: {pass_count}")
    lines.append(f"- Fail samples: {fail_count}")
    lines.append(f"- Fail rate: {fail_rate:.4f}")
    lines.append(f"- Pass-to-fail imbalance ratio: {imbalance_ratio:.2f}:1")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "The SECOM dataset is highly imbalanced. This means accuracy alone can be "
        "misleading. Later modeling stages should emphasize recall, precision, PR-AUC, "
        "confusion matrix, and false alarm burden instead of relying only on accuracy."
    )

    CLASS_IMBALANCE_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    check_required_files()

    sensor_df = load_sensor_data()
    labels_df = load_labels()

    build_missing_value_report(sensor_df, labels_df)
    print("Missing value report generated")

    build_class_imbalance_report(labels_df)
    print("Class imbalance report generated")


if __name__ == "__main__":
    main()