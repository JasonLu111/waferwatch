"""
Evidence-grounded root-cause triage module for WaferWatch.

This module does not claim true causal discovery. Instead, it creates a
structured engineering triage layer:

1. Identify suspicious lots.
2. Compare each lot against normal-reference feature statistics.
3. Rank feature-level evidence.
4. Map feature evidence to cause hypotheses.
5. Generate lot-level triage reports.

The goal is to bridge anomaly detection outputs to actionable engineering
review.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import DATA_PROCESSED_DIR, REPORTS_DIR, ensure_directories_exist
from src.utils.logger import get_logger


logger = get_logger(__name__)


FEATURE_HYPOTHESIS_MAP: dict[str, dict[str, str]] = {
    "spc_violation_count": {
        "cause_family": "SPC rule violation",
        "hypothesis": "The lot may have repeated process-control rule violations.",
        "evidence_type": "SPC count evidence",
        "recommended_review": "Review control-chart rule violations, recipe context, and recent process-window changes.",
    },
    "spc_max_abs_zscore": {
        "cause_family": "SPC excursion",
        "hypothesis": "The lot may contain a large standardized process excursion.",
        "evidence_type": "SPC magnitude evidence",
        "recommended_review": "Review the strongest SPC excursion, nearby lots, and whether the excursion aligns with known process shifts.",
    },
    "sensor_mean": {
        "cause_family": "Process center shift",
        "hypothesis": "The average sensor behavior may have shifted away from the normal reference pattern.",
        "evidence_type": "Sensor mean evidence",
        "recommended_review": "Review process setpoints, chamber matching, calibration records, and lot-level recipe conditions.",
    },
    "sensor_std": {
        "cause_family": "Process instability",
        "hypothesis": "The lot may show higher within-lot sensor variability or unstable process behavior.",
        "evidence_type": "Sensor variation evidence",
        "recommended_review": "Review tool stability, run-to-run variation, maintenance history, and abnormal sensor fluctuation windows.",
    },
    "sensor_min": {
        "cause_family": "Lower-tail excursion",
        "hypothesis": "The lot may contain an unusual lower-tail sensor pattern relative to normal-reference behavior.",
        "evidence_type": "Sensor low-tail evidence",
        "recommended_review": "Review lower-tail sensor readings, possible under-processing, sensor dropout, or transient boundary shifts.",
    },
    "sensor_max": {
        "cause_family": "Upper-tail excursion",
        "hypothesis": "The lot may contain an unusual upper-tail sensor pattern relative to normal-reference behavior.",
        "evidence_type": "Sensor high-tail evidence",
        "recommended_review": "Review upper-tail sensor readings, possible over-processing, transient spikes, tool instability, or boundary shifts.",
    },
}


def load_selected_feature_table(
    feature_table_file_name: str = "demo_spc_selected_feature_table.csv",
) -> pd.DataFrame:
    """
    Load the selected SPC-enhanced feature table.
    """

    path = DATA_PROCESSED_DIR / feature_table_file_name

    if not path.exists():
        raise FileNotFoundError(f"Missing selected feature table: {path}")

    logger.info("Loading selected feature table from: %s", path)

    return pd.read_csv(path)


def infer_label_column(df: pd.DataFrame) -> str:
    """Infer the binary label column from the selected feature table."""

    candidate_columns = [
        "pass_fail_label",
        "label",
        "target",
        "is_anomaly",
        "failure_label",
    ]

    for column in candidate_columns:
        if column in df.columns:
            return column

    raise ValueError(
        "Could not infer label column. Expected one of: "
        "pass_fail_label, label, target, is_anomaly, failure_label."
    )

def infer_lot_id_column(df: pd.DataFrame) -> str:
    """
    Infer the lot identifier column name.
    """

    candidates = ["lot_id", "lot", "lot_number"]

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    raise ValueError("Could not infer lot ID column. Expected one of: lot_id, lot, lot_number.")


def get_feature_columns(
    df: pd.DataFrame,
    lot_id_column: str,
    label_column: str,
) -> list[str]:
    """
    Identify numeric feature columns.
    """

    excluded = {lot_id_column, label_column}

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(df[column])
    ]

    if not feature_columns:
        raise ValueError("No numeric feature columns found.")

    return feature_columns


def classify_evidence_strength(abs_z_score: float) -> str:
    """
    Convert absolute z-score into a human-readable evidence strength.
    """

    if abs_z_score >= 3.0:
        return "critical"
    if abs_z_score >= 2.0:
        return "high"
    if abs_z_score >= 1.0:
        return "moderate"
    return "low"


def classify_feature_evidence_strength(
    feature: str,
    feature_value: float,
    abs_z_score: float,
) -> str:
    """
    Convert feature evidence into a human-readable strength label.

    Some count-based SPC features can have zero variance among normal-reference lots.
    For those features, domain rules are clearer than raw z-score magnitude.
    """

    if feature == "spc_violation_count":
        if feature_value >= 2:
            return "critical"
        if feature_value >= 1:
            return "high"
        return "low"

    return classify_evidence_strength(abs_z_score)


def calculate_normal_reference_stats(
    df: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
) -> pd.DataFrame:
    """
    Calculate normal-reference feature statistics.
    """

    normal_df = df[df[label_column] == 0]

    if normal_df.empty:
        raise ValueError("No normal-reference rows found.")

    stats = []

    for feature in feature_columns:
        mean_value = float(normal_df[feature].mean())
        std_value = float(normal_df[feature].std(ddof=0))

        if std_value == 0.0 or np.isnan(std_value):
            std_value = 1.0

        stats.append(
            {
                "feature": feature,
                "normal_mean": mean_value,
                "normal_std": std_value,
            }
        )

    return pd.DataFrame(stats)


def build_cause_hypothesis_table(feature_columns: list[str]) -> pd.DataFrame:
    """
    Build the structured cause-hypothesis table.
    """

    rows = []

    for feature in feature_columns:
        mapping = FEATURE_HYPOTHESIS_MAP.get(
            feature,
            {
                "cause_family": "Unmapped feature signal",
                "hypothesis": "The feature deviates from normal-reference behavior and should be reviewed.",
                "evidence_type": "Feature deviation evidence",
                "recommended_review": "Review this feature with process and equipment context.",
            },
        )

        rows.append(
            {
                "feature": feature,
                "cause_family": mapping["cause_family"],
                "hypothesis": mapping["hypothesis"],
                "evidence_type": mapping["evidence_type"],
                "recommended_review": mapping["recommended_review"],
            }
        )

    return pd.DataFrame(rows)


def build_feature_contributions(
    df: pd.DataFrame,
    feature_columns: list[str],
    lot_id_column: str,
    label_column: str,
    normal_stats: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build feature-level evidence table for each lot.
    """

    stats_lookup = normal_stats.set_index("feature").to_dict(orient="index")
    rows = []

    for _, lot_row in df.iterrows():
        lot_id = str(lot_row[lot_id_column])
        true_label = int(lot_row[label_column])

        for feature in feature_columns:
            normal_mean = float(stats_lookup[feature]["normal_mean"])
            normal_std = float(stats_lookup[feature]["normal_std"])
            feature_value = float(lot_row[feature])
            z_score = (feature_value - normal_mean) / normal_std
            abs_z_score = abs(z_score)

            mapping = FEATURE_HYPOTHESIS_MAP.get(
                feature,
                {
                    "cause_family": "Unmapped feature signal",
                    "hypothesis": "The feature deviates from normal-reference behavior and should be reviewed.",
                    "evidence_type": "Feature deviation evidence",
                    "recommended_review": "Review this feature with process and equipment context.",
                },
            )

            rows.append(
                {
                    "lot_id": lot_id,
                    "true_label": true_label,
                    "feature": feature,
                    "feature_value": feature_value,
                    "normal_mean": normal_mean,
                    "normal_std": normal_std,
                    "z_score": float(z_score),
                    "abs_z_score": float(abs_z_score),
                    "evidence_strength": classify_feature_evidence_strength(
                        feature=feature,
                        feature_value=feature_value,
                        abs_z_score=abs_z_score,
                    ),
                    "cause_family": mapping["cause_family"],
                    "hypothesis": mapping["hypothesis"],
                    "evidence_type": mapping["evidence_type"],
                    "recommended_review": mapping["recommended_review"],
                }
            )

    contribution_df = pd.DataFrame(rows)
    contribution_df = contribution_df.sort_values(
        ["lot_id", "abs_z_score"],
        ascending=[True, False],
    ).reset_index(drop=True)

    return contribution_df


def select_triage_lots(
    df: pd.DataFrame,
    lot_id_column: str,
    label_column: str,
    contribution_df: pd.DataFrame,
    max_lots: int = 10,
) -> pd.DataFrame:
    """
    Select lots for triage.

    In this demo, lots are prioritized by:
    1. Known failed-lot label, used only because this is a controlled demo.
    2. Maximum absolute feature z-score.
    """

    max_evidence = (
        contribution_df.groupby("lot_id", as_index=False)["abs_z_score"]
        .max()
        .rename(columns={"abs_z_score": "max_abs_feature_zscore"})
    )

    lot_table = df[[lot_id_column, label_column]].copy()
    lot_table[lot_id_column] = lot_table[lot_id_column].astype(str)
    lot_table = lot_table.rename(
        columns={
            lot_id_column: "lot_id",
            label_column: "true_label",
        }
    )

    lot_table = lot_table.merge(max_evidence, on="lot_id", how="left")
    lot_table["true_label"] = lot_table["true_label"].astype(int)

    lot_table = lot_table.sort_values(
        ["true_label", "max_abs_feature_zscore"],
        ascending=[False, False],
    ).head(max_lots)

    lot_table["triage_rank"] = range(1, len(lot_table) + 1)

    return lot_table.reset_index(drop=True)


def build_lot_triage_reports(
    triage_lots: pd.DataFrame,
    contribution_df: pd.DataFrame,
    top_features_per_lot: int = 3,
) -> list[dict[str, Any]]:
    """
    Build structured lot-level triage reports.
    """

    reports = []

    for _, lot_row in triage_lots.iterrows():
        lot_id = str(lot_row["lot_id"])
        lot_contrib = contribution_df[contribution_df["lot_id"] == lot_id]
        top_contrib = lot_contrib.head(top_features_per_lot)

        evidence_items = []

        for _, evidence_row in top_contrib.iterrows():
            evidence_items.append(
                {
                    "feature": str(evidence_row["feature"]),
                    "feature_value": float(evidence_row["feature_value"]),
                    "normal_mean": float(evidence_row["normal_mean"]),
                    "normal_std": float(evidence_row["normal_std"]),
                    "z_score": float(evidence_row["z_score"]),
                    "abs_z_score": float(evidence_row["abs_z_score"]),
                    "evidence_strength": str(evidence_row["evidence_strength"]),
                    "cause_family": str(evidence_row["cause_family"]),
                    "hypothesis": str(evidence_row["hypothesis"]),
                    "recommended_review": str(evidence_row["recommended_review"]),
                }
            )

        cause_families = list(dict.fromkeys(item["cause_family"] for item in evidence_items))
        recommended_reviews = list(
            dict.fromkeys(item["recommended_review"] for item in evidence_items)
        )

        reports.append(
            {
                "triage_rank": int(lot_row["triage_rank"]),
                "lot_id": lot_id,
                "true_label": int(lot_row["true_label"]),
                "max_abs_feature_zscore": float(lot_row["max_abs_feature_zscore"]),
                "top_cause_families": cause_families,
                "evidence_items": evidence_items,
                "recommended_reviews": recommended_reviews,
                "triage_note": (
                    "This is a cause-hypothesis triage report. It links feature evidence "
                    "to engineering review directions, but it does not prove causal root cause."
                ),
            }
        )

    return reports


def build_lot_summary_table(lot_reports: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Convert lot triage reports into a compact table.
    """

    rows = []

    for report in lot_reports:
        top_feature = report["evidence_items"][0]["feature"]
        top_cause = report["evidence_items"][0]["cause_family"]
        top_strength = report["evidence_items"][0]["evidence_strength"]

        rows.append(
            {
                "triage_rank": report["triage_rank"],
                "lot_id": report["lot_id"],
                "true_label": report["true_label"],
                "max_abs_feature_zscore": report["max_abs_feature_zscore"],
                "top_feature": top_feature,
                "top_cause_family": top_cause,
                "top_evidence_strength": top_strength,
                "cause_family_summary": "; ".join(report["top_cause_families"]),
            }
        )

    return pd.DataFrame(rows)


def build_markdown_report(report: dict[str, Any]) -> str:
    """
    Build Markdown report for root-cause triage.
    """

    lines: list[str] = []

    lines.append("# WaferWatch Evidence-Grounded Root-Cause Triage Report")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append(
        "This report converts anomaly-monitoring outputs into structured engineering triage evidence."
    )
    lines.append(
        "It does not claim true causal discovery. It generates cause hypotheses based on feature deviations from normal-reference behavior."
    )
    lines.append("")
    lines.append("## 2. Cause-Hypothesis Table")
    lines.append("")
    lines.append("| Feature | Cause family | Evidence type | Recommended review |")
    lines.append("|---|---|---|---|")

    for row in report["cause_hypothesis_table"]:
        lines.append(
            f"| `{row['feature']}` | {row['cause_family']} | {row['evidence_type']} | {row['recommended_review']} |"
        )

    lines.append("")
    lines.append("## 3. Triage Lot Summary")
    lines.append("")
    lines.append(
        "| Rank | Lot ID | True Label | Max Abs Z-score | Top Feature | Top Cause Family | Evidence Strength |"
    )
    lines.append("|---:|---|---:|---:|---|---|---|")

    for row in report["lot_summary_table"]:
        lines.append(
            f"| {row['triage_rank']} | `{row['lot_id']}` | {row['true_label']} | "
            f"{row['max_abs_feature_zscore']:.6f} | `{row['top_feature']}` | "
            f"{row['top_cause_family']} | {row['top_evidence_strength']} |"
        )

    lines.append("")
    lines.append("## 4. Lot-Level Evidence Reports")
    lines.append("")

    for lot_report in report["lot_triage_reports"]:
        lines.append(f"### 4.{lot_report['triage_rank']} Lot `{lot_report['lot_id']}`")
        lines.append("")
        lines.append(f"- True label in controlled demo: `{lot_report['true_label']}`")
        lines.append(f"- Max absolute feature z-score: `{lot_report['max_abs_feature_zscore']:.6f}`")
        lines.append(
            f"- Top cause families: `{'; '.join(lot_report['top_cause_families'])}`"
        )
        lines.append("")
        lines.append("| Feature | Value | Normal Mean | Z-score | Strength | Cause family | Hypothesis |")
        lines.append("|---|---:|---:|---:|---|---|---|")

        for evidence in lot_report["evidence_items"]:
            lines.append(
                f"| `{evidence['feature']}` | {evidence['feature_value']:.6f} | "
                f"{evidence['normal_mean']:.6f} | {evidence['z_score']:.6f} | "
                f"{evidence['evidence_strength']} | {evidence['cause_family']} | "
                f"{evidence['hypothesis']} |"
            )

        lines.append("")
        lines.append("Recommended engineering reviews:")
        lines.append("")

        for review in lot_report["recommended_reviews"]:
            lines.append(f"- {review}")

        lines.append("")

    lines.append("## 5. Interpretation")
    lines.append("")
    lines.append(
        "The triage module links anomalous feature evidence to structured cause hypotheses and review actions."
    )
    lines.append(
        "This creates a bridge from model output to engineering investigation, while avoiding unsupported causal claims."
    )
    lines.append(
        "In a real deployment, these hypotheses would need to be validated with process history, tool logs, maintenance records, metrology, and engineering judgment."
    )
    lines.append("")

    return "\n".join(lines)


def run_root_cause_triage(
    feature_table_file_name: str = "demo_spc_selected_feature_table.csv",
    report_file_name: str = "root_cause_triage_report.json",
    markdown_file_name: str = "root_cause_triage_report.md",
    cause_table_file_name: str = "demo_cause_hypothesis_table.csv",
    contribution_file_name: str = "demo_root_cause_feature_contributions.csv",
    lot_summary_file_name: str = "demo_root_cause_triage_lot_summary.csv",
    max_lots: int = 10,
    top_features_per_lot: int = 3,
) -> dict[str, Any]:
    """
    Run evidence-grounded root-cause triage.
    """

    ensure_directories_exist()

    df = load_selected_feature_table(feature_table_file_name)
    lot_id_column = infer_lot_id_column(df)
    label_column = infer_label_column(df)
    feature_columns = get_feature_columns(
        df=df,
        lot_id_column=lot_id_column,
        label_column=label_column,
    )

    normal_stats = calculate_normal_reference_stats(
        df=df,
        feature_columns=feature_columns,
        label_column=label_column,
    )

    cause_hypothesis_table = build_cause_hypothesis_table(feature_columns)

    contribution_df = build_feature_contributions(
        df=df,
        feature_columns=feature_columns,
        lot_id_column=lot_id_column,
        label_column=label_column,
        normal_stats=normal_stats,
    )

    triage_lots = select_triage_lots(
        df=df,
        lot_id_column=lot_id_column,
        label_column=label_column,
        contribution_df=contribution_df,
        max_lots=max_lots,
    )

    lot_triage_reports = build_lot_triage_reports(
        triage_lots=triage_lots,
        contribution_df=contribution_df,
        top_features_per_lot=top_features_per_lot,
    )

    lot_summary_table = build_lot_summary_table(lot_triage_reports)

    cause_table_path = DATA_PROCESSED_DIR / cause_table_file_name
    contribution_path = DATA_PROCESSED_DIR / contribution_file_name
    lot_summary_path = DATA_PROCESSED_DIR / lot_summary_file_name

    cause_hypothesis_table.to_csv(cause_table_path, index=False)
    contribution_df.to_csv(contribution_path, index=False)
    lot_summary_table.to_csv(lot_summary_path, index=False)

    report: dict[str, Any] = {
        "module_name": "Evidence-grounded root-cause triage",
        "feature_table": str(DATA_PROCESSED_DIR / feature_table_file_name),
        "label_column": label_column,
        "lot_id_column": lot_id_column,
        "n_rows": int(len(df)),
        "n_features": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "normal_reference_rows": int((df[label_column] == 0).sum()),
        "triage_lots": int(len(lot_triage_reports)),
        "top_features_per_lot": int(top_features_per_lot),
        "cause_hypothesis_table_file": str(cause_table_path),
        "feature_contribution_file": str(contribution_path),
        "lot_summary_file": str(lot_summary_path),
        "cause_hypothesis_table": cause_hypothesis_table.to_dict(orient="records"),
        "lot_summary_table": lot_summary_table.to_dict(orient="records"),
        "lot_triage_reports": lot_triage_reports,
        "interpretation_note": (
            "This module provides evidence-grounded cause hypotheses for engineering triage. "
            "It does not prove causal root cause."
        ),
    }

    report_path = REPORTS_DIR / report_file_name

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    markdown_path = REPORTS_DIR / markdown_file_name
    markdown_text = build_markdown_report(report)

    with markdown_path.open("w", encoding="utf-8") as file:
        file.write(markdown_text)

    logger.info("Saved root-cause triage report JSON to: %s", report_path)
    logger.info("Saved root-cause triage report Markdown to: %s", markdown_path)

    print("Evidence-grounded root-cause triage summary")
    print("-------------------------------------------")
    print(f"Feature table: {DATA_PROCESSED_DIR / feature_table_file_name}")
    print(f"Rows: {len(df)}")
    print(f"Features: {len(feature_columns)}")
    print(f"Triage lots: {len(lot_triage_reports)}")
    print()
    print("Top triage lots:")
    print(
        lot_summary_table[
            [
                "triage_rank",
                "lot_id",
                "true_label",
                "max_abs_feature_zscore",
                "top_feature",
                "top_cause_family",
                "top_evidence_strength",
            ]
        ]
    )

    return report


def _demo() -> None:
    """
    Run the evidence-grounded root-cause triage demo.
    """

    run_root_cause_triage()


if __name__ == "__main__":
    _demo()