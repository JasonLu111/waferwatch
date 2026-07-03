"""
Monitoring alert utilities for WaferWatch.

This module combines drift monitoring and performance monitoring outputs
into an engineer-readable alert summary.

It reads:
- reports/drift_monitoring_report.json
- reports/performance_monitoring_report.json

It writes:
- reports/monitoring_alert_summary.json
- reports/monitoring_alert_summary.md

The goal is to convert raw monitoring metrics into a practical operational
summary for process, equipment, yield, and data teams.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.utils.config import REPORTS_DIR, ensure_directories_exist
from src.utils.logger import get_logger


logger = get_logger(__name__)


def load_json_report(report_path: Path) -> dict[str, Any]:
    """
    Load a JSON report.
    """

    if not report_path.exists():
        raise FileNotFoundError(f"Report not found: {report_path}")

    with report_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def determine_alert_level(
    drift_alert: bool,
    performance_alert: bool,
    n_drifted_features: int,
    alert_reasons: list[str],
) -> str:
    """
    Determine alert level from drift and performance signals.
    """

    if drift_alert and performance_alert:
        return "critical"

    if performance_alert:
        return "high"

    if drift_alert and n_drifted_features >= 3:
        return "medium"

    if drift_alert:
        return "low"

    if alert_reasons:
        return "low"

    return "normal"


def build_recommended_actions(
    drift_alert: bool,
    performance_alert: bool,
    drifted_features: list[str],
    alert_reasons: list[str],
) -> list[str]:
    """
    Build recommended follow-up actions for engineers and data teams.
    """

    actions: list[str] = []

    if drift_alert:
        actions.append(
            "Review drifted features and compare them with recent tool, chamber, recipe, and maintenance changes."
        )

        if drifted_features:
            actions.append(
                "Prioritize investigation of drifted features: "
                + ", ".join(drifted_features[:10])
                + "."
            )

    if performance_alert:
        actions.append(
            "Review current-period false positives and false negatives to understand whether the alert threshold or model signal has degraded."
        )

        actions.append(
            "Check whether current labels reflect a new operating regime, recipe mix, tool condition, or process variation."
        )

    if "false_alarm_increase_exceeds_threshold" in alert_reasons:
        actions.append(
            "Evaluate alert fatigue risk and consider raising the threshold or limiting escalation volume per shift."
        )

    if (
        "recall_drop_exceeds_threshold" in alert_reasons
        or "current_recall_below_minimum" in alert_reasons
    ):
        actions.append(
            "Investigate missed risky lots and consider retraining, recalibration, or adding new process/event features."
        )

    if (
        "pr_auc_drop_exceeds_threshold" in alert_reasons
        or "current_pr_auc_below_minimum" in alert_reasons
    ):
        actions.append(
            "Check whether risk ranking quality degraded; compare top-K lots against actual outcomes."
        )

    if not actions:
        actions.append(
            "No immediate action required. Continue routine monitoring."
        )

    return actions


def summarize_drift_report(
    drift_report: dict[str, Any],
) -> dict[str, Any]:
    """
    Extract key drift monitoring information.
    """

    return {
        "overall_drift_detected": bool(
            drift_report.get("overall_drift_detected", False)
        ),
        "reference_rows": drift_report.get("reference_rows"),
        "current_rows": drift_report.get("current_rows"),
        "n_features_monitored": drift_report.get("n_features_monitored"),
        "n_features_with_drift": drift_report.get("n_features_with_drift"),
        "drifted_features": drift_report.get("drifted_features", []),
        "thresholds": drift_report.get("thresholds", {}),
    }


def summarize_performance_report(
    performance_report: dict[str, Any],
) -> dict[str, Any]:
    """
    Extract key performance monitoring information.
    """

    comparison = performance_report.get("performance_comparison", {})

    reference_metrics = performance_report.get("reference_metrics", {})
    current_metrics = performance_report.get("current_metrics", {})

    return {
        "performance_alert": bool(
            comparison.get("performance_alert", False)
        ),
        "alert_reasons": comparison.get("alert_reasons", []),
        "metric_deltas": comparison.get("metric_deltas", {}),
        "reference_metrics_preview": {
            "accuracy": reference_metrics.get("accuracy"),
            "precision": reference_metrics.get("precision"),
            "recall": reference_metrics.get("recall"),
            "f1": reference_metrics.get("f1"),
            "pr_auc": reference_metrics.get("pr_auc"),
            "false_alarms_per_100_lots": reference_metrics.get(
                "false_alarms_per_100_lots"
            ),
            "confusion_matrix": reference_metrics.get("confusion_matrix"),
        },
        "current_metrics_preview": {
            "accuracy": current_metrics.get("accuracy"),
            "precision": current_metrics.get("precision"),
            "recall": current_metrics.get("recall"),
            "f1": current_metrics.get("f1"),
            "pr_auc": current_metrics.get("pr_auc"),
            "false_alarms_per_100_lots": current_metrics.get(
                "false_alarms_per_100_lots"
            ),
            "confusion_matrix": current_metrics.get("confusion_matrix"),
        },
        "thresholds": comparison.get("thresholds", {}),
    }


def build_monitoring_alert_summary(
    drift_report_file_name: str = "drift_monitoring_report.json",
    performance_report_file_name: str = "performance_monitoring_report.json",
) -> dict[str, Any]:
    """
    Build a combined monitoring alert summary.
    """

    ensure_directories_exist()

    drift_report_path = REPORTS_DIR / drift_report_file_name
    performance_report_path = REPORTS_DIR / performance_report_file_name

    logger.info("Loading drift report from: %s", drift_report_path)
    drift_report = load_json_report(drift_report_path)

    logger.info("Loading performance report from: %s", performance_report_path)
    performance_report = load_json_report(performance_report_path)

    drift_summary = summarize_drift_report(drift_report)
    performance_summary = summarize_performance_report(performance_report)

    drift_alert = drift_summary["overall_drift_detected"]
    performance_alert = performance_summary["performance_alert"]

    drifted_features = drift_summary["drifted_features"]
    alert_reasons = performance_summary["alert_reasons"]

    alert_level = determine_alert_level(
        drift_alert=drift_alert,
        performance_alert=performance_alert,
        n_drifted_features=drift_summary["n_features_with_drift"],
        alert_reasons=alert_reasons,
    )

    recommended_actions = build_recommended_actions(
        drift_alert=drift_alert,
        performance_alert=performance_alert,
        drifted_features=drifted_features,
        alert_reasons=alert_reasons,
    )

    summary: dict[str, Any] = {
        "alert_level": alert_level,
        "requires_engineer_review": alert_level in ["critical", "high", "medium"],
        "drift_summary": drift_summary,
        "performance_summary": performance_summary,
        "recommended_actions": recommended_actions,
        "interpretation_note": (
            "This alert summary combines data drift and model performance signals. "
            "It is a decision-support artifact, not an automatic root-cause diagnosis."
        ),
    }

    return summary


def save_alert_summary_json(
    summary: dict[str, Any],
    output_file_name: str = "monitoring_alert_summary.json",
) -> Path:
    """
    Save alert summary as JSON.
    """

    output_path = REPORTS_DIR / output_file_name

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    logger.info("Saved monitoring alert summary JSON to: %s", output_path)

    return output_path


def build_alert_summary_markdown(
    summary: dict[str, Any],
) -> str:
    """
    Build an engineer-readable Markdown alert summary.
    """

    drift_summary = summary["drift_summary"]
    performance_summary = summary["performance_summary"]

    reference_metrics = performance_summary["reference_metrics_preview"]
    current_metrics = performance_summary["current_metrics_preview"]
    metric_deltas = performance_summary["metric_deltas"]

    lines: list[str] = []

    lines.append("# WaferWatch Monitoring Alert Summary")
    lines.append("")
    lines.append(f"**Alert level:** `{summary['alert_level']}`")
    lines.append(f"**Requires engineer review:** `{summary['requires_engineer_review']}`")
    lines.append("")
    lines.append("## 1. Drift Status")
    lines.append("")
    lines.append(f"- Overall drift detected: `{drift_summary['overall_drift_detected']}`")
    lines.append(f"- Features monitored: `{drift_summary['n_features_monitored']}`")
    lines.append(f"- Features with drift: `{drift_summary['n_features_with_drift']}`")
    lines.append(
        "- Drifted features: "
        + (
            ", ".join(drift_summary["drifted_features"])
            if drift_summary["drifted_features"]
            else "none"
        )
    )
    lines.append("")
    lines.append("## 2. Performance Status")
    lines.append("")
    lines.append(f"- Performance alert: `{performance_summary['performance_alert']}`")
    lines.append(
        "- Alert reasons: "
        + (
            ", ".join(performance_summary["alert_reasons"])
            if performance_summary["alert_reasons"]
            else "none"
        )
    )
    lines.append("")
    lines.append("### Reference vs Current Metrics")
    lines.append("")
    lines.append("| Metric | Reference | Current | Delta current - reference |")
    lines.append("|---|---:|---:|---:|")

    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "false_alarms_per_100_lots",
    ]

    delta_key_map = {
        "accuracy": "accuracy_delta_current_minus_reference",
        "precision": "precision_delta_current_minus_reference",
        "recall": "recall_delta_current_minus_reference",
        "f1": "f1_delta_current_minus_reference",
        "pr_auc": "pr_auc_delta_current_minus_reference",
        "false_alarms_per_100_lots": (
            "false_alarms_per_100_lots_delta_current_minus_reference"
        ),
    }

    for metric_name in metric_names:
        reference_value = reference_metrics.get(metric_name)
        current_value = current_metrics.get(metric_name)
        delta_value = metric_deltas.get(delta_key_map[metric_name])

        lines.append(
            f"| {metric_name} | {reference_value} | {current_value} | {delta_value} |"
        )

    lines.append("")
    lines.append("## 3. Recommended Actions")
    lines.append("")

    for action in summary["recommended_actions"]:
        lines.append(f"- {action}")

    lines.append("")
    lines.append("## 4. Interpretation Note")
    lines.append("")
    lines.append(summary["interpretation_note"])
    lines.append("")

    return "\n".join(lines)


def save_alert_summary_markdown(
    summary: dict[str, Any],
    output_file_name: str = "monitoring_alert_summary.md",
) -> Path:
    """
    Save alert summary as Markdown.
    """

    output_path = REPORTS_DIR / output_file_name
    markdown_text = build_alert_summary_markdown(summary)

    with output_path.open("w", encoding="utf-8") as file:
        file.write(markdown_text)

    logger.info("Saved monitoring alert summary Markdown to: %s", output_path)

    return output_path


def run_alert_summary_demo() -> dict[str, Any]:
    """
    Run monitoring alert summary demo.
    """

    summary = build_monitoring_alert_summary()

    save_alert_summary_json(summary)
    save_alert_summary_markdown(summary)

    print("Monitoring alert summary")
    print("------------------------")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    return summary


def _demo() -> None:
    """
    Run alert summary demo.
    """

    run_alert_summary_demo()


if __name__ == "__main__":
    _demo()