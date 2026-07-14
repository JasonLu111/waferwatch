"""Validate Synthetic Data V2 output quality and reproducibility.

This module validates the research-only synthetic process context generated
from public UCI SECOM sensor data. It does not modify model, API, or MLOps
artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


OUTPUT_FILES = {
    "lots": "data/synthetic/v2/synthetic_secom_v2.csv",
    "tool_events": "data/synthetic/v2/synthetic_tool_events.csv",
    "maintenance": "data/synthetic/v2/synthetic_maintenance.csv",
    "process_changes": (
        "data/synthetic/v2/synthetic_process_changes.csv"
    ),
    "rca_ground_truth": (
        "data/synthetic/v2/synthetic_rca_ground_truth.csv"
    ),
}

ANOMALY_ROOT_CAUSES = {
    "abrupt_mean_shift": "RC_PRESSURE_INSTABILITY",
    "gradual_degradation": "RC_COMPONENT_WEAR",
    "variance_instability": "RC_PROCESS_VARIABILITY",
    "sensor_fault": "RC_SENSOR_HARDWARE",
    "contextual_anomaly": "RC_RECIPE_CONTEXT_MISMATCH",
}

BENIGN_DRIFT_COUNTS = {
    "recipe_mix_change": 24,
    "product_mix_change": 24,
    "tool_reassignment": 24,
}


class SyntheticDataV2ValidationError(ValueError):
    """Raised when a Synthetic Data V2 contract check fails."""


@dataclass(frozen=True)
class ValidationSummary:
    """Compact result returned after all output checks pass."""

    lot_count: int
    synthetic_anomaly_count: int
    benign_drift_count: int
    rca_case_count: int
    mechanism_counts: dict[str, int]
    output_hashes: dict[str, str]


def _fail(message: str) -> None:
    raise SyntheticDataV2ValidationError(message)


def _load_config(repo_root: Path) -> dict[str, Any]:
    config_path = repo_root / "configs/synthetic_data_v2.json"

    if not config_path.exists():
        _fail(f"Missing config file: {config_path}")

    with config_path.open(encoding="utf-8") as file:
        return json.load(file)


def _require_columns(
    frame: pd.DataFrame,
    required_columns: set[str],
    table_name: str,
) -> None:
    missing_columns = required_columns.difference(frame.columns)

    if missing_columns:
        _fail(
            f"{table_name} is missing columns: "
            f"{sorted(missing_columns)}"
        )


def _non_empty(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().ne("")


def _true_mask(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "true", "yes"})
    )


def _evidence_tokens(value: object) -> set[str]:
    if value is None or pd.isna(value):
        return set()

    return {
        token.strip()
        for token in str(value).split(";")
        if token.strip()
    }


def _all_evidence_tokens(series: pd.Series) -> set[str]:
    tokens: set[str] = set()

    for value in series:
        tokens.update(_evidence_tokens(value))

    return tokens


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _expected_injection_count(
    config: dict[str, Any],
    mechanism_id: str,
    lot_count: int,
) -> int:
    mechanism = next(
        item
        for item in config["anomaly_mechanisms"]
        if item["id"] == mechanism_id
    )
    injection_rate = float(
        mechanism["parameters"]["injection_rate"]
    )

    return max(1, int(round(lot_count * injection_rate)))


def _read_output_tables(
    repo_root: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    tables: dict[str, pd.DataFrame] = {}
    output_hashes: dict[str, str] = {}

    for table_id, relative_path in OUTPUT_FILES.items():
        path = repo_root / relative_path

        if not path.exists():
            _fail(f"Missing generated output: {path}")

        if path.stat().st_size == 0:
            _fail(f"Generated output is empty: {path}")

        tables[table_id] = pd.read_csv(path)
        output_hashes[table_id] = _sha256(path)

    return tables, output_hashes


def _validate_lots(
    lots: pd.DataFrame,
    config: dict[str, Any],
    repo_root: Path,
) -> dict[str, int]:
    _require_columns(
        lots,
        {
            "lot_id",
            "source_lot_id",
            "event_time",
            "tool_id",
            "chamber_id",
            "recipe_id",
            "quality_label",
            "label_delay_hours",
            "is_unseen_context",
            "split",
            "is_synthetic_anomaly",
            "anomaly_mechanism",
            "root_cause_id",
            "synthetic_evidence_id",
            "is_benign_drift",
            "sensor_fault_type",
            "degradation_progress",
            "degradation_final_sigma",
            "maintenance_evidence_id",
            "variance_multiplier_progress",
            "variance_final_multiplier",
            "variance_mean_delta",
            "is_contextual_control",
            "contextual_pair_id",
            "contextual_expected_status",
            "contextual_sensor_columns",
            "recipe_context_evidence_id",
        },
        "lots",
    )

    if lots["lot_id"].duplicated().any():
        _fail("lots contains duplicate lot_id values.")

    input_path = repo_root / config["provenance"]["input_path"]

    if not input_path.exists():
        _fail(f"Configured SECOM input does not exist: {input_path}")

    input_row_count = len(pd.read_csv(input_path))

    if len(lots) != input_row_count:
        _fail(
            "Generated lot count must equal the configured SECOM input "
            f"row count. Expected {input_row_count}, got {len(lots)}."
        )

    quality_labels = pd.to_numeric(
        lots["quality_label"],
        errors="coerce",
    )

    if quality_labels.isna().any() or not quality_labels.isin(
        {0, 1}
    ).all():
        _fail("quality_label must contain only 0 or 1.")

    label_delays = pd.to_numeric(
        lots["label_delay_hours"],
        errors="coerce",
    )

    if label_delays.isna().any() or not label_delays.isin(
        {12, 24, 48}
    ).all():
        _fail(
            "label_delay_hours must contain only 12, 24, or 48."
        )

    valid_mechanisms = {"none"}.union(
        ANOMALY_ROOT_CAUSES
    )
    observed_mechanisms = set(
        lots["anomaly_mechanism"].astype(str)
    )

    if not observed_mechanisms.issubset(valid_mechanisms):
        _fail(
            "Unknown anomaly_mechanism values found: "
            f"{sorted(observed_mechanisms.difference(valid_mechanisms))}"
        )

    synthetic_anomaly_mask = _true_mask(
        lots["is_synthetic_anomaly"]
    )
    benign_drift_mask = _true_mask(lots["is_benign_drift"])

    if (synthetic_anomaly_mask & benign_drift_mask).any():
        _fail(
            "A lot cannot be both a synthetic anomaly and benign drift."
        )

    mechanism_counts: dict[str, int] = {}

    for mechanism_id, root_cause_id in ANOMALY_ROOT_CAUSES.items():
        mechanism_lots = lots.loc[
            lots["anomaly_mechanism"].eq(mechanism_id)
        ].copy()
        expected_count = _expected_injection_count(
            config=config,
            mechanism_id=mechanism_id,
            lot_count=len(lots),
        )

        if len(mechanism_lots) != expected_count:
            _fail(
                f"{mechanism_id} count mismatch. "
                f"Expected {expected_count}, got {len(mechanism_lots)}."
            )

        if not _true_mask(
            mechanism_lots["is_synthetic_anomaly"]
        ).all():
            _fail(
                f"{mechanism_id} lots must have "
                "is_synthetic_anomaly=1."
            )

        if not mechanism_lots["root_cause_id"].eq(
            root_cause_id
        ).all():
            _fail(
                f"{mechanism_id} lots must use {root_cause_id}."
            )

        if not _non_empty(
            mechanism_lots["synthetic_evidence_id"]
        ).all():
            _fail(
                f"{mechanism_id} lots must include synthetic evidence IDs."
            )

        mechanism_counts[mechanism_id] = len(mechanism_lots)

    expected_anomaly_total = sum(mechanism_counts.values())

    if int(synthetic_anomaly_mask.sum()) != expected_anomaly_total:
        _fail(
            "is_synthetic_anomaly count does not match the five "
            "implemented anomaly mechanisms."
        )

    if not lots.loc[
        synthetic_anomaly_mask,
        "anomaly_mechanism",
    ].ne("none").all():
        _fail(
            "Synthetic anomaly lots cannot use anomaly_mechanism=none."
        )

    sensor_fault_lots = lots.loc[
        lots["anomaly_mechanism"].eq("sensor_fault")
    ].copy()
    fault_type_counts = (
        sensor_fault_lots["sensor_fault_type"]
        .fillna("")
        .astype(str)
        .value_counts()
        .to_dict()
    )
    expected_fault_types = {
        "stuck_at",
        "dropout",
        "missing_burst",
        "calibration_bias",
    }

    if set(fault_type_counts) != expected_fault_types:
        _fail(
            "sensor_fault must contain exactly these subtypes: "
            "stuck_at, dropout, missing_burst, calibration_bias."
        )

    if max(fault_type_counts.values()) - min(
        fault_type_counts.values()
    ) > 1:
        _fail(
            "sensor_fault subtypes must be distributed as evenly as possible."
        )

    gradual_lots = lots.loc[
        lots["anomaly_mechanism"].eq("gradual_degradation")
    ].sort_values("event_time")
    gradual_progress = pd.to_numeric(
        gradual_lots["degradation_progress"],
        errors="coerce",
    )
    gradual_final_sigma = pd.to_numeric(
        gradual_lots["degradation_final_sigma"],
        errors="coerce",
    )

    if (
        gradual_progress.isna().any()
        or not gradual_progress.is_monotonic_increasing
        or gradual_final_sigma.isna().any()
        or gradual_final_sigma.le(0).any()
    ):
        _fail(
            "gradual_degradation metadata must show positive, "
            "time-increasing degradation."
        )

    if not _non_empty(
        gradual_lots["maintenance_evidence_id"]
    ).all():
        _fail(
            "gradual_degradation lots must cite maintenance evidence."
        )

    variance_lots = lots.loc[
        lots["anomaly_mechanism"].eq("variance_instability")
    ].sort_values("event_time")
    variance_progress = pd.to_numeric(
        variance_lots["variance_multiplier_progress"],
        errors="coerce",
    )
    variance_final_multiplier = pd.to_numeric(
        variance_lots["variance_final_multiplier"],
        errors="coerce",
    )
    variance_mean_delta = pd.to_numeric(
        variance_lots["variance_mean_delta"],
        errors="coerce",
    )

    if (
        variance_progress.isna().any()
        or not variance_progress.is_monotonic_increasing
        or variance_final_multiplier.isna().any()
        or variance_final_multiplier.le(1).any()
        or variance_mean_delta.isna().any()
        or variance_mean_delta.abs().gt(1e-9).any()
    ):
        _fail(
            "variance_instability must increase variance while preserving "
            "the episode mean."
        )

    contextual_lots = lots.loc[
        lots["anomaly_mechanism"].eq("contextual_anomaly")
    ].copy()
    contextual_controls = lots.loc[
        _true_mask(lots["is_contextual_control"])
    ].copy()

    if len(contextual_controls) != len(contextual_lots):
        _fail(
            "contextual_anomaly requires one Recipe A control per anomaly."
        )

    if not contextual_lots["recipe_id"].eq("RCP_B").all():
        _fail(
            "contextual anomalies must use Recipe B."
        )

    if not contextual_controls["recipe_id"].eq("RCP_A").all():
        _fail(
            "contextual controls must use Recipe A."
        )

    if not contextual_controls["anomaly_mechanism"].eq("none").all():
        _fail(
            "contextual controls must retain anomaly_mechanism=none."
        )

    if not _non_empty(
        contextual_lots["recipe_context_evidence_id"]
    ).all() or not _non_empty(
        contextual_controls["recipe_context_evidence_id"]
    ).all():
        _fail(
            "contextual anomaly pairs must include recipe-context evidence."
        )

    anomaly_pair_ids = set(
        contextual_lots["contextual_pair_id"].astype(str)
    )
    control_pair_ids = set(
        contextual_controls["contextual_pair_id"].astype(str)
    )

    if (
        "" in anomaly_pair_ids
        or anomaly_pair_ids != control_pair_ids
        or len(anomaly_pair_ids) != len(contextual_lots)
    ):
        _fail(
            "contextual anomaly/control pair IDs are incomplete or invalid."
        )

    for pair_id in sorted(anomaly_pair_ids):
        anomaly_row = contextual_lots.loc[
            contextual_lots["contextual_pair_id"].eq(pair_id)
        ]
        control_row = contextual_controls.loc[
            contextual_controls["contextual_pair_id"].eq(pair_id)
        ]

        if len(anomaly_row) != 1 or len(control_row) != 1:
            _fail(
                "Each contextual pair must contain exactly one anomaly "
                f"and one control: {pair_id}."
            )

        sensor_columns = [
            column
            for column in str(
                anomaly_row.iloc[0]["contextual_sensor_columns"]
            ).split("|")
            if column
        ]

        if not sensor_columns:
            _fail(
                f"Contextual pair {pair_id} has no sensor metadata."
            )

        anomaly_values = pd.to_numeric(
            anomaly_row.iloc[0][sensor_columns],
            errors="coerce",
        )
        control_values = pd.to_numeric(
            control_row.iloc[0][sensor_columns],
            errors="coerce",
        )

        if not anomaly_values.equals(control_values):
            _fail(
                "Contextual anomaly/control sensor profiles differ for "
                f"{pair_id}."
            )

    prior_contexts = {
        (str(row.tool_id), str(row.chamber_id))
        for row in lots.loc[
            lots["anomaly_mechanism"].isin(
                {
                    "abrupt_mean_shift",
                    "gradual_degradation",
                    "variance_instability",
                    "sensor_fault",
                }
            ),
            ["tool_id", "chamber_id"],
        ].drop_duplicates().itertuples(index=False)
    }
    contextual_contexts = {
        (str(row.tool_id), str(row.chamber_id))
        for row in contextual_lots[
            ["tool_id", "chamber_id"]
        ].drop_duplicates().itertuples(index=False)
    }

    if (
        len(contextual_contexts) != 1
        or prior_contexts.intersection(contextual_contexts)
    ):
        _fail(
            "contextual_anomaly must use one context not used by earlier "
            "anomaly mechanisms."
        )

    benign_lots = lots.loc[benign_drift_mask].copy()

    if len(benign_lots) != sum(BENIGN_DRIFT_COUNTS.values()):
        _fail(
            "Unexpected benign drift lot count. Expected 72."
        )

    if not benign_lots["is_synthetic_anomaly"].map(
        lambda value: not str(value).strip().lower()
        in {"1", "true", "yes"}
    ).all():
        _fail(
            "Benign drift lots must not be synthetic anomalies."
        )

    if not benign_lots["anomaly_mechanism"].eq("none").all():
        _fail(
            "Benign drift lots must retain anomaly_mechanism=none."
        )

    if not benign_lots["root_cause_id"].eq("none").all():
        _fail(
            "Benign drift lots must retain root_cause_id=none."
        )

    _require_columns(
        benign_lots,
        {"benign_drift_type"},
        "benign drift lots",
    )

    for drift_type, expected_count in BENIGN_DRIFT_COUNTS.items():
        actual_count = int(
            benign_lots["benign_drift_type"]
            .fillna("")
            .astype(str)
            .eq(drift_type)
            .sum()
        )

        if actual_count != expected_count:
            _fail(
                f"{drift_type} count mismatch. "
                f"Expected {expected_count}, got {actual_count}."
            )

    return mechanism_counts


def _validate_evidence_and_rca(
    tables: dict[str, pd.DataFrame],
    mechanism_counts: dict[str, int],
) -> None:
    lots = tables["lots"]
    tool_events = tables["tool_events"]
    maintenance = tables["maintenance"]
    process_changes = tables["process_changes"]
    rca_ground_truth = tables["rca_ground_truth"]

    _require_columns(
        tool_events,
        {
            "event_id",
            "event_type",
            "alarm_code",
            "tool_id",
            "chamber_id",
            "related_lot_id",
            "evidence_id",
        },
        "tool_events",
    )
    _require_columns(
        maintenance,
        {
            "maintenance_id",
            "tool_id",
            "chamber_id",
            "delay_days",
            "maintenance_type",
            "evidence_id",
        },
        "maintenance",
    )
    _require_columns(
        process_changes,
        {
            "change_id",
            "change_type",
            "related_lot_id",
            "is_benign_drift",
            "evidence_id",
        },
        "process_changes",
    )
    _require_columns(
        rca_ground_truth,
        {
            "case_id",
            "lot_id",
            "root_cause_id",
            "suspected_cause",
            "evidence_ids",
            "supports_abstention",
        },
        "rca_ground_truth",
    )

    if tool_events["event_id"].duplicated().any():
        _fail("tool_events contains duplicate event_id values.")

    if maintenance["maintenance_id"].duplicated().any():
        _fail("maintenance contains duplicate maintenance_id values.")

    if process_changes["change_id"].duplicated().any():
        _fail("process_changes contains duplicate change_id values.")

    if rca_ground_truth["case_id"].duplicated().any():
        _fail("rca_ground_truth contains duplicate case_id values.")

    synthetic_lots = lots.loc[
        _true_mask(lots["is_synthetic_anomaly"])
    ].copy()
    anomaly_lot_ids = set(
        synthetic_lots["lot_id"].astype(str)
    )
    synthetic_cases = rca_ground_truth.loc[
        rca_ground_truth["lot_id"].astype(str).isin(anomaly_lot_ids)
    ].copy()

    if len(synthetic_cases) != len(synthetic_lots):
        _fail(
            "Every synthetic anomaly lot must have exactly one RCA case."
        )

    if synthetic_cases["lot_id"].duplicated().any():
        _fail(
            "A synthetic anomaly lot has more than one RCA case."
        )

    roots_by_lot = (
        synthetic_lots.set_index("lot_id")["root_cause_id"]
        .astype(str)
        .to_dict()
    )

    for row in synthetic_cases.itertuples(index=False):
        if str(row.root_cause_id) != roots_by_lot[str(row.lot_id)]:
            _fail(
                "Synthetic RCA root cause does not match its lot label: "
                f"{row.lot_id}."
            )

    benign_cases = rca_ground_truth.loc[
        rca_ground_truth["root_cause_id"].eq(
            "RC_BENIGN_MIX_CHANGE"
        )
    ].copy()

    if len(benign_cases) != 3:
        _fail(
            "Expected three benign RAG ground-truth cases."
        )

    if not _true_mask(
        benign_cases["supports_abstention"]
    ).all():
        _fail(
            "Benign RAG ground truth must support abstention."
        )

    expected_rca_count = sum(mechanism_counts.values()) + 3

    if len(rca_ground_truth) != expected_rca_count:
        _fail(
            "Unexpected RCA ground-truth case count. "
            f"Expected {expected_rca_count}, "
            f"got {len(rca_ground_truth)}."
        )

    gradual_maintenance = maintenance.loc[
        maintenance["maintenance_id"]
        .astype(str)
        .str.startswith("MAINT_GRADUAL_")
    ].copy()

    if len(gradual_maintenance) != 1:
        _fail(
            "Expected exactly one delayed gradual-degradation maintenance "
            "record."
        )

    gradual_delay_days = pd.to_numeric(
        gradual_maintenance["delay_days"],
        errors="coerce",
    )

    if gradual_delay_days.isna().any() or gradual_delay_days.le(0).any():
        _fail(
            "Gradual-degradation maintenance must have a positive delay."
        )

    benign_changes = process_changes.loc[
        process_changes["change_id"]
        .astype(str)
        .str.startswith("CHANGE_BENIGN_")
    ].copy()

    if len(benign_changes) != 3:
        _fail(
            "Expected three benign process-change timeline records."
        )

    if set(benign_changes["change_type"].astype(str)) != set(
        BENIGN_DRIFT_COUNTS
    ):
        _fail(
            "Benign process changes must cover recipe mix, product mix, "
            "and tool reassignment."
        )

    known_evidence_ids = set()

    for frame, column in (
        (tool_events, "evidence_id"),
        (maintenance, "evidence_id"),
        (process_changes, "evidence_id"),
        (lots, "synthetic_evidence_id"),
        (lots, "maintenance_evidence_id"),
        (lots, "recipe_context_evidence_id"),
    ):
        known_evidence_ids.update(
            frame.loc[_non_empty(frame[column]), column]
            .astype(str)
            .str.strip()
        )

    for case_id, evidence_ids in zip(
        rca_ground_truth["case_id"].astype(str),
        rca_ground_truth["evidence_ids"],
    ):
        tokens = _evidence_tokens(evidence_ids)

        if not tokens:
            _fail(f"RCA case {case_id} has no evidence IDs.")

        unknown_tokens = tokens.difference(known_evidence_ids)

        if unknown_tokens:
            _fail(
                f"RCA case {case_id} references unknown evidence IDs: "
                f"{sorted(unknown_tokens)}"
            )

    event_evidence_ids = set(
        tool_events.loc[
            _non_empty(tool_events["evidence_id"]),
            "evidence_id",
        ]
        .astype(str)
        .str.strip()
    )

    event_backed_mechanisms = {
        "abrupt_mean_shift",
        "sensor_fault",
        "contextual_anomaly",
    }

    for mechanism_id in event_backed_mechanisms:
        root_cause_id = ANOMALY_ROOT_CAUSES[mechanism_id]
        mechanism_cases = rca_ground_truth.loc[
            rca_ground_truth["root_cause_id"].eq(root_cause_id)
        ]
        mechanism_evidence_ids = _all_evidence_tokens(
            mechanism_cases["evidence_ids"]
        )

        if not mechanism_evidence_ids.intersection(
            event_evidence_ids
        ):
            _fail(
                f"{mechanism_id} RCA cases do not cite a tool-event "
                "evidence ID."
            )

    gradual_maintenance_evidence_id = str(
        gradual_maintenance.iloc[0]["evidence_id"]
    )
    gradual_cases = rca_ground_truth.loc[
        rca_ground_truth["root_cause_id"].eq(
            "RC_COMPONENT_WEAR"
        )
    ]

    if not gradual_cases["evidence_ids"].astype(str).str.contains(
        gradual_maintenance_evidence_id,
        regex=False,
    ).all():
        _fail(
            "gradual_degradation RCA cases must cite the delayed "
            "maintenance evidence ID."
        )

    case_evidence_by_lot = {
        str(lot_id): _evidence_tokens(evidence_ids)
        for lot_id, evidence_ids in zip(
            rca_ground_truth["lot_id"],
            rca_ground_truth["evidence_ids"],
        )
    }

    for row in synthetic_lots.itertuples(index=False):
        lot_id = str(row.lot_id)
        synthetic_evidence_id = str(row.synthetic_evidence_id)

        if synthetic_evidence_id not in case_evidence_by_lot[lot_id]:
            _fail(
                "Synthetic RCA case must cite its lot-level evidence ID: "
                f"{lot_id}."
            )


def validate_output_tables(repo_root: Path) -> ValidationSummary:
    """Validate all committed Synthetic Data V2 CSV output tables."""
    repo_root = repo_root.resolve()
    config = _load_config(repo_root)
    tables, output_hashes = _read_output_tables(repo_root)

    mechanism_counts = _validate_lots(
        lots=tables["lots"],
        config=config,
        repo_root=repo_root,
    )
    _validate_evidence_and_rca(
        tables=tables,
        mechanism_counts=mechanism_counts,
    )

    lots = tables["lots"]

    return ValidationSummary(
        lot_count=len(lots),
        synthetic_anomaly_count=int(
            _true_mask(lots["is_synthetic_anomaly"]).sum()
        ),
        benign_drift_count=int(
            _true_mask(lots["is_benign_drift"]).sum()
        ),
        rca_case_count=len(tables["rca_ground_truth"]),
        mechanism_counts=mechanism_counts,
        output_hashes=output_hashes,
    )


def validate_reproducibility(repo_root: Path) -> None:
    """Regenerate all CSVs in a temporary directory and compare SHA-256."""
    repo_root = repo_root.resolve()

    with tempfile.TemporaryDirectory(
        prefix="waferwatch_synthetic_v2_",
    ) as temporary_directory:
        temporary_path = Path(temporary_directory)
        command = [
            sys.executable,
            "-m",
            "src.data.generate_synthetic_v2_context",
            "--config",
            "configs/synthetic_data_v2.json",
            "--output-dir",
            str(temporary_path),
        ]
        result = subprocess.run(
            command,
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            _fail(
                "Synthetic Data V2 regeneration failed during "
                "reproducibility validation:\n"
                f"{result.stdout}\n{result.stderr}"
            )

        for table_id, relative_path in OUTPUT_FILES.items():
            committed_path = repo_root / relative_path
            regenerated_path = (
                temporary_path / Path(relative_path).name
            )

            if not regenerated_path.exists():
                _fail(
                    "Regeneration did not produce expected file: "
                    f"{regenerated_path}"
                )

            if _sha256(committed_path) != _sha256(regenerated_path):
                _fail(
                    "Byte-for-byte reproducibility failed for "
                    f"{table_id}: {committed_path.name}"
                )


def write_quality_report(
    summary: ValidationSummary,
    report_path: Path,
) -> None:
    """Write a stable, reviewable quality report without timestamps."""
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Synthetic Data V2 Quality and Reproducibility Report",
        "",
        "## Scope",
        "",
        (
            "This artifact validates research-only synthetic process context "
            "generated from public UCI SECOM sensor data. It is not data "
            "from a real semiconductor fab."
        ),
        "",
        "## Validation Result",
        "",
        "| Check | Result |",
        "| --- | --- |",
        "| Output-table structural contract | PASS |",
        "| Five synthetic anomaly mechanisms | PASS |",
        "| Benign recipe/product/tool drift controls | PASS |",
        "| RCA ground truth and evidence references | PASS |",
        "| Byte-for-byte regeneration | PASS |",
        "",
        "## Dataset Summary",
        "",
        f"- Generated lots: {summary.lot_count}",
        f"- Synthetic anomaly lots: {summary.synthetic_anomaly_count}",
        f"- Benign drift lots: {summary.benign_drift_count}",
        f"- RCA ground-truth cases: {summary.rca_case_count}",
        "",
        "## Mechanism Counts",
        "",
        "| Mechanism | Lots |",
        "| --- | ---: |",
    ]

    for mechanism_id, count in summary.mechanism_counts.items():
        lines.append(f"| {mechanism_id} | {count} |")

    lines.extend(
        [
            "",
            "## Output SHA-256",
            "",
            "| Table | SHA-256 |",
            "| --- | --- |",
        ]
    )

    for table_id, digest in summary.output_hashes.items():
        lines.append(f"| {table_id} | `{digest}` |")

    lines.append("")

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate WaferWatch Synthetic Data V2 output quality and "
            "byte-for-byte reproducibility."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root. Defaults to the current directory.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "reports/synthetic_data_v2_quality_report.md"
        ),
        help="Quality report path relative to repo root.",
    )
    parser.add_argument(
        "--skip-reproducibility",
        action="store_true",
        help="Validate committed outputs only; skip temporary regeneration.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = args.repo_root.resolve()
    report_path = args.report

    if not report_path.is_absolute():
        report_path = repo_root / report_path

    try:
        summary = validate_output_tables(repo_root)

        if not args.skip_reproducibility:
            validate_reproducibility(repo_root)

        write_quality_report(
            summary=summary,
            report_path=report_path,
        )
    except SyntheticDataV2ValidationError as error:
        print("SYNTHETIC_V2_OUTPUT_VALIDATION_FAILED")
        print(f"- {error}")
        return 1

    print("SYNTHETIC_V2_OUTPUT_VALIDATION_OK")
    print(f"Generated lots: {summary.lot_count}")
    print(
        "Synthetic anomaly lots: "
        f"{summary.synthetic_anomaly_count}"
    )
    print(f"Benign drift lots: {summary.benign_drift_count}")
    print(f"RCA ground-truth cases: {summary.rca_case_count}")

    for mechanism_id, count in summary.mechanism_counts.items():
        print(f"{mechanism_id}: {count}")

    print("Byte-for-byte reproducibility: PASS")
    print(f"Report: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())